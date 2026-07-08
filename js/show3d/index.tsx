/// <reference types="@webgpu/types" />
/**
 * Show3D - Interactive 3D stack viewer with playback controls.
 *
 * Features:
 * - Scroll to zoom, double-click to reset
 * - Adjustable ROI size via slider
 * - FPS slider control
 * - WebGPU-accelerated FFT
 * - Equal-sized FFT and histogram panels
 * - Automatic theme detection (light/dark mode)
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import IconButton from "@mui/material/IconButton";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import FastForwardIcon from "@mui/icons-material/FastForward";
import StopIcon from "@mui/icons-material/Stop";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import DragIndicatorIcon from "@mui/icons-material/DragIndicator";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar, roundToNiceValue, unitSymbol, formatScaleLabel } from "../figure";
import { downloadBlob, extractBytes, extractFloat32, formatNumber, preserveRestoredWidgetModelsOnSave } from "../format";
import { useHideStaticFallback } from "../staticFallback";
import { findDataRange, applyLogScale, applyLogScaleInPlace, percentileClip, sliderRange, computeStats, computeHistogramFromBytes } from "../stats";
import { MetadataSection } from "../widgetInfo";
import { EmbeddedWidgetView } from "../embeddedWidget";

const SHOW3D_TO_SHOW2D_LINKED_TRAITS = [
  { source: "cmap" },
  { source: "log_scale" },
  { source: "auto_contrast" },
  { source: "vmin" },
  { source: "vmax" },
  { source: "show_stats" },
  { source: "show_controls" },
  { source: "controls_collapsed" },
  { source: "link_contrast" },
  { source: "show_fft" },
  { source: "hidden_panels" },
];
// ============================================================================
// Style tokens (inlined - matches Show2D/Show4DSTEM single-file convention)
// ============================================================================
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const UI_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const controlRow = {
  display: "flex",
  alignItems: "center",
  flexWrap: "wrap",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
  maxWidth: "100%",
  boxSizing: "border-box",
} as const;
const compactButton = {
  fontSize: 10,
  fontFamily: "inherit",
  textTransform: "none" as const,
  letterSpacing: 0,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};
const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};
const sliderStyles = {
  small: {
    py: 0,
    "& .MuiSlider-thumb": { width: 10, height: 10 },
    "& .MuiSlider-rail": { height: 2 },
    "& .MuiSlider-track": { height: 2 },
  },
};
const PAGE_PLAY_FPS_OPTIONS = [1, 2, 3, 4] as const;
const AVG_WINDOW_OPTIONS = Array.from({ length: 15 }, (_, idx) => idx + 1);
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: UI_FONT },
  title: { fontWeight: "bold" as const },
};
type FftOverlayPosition = "top-left" | "top-right" | "bottom-left" | "bottom-right";
type ReorderPlacement = "before" | "after";
type ReorderDragVisual = {
  panel: number;
  label: string;
  imageUrl: string;
  width: number;
  height: number;
  x: number;
  y: number;
  offsetX: number;
  offsetY: number;
};
type ReorderDragStart = {
  x: number;
  y: number;
};
const REORDER_DRAG_THRESHOLD_PX = 8;

function useMobileViewport(): boolean {
  const getIsMobile = React.useCallback(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return false;
    }
    return window.matchMedia("(pointer: coarse)").matches || window.matchMedia("(max-width: 768px)").matches;
  }, []);
  const [isMobile, setIsMobile] = React.useState(getIsMobile);

  React.useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }
    const coarsePointer = window.matchMedia("(pointer: coarse)");
    const narrowViewport = window.matchMedia("(max-width: 768px)");
    const update = () => setIsMobile(getIsMobile());
    const addQueryListener = (query: MediaQueryList) => {
      if (typeof query.addEventListener === "function") query.addEventListener("change", update);
      else query.addListener(update);
    };
    const removeQueryListener = (query: MediaQueryList) => {
      if (typeof query.removeEventListener === "function") query.removeEventListener("change", update);
      else query.removeListener(update);
    };
    update();
    addQueryListener(coarsePointer);
    addQueryListener(narrowViewport);
    window.addEventListener("resize", update);
    return () => {
      removeQueryListener(coarsePointer);
      removeQueryListener(narrowViewport);
      window.removeEventListener("resize", update);
    };
  }, [getIsMobile]);

  return isMobile;
}

// ============================================================================
// Inlined utilities (matches Show2D/Show4DSTEM single-file convention)
// ============================================================================
const signedLog1p = (x: number): number => x >= 0 ? Math.log1p(x) : -Math.log1p(-x);

type Show3DWritableFile = {
  write: (data: BlobPart) => Promise<void>;
  close: () => Promise<void>;
};

type Show3DFileHandle = {
  createWritable: () => Promise<Show3DWritableFile>;
};

type Show3DSavePickerOptions = {
  suggestedName?: string;
  types?: { description: string; accept: Record<string, string[]> }[];
};

type Show3DWindow = Window & typeof globalThis & {
  showSaveFilePicker?: (options?: Show3DSavePickerOptions) => Promise<Show3DFileHandle>;
};

type PanelStats = {
  panel: number;
  mean: number;
  min: number;
  max: number;
  std: number;
};

type CursorInfo = {
  row: number;
  col: number;
  value: number;
  panelIdx: number;
};

function makeExportFilename(
  title: string,
  nSlices: number,
  height: number,
  width: number,
  mode: string,
  quality = "medium",
  downsample = 1,
): string {
  let slug = (title || "show3d")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  while (slug.includes("__")) slug = slug.replace(/__/g, "_");
  if (!slug) slug = "show3d";
  if (mode === "gif" || mode === "mp4") {
    return `${slug}_${nSlices}x${height}x${width}_${quality}.${mode}`;
  }
  const binSuffix = mode === "quantized" && downsample > 1 ? `_${downsample}xbin` : "";
  const suffix = mode === "quantized" ? `quantized${binSuffix}` : "exact";
  return `${slug}_${nSlices}x${height}x${width}_${suffix}.html`;
}

function exportPickerType(mode: string): { description: string; accept: Record<string, string[]> } {
  if (mode === "gif") return { description: "Animated GIF", accept: { "image/gif": [".gif"] } };
  if (mode === "mp4") return { description: "MP4 video", accept: { "video/mp4": [".mp4"] } };
  return { description: "Standalone HTML", accept: { "text/html": [".html"] } };
}

function exportBlobType(mode: string): string {
  if (mode === "gif") return "image/gif";
  if (mode === "mp4") return "video/mp4";
  return "text/html;charset=utf-8";
}

function formatSavedBytes(bytes: number): string {
  const mb = Math.max(0, bytes) / (1024 * 1024);
  if (mb >= 100) return `${Math.round(mb)} MB`;
  if (mb >= 10) return `${mb.toFixed(1)} MB`;
  return `${mb.toFixed(2)} MB`;
}

function isAbortLikeError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function float32FrameFromDataView(stack: DataView, frameIdx: number, pixelCount: number, copy: boolean): Float32Array | null {
  const byteStart = frameIdx * pixelCount * 4;
  const byteLength = pixelCount * 4;
  if (byteStart < 0 || byteStart + byteLength > stack.byteLength) return null;
  const byteOffset = stack.byteOffset + byteStart;
  let view: Float32Array;
  if (byteOffset % 4 === 0) {
    view = new Float32Array(stack.buffer, byteOffset, pixelCount);
  } else {
    const bytes = new Uint8Array(stack.buffer, byteOffset, byteLength);
    const aligned = new Uint8Array(byteLength);
    aligned.set(bytes);
    view = new Float32Array(aligned.buffer);
  }
  return copy ? new Float32Array(view) : view;
}

const clampPct = (x: number): number => Math.max(0, Math.min(100, x));
const valueToPct = (value: number | null | undefined, min: number, max: number, fallback: number): number => {
  if (value == null || !Number.isFinite(value) || max <= min) return fallback;
  return clampPct(((value - min) / (max - min)) * 100);
};
const pctToValue = (pct: number, min: number, max: number): number => min + (max - min) * (clampPct(pct) / 100);

function shouldIgnoreWidgetShortcut(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return target.closest([
    "input", "textarea", "button", "select",
    "[contenteditable='true']", "[role='button']", "[role='slider']",
    "[role='switch']", "[role='textbox']", "[role='combobox']", "[role='menuitem']",
    ".MuiSlider-root", ".MuiSelect-select",
  ].join(",")) !== null;
}

function findFFTPeak(
  mag: Float32Array, width: number, height: number,
  col: number, row: number, radius: number,
): { row: number; col: number } {
  const c0 = Math.max(0, Math.floor(col) - radius);
  const r0 = Math.max(0, Math.floor(row) - radius);
  const c1 = Math.min(width - 1, Math.floor(col) + radius);
  const r1 = Math.min(height - 1, Math.floor(row) + radius);
  let bestCol = Math.round(col), bestRow = Math.round(row), bestVal = -Infinity;
  for (let ir = r0; ir <= r1; ir++) {
    for (let ic = c0; ic <= c1; ic++) {
      const val = mag[ir * width + ic];
      if (val > bestVal) { bestVal = val; bestCol = ic; bestRow = ir; }
    }
  }
  const wc0 = Math.max(0, bestCol - 1), wc1 = Math.min(width - 1, bestCol + 1);
  const wr0 = Math.max(0, bestRow - 1), wr1 = Math.min(height - 1, bestRow + 1);
  let sumW = 0, sumWC = 0, sumWR = 0;
  for (let ir = wr0; ir <= wr1; ir++) {
    for (let ic = wc0; ic <= wc1; ic++) {
      const w = mag[ir * width + ic];
      sumW += w; sumWC += w * ic; sumWR += w * ir;
    }
  }
  if (sumW > 0) return { row: sumWR / sumW, col: sumWC / sumW };
  return { row: bestRow, col: bestCol };
}

function findFFTPeakInBounds(
  mag: Float32Array, width: number, height: number,
  col: number, row: number, radius: number,
  minCol: number, maxCol: number, minRow: number, maxRow: number,
): { row: number; col: number } {
  const c0 = Math.max(0, minCol, Math.floor(col) - radius);
  const r0 = Math.max(0, minRow, Math.floor(row) - radius);
  const c1 = Math.min(width - 1, maxCol, Math.floor(col) + radius);
  const r1 = Math.min(height - 1, maxRow, Math.floor(row) + radius);
  let bestCol = Math.round(col), bestRow = Math.round(row), bestVal = -Infinity;
  for (let ir = r0; ir <= r1; ir++) {
    for (let ic = c0; ic <= c1; ic++) {
      const val = mag[ir * width + ic];
      if (val > bestVal) { bestVal = val; bestCol = ic; bestRow = ir; }
    }
  }
  const wc0 = Math.max(0, minCol, bestCol - 1), wc1 = Math.min(width - 1, maxCol, bestCol + 1);
  const wr0 = Math.max(0, minRow, bestRow - 1), wr1 = Math.min(height - 1, maxRow, bestRow + 1);
  let sumW = 0, sumWC = 0, sumWR = 0;
  for (let ir = wr0; ir <= wr1; ir++) {
    for (let ic = wc0; ic <= wc1; ic++) {
      const w = mag[ir * width + ic];
      sumW += w; sumWC += w * ic; sumWR += w * ir;
    }
  }
  if (sumW > 0) return { row: sumWR / sumW, col: sumWC / sumW };
  return { row: bestRow, col: bestCol };
}

function resolveDisplayRange(
  dataMin: number, dataMax: number,
  traitVmin: number | null | undefined, traitVmax: number | null | undefined,
  logScale: boolean, vminPct: number, vmaxPct: number,
): { vmin: number; vmax: number } {
  const baseMin = logScale ? signedLog1p(traitVmin ?? dataMin) : (traitVmin ?? dataMin);
  const baseMax = logScale ? signedLog1p(traitVmax ?? dataMax) : (traitVmax ?? dataMax);
  return sliderRange(baseMin, baseMax, vminPct, vmaxPct);
}

function resolveDisplayBounds(
  dataMin: number, dataMax: number,
  traitVmin: number | null | undefined, traitVmax: number | null | undefined,
  logScale: boolean,
): { min: number; max: number } {
  return {
    min: logScale ? signedLog1p(traitVmin ?? dataMin) : (traitVmin ?? dataMin),
    max: logScale ? signedLog1p(traitVmax ?? dataMax) : (traitVmax ?? dataMax),
  };
}

function cachedAutoRange(
  vmins: number[] | null | undefined,
  vmaxs: number[] | null | undefined,
  idx: number,
): { vmin: number; vmax: number } | null {
  const vmin = vmins?.[idx];
  const vmax = vmaxs?.[idx];
  if (typeof vmin !== "number" || typeof vmax !== "number") return null;
  return Number.isFinite(vmin) && Number.isFinite(vmax) && vmax > vmin ? { vmin, vmax } : null;
}

function cachedAutoDisplayRange(
  vmins: number[] | null | undefined,
  vmaxs: number[] | null | undefined,
  idx: number,
  logScale: boolean,
): { vmin: number; vmax: number } | null {
  const range = cachedAutoRange(vmins, vmaxs, idx);
  if (!range) return null;
  if (!logScale) return range;
  return { vmin: signedLog1p(range.vmin), vmax: signedLog1p(range.vmax) };
}

const show3dPerfDebugFallback: Record<string, unknown> = {};

function show3dPerfDebug(): Record<string, unknown> | null {
  if (typeof window === "undefined") return null;
  const host = window as unknown as { __quantemShow3DPerf?: Record<string, unknown> };
  if (host.__quantemShow3DPerf) return host.__quantemShow3DPerf;
  try {
    host.__quantemShow3DPerf = {};
    return host.__quantemShow3DPerf;
  } catch {
    try {
      (document.documentElement as unknown as { __quantemShow3DPerf?: Record<string, unknown> }).__quantemShow3DPerf = show3dPerfDebugFallback;
    } catch {
      // Ignore locked-down standalone export environments; diagnostics must not
      // affect the rendering path.
    }
    return show3dPerfDebugFallback;
  }
}

const FRAME_INTERVAL_HISTORY = 512;

function resetFramePacingDebug(dbg: Record<string, unknown>, targetMs: number): void {
  dbg.frameIntervalTargetMs = Number(targetMs.toFixed(2));
  dbg.frameIntervalCount = 0;
  dbg.frameIntervalSumMs = 0;
  dbg.frameIntervalAvgMs = 0;
  dbg.lastFrameIntervalMs = null;
  dbg.maxFrameIntervalMs = 0;
  dbg.overBudgetFrames = 0;
  dbg.frameIntervalHistory = [];
  dbg.lastRenderedAt = null;
}

function recordFramePacingDebug(dbg: Record<string, unknown>, now: number, targetMs: number): void {
  const lastRenderedAt = Number(dbg.lastRenderedAt ?? 0);
  if (lastRenderedAt > 0) {
    const interval = Math.max(0, now - lastRenderedAt);
    const count = Number(dbg.frameIntervalCount ?? 0) + 1;
    const sum = Number(dbg.frameIntervalSumMs ?? 0) + interval;
    const longFrameBudgetMs = Math.max(targetMs * 1.5, targetMs + 8);
    const history = Array.isArray(dbg.frameIntervalHistory)
      ? (dbg.frameIntervalHistory as number[])
      : [];
    history.push(Number(interval.toFixed(2)));
    if (history.length > FRAME_INTERVAL_HISTORY) history.splice(0, history.length - FRAME_INTERVAL_HISTORY);

    dbg.frameIntervalCount = count;
    dbg.frameIntervalSumMs = Number(sum.toFixed(2));
    dbg.frameIntervalAvgMs = Number((sum / count).toFixed(2));
    dbg.lastFrameIntervalMs = Number(interval.toFixed(2));
    dbg.maxFrameIntervalMs = Number(Math.max(Number(dbg.maxFrameIntervalMs ?? 0), interval).toFixed(2));
    dbg.overBudgetFrames = Number(dbg.overBudgetFrames ?? 0) + (interval > longFrameBudgetMs ? 1 : 0);
    dbg.frameIntervalHistory = history;
  }
  dbg.lastRenderedAt = now;
}

function percentileFromHistory(values: unknown, percentile: number): number | null {
  if (!Array.isArray(values) || values.length === 0) return null;
  const nums = values
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v))
    .sort((a, b) => a - b);
  if (nums.length === 0) return null;
  const idx = Math.min(nums.length - 1, Math.max(0, Math.ceil((percentile / 100) * nums.length) - 1));
  return Number(nums[idx].toFixed(2));
}

function estimateRafFps(sampleMs: number): Promise<number | null> {
  if (typeof window === "undefined" || typeof window.requestAnimationFrame !== "function") {
    return Promise.resolve(null);
  }
  return new Promise(resolve => {
    let first = 0;
    let last = 0;
    let frames = 0;
    const tick = (ts: number) => {
      if (first === 0) first = ts;
      last = ts;
      frames++;
      if (ts - first >= sampleMs) {
        const elapsed = Math.max(1, last - first);
        resolve(frames > 1 ? (frames - 1) * 1000 / elapsed : null);
        return;
      }
      window.requestAnimationFrame(tick);
    };
    window.requestAnimationFrame(tick);
  });
}

const FRAME_SERVER_STREAM_CACHE_BYTES = 4 * 1024 * 1024 * 1024;
const FRAME_SERVER_FULL_STACK_CACHE_BYTES = 24 * 1024 * 1024 * 1024;
const FRAME_SERVER_JS_FULL_STACK_CACHE_BYTES = 8 * 1024 * 1024 * 1024;
const FRAME_SERVER_SEPARATE_PANEL_GPU_CACHE_BYTES = 1024 * 1024 * 1024;
const FRAME_SERVER_MIN_CACHE_FRAMES = 6;
const FRAME_SERVER_PREFETCH_FRAMES = 8;

// ============================================================================
// Inlined components (matches Show2D single-file convention)
// ============================================================================
function InfoTooltip({ text, theme = "dark" }: { text: React.ReactNode; theme?: "light" | "dark" }) {
  const isDark = theme === "dark";
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  return (
    <Tooltip
      title={content} arrow placement="bottom"
      componentsProps={{
        tooltip: { sx: { bgcolor: isDark ? "#333" : "#fff", color: isDark ? "#ddd" : "#333", border: `1px solid ${isDark ? "#555" : "#ccc"}`, maxWidth: 360, p: 1 } },
        arrow: { sx: { color: isDark ? "#333" : "#fff", "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` } } },
      }}
    >
      <Typography component="span" sx={{ fontSize: 12, color: isDark ? "#888" : "#666", cursor: "help", ml: 0.5, "&:hover": { color: isDark ? "#aaa" : "#444" } }}>
        ⓘ
      </Typography>
    </Tooltip>
  );
}

function KeyboardShortcuts({ items }: { items: [string, string][] }) {
  return (
    <Box
      component="table"
      sx={{
        borderCollapse: "collapse",
        "& td": { py: 0.25, fontSize: 11, lineHeight: 1.3, verticalAlign: "top" },
        "& td:first-of-type": { pr: 1.5, opacity: 0.7, fontFamily: "monospace", fontSize: 10, whiteSpace: "nowrap" },
      }}
    >
      <tbody>
        {items.map(([key, desc], i) => (
          <tr key={i}><td>{key}</td><td>{desc}</td></tr>
        ))}
      </tbody>
    </Box>
  );
}

interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
  pinBinsToRange?: boolean;
  ariaHidden?: boolean;
  // Pre-computed 256-element bin array (e.g. from GPU). When provided, the
  // CPU `computeHistogramFromBytes` fallback is skipped entirely.
  bins?: number[] | null;
}

const Histogram = React.memo(function Histogram({
  data, vminPct, vmaxPct, onRangeChange,
  width = 110, height = 40, theme = "dark",
  dataMin = 0, dataMax = 1, pinBinsToRange = true, ariaHidden = false,
  bins: precomputedBins = null,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const sliderRef = React.useRef<HTMLDivElement | null>(null);
  const minLabelRef = React.useRef<HTMLElement | null>(null);
  const maxLabelRef = React.useRef<HTMLElement | null>(null);
  const onRangeChangeRef = React.useRef(onRangeChange);
  const pendingRangeRef = React.useRef<[number, number] | null>(null);
  const rangeRafRef = React.useRef<number | null>(null);
  // Bins source priority: GPU-precomputed > CPU memoized scan. The CPU path
  // is an O(N) pass over 16.8 M Float32 at 4k (89% of scrub cost in profiling)
  // so we only run it when the GPU path didn't produce bins.
  const bins = React.useMemo(
    () => {
      // Use GPU-precomputed bins only if non-empty. The GPU path can return
      // an all-zero array when the engine slot has no data yet (e.g. the
      // colormap render effect hasn't run yet), which would draw a blank
      // histogram. Falling back to the CPU bin scan in that case keeps the
      // first paint correct; subsequent renders use the GPU bins.
      if (precomputedBins && precomputedBins.length === 256) {
        let total = 0;
        for (let i = 0; i < precomputedBins.length; i++) total += precomputedBins[i];
        if (total > 0) return precomputedBins;
      }
      return pinBinsToRange
        ? computeHistogramFromBytes(data, 256, dataMin, dataMax)
        : computeHistogramFromBytes(data);
    },
    [precomputedBins, data, dataMin, dataMax, pinBinsToRange],
  );
  const colors = theme === "dark"
    ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" }
    : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };
  const formatValue = React.useCallback((pct: number) => {
    const val = dataMin + (pct / 100) * (dataMax - dataMin);
    return val >= 1000 ? val.toExponential(2) : val.toFixed(2);
  }, [dataMax, dataMin]);
  const drawHistogram = React.useCallback((loPct: number, hiPct: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    // setTransform (not scale) so React 19 StrictMode double-invoke doesn't stack.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);
    const displayBins = 64;
    const binRatio = Math.max(1, Math.floor(bins.length / displayBins));
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) sum += bins[i * binRatio + j] || 0;
      reducedBins.push(sum / binRatio);
    }
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((loPct / 100) * displayBins);
    const vmaxBin = Math.floor((hiPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;
      ctx.fillStyle = i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, colors, height, width]);
  const applyRangePreview = React.useCallback((next: [number, number]) => {
    const [lo, hi] = next;
    const slider = sliderRef.current?.querySelector(".MuiSlider-root") as HTMLElement | null;
    const thumbs = slider?.querySelectorAll(".MuiSlider-thumb");
    const track = slider?.querySelector(".MuiSlider-track") as HTMLElement | null;
    if (thumbs && thumbs.length >= 2) {
      (thumbs[0] as HTMLElement).style.left = `${lo}%`;
      (thumbs[1] as HTMLElement).style.left = `${hi}%`;
    }
    if (track) {
      track.style.left = `${lo}%`;
      track.style.width = `${Math.max(0, hi - lo)}%`;
    }
    if (minLabelRef.current) minLabelRef.current.textContent = formatValue(lo);
    if (maxLabelRef.current) maxLabelRef.current.textContent = formatValue(hi);
    drawHistogram(lo, hi);
  }, [drawHistogram, formatValue]);
  React.useEffect(() => {
    drawHistogram(vminPct, vmaxPct);
  }, [drawHistogram, vmaxPct, vminPct]);
  React.useEffect(() => {
    onRangeChangeRef.current = onRangeChange;
  }, [onRangeChange]);
  const flushRangePreview = React.useCallback(() => {
    if (rangeRafRef.current != null) {
      window.cancelAnimationFrame(rangeRafRef.current);
      rangeRafRef.current = null;
    }
    const pending = pendingRangeRef.current;
    pendingRangeRef.current = null;
    if (pending) {
      applyRangePreview(pending);
      onRangeChangeRef.current(pending[0], pending[1]);
    }
  }, [applyRangePreview]);
  React.useEffect(() => () => {
    if (rangeRafRef.current != null) window.cancelAnimationFrame(rangeRafRef.current);
  }, []);
  const beginRangeDrag = React.useCallback((event: React.MouseEvent, dragWidth: number, lo0: number, hi0: number) => {
    const startX = event.clientX;
    const span = Math.max(1, hi0 - lo0);
    const previousCursor = document.body.style.cursor;
    document.body.style.cursor = "grabbing";
    const onMove = (moveEvent: MouseEvent) => {
      moveEvent.preventDefault();
      const deltaPct = ((moveEvent.clientX - startX) / Math.max(1, dragWidth)) * 100;
      const lo = Math.max(0, Math.min(100 - span, lo0 + deltaPct));
      const next: [number, number] = [lo, lo + span];
      pendingRangeRef.current = next;
      if (rangeRafRef.current == null) {
        rangeRafRef.current = window.requestAnimationFrame(() => {
          rangeRafRef.current = null;
          const pending = pendingRangeRef.current;
          if (pending) {
            applyRangePreview(pending);
            onRangeChangeRef.current(pending[0], pending[1]);
          }
        });
      }
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = previousCursor;
      flushRangePreview();
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [applyRangePreview, flushRangePreview]);

  const sliderInset = 4;
  const sliderWidth = Math.max(1, width - sliderInset * 2);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0, width, overflow: "visible" }}>
      <Box sx={{ position: "relative", width, height: height + 6, overflow: "visible" }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}`, display: "block" }}
        role={ariaHidden ? undefined : "img"}
        aria-hidden={ariaHidden ? "true" : undefined}
        aria-label={ariaHidden ? undefined : "Histogram of intensity values with min and max clip handles"}
      />
      <Box
        ref={sliderRef}
        onMouseDownCapture={(e) => {
          if ((e.target as HTMLElement).closest(".MuiSlider-thumb")) return;
          const rect = sliderRef.current?.getBoundingClientRect();
          if (!rect) return;
          const lo = Math.max(0, Math.min(100, Math.min(vminPct, vmaxPct)));
          const hi = Math.max(0, Math.min(100, Math.max(vminPct, vmaxPct)));
          const pct = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 100;
          if (pct < lo || pct > hi) return;
            const thumbGuardPct = Math.max(4, (10 / Math.max(1, rect.width)) * 100);
            if (Math.abs(pct - lo) <= thumbGuardPct || Math.abs(pct - hi) <= thumbGuardPct) return;
            beginRangeDrag(e, rect.width, lo, hi);
            e.preventDefault();
            e.stopPropagation();
          e.nativeEvent.stopImmediatePropagation();
        }}
        sx={{ position: "absolute", left: sliderInset, top: height - 1, width: sliderWidth, height: 8, display: "flex", alignItems: "flex-start", cursor: "grab", zIndex: 2, overflow: "visible" }}
      >
        <Slider
          value={[vminPct, vmaxPct]}
          onChange={(_, v) => {
            const [newMin, newMax] = v as number[];
            onRangeChange(Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1));
          }}
          min={0} max={100} size="small"
          valueLabelDisplay="auto" valueLabelFormat={formatValue}
          aria-label="Histogram intensity clip range"
          sx={{
            width: sliderWidth, py: 0,
            position: "relative",
            zIndex: 3,
            overflow: "visible",
            "& .MuiSlider-rail": { height: 2, zIndex: 1 },
            "& .MuiSlider-track": { height: 2, cursor: "grab", zIndex: 2 },
            "& .MuiSlider-thumb": { width: 8, height: 8, zIndex: 4 },
            "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px", zIndex: 5 },
          }}
        />
      </Box>
      </Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography ref={minLabelRef} sx={{ fontSize: 8, fontFamily: UI_FONT, opacity: 0.6, lineHeight: 1 }}>{formatValue(vminPct)}</Typography>
        <Typography ref={maxLabelRef} sx={{ fontSize: 8, fontFamily: UI_FONT, opacity: 0.6, lineHeight: 1 }}>{formatValue(vmaxPct)}</Typography>
      </Box>
    </Box>
  );
});

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const container = {
  // Match the shared canvas scale-bar typography for a clean microscope-viewer UI.
  root: {
    p: 2,
    bgcolor: "transparent",
    color: "inherit",
    fontFamily: UI_FONT,
    overflow: "visible",
    "& .MuiTypography-root, & .MuiButton-root, & .MuiInputBase-root": { fontFamily: "inherit" },
  },
  imageBox: { bgcolor: "transparent", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

import { COLORMAPS, COLORMAP_NAMES, applyColormap, renderToOffscreen, renderToOffscreenReuse, createGPUColormapEngine, GPUColormapEngine } from "../colormaps";

const DPR = window.devicePixelRatio || 1;
const RESIZE_HIT_AREA_PX = 10;
const ENABLE_GPU_CANVAS_DISPLAY = true;

function packedRgbFromHex(color: string): number {
  const raw = (color.startsWith("#") ? color.slice(1) : color).trim();
  const expanded = raw.length === 3
    ? raw.split("").map(ch => ch + ch).join("")
    : raw.slice(0, 6);
  const parsed = Number.parseInt(expanded, 16);
  return Number.isFinite(parsed) ? parsed & 0xFFFFFF : 0;
}

// ROI drawing
function drawROI(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  shape: "circle" | "square" | "rectangle" | "annular",
  radius: number,
  width: number,
  height: number,
  activeColor: string,
  inactiveColor: string,
  active: boolean = false,
  innerRadius: number = 0
): void {
  const strokeColor = active ? activeColor : inactiveColor;
  ctx.strokeStyle = strokeColor;
  // Caller sets ctx.lineWidth from roi.line_width; don't clobber.
  if (shape === "circle") {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
  } else if (shape === "square") {
    ctx.strokeRect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (shape === "rectangle") {
    ctx.strokeRect(x - width / 2, y - height / 2, width, height);
  } else if (shape === "annular") {
    // Outer circle
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
    // Inner circle (cyan)
    ctx.strokeStyle = active ? "#0ff" : inactiveColor;
    ctx.beginPath();
    ctx.arc(x, y, innerRadius, 0, Math.PI * 2);
    ctx.stroke();
    // Annular fill
    ctx.fillStyle = (active ? activeColor : inactiveColor) + "15";
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.arc(x, y, innerRadius, 0, Math.PI * 2, true);
    ctx.fill();
    ctx.strokeStyle = strokeColor;
  }
  if (active) {
    ctx.beginPath();
    ctx.moveTo(x - 5, y);
    ctx.lineTo(x + 5, y);
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x, y + 5);
    ctx.stroke();
  }
}

import { WebGPUFFT, getWebGPUFFT, getGPUInfo, fft2d, fft2dAsync, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2, applyHannWindow2D } from "../fft";
import { computeFftQualityMetrics, formatFftQualityLabel, summarizeFftQualityMetrics, type FftQualityMetrics } from "../fftMetrics";

const FFT_SNAP_RADIUS = 5;

/** Sample intensity values along a line using bilinear interpolation. */
function sampleSingleLine(data: Float32Array, w: number, h: number, row0: number, col0: number, row1: number, col1: number): Float32Array {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  const n = Math.max(2, Math.ceil(len));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const c = col0 + t * dc;
    const r = row0 + t * dr;
    const ci = Math.floor(c), ri = Math.floor(r);
    const cf = c - ci, rf = r - ri;
    const c0c = Math.max(0, Math.min(w - 1, ci));
    const c1c = Math.max(0, Math.min(w - 1, ci + 1));
    const r0c = Math.max(0, Math.min(h - 1, ri));
    const r1c = Math.max(0, Math.min(h - 1, ri + 1));
    out[i] = data[r0c * w + c0c] * (1 - cf) * (1 - rf) +
             data[r0c * w + c1c] * cf * (1 - rf) +
             data[r1c * w + c0c] * (1 - cf) * rf +
             data[r1c * w + c1c] * cf * rf;
  }
  return out;
}

/** Sample intensity along a line, averaging over profileWidth perpendicular pixels. */
function sampleLineProfile(data: Float32Array, w: number, h: number, row0: number, col0: number, row1: number, col1: number, profileWidth: number = 1): Float32Array {
  if (profileWidth <= 1) return sampleSingleLine(data, w, h, row0, col0, row1, col1);
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  if (len < 1e-8) return sampleSingleLine(data, w, h, row0, col0, row1, col1);
  const perpR = -dc / len;
  const perpC = dr / len;
  const half = (profileWidth - 1) / 2;
  let accumulated: Float32Array | null = null;
  for (let k = 0; k < profileWidth; k++) {
    const off = -half + k;
    const vals = sampleSingleLine(data, w, h, row0 + off * perpR, col0 + off * perpC, row1 + off * perpR, col1 + off * perpC);
    if (!accumulated) {
      accumulated = vals;
    } else {
      for (let i = 0; i < vals.length; i++) accumulated[i] += vals[i];
    }
  }
  if (accumulated) for (let i = 0; i < accumulated.length; i++) accumulated[i] /= profileWidth;
  return accumulated || new Float32Array(0);
}

// uint8-stack variants: dequantize ONLY the bilinear corners at each sample
// point instead of materializing the whole frame. Critical for kymograph on 4k
// stacks - sampling a line touches ~lineLen*4*width pixels, not width*height*N.
// `u8` is the packed offline stack; `base` = frameIdx * w * h; value =
// u8[base + idx] * scale + offset.
function sampleSingleLineU8(u8: Uint8Array, base: number, w: number, h: number, scale: number, offset: number, row0: number, col0: number, row1: number, col1: number): Float32Array {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  const n = Math.max(2, Math.ceil(len));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const c = col0 + t * dc;
    const r = row0 + t * dr;
    const ci = Math.floor(c), ri = Math.floor(r);
    const cf = c - ci, rf = r - ri;
    const c0c = Math.max(0, Math.min(w - 1, ci));
    const c1c = Math.max(0, Math.min(w - 1, ci + 1));
    const r0c = Math.max(0, Math.min(h - 1, ri));
    const r1c = Math.max(0, Math.min(h - 1, ri + 1));
    const v00 = u8[base + r0c * w + c0c] * scale + offset;
    const v01 = u8[base + r0c * w + c1c] * scale + offset;
    const v10 = u8[base + r1c * w + c0c] * scale + offset;
    const v11 = u8[base + r1c * w + c1c] * scale + offset;
    out[i] = v00 * (1 - cf) * (1 - rf) + v01 * cf * (1 - rf) + v10 * (1 - cf) * rf + v11 * cf * rf;
  }
  return out;
}

function sampleLineProfileU8(u8: Uint8Array, base: number, w: number, h: number, scale: number, offset: number, row0: number, col0: number, row1: number, col1: number, profileWidth: number = 1): Float32Array {
  if (profileWidth <= 1) return sampleSingleLineU8(u8, base, w, h, scale, offset, row0, col0, row1, col1);
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  if (len < 1e-8) return sampleSingleLineU8(u8, base, w, h, scale, offset, row0, col0, row1, col1);
  const perpR = -dc / len;
  const perpC = dr / len;
  const half = (profileWidth - 1) / 2;
  let accumulated: Float32Array | null = null;
  for (let k = 0; k < profileWidth; k++) {
    const off = -half + k;
    const vals = sampleSingleLineU8(u8, base, w, h, scale, offset, row0 + off * perpR, col0 + off * perpC, row1 + off * perpR, col1 + off * perpC);
    if (!accumulated) accumulated = vals;
    else for (let i = 0; i < vals.length; i++) accumulated[i] += vals[i];
  }
  if (accumulated) for (let i = 0; i < accumulated.length; i++) accumulated[i] /= profileWidth;
  return accumulated || new Float32Array(0);
}

function pointToSegmentDistance(col: number, row: number, col0: number, row0: number, col1: number, row1: number): number {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const lenSq = dc * dc + dr * dr;
  if (lenSq <= 1e-12) return Math.sqrt((col - col0) ** 2 + (row - row0) ** 2);
  const tRaw = ((col - col0) * dc + (row - row0) * dr) / lenSq;
  const t = Math.max(0, Math.min(1, tRaw));
  const projCol = col0 + t * dc;
  const projRow = row0 + t * dr;
  return Math.sqrt((col - projCol) ** 2 + (row - projRow) ** 2);
}

// ============================================================================
// Constants
// ============================================================================
// Reserved GPU slot for offline-mode histogram compute (well above any
// frame-server slot index = nSlices*nPanels), so uploading the scratch frame
// never clobbers a cached playback slot.
const OFFLINE_HIST_SLOT = 1_000_000;
const CANVAS_TARGET_SIZE = 600;
const MAX_PANEL_COLUMNS = 12;
const FFT_OVERLAY_MAX_SOURCE_SIZE = 512;
const MIN_ZOOM = 0.5;
const MIN_IMAGE_ZOOM = 1;
const MAX_ZOOM = 30;
const MAX_PLAYBACK_FPS = 30;
const HTML_EXPORT_OVERHEAD_BYTES = 700_000;
const ANIMATION_QUALITY_SCALE: Record<string, number> = { low: 0.35, medium: 0.6, high: 1.0 };

function formatEstimatedHtmlSize(payloadBytes: number): string {
  const htmlBytes = Math.max(0, payloadBytes) * 4 / 3 + HTML_EXPORT_OVERHEAD_BYTES;
  const mb = htmlBytes / (1024 * 1024);
  if (mb >= 100) return `~${Math.round(mb)} MB`;
  if (mb >= 10) return `~${mb.toFixed(1)} MB`;
  return `~${mb.toFixed(2)} MB`;
}

function formatEstimatedAnimationWork(
  width: number,
  height: number,
  nSlices: number,
  visiblePanels: number,
  maxCols: number,
  panelGap: number,
  quality: string,
): string {
  const scale = ANIMATION_QUALITY_SCALE[quality] ?? ANIMATION_QUALITY_SCALE.medium;
  const panelW = Math.max(1, Math.floor(Math.max(1, width) * scale));
  const panelH = Math.max(1, Math.floor(Math.max(1, height) * scale));
  const panels = Math.max(1, visiblePanels);
  const cols = maxCols <= 0 ? panels : Math.max(1, Math.min(maxCols, panels));
  const rows = Math.max(1, Math.ceil(panels / cols));
  const gap = Math.max(0, Math.round(panelGap || 0));
  const outW = cols * panelW + Math.max(0, cols - 1) * gap;
  const outH = rows * panelH + Math.max(0, rows - 1) * gap;
  const rgbBytes = outW * outH * Math.max(1, nSlices) * 3;
  const mb = rgbBytes / (1024 * 1024);
  if (mb >= 100) return `~${Math.round(mb)} MB work`;
  if (mb >= 10) return `~${mb.toFixed(1)} MB work`;
  return `~${mb.toFixed(2)} MB work`;
}

const clampPlaybackFps = (value: number) => {
  const fps = Number.isFinite(value) ? value : 1;
  return Math.max(1, Math.min(MAX_PLAYBACK_FPS, fps));
};

const playbackIntervalMs = (value: number) => {
  const fps = clampPlaybackFps(value);
  return 1000 / fps;
};

function suppressFftRadialBackgroundInPlace(data: Float32Array, width: number, height: number): void {
  if (width < 16 || height < 16 || data.length !== width * height) return;
  const cx = Math.floor(width / 2);
  const cy = Math.floor(height / 2);
  const maxRadius = Math.ceil(Math.hypot(Math.max(cx, width - cx), Math.max(cy, height - cy)));
  const sums = new Float64Array(maxRadius + 1);
  const counts = new Uint32Array(maxRadius + 1);

  for (let y = 0; y < height; y++) {
    const dy = y - cy;
    const offset = y * width;
    for (let x = 0; x < width; x++) {
      const radius = Math.min(maxRadius, Math.floor(Math.hypot(x - cx, dy)));
      sums[radius] += data[offset + x];
      counts[radius]++;
    }
  }

  for (let radius = 0; radius <= maxRadius; radius++) {
    if (counts[radius] > 0) sums[radius] /= counts[radius];
  }

  // Display-only whitening: remove the smooth radial pedestal so Bragg spots and
  // lattice peaks remain visible in small FFT overlays without changing the
  // underlying magnitude data used for measurements.
  for (let y = 0; y < height; y++) {
    const dy = y - cy;
    const offset = y * width;
    for (let x = 0; x < width; x++) {
      const radius = Math.min(maxRadius, Math.floor(Math.hypot(x - cx, dy)));
      data[offset + x] -= sums[radius];
    }
  }
}

type ROIItem = {
  row: number;
  col: number;
  shape: string;
  radius: number;
  radius_inner: number;
  width: number;
  height: number;
  color: string;
  line_width: number;
  highlight: boolean;
};
const ROI_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];

function createROI(row: number, col: number, shape: string, index: number, imgW: number = 0, imgH: number = 0): ROIItem {
  const defR = imgW > 0 && imgH > 0 ? Math.max(10, Math.round(Math.min(imgW, imgH) * 0.05)) : 10;
  return {
    row,
    col,
    shape,
    radius: defR,
    radius_inner: Math.max(5, Math.round(defR * 0.5)),
    width: defR * 2,
    height: defR * 2,
    color: ROI_COLORS[index % ROI_COLORS.length],
    line_width: 2,
    highlight: false,
  };
}

function normalizeROI(roi: ROIItem, index: number): ROIItem {
  return {
    ...roi,
    color: roi.color || ROI_COLORS[index % ROI_COLORS.length],
    shape: roi.shape || "circle",
    radius: roi.radius ?? 10,
    radius_inner: roi.radius_inner ?? 5,
    width: roi.width ?? 20,
    height: roi.height ?? 20,
    line_width: roi.line_width ?? 2,
    highlight: !!roi.highlight,
  };
}

/** Extract a single frame from the playback buffer (zero-copy subarray). */
function getFrameFromBuffer(
  buffer: Float32Array | null,
  bufStart: number,
  bufCount: number,
  nSlices: number,
  frameIdx: number,
  frameSize: number,
): Float32Array | null {
  if (!buffer || bufCount === 0) return null;
  let offset = frameIdx - bufStart;
  if (offset < 0) offset += nSlices;
  if (offset < 0 || offset >= bufCount) return null;
  const start = offset * frameSize;
  const end = start + frameSize;
  if (end > buffer.length) return null;
  return buffer.subarray(start, end);
}

/** Fused single-pass render: optional log scale + normalize + colormap → RGBA.
 *  Eliminates multiple data passes during playback for maximum frame rate. */
function renderFramePlayback(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
): void {
  const range = vmax - vmin;
  const invRange = range > 0 ? 255 / range : 0;
  if (logScale) {
    for (let i = 0; i < data.length; i++) {
      const d = data[i];
      const v = d >= 0 ? Math.log1p(d) : -Math.log1p(-d);
      const idx = v <= vmin ? 0 : v >= vmax ? 255 : ((v - vmin) * invRange) | 0;
      const j = i << 2;
      const k = idx * 3;
      rgba[j] = lut[k];
      rgba[j + 1] = lut[k + 1];
      rgba[j + 2] = lut[k + 2];
      rgba[j + 3] = 255;
    }
  } else {
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      const idx = v <= vmin ? 0 : v >= vmax ? 255 : ((v - vmin) * invRange) | 0;
      const j = i << 2;
      const k = idx * 3;
      rgba[j] = lut[k];
      rgba[j + 1] = lut[k + 1];
      rgba[j + 2] = lut[k + 2];
      rgba[j + 3] = 255;
    }
  }
}

function renderFrameScaledPlayback(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  xMap: Uint32Array,
  yMap: Uint32Array,
  outW: number,
  outH: number,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
): void {
  const range = vmax - vmin;
  const invRange = range > 0 ? 255 / range : 0;
  for (let y = 0; y < outH; y++) {
    const srcRow = yMap[y];
    const outRow = y * outW;
    for (let x = 0; x < outW; x++) {
      let v = data[srcRow + xMap[x]];
      if (logScale) v = v >= 0 ? Math.log1p(v) : -Math.log1p(-v);
      const idx = v <= vmin ? 0 : v >= vmax ? 255 : ((v - vmin) * invRange) | 0;
      const j = (outRow + x) << 2;
      const k = idx * 3;
      rgba[j] = lut[k];
      rgba[j + 1] = lut[k + 1];
      rgba[j + 2] = lut[k + 2];
      rgba[j + 3] = 255;
    }
  }
}

// ============================================================================
// Crop ROI region from raw float32 data for ROI-scoped FFT
// ============================================================================
function cropROIRegion(
  data: Float32Array, imgW: number, imgH: number,
  roi: ROIItem,
): { cropped: Float32Array; cropW: number; cropH: number } | null {
  const shape = roi.shape || "circle";
  let col0: number, row0: number, col1: number, row1: number;

  if (shape === "rectangle") {
    const hw = roi.width / 2;
    const hh = roi.height / 2;
    col0 = Math.max(0, Math.floor(roi.col - hw));
    row0 = Math.max(0, Math.floor(roi.row - hh));
    col1 = Math.min(imgW, Math.ceil(roi.col + hw));
    row1 = Math.min(imgH, Math.ceil(roi.row + hh));
  } else {
    const r = roi.radius;
    col0 = Math.max(0, Math.floor(roi.col - r));
    row0 = Math.max(0, Math.floor(roi.row - r));
    col1 = Math.min(imgW, Math.ceil(roi.col + r));
    row1 = Math.min(imgH, Math.ceil(roi.row + r));
  }

  const cropW = col1 - col0;
  const cropH = row1 - row0;
  if (cropW < 2 || cropH < 2) return null;

  const cropped = new Float32Array(cropW * cropH);

  if (shape === "circle" || shape === "annular") {
    const r = roi.radius;
    const rSq = r * r;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const imgCol = col0 + dx;
        const imgRow = row0 + dy;
        const distSq = (imgCol - roi.col) * (imgCol - roi.col) + (imgRow - roi.row) * (imgRow - roi.row);
        cropped[dy * cropW + dx] = distSq <= rSq ? data[imgRow * imgW + imgCol] : 0;
      }
    }
  } else {
    for (let dy = 0; dy < cropH; dy++) {
      const srcOffset = (row0 + dy) * imgW + col0;
      cropped.set(data.subarray(srcOffset, srcOffset + cropW), dy * cropW);
    }
  }

  return { cropped, cropW, cropH };
}

// ============================================================================
// Compute stats for pixels inside a single ROI (mean/min/max/std)
// ============================================================================
function computeROIPixelStats(
  data: Float32Array, imgW: number, imgH: number,
  roi: ROIItem,
): { mean: number; min: number; max: number; std: number } | null {
  const shape = roi.shape || "circle";
  let col0: number, row0: number, col1: number, row1: number;

  if (shape === "rectangle") {
    const hw = roi.width / 2;
    const hh = roi.height / 2;
    col0 = Math.max(0, Math.floor(roi.col - hw));
    row0 = Math.max(0, Math.floor(roi.row - hh));
    col1 = Math.min(imgW, Math.ceil(roi.col + hw));
    row1 = Math.min(imgH, Math.ceil(roi.row + hh));
  } else {
    const r = roi.radius;
    col0 = Math.max(0, Math.floor(roi.col - r));
    row0 = Math.max(0, Math.floor(roi.row - r));
    col1 = Math.min(imgW, Math.ceil(roi.col + r));
    row1 = Math.min(imgH, Math.ceil(roi.row + r));
  }

  const cropW = col1 - col0;
  const cropH = row1 - row0;
  if (cropW < 1 || cropH < 1) return null;

  let sum = 0, sumSq = 0, minVal = Infinity, maxVal = -Infinity, n = 0;

  if (shape === "circle") {
    const rSq = roi.radius * roi.radius;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const imgCol = col0 + dx, imgRow = row0 + dy;
        const distSq = (imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2;
        if (distSq > rSq) continue;
        const v = data[imgRow * imgW + imgCol];
        sum += v; sumSq += v * v;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
        n++;
      }
    }
  } else if (shape === "annular") {
    const rSq = roi.radius * roi.radius;
    const riSq = (roi.radius_inner || 0) ** 2;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const imgCol = col0 + dx, imgRow = row0 + dy;
        const distSq = (imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2;
        if (distSq > rSq || distSq < riSq) continue;
        const v = data[imgRow * imgW + imgCol];
        sum += v; sumSq += v * v;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
        n++;
      }
    }
  } else {
    // square or rectangle - all pixels in bounding box
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const v = data[(row0 + dy) * imgW + (col0 + dx)];
        sum += v; sumSq += v * v;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
        n++;
      }
    }
  }

  if (n === 0) return null;
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
  return { mean, min: minVal, max: maxVal, std };
}

// ============================================================================
// Main Component
// ============================================================================
function Show3D() {
  const isMobileViewport = useMobileViewport();
  const model = useModel();
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);

  // Theme detection (offline HTML exports force a light/white background)
  const [offlineForTheme] = useModelState<boolean>("_export_light");
  const { themeInfo, colors: baseColors } = useTheme(offlineForTheme);
  const themeColors = {
    ...baseColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#b08800",
  };
  const mobileControlRowSx = isMobileViewport
    ? ({ columnGap: "8px", rowGap: "4px", px: 0.75, py: 0.25 } as const)
    : ({} as const);

  // Theme-aware select style (matching Show4DSTEM)
  const themedSelect = {
    ...controlPanel.select,
    fontFamily: "inherit",
    flexShrink: 0,  // never compress a dropdown below its width -> no truncated label
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiSelect-select": { py: 0.5, fontFamily: "inherit", textOverflow: "clip", overflow: "visible" },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}`, fontFamily: UI_FONT, "& .MuiMenuItem-root": { fontFamily: "inherit" } } },
  };

  // Model state (synced with Python)
  const [sliceIdx, setSliceIdx] = useModelState<number>("slice_idx");
  const [nSlices] = useModelState<number>("n_slices");
  const [labels] = useModelState<string[]>("labels");
  const [panelFrameLabels] = useModelState<string[][]>("panel_frame_labels");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [rawFrameBytes] = useModelState<DataView>("frame_bytes");
  const [staticFallbackJpeg] = useModelState<string>("_static_fallback_jpeg");
  const [staticFallbackMime] = useModelState<string>("_static_fallback_mime");
  // Defensive: traitlets.Bytes can identity-suppress trait events when content
  // and length are similar. frame_seq is incremented Python-side on every write
  // so JS effects always see a change. Use it in dep arrays alongside frameBytes.
  const [frameSeq] = useModelState<number>("frame_seq");
  // Offline mode: standalone HTML can carry either a compact uint8 stack
  // (_offline_stack) or an exact float32 stack (_offline_float_stack). JS
  // slices locally on scrub so exported reports do not need a Python kernel.
  const [offline] = useModelState<boolean>("offline");
  const [offlineStack] = useModelState<DataView>("_offline_stack");
  const [offlineFloatStack] = useModelState<DataView>("_offline_float_stack");
  const [offlineMin] = useModelState<number>("_offline_min");
  const [offlineMax] = useModelState<number>("_offline_max");
  const [offlineMins] = useModelState<number[]>("_offline_mins");
  const [offlineMaxs] = useModelState<number[]>("_offline_maxs");
  const [nPanels] = useModelState<number>("n_panels");
  const [panelWidthPx] = useModelState<number>("panel_width_px");
  const [sharedPanelSource] = useModelState<boolean>("shared_panel_source");
  const [separatePanelFrames] = useModelState<boolean>("separate_panel_frames");
  // Reused scratch Float32Array sized to one frame so per-scrub dequant
  // doesn't re-allocate. Indexed by (width, height) since reshape resets it.
  const offlineScratch = React.useRef<Float32Array | null>(null);
  const offlineScratchKey = React.useRef<number>(-1);
  // Local live index used by the offline frameBytes useMemo. During MUI Slider
  // drag, `setSliceIdx` (anywidget useModelState) goes through model.set +
  // save_changes which appears to batch under rapid mousemove ticks - useMemo
  // doesn't see sliceIdx change until the user releases the slider, so the
  // canvas stays on the old frame the whole drag. Local React state updates
  // synchronously per tick. scrubToSlice writes both; the model trait is still
  // updated for state.dict round-trips and observers.
  const [liveSliceIdx, setLiveSliceIdx] = React.useState<number>(sliceIdx);
  React.useEffect(() => { setLiveSliceIdx(sliceIdx); }, [sliceIdx]);
  const frameBytes = React.useMemo<DataView>(() => {
    const pixelCount = width * height;
    if (offline && offlineFloatStack && offlineFloatStack.byteLength > 0 && pixelCount > 0) {
      const f32 = float32FrameFromDataView(offlineFloatStack, liveSliceIdx, pixelCount, false);
      if (f32) return new DataView(f32.buffer, f32.byteOffset, f32.byteLength);
    }
    if (offline && offlineStack && offlineStack.byteLength > 0 && width > 0 && height > 0) {
      const start = liveSliceIdx * pixelCount;
      if (start + pixelCount <= offlineStack.byteLength) {
        const u8 = new Uint8Array(offlineStack.buffer, offlineStack.byteOffset + start, pixelCount);
        const key = (width << 16) | height;
        if (offlineScratchKey.current !== key || offlineScratch.current === null) {
          offlineScratch.current = new Float32Array(pixelCount);
          offlineScratchKey.current = key;
        }
        const f32 = offlineScratch.current;
        const panelCount = Math.max(1, nPanels || 1);
        const panelRanges = panelCount > 1 && offlineMins?.length >= panelCount && offlineMaxs?.length >= panelCount;
        const panelW = Math.max(1, panelWidthPx || Math.floor(width / panelCount) || width);
        if (panelRanges) {
          for (let r = 0; r < height; r++) {
            const rowOffset = r * width;
            for (let c = 0; c < width; c++) {
              const panel = Math.max(0, Math.min(panelCount - 1, Math.floor(c / panelW)));
              const lo = offlineMins[panel] ?? offlineMin;
              const hi = offlineMaxs[panel] ?? offlineMax;
              f32[rowOffset + c] = u8[rowOffset + c] * ((hi - lo) / 255.0) + lo;
            }
          }
        } else {
          const scale = (offlineMax - offlineMin) / 255.0;
          for (let i = 0; i < pixelCount; i++) f32[i] = u8[i] * scale + offlineMin;
        }
        return new DataView(f32.buffer);
      }
    }
    return rawFrameBytes;
  }, [offline, offlineStack, offlineFloatStack, offlineMin, offlineMax, offlineMins, offlineMaxs, rawFrameBytes, liveSliceIdx, width, height, nPanels, panelWidthPx]);
  const getOfflineFrame = (idx: number): Float32Array | null => {
    // Allocate a FRESH Float32Array per call so the GPU upload path's
    // pointer-equality cache can't short-circuit the upload and leave
    // the canvas painted with a stale frame. Previously reused a single
    // scratch buffer in place; engine.uploadData saw identical reference
    // every tick and skipped the texture refresh — autoplay frame counter
    // advanced but canvas stayed on initial frame. Verified 2026-05-24.
    if (!offline || width <= 0 || height <= 0) return null;
    const pixelCount = width * height;
    if (offlineFloatStack && offlineFloatStack.byteLength > 0) {
      return float32FrameFromDataView(offlineFloatStack, idx, pixelCount, true);
    }
    if (!offlineStack || offlineStack.byteLength === 0) return null;
    const start = idx * pixelCount;
    if (start < 0 || start + pixelCount > offlineStack.byteLength) return null;
    const u8 = new Uint8Array(offlineStack.buffer, offlineStack.byteOffset + start, pixelCount);
    const f32 = new Float32Array(pixelCount);
    const panelCount = Math.max(1, nPanels || 1);
    const panelRanges = panelCount > 1 && offlineMins?.length >= panelCount && offlineMaxs?.length >= panelCount;
    const panelW = Math.max(1, panelWidthPx || Math.floor(width / panelCount) || width);
    if (panelRanges) {
      for (let r = 0; r < height; r++) {
        const rowOffset = r * width;
        for (let c = 0; c < width; c++) {
          const panel = Math.max(0, Math.min(panelCount - 1, Math.floor(c / panelW)));
          const lo = offlineMins[panel] ?? offlineMin;
          const hi = offlineMaxs[panel] ?? offlineMax;
          f32[rowOffset + c] = u8[rowOffset + c] * ((hi - lo) / 255.0) + lo;
        }
      }
    } else {
      const scale = (offlineMax - offlineMin) / 255.0;
      for (let i = 0; i < pixelCount; i++) f32[i] = u8[i] * scale + offlineMin;
    }
    return f32;
  };

  // Truthful first-render signal: flipped ONCE after the first frame_bytes
  // arrives and the browser has had time to composite two frames.  Python side
  // observes `_js_rendered` and prints the real end-to-end wall clock, not the
  // misleading Python-only __init__ number.
  const [, setJsRendered] = useModelState<boolean>("_js_rendered");
  const firstRenderFiredRef = React.useRef(false);
  React.useEffect(() => {
    if (firstRenderFiredRef.current) return;
    if (!frameBytes || frameBytes.byteLength === 0) return;
    firstRenderFiredRef.current = true;
    requestAnimationFrame(() => requestAnimationFrame(() => setJsRendered(true)));
  }, [frameBytes, setJsRendered]);

  const [title] = useModelState<string>("title");
  const [showTitle] = useModelState<boolean>("show_title");
  const [dimLabel] = useModelState<string>("dim_label");
  const [dimSampling] = useModelState<number>("dim_sampling");
  const [dimUnit] = useModelState<string>("dim_unit");
  const [panelTitles] = useModelState<string[]>("panel_titles");
  const [panelRealFrames] = useModelState<number[]>("panel_real_frames");
  const [starred, setStarred] = useModelState<number[]>("starred");
  const [hiddenPanels, setHiddenPanels] = useModelState<number[]>("hidden_panels");
  const [panelOrder, setPanelOrder] = useModelState<number[]>("panel_order");
  const [nPages] = useModelState<number>("n_pages");
  const [pageIdx, setPageIdx] = useModelState<number>("page_idx");
  const [panelsPerPage] = useModelState<number>("panels_per_page");
  const [pageLabels] = useModelState<string[]>("page_labels");
  const [pageStarred, setPageStarred] = useModelState<number[]>("page_starred");
  const [pagePlaying, setPagePlaying] = React.useState(false);
  const [pagePlayFps, setPagePlayFps] = React.useState<number>(2);
  const [reorderMode, setReorderMode] = React.useState(false);
  const [dragOverPanel, setDragOverPanel] = React.useState<number | null>(null);
  const [reorderPreviewOrder, setReorderPreviewOrder] = React.useState<number[] | null>(null);
  const [reorderDragVisual, setReorderDragVisual] = React.useState<ReorderDragVisual | null>(null);
  const draggedPanelRef = React.useRef<number | null>(null);
  const pointerReorderPanelRef = React.useRef<number | null>(null);
  const reorderPreviewOrderRef = React.useRef<number[] | null>(null);
  const reorderDragVisualRef = React.useRef<ReorderDragVisual | null>(null);
  const reorderGhostRef = React.useRef<HTMLDivElement>(null);
  const reorderGhostRafRef = React.useRef<number | null>(null);
  const reorderGhostPendingRef = React.useRef<{ x: number; y: number } | null>(null);
  const reorderDragStartRef = React.useRef<ReorderDragStart | null>(null);
  const reorderDragActivatedRef = React.useRef(false);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const gpuCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const canvasContainerRef = React.useRef<HTMLDivElement>(null);
  const totalPanelCount = Math.max(1, nPanels || 1);
  const isPaged = (nPages || 1) > 1 && (panelsPerPage || 0) > 0;
  const currentPageIdx = Math.max(0, Math.min((nPages || 1) - 1, Math.round(pageIdx || 0)));
  const currentPageLabel = pageLabels?.[currentPageIdx] || `Page ${currentPageIdx + 1}`;
  const currentPageStatus = `${currentPageLabel} ${currentPageIdx + 1}/${nPages || 1}`;
  React.useEffect(() => {
    if (!isPaged || (nPages || 1) <= 1) setPagePlaying(false);
  }, [isPaged, nPages]);
  React.useEffect(() => {
    if (!pagePlaying || !isPaged || (nPages || 1) <= 1) return;
    const timeout = window.setTimeout(() => {
      setPageIdx((currentPageIdx + 1) % Math.max(1, nPages || 1));
    }, 1000 / Math.max(1, pagePlayFps));
    return () => window.clearTimeout(timeout);
  }, [currentPageIdx, isPaged, nPages, pagePlayFps, pagePlaying, setPageIdx]);
  const activePageStart = isPaged ? currentPageIdx * Math.max(1, panelsPerPage || 1) : 0;
  const activePageEnd = isPaged ? Math.min(totalPanelCount, activePageStart + Math.max(1, panelsPerPage || 1)) : totalPanelCount;
  const activePageIndices = React.useMemo(
    () => Array.from({ length: Math.max(0, activePageEnd - activePageStart) }, (_, i) => activePageStart + i),
    [activePageStart, activePageEnd]
  );
  const activePanelCount = isPaged ? activePageIndices.length : totalPanelCount;
  const [hiddenPageSlots, setHiddenPageSlots] = React.useState<number[]>([]);
  const hiddenPageSlotsInitializedRef = React.useRef(false);
  React.useEffect(() => {
    if (!isPaged) {
      hiddenPageSlotsInitializedRef.current = false;
      return;
    }
    if (hiddenPageSlotsInitializedRef.current) return;
    hiddenPageSlotsInitializedRef.current = true;
    const slots = new Set<number>();
    for (const value of hiddenPanels || []) {
      const idx = Math.trunc(Number(value));
      if (Number.isFinite(idx) && idx >= activePageStart && idx < activePageEnd) {
        slots.add(idx - activePageStart);
      }
    }
    setHiddenPageSlots(Array.from(slots).filter(slot => slot >= 0 && slot < activePanelCount).sort((a, b) => a - b));
  }, [activePageEnd, activePageStart, activePanelCount, hiddenPanels, isPaged]);
  React.useEffect(() => {
    if (!isPaged) return;
    const mapped = Array.from(new Set(hiddenPageSlots
      .map(slot => activePageStart + Math.trunc(Number(slot)))
      .filter(panel => Number.isFinite(panel) && panel >= activePageStart && panel < activePageEnd)))
      .sort((a, b) => a - b);
    const current = (hiddenPanels || [])
      .map(value => Math.trunc(Number(value)))
      .filter(panel => Number.isFinite(panel) && panel >= 0 && panel < totalPanelCount)
      .sort((a, b) => a - b);
    if (mapped.length === current.length && mapped.every((panel, idx) => panel === current[idx])) return;
    setHiddenPanels(mapped);
  }, [activePageEnd, activePageStart, hiddenPageSlots, hiddenPanels, isPaged, setHiddenPanels, totalPanelCount]);
  const hiddenPanelSet = React.useMemo(() => {
    const clean = new Set<number>();
    if (isPaged) {
      for (const value of hiddenPageSlots || []) {
        const slot = Math.trunc(Number(value));
        const idx = activePageStart + slot;
        if (Number.isFinite(slot) && slot >= 0 && slot < activePanelCount && idx >= activePageStart && idx < activePageEnd) {
          clean.add(idx);
        }
      }
    } else {
      for (const value of hiddenPanels || []) {
        const idx = Math.trunc(Number(value));
        if (Number.isFinite(idx) && idx >= 0 && idx < totalPanelCount) clean.add(idx);
      }
    }
    const activeHiddenCount = (isPaged ? activePageIndices : Array.from({ length: totalPanelCount }, (_, panel) => panel))
      .filter((panel) => clean.has(panel)).length;
    if (activeHiddenCount >= Math.max(1, activePanelCount)) {
      const fallback = (isPaged ? activePageIndices : [totalPanelCount - 1])[Math.max(0, activePanelCount - 1)];
      clean.delete(fallback);
    }
    return clean;
  }, [activePageEnd, activePageIndices, activePageStart, activePanelCount, hiddenPageSlots, hiddenPanels, totalPanelCount, isPaged]);
  const naturalPanelOrder = React.useMemo(
    () => isPaged ? activePageIndices : Array.from({ length: totalPanelCount }, (_, panel) => panel),
    [activePageIndices, isPaged, totalPanelCount]
  );
  const orderedPanelIndices = React.useMemo(() => {
    if (isPaged) return naturalPanelOrder;
    const values = Array.isArray(panelOrder) ? panelOrder.map(value => Math.trunc(Number(value))) : [];
    const valid = (
      values.length === totalPanelCount &&
      values.every((value) => Number.isFinite(value) && value >= 0 && value < totalPanelCount) &&
      new Set(values).size === totalPanelCount
    );
    return valid ? values : naturalPanelOrder;
  }, [panelOrder, naturalPanelOrder, totalPanelCount, isPaged]);
  const previewOrderedPanelIndices = React.useMemo(() => {
    if (isPaged) return null;
    const values = Array.isArray(reorderPreviewOrder) ? reorderPreviewOrder.map(value => Math.trunc(Number(value))) : [];
    const valid = (
      values.length === totalPanelCount &&
      values.every((value) => Number.isFinite(value) && value >= 0 && value < totalPanelCount) &&
      new Set(values).size === totalPanelCount
    );
    return valid ? values : null;
  }, [reorderPreviewOrder, totalPanelCount, isPaged]);
  const displayOrderedPanelIndices = previewOrderedPanelIndices || orderedPanelIndices;
  const visiblePanelIndices = React.useMemo(
    () => displayOrderedPanelIndices.filter(panel => !hiddenPanelSet.has(panel)),
    [hiddenPanelSet, displayOrderedPanelIndices]
  );
  const visiblePanelCount = visiblePanelIndices.length;
  const panelMenuTotal = isPaged ? activePanelCount : totalPanelCount;
  const panelLabel = React.useCallback((panel: number) => (
    (panelTitles && panelTitles[panel]) || `Panel ${panel + 1}`
  ), [panelTitles]);
  const setPanelHidden = React.useCallback((panel: number, hidden: boolean) => {
    if (panel < 0 || panel >= totalPanelCount) return;
    if (isPaged) {
      if (panel < activePageStart || panel >= activePageEnd) return;
      const slot = panel - activePageStart;
      const next = new Set<number>();
      for (const value of hiddenPageSlots || []) {
        const idx = Math.trunc(Number(value));
        if (Number.isFinite(idx) && idx >= 0 && idx < activePanelCount) next.add(idx);
      }
      if (hidden) {
        if (!next.has(slot) && activePanelCount - next.size <= 1) return;
        next.add(slot);
      } else {
        next.delete(slot);
      }
      setHiddenPageSlots(Array.from(next).sort((a, b) => a - b));
      return;
    }
    const next = new Set<number>();
    for (const value of hiddenPanels || []) {
      const idx = Math.trunc(Number(value));
      if (Number.isFinite(idx) && idx >= 0 && idx < totalPanelCount) next.add(idx);
    }
    if (hidden) {
      const activeVisible = (isPaged ? activePageIndices : Array.from({ length: totalPanelCount }, (_, idx) => idx))
        .filter((idx) => !next.has(idx)).length;
      if (!next.has(panel) && activeVisible <= 1) return;
      next.add(panel);
    } else {
      next.delete(panel);
    }
    setHiddenPanels(Array.from(next).sort((a, b) => a - b));
  }, [activePageEnd, activePageStart, activePanelCount, hiddenPageSlots, hiddenPanels, totalPanelCount, isPaged, activePageIndices, setHiddenPanels]);
  const applyPanelOrder = React.useCallback((order: number[]) => {
    const clean = order.filter((value) => Number.isInteger(value) && value >= 0 && value < totalPanelCount);
    if (clean.length !== totalPanelCount || new Set(clean).size !== totalPanelCount) return;
    const natural = clean.every((value, idx) => value === idx);
    setPanelOrder(natural ? [] : clean);
  }, [setPanelOrder, totalPanelCount]);
  const setReorderPreviewOrderValue = React.useCallback((order: number[] | null) => {
    reorderPreviewOrderRef.current = order;
    setReorderPreviewOrder(order);
  }, []);
  const setReorderDragVisualValue = React.useCallback((visual: ReorderDragVisual | null) => {
    reorderDragVisualRef.current = visual;
    setReorderDragVisual(visual);
  }, []);
  const captureReorderPanelImage = React.useCallback((panelRect: DOMRect, containerRect: DOMRect): string => {
    const container = canvasContainerRef.current;
    if (!container) return "";
    const canvases = Array.from(container.querySelectorAll("canvas")) as HTMLCanvasElement[];
    const source = canvases.find((canvas) => {
      const rect = canvas.getBoundingClientRect();
      const style = window.getComputedStyle(canvas);
      const opacity = Number(style.opacity || "1");
      return rect.width > 0 && rect.height > 0 && canvas.width > 0 && canvas.height > 0 &&
        style.display !== "none" && opacity > 0.5;
    }) || canvases.find((canvas) => canvas.width > 0 && canvas.height > 0);
    if (!source) return "";
    const scaleX = source.width / Math.max(1, containerRect.width);
    const scaleY = source.height / Math.max(1, containerRect.height);
    const sx = Math.max(0, Math.round((panelRect.left - containerRect.left) * scaleX));
    const sy = Math.max(0, Math.round((panelRect.top - containerRect.top) * scaleY));
    const sw = Math.max(1, Math.min(source.width - sx, Math.round(panelRect.width * scaleX)));
    const sh = Math.max(1, Math.min(source.height - sy, Math.round(panelRect.height * scaleY)));
    if (sw <= 0 || sh <= 0) return "";
    const scratch = document.createElement("canvas");
    scratch.width = sw;
    scratch.height = sh;
    const ctx = scratch.getContext("2d");
    if (!ctx) return "";
    try {
      ctx.drawImage(source, sx, sy, sw, sh, 0, 0, sw, sh);
      return scratch.toDataURL("image/png");
    } catch {
      return "";
    }
  }, []);
  const updateReorderGhostPosition = React.useCallback((clientX: number, clientY: number) => {
    const visual = reorderDragVisualRef.current;
    const container = canvasContainerRef.current;
    if (!visual || !container) return;
    const rect = container.getBoundingClientRect();
    const x = Math.max(0, Math.min(Math.max(0, rect.width - visual.width), clientX - rect.left - visual.offsetX));
    const y = Math.max(0, Math.min(Math.max(0, rect.height - visual.height), clientY - rect.top - visual.offsetY));
    reorderGhostPendingRef.current = { x, y };
    if (reorderGhostRafRef.current !== null) return;
    reorderGhostRafRef.current = window.requestAnimationFrame(() => {
      reorderGhostRafRef.current = null;
      const pending = reorderGhostPendingRef.current;
      const ghost = reorderGhostRef.current;
      if (!pending || !ghost) return;
      ghost.style.transform = `translate3d(${pending.x}px, ${pending.y}px, 0)`;
    });
  }, []);
  const beginReorderDragVisual = React.useCallback((event: React.PointerEvent, panel: number) => {
    const container = canvasContainerRef.current;
    if (!container) return;
    const panelRect = event.currentTarget.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const offsetX = Math.max(0, Math.min(panelRect.width, event.clientX - panelRect.left));
    const offsetY = Math.max(0, Math.min(panelRect.height, event.clientY - panelRect.top));
    const x = Math.max(0, Math.min(Math.max(0, containerRect.width - panelRect.width), event.clientX - containerRect.left - offsetX));
    const y = Math.max(0, Math.min(Math.max(0, containerRect.height - panelRect.height), event.clientY - containerRect.top - offsetY));
    setReorderDragVisualValue({
      panel,
      label: panelLabel(panel),
      imageUrl: captureReorderPanelImage(panelRect, containerRect),
      width: panelRect.width,
      height: panelRect.height,
      x,
      y,
      offsetX,
      offsetY,
    });
    reorderGhostPendingRef.current = { x, y };
    requestAnimationFrame(() => updateReorderGhostPosition(event.clientX, event.clientY));
  }, [captureReorderPanelImage, panelLabel, setReorderDragVisualValue, updateReorderGhostPosition]);
  const clearReorderDragVisual = React.useCallback(() => {
    if (reorderGhostRafRef.current !== null) {
      window.cancelAnimationFrame(reorderGhostRafRef.current);
      reorderGhostRafRef.current = null;
    }
    reorderGhostPendingRef.current = null;
    reorderDragStartRef.current = null;
    reorderDragActivatedRef.current = false;
    setReorderDragVisualValue(null);
  }, [setReorderDragVisualValue]);
  const reorderDragHasPassedThreshold = React.useCallback((clientX: number, clientY: number) => {
    const start = reorderDragStartRef.current;
    if (!start) return true;
    if (reorderDragActivatedRef.current) return true;
    const distance = Math.hypot(clientX - start.x, clientY - start.y);
    if (distance < REORDER_DRAG_THRESHOLD_PX) return false;
    reorderDragActivatedRef.current = true;
    return true;
  }, []);
  const buildPanelMovedOrder = React.useCallback((
    source: number,
    target: number,
    placement: ReorderPlacement,
    baseOrder?: number[] | null,
  ): number[] | null => {
    if (source === target) return null;
    const base = Array.isArray(baseOrder) && baseOrder.length === totalPanelCount
      ? baseOrder
      : orderedPanelIndices;
    const next = [...base];
    const from = next.indexOf(source);
    if (from < 0) return null;
    next.splice(from, 1);
    const targetIndex = next.indexOf(target);
    if (targetIndex < 0) return null;
    const insertAt = placement === "after" ? targetIndex + 1 : targetIndex;
    next.splice(insertAt, 0, source);
    return next;
  }, [orderedPanelIndices, totalPanelCount]);
  const panelReorderTargetFromPoint = React.useCallback((clientX: number, clientY: number): { panel: number; placement: ReorderPlacement } | null => {
    if (typeof document === "undefined") return null;
    const elements = document.elementsFromPoint(clientX, clientY);
    let targetEl: HTMLElement | null = null;
    for (const element of elements) {
      if (!(element instanceof HTMLElement)) continue;
      const candidate = element.closest("[data-show3d-reorder-panel]");
      if (candidate instanceof HTMLElement) {
        targetEl = candidate;
        break;
      }
    }
    const allTargets = Array.from(document.querySelectorAll<HTMLElement>("[data-show3d-reorder-panel]"))
      .map((element) => {
        const rect = element.getBoundingClientRect();
        const raw = element.dataset.show3dReorderPanel;
        const panel = raw == null ? Number.NaN : Math.trunc(Number(raw));
        return { element, rect, panel };
      })
      .filter((item) => Number.isFinite(item.panel) && item.rect.width > 0 && item.rect.height > 0);
    if (!targetEl && allTargets.length) {
      let best = allTargets[0];
      let bestDistance = Number.POSITIVE_INFINITY;
      for (const item of allTargets) {
        const dx = Math.max(item.rect.left - clientX, 0, clientX - item.rect.right);
        const dy = Math.max(item.rect.top - clientY, 0, clientY - item.rect.bottom);
        const distance = dx * dx + dy * dy;
        if (distance < bestDistance) {
          best = item;
          bestDistance = distance;
        }
      }
      targetEl = best.element;
    }
    if (!targetEl) return null;
    const raw = targetEl.dataset.show3dReorderPanel;
    const panel = raw == null ? Number.NaN : Math.trunc(Number(raw));
    if (!Number.isFinite(panel) || panel < 0 || panel >= totalPanelCount) return null;
    const rect = targetEl.getBoundingClientRect();
    const sameRowNeighbor = allTargets.some((item) => item.panel !== panel && Math.abs(item.rect.top - rect.top) < 8);
    const sameColumnNeighbor = allTargets.some((item) => item.panel !== panel && Math.abs(item.rect.left - rect.left) < 8);
    const useHorizontal = sameRowNeighbor || !sameColumnNeighbor;
    const placement: ReorderPlacement = useHorizontal
      ? (clientX >= rect.left + rect.width / 2 ? "after" : "before")
      : (clientY >= rect.top + rect.height / 2 ? "after" : "before");
    return { panel, placement };
  }, [totalPanelCount]);
  const previewPanelReorderFromPoint = React.useCallback((clientX: number, clientY: number) => {
    const source = pointerReorderPanelRef.current ?? draggedPanelRef.current;
    if (source === null) return;
    const target = panelReorderTargetFromPoint(clientX, clientY);
    if (!target) return;
    setDragOverPanel(target.panel);
    const base = reorderPreviewOrderRef.current || orderedPanelIndices;
    const next = buildPanelMovedOrder(source, target.panel, target.placement, base);
    if (!next) return;
    const current = reorderPreviewOrderRef.current || orderedPanelIndices;
    if (next.length === current.length && next.every((value, idx) => value === current[idx])) return;
    setReorderPreviewOrderValue(next);
  }, [buildPanelMovedOrder, orderedPanelIndices, panelReorderTargetFromPoint, setReorderPreviewOrderValue]);
  const commitPanelReorderPreview = React.useCallback(() => {
    const next = reorderPreviewOrderRef.current;
    if (next) applyPanelOrder(next);
    setReorderPreviewOrderValue(null);
    setDragOverPanel(null);
    draggedPanelRef.current = null;
    pointerReorderPanelRef.current = null;
    clearReorderDragVisual();
  }, [applyPanelOrder, clearReorderDragVisual, setReorderPreviewOrderValue]);
  const cancelPanelReorderPreview = React.useCallback(() => {
    setReorderPreviewOrderValue(null);
    setDragOverPanel(null);
    draggedPanelRef.current = null;
    pointerReorderPanelRef.current = null;
    clearReorderDragVisual();
  }, [clearReorderDragVisual, setReorderPreviewOrderValue]);
  const handlePanelDragStart = React.useCallback((event: React.DragEvent, panel: number) => {
    if (!reorderMode) return;
    draggedPanelRef.current = panel;
    setReorderPreviewOrderValue(orderedPanelIndices);
    setDragOverPanel(panel);
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", String(panel));
    const blankDragImage = document.createElement("canvas");
    blankDragImage.width = 1;
    blankDragImage.height = 1;
    event.dataTransfer.setDragImage(blankDragImage, 0, 0);
    event.stopPropagation();
  }, [orderedPanelIndices, reorderMode, setReorderPreviewOrderValue]);
  const handlePanelDragOver = React.useCallback((event: React.DragEvent, panel: number) => {
    if (!reorderMode) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    if (dragOverPanel !== panel) setDragOverPanel(panel);
    previewPanelReorderFromPoint(event.clientX, event.clientY);
    event.stopPropagation();
  }, [dragOverPanel, previewPanelReorderFromPoint, reorderMode]);
  const handlePanelDrop = React.useCallback((event: React.DragEvent) => {
    if (!reorderMode) return;
    event.preventDefault();
    const raw = event.dataTransfer.getData("text/plain");
    const source = raw.trim() !== "" && Number.isFinite(Number(raw))
      ? Math.trunc(Number(raw))
      : draggedPanelRef.current;
    if (source !== null && source !== undefined) {
      draggedPanelRef.current = source;
      previewPanelReorderFromPoint(event.clientX, event.clientY);
    }
    commitPanelReorderPreview();
    event.stopPropagation();
  }, [commitPanelReorderPreview, previewPanelReorderFromPoint, reorderMode]);
  const handlePanelDragEnd = React.useCallback(() => {
    cancelPanelReorderPreview();
  }, [cancelPanelReorderPreview]);
  const handlePanelReorderPointerDown = React.useCallback((event: React.PointerEvent, panel: number) => {
    if (!reorderMode) return;
    pointerReorderPanelRef.current = panel;
    draggedPanelRef.current = panel;
    reorderDragStartRef.current = { x: event.clientX, y: event.clientY };
    reorderDragActivatedRef.current = false;
    setReorderPreviewOrderValue(orderedPanelIndices);
    setDragOverPanel(panel);
    beginReorderDragVisual(event, panel);
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Some browser automation paths do not expose pointer capture.
    }
    event.preventDefault();
    event.stopPropagation();
  }, [beginReorderDragVisual, orderedPanelIndices, reorderMode, setReorderPreviewOrderValue]);
  const handlePanelReorderPointerEnter = React.useCallback((event: React.PointerEvent, panel: number) => {
    if (!reorderMode || pointerReorderPanelRef.current === null) return;
    if (dragOverPanel !== panel) setDragOverPanel(panel);
    event.stopPropagation();
  }, [dragOverPanel, reorderMode]);
  const handlePanelReorderPointerMove = React.useCallback((event: React.PointerEvent) => {
    if (!reorderMode || pointerReorderPanelRef.current === null) return;
    updateReorderGhostPosition(event.clientX, event.clientY);
    if (!reorderDragHasPassedThreshold(event.clientX, event.clientY)) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    previewPanelReorderFromPoint(event.clientX, event.clientY);
    event.preventDefault();
    event.stopPropagation();
  }, [previewPanelReorderFromPoint, reorderDragHasPassedThreshold, reorderMode, updateReorderGhostPosition]);
  const handlePanelReorderPointerUp = React.useCallback((event: React.PointerEvent) => {
    if (!reorderMode) return;
    updateReorderGhostPosition(event.clientX, event.clientY);
    if (reorderDragHasPassedThreshold(event.clientX, event.clientY)) {
      previewPanelReorderFromPoint(event.clientX, event.clientY);
      commitPanelReorderPreview();
    } else {
      cancelPanelReorderPreview();
    }
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // Ignore capture release failures from synthetic pointer streams.
    }
    event.preventDefault();
    event.stopPropagation();
  }, [cancelPanelReorderPreview, commitPanelReorderPreview, previewPanelReorderFromPoint, reorderDragHasPassedThreshold, reorderMode, updateReorderGhostPosition]);
  const resetPanelOrder = React.useCallback(() => {
    setPanelOrder([]);
    cancelPanelReorderPreview();
  }, [cancelPanelReorderPreview, setPanelOrder]);
  React.useEffect(() => {
    if (((nPanels || 1) <= 1 || isPaged) && reorderMode) setReorderMode(false);
  }, [nPanels, isPaged, reorderMode]);
  React.useEffect(() => {
    if (reorderMode) return;
    cancelPanelReorderPreview();
  }, [cancelPanelReorderPreview, reorderMode]);
  React.useEffect(() => () => {
    if (reorderGhostRafRef.current !== null) {
      window.cancelAnimationFrame(reorderGhostRafRef.current);
      reorderGhostRafRef.current = null;
    }
  }, []);
  const [hiddenIndices] = useModelState<number[]>("hidden_indices");
  const hiddenSet = new Set(hiddenIndices || []);
  const nextVisible = (from: number, dir: 1 | -1, allowWrap = true): number => {
    if (!hiddenSet.size) return from + dir;
    let n = from + dir;
    while (n >= 0 && n < nSlices) {
      if (!hiddenSet.has(n)) return n;
      n += dir;
    }
    if (!allowWrap) return from;
    n = dir > 0 ? 0 : nSlices - 1;
    while (n !== from) {
      if (!hiddenSet.has(n)) return n;
      n += dir;
      if (n < 0 || n >= nSlices) return from;
    }
    return from;
  };
  const visibleCount = nSlices - hiddenSet.size;
  // If the user hides the currently-displayed slice, snap to next visible.
  React.useEffect(() => {
    if (!hiddenSet.has(sliceIdx)) return;
    const next = nextVisible(sliceIdx, 1, true);
    if (next !== sliceIdx) setSliceIdx(next);
  }, [hiddenIndices]);
  const [maxCols, setMaxCols] = useModelState<number>("max_cols");
  const [linkPanels, setLinkPanels] = useModelState<boolean>("link_panels");
  const [showResizeHandles] = useModelState<boolean>("show_resize_handles");
  const allowResizeControls = showResizeHandles !== false && !isMobileViewport;
  const [showZoomIndicator] = useModelState<boolean>("show_zoom_indicator");
  const [showPanelTitles] = useModelState<boolean>("show_panel_titles");
  const [panelTitleFontSize] = useModelState<number>("panel_title_font_size");
  const [panelGapTrait] = useModelState<number>("panel_gap");
  const [linkContrast, setLinkContrast] = useModelState<boolean>("link_contrast");
  const [cmap, setCmap] = useModelState<string>("cmap");

  // Playback
  const [playing, setPlaying] = useModelState<boolean>("playing");
  const [reverse, setReverse] = useModelState<boolean>("reverse");
  const [boomerang, setBoomerang] = useModelState<boolean>("boomerang");
  const [fps, setFpsModel] = useModelState<number>("fps");
  const playbackFps = clampPlaybackFps(fps);
  const setPlaybackFps = React.useCallback((value: number) => {
    setFpsModel(clampPlaybackFps(value));
  }, [setFpsModel]);
  React.useEffect(() => {
    if (fps !== playbackFps) setFpsModel(playbackFps);
  }, [fps, playbackFps, setFpsModel]);
  const [loop, setLoop] = useModelState<boolean>("loop");
  const [loopStart, setLoopStart] = useModelState<number>("loop_start");
  const [loopEnd, setLoopEnd] = useModelState<number>("loop_end");
  const [bookmarkedFrames] = useModelState<number[]>("bookmarked_frames");
  const [playbackPath] = useModelState<number[]>("playback_path");

  // Boomerang direction ref (avoids stale closure in setInterval)
  const bounceDirRef = React.useRef<1 | -1>(1);

  // Stats
  const [showStats, setShowStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [controlsCollapsed, setControlsCollapsed] = useModelState<boolean>("controls_collapsed");
  const controlsVisible = showControls && !controlsCollapsed;
  const panelChromeVisible = controlsVisible;
  const showResizeControls = allowResizeControls && panelChromeVisible;
  const [statsMean] = useModelState<number>("stats_mean");
  const [statsMin] = useModelState<number>("stats_min");
  const [statsMax] = useModelState<number>("stats_max");
  const [statsStd] = useModelState<number>("stats_std");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [percentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh] = useModelState<number>("percentile_high");
  const [traitVmin] = useModelState<number | null>("vmin");
  const [traitVmax] = useModelState<number | null>("vmax");
  const [imageVminPct, setImageVminPct] = useModelState<number>("image_vmin_pct");
  const [imageVmaxPct, setImageVmaxPct] = useModelState<number>("image_vmax_pct");
  const manualImageRangeBeforeAutoRef = React.useRef<{ min: number; max: number } | null>(null);
  const [vminPerPanel, setVminPerPanel] = useModelState<(number | null)[]>("vmin_per_panel");
  const [vmaxPerPanel, setVmaxPerPanel] = useModelState<(number | null)[]>("vmax_per_panel");
  const vminPerPanelLiveRef = React.useRef<(number | null)[]>(vminPerPanel);
  const vmaxPerPanelLiveRef = React.useRef<(number | null)[]>(vmaxPerPanel);
  React.useEffect(() => {
    vminPerPanelLiveRef.current = vminPerPanel;
  }, [vminPerPanel]);
  React.useEffect(() => {
    vmaxPerPanelLiveRef.current = vmaxPerPanel;
  }, [vmaxPerPanel]);
  const [dataMin] = useModelState<number>("data_min");
  const [dataMax] = useModelState<number>("data_max");
  const [autoVmins] = useModelState<number[]>("auto_vmins");
  const [autoVmaxs] = useModelState<number[]>("auto_vmaxs");
  // Scale bar
  const [pixelSize] = useModelState<number>("pixel_size");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [imageRotation] = useModelState<number>("image_rotation");

  // Customization
  const [canvasSizeTrait, setCanvasSizeTrait] = useModelState<number>("size");

  // ROI
  const [roiActive, setRoiActive] = useModelState<boolean>("roi_active");
  const [roiList, setRoiList] = useModelState<ROIItem[]>("roi_list");
  const [roiSelectedIdx, setRoiSelectedIdx] = useModelState<number>("roi_selected_idx");
  const [roiPlotData] = useModelState<DataView>("roi_plot_data");
  const [newRoiShape, setNewRoiShape] = React.useState<"circle" | "square" | "rectangle" | "annular">("square");

  // Diff mode
  const [diffMode, setDiffMode] = useModelState<string>("diff_mode");
  const [avgWindow, setAvgWindow] = useModelState<number>("avg_window");

  // FFT
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftLayout, setFftLayout] = useModelState<string>("fft_layout");
  const [fftOverlayPosition, setFftOverlayPosition] = useModelState<string>("fft_overlay_position");
  const [fftOverlaySize, setFftOverlaySize] = useModelState<number>("fft_overlay_size");
  const [fftOverlayZoomTrait, setFftOverlayZoomTrait] = useModelState<number>("fft_overlay_zoom");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const [fftMetricsTrait] = useModelState<boolean>("fft_metrics");
  const fftMetricsEnabled = fftMetricsTrait !== false;
  const resolvedFftLayout = (["bottom", "right", "overlay"].includes(String(fftLayout)) ? String(fftLayout) : "bottom") as "bottom" | "right" | "overlay";
  const fftLayoutBottom = resolvedFftLayout === "bottom";
  const fftLayoutOverlay = resolvedFftLayout === "overlay";
  const resolvedFftOverlayPosition = (["top-left", "top-right", "bottom-left", "bottom-right"].includes(String(fftOverlayPosition)) ? String(fftOverlayPosition) : "top-left") as FftOverlayPosition;
  const resolvedFftOverlaySize = Math.max(0.2, Math.min(0.7, Number.isFinite(fftOverlaySize) ? fftOverlaySize : 0.35));
  const resolvedFftOverlayZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, Number.isFinite(fftOverlayZoomTrait) ? fftOverlayZoomTrait : 1));


  // Playback buffer (sliding prefetch)
  const [bufferBytes] = useModelState<DataView>("_buffer_bytes");
  const [bufferStart] = useModelState<number>("_buffer_start");
  const [bufferCount] = useModelState<number>("_buffer_count");
  const [, setPrefetchRequest] = useModelState<number>("_prefetch_request");
  const [frameServerUrl] = useModelState<string>("frame_server_url");
  const [frameServerVersion] = useModelState<number>("frame_server_version");
  const [benchmarkRequest] = useModelState<Record<string, unknown>>("benchmark_request");
  const [, setBenchmarkResult] = useModelState<Record<string, unknown>>("benchmark_result");
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

  // Canvas refs
  const rootRef = React.useRef<HTMLDivElement>(null);
  const hasLiveFrameBytes = !!rawFrameBytes && rawFrameBytes.byteLength > 0;
  const hasOfflineStack = !!offlineStack && offlineStack.byteLength > 0;
  const hasOfflineFloatStack = !!offlineFloatStack && offlineFloatStack.byteLength > 0;
  const hasFrameServer = !offline && !!frameServerUrl;
  const canRenderLive = hasLiveFrameBytes || hasOfflineStack || hasOfflineFloatStack || hasFrameServer;
  const staticFallbackUrl = staticFallbackJpeg
    ? `data:${staticFallbackMime || "image/jpeg"};base64,${staticFallbackJpeg}`
    : "";
  const hasSavedStaticFallback = staticFallbackUrl.length > 0;
  useHideStaticFallback(model, rootRef, canRenderLive || hasSavedStaticFallback);
  const gpuCanvasCtxRef = React.useRef<GPUCanvasContext | null>(null);
  const gpuCanvasSizeRef = React.useRef<{ w: number; h: number } | null>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const uiRef = React.useRef<HTMLCanvasElement>(null);
  const canvasWheelHandlerRef = React.useRef<((event: WheelEvent) => void) | null>(null);
  const fftInsetNativeWheelHandlerRef = React.useRef<((event: WheelEvent) => boolean) | null>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftInsetLayerRef = React.useRef<HTMLCanvasElement>(null);

  const [exportMenuAnchor, setExportMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [panelMenuAnchor, setPanelMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [viewMenuAnchor, setViewMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [exportBusy, setExportBusy] = React.useState(false);
  const [localExportStatus, setLocalExportStatus] = React.useState("");
  const fftOverlayDragRef = React.useRef<{
    pointerId: number;
    startClientX: number;
    startClientY: number;
    startInsetX: number;
    startInsetY: number;
    panelLeft: number;
    panelTop: number;
    panelW: number;
    panelH: number;
    insetW: number;
    insetH: number;
    moved: boolean;
  } | null>(null);
  const [fftOverlayDragPreview, setFftOverlayDragPreview] = React.useState<{ x: number; y: number } | null>(null);
  const pendingExportRef = React.useRef<{
    id: string;
    filename: string;
    mode: string;
    downsample: number;
    handle: Show3DFileHandle | null;
  } | null>(null);
  React.useEffect(() => {
    if (!exportStatus) return;
    const preparing = exportStatus.startsWith("Preparing ") || exportStatus.startsWith("Exporting ");
    if (preparing) {
      setExportBusy(true);
    } else if (!pendingExportRef.current) {
      setExportBusy(false);
    }
  }, [exportStatus]);
  React.useEffect(() => {
    if (!localExportStatus || exportBusy) return;
    if (localExportStatus.startsWith("Preparing ") || localExportStatus.startsWith("Saving ")) return;
    const id = window.setTimeout(() => {
      setLocalExportStatus((current) => current === localExportStatus ? "" : current);
    }, 5000);
    return () => window.clearTimeout(id);
  }, [localExportStatus, exportBusy]);
  const voxelCount = Math.max(0, Math.floor(nSlices) * Math.floor(height) * Math.floor(width));
  const exactExportSize = formatEstimatedHtmlSize(voxelCount * 4);
  const quantizedExportSize = formatEstimatedHtmlSize(voxelCount);
  const quantizedExportSize2 = formatEstimatedHtmlSize(Math.ceil(voxelCount / 4));
  const quantizedExportSize4 = formatEstimatedHtmlSize(Math.ceil(voxelCount / 16));
  const quantizedExportSize8 = formatEstimatedHtmlSize(Math.ceil(voxelCount / 64));
  const gifLowEstimate = formatEstimatedAnimationWork(width, height, nSlices, visiblePanelCount, maxCols, panelGapTrait ?? 10, "low");
  const gifMediumEstimate = formatEstimatedAnimationWork(width, height, nSlices, visiblePanelCount, maxCols, panelGapTrait ?? 10, "medium");
  const gifHighEstimate = formatEstimatedAnimationWork(width, height, nSlices, visiblePanelCount, maxCols, panelGapTrait ?? 10, "high");
  const handleExportMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setExportMenuAnchor(event.currentTarget);
  };
  const handleExportMenuClose = () => {
    setExportMenuAnchor(null);
  };
  const handleExportSelect = async (mode: string, quality = "medium", downsample = 1) => {
    setExportMenuAnchor(null);
    if (mode !== "exact" && mode !== "quantized" && mode !== "gif" && mode !== "mp4") return;
    const filename = makeExportFilename(title, nSlices, height, width, mode, quality, downsample);
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setExportBusy(true);
    setLocalExportStatus("Choose export location...");
    const picker = (window as Show3DWindow).showSaveFilePicker;
    let handle: Show3DFileHandle | null = null;
    if (picker) {
      try {
        handle = await picker({
          suggestedName: filename,
          types: [exportPickerType(mode)],
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
    pendingExportRef.current = { id, filename, mode, downsample, handle };
    setLocalExportStatus(`Preparing ${filename}...`);
    setExportRequest(JSON.stringify({ mode, quality, downsample, id, filename, download: true }));
  };

  React.useEffect(() => {
    const pending = pendingExportRef.current;
    if (!pending || exportPayloadId !== pending.id) return;
    const bytes = extractBytes(exportPayload);
    if (bytes.length === 0) return;
    let canceled = false;
    const save = async () => {
      const payload = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? bytes
        : bytes.slice();
      const filename = exportPayloadFilename || pending.filename;
      const blob = new Blob([payload as BlobPart], { type: exportBlobType(pending.mode) });
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

  // Local state
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false);
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const resizeAspectRef = React.useRef<number | null>(null);
  const roiItems = (roiList || []).map((roi, i) => normalizeROI(roi, i));
  const selectedRoi = roiSelectedIdx >= 0 && roiSelectedIdx < roiItems.length ? roiItems[roiSelectedIdx] : null;
  const [showRoiResizeHint, setShowRoiResizeHint] = React.useState(true);
  const pendingRoiAddRef = React.useRef<{ row: number; col: number } | null>(null);

  // Preview panel state (JS-only, shows ROI crop at full resolution - auto-shows when ROI selected)
  const [previewZoom, setPreviewZoom] = React.useState({ zoom: 1, panX: 0, panY: 0 });
  const previewCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const previewOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const previewContainerRef = React.useRef<HTMLDivElement>(null);
  const [isDraggingPreviewPan, setIsDraggingPreviewPan] = React.useState(false);
  const [previewPanStart, setPreviewPanStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);
  const [previewCropDims, setPreviewCropDims] = React.useState<{ w: number; h: number } | null>(null);
  const previewOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const [previewVersion, setPreviewVersion] = React.useState(0);

  const updateSelectedRoi = (updates: Partial<ROIItem>) => {
    if (roiSelectedIdx < 0 || !roiList) return;
    const newList = [...roiList];
    newList[roiSelectedIdx] = { ...newList[roiSelectedIdx], ...updates };
    setRoiList(newList);
  };
  // Per-panel zoom/pan: index 0 is also used as the shared linked state.
  // Each panel keeps its own state when unlinked.
  type PanelState = {
    zoom: number;
    panX: number;
    panY: number;
    imageVminPct: number;
    imageVmaxPct: number;
  };
  type TouchTransformState = {
    panelIdx: number;
    mode: "pan" | "pinch";
    startX: number;
    startY: number;
    startDistance: number;
    startMidX: number;
    startMidY: number;
    startState: PanelState;
  };
  type FftTouchTransformState = {
    mode: "pan" | "pinch";
    startX: number;
    startY: number;
    startDistance: number;
    startMidX: number;
    startMidY: number;
    startState: { zoom: number; panX: number; panY: number };
  };
  const initialState: PanelState = {
    zoom: 1,
    panX: 0,
    panY: 0,
    imageVminPct: 0,
    imageVmaxPct: 100,
  };
  type Show3DViewState = {
    linked_state?: Partial<PanelState>;
    panel_states?: Partial<PanelState>[];
  };
  const [viewState, setViewState] = useModelState<Show3DViewState>("view_state");
  const readNumber = (value: unknown, fallback: number): number => (
    typeof value === "number" && Number.isFinite(value) ? value : fallback
  );
  const normalizePanelState = (value: Partial<PanelState> | undefined, fallback: PanelState): PanelState => ({
    zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, readNumber(value?.zoom, fallback.zoom))),
    panX: readNumber(value?.panX, readNumber((value as { pan_x?: unknown } | undefined)?.pan_x, fallback.panX)),
    panY: readNumber(value?.panY, readNumber((value as { pan_y?: unknown } | undefined)?.pan_y, fallback.panY)),
    imageVminPct: readNumber(value?.imageVminPct, readNumber((value as { image_vmin_pct?: unknown } | undefined)?.image_vmin_pct, fallback.imageVminPct)),
    imageVmaxPct: readNumber(value?.imageVmaxPct, readNumber((value as { image_vmax_pct?: unknown } | undefined)?.image_vmax_pct, fallback.imageVmaxPct)),
  });
  const savedPanelStates = Array.isArray(viewState?.panel_states)
    ? viewState.panel_states.map(v => normalizePanelState(v, initialState))
    : [initialState];
  const [linkedState, setLinkedState] = React.useState<PanelState>(() => normalizePanelState(viewState?.linked_state, savedPanelStates[0] || initialState));
  const [panelStates, setPanelStates] = React.useState<PanelState[]>(() => savedPanelStates.length ? savedPanelStates : [initialState]);
  const linkedStateLiveRef = React.useRef<PanelState>(linkedState);
  const panelStatesLiveRef = React.useRef<PanelState[]>(panelStates);
  const transformRenderRafRef = React.useRef<number | null>(null);
  const transformStateCommitTimerRef = React.useRef<number | null>(null);
  const transformInputAtRef = React.useRef(0);
  React.useEffect(() => {
    const n = Math.max(1, nPanels || 1);
    setPanelStates(prev => {
      if (prev.length === n) return prev;
      const next = Array.from({ length: n }, (_, i) => prev[i] || { ...initialState });
      return next;
    });
  }, [nPanels]);
  // Seamless toggle: on link→unlink, copy linkedState into every panel; on
  // unlink→link, copy panel 0 into linkedState. Single effect so both axes
  // sync atomically.
  const prevLinkRef = React.useRef(linkPanels);
  React.useEffect(() => {
    if (prevLinkRef.current && !linkPanels) {
      // Linked → unlinked: distribute linkedState to all panels
      const s = linkedState;
      setPanelStates(arr => {
        const next = arr.map(() => ({ ...s }));
        setViewState({ linked_state: { ...s }, panel_states: next.map(v => ({ ...v })) });
        return next;
      });
    } else if (!prevLinkRef.current && linkPanels) {
      // Unlinked → linked: adopt panel 0's state as the shared linked state
      const s0 = panelStates[0] || initialState;
      setLinkedState({ ...s0 });
      setViewState({ linked_state: { ...s0 }, panel_states: panelStates.map(v => ({ ...v })) });
    }
    prevLinkRef.current = linkPanels;
  }, [linkPanels]);
  const stateFor = (panelIdx: number): PanelState => {
    const livePanels = panelStatesLiveRef.current;
    return linkPanels
      ? linkedStateLiveRef.current
      : (livePanels[panelIdx] || panelStates[panelIdx] || initialState);
  };
  const syncPlaybackPanelTransform = (panelIdx: number, nextZoom: number, nextPanX: number, nextPanY: number) => {
    const clampAxis = (pan: number, viewport: number, zoomValue: number) => {
      if (viewport <= 0) return 0;
      if (zoomValue <= 1) return viewport * (1 - zoomValue) / 2;
      return Math.max(viewport * (1 - zoomValue), Math.min(0, pan));
    };
    const panelCount = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(panelCount);
    const rows = Math.max(1, Math.ceil(panelCount / cols));
    const gap = panelCount > 1 ? (panelGapTrait ?? 10) : 0;
    const viewportW = (canvasW - gap * (cols - 1)) / cols;
    const viewportH = (canvasH - gap * (rows - 1)) / rows;
    const zoomValue = Math.max(MIN_IMAGE_ZOOM, Math.min(MAX_ZOOM, nextZoom));
    const panXValue = clampAxis(nextPanX, viewportW, zoomValue);
    const panYValue = clampAxis(nextPanY, viewportH, zoomValue);
    const c = playRef.current;
    if (c.linkPanels) {
      const nextLinked = { ...c.linkedState, zoom: zoomValue, panX: panXValue, panY: panYValue };
      c.linkedState = nextLinked;
      linkedStateLiveRef.current = nextLinked;
    } else {
      const next = c.panelStates.slice();
      const prev = next[panelIdx] || initialState;
      next[panelIdx] = { ...prev, zoom: zoomValue, panX: panXValue, panY: panYValue };
      c.panelStates = next;
      panelStatesLiveRef.current = next;
    }
  };
  // Back-compat aliases for the single-panel code paths (ROI, profile, etc.)
  // which still expect plain zoom/panX/panY. Use panel 0's state.
  const zoom = stateFor(0).zoom;
  const panX = stateFor(0).panX;
  const panY = stateFor(0).panY;
  const [isDraggingPan, setIsDraggingPan] = React.useState(false);
  const [panStart, setPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);
  const panStartPanelRef = React.useRef<number>(0);
  const [mainCanvasSize, setMainCanvasSize] = React.useState(CANVAS_TARGET_SIZE);
  const rawFrameDataRef = React.useRef<Float32Array | null>(null);
  const initialCanvasSizeRef = React.useRef<number>(canvasSizeTrait > 0 ? canvasSizeTrait : CANVAS_TARGET_SIZE);
  const panelColsForCount = React.useCallback((count: number) => {
    const n = Math.max(1, count || 1);
    const requestedCols = (maxCols && maxCols > 0) ? Math.min(maxCols, n, MAX_PANEL_COLUMNS) : Math.min(n, MAX_PANEL_COLUMNS);
    if (n <= 1) return 1;
    return Math.max(1, requestedCols);
  }, [maxCols]);
  const show3dColumnOptions = React.useMemo(() => {
    const n = Math.max(1, visiblePanelCount || 1);
    const values = new Set<number>([1, 2, 3, 4, 5, 6, 8, 10, 12]);
    return Array.from(values).filter((cols) => cols >= 1 && cols <= n).sort((a, b) => a - b);
  }, [visiblePanelCount]);
  const clampedMaxCols = Math.max(1, Math.min((maxCols && maxCols > 0) ? maxCols : visiblePanelCount || 1, visiblePanelCount || 1, MAX_PANEL_COLUMNS));

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<CursorInfo | null>(null);
  const [cursorReadoutVisible, setCursorReadoutVisible] = React.useState(false);
  const cursorReadoutVisibleRef = React.useRef(false);
  const cursorInfoPendingRef = React.useRef<CursorInfo | null>(null);
  const cursorInfoRafRef = React.useRef<number | null>(null);
  const [showRoiPlot, setShowRoiPlot] = React.useState(true);
  const roiPlotCanvasRef = React.useRef<HTMLCanvasElement>(null);

  // Lens (magnifier inset)
  const [showLens, setShowLens] = React.useState(false);
  const [lensPos, setLensPos] = React.useState<{ row: number; col: number } | null>(null);
  const [lensMag, setLensMag] = React.useState(4);
  const [lensDisplaySize, setLensDisplaySize] = React.useState(128);
  const [lensAnchor, setLensAnchor] = React.useState<{ x: number; y: number } | null>(null);
  const [isDraggingLens, setIsDraggingLens] = React.useState(false);
  const lensCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const lensDragStartRef = React.useRef<{ mx: number; my: number; ax: number; ay: number } | null>(null);
  const [isResizingLens, setIsResizingLens] = React.useState(false);
  const [isHoveringLensEdge, setIsHoveringLensEdge] = React.useState(false);
  const lensResizeStartRef = React.useRef<{ my: number; startSize: number } | null>(null);

  const scheduleCursorInfo = React.useCallback((next: CursorInfo | null) => {
    cursorInfoPendingRef.current = next;
    if (cursorReadoutVisibleRef.current !== Boolean(next)) {
      cursorReadoutVisibleRef.current = Boolean(next);
      setCursorReadoutVisible(Boolean(next));
    }
    if (cursorInfoRafRef.current != null) return;
    if (typeof window === "undefined" || typeof window.requestAnimationFrame !== "function") {
      if (next) setCursorInfo(next);
      return;
    }
    cursorInfoRafRef.current = window.requestAnimationFrame(() => {
      cursorInfoRafRef.current = null;
      const pending = cursorInfoPendingRef.current;
      if (!pending) return;
      setCursorInfo((prev) => (
        prev &&
        prev.row === pending.row &&
        prev.col === pending.col &&
        prev.panelIdx === pending.panelIdx &&
        prev.value === pending.value
          ? prev
          : pending
      ));
    });
  }, []);

  React.useEffect(() => () => {
    if (cursorInfoRafRef.current != null && typeof window !== "undefined") {
      window.cancelAnimationFrame(cursorInfoRafRef.current);
    }
  }, []);

  // Reusable rendering buffers (avoid per-frame allocation)
  const mainOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const mainImgDataRef = React.useRef<ImageData | null>(null);
  const scaledPlaybackImgDataRef = React.useRef<{ width: number; height: number; imageData: ImageData } | null>(null);
  const scaledPlaybackMapRef = React.useRef<{
    srcW: number;
    srcH: number;
    outW: number;
    outH: number;
    xMap: Uint32Array;
    yMap: Uint32Array;
  } | null>(null);
  const logBufferRef = React.useRef<Float32Array | null>(null);

  // Playback buffer refs (double-buffer: current + next to avoid overwrite stalls)
  const bufferRef = React.useRef<Float32Array | null>(null);
  const bufferStartRef = React.useRef(0);
  const bufferCountRef = React.useRef(0);
  const nextBufferRef = React.useRef<Float32Array | null>(null);
  const nextBufferStartRef = React.useRef(0);
  const nextBufferCountRef = React.useRef(0);
  const prefetchPendingRef = React.useRef(false);
  // Seed from the model's slice_idx (not 0): on mount the not-playing branch
  // of the playback effect syncs this ref back onto slice_idx in offline mode,
  // and a stale 0 would clobber a baked middle-slice start.
  const playbackIdxRef = React.useRef(Number.isFinite(sliceIdx) ? sliceIdx : 0);
  const playbackSliderRef = React.useRef<HTMLSpanElement>(null);
  const playbackLiveCountRef = React.useRef<HTMLElement>(null);
  const frameFetchCacheRef = React.useRef<Map<number, Float32Array>>(new Map());
  const frameFetchPendingRef = React.useRef<Map<number, Promise<Float32Array | null>>>(new Map());
  const panelGpuFramePendingRef = React.useRef<Map<number, Promise<boolean>>>(new Map());
  const frameFetchSerialRef = React.useRef(0);
  const localAutoVminsRef = React.useRef<number[]>([]);
  const localAutoVmaxsRef = React.useRef<number[]>([]);
  const autoRangeComputeTokenRef = React.useRef(0);

  const [displaySliceIdx, setDisplaySliceIdx] = React.useState(sliceIdx);
  const [playbackUiSliceIdx, setPlaybackUiSliceIdx] = React.useState(sliceIdx);
  const [localStats, setLocalStats] = React.useState<{ mean: number; min: number; max: number; std: number } | null>(null);
  const [localPanelStats, setLocalPanelStats] = React.useState<PanelStats[] | null>(null);

  // WebGPU FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const gpuFftInitPromiseRef = React.useRef<Promise<WebGPUFFT | null> | null>(null);
  const offlineFftGpuDisabledRef = React.useRef(false);
  const offlineFftGpuInFlightRef = React.useRef(false);
  const [, setGpuReady] = React.useState(false);  // value unused; setter gates FFT-ready re-renders
  const [fftBackendInfo, setFftBackendInfo] = React.useState<{
    webgpu: "unknown" | "ready" | "software" | "unavailable";
    adapter: string;
    source: string;
    ms: number | null;
    panels: number | null;
    grid: string;
  }>({ webgpu: "unknown", adapter: "", source: "", ms: null, panels: null, grid: "" });
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const kymoOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  // WebGPU colormap engine (GPU-accelerated colormap for 4K frames)
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const gpuCmapReadyRef = React.useRef(false);
  const gpuFrameCacheUploadedRef = React.useRef<Set<number>>(new Set());
  const gpuUploadRef = React.useRef<{
    source: Float32Array | null;
    data: Float32Array | null;
    width: number;
    height: number;
    logScale: boolean;
  } | null>(null);
  const gpuRenderSerialRef = React.useRef(0);
  const gpuDisplayVisibleRef = React.useRef<boolean | null>(null);
  const [gpuDisplayVisible, setGpuDisplayVisibleState] = React.useState(false);

  const ensureFftGpu = React.useCallback(async (): Promise<WebGPUFFT | null> => {
    if (gpuFFTRef.current) return gpuFFTRef.current;
    if (!gpuFftInitPromiseRef.current) {
      gpuFftInitPromiseRef.current = getWebGPUFFT().then(fft => {
        if (fft) {
          const info = getGPUInfo();
          if (/swiftshader|software/i.test(info)) {
            console.log(`[Show3D] Software WebGPU adapter detected (${info}); using CPU FFT fallback`);
            setFftBackendInfo(prev => ({ ...prev, webgpu: "software", adapter: info || "software adapter" }));
            return null;
          }
          gpuFFTRef.current = fft;
          setGpuReady(true);
          setFftBackendInfo(prev => ({ ...prev, webgpu: "ready", adapter: info || "GPU" }));
          console.log(`[Show3D] WebGPU FFT initialized - ${info || "GPU"}`);
        } else {
          setFftBackendInfo(prev => ({ ...prev, webgpu: "unavailable", adapter: "" }));
          console.log("[Show3D] WebGPU FFT unavailable - CPU fallback will be used");
        }
        return fft;
      }).catch(err => {
        console.warn("[Show3D] WebGPU FFT init failed; CPU fallback will be used.", err);
        setFftBackendInfo(prev => ({ ...prev, webgpu: "unavailable", adapter: "" }));
        return null;
      });
    }
    return gpuFftInitPromiseRef.current;
  }, []);

  const setGpuDisplayVisible = React.useCallback((visible: boolean) => {
    if (gpuDisplayVisibleRef.current === visible) return;
    gpuDisplayVisibleRef.current = visible;
    const gpuCanvas = gpuCanvasRef.current;
    const canvas = canvasRef.current;
    const gpuVisible = ENABLE_GPU_CANVAS_DISPLAY && visible;
    setGpuDisplayVisibleState(gpuVisible);
    if (gpuCanvas) gpuCanvas.style.opacity = gpuVisible ? "1" : "0";
    if (canvas) {
      canvas.style.opacity = gpuVisible ? "0" : "1";
      canvas.style.display = "block";
    }
  }, []);

  const ensureGpuDisplayContext = React.useCallback((
    engine: GPUColormapEngine,
    w: number,
    h: number,
  ): GPUCanvasContext | null => {
    const canvas = gpuCanvasRef.current;
    if (!canvas) return null;
    const widthPx = Math.max(1, Math.round(w));
    const heightPx = Math.max(1, Math.round(h));
    const size = gpuCanvasSizeRef.current;
    if (!gpuCanvasCtxRef.current || !size || size.w !== widthPx || size.h !== heightPx) {
      gpuCanvasCtxRef.current = engine.configureCanvas(canvas, widthPx, heightPx);
      gpuCanvasSizeRef.current = { w: widthPx, h: heightPx };
    }
    return gpuCanvasCtxRef.current;
  }, []);

  const ensureLocalAutoRange = React.useCallback((
    idx: number,
    data: Float32Array,
    low: number,
    high: number,
  ): { vmin: number; vmax: number } => {
    const synced = cachedAutoRange(autoVmins, autoVmaxs, idx);
    if (synced) return synced;
    const local = cachedAutoRange(localAutoVminsRef.current, localAutoVmaxsRef.current, idx);
    if (local) return local;

    const range = percentileClip(data, low, high);
    const needed = Math.max(nSlices, idx + 1);
    while (localAutoVminsRef.current.length < needed) {
      localAutoVminsRef.current.push(Number.NaN);
      localAutoVmaxsRef.current.push(Number.NaN);
    }
    localAutoVminsRef.current[idx] = range.vmin;
    localAutoVmaxsRef.current[idx] = range.vmax;
    return range;
  }, [autoVmins, autoVmaxs, nSlices]);

  React.useEffect(() => {
    localAutoVminsRef.current = [];
    localAutoVmaxsRef.current = [];
    autoRangeComputeTokenRef.current++;
  }, [percentileLow, percentileHigh, nSlices, width, height]);

  const [gpuCmapReady, setGpuCmapReady] = React.useState(false);
  React.useEffect(() => {
    let disposed = false;
    ensureFftGpu().then(fft => {
      if (disposed || !fft) return;
      gpuFFTRef.current = fft;
      setGpuReady(true);
    });
    createGPUColormapEngine().then(engine => {
      if (disposed) {
        engine?.destroy();
        return;
      }
      if (engine) {
        gpuCmapRef.current = engine;
        gpuCmapReadyRef.current = true;
        // State counterpart of the ref so downstream useEffects re-fire
        // when the GPU engine becomes available. Without this, the data
        // effect that fires at mount paints via the CPU fallback BEFORE
        // the engine is ready and never re-paints when it IS ready.
        setGpuCmapReady(true);
      }
    });
    return () => {
      disposed = true;
      gpuCmapRef.current?.destroy();
      gpuCmapRef.current = null;
      gpuCmapReadyRef.current = false;
      setGpuCmapReady(false);
      gpuCanvasCtxRef.current = null;
      gpuCanvasSizeRef.current = null;
      frameFetchSerialRef.current++;
      frameFetchCacheRef.current.clear();
      frameFetchPendingRef.current.clear();
      panelGpuFramePendingRef.current.clear();
      gpuFrameCacheUploadedRef.current.clear();
    };
  }, []);

  const getFrameServerCacheLimit = React.useCallback(() => {
    const frameByteLength = Math.max(1, width * height * 4);
    const stackByteLength = frameByteLength * Math.max(1, nSlices);
    const cacheBudget = stackByteLength <= FRAME_SERVER_JS_FULL_STACK_CACHE_BYTES
      ? stackByteLength
      : FRAME_SERVER_STREAM_CACHE_BYTES;
    const budgetFrames = Math.floor(cacheBudget / frameByteLength);
    const minFrames = frameByteLength <= cacheBudget / FRAME_SERVER_MIN_CACHE_FRAMES
      ? FRAME_SERVER_MIN_CACHE_FRAMES
      : 1;
    return Math.max(1, Math.min(Math.max(1, nSlices), Math.max(minFrames, budgetFrames)));
  }, [width, height, nSlices]);

  const getSeparatePanelGpuCacheLimit = React.useCallback(() => {
    const visiblePanels = Math.max(1, visiblePanelCount || 1);
    const frameByteLength = Math.max(1, panelWidthPx * height * 4 * visiblePanels);
    const budgetFrames = Math.floor(FRAME_SERVER_SEPARATE_PANEL_GPU_CACHE_BYTES / frameByteLength);
    const minFrames = frameByteLength <= FRAME_SERVER_SEPARATE_PANEL_GPU_CACHE_BYTES / FRAME_SERVER_MIN_CACHE_FRAMES
      ? FRAME_SERVER_MIN_CACHE_FRAMES
      : 1;
    return Math.max(1, Math.min(Math.max(1, nSlices), Math.max(minFrames, budgetFrames)));
  }, [panelWidthPx, height, visiblePanelCount, nSlices]);

  const releasePanelGpuFrame = React.useCallback((idx: number) => {
    const normalized = ((Math.round(idx) % Math.max(1, nSlices)) + Math.max(1, nSlices)) % Math.max(1, nSlices);
    const engine = gpuCmapRef.current;
    const n = Math.max(1, Math.round(nPanels || 1));
    for (let panel = 0; panel < n; panel++) {
      engine?.releaseSlot(normalized * n + panel);
    }
    gpuFrameCacheUploadedRef.current.delete(normalized);
  }, [nPanels, nSlices]);

  const getCachedServerFrame = React.useCallback((idx: number): Float32Array | null => {
    const cache = frameFetchCacheRef.current;
    const frame = cache.get(idx);
    if (!frame) return null;
    cache.delete(idx);
    cache.set(idx, frame);
    return frame;
  }, []);

  const putCachedServerFrame = React.useCallback((idx: number, frame: Float32Array) => {
    const cache = frameFetchCacheRef.current;
    if (cache.has(idx)) cache.delete(idx);
    cache.set(idx, frame);
    const limit = getFrameServerCacheLimit();
    while (cache.size > limit) {
      const oldest = cache.keys().next().value;
      if (oldest === undefined) break;
      cache.delete(oldest);
    }
  }, [getFrameServerCacheLimit]);

  React.useEffect(() => {
    frameFetchSerialRef.current++;
    frameFetchCacheRef.current.clear();
    frameFetchPendingRef.current.clear();
    panelGpuFramePendingRef.current.clear();
    gpuFrameCacheUploadedRef.current.clear();
    gpuCmapRef.current?.destroy();
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.frameServerUrl = frameServerUrl || "";
      dbg.frameServerVersion = frameServerVersion;
      dbg.frameFetchCacheSize = 0;
      dbg.frameFetchPendingSize = 0;
    }
  }, [frameServerUrl, frameServerVersion, width, height, nSlices]);

  const fetchFrameFromServer = React.useCallback(async (idx: number): Promise<Float32Array | null> => {
    if (offline || !frameServerUrl || width <= 0 || height <= 0 || nSlices <= 0) return null;
    const normalized = ((Math.round(idx) % nSlices) + nSlices) % nSlices;
    const cached = getCachedServerFrame(normalized);
    const dbg = show3dPerfDebug();
    if (cached) {
      if (dbg) {
        dbg.frameFetchHits = ((dbg.frameFetchHits as number | undefined) ?? 0) + 1;
        dbg.frameFetchCacheSize = frameFetchCacheRef.current.size;
      }
      return cached;
    }
    const pending = frameFetchPendingRef.current.get(normalized);
    if (pending) return pending;

    let url: URL;
    try {
      url = new URL(frameServerUrl);
    } catch {
      return null;
    }
    url.searchParams.set("idx", String(normalized));
    url.searchParams.set("version", String(frameServerVersion));

    const serial = frameFetchSerialRef.current;
    const t0 = performance.now();
    let promise!: Promise<Float32Array | null>;
    promise = (async () => {
      try {
        if (dbg) {
          dbg.frameFetchMisses = ((dbg.frameFetchMisses as number | undefined) ?? 0) + 1;
          dbg.frameFetchPendingSize = frameFetchPendingRef.current.size + 1;
        }
        const response = await fetch(url.toString(), { cache: "no-store" });
        if (!response.ok) throw new Error(`frame fetch ${response.status}`);
        const buffer = await response.arrayBuffer();
        if (serial !== frameFetchSerialRef.current) return null;
        const expectedBytes = width * height * 4;
        if (buffer.byteLength !== expectedBytes) {
          throw new Error(`expected ${expectedBytes} bytes, got ${buffer.byteLength}`);
        }
        const frame = new Float32Array(buffer);
        putCachedServerFrame(normalized, frame);
        if (dbg) {
          dbg.lastFrameFetchMs = performance.now() - t0;
          dbg.lastFetchedFrame = normalized;
          dbg.frameFetchCacheSize = frameFetchCacheRef.current.size;
        }
        return frame;
      } catch (err) {
        if (dbg) {
          dbg.lastFrameFetchError = err instanceof Error ? err.message : String(err);
          dbg.lastFrameFetchErrorAt = performance.now();
        }
        return null;
      }
    })().finally(() => {
      if (frameFetchPendingRef.current.get(normalized) === promise) {
        frameFetchPendingRef.current.delete(normalized);
      }
      const d = show3dPerfDebug();
      if (d) d.frameFetchPendingSize = frameFetchPendingRef.current.size;
    });
    frameFetchPendingRef.current.set(normalized, promise);
    return promise;
  }, [offline, frameServerUrl, frameServerVersion, width, height, nSlices, getCachedServerFrame, putCachedServerFrame]);

  const fetchPanelFrameFromServer = React.useCallback(async (idx: number, panel: number): Promise<Float32Array | null> => {
    if (offline || !frameServerUrl || panelWidthPx <= 0 || height <= 0 || nSlices <= 0) return null;
    const normalized = ((Math.round(idx) % nSlices) + nSlices) % nSlices;
    const panelIdx = Math.max(0, Math.min(Math.max(0, nPanels - 1), Math.round(panel)));
    let url: URL;
    try {
      url = new URL(frameServerUrl);
    } catch {
      return null;
    }
    url.searchParams.set("idx", String(normalized));
    url.searchParams.set("panel", String(panelIdx));
    url.searchParams.set("version", String(frameServerVersion));

    const serial = frameFetchSerialRef.current;
    const t0 = performance.now();
    const dbg = show3dPerfDebug();
    try {
      if (dbg) {
        dbg.panelFrameFetchAttempts = ((dbg.panelFrameFetchAttempts as number | undefined) ?? 0) + 1;
        dbg.lastPanelFrameFetch = `${normalized}:${panelIdx}`;
      }
      const response = await fetch(url.toString(), { cache: "no-store" });
      if (!response.ok) throw new Error(`panel frame fetch ${response.status}`);
      const buffer = await response.arrayBuffer();
      if (serial !== frameFetchSerialRef.current) return null;
      const expectedBytes = panelWidthPx * height * 4;
      if (buffer.byteLength !== expectedBytes) {
        throw new Error(`expected ${expectedBytes} panel bytes, got ${buffer.byteLength}`);
      }
      if (dbg) dbg.lastPanelFrameFetchMs = performance.now() - t0;
      return new Float32Array(buffer);
    } catch (err) {
      if (dbg) {
        // Real misses only (failed fetch), not every attempt - the old counter
        // incremented at the top of try and read as "~every request missed".
        dbg.panelFrameFetchMisses = ((dbg.panelFrameFetchMisses as number | undefined) ?? 0) + 1;
        dbg.lastPanelFrameFetchError = err instanceof Error ? err.message : String(err);
        dbg.lastPanelFrameFetchErrorAt = performance.now();
      }
      return null;
    }
  }, [offline, frameServerUrl, frameServerVersion, panelWidthPx, height, nSlices, nPanels]);

  React.useEffect(() => {
    if (!separatePanelFrames) return;
    // Separate-panel GPU slots are keyed by frame plus panel index. A page
    // change swaps the visible panel set, so the old "frame is uploaded" mark
    // cannot be reused for the new page's panel slots.
    gpuFrameCacheUploadedRef.current.clear();
    panelGpuFramePendingRef.current.clear();
  }, [hiddenPanels, separatePanelFrames, visiblePanelIndices]);

  const ensurePanelFrameGpu = React.useCallback(async (
    idx: number,
    rgbaCapacityHint?: number,
  ): Promise<boolean> => {
    if (offline || !separatePanelFrames || !frameServerUrl || width <= 0 || height <= 0 || nSlices <= 0) return false;
    const normalized = ((Math.round(idx) % nSlices) + nSlices) % nSlices;
    if (gpuFrameCacheUploadedRef.current.has(normalized)) {
      gpuFrameCacheUploadedRef.current.delete(normalized);
      gpuFrameCacheUploadedRef.current.add(normalized);
      return true;
    }
    const pending = panelGpuFramePendingRef.current.get(normalized);
    if (pending) return pending;

    const promise = (async () => {
      while (!gpuCmapReadyRef.current || !gpuCmapRef.current) {
        await new Promise<void>(resolve => setTimeout(resolve, 25));
      }
      const engine = gpuCmapRef.current;
      if (!engine) return false;
      try {
        const n = Math.max(1, Math.round(nPanels || 1));
        for (const panel of visiblePanelIndices) {
          const frame = await fetchPanelFrameFromServer(normalized, panel);
          if (!frame) {
            const dbg = show3dPerfDebug();
            if (dbg) {
              dbg.lastPanelGpuMissFrame = normalized;
              dbg.lastPanelGpuMissPanel = panel;
            }
            return false;
          }
          engine.uploadData(normalized * n + panel, frame, panelWidthPx, height, rgbaCapacityHint, true);
        }
      } catch (err) {
        const dbg = show3dPerfDebug();
        if (dbg) {
          dbg.lastPanelGpuUploadFrame = normalized;
          dbg.lastPanelGpuUploadError = err instanceof Error ? err.message : String(err);
        }
        return false;
      }
      gpuFrameCacheUploadedRef.current.add(normalized);
      const cacheLimit = getSeparatePanelGpuCacheLimit();
      while (gpuFrameCacheUploadedRef.current.size > cacheLimit) {
        let oldest = gpuFrameCacheUploadedRef.current.keys().next().value;
        if (oldest === undefined) break;
        if (oldest === normalized) {
          gpuFrameCacheUploadedRef.current.delete(oldest);
          gpuFrameCacheUploadedRef.current.add(oldest);
          oldest = gpuFrameCacheUploadedRef.current.keys().next().value;
          if (oldest === undefined || oldest === normalized) break;
        }
        releasePanelGpuFrame(oldest);
      }
      const dbg = show3dPerfDebug();
      if (dbg) {
        dbg.gpuPreloadDone = gpuFrameCacheUploadedRef.current.size;
        dbg.gpuFrameCacheUploaded = gpuFrameCacheUploadedRef.current.size;
        dbg.gpuPanelCacheLimit = cacheLimit;
        dbg.gpuPanelCacheLayout = "panel-slots";
        dbg.lastFrameSource = "gpu-panel-cache-slots";
      }
      return true;
    })().finally(() => {
      if (panelGpuFramePendingRef.current.get(normalized) === promise) {
        panelGpuFramePendingRef.current.delete(normalized);
      }
    });
    panelGpuFramePendingRef.current.set(normalized, promise);
    return promise;
  }, [
    offline,
    separatePanelFrames,
    frameServerUrl,
    width,
    height,
    nSlices,
    nPanels,
    visiblePanelIndices,
    panelWidthPx,
    height,
    fetchPanelFrameFromServer,
    getSeparatePanelGpuCacheLimit,
    releasePanelGpuFrame,
  ]);

  React.useEffect(() => {
    if (offline || !frameServerUrl || width <= 0 || height <= 0 || nSlices <= 0) return;
    if (separatePanelFrames) return;
    const frameByteLength = Math.max(1, width * height * 4);
    const stackByteLength = frameByteLength * Math.max(1, nSlices);
    if (stackByteLength > FRAME_SERVER_JS_FULL_STACK_CACHE_BYTES) return;
    let cancelled = false;
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.frameFetchPreloadTarget = nSlices;
      dbg.frameFetchPreloadDone = 0;
      dbg.frameFetchPreloadActive = true;
    }
    void (async () => {
      for (let i = 0; i < nSlices; i++) {
        if (cancelled) break;
        await fetchFrameFromServer(i);
        const d = show3dPerfDebug();
        if (d) {
          d.frameFetchPreloadDone = i + 1;
          d.frameFetchCacheSize = frameFetchCacheRef.current.size;
        }
        await new Promise<void>(resolve => setTimeout(resolve, 0));
      }
      const d = show3dPerfDebug();
      if (d) d.frameFetchPreloadActive = false;
    })();
    return () => {
      cancelled = true;
      const d = show3dPerfDebug();
      if (d) d.frameFetchPreloadActive = false;
    };
  }, [offline, frameServerUrl, frameServerVersion, width, height, nSlices, fetchFrameFromServer, separatePanelFrames]);

  const prefetchServerFrames = React.useCallback((
    startIdx: number,
    reversePlayback = false,
    loopPlayback = false,
    loopStartIdx = 0,
    loopEndIdx = -1,
  ) => {
    if (offline || !frameServerUrl || nSlices <= 0) return;
    if (separatePanelFrames) return;
    const dir = reversePlayback ? -1 : 1;
    const rangeStart = loopPlayback ? Math.max(0, Math.min(loopStartIdx, nSlices - 1)) : 0;
    const rangeEnd = loopPlayback
      ? Math.max(rangeStart, Math.min(loopEndIdx < 0 ? nSlices - 1 : loopEndIdx, nSlices - 1))
      : nSlices - 1;
    const rangeSize = Math.max(1, rangeEnd - rangeStart + 1);
    for (let step = 0; step < FRAME_SERVER_PREFETCH_FRAMES; step++) {
      let next = Math.round(startIdx) + dir * step;
      if (loopPlayback) {
        while (next < rangeStart) next += rangeSize;
        while (next > rangeEnd) next -= rangeSize;
      } else {
        next = ((next % nSlices) + nSlices) % nSlices;
      }
      void fetchFrameFromServer(next);
    }
  }, [offline, frameServerUrl, nSlices, fetchFrameFromServer, separatePanelFrames]);

  const prefetchPanelGpuFrames = React.useCallback((
    startIdx: number,
    reversePlayback = false,
    loopPlayback = false,
    loopStartIdx = 0,
    loopEndIdx = -1,
  ) => {
    if (offline || !frameServerUrl || !separatePanelFrames || nSlices <= 0) return;
    const dir = reversePlayback ? -1 : 1;
    const rangeStart = loopPlayback ? Math.max(0, Math.min(loopStartIdx, nSlices - 1)) : 0;
    const rangeEnd = loopPlayback
      ? Math.max(rangeStart, Math.min(loopEndIdx < 0 ? nSlices - 1 : loopEndIdx, nSlices - 1))
      : nSlices - 1;
    const rangeSize = Math.max(1, rangeEnd - rangeStart + 1);
    const count = Math.min(FRAME_SERVER_PREFETCH_FRAMES, getSeparatePanelGpuCacheLimit());
    const live = playRef.current;
    const rgbaCapacity = Math.max(1, Math.round(live.canvasW * live.canvasH));
    for (let step = 0; step < count; step++) {
      let next = Math.round(startIdx) + dir * step;
      if (loopPlayback) {
        while (next < rangeStart) next += rangeSize;
        while (next > rangeEnd) next -= rangeSize;
      } else {
        next = ((next % nSlices) + nSlices) % nSlices;
      }
      void ensurePanelFrameGpu(next, rgbaCapacity);
    }
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.gpuPanelPrefetchCount = count;
      dbg.gpuPanelCacheLimit = getSeparatePanelGpuCacheLimit();
    }
  }, [
    offline,
    frameServerUrl,
    separatePanelFrames,
    nSlices,
    ensurePanelFrameGpu,
    getSeparatePanelGpuCacheLimit,
  ]);

  // Parse incoming playback buffer (double-buffer to avoid overwrite stalls)
  React.useEffect(() => {
    if (!bufferBytes || bufferBytes.byteLength === 0) return;
    const parsed = extractFloat32(bufferBytes, Math.max(0, bufferCount) * width * height);
    if (!parsed) return;
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.lastBufferByteLength = bufferBytes.byteLength;
      dbg.lastParsedFloatLength = parsed.length;
      dbg.lastBufferStart = bufferStart;
      dbg.lastBufferCount = bufferCount;
      dbg.lastBufferAt = performance.now();
    }
    if (!bufferRef.current || bufferCountRef.current === 0) {
      // No active buffer - use as current (initial load)
      bufferRef.current = parsed;
      bufferStartRef.current = bufferStart;
      bufferCountRef.current = bufferCount;
    } else {
      // Active buffer exists - store as next (prefetch)
      nextBufferRef.current = parsed;
      nextBufferStartRef.current = bufferStart;
      nextBufferCountRef.current = bufferCount;
    }
    prefetchPendingRef.current = false;

    if (autoContrast && !logScale && width > 0 && height > 0 && nSlices > 0 && bufferCount > 0) {
      const frameSize = width * height;
      const availableFrames = Math.min(bufferCount, Math.floor(parsed.length / frameSize));
      const hasSyncedRanges = (autoVmins?.length ?? 0) >= nSlices && (autoVmaxs?.length ?? 0) >= nSlices;
      if (!hasSyncedRanges && availableFrames > 0) {
        const token = ++autoRangeComputeTokenRef.current;
        let j = 0;
        const computeNextRange = () => {
          if (token !== autoRangeComputeTokenRef.current) return;
          const idx = (bufferStart + j) % nSlices;
          const start = j * frameSize;
          const end = start + frameSize;
          if (!cachedAutoRange(localAutoVminsRef.current, localAutoVmaxsRef.current, idx)) {
            ensureLocalAutoRange(idx, parsed.subarray(start, end), percentileLow, percentileHigh);
          }
          j++;
          if (j < availableFrames) {
            const ric = (window as unknown as { requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number }).requestIdleCallback;
            if (ric) ric(computeNextRange, { timeout: 150 });
            else window.setTimeout(computeNextRange, 0);
          }
        };
        window.setTimeout(computeNextRange, 0);
      }
    }
    return () => { autoRangeComputeTokenRef.current++; };
  }, [bufferBytes, bufferStart, bufferCount, autoContrast, logScale, width, height, nSlices, autoVmins, autoVmaxs, percentileLow, percentileHigh, ensureLocalAutoRange]);

  // Sync displaySliceIdx with model when not playing
  React.useEffect(() => {
    if (!playing) {
      setGpuDisplayVisible(false);
      setDisplaySliceIdx(sliceIdx);
      setPlaybackUiSliceIdx(sliceIdx);
    }
  }, [sliceIdx, playing, setGpuDisplayVisible]);

  // Histogram state for main image
  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  // GPU-computed 256-bin histogram. When non-null, the Histogram component
  // uses these bins directly and skips its CPU bin-scan fallback.
  const [imageHistogramBins, setImageHistogramBins] = React.useState<number[] | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [panelHistogramData, setPanelHistogramData] = React.useState<(Float32Array | null)[]>([]);
  const [panelDataRanges, setPanelDataRanges] = React.useState<{ min: number; max: number }[]>([]);
  const perPanelHistogramEnabled = (nPanels || 1) > 1 && !linkContrast;

  const updatePanelState = (panel: number, patch: Partial<PanelState>) => {
    setPanelStates(prev => prev.map((state, i) => i === panel ? { ...state, ...patch } : state));
  };
  const setPanelRangeValues = (panel: number, minValue: number | null, maxValue: number | null) => {
    const n = Math.max(1, nPanels || 1);
    const nextMins = Array.from({ length: n }, (_, i) => vminPerPanelLiveRef.current[i] ?? null);
    const nextMaxs = Array.from({ length: n }, (_, i) => vmaxPerPanelLiveRef.current[i] ?? null);
    nextMins[panel] = minValue;
    nextMaxs[panel] = maxValue;
    vminPerPanelLiveRef.current = nextMins;
    vmaxPerPanelLiveRef.current = nextMaxs;
    setVminPerPanel(nextMins);
    setVmaxPerPanel(nextMaxs);
  };
  const extractPanelSlice = (
    raw: Float32Array,
    panel: number,
    panelLogScale: boolean,
  ): Float32Array | null => {
    const n = Math.max(1, nPanels || 1);
    if (height <= 0 || raw.length === 0) return null;
    const panelW = Math.max(1, sourcePanelWidth);
    const fullW = raw.length === height * panelW ? panelW : width;
    const srcPanel = sharedPanelSource ? 0 : panel;
    const x0 = Math.min(Math.max(0, srcPanel * panelW), Math.max(0, fullW - panelW));
    if (raw.length < height * fullW || x0 + panelW > fullW || panel >= n) return null;
    const out = new Float32Array(height * panelW);
    for (let r = 0; r < height; r++) {
      out.set(raw.subarray(r * fullW + x0, r * fullW + x0 + panelW), r * panelW);
    }
    return panelLogScale ? applyLogScale(out) : out;
  };

  const resolvePanelRange = (
    panel: number,
    range: { min: number; max: number },
    sharedAutoRange?: { vmin: number; vmax: number } | null,
  ): { vmin: number; vmax: number; logScale: boolean } => {
    const state = panelStates[panel] || initialState;
    if (sharedAutoRange && !perPanelHistogramEnabled) {
      return { ...sharedAutoRange, logScale };
    }
    // Per-panel mode: always interpret slider pct in THIS panel's data
    // range. Stack-wide bounds (for mixed BF/DF counts vs SSB radians)
    // would decode SSB sliders to count-territory values → black image.
    const pdr = panelDataRanges[panel];
    const effectiveRange = (perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
      ? pdr
      : range;
    const useStoredManual = !autoContrast;
    if (useStoredManual) {
      const storedMin = vminPerPanelLiveRef.current[panel];
      const storedMax = vmaxPerPanelLiveRef.current[panel];
      if (storedMin != null || storedMax != null) {
        const lo = storedMin ?? effectiveRange.min;
        const hi = storedMax ?? effectiveRange.max;
        return { vmin: lo, vmax: Math.max(lo, hi), logScale };
      }
    }
    const slider = sliderRange(effectiveRange.min, effectiveRange.max, state.imageVminPct, state.imageVmaxPct);
    return { ...slider, logScale };
  };

  const autoPanelRangeFromData = (
    panelData: Float32Array | null,
    fallbackRange: { min: number; max: number },
    low: number,
    high: number,
  ): { vmin: number; vmax: number; logScale: boolean } | null => {
    if (!panelData || panelData.length === 0) return null;
    const dataRange = findDataRange(panelData);
    const range = dataRange.max > dataRange.min ? dataRange : fallbackRange;
    if (range.max <= range.min) return null;
    let clipped: { vmin: number; vmax: number } = percentileClip(panelData, low, high);
    const span = range.max - range.min;
    if (!Number.isFinite(clipped.vmin) || !Number.isFinite(clipped.vmax) || clipped.vmax <= clipped.vmin || clipped.vmax - clipped.vmin < span * 1e-4) {
      clipped = { vmin: range.min, vmax: range.max };
    }
    return { vmin: clipped.vmin, vmax: Math.max(clipped.vmin, clipped.vmax), logScale };
  };

  const panelAutoClipPcts = (
    panel: number,
    state: PanelState,
    stackBounds: { min: number; max: number },
  ): (Pick<PanelState, "imageVminPct" | "imageVmaxPct"> & { vmin: number; vmax: number }) | null => {
    const panelRaw = panelHistogramData[panel];
    if (!panelRaw || panelRaw.length === 0) return null;
    // panelHistogramData is already in the active display domain; in log mode
    // refreshHistogram populated it from extractPanelSlice(..., logScale).
    const panelRange = panelDataRanges[panel];
    const range = (panelRange && panelRange.max > panelRange.min) ? panelRange : stackBounds;
    const span = range.max - range.min;
    if (span <= 0) return null;
    let clipped: { vmin: number; vmax: number } = percentileClip(panelRaw, percentileLow, percentileHigh);
    if (
      !Number.isFinite(clipped.vmin) ||
      !Number.isFinite(clipped.vmax) ||
      clipped.vmax <= clipped.vmin ||
      clipped.vmax - clipped.vmin < span * 1e-4
    ) {
      clipped = { vmin: range.min, vmax: range.max };
    }
    return {
      vmin: clipped.vmin,
      vmax: Math.max(clipped.vmin, clipped.vmax),
      imageVminPct: valueToPct(clipped.vmin, range.min, range.max, state.imageVminPct),
      imageVmaxPct: valueToPct(clipped.vmax, range.min, range.max, state.imageVmaxPct),
    };
  };

  const restorePanelManualClipPcts = () => {
    const stackBounds = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    if (stackBounds.max <= stackBounds.min) return;
    const n = Math.max(1, nPanels || 1);
    const liveStates = panelStatesLiveRef.current.length === n ? panelStatesLiveRef.current : panelStates;
    const nextStates = Array.from({ length: n }, (_, i) => {
      const state = liveStates[i] || initialState;
      const storedMin = vminPerPanelLiveRef.current[i];
      const storedMax = vmaxPerPanelLiveRef.current[i];
      if (storedMin == null && storedMax == null) {
        return { ...state, imageVminPct: 0, imageVmaxPct: 100 };
      }
      const panelRange = panelDataRanges[i];
      const range = (panelRange && panelRange.max > panelRange.min) ? panelRange : stackBounds;
      if (range.max <= range.min) return { ...state, imageVminPct: 0, imageVmaxPct: 100 };
      const lo = storedMin ?? range.min;
      const hi = Math.max(lo, storedMax ?? range.max);
      return {
        ...state,
        imageVminPct: valueToPct(lo, range.min, range.max, state.imageVminPct),
        imageVmaxPct: valueToPct(hi, range.min, range.max, state.imageVmaxPct),
      };
    });
    panelStatesLiveRef.current = nextStates;
    setPanelStates(nextStates);
  };

  const resolvePanelRenderRange = (
    panel: number,
    range: { min: number; max: number },
    sharedAutoRange: { vmin: number; vmax: number } | null,
    panelData: Float32Array | null,
    autoOn: boolean,
    low: number,
    high: number,
  ): { vmin: number; vmax: number; logScale: boolean } => {
    if (perPanelHistogramEnabled && autoOn) {
      const autoRange = autoPanelRangeFromData(panelData, range, low, high);
      if (autoRange) return autoRange;
    }
    return resolvePanelRange(panel, range, sharedAutoRange);
  };

  const handleAutoContrastChange = (on: boolean) => {
    if (on) {
      manualImageRangeBeforeAutoRef.current = { min: imageVminPct, max: imageVmaxPct };
    }
    setAutoContrast(on);
    if (perPanelHistogramEnabled) {
      if (on) {
        // Keep remembered manual per-panel clips. Auto rendering ignores them,
        // and toggling Auto back off should restore the user's manual window.
        // Per-panel snap fires automatically via the [autoContrast,
        // panelHistogramData, ...] useEffect below. Calling the legacy
        // stack-wide snap here would race-write 0/100 to every panel
        // before the effect overrode with the correct per-panel clip,
        // causing a 1-frame flash to washed contrast on every toggle.
      } else {
        // OFF restores manual contrast. If the user never set a manual range,
        // default to each panel's full local histogram range.
        restorePanelManualClipPcts();
        manualImageRangeBeforeAutoRef.current = null;
      }
      return;
    }
    if (on && imageHistogramData) {
      // ON -> snap slider thumbs to actual percentile clip so slider shows what's rendered.
      const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
      const { vmin: pmin, vmax: pmax } = cached ?? percentileClip(imageHistogramData, percentileLow, percentileHigh);
      const { min: autoMin, max: autoMax } = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
      const span = autoMax - autoMin;
      if (span > 0) {
        setImageVminPct(Math.max(0, Math.min(100, ((pmin - autoMin) / span) * 100)));
        setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - autoMin) / span) * 100)));
      }
    } else {
      // OFF -> restore the user's manual window from before Auto was enabled.
      const restore = manualImageRangeBeforeAutoRef.current;
      if (restore) {
        setImageVminPct(restore.min);
        setImageVmaxPct(restore.max);
        manualImageRangeBeforeAutoRef.current = null;
      } else {
        setImageVminPct(0);
        setImageVmaxPct(100);
      }
    }
  };

  // Histogram state for FFT
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftStats, setFftStats] = React.useState<{ mean: number; min: number; max: number; std: number }>({ mean: 0, min: 0, max: 0, std: 0 });
  const [fftQuality, setFftQuality] = React.useState<FftQualityMetrics | null>(null);
  const fftQualityKeyRef = React.useRef("");
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftLogScale, setFftLogScale] = React.useState(false);
  const [fftAuto, setFftAuto] = React.useState(true);  // Auto: mask DC + 99.9% clipping
  const [fftShowColorbar, setFftShowColorbar] = React.useState(false);
  const [fftOffscreenVersion, setFftOffscreenVersion] = React.useState(0);
  const [showColorbar, setShowColorbar] = React.useState(false);

  // Histogram state for kymograph (mirrors FFT contrast/colormap controls)
  const [kymoVminPct, setKymoVminPct] = React.useState(0);
  const [kymoVmaxPct, setKymoVmaxPct] = React.useState(100);
  const [kymoHistogramData, setKymoHistogramData] = React.useState<Float32Array | null>(null);
  const [kymoDataRange, setKymoDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [kymoStats, setKymoStats] = React.useState<{ mean: number; min: number; max: number; std: number }>({ mean: 0, min: 0, max: 0, std: 0 });
  const [kymoColormap, setKymoColormap] = React.useState("inferno");
  const [kymoLogScale, setKymoLogScale] = React.useState(false);
  const [kymoAuto, setKymoAuto] = React.useState(true);  // Auto: percentile-clip like the main image
  const [kymoShowColorbar, setKymoShowColorbar] = React.useState(false);

  const handleRootMouseDownCapture = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement | null;
    if (target?.closest("canvas")) rootRef.current?.focus();
  };

  const lastBenchmarkTokenRef = React.useRef<unknown>(null);
  const benchmarkPlaybackFpsRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    const req = benchmarkRequest ?? {};
    const token = req.token;
    const mode = typeof req.mode === "string" ? req.mode : "playback";
    if ((typeof token !== "string" && typeof token !== "number") || mode === "renderBurst" || lastBenchmarkTokenRef.current === token) return;
    lastBenchmarkTokenRef.current = token;

    let cancelled = false;
    const sleep = (ms: number) => new Promise<void>(resolve => window.setTimeout(resolve, ms));
    const numberFromReq = (key: string, fallback: number) => {
      const value = req[key];
      return typeof value === "number" && Number.isFinite(value) ? value : fallback;
    };
    const warmupMs = Math.max(0, numberFromReq("warmupMs", 3000));
    const sampleMs = Math.max(250, numberFromReq("sampleMs", 10000));
    const targetFps = clampPlaybackFps(numberFromReq("targetFps", playbackFps));
    const expectedFrames = Math.max(0, Math.floor(numberFromReq("expectedFrames", nSlices)));
    const waitForGpuPreload = req.waitForGpuPreload === true;
    const reportUrl = typeof req.reportUrl === "string" ? req.reportUrl : "";
    const label = typeof req.label === "string" ? req.label : "show3d benchmark";

    void (async () => {
      const startedAt = performance.now();
      const setStatus = (status: string, extra: Record<string, unknown> = {}) => {
        if (!cancelled) {
          setBenchmarkResult({ token, label, status, targetFps, ...extra });
        }
      };

      try {
        setStatus("warming");
        const estimatedRefresh = await estimateRafFps(Math.max(300, numberFromReq("refreshProbeMs", 750)));
        if (estimatedRefresh !== null) {
          setStatus("warming", { displayRefreshFps: Number(estimatedRefresh.toFixed(2)) });
        }
        benchmarkPlaybackFpsRef.current = targetFps;
        setPlaybackFps(targetFps);
        setPlaying(true);

        if (waitForGpuPreload && expectedFrames > 0) {
          const preloadDeadline = performance.now() + Math.max(5000, numberFromReq("preloadTimeoutMs", 120000));
          let preloadReady = false;
          while (!cancelled && performance.now() < preloadDeadline) {
            const dbg = show3dPerfDebug() ?? {};
            const gpuDone = Number(dbg.gpuPreloadDone ?? dbg.gpuFrameCacheUploaded ?? 0);
            const fetchDone = Number(dbg.frameFetchPreloadDone ?? 0);
            preloadReady = separatePanelFrames
              ? gpuDone >= expectedFrames
              : gpuDone >= expectedFrames || fetchDone >= expectedFrames;
            if (preloadReady) break;
            setStatus("preloading", { gpuPreloadDone: gpuDone, frameFetchPreloadDone: fetchDone });
            await sleep(250);
          }
          if (!preloadReady) {
            const dbg = show3dPerfDebug() ?? {};
            const gpuDone = Number(dbg.gpuPreloadDone ?? dbg.gpuFrameCacheUploaded ?? 0);
            const fetchDone = Number(dbg.frameFetchPreloadDone ?? 0);
            setStatus("error", {
              error: "GPU preload incomplete",
              gpuPreloadDone: gpuDone,
              frameFetchPreloadDone: fetchDone,
              gpuPreloadTarget: expectedFrames,
              gpuPreloadError: dbg.gpuPreloadError ?? null,
              gpuPreloadLastMiss: dbg.gpuPreloadLastMiss ?? null,
            });
            return;
          }
        }

        await sleep(warmupMs);
        if (cancelled) return;

        const dbgStart = show3dPerfDebug() ?? {};
        resetFramePacingDebug(dbgStart, playbackIntervalMs(targetFps));
        const startFrames = Number(dbgStart.renderedFrames ?? 0);
        const sampleStart = performance.now();
        setStatus("sampling");
        await sleep(sampleMs);
        if (cancelled) return;

        const dbgEnd = show3dPerfDebug() ?? {};
        const elapsedSeconds = Math.max(0.001, (performance.now() - sampleStart) / 1000);
        const endFrames = Number(dbgEnd.renderedFrames ?? 0);
        const frames = Math.max(0, endFrames - startFrames);
        const measuredFps = frames / elapsedSeconds;
        const frameIntervalCount = Number(dbgEnd.frameIntervalCount ?? 0);
        const overBudgetFrames = Number(dbgEnd.overBudgetFrames ?? 0);
        const passTarget = measuredFps >= targetFps * 0.98;
        const displayRefreshFps = estimatedRefresh !== null ? Number(estimatedRefresh.toFixed(2)) : null;
        const refreshLimited = displayRefreshFps !== null && targetFps > displayRefreshFps * 1.03;
        const result = {
          token,
          label,
          status: "done",
          targetFps,
          displayRefreshFps,
          refreshLimited,
          measuredFps: Number(measuredFps.toFixed(2)),
          frames,
          elapsedSeconds: Number(elapsedSeconds.toFixed(2)),
          passTarget,
          pass60: measuredFps >= 60 * 0.98,
          frameIntervalAvgMs: dbgEnd.frameIntervalAvgMs ?? null,
          frameIntervalP95Ms: percentileFromHistory(dbgEnd.frameIntervalHistory, 95),
          maxFrameIntervalMs: dbgEnd.maxFrameIntervalMs ?? null,
          overBudgetFrames,
          overBudgetPct: frameIntervalCount > 0 ? Number(((overBudgetFrames / frameIntervalCount) * 100).toFixed(2)) : null,
          lastRenderPath: dbgEnd.lastRenderPath ?? null,
          lastRenderMs: dbgEnd.lastRenderMs ?? null,
          lastFrameSource: dbgEnd.lastFrameSource ?? null,
          frameFetchCacheSize: dbgEnd.frameFetchCacheSize ?? null,
          gpuFrameCacheUploaded: dbgEnd.gpuFrameCacheUploaded ?? null,
          gpuPreloadDone: dbgEnd.gpuPreloadDone ?? null,
          frameFetchPreloadDone: dbgEnd.frameFetchPreloadDone ?? null,
          missingFrame: dbgEnd.missingFrame ?? null,
          totalMs: Number((performance.now() - startedAt).toFixed(1)),
        };
        setBenchmarkResult(result);
        if (reportUrl) {
          void fetch(reportUrl, { method: "POST", mode: "no-cors", body: JSON.stringify(result) }).catch(() => {});
        }
      } catch (err) {
        setStatus("error", { error: err instanceof Error ? err.message : String(err) });
      } finally {
        if (!cancelled) setPlaying(false);
        benchmarkPlaybackFpsRef.current = null;
      }
    })();

    return () => {
      cancelled = true;
      benchmarkPlaybackFpsRef.current = null;
    };
  }, [benchmarkRequest, playbackFps, nSlices, separatePanelFrames, setBenchmarkResult, setPlaybackFps, setPlaying]);

  // FFT d-spacing measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);

  // ROI FFT state: when ROI + FFT are both active, compute FFT of cropped ROI region
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);
  const fftCropDimsRef = React.useRef<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);
  const fftPanelGridRef = React.useRef<{ panelWidth: number; panelHeight: number; cols: number; rows: number; count: number } | null>(null);

  // FFT zoom/pan state
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const internalFftZoomSyncRef = React.useRef(false);
  const fftViewLiveRef = React.useRef({ zoom: 1, panX: 0, panY: 0 });
  const fftViewRafRef = React.useRef<number | null>(null);
  const fftViewReactSyncTimerRef = React.useRef<number | null>(null);
  const fftViewTraitSyncTimerRef = React.useRef<number | null>(null);
  const fftViewDirectRedrawRef = React.useRef<((view: { zoom: number; panX: number; panY: number }) => void) | null>(null);
  const fftViewCenterOnViewportRef = React.useRef(false);
  const fftOverlayInitialCenterPendingRef = React.useRef(true);
  const fftOverlayWasActiveRef = React.useRef(false);
  const fftUserAdjustedViewRef = React.useRef(false);

  const commitFftViewReactState = React.useCallback(() => {
    const live = fftViewLiveRef.current;
    setFftZoom(prev => Math.abs(prev - live.zoom) > 0.001 ? live.zoom : prev);
    setFftPanX(prev => Math.abs(prev - live.panX) > 0.5 ? live.panX : prev);
    setFftPanY(prev => Math.abs(prev - live.panY) > 0.5 ? live.panY : prev);
  }, []);

  const scheduleFftViewState = React.useCallback((next: { zoom: number; panX: number; panY: number }, syncTrait = false, directOnly = false) => {
    fftViewLiveRef.current = next;
    if (directOnly) {
      fftViewDirectRedrawRef.current?.(next);
      if (fftViewReactSyncTimerRef.current !== null) {
        window.clearTimeout(fftViewReactSyncTimerRef.current);
      }
      fftViewReactSyncTimerRef.current = window.setTimeout(() => {
        fftViewReactSyncTimerRef.current = null;
        commitFftViewReactState();
      }, 80);
    } else if (fftViewRafRef.current === null) {
      fftViewRafRef.current = window.requestAnimationFrame(() => {
        fftViewRafRef.current = null;
        commitFftViewReactState();
      });
    }
    if (syncTrait) {
      if (fftViewTraitSyncTimerRef.current !== null) {
        window.clearTimeout(fftViewTraitSyncTimerRef.current);
      }
      fftViewTraitSyncTimerRef.current = window.setTimeout(() => {
        fftViewTraitSyncTimerRef.current = null;
        internalFftZoomSyncRef.current = true;
        setFftOverlayZoomTrait(Number(fftViewLiveRef.current.zoom.toFixed(3)));
      }, 160);
    }
  }, [commitFftViewReactState, setFftOverlayZoomTrait]);

  React.useEffect(() => {
    fftViewLiveRef.current = { zoom: fftZoom, panX: fftPanX, panY: fftPanY };
  }, [fftZoom, fftPanX, fftPanY]);

  React.useEffect(() => () => {
    if (fftViewRafRef.current !== null) {
      window.cancelAnimationFrame(fftViewRafRef.current);
      fftViewRafRef.current = null;
    }
    if (fftViewReactSyncTimerRef.current !== null) {
      window.clearTimeout(fftViewReactSyncTimerRef.current);
      fftViewReactSyncTimerRef.current = null;
    }
    if (fftViewTraitSyncTimerRef.current !== null) {
      window.clearTimeout(fftViewTraitSyncTimerRef.current);
      fftViewTraitSyncTimerRef.current = null;
    }
  }, []);

  React.useEffect(() => {
    if (internalFftZoomSyncRef.current) {
      internalFftZoomSyncRef.current = false;
      return;
    }
    const reset = { zoom: resolvedFftOverlayZoom, panX: 0, panY: 0 };
    fftViewLiveRef.current = reset;
    fftViewCenterOnViewportRef.current = true;
    fftOverlayInitialCenterPendingRef.current = true;
    fftUserAdjustedViewRef.current = false;
    setFftZoom(reset.zoom);
    setFftPanX(reset.panX);
    setFftPanY(reset.panY);
  }, [resolvedFftOverlayZoom]);

  React.useEffect(() => {
    if (fftLayoutOverlay && !fftOverlayWasActiveRef.current) {
      fftOverlayInitialCenterPendingRef.current = true;
      fftViewCenterOnViewportRef.current = true;
    }
    fftOverlayWasActiveRef.current = fftLayoutOverlay;
  }, [fftLayoutOverlay]);

  React.useEffect(() => {
    if (fftLayoutOverlay && fftOverlayInitialCenterPendingRef.current) {
      fftViewCenterOnViewportRef.current = true;
    }
  }, [fftLayoutOverlay, fftOffscreenVersion]);
  const fftContainerRef = React.useRef<HTMLDivElement>(null);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileLine, setProfileLine] = useModelState<{row: number; col: number}[]>("profile_line");
  const [profileWidth, setProfileWidth] = useModelState<number>("profile_width");
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const [profilePanelIdx, setProfilePanelIdx] = React.useState(0);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profilePoints = profileLine || [];
  // Kymograph (space-time) panel: static (nFrames, lineLen) image built by
  // sampling the profile line on every frame from the offline stack. Recompute
  // is cold-path (on line / width change only), not per render tick.
  const [showKymograph, setShowKymograph] = useModelState<boolean>("show_kymograph");
  const kymoCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const kymoOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const kymoDataRef = React.useRef<{ data: Float32Array; lineLen: number; nFrames: number } | null>(null);
  const [kymoVersion, setKymoVersion] = React.useState(0);
  // Kymograph zoom/pan state (mirrors FFT)
  const [kymoZoom, setKymoZoom] = React.useState(1);
  const [kymoPanX, setKymoPanX] = React.useState(0);
  const [kymoPanY, setKymoPanY] = React.useState(0);
  const kymoContainerRef = React.useRef<HTMLDivElement>(null);
  // Click readout: cursor maps to (frame index, distance index) and looks up
  // intensity in the static kymograph image. Mirrors FFT d-spacing readout.
  const [kymoClickInfo, setKymoClickInfo] = React.useState<{
    timeVal: number; timeUnit: string; distVal: number; distUnit: string; intensity: number;
    col: number; row: number;
  } | null>(null);
  const kymoClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const [profileResizeStart, setProfileResizeStart] = React.useState<{ y: number; height: number } | null>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);

  // Sync sizes from Python and set initial minimum. In multi-panel mode the user
  // is comparing N images side-by-side; default per-panel sizing keeps each image
  // readable instead of crushed when the widget concatenates them into one wide
  // canvas (e.g. 4 panels at 500 px total → 125 px per panel = too small).
  React.useEffect(() => {
    // size is PER PANEL. For multi-panel, total canvas width = size * cols.
    // NEVER BIN rule: data is never averaged. CSS canvas scales the painted
    // image for display, source pixels stay intact. 500 px/panel default
    // gives 4 cols → 2000 px wide which fits a typical monitor; operator
    // drags the resize handle larger when they want pixel-1:1.
    const n = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(n);
    const perPanel = canvasSizeTrait > 0 ? canvasSizeTrait : (n > 1 ? 500 : CANVAS_TARGET_SIZE);
    const target = perPanel * cols;
    setMainCanvasSize(target);
    if (initialCanvasSizeRef.current === CANVAS_TARGET_SIZE) {
      initialCanvasSizeRef.current = target;
    }
  }, [canvasSizeTrait, visiblePanelCount, panelColsForCount]);

  // Calculate display scale. In multi-panel mode `width` may be either the
  // concatenated source width or one shared source frame drawn into N slots.
  // `panel_width_px` keeps the per-panel source geometry explicit.
  const _nPanelsLocal = Math.max(1, visiblePanelCount || 1);
  const _colsLocal = panelColsForCount(_nPanelsLocal);
  const _rowsLocal = Math.ceil(_nPanelsLocal / _colsLocal);
  const fftAllowed = true;
  const effectiveShowFft = showFft && fftAllowed;
  const sourcePanelWidth = totalPanelCount > 1
    ? Math.max(1, panelWidthPx || Math.round(width / totalPanelCount))
    : Math.max(1, width);
  const sourcePanelHeight = Math.max(1, height);
  const isMultiPanelSource = totalPanelCount > 1;
  const displayScale = isMultiPanelSource
    ? mainCanvasSize / Math.max(1, sourcePanelWidth * _colsLocal)
    : mainCanvasSize / Math.max(width, height);
  // For 90°/270° rotations, swap canvas dims so non-square images fit without clipping.
  const rotSwap = (imageRotation % 2) !== 0;
  const canvasW = isMultiPanelSource
    ? Math.round(sourcePanelWidth * displayScale * _colsLocal)
    : Math.round((rotSwap ? height : width) * displayScale);
  // Grid layout: when max_cols wraps panels into multiple rows, canvasH grows to fit `rows` rows.
  const _canvasHSingleRow = Math.round((rotSwap ? width : height) * displayScale);
  const _gapForLayout = _nPanelsLocal > 1 ? (panelGapTrait ?? 10) : 0;
  const _slotWForLayout = (canvasW - _gapForLayout * (_colsLocal - 1)) / _colsLocal;
  const _slotHForLayout = _slotWForLayout * (sourcePanelHeight / sourcePanelWidth);
  const canvasH = isMultiPanelSource
    ? Math.round(_slotHForLayout * _rowsLocal + _gapForLayout * (_rowsLocal - 1))
    : _canvasHSingleRow;
  const mainPanelWidth = `min(100%, ${canvasW}px)`;
  const mainPanelAspectRatio = `${Math.max(canvasW, 1)} / ${Math.max(canvasH, 1)}`;
  const effectiveLoopEnd = loopEnd < 0 ? nSlices - 1 : loopEnd;
  // ROI hidden while the kymograph is shown - both are line/region analysis on
  // the same side slot, and showing them together confuses which panel is which.
  const roiAllowed = totalPanelCount === 1 && !showKymograph;
  const effectiveRoiActive = roiAllowed && roiActive;

  type PanelGeometry = {
    panelIdx: number;
    slotX: number;
    slotY: number;
    slotW: number;
    slotH: number;
    scaleX: number;
    scaleY: number;
    state: PanelState;
  };
  const getPanelLayout = () => {
    const n = _nPanelsLocal;
    const cols = _colsLocal;
    const rows = _rowsLocal;
    const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
    const slotW = (canvasW - gap * (cols - 1)) / cols;
    const slotH = (canvasH - gap * (rows - 1)) / rows;
    return { n, cols, rows, gap, slotW, slotH };
  };
  const getPanelGeometry = (panelIdx: number): PanelGeometry | null => {
    const { n, cols, rows, gap, slotW, slotH } = getPanelLayout();
    if (panelIdx < 0 || panelIdx >= totalPanelCount) return null;
    const slotIdx = visiblePanelIndices.indexOf(panelIdx);
    if (slotIdx < 0 || slotIdx >= n) return null;
    const col = slotIdx % cols;
    const row = Math.floor(slotIdx / cols);
    if (row >= rows) return null;
    return {
      panelIdx,
      slotX: col * (slotW + gap),
      slotY: row * (slotH + gap),
      slotW,
      slotH,
      scaleX: slotW / Math.max(1, sourcePanelWidth),
      scaleY: slotH / Math.max(1, sourcePanelHeight),
      state: stateFor(panelIdx),
    };
  };
  const getFftSlot = React.useCallback((slot: number, count: number, cols: number, rows: number) => {
    const gap = count > 1 ? (panelGapTrait ?? 10) : 0;
    const slotW = (canvasW - gap * (cols - 1)) / cols;
    const slotH = (canvasH - gap * (rows - 1)) / rows;
    const col = slot % cols;
    const row = Math.floor(slot / cols);
    return {
      x: col * (slotW + gap),
      y: row * (slotH + gap),
      w: slotW,
      h: slotH,
    };
  }, [canvasW, canvasH, panelGapTrait]);
  const drawFftOffscreen = React.useCallback((ctx: CanvasRenderingContext2D, offscreen: HTMLCanvasElement) => {
    ctx.clearRect(0, 0, canvasW, canvasH);
    const grid = fftPanelGridRef.current;
    if (grid) {
      ctx.fillStyle = themeColors.bg;
      ctx.fillRect(0, 0, canvasW, canvasH);
      for (let slot = 0; slot < grid.count; slot++) {
        const srcCol = slot % grid.cols;
        const srcRow = Math.floor(slot / grid.cols);
        const srcX = srcCol * grid.panelWidth;
        const srcY = srcRow * grid.panelHeight;
        const dst = getFftSlot(slot, grid.count, grid.cols, grid.rows);
        ctx.imageSmoothingEnabled = grid.panelWidth < dst.w || grid.panelHeight < dst.h;
        ctx.save();
        ctx.beginPath();
        ctx.rect(dst.x, dst.y, dst.w, dst.h);
        ctx.clip();
        ctx.translate(dst.x + fftPanX, dst.y + fftPanY);
        ctx.scale(fftZoom, fftZoom);
        ctx.drawImage(
          offscreen,
          srcX,
          srcY,
          grid.panelWidth,
          grid.panelHeight,
          0,
          0,
          dst.w,
          dst.h,
        );
        ctx.restore();
      }
    } else {
      ctx.save();
      ctx.translate(fftPanX, fftPanY);
      ctx.scale(fftZoom, fftZoom);
      ctx.imageSmoothingEnabled = offscreen.width < canvasW || offscreen.height < canvasH;
      ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);
      ctx.restore();
    }
  }, [canvasW, canvasH, fftPanX, fftPanY, fftZoom, getFftSlot, themeColors.bg]);
  const panelGlobalColOffset = (panelIdx: number) => (totalPanelCount > 1 && !sharedPanelSource) ? panelIdx * sourcePanelWidth : 0;
  const panelLocalCol = (globalCol: number, panelIdx: number) => globalCol - panelGlobalColOffset(panelIdx);
  const panelGlobalCol = (localCol: number, panelIdx: number) => localCol + panelGlobalColOffset(panelIdx);
  const getImageHitRadius = (panelIdx: number) => {
    const geom = getPanelGeometry(panelIdx);
    if (!geom) return RESIZE_HIT_AREA_PX / Math.max(1e-6, displayScale * zoom);
    const scale = Math.max(1e-6, Math.min(geom.scaleX, geom.scaleY) * geom.state.zoom);
    return RESIZE_HIT_AREA_PX / scale;
  };
  const canvasPointFromEvent = (e: React.MouseEvent): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  React.useEffect(() => {
    if (offline || !frameServerUrl || width <= 0 || height <= 0 || nSlices <= 0 || canvasW <= 0 || canvasH <= 0) return;
    const n = Math.max(1, nPanels || 1);
    if (separatePanelFrames && n > 1) {
      let cancelled = false;
      const cacheLimit = getSeparatePanelGpuCacheLimit();
      const preloadCount = Math.min(nSlices, cacheLimit);
      const startIdx = ((Math.round(playbackIdxRef.current) % nSlices) + nSlices) % nSlices;
      const dbg = show3dPerfDebug();
      if (dbg) {
        dbg.gpuPreloadTarget = preloadCount;
        dbg.gpuPreloadDone = gpuFrameCacheUploadedRef.current.size;
        dbg.gpuPreloadActive = true;
        dbg.gpuPreloadMode = "separate-panel-direct-grid";
        dbg.gpuPanelCacheLimit = cacheLimit;
        dbg.gpuPanelTotalFrames = nSlices;
      }
      void (async () => {
        const rgbaCapacity = Math.max(1, Math.round(canvasW * canvasH));
        for (let step = 0; step < preloadCount; step++) {
          if (cancelled) break;
          const i = (startIdx + step) % nSlices;
          try {
            const ready = await ensurePanelFrameGpu(i, rgbaCapacity);
            if (!ready) {
              const d = show3dPerfDebug();
              if (d) {
                d.gpuPreloadMisses = ((d.gpuPreloadMisses as number | undefined) ?? 0) + 1;
                d.gpuPreloadLastMiss = i;
              }
              // One transient miss (stale 409, dropped socket, GPU not yet
              // ready) must NOT abort the whole preload - skip this frame and
              // keep going, matching the non-panel branch. `break` here left
              // the cache permanently below nSlices.
              continue;
            }
          } catch (err) {
            const d = show3dPerfDebug();
            if (d) d.gpuPreloadError = err instanceof Error ? err.message : String(err);
            continue;
          }
          const d = show3dPerfDebug();
          if (d) {
            d.gpuPreloadDone = gpuFrameCacheUploadedRef.current.size;
            d.gpuFrameCacheUploaded = gpuFrameCacheUploadedRef.current.size;
          }
          await new Promise<void>(resolve => setTimeout(resolve, 0));
        }
        const d = show3dPerfDebug();
        if (d) d.gpuPreloadActive = false;
      })();
      return () => {
        cancelled = true;
        const d = show3dPerfDebug();
        if (d) d.gpuPreloadActive = false;
      };
    }
    const stackByteLength = Math.max(1, width * height * 4) * Math.max(1, nSlices);
    if (stackByteLength > FRAME_SERVER_FULL_STACK_CACHE_BYTES) return;
    let cancelled = false;
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.gpuPreloadTarget = nSlices;
      dbg.gpuPreloadDone = 0;
      dbg.gpuPreloadActive = true;
      dbg.gpuPreloadMode = "direct-grid";
    }
    void (async () => {
      while (!cancelled && (!gpuCmapReadyRef.current || !gpuCmapRef.current)) {
        await new Promise<void>(resolve => setTimeout(resolve, 50));
      }
      const engine = gpuCmapRef.current;
      if (!engine || cancelled) return;
      const rgbaCapacity = Math.max(1, Math.round(canvasW * canvasH));
      for (let i = 0; i < nSlices; i++) {
        if (cancelled) break;
        const frame = await fetchFrameFromServer(i);
        if (cancelled) break;
        const d = show3dPerfDebug();
        if (!frame) {
          if (d) d.gpuPreloadMisses = ((d.gpuPreloadMisses as number | undefined) ?? 0) + 1;
          continue;
        }
        try {
          engine.uploadData(i, frame, width, height, rgbaCapacity);
          gpuFrameCacheUploadedRef.current.add(i);
          frameFetchCacheRef.current.delete(i);
          if (d) {
            d.gpuPreloadDone = gpuFrameCacheUploadedRef.current.size;
            d.gpuFrameCacheUploaded = gpuFrameCacheUploadedRef.current.size;
            d.frameFetchCacheSize = frameFetchCacheRef.current.size;
          }
        } catch (err) {
          if (d) d.gpuPreloadError = err instanceof Error ? err.message : String(err);
          break;
        }
        await new Promise<void>(resolve => setTimeout(resolve, 0));
      }
      const d = show3dPerfDebug();
      if (d) d.gpuPreloadActive = false;
    })();
    return () => {
      cancelled = true;
      const d = show3dPerfDebug();
      if (d) d.gpuPreloadActive = false;
    };
  }, [offline, frameServerUrl, frameServerVersion, width, height, nSlices, nPanels, canvasW, canvasH, fetchFrameFromServer, separatePanelFrames, ensurePanelFrameGpu, getSeparatePanelGpuCacheLimit, sliceIdx]);

  // ROI FFT active: both ROI and FFT on, with a selected ROI
  const roiFftActive = effectiveShowFft && effectiveRoiActive && roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0);

  // Preview panel visible: auto-shows when ROI active with a selected ROI
  const previewVisible = effectiveRoiActive && roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0);
  const selectedRoiKey = (() => {
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return "";
    const r = roiList[roiSelectedIdx];
    return `${r.row},${r.col},${r.radius},${r.radius_inner},${r.width},${r.height},${r.shape}`;
  })();

  // Compute stats for ALL ROIs (memoized, recomputes on frame/ROI geometry change)
  const allRoiStats = React.useMemo(() => {
    const raw = rawFrameDataRef.current;
    if (!effectiveRoiActive || !roiItems.length || !raw || !width || !height) return [];
    return roiItems.map(roi => computeROIPixelStats(raw, width, height, roi));
    // frameBytes triggers recompute on frame change; displaySliceIdx triggers recompute during playback
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveRoiActive, roiItems, width, height, frameBytes, displaySliceIdx]);

  // Initialize reusable offscreen canvas + ImageData (resized when dimensions change)
  React.useEffect(() => {
    if (width <= 0 || height <= 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    mainOffscreenRef.current = canvas;
    mainImgDataRef.current = canvas.getContext("2d")!.createImageData(width, height);
    scaledPlaybackImgDataRef.current = null;
    scaledPlaybackMapRef.current = null;
    logBufferRef.current = new Float32Array(width * height);
  }, [width, height]);

  // Prevent page scroll on secondary canvas containers. Main image wheel is
  // handled by a non-passive listener below so zoom works in notebook outputs.
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const el2 = fftContainerRef.current;
    const el3 = previewContainerRef.current;
    el2?.addEventListener("wheel", preventDefault, { passive: false });
    el3?.addEventListener("wheel", preventDefault, { passive: false });
    return () => {
      el2?.removeEventListener("wheel", preventDefault);
      el3?.removeEventListener("wheel", preventDefault);
    };
  }, [effectiveShowFft, previewVisible]);


  // Sync boomerang direction ref with reverse state
  React.useEffect(() => {
    bounceDirRef.current = reverse ? -1 : 1;
  }, [reverse]);

  // All playback params as a single ref (avoids stale closures in rAF loop)
  const pathIdxRef = React.useRef(0);
  const playRef = React.useRef({
    fps: playbackFps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    autoVmins, autoVmaxs,
    linkContrast,
    linkedState, linkPanels,
    panelStates, vminPerPanel, vmaxPerPanel,
    visiblePanelIndices,
    zoom, panX, panY, playbackPath,
    profileActive, profilePoints, profileWidth,
    traitVmin, traitVmax, smooth, imageRotation, showStats,
    diffMode, avgWindow,
  });
  React.useEffect(() => {
    linkedStateLiveRef.current = linkedState;
  }, [linkedState]);
  React.useEffect(() => {
    panelStatesLiveRef.current = panelStates;
  }, [panelStates]);
  React.useEffect(() => {
    const liveLinkedState = linkedStateLiveRef.current;
    const livePanelStates = panelStatesLiveRef.current.length === Math.max(1, nPanels || 1)
      ? panelStatesLiveRef.current
      : panelStates;
    playRef.current = {
      fps: playbackFps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
      nSlices, width, height, displayScale, canvasW, canvasH,
      logScale, autoContrast, percentileLow, percentileHigh,
      dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
      autoVmins, autoVmaxs,
      linkContrast,
      linkedState: liveLinkedState, linkPanels,
      panelStates: livePanelStates, vminPerPanel, vmaxPerPanel,
      visiblePanelIndices,
      zoom, panX, panY, playbackPath,
      profileActive, profilePoints, profileWidth,
      traitVmin, traitVmax, smooth, imageRotation, showStats,
      diffMode, avgWindow,
    };
  }, [playbackFps, reverse, boomerang, loop, loopStart, effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    autoVmins, autoVmaxs, linkContrast, linkedState, linkPanels, panelStates, vminPerPanel, vmaxPerPanel, visiblePanelIndices,
    zoom, panX, panY, playbackPath,
    profileActive, profilePoints, profileWidth,
    traitVmin, traitVmax, smooth, imageRotation, showStats, diffMode, avgWindow]);

  const updatePlaybackLiveControls = React.useCallback((idx: number) => {
    const c = playRef.current;
    const total = Math.max(1, c.nSlices || nSlices || 1);
    const rangeStart = c.loop ? Math.max(0, Math.min(c.loopStart, total - 1)) : 0;
    const rangeEnd = c.loop ? Math.max(rangeStart, Math.min(c.loopEnd, total - 1)) : total - 1;
    const clamped = Math.max(rangeStart, Math.min(rangeEnd, Math.round(idx)));
    const pct = total > 1 ? (clamped / (total - 1)) * 100 : 0;
    const slider = playbackSliderRef.current;
    const activeThumb = slider?.querySelector(c.loop ? ".MuiSlider-thumb[data-index='1']" : ".MuiSlider-thumb") as HTMLElement | null;
    const track = slider?.querySelector(".MuiSlider-track") as HTMLElement | null;
    const input = activeThumb?.querySelector("input") as HTMLInputElement | null;
    const count = playbackLiveCountRef.current;
    if (activeThumb) {
      activeThumb.style.left = `${pct}%`;
      activeThumb.setAttribute("aria-valuenow", String(clamped));
    }
    if (input) input.value = String(clamped);
    if (track && !c.loop) {
      track.style.left = "0%";
      track.style.width = `${pct}%`;
    }
    if (count) count.textContent = hiddenSet.size ? `${clamped + 1}/${visibleCount} (${total})` : `${clamped + 1}/${total}`;
  }, [hiddenSet.size, nSlices, visibleCount]);

  const frameTransformActive = () => diffMode !== "off" || Math.max(1, Math.round(avgWindow || 1)) > 1;

  const rawFrameForIndex = (idx: number, currentIdx: number, currentFrame: Float32Array | null): Float32Array | null => {
    const n = Math.max(1, nSlices || 1);
    const normalized = ((Math.round(idx) % n) + n) % n;
    if (currentFrame && normalized === ((Math.round(currentIdx) % n) + n) % n) return currentFrame;
    if (offline) return getOfflineFrame(normalized);
    const frameSize = width * height;
    const fromBuffer = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, n, normalized, frameSize)
      || getFrameFromBuffer(nextBufferRef.current, nextBufferStartRef.current, nextBufferCountRef.current, n, normalized, frameSize);
    if (fromBuffer) return fromBuffer;
    const cached = getCachedServerFrame(normalized);
    if (cached) return cached;
    return null;
  };

  // Mean of `avg_window` consecutive frames (temporal denoise). At the stack
  // ends the window SLIDES INWARD to stay full-width (frame 0, win 5 -> [0..4])
  // rather than shrinking - constant denoise strength, but the average is not
  // centered on `idx` near the ends. Even windows are front-biased.
  const averagedFrameForIndex = (idx: number, currentIdx: number, currentFrame: Float32Array | null): Float32Array | null => {
    const frameSize = width * height;
    const win = Math.max(1, Math.min(15, Math.round(avgWindow || 1)));
    if (win <= 1) return rawFrameForIndex(idx, currentIdx, currentFrame);
    const n = Math.max(1, nSlices || 1);
    const center = Math.max(0, Math.min(n - 1, Math.round(idx)));
    const half = Math.floor(win / 2);
    let start = center - half;
    let end = start + win - 1;
    if (start < 0) {
      end = Math.min(n - 1, end - start);
      start = 0;
    }
    if (end >= n) {
      start = Math.max(0, start - (end - n + 1));
      end = n - 1;
    }
    if (offline && offlineFloatStack && offlineFloatStack.byteLength >= n * frameSize * 4) {
      const out = new Float32Array(frameSize);
      let count = 0;
      for (let j = start; j <= end; j++) {
        const frame = float32FrameFromDataView(offlineFloatStack, j, frameSize, false);
        if (!frame || frame.length < frameSize) continue;
        for (let k = 0; k < frameSize; k++) out[k] += frame[k];
        count++;
      }
      if (count > 0) {
        const inv = 1 / count;
        for (let k = 0; k < frameSize; k++) out[k] *= inv;
        return out;
      }
    }
    if (offline && offlineStack && offlineStack.byteLength >= n * frameSize) {
      const out = new Float32Array(frameSize);
      let count = 0;
      for (let j = start; j <= end; j++) {
        const frame = getOfflineFrame(j);
        if (!frame || frame.length < frameSize) continue;
        for (let k = 0; k < frameSize; k++) out[k] += frame[k];
        count++;
      }
      if (count > 0) {
        const inv = 1 / count;
        for (let k = 0; k < frameSize; k++) out[k] *= inv;
        return out;
      }
    }
    const out = new Float32Array(frameSize);
    let count = 0;
    for (let j = start; j <= end; j++) {
      const frame = rawFrameForIndex(j, currentIdx, currentFrame);
      if (!frame || frame.length < frameSize) continue;
      for (let k = 0; k < frameSize; k++) out[k] += frame[k];
      count++;
    }
    if (count === 0) return rawFrameForIndex(idx, currentIdx, currentFrame);
    if (count > 1) {
      const inv = 1 / count;
      for (let k = 0; k < frameSize; k++) out[k] *= inv;
    }
    return out;
  };

  const displayFrameForIndex = (idx: number, currentFrame: Float32Array | null): Float32Array | null => {
    const frame = averagedFrameForIndex(idx, idx, currentFrame);
    if (!frame || diffMode === "off") return frame;
    const refIdx = diffMode === "first" ? 0 : Math.max(0, Math.round(idx) - 1);
    const ref = averagedFrameForIndex(refIdx, idx, currentFrame);
    if (!ref) return frame;
    const frameSize = width * height;
    const out = new Float32Array(frameSize);
    for (let k = 0; k < frameSize; k++) out[k] = frame[k] - ref[k];
    return out;
  };

  const renderGpuPanelSlice = (idx: number, updateDisplayState = true): boolean => {
    const normalized = ((Math.round(idx) % Math.max(1, nSlices)) + Math.max(1, nSlices)) % Math.max(1, nSlices);
    if (!separatePanelFrames || !gpuFrameCacheUploadedRef.current.has(normalized)) return false;
    const engine = gpuCmapRef.current;
    if (!engine || !gpuCmapReadyRef.current) return false;
    const c = playRef.current;
    if (c.imageRotation % 4 !== 0) return false;
    const sourcePanelCount = Math.max(1, nPanels || 1);
    const n = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(n);
    const rows = Math.ceil(n / cols);
    const gap = n > 1 ? (panelGapTrait ?? 10) : 0;

    const lut = COLORMAPS[c.cmap] || COLORMAPS.inferno;
    engine.uploadLUT(c.cmap, lut);
    let renderRanges: { vmin: number; vmax: number } | { vmin: number; vmax: number }[];
    let renderLogScale: boolean | boolean[];
    if (n > 1 && !c.linkContrast) {
      let sharedAutoRange: { vmin: number; vmax: number } | null = null;
      if (c.autoContrast) {
        sharedAutoRange = cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, normalized, c.logScale)
          || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, normalized, c.logScale);
        if (!sharedAutoRange) {
          sharedAutoRange = resolveDisplayRange(
            c.dataMin,
            c.dataMax,
            c.traitVmin,
            c.traitVmax,
            c.logScale,
            c.imageVminPct,
            c.imageVmaxPct,
          );
        }
      }
      renderRanges = visiblePanelIndices.map((panel) => {
        const stack = resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale);
        const pdr = panelDataRanges[panel];
        const bounds = (perPanelHistogramEnabled && pdr && pdr.max > pdr.min) ? pdr : stack;
        return resolvePanelRange(panel, bounds, sharedAutoRange);
      });
      renderLogScale = c.logScale;
    } else {
      let vmin: number, vmax: number;
      if (c.autoContrast) {
        const cached = cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, normalized, c.logScale)
          || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, normalized, c.logScale);
        if (cached) {
          ({ vmin, vmax } = cached);
        } else {
          ({ vmin, vmax } = resolveDisplayRange(
            c.dataMin,
            c.dataMax,
            c.traitVmin,
            c.traitVmax,
            c.logScale,
            c.imageVminPct,
            c.imageVmaxPct,
          ));
        }
      } else {
        ({ vmin, vmax } = resolveDisplayRange(
          c.dataMin,
          c.dataMax,
          c.traitVmin,
          c.traitVmax,
          c.logScale,
          c.imageVminPct,
          c.imageVmaxPct,
        ));
      }
      renderRanges = { vmin, vmax };
      renderLogScale = c.logScale;
    }

    const renderStartMs = performance.now();
    const gpuCtx = ensureGpuDisplayContext(engine, c.canvasW, c.canvasH);
    if (!gpuCtx) return false;
    const panelSlots = visiblePanelIndices.map((panel) => normalized * sourcePanelCount + panel);
    const transforms = visiblePanelIndices.map((panel) => {
      const base = c.panelStates[panel] || initialState;
      return {
        zoom: c.linkPanels ? c.linkedState.zoom : base.zoom,
        panX: c.linkPanels ? c.linkedState.panX : base.panX,
        panY: c.linkPanels ? c.linkedState.panY : base.panY,
      };
    });
    const rendered = engine.renderPanelSlotsDirectToCanvas(
      panelSlots,
      renderRanges,
      renderLogScale,
      gpuCtx,
      {
        width: c.canvasW,
        height: c.canvasH,
        panelCount: n,
        cols,
        rows,
        gap,
        bgRgb: packedRgbFromHex(themeColors.bg),
        transforms,
      },
    );
    if (!rendered) return false;
    setGpuDisplayVisible(true);
    playbackIdxRef.current = normalized;
    if (updateDisplayState) setDisplaySliceIdx(normalized);
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.missingFrame = null;
      dbg.lastFrame = normalized;
      dbg.lastFrameSource = "gpu-panel-cache-slots";
      dbg.lastRenderPath = "webgpu-grid-separate-panels-panel-slots-direct-fragment";
      dbg.lastRenderMs = Number((performance.now() - renderStartMs).toFixed(2));
      dbg.lastPanelTransforms = transforms.map(t => ({
        zoom: Number(t.zoom.toFixed(3)),
        panX: Number(t.panX.toFixed(1)),
        panY: Number(t.panY.toFixed(1)),
      }));
    }
    return true;
  };

  const renderGpuCachedSliceDirect = (idx: number, updateDisplayState = true): boolean => {
    if (separatePanelFrames) return renderGpuPanelSlice(idx, updateDisplayState);
    const normalized = ((Math.round(idx) % Math.max(1, nSlices)) + Math.max(1, nSlices)) % Math.max(1, nSlices);
    if (!gpuFrameCacheUploadedRef.current.has(normalized)) return false;
    const engine = gpuCmapRef.current;
    if (!engine || !gpuCmapReadyRef.current) return false;
    const c = playRef.current;
    if (c.imageRotation % 4 !== 0 || c.zoom !== 1 || c.panX !== 0 || c.panY !== 0) return false;
    const gpuCtx = ensureGpuDisplayContext(engine, c.canvasW, c.canvasH);
    if (!gpuCtx) return false;

    const lut = COLORMAPS[c.cmap] || COLORMAPS.inferno;
    engine.uploadLUT(c.cmap, lut);
    let vmin: number, vmax: number;
    if (c.autoContrast) {
      const cached = cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, normalized, c.logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, normalized, c.logScale);
      if (cached) {
        ({ vmin, vmax } = cached);
      } else {
        ({ vmin, vmax } = resolveDisplayRange(
          c.dataMin,
          c.dataMax,
          c.traitVmin,
          c.traitVmax,
          c.logScale,
          c.imageVminPct,
          c.imageVmaxPct,
        ));
      }
    } else {
      ({ vmin, vmax } = resolveDisplayRange(
        c.dataMin,
        c.dataMax,
        c.traitVmin,
        c.traitVmax,
        c.logScale,
        c.imageVminPct,
        c.imageVmaxPct,
      ));
    }

    if (hiddenPanelSet.size > 0 && !separatePanelFrames) return false;
    const n = Math.max(1, nPanels || 1);
    const cols = panelColsForCount(n);
    const rows = Math.ceil(n / cols);
    const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
    const renderStartMs = performance.now();
    if (n > 1 && !c.linkContrast && !sharedPanelSource) {
      const panelW = Math.max(1, panelWidthPx || Math.round(c.width / n));
      const regions = Array.from({ length: n }, (_, panel) => ({
        x: panel * panelW, y: 0, width: panelW, height: c.height,
      }));
      const sharedAutoRange = c.autoContrast ? { vmin, vmax } : null;
      const transformActive = c.diffMode !== "off" || Math.max(1, Math.round(c.avgWindow || 1)) > 1;
      const rawForRanges = rawFrameForIndex(normalized, normalized, rawFrameDataRef.current);
      const frameForRanges = rawForRanges && transformActive
        ? (displayFrameForIndex(normalized, rawForRanges) ?? rawForRanges)
        : rawForRanges;
      const ranges = Array.from({ length: n }, (_, panel) => {
        const stack = resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale);
        const panelData = frameForRanges ? extractPanelSlice(frameForRanges, panel, c.logScale) : null;
        const pdr = panelDataRanges[panel];
        const bounds = (panelData && panelData.length > 0)
          ? findDataRange(panelData)
          : ((perPanelHistogramEnabled && pdr && pdr.max > pdr.min) ? pdr : stack);
        return resolvePanelRenderRange(panel, bounds, sharedAutoRange, panelData, c.autoContrast, c.percentileLow, c.percentileHigh);
      });
      const logs = c.logScale;
      const bitmaps = engine.renderPerPanelGpuExplicit(normalized, regions, ranges, logs);
      const offCtx = mainOffscreenRef.current?.getContext("2d");
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!bitmaps || !offCtx || !ctx || !mainOffscreenRef.current) return false;
      offCtx.clearRect(0, 0, c.width, c.height);
      for (let panel = 0; panel < n; panel++) {
        if (bitmaps[panel]) {
          offCtx.drawImage(bitmaps[panel], panel * panelW, 0);
          bitmaps[panel].close();
        }
      }
      drawMain(ctx, mainOffscreenRef.current);
      setGpuDisplayVisible(false);
      playbackIdxRef.current = normalized;
      if (updateDisplayState) setDisplaySliceIdx(normalized);
      const dbg = show3dPerfDebug();
      if (dbg) {
        dbg.missingFrame = null;
        dbg.lastFrame = normalized;
        dbg.lastFrameSource = "gpu-cache";
        dbg.lastRenderPath = "webgpu-grid-panels-explicit-ranges";
        dbg.lastRenderMs = Number((performance.now() - renderStartMs).toFixed(2));
      }
      return true;
    }
    const sourcePanelWidthForGrid = sharedPanelSource
      ? Math.max(1, panelWidthPx || c.width)
      : Math.max(1, panelWidthPx || Math.round(c.width / n));
    const gridOpts = {
      width: c.canvasW,
      height: c.canvasH,
      panelCount: n,
      cols,
      rows,
      gap,
      bgRgb: packedRgbFromHex(themeColors.bg),
      sourcePanelWidth: sourcePanelWidthForGrid,
      sharedSource: !!sharedPanelSource,
    };
    const rendered = engine.renderSharedGridDirectToCanvas(
      normalized,
      { vmin, vmax },
      c.logScale,
      gpuCtx,
      gridOpts,
    );
    if (!rendered) return false;
    setGpuDisplayVisible(true);
    playbackIdxRef.current = normalized;
    if (updateDisplayState) setDisplaySliceIdx(normalized);
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.missingFrame = null;
      dbg.lastFrame = normalized;
      dbg.lastFrameSource = "gpu-cache";
      dbg.lastRenderPath = n === 1
        ? "webgpu-grid-single-panel-direct-fragment"
        : (sharedPanelSource ? "webgpu-grid-shared-panels-direct-fragment" : "webgpu-grid-panels-direct-fragment");
      dbg.lastRenderMs = Number((performance.now() - renderStartMs).toFixed(2));
    }
    return true;
  };

  const lastRenderBurstBenchmarkTokenRef = React.useRef<unknown>(null);
  React.useEffect(() => {
    const req = benchmarkRequest ?? {};
    const token = req.token;
    const mode = typeof req.mode === "string" ? req.mode : "playback";
    if ((typeof token !== "string" && typeof token !== "number") || mode !== "renderBurst" || lastRenderBurstBenchmarkTokenRef.current === token) return;
    lastRenderBurstBenchmarkTokenRef.current = token;

    let cancelled = false;
    const sleep = (ms: number) => new Promise<void>(resolve => window.setTimeout(resolve, ms));
    const numberFromReq = (key: string, fallback: number) => {
      const value = req[key];
      return typeof value === "number" && Number.isFinite(value) ? value : fallback;
    };
    const sampleMs = Math.max(250, numberFromReq("sampleMs", 3000));
    const expectedFrames = Math.max(0, Math.floor(numberFromReq("expectedFrames", nSlices)));
    const syncEvery = Math.max(0, Math.floor(numberFromReq("syncEvery", 1)));
    const reportUrl = typeof req.reportUrl === "string" ? req.reportUrl : "";
    const label = typeof req.label === "string" ? req.label : "show3d render burst";

    void (async () => {
      const startedAt = performance.now();
      const setStatus = (status: string, extra: Record<string, unknown> = {}) => {
        if (!cancelled) setBenchmarkResult({ token, label, status, targetFps: 0, mode, ...extra });
      };
      try {
        setStatus("preloading");
        const engine = gpuCmapRef.current;
        if (!engine || !gpuCmapReadyRef.current) throw new Error("WebGPU colormap engine is not ready");
        const rgbaCapacity = Math.max(1, Math.round(canvasW * canvasH));
        const framesToPrepare = expectedFrames > 0 ? Math.min(expectedFrames, nSlices) : nSlices;
        for (let i = 0; i < framesToPrepare; i++) {
          if (cancelled) return;
          if (separatePanelFrames) {
            const ready = await ensurePanelFrameGpu(i, rgbaCapacity);
            if (!ready) throw new Error(`panel frame ${i} was not available for GPU upload`);
          } else if (!gpuFrameCacheUploadedRef.current.has(i)) {
            const frame = await fetchFrameFromServer(i);
            if (!frame) throw new Error(`frame ${i} was not available for GPU upload`);
            engine.uploadData(i, frame, width, height, rgbaCapacity);
            gpuFrameCacheUploadedRef.current.add(i);
          }
          if (i % 4 === 0) {
            setStatus("preloading", { preparedFrames: i + 1, expectedFrames: framesToPrepare });
            await sleep(0);
          }
        }
        await engine.waitForSubmittedWork();

        setStatus("sampling", { preparedFrames: framesToPrepare, syncEvery });
        const sampleStart = performance.now();
        let frames = 0;
        let misses = 0;
        while (!cancelled && performance.now() - sampleStart < sampleMs) {
          const idx = frames % Math.max(1, framesToPrepare);
          const ok = renderGpuCachedSliceDirect(idx, false);
          if (!ok) {
            misses++;
            await sleep(0);
            continue;
          }
          frames++;
          if (syncEvery > 0 && frames % syncEvery === 0) {
            await engine.waitForSubmittedWork();
          } else if (frames % 32 === 0) {
            await sleep(0);
          }
        }
        await engine.waitForSubmittedWork();
        const elapsedSeconds = Math.max(0.001, (performance.now() - sampleStart) / 1000);
        const measuredFps = frames / elapsedSeconds;
        const dbgEnd = show3dPerfDebug() ?? {};
        const result = {
          token,
          label,
          status: "done",
          mode,
          syncEvery,
          measuredFps: Number(measuredFps.toFixed(2)),
          frames,
          misses,
          elapsedSeconds: Number(elapsedSeconds.toFixed(2)),
          preparedFrames: framesToPrepare,
          lastRenderPath: dbgEnd.lastRenderPath ?? null,
          lastRenderMs: dbgEnd.lastRenderMs ?? null,
          totalMs: Number((performance.now() - startedAt).toFixed(1)),
        };
        setBenchmarkResult(result);
        if (reportUrl) {
          void fetch(reportUrl, { method: "POST", mode: "no-cors", body: JSON.stringify(result) }).catch(() => {});
        }
      } catch (err) {
        setStatus("error", { error: err instanceof Error ? err.message : String(err) });
      }
    })();

    return () => {
      cancelled = true;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [benchmarkRequest, nSlices, separatePanelFrames, canvasW, canvasH, width, height]);

  const playbackHistogramCounterRef = React.useRef(0);
  const refreshHistogramRef = React.useRef<((idxArg?: number) => void | Promise<void>) | null>(null);

  // Playback logic - rAF-driven, zero React re-renders in hot path
  React.useEffect(() => {
    if (!playing) {
      // Playback stopped - sync final position to Python
      if (playbackIdxRef.current !== sliceIdx && (bufferRef.current || separatePanelFrames || offline)) {
        setLiveSliceIdx(playbackIdxRef.current);
        setSliceIdx(playbackIdxRef.current);
      }
      if (!playRef.current.showStats) setLocalStats(null);
      prefetchPendingRef.current = false;
      return;
    }

    // === PLAYBACK START ===
    // Snap slice_idx into [loop_start, loop_end] before first tick, otherwise
    // playback walked outside the loop range on the first frame.
    {
      const c0 = playRef.current;
      const rs0 = c0.loop ? Math.max(0, Math.min(c0.loopStart, c0.nSlices - 1)) : 0;
      const re0 = c0.loop ? Math.max(rs0, Math.min(c0.loopEnd, c0.nSlices - 1)) : c0.nSlices - 1;
      const liveStart = Number.isFinite(playbackIdxRef.current)
        ? playbackIdxRef.current
        : (Number.isFinite(displaySliceIdx) ? displaySliceIdx : sliceIdx);
      playbackIdxRef.current = Math.max(rs0, Math.min(re0, Math.round(liveStart)));
    }
    const pathLen = playRef.current.playbackPath?.length ?? 0;
    pathIdxRef.current = pathLen > 0 ? (playRef.current.reverse ? pathLen : -1) : 0;
    bounceDirRef.current = playRef.current.reverse ? -1 : 1;
    if (frameServerUrl && gpuFrameCacheUploadedRef.current.size < playRef.current.nSlices) {
      const c0 = playRef.current;
      if (separatePanelFrames) {
        prefetchPanelGpuFrames(playbackIdxRef.current, c0.reverse, c0.loop, c0.loopStart, c0.loopEnd);
      } else {
        prefetchServerFrames(playbackIdxRef.current, c0.reverse, c0.loop, c0.loopStart, c0.loopEnd);
      }
    }
    let lastFrameTime = 0;
    let lastUIUpdate = 0;
    let animId = 0;
    let tick: (now: number) => void = () => {};
    const scheduleTick = () => {
      animId = requestAnimationFrame(tick);
    };
    const startDbg = show3dPerfDebug();
    const startFps = clampPlaybackFps(benchmarkPlaybackFpsRef.current ?? playRef.current.fps);
    if (startDbg) resetFramePacingDebug(startDbg, playbackIntervalMs(startFps));

    tick = (_now: number) => {
      const tickNow = performance.now();
      const c = playRef.current;
      const effectiveFps = clampPlaybackFps(benchmarkPlaybackFpsRef.current ?? c.fps);
      const intervalMs = playbackIntervalMs(effectiveFps);
      const uiUpdateIntervalMs = effectiveFps >= 60 ? 250 : 100;
      const dbg = show3dPerfDebug();
      if (dbg) {
        dbg.playing = true;
        dbg.effectiveFps = effectiveFps;
        dbg.lastTickAt = tickNow;
        dbg.currentBufferFloatLength = bufferRef.current?.length ?? 0;
        dbg.currentBufferStart = bufferStartRef.current;
        dbg.currentBufferCount = bufferCountRef.current;
        dbg.nextBufferFloatLength = nextBufferRef.current?.length ?? 0;
        dbg.nextBufferStart = nextBufferStartRef.current;
        dbg.nextBufferCount = nextBufferCountRef.current;
      }

      // First tick paints immediately; otherwise every playback start drops
      // one frame before the cadence timer is even allowed to run.
      if (lastFrameTime === 0) {
        lastFrameTime = tickNow - intervalMs;
        lastUIUpdate = tickNow;
      }

      const elapsed = tickNow - lastFrameTime;
      // Frame-pacing tolerance: at 60 fps intervalMs (16.67) equals the vsync
      // period, so a rAF tick arriving a hair early (elapsed 16.6 < 16.67) would
      // be dropped and cost a whole vsync -> steady 17/33 ms alternation = 30 fps.
      // Allow a tick that is within tolerance of the deadline through, and
      // phase-correct lastFrameTime by the deadline (not tickNow) so drift does
      // not accumulate. Restores 60 fps on the GPU-cached multi-panel path.
      const framePacingToleranceMs = Math.min(6, intervalMs * 0.2);
      if (elapsed + framePacingToleranceMs < intervalMs) {
        scheduleTick();
        return;
      }
      lastFrameTime = tickNow - Math.max(0, elapsed - intervalMs);

      // Advance frame
      let next: number;
      if (c.playbackPath && c.playbackPath.length > 0) {
        // Custom playback path
        const pp = c.playbackPath;
        let pi = pathIdxRef.current;
        if (c.boomerang) {
          // Visit endpoints once (matches grid-mode boomerang). Earlier code
          // jumped to pp.length-2 / 1 on overshoot, skipping endpoints.
          pi += bounceDirRef.current;
          if (pi >= pp.length) { bounceDirRef.current = -1; pi = pp.length - 1; }
          else if (pi < 0) { bounceDirRef.current = 1; pi = 0; }
        } else {
          pi += (c.reverse ? -1 : 1);
          if (pi >= pp.length) { if (!c.loop) { setPlaying(false); return; } pi = 0; }
          if (pi < 0) { if (!c.loop) { setPlaying(false); return; } pi = pp.length - 1; }
        }
        pi = Math.max(0, Math.min(pp.length - 1, pi));
        pathIdxRef.current = pi;
        next = pp[pi];
      } else {
        const rangeStart = c.loop ? Math.max(0, Math.min(c.loopStart, c.nSlices - 1)) : 0;
        const rangeEnd = c.loop ? Math.max(rangeStart, Math.min(c.loopEnd, c.nSlices - 1)) : c.nSlices - 1;
        const prev = playbackIdxRef.current;

        if (c.boomerang) {
          next = prev + bounceDirRef.current;
          if (next > rangeEnd) { bounceDirRef.current = -1; next = prev - 1 >= rangeStart ? prev - 1 : prev; }
          else if (next < rangeStart) { bounceDirRef.current = 1; next = prev + 1 <= rangeEnd ? prev + 1 : prev; }
        } else {
          next = prev + (c.reverse ? -1 : 1);
          if (c.reverse) {
            if (next < rangeStart) { if (!c.loop) { setPlaying(false); return; } next = rangeEnd; }
          } else {
            if (next > rangeEnd) { if (!c.loop) { setPlaying(false); return; } next = rangeStart; }
          }
        }
      }

      // OFFLINE mode (nbconvert HTML export with packed stack in widget state):
      // bypass ALL the kernel-fed buffer paths — bufferRef/nextBufferRef/
      // gpuFrameCacheUploadedRef are stale from prior pause+resume cycles and
      // can pin the canvas to a single frame. Always re-derive from the
      // offline stack so play→pause→play repaints correctly. Verified bug
      // 2026-05-24: 2nd autoplay cycle painted same frame from buffer cache.
      const frameSize = c.width * c.height;
      const transformActive = c.diffMode !== "off" || Math.max(1, Math.round(c.avgWindow || 1)) > 1;
      let frame: Float32Array | null = null;
      let frameSource = "buffer";
      // The GPU-cache fast paths (renderGpuPanelSlice / direct-grid) only handle
      // imageRotation%4===0; renderGpuPanelSlice bails (returns false) on a 90/270
      // rotation, which froze playback (renderedFrames + canvas stuck, playing
      // true). When rotated, skip the GPU-cache path so the frame is fetched and
      // drawMain applies the rotation. Verified bug 2026-05-29.
      const rotationAllowsGpuCache = (c.imageRotation % 4) === 0;
      const gpuCachedFrameReady = (offline || transformActive || !rotationAllowsGpuCache) ? false : gpuFrameCacheUploadedRef.current.has(next);
      const gpuPanelFrameReady = separatePanelFrames && gpuCachedFrameReady;
      if (offline) {
        frame = getOfflineFrame(next);
        if (frame) frameSource = "offline";
      } else if (gpuPanelFrameReady) {
        frameSource = "gpu-panel-cache";
      } else if (separatePanelFrames && rotationAllowsGpuCache && !transformActive) {
        frameSource = "gpu-panel-fetch";
        void ensurePanelFrameGpu(next, Math.max(1, Math.round(c.canvasW * c.canvasH)));
        if (dbg) {
          dbg.lastPanelGpuRequestedFrame = next;
          dbg.lastFrameSource = frameSource;
        }
      } else if (!gpuCachedFrameReady) {
        frame = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, c.nSlices, next, frameSize);
        if (!frame && nextBufferRef.current) {
          // Current buffer doesn't have this frame - swap to next buffer
          bufferRef.current = nextBufferRef.current;
          bufferStartRef.current = nextBufferStartRef.current;
          bufferCountRef.current = nextBufferCountRef.current;
          nextBufferRef.current = null;
          nextBufferCountRef.current = 0;
          frame = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, c.nSlices, next, frameSize);
        }
        if (!frame && frameServerUrl) {
          frame = getCachedServerFrame(next);
          if (frame) {
            frameSource = "server";
          } else {
            prefetchServerFrames(next, c.reverse, c.loop, c.loopStart, c.loopEnd);
          }
        }
        if (!frame) {
          frame = getOfflineFrame(next);
          if (frame) frameSource = "offline";
        }
      }
      if (!frame && !gpuCachedFrameReady) {
        // Buffer not ready yet - keep requesting frames
        if (dbg) {
          dbg.missingFrame = next;
          dbg.missingFrameAt = tickNow;
        }
        scheduleTick();
        return;
      }
      if (dbg) {
        dbg.missingFrame = null;
        dbg.lastFrame = next;
        dbg.lastFrameSource = frameSource;
      }

      playbackIdxRef.current = next;
      updatePlaybackLiveControls(next);
      if (frame && transformActive) {
        frame = displayFrameForIndex(next, frame) ?? frame;
      }
      if (frame) rawFrameDataRef.current = frame;
      const offlineDirectRender = offline && !!frame && !!gpuCmapRef.current && gpuCmapReadyRef.current;
      // Static offline paint is driven by liveSliceIdx. When WebGPU is ready we
      // render offline frames directly in the rAF hot path and throttle React
      // state updates below so large 2k/4k exports do not double-paint.
      const liveGpuFrameRender = !offline && !!frame && !transformActive && !!gpuCmapRef.current && gpuCmapReadyRef.current;
      const gpuDirectRender = gpuCachedFrameReady || gpuPanelFrameReady || liveGpuFrameRender;
      if (!offlineDirectRender && !gpuDirectRender) setLiveSliceIdx(next);
      // Offline mode short-circuit: hand the frame to the React static paint
      // pipeline (proven smooth on slider drag) and skip the rAF direct paint
      // entirely. The two paths fought on Mac/retina (Linux didn't expose it),
      // producing the "play is flaky while drag is smooth" symptom verified
      // 2026-05-24 on sample_device_trial.html.
      if (offline && !offlineDirectRender) {
        setGpuDisplayVisible(false);
        const d = show3dPerfDebug();
        if (d) {
          recordFramePacingDebug(d, performance.now(), intervalMs);
          d.renderedFrames = ((d.renderedFrames as number | undefined) ?? 0) + 1;
        }
        if (tickNow - lastUIUpdate > uiUpdateIntervalMs) {
          lastUIUpdate = tickNow;
          setDisplaySliceIdx(next);
          setPlaybackUiSliceIdx(next);
          playbackHistogramCounterRef.current = (playbackHistogramCounterRef.current + 1) % 2;
          if (playbackHistogramCounterRef.current === 0) {
            void refreshHistogramRef.current?.(next);
          }
        }
        scheduleTick();
        return;
      }
      if (gpuPanelFrameReady) {
        if (!renderGpuPanelSlice(next, false)) {
          if (dbg) {
            dbg.missingFrame = next;
            dbg.lastRenderError = "separate panel GPU render failed";
          }
          scheduleTick();
          return;
        }
        const d = show3dPerfDebug();
        if (d) {
          recordFramePacingDebug(d, performance.now(), intervalMs);
          d.renderedFrames = ((d.renderedFrames as number | undefined) ?? 0) + 1;
        }
        if (tickNow - lastUIUpdate > uiUpdateIntervalMs) {
          lastUIUpdate = tickNow;
          setDisplaySliceIdx(next);
          setPlaybackUiSliceIdx(next);
          playbackHistogramCounterRef.current = (playbackHistogramCounterRef.current + 1) % 2;
          if (playbackHistogramCounterRef.current === 0) {
            void refreshHistogramRef.current?.(next);
          }
        }
        scheduleTick();
        return;
      }

      // Render frame. The 4k playback hot path must stay off the JS CPU:
      // one 4096^2 colormap loop alone is ~37 ms, before auto-contrast/canvas.
      const renderStartMs = performance.now();
      const lut = COLORMAPS[c.cmap] || COLORMAPS.inferno;
      if (mainOffscreenRef.current && mainImgDataRef.current) {
        let vmin: number, vmax: number;
        let cpuData: Float32Array | null = frame;
        let cpuDataAlreadyLogged = false;
        if (c.autoContrast) {
          const cached = transformActive ? null : (
            cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, next, c.logScale)
            || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, next, c.logScale)
          );
          if (cached) {
            ({ vmin, vmax } = cached);
          } else if (frame && c.logScale && logBufferRef.current) {
            applyLogScaleInPlace(frame, logBufferRef.current);
            ({ vmin, vmax } = percentileClip(logBufferRef.current, c.percentileLow, c.percentileHigh));
            cpuData = logBufferRef.current;
            cpuDataAlreadyLogged = true;
          } else if (frame) {
            ({ vmin, vmax } = percentileClip(frame, c.percentileLow, c.percentileHigh));
          } else {
            ({ vmin, vmax } = resolveDisplayRange(
              c.dataMin,
              c.dataMax,
              c.traitVmin,
              c.traitVmax,
              c.logScale,
              c.imageVminPct,
              c.imageVmaxPct,
            ));
          }
        } else {
          ({ vmin, vmax } = resolveDisplayRange(
            c.dataMin,
            c.dataMax,
            c.traitVmin,
            c.traitVmax,
            c.logScale,
            c.imageVminPct,
            c.imageVmaxPct,
          ));
        }

        let rendered = false;
        let drewDisplayDirect = false;
        const dw = Math.max(1, Math.round(c.width * c.displayScale));
        const dh = Math.max(1, Math.round(c.height * c.displayScale));
        const panelCountForGrid = Math.max(1, nPanels || 1);
        const allPanelsVisibleForDirect = c.visiblePanelIndices.length === panelCountForGrid;
        const panelTransformsForDirect = c.visiblePanelIndices.map((panel) => {
          const base = c.panelStates[panel] || initialState;
          return {
            zoom: c.linkPanels ? c.linkedState.zoom : base.zoom,
            panX: c.linkPanels ? c.linkedState.panX : base.panX,
            panY: c.linkPanels ? c.linkedState.panY : base.panY,
          };
        });
        const panelTransformsAreDefault = panelTransformsForDirect.every((transform) => (
          transform.zoom === 1 && transform.panX === 0 && transform.panY === 0
        ));
        const packedPanelDirectCanvas = panelCountForGrid > 1 && !sharedPanelSource;
        const canDirectGridCanvas =
          allPanelsVisibleForDirect &&
          c.imageRotation % 4 === 0 &&
          (
            packedPanelDirectCanvas ||
            (c.zoom === 1 && c.panX === 0 && c.panY === 0 && panelTransformsAreDefault)
          );
        const canSharedPanelScaledDirect =
          !!sharedPanelSource &&
          canDirectGridCanvas &&
          panelTransformsAreDefault &&
          panelCountForGrid > 1;
        const canScaledDirect =
          (nPanels === 1 || canSharedPanelScaledDirect) &&
          c.imageRotation % 4 === 0 &&
          c.zoom === 1 &&
          c.panX === 0 &&
          c.panY === 0 &&
          dw <= c.canvasW &&
          dh <= c.canvasH;
        const drawSharedScaledBitmap = (ctx: CanvasRenderingContext2D, bitmap: ImageBitmap) => {
          const n = Math.max(1, nPanels || 1);
          const cols = panelColsForCount(n);
          const rows = Math.ceil(n / cols);
          const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
          const outPanelW = (c.canvasW - gap * (cols - 1)) / cols;
          const outPanelH = (c.canvasH - gap * (rows - 1)) / rows;
          ctx.clearRect(0, 0, c.canvasW, c.canvasH);
          ctx.fillStyle = themeColors.bg;
          ctx.imageSmoothingEnabled = c.smooth;
          for (let i = 0; i < n; i++) {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const slotX = col * (outPanelW + gap);
            const slotY = row * (outPanelH + gap);
            ctx.fillRect(slotX, slotY, outPanelW, outPanelH);
            ctx.drawImage(bitmap, slotX, slotY, outPanelW, outPanelH);
          }
        };
        const engine = gpuCmapRef.current;
        const preferGpuScaledPlayback = !!engine && gpuCmapReadyRef.current;
        if (frame && canScaledDirect && !canSharedPanelScaledDirect && !c.smooth && !preferGpuScaledPlayback) {
          const canvas = canvasRef.current;
          const ctx = canvas?.getContext("2d");
          if (ctx) {
            let cached = scaledPlaybackImgDataRef.current;
            if (!cached || cached.width !== dw || cached.height !== dh) {
              cached = { width: dw, height: dh, imageData: ctx.createImageData(dw, dh) };
              scaledPlaybackImgDataRef.current = cached;
            }
            let map = scaledPlaybackMapRef.current;
            if (!map || map.srcW !== c.width || map.srcH !== c.height || map.outW !== dw || map.outH !== dh) {
              const xMap = new Uint32Array(dw);
              const yMap = new Uint32Array(dh);
              for (let x = 0; x < dw; x++) {
                xMap[x] = Math.min(c.width - 1, Math.floor(((x + 0.5) * c.width) / dw));
              }
              for (let y = 0; y < dh; y++) {
                yMap[y] = Math.min(c.height - 1, Math.floor(((y + 0.5) * c.height) / dh)) * c.width;
              }
              map = { srcW: c.width, srcH: c.height, outW: dw, outH: dh, xMap, yMap };
              scaledPlaybackMapRef.current = map;
            }
            renderFrameScaledPlayback(frame, cached.imageData.data, map.xMap, map.yMap, dw, dh, lut, vmin, vmax, c.logScale);
            ctx.imageSmoothingEnabled = false;
            ctx.clearRect(0, 0, c.canvasW, c.canvasH);
            ctx.putImageData(cached.imageData, 0, 0);
            setGpuDisplayVisible(false);
            rendered = true;
            drewDisplayDirect = true;
            if (dbg) dbg.lastRenderPath = "scaled-cpu";
          }
        }
        if (!rendered && engine && gpuCmapReadyRef.current) {
          try {
            engine.uploadLUT(c.cmap, lut);
            const stackByteLength = c.width * c.height * 4 * c.nSlices;
            const hasGpuSlot = gpuFrameCacheUploadedRef.current.has(next);
            const canGpuFrameCache =
              !!frameServerUrl &&
              stackByteLength <= FRAME_SERVER_FULL_STACK_CACHE_BYTES &&
              (hasGpuSlot || frameFetchCacheRef.current.size >= c.nSlices);
            const slotIdx = canGpuFrameCache ? next : 0;
            const gpuRgbaCapacityHint = canDirectGridCanvas
              ? c.canvasW * c.canvasH
              : (canScaledDirect ? dw * dh : undefined);
            if (canGpuFrameCache) {
              if (!gpuFrameCacheUploadedRef.current.has(slotIdx)) {
                if (frame) {
                  engine.uploadData(slotIdx, frame, c.width, c.height, gpuRgbaCapacityHint);
                  gpuFrameCacheUploadedRef.current.add(slotIdx);
                  if (dbg) dbg.gpuFrameCacheUploaded = gpuFrameCacheUploadedRef.current.size;
                }
              } else if (dbg) {
                dbg.gpuFrameCacheHits = ((dbg.gpuFrameCacheHits as number | undefined) ?? 0) + 1;
              }
            } else if (frame) {
              engine.uploadData(0, frame, c.width, c.height, gpuRgbaCapacityHint);
              if (dbg) dbg.gpuFrameCacheUploaded = 0;
            }
            if (canDirectGridCanvas) {
              const gpuCtx = ensureGpuDisplayContext(engine, c.canvasW, c.canvasH);
              if (gpuCtx) {
                const n = panelCountForGrid;
                const cols = panelColsForCount(n);
                const rows = Math.ceil(n / cols);
                const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
                const sourcePanelWidthForGrid = sharedPanelSource
                  ? Math.max(1, panelWidthPx || c.width)
                  : Math.max(1, panelWidthPx || Math.round(c.width / n));
                const gridOpts = {
                  width: c.canvasW,
                  height: c.canvasH,
                  panelCount: n,
                  cols,
                  rows,
                  gap,
                  bgRgb: packedRgbFromHex(themeColors.bg),
                  sourcePanelWidth: sourcePanelWidthForGrid,
                  sharedSource: !!sharedPanelSource,
                };
                const usedPackedPanelRegions = n > 1 && !sharedPanelSource;
                if (usedPackedPanelRegions) {
                  const panelW = sourcePanelWidthForGrid;
                  const sharedAutoRange = c.autoContrast ? { vmin, vmax } : null;
                  const ranges = c.linkContrast
                    ? { vmin, vmax }
                    : Array.from({ length: n }, (_, p) => {
                        const panelData = frame ? extractPanelSlice(frame, p, c.logScale) : null;
                        // In per-panel mode, ALWAYS prefer this panel's stored
                        // data range so slider pct decodes in panel space (not
                        // stack space). Without this SSB phase [±0.04] gets
                        // decoded against stack range [≈-0.04, ≈30000] → vmin
                        // and vmax both land in DF-count territory → all SSB
                        // pixels render black.
                        const pdr = panelDataRanges[p];
                        const panelRange = panelData && panelData.length > 0
                          ? findDataRange(panelData)
                          : ((perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
                              ? pdr
                              : resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale));
                        return resolvePanelRenderRange(p, panelRange, sharedAutoRange, panelData, c.autoContrast, c.percentileLow, c.percentileHigh);
                      });
                  rendered = engine.renderCombinedPanelRegionsDirectToCanvas(
                    slotIdx,
                    ranges,
                    c.logScale,
                    gpuCtx,
                    {
                      width: c.canvasW,
                      height: c.canvasH,
                      panelCount: n,
                      cols,
                      rows,
                      gap,
                      bgRgb: packedRgbFromHex(themeColors.bg),
                      sourcePanelWidth: panelW,
                      transforms: panelTransformsForDirect,
                    },
                  );
                  if (dbg) {
                    dbg.lastRenderPath = rendered
                      ? "webgpu-grid-packed-panels-direct-fragment"
                      : "webgpu-grid-packed-panels-direct-fragment-miss";
                    dbg.lastPanelTransforms = panelTransformsForDirect.map(t => ({
                      zoom: Number(t.zoom.toFixed(3)),
                      panX: Number(t.panX.toFixed(1)),
                      panY: Number(t.panY.toFixed(1)),
                    }));
                  }
                } else {
                  const renderedDirect = engine.renderSharedGridDirectToCanvas(slotIdx, { vmin, vmax }, c.logScale, gpuCtx, gridOpts);
                  rendered = renderedDirect || engine.renderSharedGridToCanvas(slotIdx, { vmin, vmax }, c.logScale, gpuCtx, gridOpts);
                  if (dbg) {
                    const gridPath = n === 1
                      ? "webgpu-grid-single-panel"
                      : (sharedPanelSource ? "webgpu-grid-shared-panels" : "webgpu-grid-panels");
                    dbg.lastRenderPath = renderedDirect ? `${gridPath}-direct-fragment` : gridPath;
                  }
                }
                if (rendered) {
                  setGpuDisplayVisible(true);
                  drewDisplayDirect = true;
                }
              }
            }
            if (canScaledDirect) {
              const bitmap = rendered
                ? null
                : engine.renderSlotScaledToImageBitmap(slotIdx, { vmin, vmax }, c.logScale, dw, dh);
              const canvas = canvasRef.current;
              const ctx = canvas?.getContext("2d");
              if (bitmap && ctx) {
                if (canSharedPanelScaledDirect) {
                  drawSharedScaledBitmap(ctx, bitmap);
                } else {
                  ctx.imageSmoothingEnabled = c.smooth;
                  ctx.clearRect(0, 0, c.canvasW, c.canvasH);
                  ctx.drawImage(bitmap, 0, 0, dw, dh);
                }
                setGpuDisplayVisible(false);
                bitmap.close();
                rendered = true;
                drewDisplayDirect = true;
                if (dbg) dbg.lastRenderPath = canSharedPanelScaledDirect ? "scaled-gpu-shared-panels" : "scaled-gpu";
              } else {
                bitmap?.close();
              }
            }
            if (!rendered && frame) {
              const bitmaps = engine.renderSlotsToImageBitmap([slotIdx], [{ vmin, vmax }], c.logScale);
              if (bitmaps && bitmaps[0]) {
                const offCtx = mainOffscreenRef.current.getContext("2d");
                if (offCtx) {
                  offCtx.drawImage(bitmaps[0], 0, 0);
                  rendered = true;
                  if (dbg) dbg.lastRenderPath = "full-gpu";
                }
                bitmaps[0].close();
              }
            }
          } catch (err) {
            if (dbg) {
              dbg.lastRenderError = err instanceof Error ? err.message : String(err);
            }
            rendered = false;
            drewDisplayDirect = false;
          }
        }
        if (!rendered) {
          if (!frame && !cpuData) {
            if (dbg) {
              dbg.missingFrame = next;
              dbg.missingFrameAt = tickNow;
            }
            scheduleTick();
            return;
          }
          if (cpuDataAlreadyLogged && cpuData) {
            renderToOffscreenReuse(cpuData, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
          } else if (frame) {
            renderFramePlayback(frame, mainImgDataRef.current.data, lut, vmin, vmax, c.logScale);
            mainOffscreenRef.current.getContext("2d")!.putImageData(mainImgDataRef.current, 0, 0);
          }
          if (dbg) dbg.lastRenderPath = "cpu";
        }

        // Draw to display canvas. Apply image_rotation so playback matches the
        // static render path (lines 1444-1453); otherwise rotated stacks
        // silently lose their rotation when the user hits Play.
        const canvas = canvasRef.current;
        if (canvas && !drewDisplayDirect) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            if ((nPanels || 1) > 1) {
              drawMain(ctx, mainOffscreenRef.current);
            } else {
              ctx.imageSmoothingEnabled = c.smooth;
              ctx.clearRect(0, 0, c.canvasW, c.canvasH);
              ctx.save();
              ctx.translate(c.panX, c.panY);
              ctx.scale(c.zoom, c.zoom);
              const dw = c.width * c.displayScale, dh = c.height * c.displayScale;
              if (c.imageRotation % 4 !== 0) {
                const cx = c.canvasW / 2 / c.zoom, cy = c.canvasH / 2 / c.zoom;
                ctx.translate(cx, cy);
                ctx.rotate((c.imageRotation * Math.PI) / 2);
                ctx.translate(-dw / 2, -dh / 2);
                ctx.drawImage(mainOffscreenRef.current, 0, 0, dw, dh);
              } else {
                ctx.drawImage(mainOffscreenRef.current, 0, 0, dw, dh);
              }
              ctx.restore();
            }
          }
        }
      }
      if (dbg) {
        dbg.lastRenderMs = Number((performance.now() - renderStartMs).toFixed(2));
        recordFramePacingDebug(dbg, performance.now(), intervalMs);
        dbg.renderedFrames = ((dbg.renderedFrames as number | undefined) ?? 0) + 1;
      }

      // Throttled UI updates for slider/stats/profile. At the 60 fps cap, keep React
      // comfortably out of the frame loop; the canvas still renders every rAF.
      // liveSliceIdx is per-tick for static offline paint and throttled for
      // direct WebGPU offline paint to avoid a competing React render path.
      if (tickNow - lastUIUpdate > uiUpdateIntervalMs) {
        lastUIUpdate = tickNow;
        if (offlineDirectRender) setLiveSliceIdx(next);
        setDisplaySliceIdx(next);
        setPlaybackUiSliceIdx(next);
        if (frame && c.showStats) setLocalStats(computeStats(frame));
        if (frame && c.profileActive && c.profilePoints.length === 2) {
          const p0 = c.profilePoints[0], p1 = c.profilePoints[1];
          setProfileData(sampleLineProfile(frame, c.width, c.height, p0.row, p0.col, p1.row, p1.col, c.profileWidth));
        }
        // Histogram refresh during playback. The non-playback effect path is keyed on
        // frameBytes/frameSeq which DON'T change during rAF playback (frames come from
        // the prefetch buffer, not via Comm), so we drive histogram updates directly
        // here at the same 10 Hz cadence. Skip every 2nd tick → ~5 Hz refresh.
        playbackHistogramCounterRef.current = (playbackHistogramCounterRef.current + 1) % 2;
        if (playbackHistogramCounterRef.current === 0) {
          if ((nPanels || 1) > 1 && !linkContrast && frame) {
            // Per-panel histograms have no single GPU slot; keep the per-panel
            // CPU extract (cold-ish, only when contrast is unlinked).
            const n = Math.max(1, nPanels || 1);
            const nextData: (Float32Array | null)[] = Array.from({ length: n }, () => null);
            const nextRanges: { min: number; max: number }[] = Array.from(
              { length: n },
              () => resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale),
            );
            for (const panel of c.visiblePanelIndices) {
              const panelData = extractPanelSlice(frame, panel, c.logScale);
              nextData[panel] = panelData;
              nextRanges[panel] = panelData && panelData.length > 0
                ? findDataRange(panelData)
                : resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale);
            }
            setPanelHistogramData(nextData);
            setPanelDataRanges(nextRanges);
          } else {
            // GPU histogram for the current frame (honors WebGPU-first-class):
            // refreshHistogram computes bins on the GPU (live slot or offline
            // scratch slot) AND sets lastHistogramFrame so it is verifiable.
            // Replaces the old CPU setImageHistogramData(frame).
            void refreshHistogramRef.current?.(next);
          }
        }
      }

      // Prefetch at 25% buffer consumed - only if no next buffer is already queued.
      // Respect loop range so we don't fetch frames outside [loop_start, loop_end].
      if (!offline && frameServerUrl && separatePanelFrames) {
        prefetchPanelGpuFrames(next, c.reverse, c.loop, c.loopStart, c.loopEnd);
      } else if (!offline && frameServerUrl && gpuFrameCacheUploadedRef.current.size < c.nSlices) {
        prefetchServerFrames(next, c.reverse, c.loop, c.loopStart, c.loopEnd);
      } else if (!prefetchPendingRef.current && !nextBufferRef.current && bufferCountRef.current > 0) {
        let idxInBuffer = next - bufferStartRef.current;
        if (idxInBuffer < 0) idxInBuffer += c.nSlices;
        if (idxInBuffer >= Math.floor(bufferCountRef.current / 4)) {
          let prefetchStart = (bufferStartRef.current + bufferCountRef.current) % c.nSlices;
          // If loop range is constrained, snap prefetch start into it so we
          // don't waste buffer on frames the loop will never display.
          if (c.loop && (c.loopStart > 0 || c.loopEnd >= 0)) {
            const rs = Math.max(0, Math.min(c.loopStart, c.nSlices - 1));
            const re = c.loopEnd < 0 ? c.nSlices - 1 : Math.max(rs, Math.min(c.loopEnd, c.nSlices - 1));
            if (prefetchStart < rs || prefetchStart > re) prefetchStart = rs;
          }
          prefetchPendingRef.current = true;
          setPrefetchRequest(prefetchStart);
        }
      }

      scheduleTick();
    };

    scheduleTick();
    return () => {
      cancelAnimationFrame(animId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  // Update frame ref when frame changes
  React.useEffect(() => {
    const parsed = extractFloat32(frameBytes, width * height);
    if (!parsed || parsed.length === 0) return;
    const displayFrame = displayFrameForIndex(offline ? liveSliceIdx : sliceIdx, parsed) ?? parsed;
    rawFrameDataRef.current = displayFrame;
    gpuUploadRef.current = null;
    if (!showStats) {
      setLocalStats(null);
      setLocalPanelStats(null);
      return;
    }
    // Recompute stats JS-side only while visible. On 4k frames this is a full
    // 16M-float scan, so keep the default hidden state paint-limited.
    const n = Math.max(1, nPanels || 1);
    const total = computeStats(displayFrame);
    setLocalStats(total);
    if (n > 1 && height > 0 && width > 0 && width % n === 0) {
      const pw = width / n;
      const panels: PanelStats[] = [];
      for (const p of visiblePanelIndices) {
        // Slice columns [p*pw, (p+1)*pw) for all rows.
        const slab = new Float32Array(height * pw);
        for (let r = 0; r < height; r++) {
          const srcOff = r * width + p * pw;
          slab.set(displayFrame.subarray(srcOff, srcOff + pw), r * pw);
        }
        panels.push({ panel: p, ...computeStats(slab) });
      }
      setLocalPanelStats(panels);
    } else {
      setLocalPanelStats(null);
    }
  }, [frameBytes, frameSeq, nPanels, visiblePanelIndices, width, height, showStats, diffMode, avgWindow, offline, liveSliceIdx, sliceIdx]);

  // Histogram bins are computed on the GPU via `engine.computeHistogramWithRange`
  // when the colormap engine is ready. CPU fallback (computeHistogramFromBytes
  // inside the Histogram component) still runs if WebGPU isn't available.
  // Debounce: 100 ms past the last scrub frame so drag doesn't fire bin scans
  // on every tick. Playback uses the established 2-tick (5 Hz) throttle.
  const histogramTimerRef = React.useRef<number | null>(null);
  const histogramRefreshInFlightRef = React.useRef(false);
  const histogramRefreshPendingIdxRef = React.useRef<number | null>(null);
  const histogramRefreshSerialRef = React.useRef(0);
  const refreshHistogram = React.useCallback(async (idxArg?: number) => {
    const renderIdx = clampSlice(idxArg ?? displaySliceIdx);
    if (histogramRefreshInFlightRef.current) {
      histogramRefreshPendingIdxRef.current = renderIdx;
      return;
    }
    histogramRefreshInFlightRef.current = true;
    const serial = ++histogramRefreshSerialRef.current;
    try {
      // Offline path: ensurePanelFrameGpu returns false offline, so the GPU
      // block below never runs and the CPU fallback hits raw==null for
      // separate-panel -> histogram frozen on frame 0 during offline playback
      // (the frame-server slots don't exist offline). Bin the dequantized
      // offline frame directly so the histogram tracks the playing frame.
      if (offline && !perPanelHistogramEnabled) {
        const offFrame = getOfflineFrame(renderIdx);
        if (offFrame && offFrame.length) {
          const engine = gpuCmapRef.current;
          let bins: number[] | null = null;
          // GPU histogram (operator: everything WebGPU). Upload the dequantized
          // offline frame to a reserved scratch slot, then compute bins on GPU.
          if (engine && gpuCmapReadyRef.current && dataMax > dataMin) {
            try {
              const rgbaCapacity = Math.max(1, width * height);
              engine.uploadData(OFFLINE_HIST_SLOT, offFrame, width, height, rgbaCapacity, true);
              bins = await engine.computeHistogramWithRange(OFFLINE_HIST_SLOT, dataMin, dataMax, logScale);
            } catch {
              bins = null;  // Histogram component CPU-bins from imageHistogramData below
            }
          }
          const dbg = show3dPerfDebug();
          if (dbg) { dbg.lastHistogramFrame = renderIdx; dbg.lastHistogramSource = bins ? "offline-gpu" : "offline-cpu"; }
          setImageDataRange(resolveDisplayBounds(dataMin, dataMax, null, null, logScale));
          setImageHistogramBins(bins);
          setImageHistogramData(bins ? null : (logScale ? applyLogScale(offFrame) : offFrame));
          return;
        }
      }
      if (!perPanelHistogramEnabled) {
        const engine = gpuCmapRef.current;
        if (
          engine &&
          gpuCmapReadyRef.current &&
          (separatePanelFrames || gpuFrameCacheUploadedRef.current.has(renderIdx)) &&
          dataMax > dataMin
        ) {
          let bins: number[] | null = null;
          try {
            if (separatePanelFrames) {
              const rgbaCapacity = Math.max(1, Math.round(canvasW * canvasH));
              const ready = await ensurePanelFrameGpu(renderIdx, rgbaCapacity);
              if (ready) {
                const summed = new Array<number>(256).fill(0);
                const sourcePanelCount = Math.max(1, nPanels || 1);
                for (const panel of visiblePanelIndices) {
                  const panelBins = await engine.computeHistogramWithRange(renderIdx * sourcePanelCount + panel, dataMin, dataMax, logScale);
                  if (!panelBins) {
                    bins = null;
                    break;
                  }
                  for (let i = 0; i < summed.length; i++) summed[i] += panelBins[i] ?? 0;
                  bins = summed;
                }
              }
            } else {
              bins = await engine.computeHistogramWithRange(renderIdx, dataMin, dataMax, logScale);
            }
          } catch {
            bins = null;
          }
          if (serial === histogramRefreshSerialRef.current && bins) {
            const dbg = show3dPerfDebug();
            if (dbg) {
              dbg.lastHistogramFrame = renderIdx;
              dbg.lastHistogramSource = separatePanelFrames ? "gpu-panel-slots" : "gpu-cache";
            }
            setImageDataRange(resolveDisplayBounds(dataMin, dataMax, null, null, logScale));
            setImageHistogramBins(bins);
            setImageHistogramData(null);
            return;
          }
        }
      }

      const raw = rawFrameDataRef.current;
      if (!raw || raw.length === 0) return;
      if (perPanelHistogramEnabled) {
        const n = Math.max(1, nPanels || 1);
        const nextData: (Float32Array | null)[] = Array.from({ length: n }, () => null);
        const nextRanges: { min: number; max: number }[] = Array.from(
          { length: n },
          () => resolveDisplayBounds(dataMin, dataMax, null, null, logScale),
        );
        for (const panel of visiblePanelIndices) {
          const panelData = extractPanelSlice(raw, panel, logScale);
          nextData[panel] = panelData;
          nextRanges[panel] = panelData && panelData.length > 0
            ? findDataRange(panelData)
            : resolveDisplayBounds(dataMin, dataMax, null, null, logScale);
        }
        setPanelHistogramData(nextData);
        setPanelDataRanges(nextRanges);
        setImageHistogramBins(null);
        return;
      }
      const data = logScale ? applyLogScale(raw) : raw;
      setImageDataRange(resolveDisplayBounds(dataMin, dataMax, null, null, logScale));
      // GPU bins: the colormap engine has the frame data uploaded to slot 0
      // already (via the render effect). Reuse that slot's buffer for a
      // 256-bin compute pass; fall back to CPU bins in the Histogram component
      // when the engine isn't ready or returns null.
      const engine = gpuCmapRef.current;
      let bins: number[] | null = null;
      if (engine && gpuCmapReadyRef.current && dataMax > dataMin) {
        try {
          // Use the requested frame's slot, not a hardcoded 0 (which is whatever
          // the data effect last uploaded, not the playing frame).
          const slot = gpuFrameCacheUploadedRef.current.has(renderIdx) ? renderIdx : 0;
          bins = await engine.computeHistogramWithRange(slot, dataMin, dataMax, logScale);
        } catch {
          bins = null;  // fall through to CPU path
        }
      }
      const dbg = show3dPerfDebug();
      if (dbg) { dbg.lastHistogramFrame = renderIdx; dbg.lastHistogramSource = bins ? "gpu-slot" : "cpu-data"; }
      setImageHistogramBins(bins);
      setImageHistogramData(data);
    } finally {
      histogramRefreshInFlightRef.current = false;
      const pending = histogramRefreshPendingIdxRef.current;
      histogramRefreshPendingIdxRef.current = null;
      if (pending !== null && pending !== renderIdx) {
        window.setTimeout(() => { void refreshHistogram(pending); }, 0);
      }
    }
  }, [logScale, dataMin, dataMax, perPanelHistogramEnabled, nPanels, visiblePanelIndices, extractPanelSlice, displaySliceIdx, separatePanelFrames, canvasW, canvasH, ensurePanelFrameGpu]);
  refreshHistogramRef.current = refreshHistogram;
  React.useEffect(() => {
    if (playing) {
      return;
    }
    playbackHistogramCounterRef.current = 0;
    if (histogramTimerRef.current !== null) {
      window.clearTimeout(histogramTimerRef.current);
    }
    histogramTimerRef.current = window.setTimeout(() => {
      refreshHistogram(displaySliceIdx);
      histogramTimerRef.current = null;
    }, 32);
  }, [frameBytes, frameSeq, playing, displaySliceIdx, refreshHistogram]);

  // Auto-snap thumbs to percentile-clip values while Auto is on. Fires once at mount
  // (so the slider visually reflects the percentile-clipped contrast that Python applies
  // when auto_contrast=True), and re-fires when logScale flips (linear vs log percentile
  // give different clip values, so the thumbs must follow). The lastLogScaleRef tracks
  // the previous logScale value so we only re-snap on transitions, not on every render.
  const initialAutoSnappedRef = React.useRef(false);
  const lastLogScaleRef = React.useRef(logScale);
  const lastAutoContrastRef = React.useRef(autoContrast);
  React.useEffect(() => {
    const logScaleChanged = lastLogScaleRef.current !== logScale;
    // Detect Auto toggled false -> true (user re-engages Auto).
    // Re-snap thumbs to auto range whenever Auto turns back on.
    const autoToggledOn = !lastAutoContrastRef.current && autoContrast;
    lastLogScaleRef.current = logScale;
    lastAutoContrastRef.current = autoContrast;
    if (perPanelHistogramEnabled) return;
    if (!autoContrast || !imageHistogramData || imageHistogramData.length === 0) return;
    // Skip initial snap if user already moved thumbs (e.g. loaded from saved state).
    if (!initialAutoSnappedRef.current && (imageVminPct !== 0 || imageVmaxPct !== 100)) {
      initialAutoSnappedRef.current = true;
      return;
    }
    // After first snap, re-snap only on logScale OR Auto-toggle-on transitions.
    if (initialAutoSnappedRef.current && !logScaleChanged && !autoToggledOn) return;
    const { min: autoMin, max: autoMax } = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    const span = autoMax - autoMin;
    if (span <= 0) return;
    const cached = frameTransformActive() ? null : (
      cachedAutoDisplayRange(autoVmins, autoVmaxs, sliceIdx, logScale)
      || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, sliceIdx, logScale)
    );
    const { vmin: pmin, vmax: pmax } = cached ?? percentileClip(imageHistogramData, percentileLow, percentileHigh);
    setImageVminPct(Math.max(0, Math.min(100, ((pmin - autoMin) / span) * 100)));
    setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - autoMin) / span) * 100)));
    initialAutoSnappedRef.current = true;
  }, [autoContrast, imageHistogramData, dataMin, dataMax, traitVmin, traitVmax, autoVmins, autoVmaxs, sliceIdx, percentileLow, percentileHigh, logScale, imageVminPct, imageVmaxPct, perPanelHistogramEnabled]);

  // useEffect (not useLayoutEffect) so the per-panel auto-snap runs AFTER
  // the data effect populates panelHistogramData for the new frame.
  // useLayoutEffect fires BEFORE useEffects → rawFrameDataRef would be
  // stale and the snap would bail at mount.
  React.useEffect(() => {
    if (!perPanelHistogramEnabled || !autoContrast || panelHistogramData.length === 0) return;
    const stackBounds = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    if (stackBounds.max <= stackBounds.min) return;
    setPanelStates(prev => {
      const out = prev.map((state, i) => {
        // PER-PANEL auto: percentile-clip THIS panel's own data, then map
        // pct in THIS panel's data range. Mixed-unit stacks (BF/DF counts
        // vs SSB radians) span many orders of magnitude — using stack
        // range squashes tight panels to pct ≈ 0.
        const clip = panelAutoClipPcts(i, state, stackBounds);
        return clip ? { ...state, imageVminPct: clip.imageVminPct, imageVmaxPct: clip.imageVmaxPct } : state;
      });
      return out;
    });
  }, [perPanelHistogramEnabled, autoContrast, panelHistogramData, panelDataRanges, dataMin, dataMax, traitVmin, traitVmax, logScale, percentileLow, percentileHigh]);

  React.useEffect(() => {
    if (!perPanelHistogramEnabled || autoContrast || panelDataRanges.length === 0) return;
    setPanelStates(prev => {
      let changed = false;
      const out = prev.map((state, i) => {
        const storedMin = vminPerPanel[i];
        const storedMax = vmaxPerPanel[i];
        if (storedMin == null && storedMax == null) return state;
        const panelRange = panelDataRanges[i];
        const range = (panelRange && panelRange.max > panelRange.min)
          ? panelRange
          : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
        if (range.max <= range.min) return state;
        const lo = storedMin ?? range.min;
        const hi = Math.max(lo, storedMax ?? range.max);
        const nextMinPct = valueToPct(lo, range.min, range.max, state.imageVminPct);
        const nextMaxPct = valueToPct(hi, range.min, range.max, state.imageVmaxPct);
        if (Math.abs(nextMinPct - state.imageVminPct) < 0.01 && Math.abs(nextMaxPct - state.imageVmaxPct) < 0.01) return state;
        changed = true;
        return { ...state, imageVminPct: nextMinPct, imageVmaxPct: nextMaxPct };
      });
      return changed ? out : prev;
    });
  }, [perPanelHistogramEnabled, autoContrast, panelDataRanges, vminPerPanel, vmaxPerPanel, dataMin, dataMax, traitVmin, traitVmax, logScale]);

  React.useEffect(() => {
    if (!effectiveRoiActive || roiItems.length === 0 || !showRoiResizeHint) return;
    const timer = window.setTimeout(() => setShowRoiResizeHint(false), 6000);
    return () => window.clearTimeout(timer);
  }, [effectiveRoiActive, roiItems.length, showRoiResizeHint]);

  // Data effect: normalize + colormap → reusable offscreen canvas, then draw
  React.useEffect(() => {
    const frameData = rawFrameDataRef.current;
    if (!frameData || frameData.length === 0) return;
    if (!mainOffscreenRef.current || !mainImgDataRef.current) return;
    // Apply log scale using reusable buffer
    const processed = logScale && logBufferRef.current
      ? applyLogScaleInPlace(frameData, logBufferRef.current)
      : frameData;

    const nP = Math.max(1, nPanels || 1);
    const perPanelContrast = nP > 1 && !linkContrast && !sharedPanelSource && width % nP === 0 && height > 0;
    const transformActive = frameTransformActive();

    // Compute vmin/vmax (per-panel branch uses GPU multi-slot below)
    let vmin: number, vmax: number;
    const hasTraitRange = traitVmin != null || traitVmax != null;
    if (hasTraitRange) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (autoContrast) {
      const renderIdx = offline ? liveSliceIdx : sliceIdx;
      const cached = transformActive ? null : (
        cachedAutoDisplayRange(autoVmins, autoVmaxs, renderIdx, logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, renderIdx, logScale)
      );
      if (cached) {
        ({ vmin, vmax } = cached);
      } else {
        ({ vmin, vmax } = percentileClip(processed, percentileLow, percentileHigh));
      }
    } else {
      // Use the global data range (loaded once at widget mount) rather than
      // re-scanning the frame on every scrub. findDataRange does an O(N) min/max
      // pass which is ~8 ms at 4k - avoidable when the stack-wide bounds already
      // bracket the per-frame range.
      const lo = logScale ? (dataMin >= 0 ? Math.log1p(dataMin) : -Math.log1p(-dataMin)) : dataMin;
      const hi = logScale ? (dataMax >= 0 ? Math.log1p(dataMax) : -Math.log1p(-dataMax)) : dataMax;
      ({ vmin, vmax } = sliderRange(lo, hi, imageVminPct, imageVmaxPct));
    }

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    if (offline) {
      const canvas = canvasRef.current;
      const offscreen = mainOffscreenRef.current;
      const imgData = mainImgDataRef.current;
      const ctx = canvas?.getContext("2d");
      const offCtx = offscreen?.getContext("2d");
      if (!canvas || !offscreen || !imgData || !ctx || !offCtx) return;
      if (perPanelContrast) {
        offCtx.clearRect(0, 0, offscreen.width, offscreen.height);
        const panelW = Math.max(1, Math.floor(width / nP));
        const panelImg = offCtx.createImageData(panelW, height);
        const sharedAutoRange = autoContrast ? { vmin, vmax } : null;
        for (const p of visiblePanelIndices) {
          const panelData = extractPanelSlice(frameData, p, logScale);
          if (!panelData) continue;
          const pdr = panelDataRanges[p];
          const panelRange = panelData.length > 0
            ? findDataRange(panelData)
            : ((perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
                ? pdr
                : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale));
          const range = resolvePanelRenderRange(p, panelRange, sharedAutoRange, panelData, autoContrast, percentileLow, percentileHigh);
          applyColormap(panelData, panelImg.data, lut, range.vmin, range.vmax);
          offCtx.putImageData(panelImg, p * panelW, 0);
        }
      } else {
        renderToOffscreenReuse(processed, lut, vmin, vmax, offscreen, imgData);
      }
      drawMain(ctx, offscreen);
      return;
    }

    // GPU colormap path (single frame) - zero-copy via OffscreenCanvas→ImageBitmap
    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current) {
      engine.uploadLUT(cmap, lut);
      // Per-panel contrast: upload the FULL frame ONCE as slot 0, then run a
      // fused GPU pipeline that, per panel: reduces a sub-region → vmin/vmax,
      // colormaps the panel sub-image using those values + slider pcts, and
      // blits to a panel-sized OffscreenCanvas. No JS slab extraction, no
      // findDataRange loop, no CPU readback between range and colormap.
      const dataForGpu = perPanelContrast ? frameData : (logScale ? processed : frameData);
      const ensureGpuUpload = () => {
        const prev = gpuUploadRef.current;
        if (
          prev &&
          prev.source === frameData &&
          prev.data === dataForGpu &&
          prev.width === width &&
          prev.height === height &&
          prev.logScale === logScale
        ) {
          return;
        }
        engine.uploadData(0, dataForGpu, width, height);
        gpuUploadRef.current = { source: frameData, data: dataForGpu, width, height, logScale };
      };
      const renderSerial = ++gpuRenderSerialRef.current;
      if (perPanelContrast) {
        const pw = width / nP;
        ensureGpuUpload();
        const activePanels = visiblePanelIndices.filter((p) => p >= 0 && p < nP);
        const regions = activePanels.map((p) => ({
          x: p * pw, y: 0, width: pw, height,
        }));
        const sharedAutoRange = autoContrast ? { vmin, vmax } : null;
        const panelRanges = activePanels.map((p) => {
          const panelData = extractPanelSlice(frameData, p, logScale);
          const pdr = panelDataRanges[p];
          const panelRange = panelData && panelData.length > 0
            ? findDataRange(panelData)
            : ((perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
                ? pdr
                : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale));
          return resolvePanelRenderRange(p, panelRange, sharedAutoRange, panelData, autoContrast, percentileLow, percentileHigh);
        });
        const panelLogs = logScale;
        requestAnimationFrame(() => {
          if (renderSerial !== gpuRenderSerialRef.current) return;
          if (!mainOffscreenRef.current) return;
          const bitmaps = engine.renderPerPanelGpuExplicit(0, regions, panelRanges, panelLogs);
          if (bitmaps) {
            const ctx = mainOffscreenRef.current.getContext("2d");
            if (ctx) {
              for (let slot = 0; slot < activePanels.length; slot++) {
                const p = activePanels[slot];
                if (bitmaps[slot]) {
                  ctx.drawImage(bitmaps[slot], p * pw, 0);
                  bitmaps[slot].close();
                }
              }
            }
          }
          const canvas = canvasRef.current;
          if (!canvas) return;
          const ctx2 = canvas.getContext("2d");
          if (renderSerial !== gpuRenderSerialRef.current) return;
          if (ctx2 && mainOffscreenRef.current) drawMain(ctx2, mainOffscreenRef.current);
        });
        return;
      }
      ensureGpuUpload();
      const capturedVmin = vmin, capturedVmax = vmax;
      const blitAndDraw = async (): Promise<boolean> => {
        if (renderSerial !== gpuRenderSerialRef.current) return false;
        if (!mainOffscreenRef.current) return false;
        // Zero-copy: GPU → OffscreenCanvas → ImageBitmap → drawImage
        const bitmaps = engine.renderSlotsToImageBitmap([0], [{ vmin: capturedVmin, vmax: capturedVmax }], false);
        if (bitmaps && bitmaps[0]) {
          const ctx = mainOffscreenRef.current.getContext("2d");
          if (ctx) ctx.drawImage(bitmaps[0], 0, 0);
          // ImageBitmap holds external GPU/CPU memory not reclaimed by GC. Must close()
          // explicitly or repeated render calls (cmap/contrast/scrub) leak ~MB per call.
          bitmaps[0].close();
        } else {
          // Fallback: mapAsync path
          if (mainImgDataRef.current) {
            const rendered = await engine.renderSlots(
              [0], [{ vmin: capturedVmin, vmax: capturedVmax }],
              [mainOffscreenRef.current], [mainImgDataRef.current], false,
            );
            if (renderSerial !== gpuRenderSerialRef.current) return false;
            if (rendered === 0) {
              renderToOffscreenReuse(processed, lut, capturedVmin, capturedVmax, mainOffscreenRef.current!, mainImgDataRef.current!);
            }
          }
        }
        // Redraw main canvas (per-panel)
        const canvas = canvasRef.current;
        if (!canvas) return false;
        const ctx = canvas.getContext("2d");
        if (renderSerial !== gpuRenderSerialRef.current) return false;
        if (ctx && mainOffscreenRef.current) drawMain(ctx, mainOffscreenRef.current);
        return true;
      };
      requestAnimationFrame(async () => {
        const ok = await blitAndDraw();
        // Mac/Metal flush race: a one-shot static render captures the ImageBitmap
        // before the GPU submit has flushed ~2/3 of the time, leaving the canvas
        // blank until something re-renders. Playback's continuous rAF self-heals;
        // a static offline mount has no follow-up frame, so the panels stay black
        // (D6). Re-blit on a confirming second rAF when NOT playing - by the next
        // frame the GPU work has flushed and the bitmap is valid. Idempotent.
        if (ok && !playing) requestAnimationFrame(() => { void blitAndDraw(); });
      });
    } else {
      // WebGPU-only per CLAUDE.md "WebGPU is THE pipeline" rule.
      // setGpuCmapReady(true) upstream triggers this effect to re-fire as
      // soon as the engine resolves. Skip painting until then (canvas
      // briefly blank for ~50-200 ms on some GPU workstations, never CPU-rendered).
      gpuRenderSerialRef.current++;
    }

    // Draw to main canvas (CPU path only - GPU path draws in its own rAF above)
    if (!engine || !gpuCmapReadyRef.current) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (ctx && mainOffscreenRef.current) drawMain(ctx, mainOffscreenRef.current);
    }
  }, [frameBytes, frameSeq, width, height, cmap, displayScale, canvasW, canvasH, imageVminPct, imageVmaxPct, logScale, autoContrast, percentileLow, percentileHigh, traitVmin, traitVmax, dataMin, dataMax, autoVmins, autoVmaxs, smooth, imageRotation, nPanels, linkContrast, panelStates, panelDataRanges, vminPerPanel, vmaxPerPanel, offline, liveSliceIdx, sliceIdx, diffMode, avgWindow, playing, gpuCmapReady]);

  // Per-panel render: each slot gets its own zoom/pan transform. 2px gap
  // between slots painted as the canvas bg (transparent through clearRect).
  const drawMain = (
    ctx: CanvasRenderingContext2D,
    offscreen: HTMLCanvasElement | OffscreenCanvas,
    options: { preserveGpuDisplay?: boolean } = {},
  ) => {
    const drawSliceIdx = offline ? liveSliceIdx : displaySliceIdx;
    const keepDirectGpuVisible =
      !offline &&
      gpuCmapReadyRef.current &&
      gpuFrameCacheUploadedRef.current.has(displaySliceIdx) &&
      imageRotation % 4 === 0 &&
      (separatePanelFrames || hiddenPanelSet.size === 0) &&
      (separatePanelFrames || (
        linkedState.zoom === 1 &&
        linkedState.panX === 0 &&
        linkedState.panY === 0
      ));
    if (!keepDirectGpuVisible && !options.preserveGpuDisplay) {
      setGpuDisplayVisible(false);
    }
    ctx.imageSmoothingEnabled = smooth;
    // Clear entire canvas. Slot-level bg fill happens inside the per-panel
    // loop so empty grid cells (partial last row) stay transparent - the
    // page bg shows through instead of a dead white block.
    ctx.clearRect(0, 0, canvasW, canvasH);
    const n = Math.max(1, visiblePanelCount || 1);
    const sourcePanelCount = Math.max(1, nPanels || 1);
    const cols = panelColsForCount(n);
    const rows = Math.ceil(n / cols);
    const srcPanelW = sharedPanelSource
      ? offscreen.width
      : Math.max(1, panelWidthPx || offscreen.width / sourcePanelCount);
    const srcH = offscreen.height;
    const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
    const outPanelW = (canvasW - gap * (cols - 1)) / cols;
    const outPanelH = (canvasH - gap * (rows - 1)) / rows;
    for (let slot = 0; slot < n; slot++) {
      const i = visiblePanelIndices[slot] ?? slot;
      const panelState = stateFor(i);
      const col = slot % cols;
      const row = Math.floor(slot / cols);
      const slotX = col * (outPanelW + gap);
      const slotY = row * (outPanelH + gap);
      // Per-slot bg fill - only real panels get the theme bg; empty grid
      // cells in a partial last row stay transparent.
      ctx.fillStyle = themeColors.bg;
      ctx.fillRect(slotX, slotY, outPanelW, outPanelH);
      // End-of-stack: when current frame exceeds this panel's real frame
      // count, blur the (repeated last) frame + draw "end ({real}/{real})"
      // badge so operator sees they're scrubbing past real data.
      const realN = panelRealFrames && panelRealFrames[i];
      const pastEnd = !!(realN && drawSliceIdx >= realN);
      ctx.save();
      ctx.beginPath();
      ctx.rect(slotX, slotY, outPanelW, outPanelH);
      ctx.clip();
      ctx.translate(slotX + panelState.panX, slotY + panelState.panY);
      ctx.scale(panelState.zoom, panelState.zoom);
      const w = outPanelW, h = outPanelH;
      if (imageRotation % 4 !== 0) {
        const cx = w / 2 / panelState.zoom, cy = h / 2 / panelState.zoom;
        ctx.translate(cx, cy);
        ctx.rotate((imageRotation * Math.PI) / 2);
        ctx.translate(-w / 2, -h / 2);
      }
      if (pastEnd) ctx.filter = "blur(4px)";
      const srcX = sharedPanelSource ? 0 : i * srcPanelW;
      ctx.drawImage(offscreen as CanvasImageSource, srcX, 0, srcPanelW, srcH, 0, 0, w, h);
      ctx.restore();
      // No end badge - blur alone signals past-real-frame.
    }
  };

  const renderFloatFrameSlice = (inputFrame: Float32Array, idx: number): boolean => {
    const c = playRef.current;
    if (!mainOffscreenRef.current || !mainImgDataRef.current) return false;
    const transformActive = c.diffMode !== "off" || Math.max(1, Math.round(c.avgWindow || 1)) > 1;
    const frame = transformActive ? (displayFrameForIndex(idx, inputFrame) ?? inputFrame) : inputFrame;

    gpuRenderSerialRef.current++;
    playbackIdxRef.current = idx;
    rawFrameDataRef.current = frame;
    setDisplaySliceIdx(idx);

    const lut = COLORMAPS[c.cmap] || COLORMAPS.inferno;
    let vmin: number, vmax: number;
    let cpuData: Float32Array = frame;
    let cpuDataAlreadyLogged = false;
    if (c.autoContrast) {
      const cached = transformActive ? null : (
        cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, idx, c.logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, idx, c.logScale)
      );
      if (cached) {
        ({ vmin, vmax } = cached);
      } else if (c.logScale && logBufferRef.current) {
        applyLogScaleInPlace(frame, logBufferRef.current);
        ({ vmin, vmax } = percentileClip(logBufferRef.current, c.percentileLow, c.percentileHigh));
        cpuData = logBufferRef.current;
        cpuDataAlreadyLogged = true;
      } else {
        ({ vmin, vmax } = percentileClip(frame, c.percentileLow, c.percentileHigh));
      }
    } else {
      ({ vmin, vmax } = resolveDisplayRange(
        c.dataMin,
        c.dataMax,
        c.traitVmin,
        c.traitVmax,
        c.logScale,
        c.imageVminPct,
        c.imageVmaxPct,
      ));
    }

    let rendered = false;
    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current) {
      try {
        engine.uploadLUT(c.cmap, lut);
        engine.uploadData(0, frame, c.width, c.height);
        const bitmaps = engine.renderSlotsToImageBitmap([0], [{ vmin, vmax }], c.logScale);
        if (bitmaps && bitmaps[0]) {
          const offCtx = mainOffscreenRef.current.getContext("2d");
          if (offCtx) {
            offCtx.drawImage(bitmaps[0], 0, 0);
            rendered = true;
          }
          bitmaps[0].close();
        }
      } catch {
        rendered = false;
      }
    }
    if (!rendered) {
      if (cpuDataAlreadyLogged) {
        renderToOffscreenReuse(cpuData, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
      } else {
        renderFramePlayback(frame, mainImgDataRef.current.data, lut, vmin, vmax, c.logScale);
        mainOffscreenRef.current.getContext("2d")!.putImageData(mainImgDataRef.current, 0, 0);
      }
    }

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (ctx) drawMain(ctx, mainOffscreenRef.current);
    if (c.showStats) setLocalStats(computeStats(frame));
    if (c.profileActive && c.profilePoints.length === 2) {
      const p0 = c.profilePoints[0], p1 = c.profilePoints[1];
      setProfileData(sampleLineProfile(frame, c.width, c.height, p0.row, p0.col, p1.row, p1.col, c.profileWidth));
    }
    return true;
  };

  const renderBufferedSlice = (idx: number): boolean => {
    const c = playRef.current;
    const frameSize = c.width * c.height;
    let frame = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, c.nSlices, idx, frameSize);
    if (!frame && nextBufferRef.current) {
      const nextFrame = getFrameFromBuffer(nextBufferRef.current, nextBufferStartRef.current, nextBufferCountRef.current, c.nSlices, idx, frameSize);
      if (nextFrame) {
        bufferRef.current = nextBufferRef.current;
        bufferStartRef.current = nextBufferStartRef.current;
        bufferCountRef.current = nextBufferCountRef.current;
        nextBufferRef.current = null;
        nextBufferCountRef.current = 0;
        frame = nextFrame;
      }
    }
    if (!frame) return false;
    return renderFloatFrameSlice(frame, idx);
  };

  const renderFetchedSlice = async (idx: number): Promise<boolean> => {
    const transformActive = diffMode !== "off" || Math.max(1, Math.round(avgWindow || 1)) > 1;
    if (!transformActive && renderGpuCachedSliceDirect(idx)) return true;
    if (separatePanelFrames) {
      const c = playRef.current;
      const rgbaCapacity = Math.max(1, Math.round(c.canvasW * c.canvasH));
      const ready = await ensurePanelFrameGpu(idx, rgbaCapacity);
      if (!ready) return false;
      return renderGpuPanelSlice(idx);
    }
    const frame = getCachedServerFrame(idx) ?? await fetchFrameFromServer(idx);
    if (!frame) return false;
    return renderFloatFrameSlice(frame, idx);
  };

  const commitLivePanelTransforms = () => {
    if (transformStateCommitTimerRef.current !== null) {
      window.clearTimeout(transformStateCommitTimerRef.current);
      transformStateCommitTimerRef.current = null;
    }
    const nextLinked = linkedStateLiveRef.current;
    const nextPanels = panelStatesLiveRef.current;
    setViewState({ linked_state: { ...nextLinked }, panel_states: nextPanels.map(v => ({ ...v })) });
    setLinkedState(prev => (
      prev.zoom === nextLinked.zoom &&
      prev.panX === nextLinked.panX &&
      prev.panY === nextLinked.panY
        ? prev
        : { ...prev, zoom: nextLinked.zoom, panX: nextLinked.panX, panY: nextLinked.panY }
    ));
    setPanelStates(prev => {
      const n = Math.max(prev.length, nextPanels.length);
      let changed = prev.length !== n;
      const merged = Array.from({ length: n }, (_, i) => {
        const base = prev[i] || initialState;
        const live = nextPanels[i] || base;
        if (
          base.zoom !== live.zoom ||
          base.panX !== live.panX ||
          base.panY !== live.panY ||
          base.imageVminPct !== live.imageVminPct ||
          base.imageVmaxPct !== live.imageVmaxPct
        ) {
          changed = true;
        }
        return { ...base, ...live };
      });
      return changed ? merged : prev;
    });
  };

  const scheduleTransformStateCommit = (delayMs = 120) => {
    if (transformStateCommitTimerRef.current !== null) {
      window.clearTimeout(transformStateCommitTimerRef.current);
    }
    transformStateCommitTimerRef.current = window.setTimeout(commitLivePanelTransforms, delayMs);
  };

  const renderCurrentPanelTransformDirect = (): boolean => {
    if (!separatePanelFrames) {
      const canvas = canvasRef.current;
      const offscreen = mainOffscreenRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !offscreen || !ctx) return false;
      const start = performance.now();
      drawMain(ctx, offscreen);
      const dbg = show3dPerfDebug();
      if (dbg) {
        const latencyMs = transformInputAtRef.current > 0 ? performance.now() - transformInputAtRef.current : 0;
        dbg.lastInteractionRenderMs = Number((performance.now() - start).toFixed(2));
        dbg.lastInteractionLatencyMs = Number(latencyMs.toFixed(2));
        dbg.lastInteractionRenderFrame = playbackIdxRef.current;
        dbg.lastInteractionRenderPath = "canvas-packed-transform";
      }
      return true;
    }
    if (offline || imageRotation % 4 !== 0) return false;
    const n = Math.max(1, nSlices || 1);
    const idx = ((Math.round(playbackIdxRef.current) % n) + n) % n;
    if (!gpuFrameCacheUploadedRef.current.has(idx)) return false;
    const start = performance.now();
    const rendered = renderGpuPanelSlice(idx, false);
    const dbg = show3dPerfDebug();
    if (dbg) {
      const latencyMs = transformInputAtRef.current > 0 ? performance.now() - transformInputAtRef.current : 0;
      dbg.lastInteractionRenderMs = Number((performance.now() - start).toFixed(2));
      dbg.lastInteractionLatencyMs = Number(latencyMs.toFixed(2));
      dbg.lastInteractionRenderFrame = idx;
      dbg.lastInteractionRenderPath = rendered ? "webgpu-panel-transform" : "miss";
    }
    return rendered;
  };

  const scheduleTransformRender = (): boolean => {
    if (separatePanelFrames && (offline || imageRotation % 4 !== 0)) return false;
    if (transformRenderRafRef.current !== null) return true;
    transformRenderRafRef.current = window.requestAnimationFrame(() => {
      transformRenderRafRef.current = null;
      renderCurrentPanelTransformDirect();
    });
    return true;
  };

  React.useEffect(() => () => {
    if (transformRenderRafRef.current !== null) {
      window.cancelAnimationFrame(transformRenderRafRef.current);
      transformRenderRafRef.current = null;
    }
    if (transformStateCommitTimerRef.current !== null) {
      window.clearTimeout(transformStateCommitTimerRef.current);
      transformStateCommitTimerRef.current = null;
    }
  }, []);

  React.useEffect(() => {
    if (offline || !separatePanelFrames || !frameServerUrl || playing) return;
    void renderFetchedSlice(sliceIdx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offline, separatePanelFrames, frameServerUrl, frameServerVersion, sliceIdx, playing, canvasW, canvasH, cmap, imageVminPct, imageVmaxPct, autoContrast, logScale, panelStates, linkedState, linkPanels, panelGapTrait, maxCols]);

  React.useLayoutEffect(() => {
    if (!mainOffscreenRef.current || !canvasRef.current) return;
    const preserveGpuDisplay = playing && gpuDisplayVisibleRef.current === true && imageRotation % 4 === 0;
    if (preserveGpuDisplay && separatePanelFrames) return;
    const ctx = canvasRef.current.getContext("2d");
    if (ctx) drawMain(ctx, mainOffscreenRef.current, { preserveGpuDisplay });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [smooth, canvasW, canvasH, nPanels, maxCols, imageRotation, panelStates, linkedState, linkPanels, themeColors.bg, panelRealFrames, panelTitles, showPanelTitles, panelGapTrait, panelTitleFontSize, panelWidthPx, sharedPanelSource, sliceIdx, displaySliceIdx, liveSliceIdx, offline, playing, nSlices]);

  // Render overlay (ROI only) - HiDPI aware
  React.useEffect(() => {
    if (!overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);
    // Match the main image's rotation so ROIs / profile sit on the right pixels.
    // Image draw applies `translate(panX,panY) → scale(zoom) → rotate(around cx)`,
    // so the rotation pivot in screen pixels is (canvasW/2+panX, canvasH/2+panY).
    // Overlay must use the SAME screen-space pivot - earlier bug used (canvasW/2,
    // canvasH/2) without pan offset, drifting ROIs when user panned + rotated.
    if (imageRotation % 4 !== 0) {
      const cx = canvasW / 2 + panX;
      const cy = canvasH / 2 + panY;
      ctx.translate(cx, cy);
      ctx.rotate((imageRotation * Math.PI) / 2);
      ctx.translate(-cx, -cy);
    }
    if (effectiveRoiActive && roiItems.length > 0) {
      const highlightedRois = roiItems.filter(r => r.highlight);
      if (highlightedRois.length > 0) {
        ctx.save();
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(0, 0, canvasW, canvasH);
        ctx.globalCompositeOperation = "destination-out";
        for (const roi of highlightedRois) {
          const sx = roi.col * displayScale * zoom + panX;
          const sy = roi.row * displayScale * zoom + panY;
          const sr = roi.radius * displayScale * zoom;
          const shape = roi.shape || "circle";
          ctx.fillStyle = "rgba(0,0,0,1)";
          if (shape === "circle") {
            ctx.beginPath(); ctx.arc(sx, sy, sr, 0, Math.PI * 2); ctx.fill();
          } else if (shape === "square") {
            ctx.fillRect(sx - sr, sy - sr, sr * 2, sr * 2);
          } else if (shape === "rectangle") {
            const sw = roi.width * displayScale * zoom;
            const sh = roi.height * displayScale * zoom;
            ctx.fillRect(sx - sw / 2, sy - sh / 2, sw, sh);
          } else if (shape === "annular") {
            ctx.beginPath(); ctx.arc(sx, sy, sr, 0, Math.PI * 2); ctx.fill();
            ctx.globalCompositeOperation = "source-over";
            ctx.fillStyle = "rgba(0,0,0,0.6)";
            const sir = roi.radius_inner * displayScale * zoom;
            ctx.beginPath(); ctx.arc(sx, sy, sir, 0, Math.PI * 2); ctx.fill();
            ctx.globalCompositeOperation = "destination-out";
          }
        }
        ctx.restore();
      }

      for (let roiIdx = 0; roiIdx < roiItems.length; roiIdx++) {
        const roi = roiItems[roiIdx];
        const isSelected = roiIdx === roiSelectedIdx;
        const screenX = roi.col * displayScale * zoom + panX;
        const screenY = roi.row * displayScale * zoom + panY;
        const screenRadius = roi.radius * displayScale * zoom;
        const screenWidth = roi.width * displayScale * zoom;
        const screenHeight = roi.height * displayScale * zoom;
        const screenRadiusInner = roi.radius_inner * displayScale * zoom;
        const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
        ctx.lineWidth = roi.line_width || 2;
        const color = roi.color || ROI_COLORS[roiIdx % ROI_COLORS.length];
        drawROI(ctx, screenX, screenY, shape, screenRadius, screenWidth, screenHeight, color, color, isSelected && isDraggingROI, screenRadiusInner);
        if (isSelected) {
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1;
          if (shape === "circle" || shape === "annular") {
            ctx.beginPath(); ctx.arc(screenX, screenY, screenRadius + 3, 0, Math.PI * 2); ctx.stroke();
          } else if (shape === "square") {
            ctx.strokeRect(screenX - screenRadius - 3, screenY - screenRadius - 3, (screenRadius + 3) * 2, (screenRadius + 3) * 2);
          } else if (shape === "rectangle") {
            ctx.strokeRect(screenX - screenWidth / 2 - 3, screenY - screenHeight / 2 - 3, screenWidth + 6, screenHeight + 6);
          }
          ctx.setLineDash([]);
        }
      }
    }

    // Line profile overlay. Use the same slot, clip, zoom, pan, and rotation
    // transform as drawMain so profiles stay attached to their panel.
    if (profileActive && profilePoints.length > 0) {
      const ownerPanel = Math.max(0, Math.min(_nPanelsLocal - 1, profilePanelIdx));
      const geom = getPanelGeometry(ownerPanel);
      if (geom) {
        const toPanelX = (col: number) => panelLocalCol(col, ownerPanel) * geom.scaleX;
        const toPanelY = (row: number) => row * geom.scaleY;
        const markerR = 4 / Math.max(1, geom.state.zoom);
        ctx.save();
        ctx.beginPath();
        ctx.rect(geom.slotX, geom.slotY, geom.slotW, geom.slotH);
        ctx.clip();
        ctx.translate(geom.slotX + geom.state.panX, geom.slotY + geom.state.panY);
        ctx.scale(geom.state.zoom, geom.state.zoom);
        if (imageRotation % 4 !== 0) {
          const cx = geom.slotW / 2 / geom.state.zoom;
          const cy = geom.slotH / 2 / geom.state.zoom;
          ctx.translate(cx, cy);
          ctx.rotate((imageRotation * Math.PI) / 2);
          ctx.translate(-geom.slotW / 2, -geom.slotH / 2);
        }

        // Draw point A
        const ax = toPanelX(profilePoints[0].col);
        const ay = toPanelY(profilePoints[0].row);
        ctx.fillStyle = themeColors.accent;
        ctx.beginPath();
        ctx.arc(ax, ay, markerR, 0, Math.PI * 2);
        ctx.fill();

        if (profilePoints.length === 2) {
          const bx = toPanelX(profilePoints[1].col);
          const by = toPanelY(profilePoints[1].row);

          // Draw band when profile width > 1
          if (profileWidth > 1) {
            const dc = profilePoints[1].col - profilePoints[0].col;
            const dr = profilePoints[1].row - profilePoints[0].row;
            const lineLen = Math.sqrt(dc * dc + dr * dr);
            if (lineLen > 0) {
              const halfW = (profileWidth - 1) / 2;
              const perpR = -dc / lineLen * halfW;
              const perpC = dr / lineLen * halfW;
              ctx.fillStyle = themeColors.accent + "20";
              ctx.strokeStyle = themeColors.accent;
              ctx.lineWidth = 1 / Math.max(1, geom.state.zoom);
              ctx.setLineDash([3 / Math.max(1, geom.state.zoom), 3 / Math.max(1, geom.state.zoom)]);
              ctx.beginPath();
              ctx.moveTo(toPanelX(profilePoints[0].col + perpC), toPanelY(profilePoints[0].row + perpR));
              ctx.lineTo(toPanelX(profilePoints[1].col + perpC), toPanelY(profilePoints[1].row + perpR));
              ctx.lineTo(toPanelX(profilePoints[1].col - perpC), toPanelY(profilePoints[1].row - perpR));
              ctx.lineTo(toPanelX(profilePoints[0].col - perpC), toPanelY(profilePoints[0].row - perpR));
              ctx.closePath();
              ctx.fill();
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }

          ctx.strokeStyle = themeColors.accent;
          ctx.lineWidth = 1.5 / Math.max(1, geom.state.zoom);
          ctx.setLineDash([4 / Math.max(1, geom.state.zoom), 3 / Math.max(1, geom.state.zoom)]);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = themeColors.accent;
          ctx.beginPath();
          ctx.arc(bx, by, markerR, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }
    }
  }, [effectiveRoiActive, roiItems, roiSelectedIdx, isDraggingROI, canvasW, canvasH, displayScale, zoom, panX, panY, themeColors, profileActive, profilePoints, profileWidth, profilePanelIdx, nPanels, panelTitles, imageRotation, width, height, panelStates, linkedState, linkPanels, panelGapTrait, sourcePanelWidth, sourcePanelHeight, sharedPanelSource]);

  // Lens inset rendering
  React.useEffect(() => {
    const lensCanvas = lensCanvasRef.current;
    if (lensCanvas) {
      const lctx = lensCanvas.getContext("2d");
      if (lctx) lctx.clearRect(0, 0, lensCanvas.width, lensCanvas.height);
    }
    if (!showLens || !lensPos || !rawFrameDataRef.current) return;
    if ((nPanels || 1) > 1) return;  // Lens disabled in multi-panel mode
    if (!lensCanvas) return;
    const ctx = lensCanvas.getContext("2d");
    if (!ctx) return;

    const raw = rawFrameDataRef.current;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const processed = logScale ? applyLogScale(raw) : raw;
    let vmin: number, vmax: number;
    if (traitVmin != null || traitVmax != null) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (autoContrast) {
      const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
      ({ vmin, vmax } = cached ?? percentileClip(processed, percentileLow, percentileHigh));
    } else if (imageDataRange.min !== imageDataRange.max) {
      ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
    } else {
      const r = findDataRange(processed);
      vmin = r.min; vmax = r.max;
    }

    const regionSize = Math.max(4, Math.round(lensDisplaySize / lensMag));
    const lensSize = lensDisplaySize;
    const margin = 12;
    const half = Math.floor(regionSize / 2);
    const r0 = lensPos.row - half;
    const c0 = lensPos.col - half;

    const regionCanvas = document.createElement("canvas");
    regionCanvas.width = regionSize;
    regionCanvas.height = regionSize;
    const rctx = regionCanvas.getContext("2d");
    if (!rctx) return;
    const imgData = rctx.createImageData(regionSize, regionSize);
    const range = vmax - vmin || 1;
    for (let dr = 0; dr < regionSize; dr++) {
      for (let dc = 0; dc < regionSize; dc++) {
        const sr = r0 + dr;
        const sc = c0 + dc;
        const idx = (dr * regionSize + dc) * 4;
        if (sr < 0 || sr >= height || sc < 0 || sc >= width) {
          imgData.data[idx] = 0; imgData.data[idx + 1] = 0; imgData.data[idx + 2] = 0; imgData.data[idx + 3] = 255;
        } else {
          const val = processed[sr * width + sc];
          const t = Math.max(0, Math.min(1, (val - vmin) / range));
          const li = Math.round(t * 255);
          imgData.data[idx] = lut[li * 3]; imgData.data[idx + 1] = lut[li * 3 + 1]; imgData.data[idx + 2] = lut[li * 3 + 2]; imgData.data[idx + 3] = 255;
        }
      }
    }
    rctx.putImageData(imgData, 0, 0);

    ctx.save();
    ctx.scale(DPR, DPR);
    // Clamp anchor + default position to canvas bounds. Without clamp a small canvas
    // (e.g. multi-panel 100 px tall) puts the inset off-screen (-60 px) because
    // default ly = canvasH - lensSize - margin - 20 goes negative.
    const cssH = canvasH;
    const cssW = canvasW;
    const rawLx = lensAnchor ? lensAnchor.x : margin;
    const rawLy = lensAnchor ? lensAnchor.y : cssH - lensSize - margin - 20;
    const lx = Math.max(0, Math.min(cssW - lensSize, rawLx));
    const ly = Math.max(0, Math.min(cssH - lensSize, rawLy));
    ctx.imageSmoothingEnabled = smooth;
    ctx.drawImage(regionCanvas, lx, ly, lensSize, lensSize);
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(lx, ly, lensSize, lensSize);
    const cx = lx + lensSize / 2;
    const cy = ly + lensSize / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "10px monospace";
    ctx.fillText(`${lensMag}×`, lx + 4, ly + lensSize - 4);
    ctx.restore();
  }, [showLens, lensPos, cmap, logScale, autoContrast, imageDataRange, imageVminPct, imageVmaxPct, dataMin, dataMax, traitVmin, traitVmax, width, height, canvasW, canvasH, themeColors, lensMag, lensDisplaySize, lensAnchor, percentileLow, percentileHigh, frameBytes, sliceIdx, displaySliceIdx, nPanels]);

  // ROI sparkline plot
  React.useEffect(() => {
    const canvas = roiPlotCanvasRef.current;
    if (!canvas || !showRoiPlot || !effectiveRoiActive) return;
    const plotW = canvasW;
    const plotH = 76;
    canvas.width = Math.round(plotW * DPR);
    canvas.height = Math.round(plotH * DPR);
    canvas.style.width = `${plotW}px`;
    canvas.style.height = `${plotH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, plotW, plotH);

    if (!roiPlotData || roiPlotData.byteLength < 4) return;
    const values = extractFloat32(roiPlotData);
    if (!values || values.length === 0) return;
    let min = values[0], max = values[0];
    for (let i = 1; i < values.length; i++) {
      if (values[i] < min) min = values[i];
      if (values[i] > max) max = values[i];
    }
    const range = max - min || 1;
    const padY = 14;
    const drawH = plotH - padY * 2;

    // Draw plot line
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const denom = Math.max(1, values.length - 1);
    for (let i = 0; i < values.length; i++) {
      const x = (i / denom) * plotW;
      const y = padY + drawH - ((values[i] - min) / range) * drawH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw current frame marker
    const activeIdx = displaySliceIdx;
    const markerIdx = Math.max(0, Math.min(values.length - 1, activeIdx));
    const markerX = (markerIdx / denom) * plotW;
    ctx.strokeStyle = themeColors.textMuted;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(markerX, padY);
    ctx.lineTo(markerX, padY + drawH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Current value dot
    if (values.length > 0) {
      const cy = padY + drawH - ((values[markerIdx] - min) / range) * drawH;
      ctx.fillStyle = themeColors.accent;
      ctx.beginPath();
      ctx.arc(markerX, cy, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Y-axis labels
    ctx.fillStyle = themeColors.textMuted;
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText(formatNumber(max), 2, padY - 2);
    ctx.fillText(formatNumber(min), 2, padY + drawH + 10);
  }, [roiPlotData, effectiveRoiActive, showRoiPlot, canvasW, themeColors, sliceIdx, displaySliceIdx, playing]);

  // Keep sampled profile data current, but do not reopen the profile UI after
  // the user has turned it off. The line stays cached so toggling Profile back
  // on restores the latest sampled data.
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawFrameDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      const data = rawFrameDataRef.current;
      setProfileData(sampleLineProfile(data, width, height, p0.row, p0.col, p1.row, p1.col, profileWidth));
    } else {
      setProfileData(null);
    }
  }, [profilePoints, profileWidth, frameBytes]);

  // Render profile sparkline
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasW;
    const cssH = profileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, cssW, cssH);

    if (!profileData || profileData.length < 2) {
      ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = isDark ? "#555" : "#999";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Click two points on the image to draw a profile", cssW / 2, cssH / 2);
      return;
    }

    const padLeft = 40;
    const padRight = 8;
    const padTop = 6;
    const padBottom = 18;
    const plotW = cssW - padLeft - padRight;
    const plotH = cssH - padTop - padBottom;

    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < profileData.length; i++) {
      if (profileData[i] < gMin) gMin = profileData[i];
      if (profileData[i] > gMax) gMax = profileData[i];
    }
    const range = gMax - gMin || 1;

    // Draw profile line
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < profileData.length; i++) {
      const x = padLeft + (i / (profileData.length - 1)) * plotW;
      const y = padTop + plotH - ((profileData[i] - gMin) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // X-axis: calibrated distance
    let totalDist = profileData.length - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const dx = profilePoints[1].col - profilePoints[0].col;
      const dy = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      if (pixelSize > 0) {
        const distA = distPx * pixelSize;
        if (distA >= 10) { totalDist = distA / 10; xUnit = "nm"; }
        else { totalDist = distA; xUnit = "Å"; }
      } else {
        totalDist = distPx;
      }
    }

    // Draw x-axis ticks
    const tickY = padTop + plotH;
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    const idealTicks = Math.max(2, Math.floor(plotW / 70));
    const tickStep = roundToNiceValue(totalDist / idealTicks);
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textBaseline = "top";
    const ticks: number[] = [];
    for (let v = 0; v <= totalDist + tickStep * 0.01; v += tickStep) {
      if (v > totalDist * 1.001) break;
      ticks.push(v);
    }
    for (let i = 0; i < ticks.length; i++) {
      const v = ticks[i];
      const frac = totalDist > 0 ? v / totalDist : 0;
      const x = padLeft + frac * plotW;
      ctx.beginPath(); ctx.moveTo(x, tickY); ctx.lineTo(x, tickY + 3); ctx.stroke();
      ctx.textAlign = frac < 0.05 ? "left" : frac > 0.95 ? "right" : "center";
      const valStr = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${valStr} ${xUnit}` : valStr, x, tickY + 4);
    }

    // Y-axis min/max labels
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), padLeft - 3, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), padLeft - 3, padTop + plotH);

    // Draw axis lines
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Save base rendering + layout for hover overlay
    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [profileData, profilePoints, pixelSize, canvasW, themeInfo.theme, themeColors.accent, profileHeight]);

  // Profile hover handler - draws crosshair + value readout
  const handleProfileMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    const layout = profileLayoutRef.current;
    if (!canvas || !base || !layout) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit } = layout;
    const range = gMax - gMin || 1;

    ctx.putImageData(base, 0, 0);
    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Vertical crosshair
    ctx.strokeStyle = themeInfo.theme === "dark" ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on profile line + value
    if (profileData && profileData.length >= 2) {
      const dataIdx = Math.min(profileData.length - 1, Math.max(0, Math.round(frac * (profileData.length - 1))));
      const val = profileData[dataIdx];
      const y = padTop + plotH - ((val - gMin) / range) * plotH;
      ctx.fillStyle = themeColors.accent;
      ctx.beginPath();
      ctx.arc(cssX, y, 3, 0, Math.PI * 2);
      ctx.fill();

      // Value readout label
      const dist = frac * totalDist;
      const label = `${formatNumber(val)}  @  ${dist.toFixed(1)} ${xUnit}`;
      const isDark = themeInfo.theme === "dark";
      ctx.font = "bold 9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      const textW = ctx.measureText(label).width;
      const labelX = Math.min(cssX + 6, padLeft + plotW - textW - 2);
      const labelY = padTop + 2;
      ctx.fillStyle = isDark ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.8)";
      ctx.fillRect(labelX - 2, labelY - 1, textW + 4, 11);
      ctx.fillStyle = isDark ? "#fff" : "#000";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(label, labelX, labelY);
    }

    ctx.restore();
  };

  const handleProfileMouseLeave = () => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  };

  // Profile height resize
  React.useEffect(() => {
    if (!isResizingProfile) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!profileResizeStart) return;
      const delta = e.clientY - profileResizeStart.y;
      setProfileHeight(Math.max(40, Math.min(300, profileResizeStart.height + delta)));
    };
    const handleMouseUp = () => {
      setIsResizingProfile(false);
      setProfileResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingProfile, profileResizeStart]);

  // Render HiDPI scale bar + zoom indicator + colorbar
  React.useEffect(() => {
    if (!uiRef.current) return;
    const ctx = uiRef.current.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, uiRef.current.width, uiRef.current.height);
    if (scaleBarVisible) {
      const unit = pixelSize > 0 ? pixelUnit : "px";
      const pxSize = pixelSize > 0 ? pixelSize : 1;
      // Per-panel scale bar + zoom indicator. Each panel slot uses its
      // own panelStates[i].zoom so panels at different zoom levels show
      // their own length bar.
      const n = Math.max(1, visiblePanelCount || 1);
      const cols = panelColsForCount(n);
      const rows = Math.ceil(n / cols);
      const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
      const cssW = uiRef.current.width / DPR;
      const cssH = uiRef.current.height / DPR;
      const slotW = (cssW - gap * (cols - 1)) / cols;
      const slotH = (cssH - gap * (rows - 1)) / rows;
      ctx.save();
      ctx.scale(DPR, DPR);
      // Exact Show2D drawScaleBarHiDPI style: 60 px target, 5 px thickness,
      // 16 px font, 12 px margin. Per-panel: each slot acts as its own
      // canvas region with width=slotW, image source width=`width`.
      const targetBarPxSpec = 60;
      const barThickness = 5;
      const fontSize = 16;
      const margin = 12;
      ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
      for (let slot = 0; slot < n; slot++) {
        const i = visiblePanelIndices[slot] ?? slot;
        const panelState = stateFor(i);
        const col = slot % cols;
        const row = Math.floor(slot / cols);
        const slotX = col * (slotW + gap);
        const slotY = row * (slotH + gap);
        // Cap bar at 25% of slot width so it never overflows a small slot.
        const targetBarPx = Math.min(targetBarPxSpec, slotW * 0.25);
        const slotScale = slotW / sourcePanelWidth;
        const effectiveZoom = panelState.zoom * slotScale;
        const targetPhysical = (targetBarPx / effectiveZoom) * pxSize;
        const nicePhysical = (function (v: number) {
          if (v <= 0) return 1;
          const mag = Math.pow(10, Math.floor(Math.log10(v)));
          const norm = v / mag;
          if (norm < 1.5) return mag;
          if (norm < 3.5) return 2 * mag;
          if (norm < 7.5) return 5 * mag;
          return 10 * mag;
        })(targetPhysical);
        const barPx = (nicePhysical / pxSize) * effectiveZoom;
        const barY = slotY + slotH - margin;
        const barX = slotX + slotW - barPx - margin;
        ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
        ctx.shadowBlur = 2;
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;
        ctx.fillStyle = "white";
        ctx.fillRect(barX, barY, barPx, barThickness);
        const label = formatScaleLabel(nicePhysical, unit);
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(label, barX + barPx / 2, barY - 4);
        if (showZoomIndicator !== false && panelChromeVisible) {
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(`${panelState.zoom.toFixed(1)}×`, slotX + margin, slotY + slotH - margin + barThickness);
        }
      }
      ctx.restore();
    }
    if (showColorbar) {
      const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
      // Colorbar must match what's painted on the image, not the raw data range.
      // When autoContrast is on, the image uses percentileClip(low, high) of the
      // current frame - show that range. Otherwise use slider range over data.
      let vmin: number, vmax: number;
      if (traitVmin != null || traitVmax != null) {
        ({ vmin, vmax } = resolveDisplayRange(
          dataMin,
          dataMax,
          traitVmin,
          traitVmax,
          logScale,
          imageVminPct,
          imageVmaxPct,
        ));
      } else if (autoContrast && imageHistogramData && imageHistogramData.length > 0) {
        const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
          || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
        ({ vmin, vmax } = cached ?? percentileClip(imageHistogramData, percentileLow, percentileHigh));
      } else {
        ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
      }
      ctx.save();
      ctx.scale(DPR, DPR);
      const n = Math.max(1, visiblePanelCount || 1);
      const cols = panelColsForCount(n);
      const rows = Math.ceil(n / cols);
      const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
      const cssW = uiRef.current.width / DPR;
      const cssH = uiRef.current.height / DPR;
      const slotW = (cssW - gap * (cols - 1)) / cols;
      const slotH = (cssH - gap * (rows - 1)) / rows;
      const perPanelColorbar = n > 1 && !linkContrast && !sharedPanelSource;
      const currentFrame = rawFrameDataRef.current;
      const sharedAutoRange = autoContrast ? { vmin, vmax } : null;
      for (let slot = 0; slot < n; slot++) {
        const panel = visiblePanelIndices[slot] ?? slot;
        let panelVmin = vmin;
        let panelVmax = vmax;
        if (perPanelColorbar) {
          const panelData = currentFrame ? extractPanelSlice(currentFrame, panel, logScale) : null;
          const pdr = panelDataRanges[panel];
          const panelRange = panelData && panelData.length > 0
            ? findDataRange(panelData)
            : ((perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
                ? pdr
                : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale));
          const resolved = resolvePanelRenderRange(panel, panelRange, sharedAutoRange, panelData, autoContrast, percentileLow, percentileHigh);
          panelVmin = resolved.vmin;
          panelVmax = resolved.vmax;
        }
        const col = slot % cols;
        const row = Math.floor(slot / cols);
        const slotX = col * (slotW + gap);
        const slotY = row * (slotH + gap);
        ctx.save();
        ctx.beginPath();
        ctx.rect(slotX, slotY, slotW, slotH);
        ctx.clip();
        ctx.translate(slotX, slotY);
        drawColorbar(ctx, slotW, slotH, lut, panelVmin, panelVmax, logScale);
        ctx.restore();
      }
      ctx.restore();
    }
  }, [pixelSize, pixelUnit, scaleBarVisible, width, sourcePanelWidth, canvasW, canvasH, displayScale, zoom, nPanels, visiblePanelCount, visiblePanelIndices, maxCols, panelStates, linkedState, linkPanels, panelGapTrait, showZoomIndicator, panelChromeVisible, showColorbar, cmap, imageDataRange, imageVminPct, imageVmaxPct, logScale, autoContrast, imageHistogramData, autoVmins, autoVmaxs, displaySliceIdx, percentileLow, percentileHigh, dataMin, dataMax, traitVmin, traitVmax, linkContrast, sharedPanelSource, panelDataRanges, vminPerPanel, vmaxPerPanel]);

  // Compute FFT magnitude (expensive, async - only re-run on data/GPU changes)
  // Supports ROI-scoped FFT: when ROI is active with a selected ROI, compute
  // FFT of the cropped region instead of the full frame.
  type FftMagnitudeCacheEntry = {
    mag: Float32Array;
    cropDims: { cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null;
    grid: { panelWidth: number; panelHeight: number; cols: number; rows: number; count: number } | null;
    source: string;
    panels: number;
    gridLabel: string | null;
    sizeLabel: string;
  };
  const fftMagnitudeCacheBaseMaxBytes = 256 * 1024 * 1024;
  const fftMagRef = React.useRef<Float32Array | null>(null);
  const fftMagnitudeCacheRef = React.useRef<Map<string, FftMagnitudeCacheEntry>>(new Map());
  const fftActiveCacheKeyRef = React.useRef<string | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);

  React.useEffect(() => {
    if (!effectiveShowFft) return;
    // FFT is useful context, but for heavy multi-panel stacks it is far more
    // expensive than showing the next image frame. During playback keep the
    // last FFT visible and let the image path own the frame budget; when the
    // user pauses or finishes scrubbing, compute immediately for that settled
    // frame so overlay/bottom FFT does not sit blank behind render churn.
    if (playing) return;
    if (!rawFrameDataRef.current) {
      const idx = offline ? liveSliceIdx : displaySliceIdx;
      const offlineFrame = offline ? getOfflineFrame(idx) : null;
      const parsed = offlineFrame ?? extractFloat32(frameBytes, width * height);
      const frame = parsed
        ? (displayFrameForIndex(idx, parsed) ?? parsed)
        : null;
      if (frame) rawFrameDataRef.current = frame;
    }
    if (!rawFrameDataRef.current) return;
    let cancelled = false;
    const doCompute = async () => {
      const fftStartMs = performance.now();
      const data = rawFrameDataRef.current!;
      const panelCount = Math.max(1, nPanels || 1);
      const multiPanelFft = panelCount > 1 && !roiFftActive;
      const fftFrameIdx = clampSlice(offline ? liveSliceIdx : displaySliceIdx);
      const selectedRoi = roiFftActive && roiList && roiSelectedIdx >= 0 && roiSelectedIdx < roiList.length
        ? roiList[roiSelectedIdx]
        : null;
      const roiKey = selectedRoi
        ? JSON.stringify({
          idx: roiSelectedIdx,
          row: Math.round(Number(selectedRoi.row ?? 0) * 100) / 100,
          col: Math.round(Number(selectedRoi.col ?? 0) * 100) / 100,
          radius: Math.round(Number(selectedRoi.radius ?? 0) * 100) / 100,
          radius_inner: Math.round(Number(selectedRoi.radius_inner ?? 0) * 100) / 100,
          width: Math.round(Number(selectedRoi.width ?? 0) * 100) / 100,
          height: Math.round(Number(selectedRoi.height ?? 0) * 100) / 100,
          shape: selectedRoi.shape,
        })
        : "none";
      const fftGridCols = multiPanelFft ? panelColsForCount(Math.max(1, visiblePanelIndices.length || 1)) : 1;
      const fftCacheKey = [
        offline ? "offline" : "live",
        `frame=${fftFrameIdx}`,
        `seq=${frameSeq || 0}`,
        `server=${frameServerVersion || 0}`,
        `dims=${width}x${height}`,
        `panels=${panelCount}`,
        `visible=${visiblePanelIndices.join(",")}`,
        `cols=${fftGridCols}`,
        `sourceW=${sourcePanelWidth}`,
        `overlay=${fftLayoutOverlay ? 1 : 0}`,
        `overlayCap=${fftLayoutOverlay ? FFT_OVERLAY_MAX_SOURCE_SIZE : 0}`,
        `shared=${sharedPanelSource ? 1 : 0}`,
        `roi=${roiFftActive ? roiKey : "none"}`,
        `window=${fftWindow ? 1 : 0}`,
      ].join("|");
      const cache = fftMagnitudeCacheRef.current;
      const cached = cache.get(fftCacheKey);
      if (cached) {
        cache.delete(fftCacheKey);
        cache.set(fftCacheKey, cached);
        if (fftActiveCacheKeyRef.current === fftCacheKey) {
          return;
        }
        fftActiveCacheKeyRef.current = fftCacheKey;
        fftMagRef.current = cached.mag;
        fftMagCacheRef.current = cached.mag;
        fftPanelGridRef.current = cached.grid;
        fftCropDimsRef.current = cached.cropDims;
        setFftCropDims(cached.cropDims);
        setFftMagVersion(v => v + 1);
        const dbg = show3dPerfDebug();
        setFftBackendInfo(prev => ({
          ...prev,
          source: `${cached.source}-cache`,
          ms: 0,
          panels: cached.panels,
          grid: cached.gridLabel || "",
        }));
        if (dbg) {
          dbg.lastFftMs = 0;
          dbg.lastFftSource = `${cached.source}-cache`;
          dbg.lastFftPanels = cached.panels;
          dbg.lastFftSize = cached.sizeLabel;
          dbg.lastFftGrid = cached.gridLabel;
          dbg.fftCacheHits = Number(dbg.fftCacheHits || 0) + 1;
          dbg.fftCacheSize = cache.size;
          dbg.fftCacheBytes = Array.from(cache.values()).reduce((total, item) => total + item.mag.byteLength, 0);
        }
        return;
      }
      const rememberFft = (entry: FftMagnitudeCacheEntry) => {
        cache.set(fftCacheKey, entry);
        const maxEntries = Math.max(2, Math.min(24, nSlices || 12));
        const maxBytes = Math.max(fftMagnitudeCacheBaseMaxBytes, Math.min(1024 * 1024 * 1024, entry.mag.byteLength * 3));
        let totalBytes = Array.from(cache.values()).reduce((total, item) => total + item.mag.byteLength, 0);
        while (cache.size > maxEntries || totalBytes > maxBytes) {
          const oldest = cache.keys().next().value;
          if (oldest === undefined) break;
          const oldestEntry = cache.get(oldest);
          cache.delete(oldest);
          totalBytes -= oldestEntry?.mag.byteLength ?? 0;
        }
        const dbg = show3dPerfDebug();
        if (dbg) {
          dbg.fftCacheMisses = Number(dbg.fftCacheMisses || 0) + 1;
          dbg.fftCacheSize = cache.size;
          dbg.fftCacheBytes = totalBytes;
          dbg.fftComputes = Number(dbg.fftComputes || 0) + 1;
        }
      };

      if (multiPanelFft) {
        const panelW = sharedPanelSource
          ? Math.max(1, sourcePanelWidth)
          : Math.max(1, Math.floor(width / panelCount));
        const panelH = height;
        const overlayScale = fftLayoutOverlay
          ? Math.max(1, Math.ceil(Math.max(panelW, panelH) / FFT_OVERLAY_MAX_SOURCE_SIZE))
          : 1;
        const fftSourceW = Math.max(1, Math.ceil(panelW / overlayScale));
        const fftSourceH = Math.max(1, Math.ceil(panelH / overlayScale));
        const fftW = nextPow2(fftSourceW);
        const fftH = nextPow2(fftSourceH);
        const panels: { real: Float32Array; imag: Float32Array }[] = [];
        const fullW = data.length === height * panelW ? panelW : width;
        for (const panel of visiblePanelIndices) {
          const srcPanel = sharedPanelSource ? 0 : panel;
          const x0 = Math.min(Math.max(0, srcPanel * panelW), Math.max(0, fullW - panelW));
          if (data.length < height * fullW || x0 + panelW > fullW) continue;
          const real = new Float32Array(fftW * fftH);
          if (overlayScale > 1) {
            for (let y = 0; y < fftSourceH; y++) {
              const srcY = Math.min(panelH - 1, y * overlayScale);
              const srcOffset = srcY * fullW + x0;
              const dstOffset = y * fftW;
              for (let x = 0; x < fftSourceW; x++) {
                real[dstOffset + x] = data[srcOffset + Math.min(panelW - 1, x * overlayScale)];
              }
            }
          } else {
            for (let y = 0; y < panelH; y++) {
              real.set(data.subarray(y * fullW + x0, y * fullW + x0 + panelW), y * fftW);
            }
          }
          if (fftWindow) applyHannWindow2D(real, fftW, fftH);
          panels.push({ real, imag: new Float32Array(real.length) });
        }
        if (panels.length === 0) return;

        let results: { real: Float32Array; imag: Float32Array }[];
        let fftSource = "worker-batch";
        const offlineGpuTimeoutMs = 5000;
        const withOfflineTimeout = <T,>(promise: Promise<T>): Promise<T> => {
          if (!offline) return promise;
          return Promise.race([
            promise,
            new Promise<T>((_, reject) => window.setTimeout(() => reject(new Error("offline WebGPU FFT timed out")), offlineGpuTimeoutMs)),
          ]);
        };
        const offlineGpuDisabled = () => offlineFftGpuDisabledRef.current;
        const disableOfflineGpu = () => {
          offlineFftGpuDisabledRef.current = true;
        };
        const offlineGpuInFlight = () => offlineFftGpuInFlightRef.current;
        const skipOfflineWebGpu = offline && /HeadlessChrome/i.test(navigator.userAgent);
        if (offlineGpuInFlight()) return;
        const fftGpu = (!skipOfflineWebGpu && !offlineGpuDisabled() && !offlineGpuInFlight())
          ? await withOfflineTimeout(ensureFftGpu())
          : null;
        if (cancelled) return;
        if (fftGpu && panels.length > 1) {
          let startedOfflineGpu = false;
          try {
            if (offline) {
              offlineFftGpuInFlightRef.current = true;
              startedOfflineGpu = true;
            }
            results = await withOfflineTimeout(
              fftGpu.fft2DBatch(
                panels.map(({ real, imag }) => ({ real, imag })),
                fftW,
                fftH,
              )
            );
            fftSource = "webgpu-batch";
          } catch (err) {
            console.warn("Show3D WebGPU FFT failed; falling back to worker FFT.", err);
            if (offline) {
              disableOfflineGpu();
              results = panels.map(({ real, imag }) => {
                fft2d(real, imag, fftW, fftH, false);
                fftshift(real, fftW, fftH);
                fftshift(imag, fftW, fftH);
                return { real, imag };
              });
              fftSource = "cpu-sync-shifted";
            } else {
              results = (await Promise.all(panels.map(({ real, imag }) => fft2dAsync(real, imag, fftW, fftH, false))))
                .map(({ real, imag }) => ({ real, imag }));
            }
          } finally {
            if (startedOfflineGpu) offlineFftGpuInFlightRef.current = false;
          }
        } else if (offline) {
          results = panels.map(({ real, imag }) => {
            fft2d(real, imag, fftW, fftH, false);
            fftshift(real, fftW, fftH);
            fftshift(imag, fftW, fftH);
            return { real, imag };
          });
          fftSource = "cpu-sync-shifted";
        } else {
          results = (await Promise.all(panels.map(({ real, imag }) => fft2dAsync(real, imag, fftW, fftH, false))))
            .map(({ real, imag }) => ({ real, imag }));
        }
        if (cancelled) return;

        const cols = panelColsForCount(panels.length);
        const rows = Math.ceil(panels.length / cols);
        const gridW = cols * fftW;
        const gridH = rows * fftH;
        const gridMag = new Float32Array(gridW * gridH);
        const resultsAlreadyShifted = fftSource === "worker-batch" || fftSource === "cpu-sync-shifted";
        for (let panel = 0; panel < results.length; panel++) {
          const { real, imag } = results[panel];
          if (!resultsAlreadyShifted) {
            fftshift(real, fftW, fftH);
            fftshift(imag, fftW, fftH);
          }
          const mag = computeMagnitude(real, imag);
          const col = panel % cols;
          const row = Math.floor(panel / cols);
          const dstX = col * fftW;
          const dstY = row * fftH;
          for (let y = 0; y < fftH; y++) {
            gridMag.set(mag.subarray(y * fftW, y * fftW + fftW), (dstY + y) * gridW + dstX);
          }
        }

        fftMagRef.current = gridMag;
        fftActiveCacheKeyRef.current = fftCacheKey;
        fftMagCacheRef.current = gridMag;
        const gridInfo = { panelWidth: fftW, panelHeight: fftH, cols, rows, count: panels.length };
        const cropDims = { cropWidth: fftSourceW, cropHeight: fftSourceH, fftWidth: gridW, fftHeight: gridH };
        fftPanelGridRef.current = gridInfo;
        fftCropDimsRef.current = cropDims;
        setFftCropDims(cropDims);
        rememberFft({
          mag: gridMag,
          cropDims,
          grid: gridInfo,
          source: fftSource,
          panels: panels.length,
          gridLabel: `${gridW}x${gridH}`,
          sizeLabel: overlayScale > 1 ? `${fftW}x${fftH} overlay/${overlayScale}x` : `${fftW}x${fftH}`,
        });
        setFftMagVersion(v => v + 1);
        const dbg = show3dPerfDebug();
        const elapsedMs = Number((performance.now() - fftStartMs).toFixed(2));
        setFftBackendInfo(prev => ({
          ...prev,
          source: fftSource,
          ms: elapsedMs,
          panels: panels.length,
          grid: `${gridW}x${gridH}`,
        }));
        if (dbg) {
          dbg.lastFftMs = elapsedMs;
          dbg.lastFftSource = fftSource;
          dbg.lastFftPanels = panels.length;
          dbg.lastFftSize = overlayScale > 1 ? `${fftW}x${fftH} overlay/${overlayScale}x` : `${fftW}x${fftH}`;
          dbg.lastFftGrid = `${gridW}x${gridH}`;
        }
        return;
      }

      fftPanelGridRef.current = null;
      fftCropDimsRef.current = null;
      let fftW = width;
      let fftH = height;
      let inputData = data;

      // ROI crop: extract bounding box and optionally zero-mask outside radius
      let origCropW = 0, origCropH = 0;
      if (roiFftActive && roiList && roiSelectedIdx >= 0 && roiSelectedIdx < roiList.length) {
        const roi = roiList[roiSelectedIdx];
        const crop = cropROIRegion(data, width, height, roi);
        if (crop) {
          origCropW = crop.cropW;
          origCropH = crop.cropH;
          // Apply Hann window to crop at native dimensions BEFORE zero-padding
          if (fftWindow) applyHannWindow2D(crop.cropped, crop.cropW, crop.cropH);
          // Pad to next power-of-2 so fft2d doesn't truncate frequency data
          const padW = nextPow2(crop.cropW);
          const padH = nextPow2(crop.cropH);
          const padded = new Float32Array(padW * padH);
          for (let y = 0; y < crop.cropH; y++) {
            for (let x = 0; x < crop.cropW; x++) {
              padded[y * padW + x] = crop.cropped[y * crop.cropW + x];
            }
          }
          inputData = padded;
          fftW = padW;
          fftH = padH;
        }
      }

      // Pre-pad non-power-of-2 full images so fft2d doesn't truncate frequency data
      if (origCropW === 0) {
        const padW = nextPow2(fftW);
        const padH = nextPow2(fftH);
        if (padW !== fftW || padH !== fftH) {
          const padded = new Float32Array(padW * padH);
          for (let y = 0; y < fftH; y++) {
            for (let x = 0; x < fftW; x++) {
              padded[y * padW + x] = inputData[y * fftW + x];
            }
          }
          inputData = padded;
          fftW = padW;
          fftH = padH;
        }
      }

      let real: Float32Array, imag: Float32Array;

      let fftSource = "cpu";
      const fftGpu = await ensureFftGpu();
      if (cancelled) return;
      if (fftGpu) {
        try {
          const gpuReal = inputData.slice();
          const gpuImag = new Float32Array(inputData.length);
          const result = await fftGpu.fft2D(gpuReal, gpuImag, fftW, fftH, false);
          real = result.real;
          imag = result.imag;
          fftSource = "webgpu";
        } catch (err) {
          console.warn("Show3D WebGPU FFT failed; falling back to worker FFT.", err);
          const result = await fft2dAsync(inputData.slice(), new Float32Array(inputData.length), fftW, fftH, false);
          real = result.real;
          imag = result.imag;
          fftSource = "worker";
        }
      } else {
        const result = await fft2dAsync(inputData.slice(), new Float32Array(inputData.length), fftW, fftH, false);
        real = result.real;
        imag = result.imag;
        fftSource = "worker";
      }

      if (cancelled) return;
      if (fftSource !== "worker") {
        fftshift(real, fftW, fftH);
        fftshift(imag, fftW, fftH);
      }

      fftMagRef.current = computeMagnitude(real, imag);
      fftActiveCacheKeyRef.current = fftCacheKey;
      fftMagCacheRef.current = fftMagRef.current;
      // Track FFT dimensions when they differ from image dimensions (ROI crop or non-pow2 padding)
      let cropDims: { cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null = null;
      if (origCropW > 0) {
        cropDims = { cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH };
      } else if (fftW !== width || fftH !== height) {
        cropDims = { cropWidth: width, cropHeight: height, fftWidth: fftW, fftHeight: fftH };
      }
      fftCropDimsRef.current = cropDims;
      setFftCropDims(cropDims);
      rememberFft({
        mag: fftMagRef.current,
        cropDims,
        grid: null,
        source: fftSource,
        panels: 1,
        gridLabel: `${fftW}x${fftH}`,
        sizeLabel: `${fftW}x${fftH}`,
      });
      setFftMagVersion(v => v + 1);
      const dbg = show3dPerfDebug();
      const elapsedMs = Number((performance.now() - fftStartMs).toFixed(2));
      setFftBackendInfo(prev => ({
        ...prev,
        source: fftSource,
        ms: elapsedMs,
        panels: 1,
        grid: `${fftW}x${fftH}`,
      }));
      if (dbg) {
        dbg.lastFftMs = elapsedMs;
        dbg.lastFftSource = fftSource;
        dbg.lastFftPanels = 1;
        dbg.lastFftSize = `${fftW}x${fftH}`;
        dbg.lastFftGrid = null;
      }
    };

    void doCompute();

    return () => {
      cancelled = true;
    };
  }, [effectiveShowFft, playing, frameBytes, frameSeq, frameServerVersion, offline, liveSliceIdx, displaySliceIdx, width, height, roiFftActive, roiList, roiSelectedIdx, fftWindow, nPanels, nSlices, visiblePanelIndices, sourcePanelWidth, sharedPanelSource, maxCols, panelColsForCount, fftLayoutOverlay, extractPanelSlice, ensureFftGpu]);

  // Clear FFT measurement when ROI FFT state changes
  React.useEffect(() => { setFftClickInfo(null); }, [roiFftActive, roiSelectedIdx]);

  // Process FFT magnitude → histogram + colormap rendering (cheap, sync)
  React.useEffect(() => {
    const mag = fftMagRef.current;
    if (!effectiveShowFft || !mag) return;

    // Use ref-backed dimensions so the magnitude and its layout metadata remain
    // consistent in the same render tick; React state may lag by one effect.
    const cropDimsForRender = fftCropDimsRef.current;
    const fftW = cropDimsForRender?.fftWidth ?? width;
    const fftH = cropDimsForRender?.fftHeight ?? height;
    const grid = fftPanelGridRef.current;
    if (fftMetricsEnabled) {
      const qualityKey = `${fftMagVersion}:${fftW}x${fftH}:${pixelSize || 0}:${pixelUnit || ""}:${grid ? `${grid.panelWidth}x${grid.panelHeight}x${grid.cols}x${grid.count}` : "single"}`;
      if (fftQualityKeyRef.current !== qualityKey) {
        fftQualityKeyRef.current = qualityKey;
        const metricStartMs = performance.now();
        let nextQuality: FftQualityMetrics | null;
        if (grid) {
          const panelMetrics: Array<FftQualityMetrics | null> = [];
          for (let panel = 0; panel < grid.count; panel++) {
            panelMetrics.push(computeFftQualityMetrics(mag, fftW, fftH, {
              sampling: pixelSize,
              unit: pixelUnit,
              region: {
                x: (panel % grid.cols) * grid.panelWidth,
                y: Math.floor(panel / grid.cols) * grid.panelHeight,
                width: grid.panelWidth,
                height: grid.panelHeight,
              },
            }));
          }
          nextQuality = summarizeFftQualityMetrics(panelMetrics);
        } else {
          nextQuality = computeFftQualityMetrics(mag, fftW, fftH, { sampling: pixelSize, unit: pixelUnit });
        }
        setFftQuality(nextQuality);
        const dbg = show3dPerfDebug();
        if (dbg) {
          dbg.fftMetricComputes = Number(dbg.fftMetricComputes || 0) + 1;
          dbg.lastFftMetricMs = Number((performance.now() - metricStartMs).toFixed(2));
          dbg.lastFftMetricKey = qualityKey;
          dbg.lastFftMetricLabel = formatFftQualityLabel(nextQuality);
        }
      }
    } else if (fftQualityKeyRef.current) {
      fftQualityKeyRef.current = "";
      setFftQuality(null);
    }

    let displayMin: number, displayMax: number;
    let displayData: Float32Array;
    if (fftAuto && grid) {
      // Multi-panel FFTs can differ by orders of magnitude (BF/DF vs SSB).
      // Auto mode should reveal each panel, so normalize every FFT tile before
      // composing the shared canvas. Manual mode below intentionally stays global.
      displayData = new Float32Array(mag.length);
      const panelDisplay = new Float32Array(grid.panelWidth * grid.panelHeight);
      for (let panel = 0; panel < grid.count; panel++) {
        const col = panel % grid.cols;
        const row = Math.floor(panel / grid.cols);
        const srcX = col * grid.panelWidth;
        const srcY = row * grid.panelHeight;
        for (let y = 0; y < grid.panelHeight; y++) {
          const srcOffset = (srcY + y) * fftW + srcX;
          const dstOffset = y * grid.panelWidth;
          for (let x = 0; x < grid.panelWidth; x++) {
            // FFT magnitudes are extremely heavy-tailed; even in "Lin" UI mode,
            // auto contrast should reveal Bragg/fringe peaks instead of letting
            // the DC/low-frequency pedestal flatten the tile.
            panelDisplay[dstOffset + x] = Math.log1p(Math.max(0, mag[srcOffset + x]));
          }
        }
        const cx = Math.floor(grid.panelWidth / 2);
        const cy = Math.floor(grid.panelHeight / 2);
        const dcRadius = Math.max(2, Math.round(Math.min(grid.panelWidth, grid.panelHeight) * 0.01));
        const ringRadius = dcRadius + 2;
        let ringSum = 0;
        let ringCount = 0;
        for (let yy = Math.max(0, cy - ringRadius); yy <= Math.min(grid.panelHeight - 1, cy + ringRadius); yy++) {
          for (let xx = Math.max(0, cx - ringRadius); xx <= Math.min(grid.panelWidth - 1, cx + ringRadius); xx++) {
            const dist = Math.hypot(xx - cx, yy - cy);
            if (dist > dcRadius && dist <= ringRadius) {
              ringSum += panelDisplay[yy * grid.panelWidth + xx];
              ringCount++;
            }
          }
        }
        const dcFill = ringCount > 0 ? ringSum / ringCount : 0;
        for (let yy = Math.max(0, cy - dcRadius); yy <= Math.min(grid.panelHeight - 1, cy + dcRadius); yy++) {
          for (let xx = Math.max(0, cx - dcRadius); xx <= Math.min(grid.panelWidth - 1, cx + dcRadius); xx++) {
            panelDisplay[yy * grid.panelWidth + xx] = dcFill;
          }
        }
        suppressFftRadialBackgroundInPlace(panelDisplay, grid.panelWidth, grid.panelHeight);

        const range = findDataRange(panelDisplay);
        const clipped = percentileClip(panelDisplay, 5, 99.99);
        const pMin = clipped.vmin < clipped.vmax ? clipped.vmin : range.min;
        const pMax = clipped.vmax > pMin ? clipped.vmax : range.max;
        const denom = pMax > pMin ? pMax - pMin : 1;
        for (let y = 0; y < grid.panelHeight; y++) {
          const dstOffset = (srcY + y) * fftW + srcX;
          const srcOffset = y * grid.panelWidth;
          for (let x = 0; x < grid.panelWidth; x++) {
            const normalized = (panelDisplay[srcOffset + x] - pMin) / denom;
            displayData[dstOffset + x] = Math.max(0, Math.min(1, normalized));
          }
        }
      }
      displayMin = 0;
      displayMax = 1;
    } else {
      if (fftAuto) {
        ({ min: displayMin, max: displayMax } = autoEnhanceFFT(mag, fftW, fftH));
      } else {
        ({ min: displayMin, max: displayMax } = findDataRange(mag));
      }
      displayData = fftLogScale ? applyLogScale(mag) : mag;
      if (fftLogScale) {
        displayMin = Math.log1p(displayMin);
        displayMax = Math.log1p(displayMax);
      }
    }

    setFftHistogramData(displayData);
    setFftDataRange({ min: displayMin, max: displayMax });
    setFftStats(computeStats(displayData));

    const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(displayData, fftW, fftH, lut, vmin, vmax);
    if (!offscreen) return;

    fftOffscreenRef.current = offscreen;
    setFftOffscreenVersion(v => v + 1);

    if (fftCanvasRef.current) {
      const ctx = fftCanvasRef.current.getContext("2d");
      if (ctx) {
        drawFftOffscreen(ctx, offscreen);
      }
    }
  }, [effectiveShowFft, fftMagVersion, fftLogScale, fftAuto, fftVminPct, fftVmaxPct, fftColormap, width, height, canvasW, canvasH, fftCropDims, drawFftOffscreen, pixelSize, pixelUnit, fftMetricsEnabled]);

  // Redraw cached FFT with zoom/pan/resize before paint. Changing a canvas
  // width/height attribute clears its bitmap, so a normal effect can expose a
  // one-frame blank flash during resize drags.
  React.useLayoutEffect(() => {
    if (!effectiveShowFft || !fftCanvasRef.current || !fftOffscreenRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    drawFftOffscreen(ctx, fftOffscreenRef.current);
  }, [effectiveShowFft, fftOffscreenVersion, fftZoom, fftPanX, fftPanY, canvasW, canvasH, drawFftOffscreen]);

  const drawFftInsetLayer = React.useCallback((
    view: { zoom: number; panX: number; panY: number } = fftViewLiveRef.current,
  ) => {
    const canvas = fftInsetLayerRef.current;
    if (!canvas || !effectiveShowFft || !fftLayoutOverlay || !fftOffscreenRef.current) return;
    const offscreen = fftOffscreenRef.current;
    const grid = fftPanelGridRef.current;
    const count = grid ? grid.count : 1;
    const n = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(n);
    const rows = Math.ceil(n / cols);
    const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
    const panelW = (canvasW - gap * (cols - 1)) / cols;
    const panelH = (canvasH - gap * (rows - 1)) / rows;
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const srcW = grid ? grid.panelWidth : fftW;
    const srcH = grid ? grid.panelHeight : fftH;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = srcW < panelW || srcH < panelH;
    visiblePanelIndices.forEach((_panel, slot) => {
      if (slot >= count) return;
      const panelLeft = (slot % cols) * (panelW + gap);
      const panelTop = Math.floor(slot / cols) * (panelH + gap);
      const insetPad = Math.min(8, Math.max(3, panelW * 0.025));
      const insetMaxW = Math.max(24, panelW - insetPad * 2);
      const insetMaxH = Math.max(20, panelH - insetPad * 2);
      const insetBase = Math.min(insetMaxW, insetMaxH);
      const insetW = Math.max(24, Math.min(insetMaxW, insetBase * resolvedFftOverlaySize));
      const insetH = Math.max(20, Math.min(insetMaxH, insetBase * resolvedFftOverlaySize));
      const insetX = resolvedFftOverlayPosition.endsWith("right")
        ? panelLeft + panelW - insetW - insetPad
        : panelLeft + insetPad;
      const insetY = resolvedFftOverlayPosition.startsWith("bottom")
        ? panelTop + panelH - insetH - insetPad
        : panelTop + insetPad;
      const dstX = fftOverlayDragPreview ? panelLeft + fftOverlayDragPreview.x : insetX;
      const dstY = fftOverlayDragPreview ? panelTop + fftOverlayDragPreview.y : insetY;
      const srcX = grid ? (slot % grid.cols) * grid.panelWidth : 0;
      const srcY = grid ? Math.floor(slot / grid.cols) * grid.panelHeight : 0;
      const insetPanX = !fftUserAdjustedViewRef.current && view.zoom > 1
        ? insetW * (1 - view.zoom) / 2
        : view.panX;
      const insetPanY = !fftUserAdjustedViewRef.current && view.zoom > 1
        ? insetH * (1 - view.zoom) / 2
        : view.panY;
      ctx.save();
      ctx.fillStyle = "#000";
      ctx.fillRect(dstX, dstY, insetW, insetH);
      ctx.beginPath();
      ctx.rect(dstX, dstY, insetW, insetH);
      ctx.clip();
      ctx.translate(dstX + insetPanX, dstY + insetPanY);
      ctx.scale(view.zoom, view.zoom);
      ctx.drawImage(offscreen, srcX, srcY, srcW, srcH, 0, 0, insetW, insetH);
      ctx.restore();
      ctx.strokeStyle = "rgba(255,255,255,0.48)";
      ctx.lineWidth = 1;
      ctx.strokeRect(dstX + 0.5, dstY + 0.5, Math.max(0, insetW - 1), Math.max(0, insetH - 1));
    });
  }, [effectiveShowFft, fftLayoutOverlay, fftCropDims, width, height, visiblePanelCount, visiblePanelIndices, panelColsForCount, panelGapTrait, canvasW, canvasH, resolvedFftOverlaySize, resolvedFftOverlayPosition, fftOverlayDragPreview]);

  React.useEffect(() => {
    fftViewDirectRedrawRef.current = () => {
      if (!effectiveShowFft || !fftLayoutOverlay || !fftOffscreenRef.current) return;
      if (fftViewRafRef.current !== null) return;
      fftViewRafRef.current = window.requestAnimationFrame(() => {
        fftViewRafRef.current = null;
        drawFftInsetLayer(fftViewLiveRef.current);
      });
    };
    return () => {
      fftViewDirectRedrawRef.current = null;
    };
  }, [drawFftInsetLayer, effectiveShowFft, fftLayoutOverlay]);

  React.useLayoutEffect(() => {
    if (!effectiveShowFft || !fftLayoutOverlay || !fftOffscreenRef.current) return;
    drawFftInsetLayer();
  }, [effectiveShowFft, fftLayoutOverlay, fftOffscreenVersion, fftZoom, fftPanX, fftPanY, fftCropDims, width, height, drawFftInsetLayer]);

  // === Kymograph (space-time) ===
  // A sub-feature of the line profile (Henry: "the profile feature created a 2D
  // image ... distance along the line ... time axis"). Requires the profile tool
  // ON with a drawn line and some way to read every frame: the offline pack for
  // exported HTML, or the live frame server while the notebook kernel is up.
  const kymoExactStackReady = offline && !!offlineFloatStack && offlineFloatStack.byteLength > 0;
  const kymoQuantizedStackReady = offline && !!offlineStack && offlineStack.byteLength > 0;
  const kymoOfflineStackReady = kymoExactStackReady || kymoQuantizedStackReady;
  const kymoLiveStackReady = !offline && !!frameServerUrl;
  const kymographAvailable = (nPanels || 1) === 1
    && (kymoOfflineStackReady || kymoLiveStackReady)
    && width > 0 && height > 0 && nSlices > 1;
  const canKymograph = kymographAvailable && profileActive && profilePoints.length === 2;
  const kymoReady = canKymograph && showKymograph;

  // Compute the (nFrames, lineLen) image: sample the profile line on every
  // frame. Cold path - fires on line / width / stack change, never per tick.
  React.useEffect(() => {
    if (!kymoReady) { kymoDataRef.current = null; return; }
    const p0 = profilePoints[0], p1 = profilePoints[1];
    const pixelCount = width * height;
    const panelIdx = Math.max(0, Math.min(_nPanelsLocal - 1, profilePanelIdx));
    const colOffset = panelGlobalColOffset(panelIdx);
    const row0 = p0.row, col0 = p0.col + colOffset;
    const row1 = p1.row, col1 = p1.col + colOffset;
    let cancelled = false;

    const publish = (kymo: Float32Array, lineLen: number) => {
      if (cancelled) return;
      kymoDataRef.current = { data: kymo, lineLen, nFrames: nSlices };
      setKymoVersion(v => v + 1);
    };

    if (kymoExactStackReady && offlineFloatStack) {
      const sampleFrame = (frameIdx: number): Float32Array => {
        const frame = float32FrameFromDataView(offlineFloatStack, frameIdx, pixelCount, false);
        return frame
          ? sampleLineProfile(frame, width, height, row0, col0, row1, col1, profileWidth)
          : new Float32Array(0);
      };
      const first = sampleFrame(0);
      const lineLen = first.length;
      if (lineLen < 2) { kymoDataRef.current = null; return; }
      const kymo = new Float32Array(nSlices * lineLen);
      kymo.set(first.subarray(0, lineLen), 0);
      for (let f = 1; f < nSlices; f++) {
        kymo.set(sampleFrame(f).subarray(0, lineLen), f * lineLen);
      }
      publish(kymo, lineLen);
      return () => { cancelled = true; };
    }

    if (kymoQuantizedStackReady && offlineStack) {
      const scale = (offlineMax - offlineMin) / 255.0;
      // Read straight from the packed uint8 stack, dequantizing only the
      // bilinear corners per sample point. No whole-frame dequant.
      const u8 = new Uint8Array(offlineStack.buffer, offlineStack.byteOffset, offlineStack.byteLength);
      const sampleFrame = (frameIdx: number) =>
        sampleLineProfileU8(u8, frameIdx * pixelCount, width, height, scale, offlineMin,
          row0, col0, row1, col1, profileWidth);
      const first = sampleFrame(0);
      const lineLen = first.length;
      if (lineLen < 2) { kymoDataRef.current = null; return; }
      const kymo = new Float32Array(nSlices * lineLen);
      kymo.set(first.subarray(0, lineLen), 0);
      for (let f = 1; f < nSlices; f++) {
        kymo.set(sampleFrame(f).subarray(0, lineLen), f * lineLen);
      }
      publish(kymo, lineLen);
      return () => { cancelled = true; };
    }

    if (kymoLiveStackReady) {
      void (async () => {
        const firstFrame = await fetchFrameFromServer(0);
        if (cancelled || !firstFrame || firstFrame.length < pixelCount) {
          if (!cancelled) kymoDataRef.current = null;
          return;
        }
        const first = sampleLineProfile(firstFrame, width, height, row0, col0, row1, col1, profileWidth);
        const lineLen = first.length;
        if (lineLen < 2) {
          if (!cancelled) kymoDataRef.current = null;
          return;
        }
        const kymo = new Float32Array(nSlices * lineLen);
        kymo.set(first.subarray(0, lineLen), 0);
        for (let f = 1; f < nSlices; f++) {
          if (cancelled) return;
          const frame = await fetchFrameFromServer(f);
          if (!frame || frame.length < pixelCount) {
            if (!cancelled) kymoDataRef.current = null;
            return;
          }
          kymo.set(sampleLineProfile(frame, width, height, row0, col0, row1, col1, profileWidth).subarray(0, lineLen), f * lineLen);
          await new Promise<void>(resolve => setTimeout(resolve, 0));
        }
        publish(kymo, lineLen);
      })();
      return () => { cancelled = true; };
    }

    kymoDataRef.current = null;
    return undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kymoReady, kymoExactStackReady, kymoQuantizedStackReady, kymoLiveStackReady, offlineStack, offlineFloatStack, offlineMin, offlineMax, width, height, nSlices,
      profileWidth, profilePoints[0]?.row, profilePoints[0]?.col,
      profilePoints[1]?.row, profilePoints[1]?.col, profilePanelIdx, fetchFrameFromServer]);

  // Process kymograph data → histogram + colormap rendering (cheap, sync).
  // Mirrors the FFT pipeline: range → log scale → histogram/stats → slider
  // range → LUT → offscreen → draw with zoom/pan. Cold path, image is tiny.
  React.useEffect(() => {
    const kymo = kymoDataRef.current;
    if (!kymoReady || !kymo) return;
    const { data, lineLen, nFrames } = kymo;

    let displayMin: number, displayMax: number;
    if (kymoAuto) {
      ({ vmin: displayMin, vmax: displayMax } = percentileClip(data, percentileLow, percentileHigh));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(data));
    }

    const displayData = kymoLogScale ? applyLogScale(data) : data;
    if (kymoLogScale) {
      displayMin = Math.log1p(displayMin);
      displayMax = Math.log1p(displayMax);
    }

    setKymoHistogramData(displayData);
    setKymoDataRange({ min: displayMin, max: displayMax });
    setKymoStats(computeStats(displayData));

    const { vmin, vmax } = sliderRange(displayMin, displayMax, kymoVminPct, kymoVmaxPct);
    const lut = COLORMAPS[kymoColormap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(displayData, lineLen, nFrames, lut, vmin, vmax);
    if (!offscreen) return;

    kymoOffscreenRef.current = offscreen;

    if (kymoCanvasRef.current) {
      const ctx = kymoCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.imageSmoothingEnabled = lineLen < canvasW || nFrames < canvasH;
        ctx.clearRect(0, 0, canvasW, canvasH);
        ctx.save();
        ctx.translate(kymoPanX, kymoPanY);
        ctx.scale(kymoZoom, kymoZoom);
        ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);
        ctx.restore();
      }
    }
  }, [kymoReady, kymoVersion, kymoLogScale, kymoAuto, kymoVminPct, kymoVmaxPct, kymoColormap,
      percentileLow, percentileHigh, canvasW, canvasH]);

  // Redraw cached kymograph with zoom/pan (cheap - no recomputation)
  React.useEffect(() => {
    if (!kymoReady || !kymoCanvasRef.current || !kymoOffscreenRef.current) return;
    const canvas = kymoCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const offW = kymoOffscreenRef.current.width;
    const offH = kymoOffscreenRef.current.height;
    ctx.imageSmoothingEnabled = offW < canvasW || offH < canvasH;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(kymoPanX, kymoPanY);
    ctx.scale(kymoZoom, kymoZoom);
    ctx.drawImage(kymoOffscreenRef.current, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [kymoReady, kymoZoom, kymoPanX, kymoPanY, canvasW, canvasH]);

  // Render kymograph overlay (playhead + axis scale bars + colorbar + click
  // crosshair). Mirrors the FFT overlay structure; the playhead is the only
  // part that tracks the current frame. Never recomputes the image.
  React.useEffect(() => {
    const overlay = kymoOverlayRef.current;
    const kymo = kymoDataRef.current;
    if (!overlay || !kymoReady || !kymo) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    overlay.width = Math.round(canvasW * DPR);
    overlay.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Playhead row marker - tracks the current frame in zoomed/panned space.
    const y = kymoPanY + kymoZoom * (((liveSliceIdx + 0.5) / kymo.nFrames) * canvasH);
    ctx.save();
    ctx.scale(DPR, DPR);
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvasW, y);
    ctx.stroke();
    ctx.restore();

    // Distance scale bar along the bottom edge (distance axis, pixelUnit).
    if (pixelSize > 0) {
      drawScaleBarHiDPI(overlay, DPR, kymoZoom, pixelSize, pixelUnit || "px", kymo.lineLen);
    }

    // Time scale bar along the left edge (time axis, dimUnit). Vertical bar +
    // label so the operator can read the temporal extent of the kymograph.
    if (dimSampling > 0 && dimUnit) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const targetBarPx = 60;
      const barThickness = 5;
      const margin = 12;
      const scaleY = canvasH / kymo.nFrames;
      const effectiveZoom = kymoZoom * scaleY;
      const targetPhysical = (targetBarPx / effectiveZoom) * dimSampling;
      const nicePhysical = roundToNiceValue(targetPhysical);
      const barPx = (nicePhysical / dimSampling) * effectiveZoom;
      const barX = margin;
      const barY = margin;
      ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
      ctx.shadowBlur = 2;
      ctx.shadowOffsetX = 1;
      ctx.shadowOffsetY = 1;
      ctx.fillStyle = "white";
      ctx.fillRect(barX, barY, barThickness, barPx);
      ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const label = nicePhysical >= 1 ? `${nicePhysical} ${dimUnit}` : `${nicePhysical.toPrecision(2)} ${dimUnit}`;
      ctx.fillText(label, barX + barThickness + 4, barY + barPx / 2);
      ctx.restore();
    }

    // Colorbar when enabled (mirror FFT colorbar draw).
    if (kymoShowColorbar && kymoDataRange.min !== kymoDataRange.max) {
      const { vmin, vmax } = sliderRange(kymoDataRange.min, kymoDataRange.max, kymoVminPct, kymoVmaxPct);
      const lut = COLORMAPS[kymoColormap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      drawColorbar(ctx, overlay.width / DPR, overlay.height / DPR, lut, vmin, vmax, kymoLogScale);
      ctx.restore();
    }

    // Click crosshair marker - mirror FFT marker, coordinates in zoomed space.
    if (kymoClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const screenX = kymoPanX + kymoZoom * (kymoClickInfo.col / kymo.lineLen * canvasW);
      const screenY = kymoPanY + kymoZoom * (kymoClickInfo.row / kymo.nFrames * canvasH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
      ctx.shadowBlur = 2;
      ctx.lineWidth = 1.5;
      const r = 8;
      ctx.beginPath();
      ctx.moveTo(screenX - r, screenY); ctx.lineTo(screenX - 3, screenY);
      ctx.moveTo(screenX + 3, screenY); ctx.lineTo(screenX + r, screenY);
      ctx.moveTo(screenX, screenY - r); ctx.lineTo(screenX, screenY - 3);
      ctx.moveTo(screenX, screenY + 3); ctx.lineTo(screenX, screenY + r);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }, [kymoReady, kymoVersion, liveSliceIdx, canvasW, canvasH, themeColors.accent, kymoZoom, kymoPanX, kymoPanY,
      pixelSize, pixelUnit, dimSampling, dimUnit, kymoShowColorbar, kymoDataRange, kymoVminPct, kymoVmaxPct,
      kymoColormap, kymoLogScale, kymoClickInfo]);

  // Render FFT overlay (reciprocal-space scale bar + colorbar)
  React.useEffect(() => {
    const overlay = fftOverlayRef.current;
    if (!overlay || !effectiveShowFft) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    overlay.width = Math.round(canvasW * DPR);
    overlay.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Use crop dimensions for reciprocal-space calculations
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;

    // Reciprocal-space scale bar (pixelSize is in Å)
    if (pixelSize > 0) {
      const panelGrid = fftPanelGridRef.current;
      const reciprocalWidth = panelGrid ? panelGrid.panelWidth : fftW;
      const fftPixelSize = 1 / (reciprocalWidth * pixelSize);
      drawFFTScaleBarHiDPI(overlay, DPR, fftZoom, fftPixelSize, fftW, `${unitSymbol(pixelUnit || "px")}⁻¹`);
    }

    // FFT colorbar
    if (fftShowColorbar && fftDataRange.min !== fftDataRange.max) {
      const { vmin, vmax } = sliderRange(fftDataRange.min, fftDataRange.max, fftVminPct, fftVmaxPct);
      const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      const cssW = overlay.width / DPR;
      const cssH = overlay.height / DPR;
      drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, fftLogScale);
      ctx.restore();
    }

    // D-spacing crosshair marker - use crop dims for coordinate mapping
    if (fftClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      let screenX = fftPanX + fftZoom * (fftClickInfo.col / fftW * canvasW);
      let screenY = fftPanY + fftZoom * (fftClickInfo.row / fftH * canvasH);
      let centerX = fftPanX + fftZoom * (canvasW / 2);
      let centerY = fftPanY + fftZoom * (canvasH / 2);
      let radiusX = fftZoom * (fftClickInfo.distPx / Math.max(1, fftW)) * canvasW;
      let radiusY = fftZoom * (fftClickInfo.distPx / Math.max(1, fftH)) * canvasH;
      let clipRect: { x: number; y: number; w: number; h: number } | null = null;
      const panelGrid = fftPanelGridRef.current;
      if (panelGrid) {
        const slot = Math.max(0, Math.min(panelGrid.count - 1, Math.floor(fftClickInfo.row / panelGrid.panelHeight) * panelGrid.cols + Math.floor(fftClickInfo.col / panelGrid.panelWidth)));
        const dst = getFftSlot(slot, panelGrid.count, panelGrid.cols, panelGrid.rows);
        const localCol = fftClickInfo.col - (slot % panelGrid.cols) * panelGrid.panelWidth;
        const localRow = fftClickInfo.row - Math.floor(slot / panelGrid.cols) * panelGrid.panelHeight;
        screenX = dst.x + fftPanX + fftZoom * ((localCol / panelGrid.panelWidth) * dst.w);
        screenY = dst.y + fftPanY + fftZoom * ((localRow / panelGrid.panelHeight) * dst.h);
        centerX = dst.x + fftPanX + fftZoom * (dst.w / 2);
        centerY = dst.y + fftPanY + fftZoom * (dst.h / 2);
        radiusX = fftZoom * (fftClickInfo.distPx / Math.max(1, panelGrid.panelWidth)) * dst.w;
        radiusY = fftZoom * (fftClickInfo.distPx / Math.max(1, panelGrid.panelHeight)) * dst.h;
        clipRect = dst;
      }
      ctx.lineCap = "round";
      ctx.shadowBlur = 0;
      const r = 8;
      const drawRing = () => {
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
        ctx.stroke();
      };
      const drawMarker = () => {
        ctx.beginPath();
        ctx.moveTo(screenX - r, screenY); ctx.lineTo(screenX - 3, screenY);
        ctx.moveTo(screenX + 3, screenY); ctx.lineTo(screenX + r, screenY);
        ctx.moveTo(screenX, screenY - r); ctx.lineTo(screenX, screenY - 3);
        ctx.moveTo(screenX, screenY + 3); ctx.lineTo(screenX, screenY + r);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
        ctx.stroke();
      };
      if (clipRect) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(clipRect.x, clipRect.y, clipRect.w, clipRect.h);
        ctx.clip();
      }
      ctx.strokeStyle = "rgba(0, 0, 0, 0.78)";
      ctx.lineWidth = 4;
      drawRing();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.64)";
      ctx.lineWidth = 1.25;
      drawRing();
      if (clipRect) ctx.restore();
      ctx.strokeStyle = "rgba(0, 0, 0, 0.92)";
      ctx.lineWidth = 4;
      drawMarker();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.96)";
      ctx.lineWidth = 1.5;
      drawMarker();
      const label = fftClickInfo.dSpacing != null
        ? (() => {
          const d = fftClickInfo.dSpacing!;
          return d >= 10 ? `d = ${(d / 10).toFixed(2)} nm` : `d = ${d.toFixed(2)} Å`;
        })()
        : `dist = ${fftClickInfo.distPx.toFixed(1)} px`;
      ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const padX = 5;
      const labelW = Math.ceil(ctx.measureText(label).width + padX * 2);
      const labelH = 18;
      const cssW = overlay.width / DPR;
      const cssH = overlay.height / DPR;
      const labelX = Math.max(2, Math.min(cssW - labelW - 2, screenX + 10));
      const labelY = Math.max(labelH / 2 + 2, Math.min(cssH - labelH / 2 - 2, screenY - 10));
      ctx.fillStyle = "rgba(0, 0, 0, 0.74)";
      ctx.strokeStyle = "rgba(255, 255, 255, 0.82)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(labelX, labelY - labelH / 2, labelW, labelH, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "white";
      ctx.fillText(label, labelX + padX, labelY);
      ctx.restore();
    }
  }, [effectiveShowFft, fftZoom, fftPanX, fftPanY, canvasW, canvasH, pixelSize, width, height, fftDataRange, fftVminPct, fftVmaxPct, fftColormap, fftLogScale, fftShowColorbar, fftClickInfo, fftCropDims, getFftSlot]);

  // -------------------------------------------------------------------------
  // Preview panel - cache colormapped offscreen (only recomputes when ROI
  // geometry, data, or display settings change - NOT on zoom/pan)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!previewVisible || !rawFrameDataRef.current) {
      previewOffscreenRef.current = null;
      return;
    }

    const raw = rawFrameDataRef.current;
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return;

    const roi = roiList[roiSelectedIdx];
    const crop = cropROIRegion(raw, width, height, roi);
    if (!crop) {
      previewOffscreenRef.current = null;
      setPreviewCropDims(null);
      setPreviewVersion(v => v + 1);
      return;
    }

    setPreviewCropDims({ w: crop.cropW, h: crop.cropH });

    const processed = logScale ? applyLogScale(crop.cropped) : crop.cropped;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    let vmin: number, vmax: number;
    const nP = Math.max(1, nPanels || 1);
    const hasTraitRange = traitVmin != null || traitVmax != null;
    const perPanelContrast = nP > 1 && !linkContrast && !sharedPanelSource && width % nP === 0 && height > 0;
    if (hasTraitRange) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (autoContrast) {
      const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
      const mainProcessed = logScale ? applyLogScale(raw) : raw;
      ({ vmin, vmax } = cached ?? percentileClip(mainProcessed, percentileLow, percentileHigh));
    } else if (perPanelContrast) {
      const panelW = width / nP;
      const panel = Math.max(0, Math.min(nP - 1, Math.floor((Number(roi.col) || 0) / panelW)));
      const panelData = extractPanelSlice(raw, panel, logScale);
      const pdr = panelDataRanges[panel];
      const panelRange = (perPanelHistogramEnabled && pdr && pdr.max > pdr.min)
        ? pdr
        : (panelData && panelData.length > 0
            ? findDataRange(panelData)
            : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale));
      const resolved = resolvePanelRange(panel, panelRange, null);
      vmin = resolved.vmin;
      vmax = resolved.vmax;
    } else {
      const lo = logScale ? (dataMin >= 0 ? Math.log1p(dataMin) : -Math.log1p(-dataMin)) : dataMin;
      const hi = logScale ? (dataMax >= 0 ? Math.log1p(dataMax) : -Math.log1p(-dataMax)) : dataMax;
      ({ vmin, vmax } = sliderRange(lo, hi, imageVminPct, imageVmaxPct));
    }

    const offscreen = renderToOffscreen(processed, crop.cropW, crop.cropH, lut, vmin, vmax);
    previewOffscreenRef.current = offscreen;
    setPreviewVersion(v => v + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewVisible, selectedRoiKey, cmap, logScale, autoContrast, imageVminPct, imageVmaxPct, dataMin, dataMax, traitVmin, traitVmax, percentileLow, percentileHigh, width, height, frameBytes, displaySliceIdx, autoVmins, autoVmaxs, nPanels, linkContrast, sharedPanelSource, panelStates, vminPerPanel, vmaxPerPanel]);

  // -------------------------------------------------------------------------
  // Preview panel - compute aspect-ratio-aware canvas dimensions
  // -------------------------------------------------------------------------
  const previewCanvasDims = (() => {
    if (!previewCropDims) return { w: canvasW, h: canvasH };
    const { w: cropW, h: cropH } = previewCropDims;
    const aspect = cropW / cropH;
    if (aspect >= 1) {
      return { w: canvasW, h: Math.max(20, Math.round(canvasW / aspect)) };
    } else {
      return { w: Math.max(20, Math.round(canvasH * aspect)), h: canvasH };
    }
  })();

  // -------------------------------------------------------------------------
  // Preview panel - draw cached offscreen with zoom/pan (fast, no recompute)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = previewCanvasRef.current;
    if (!canvas || !previewVisible) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const pw = previewCanvasDims.w;
    const ph = previewCanvasDims.h;
    const offscreen = previewOffscreenRef.current;
    if (!offscreen || !previewCropDims) {
      ctx.clearRect(0, 0, pw, ph);
      return;
    }

    ctx.imageSmoothingEnabled = smooth;
    ctx.clearRect(0, 0, pw, ph);

    const { zoom: pz, panX: ppX, panY: ppY } = previewZoom;
    if (pz !== 1 || ppX !== 0 || ppY !== 0) {
      ctx.save();
      const cx = pw / 2;
      const cy = ph / 2;
      ctx.translate(cx + ppX, cy + ppY);
      ctx.scale(pz, pz);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, previewCropDims.w, previewCropDims.h, 0, 0, pw, ph);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, previewCropDims.w, previewCropDims.h, 0, 0, pw, ph);
    }
  }, [previewVisible, previewVersion, previewZoom, previewCanvasDims, previewCropDims]);

  // Preview overlay - scale bar + zoom indicator
  React.useEffect(() => {
    const overlay = previewOverlayRef.current;
    if (!overlay || !previewVisible) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (previewCropDims && pixelSize > 0) {
      const unit = "Å" as const;
      drawScaleBarHiDPI(overlay, DPR, previewZoom.zoom, pixelSize, unit, previewCropDims.w);
    }
  }, [previewVisible, previewZoom, previewCropDims, previewCanvasDims, pixelSize]);

  // Mouse handlers
  const panelIdxFromXY = (cssX: number, cssY: number): number => {
    const { n, cols, rows, gap, slotW, slotH } = getPanelLayout();
    if (n === 1) {
      return cssX >= 0 && cssX <= canvasW && cssY >= 0 && cssY <= canvasH
        ? (visiblePanelIndices[0] ?? 0)
        : -1;
    }
    const col = Math.floor(cssX / Math.max(1, slotW + gap));
    const row = Math.floor(cssY / Math.max(1, slotH + gap));
    if (col < 0 || col >= cols || row < 0 || row >= rows) return -1;
    const localX = cssX - col * (slotW + gap);
    const localY = cssY - row * (slotH + gap);
    if (localX < 0 || localX > slotW || localY < 0 || localY > slotH) return -1;
    const idx = row * cols + col;
    // Empty grid cells past N panels (partial last row) are not panels.
    return idx >= n ? -1 : (visiblePanelIndices[idx] ?? -1);
  };
  const panelIdxFromEvent = (e: React.MouseEvent): number => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    const cssX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const cssY = (e.clientY - rect.top) * (canvas.height / rect.height);
    return panelIdxFromXY(cssX, cssY);
  };
  const canvasPointFromClient = (clientX: number, clientY: number): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  };
  const panelIdxFromClient = (clientX: number, clientY: number): number => {
    const pt = canvasPointFromClient(clientX, clientY);
    return pt ? panelIdxFromXY(pt.x, pt.y) : -1;
  };
  const beginPan = (e: React.MouseEvent) => {
    const idx = panelIdxFromEvent(e);
    if (idx < 0) return;
    panStartPanelRef.current = idx;
    const live = playRef.current;
    const base = live.panelStates[idx] || stateFor(idx);
    const s = {
      ...base,
      zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
      panX: live.linkPanels ? live.linkedState.panX : base.panX,
      panY: live.linkPanels ? live.linkedState.panY : base.panY,
    };
    setIsDraggingPan(true);
    setPanStart({ x: e.clientX, y: e.clientY, pX: s.panX, pY: s.panY });
  };
  const applyCanvasWheelZoom = (clientX: number, clientY: number, deltaY: number): boolean => {
    const canvas = canvasRef.current;
    if (!canvas) return false;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    const mouseX = (clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (clientY - rect.top) * (canvas.height / rect.height);
    const panelIdx = panelIdxFromXY(mouseX, mouseY);
    if (panelIdx < 0) return false;
    const live = playRef.current;
    const base = live.panelStates[panelIdx] || stateFor(panelIdx);
    const cur = {
      ...base,
      zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
      panX: live.linkPanels ? live.linkedState.panX : base.panX,
      panY: live.linkPanels ? live.linkedState.panY : base.panY,
    };
    const zoomFactor = Math.max(0.75, Math.min(1.35, Math.exp(-deltaY * 0.002)));
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, cur.zoom * zoomFactor));
    const zoomRatio = newZoom / cur.zoom;
    // Mouse position relative to this panel's slot (so zoom anchors to cursor within slot).
    const geom = getPanelGeometry(panelIdx);
    if (!geom) return false;
    const localX = mouseX - geom.slotX;
    const localY = mouseY - geom.slotY;
    const newPanX = localX - (localX - cur.panX) * zoomRatio;
    const newPanY = localY - (localY - cur.panY) * zoomRatio;
    syncPlaybackPanelTransform(panelIdx, newZoom, newPanX, newPanY);
    transformInputAtRef.current = performance.now();
    if (scheduleTransformRender()) {
      scheduleTransformStateCommit();
    } else {
      commitLivePanelTransforms();
    }
    const dbg = show3dPerfDebug();
    if (dbg) {
      dbg.lastWheelZoom = {
        panelIdx,
        zoom: Number(newZoom.toFixed(3)),
        panX: Number(newPanX.toFixed(1)),
        panY: Number(newPanY.toFixed(1)),
        deltaY: Number(deltaY.toFixed(3)),
      };
    }
    return true;
  };

  canvasWheelHandlerRef.current = (event: WheelEvent) => {
    if (fftInsetNativeWheelHandlerRef.current?.(event)) return;
    event.preventDefault();
    event.stopPropagation();
    if (reorderMode) return;
    applyCanvasWheelZoom(event.clientX, event.clientY, event.deltaY);
  };

  React.useEffect(() => {
    const el = canvasContainerRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => canvasWheelHandlerRef.current?.(event);
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [canvasW, canvasH]);

  React.useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const onFftInsetWheelCapture = (event: WheelEvent) => {
      fftInsetNativeWheelHandlerRef.current?.(event);
    };
    root.addEventListener("wheel", onFftInsetWheelCapture, { capture: true, passive: false });
    return () => root.removeEventListener("wheel", onFftInsetWheelCapture, { capture: true });
  }, []);

  const handleDoubleClick = () => {
    const resetPanels = Array.from({ length: Math.max(1, nPanels || 1) }, (_, i) => ({
      ...(playRef.current.panelStates[i] || initialState),
      zoom: 1,
      panX: 0,
      panY: 0,
    }));
    const resetLinked = { ...playRef.current.linkedState, zoom: 1, panX: 0, panY: 0 };
    playRef.current.linkedState = resetLinked;
    playRef.current.panelStates = resetPanels;
    linkedStateLiveRef.current = resetLinked;
    panelStatesLiveRef.current = resetPanels;
    setLinkedState(s => ({ ...s, zoom: 1, panX: 0, panY: 0 }));
    setPanelStates(arr => arr.map(s => ({ ...s, zoom: 1, panX: 0, panY: 0 })));
    setViewState({ linked_state: { ...resetLinked }, panel_states: resetPanels.map(v => ({ ...v })) });
    scheduleTransformRender();
  };

  const addROIAt = (row: number, col: number, shape: "circle" | "square" | "rectangle" | "annular" = newRoiShape) => {
    const clampedRow = Math.max(0, Math.min(height - 1, Math.round(row)));
    const clampedCol = Math.max(0, Math.min(width - 1, Math.round(col)));
    const next = [...roiItems, createROI(clampedRow, clampedCol, shape, roiItems.length, width, height)];
    setRoiList(next);
    setRoiSelectedIdx(next.length - 1);
    setShowRoiResizeHint(true);
  };

  const deleteSelectedROI = () => {
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return;
    const next = roiList.filter((_, i) => i !== roiSelectedIdx);
    setRoiList(next);
    setRoiSelectedIdx(next.length > 0 ? Math.min(roiSelectedIdx, next.length - 1) : -1);
  };

  const duplicateSelectedROI = () => {
    if (!selectedRoi) return;
    const duplicated: ROIItem = {
      ...selectedRoi,
      row: Math.max(0, Math.min(height - 1, Math.round(selectedRoi.row + 3))),
      col: Math.max(0, Math.min(width - 1, Math.round(selectedRoi.col + 3))),
      shape: selectedRoi.shape,
      radius: selectedRoi.radius,
      radius_inner: selectedRoi.radius_inner,
      width: selectedRoi.width,
      height: selectedRoi.height,
      color: ROI_COLORS[roiItems.length % ROI_COLORS.length],
      line_width: selectedRoi.line_width,
      highlight: false,
    };
    const next = [...roiItems, duplicated];
    setRoiList(next);
    setRoiSelectedIdx(next.length - 1);
  };


  const handleCopy = async () => {
    if (!canvasRef.current) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvasRef.current!.toBlob(resolve, "image/png"));
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch (err) {
      console.warn("Show3D copy failed", err);
    }
  };

  const handleHandoffToShow2D = React.useCallback(() => {
    setViewMenuAnchor(null);
    setHandoffRequest(JSON.stringify({
      mode: "show2d",
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      frame: displaySliceIdx,
      panel: visiblePanelIndices,
    }));
  }, [displaySliceIdx, visiblePanelIndices, setHandoffRequest]);

  const handleClosePreparedView = React.useCallback(() => {
    setHandoffRequest(JSON.stringify({
      mode: "clear",
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    }));
  }, [setHandoffRequest]);

  const clickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const touchTransformRef = React.useRef<TouchTransformState | null>(null);
  const fftTouchTransformRef = React.useRef<FftTouchTransformState | null>(null);
  const kymoTouchTransformRef = React.useRef<FftTouchTransformState | null>(null);
  const lastTapRef = React.useRef<{ time: number; panelIdx: number } | null>(null);
  const lastFftTapRef = React.useRef<{ time: number } | null>(null);
  const lastKymoTapRef = React.useRef<{ time: number } | null>(null);
  const [draggingProfileEndpoint, setDraggingProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingProfileLine, setIsDraggingProfileLine] = React.useState(false);
  const [hoveredProfileEndpoint, setHoveredProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringProfileLine, setIsHoveringProfileLine] = React.useState(false);
  const profileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);

  const screenToImg = (e: React.MouseEvent): { imgCol: number; imgRow: number; panelIdx: number; panelCol: number } => {
    const pt = canvasPointFromEvent(e);
    if (!pt) return { imgCol: 0, imgRow: 0, panelIdx: -1, panelCol: 0 };
    const panelIdx = panelIdxFromXY(pt.x, pt.y);
    const geom = getPanelGeometry(panelIdx);
    if (!geom) return { imgCol: 0, imgRow: 0, panelIdx: -1, panelCol: 0 };
    // Undo slot offset, pan, zoom, then panel source scaling.
    let localCol = (pt.x - geom.slotX - geom.state.panX) / (geom.scaleX * geom.state.zoom);
    let row = (pt.y - geom.slotY - geom.state.panY) / (geom.scaleY * geom.state.zoom);
    // Undo image_rotation in panel-local source coordinates.
    const r = (((imageRotation % 4) + 4) % 4) | 0;
    if (r !== 0) {
      const rotSwap = (r % 2) !== 0;
      const visW = rotSwap ? sourcePanelHeight : sourcePanelWidth;
      const visH = rotSwap ? sourcePanelWidth : sourcePanelHeight;
      const cx = localCol - visW / 2;
      const cy = row - visH / 2;
      let ux: number, uy: number;
      if (r === 1) { ux = cy; uy = -cx; }
      else if (r === 2) { ux = -cx; uy = -cy; }
      else { ux = -cy; uy = cx; }
      localCol = ux + sourcePanelWidth / 2;
      row = uy + sourcePanelHeight / 2;
    }
    return { imgCol: panelGlobalCol(localCol, panelIdx), imgRow: row, panelIdx, panelCol: localCol };
  };

  const hitTestROI = (imgCol: number, imgRow: number): number => {
    if (!effectiveRoiActive || roiItems.length === 0) return -1;
    for (let roiIdx = roiItems.length - 1; roiIdx >= 0; roiIdx--) {
      const roi = roiItems[roiIdx];
      const shape = roi.shape || "circle";
      if (shape === "circle" || shape === "annular") {
        if (Math.sqrt((imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2) <= roi.radius) return roiIdx;
      } else if (shape === "square") {
        if (Math.abs(imgCol - roi.col) <= roi.radius && Math.abs(imgRow - roi.row) <= roi.radius) return roiIdx;
      } else if (shape === "rectangle") {
        if (Math.abs(imgCol - roi.col) <= roi.width / 2 && Math.abs(imgRow - roi.row) <= roi.height / 2) return roiIdx;
      }
    }
    return -1;
  };

  const getHitArea = () => RESIZE_HIT_AREA_PX / (displayScale * zoom);

  const isNearEdge = (imgCol: number, imgRow: number, roi: ROIItem): boolean => {
    const hitArea = getHitArea();
    const shape = roi.shape || "circle";
    if (shape === "circle" || shape === "annular") {
      const dist = Math.sqrt((imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2);
      return Math.abs(dist - roi.radius) < hitArea;
    }
    if (shape === "square") {
      const dx = Math.abs(imgCol - roi.col);
      const dy = Math.abs(imgRow - roi.row);
      const r = roi.radius;
      return (dx <= r + hitArea && dy <= r + hitArea) && (Math.abs(dx - r) < hitArea || Math.abs(dy - r) < hitArea);
    }
    if (shape === "rectangle") {
      const dx = Math.abs(imgCol - roi.col);
      const dy = Math.abs(imgRow - roi.row);
      const hw = roi.width / 2;
      const hh = roi.height / 2;
      return (dx <= hw + hitArea && dy <= hh + hitArea) && (Math.abs(dx - hw) < hitArea || Math.abs(dy - hh) < hitArea);
    }
    return false;
  };

  const isNearResizeHandle = (imgCol: number, imgRow: number): boolean => {
    if (!effectiveRoiActive || !selectedRoi) return false;
    return isNearEdge(imgCol, imgRow, selectedRoi);
  };

  const isNearAnyEdge = (imgCol: number, imgRow: number): boolean => {
    if (!effectiveRoiActive || roiItems.length === 0) return false;
    return roiItems.some(roi => isNearEdge(imgCol, imgRow, roi));
  };

  const isNearResizeHandleInner = (imgCol: number, imgRow: number): boolean => {
    if (!effectiveRoiActive || !selectedRoi || selectedRoi.shape !== "annular") return false;
    const hitArea = getHitArea();
    const dist = Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
    return Math.abs(dist - selectedRoi.radius_inner) < hitArea;
  };

  const updateROI = (e: React.MouseEvent) => {
    if (!selectedRoi) return;
    const { imgCol, imgRow } = screenToImg(e);
    updateSelectedRoi({
      col: Math.max(0, Math.min(width - 1, Math.floor(imgCol))),
      row: Math.max(0, Math.min(height - 1, Math.floor(imgRow))),
    });
  };

  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    // Ignore clicks in empty grid cells (partial last row when N isn't a
    // multiple of max_cols). Otherwise the click attributes to the last
    // real panel and zoom/pan jumps unexpectedly.
    if (panelIdxFromEvent(e) < 0) return;
    clickStartRef.current = { x: e.clientX, y: e.clientY };
    pendingRoiAddRef.current = null;
    // Check if clicking on lens inset for drag or resize
    if (showLens) {
      const rect = canvasContainerRef.current?.getBoundingClientRect();
      if (rect) {
        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        if (cssX >= lx && cssX <= lx + lensDisplaySize && cssY >= ly && cssY <= ly + lensDisplaySize) {
          const edgeHit = 8;
          const nearEdge = cssX - lx < edgeHit || lx + lensDisplaySize - cssX < edgeHit ||
                           cssY - ly < edgeHit || ly + lensDisplaySize - cssY < edgeHit;
          if (nearEdge) {
            setIsResizingLens(true);
            lensResizeStartRef.current = { my: e.clientY, startSize: lensDisplaySize };
          } else {
            setIsDraggingLens(true);
            lensDragStartRef.current = { mx: e.clientX, my: e.clientY, ax: lx, ay: ly };
          }
          return;
        }
      }
    }
    if (profileActive) {
      const { imgCol, imgRow, panelIdx } = screenToImg(e);
      if (profilePoints.length === 2) {
        if (panelIdx !== profilePanelIdx) {
          beginPan(e);
          return;
        }
        const p0 = profilePoints[0];
        const p1 = profilePoints[1];
        const hitRadius = getImageHitRadius(profilePanelIdx);
        const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
        const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingPan(false);
          setPanStart(null);
          return;
        }
        if (pointToSegmentDistance(imgCol, imgRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
          setIsDraggingProfileLine(true);
          profileDragStartRef.current = {
            row: imgRow,
            col: imgCol,
            p0: { row: p0.row, col: p0.col },
            p1: { row: p1.row, col: p1.col },
          };
          setIsDraggingPan(false);
          setPanStart(null);
          return;
        }
      }
      beginPan(e);
      return;
    }
    if (effectiveRoiActive) {
      const { imgCol, imgRow } = screenToImg(e);
      if (isNearResizeHandleInner(imgCol, imgRow)) {
        setIsDraggingResizeInner(true);
        return;
      }
      if (isNearResizeHandle(imgCol, imgRow)) {
        e.preventDefault();
        resizeAspectRef.current = selectedRoi && (selectedRoi.shape === "rectangle") && selectedRoi.width > 0 && selectedRoi.height > 0 ? selectedRoi.width / selectedRoi.height : null;
        setIsDraggingResize(true);
        return;
      }
      if (roiItems.length > 0) {
        for (let roiIdx = roiItems.length - 1; roiIdx >= 0; roiIdx--) {
          const roi = roiItems[roiIdx];
          if (isNearEdge(imgCol, imgRow, roi)) {
            e.preventDefault();
            resizeAspectRef.current = roi && (roi.shape === "rectangle") && roi.width > 0 && roi.height > 0 ? roi.width / roi.height : null;
            setRoiSelectedIdx(roiIdx);
            setIsDraggingResize(true);
            return;
          }
        }
      }
      const hitIdx = hitTestROI(imgCol, imgRow);
      if (hitIdx >= 0) {
        setRoiSelectedIdx(hitIdx);
        setIsDraggingROI(true);
        return;
      }
      setRoiSelectedIdx(-1);
      pendingRoiAddRef.current = {
        row: Math.max(0, Math.min(height - 1, Math.round(imgRow))),
        col: Math.max(0, Math.min(width - 1, Math.round(imgCol))),
      };
      return;
    }
    beginPan(e);
  };

  type TouchPoint = { clientX: number; clientY: number };
  const touchDistance = (a: TouchPoint, b: TouchPoint) => Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
  const touchMidpoint = (a: TouchPoint, b: TouchPoint) => ({ x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 });

  const handleCanvasTouchStart = (e: React.TouchEvent) => {
    if (profileActive || effectiveRoiActive) return;
    if (e.touches.length === 1) {
      const t = e.touches[0];
      const panelIdx = panelIdxFromClient(t.clientX, t.clientY);
      if (panelIdx < 0) return;
      const now = Date.now();
      const lastTap = lastTapRef.current;
      if (lastTap && lastTap.panelIdx === panelIdx && now - lastTap.time < 320) {
        e.preventDefault();
        handleDoubleClick();
        lastTapRef.current = null;
        touchTransformRef.current = null;
        return;
      }
      lastTapRef.current = { time: now, panelIdx };
      if (showLens) return;
      const live = playRef.current;
      const base = live.panelStates[panelIdx] || stateFor(panelIdx);
      touchTransformRef.current = {
        panelIdx,
        mode: "pan",
        startX: t.clientX,
        startY: t.clientY,
        startDistance: 0,
        startMidX: t.clientX,
        startMidY: t.clientY,
        startState: {
          ...base,
          zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
          panX: live.linkPanels ? live.linkedState.panX : base.panX,
          panY: live.linkPanels ? live.linkedState.panY : base.panY,
        },
      };
      e.preventDefault();
      return;
    }
    if (e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      const panelIdx = panelIdxFromClient(mid.x, mid.y);
      if (panelIdx < 0) return;
      const live = playRef.current;
      const base = live.panelStates[panelIdx] || stateFor(panelIdx);
      touchTransformRef.current = {
        panelIdx,
        mode: "pinch",
        startX: mid.x,
        startY: mid.y,
        startDistance: Math.max(1, touchDistance(a, b)),
        startMidX: mid.x,
        startMidY: mid.y,
        startState: {
          ...base,
          zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
          panX: live.linkPanels ? live.linkedState.panX : base.panX,
          panY: live.linkPanels ? live.linkedState.panY : base.panY,
        },
      };
      e.preventDefault();
    }
  };

  const handleCanvasTouchMove = (e: React.TouchEvent) => {
    const start = touchTransformRef.current;
    if (!start) return;
    const canvas = canvasRef.current;
    const geom = getPanelGeometry(start.panelIdx);
    if (!canvas || !geom) return;
    e.preventDefault();
    const base = start.startState;
    if (start.mode === "pinch" && e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      const startPoint = canvasPointFromClient(start.startMidX, start.startMidY);
      const currentPoint = canvasPointFromClient(mid.x, mid.y);
      if (!startPoint || !currentPoint) return;
      const startLocalX = startPoint.x - geom.slotX;
      const startLocalY = startPoint.y - geom.slotY;
      const currentLocalX = currentPoint.x - geom.slotX;
      const currentLocalY = currentPoint.y - geom.slotY;
      const imageX = (startLocalX - base.panX) / base.zoom;
      const imageY = (startLocalY - base.panY) / base.zoom;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base.zoom * (touchDistance(a, b) / start.startDistance)));
      syncPlaybackPanelTransform(start.panelIdx, newZoom, currentLocalX - imageX * newZoom, currentLocalY - imageY * newZoom);
    } else if (start.mode === "pan" && e.touches.length === 1) {
      const t = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / Math.max(1, rect.width);
      const scaleY = canvas.height / Math.max(1, rect.height);
      syncPlaybackPanelTransform(
        start.panelIdx,
        base.zoom,
        base.panX + (t.clientX - start.startX) * scaleX,
        base.panY + (t.clientY - start.startY) * scaleY,
      );
    }
    transformInputAtRef.current = performance.now();
    if (scheduleTransformRender()) scheduleTransformStateCommit();
    else commitLivePanelTransforms();
  };

  const handleCanvasTouchEnd = (e: React.TouchEvent) => {
    if (e.touches.length > 0 || !touchTransformRef.current) return;
    commitLivePanelTransforms();
    touchTransformRef.current = null;
  };

  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    // Fast path: during pan drag, skip all cursor/hover/lens work - just update pan
    if (isDraggingPan && panStart) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - panStart.x) * scaleX;
      const dy = (e.clientY - panStart.y) * scaleY;
      const newPanX = panStart.pX + dx;
      const newPanY = panStart.pY + dy;
      const live = playRef.current;
      const base = live.panelStates[panStartPanelRef.current] || stateFor(panStartPanelRef.current);
      const current = {
        ...base,
        zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
        panX: live.linkPanels ? live.linkedState.panX : base.panX,
        panY: live.linkPanels ? live.linkedState.panY : base.panY,
      };
      syncPlaybackPanelTransform(panStartPanelRef.current, current.zoom, newPanX, newPanY);
      transformInputAtRef.current = performance.now();
      if (scheduleTransformRender()) scheduleTransformStateCommit();
      else commitLivePanelTransforms();
      return;
    }

    // Cursor readout: convert screen position to image pixel coordinates.
    // Skip when hovering an empty grid cell (partial last row when nPanels
    // isn't a multiple of max_cols) so dead space doesn't flash row/col
    // numbers from a phantom panel.
    const canvas = canvasRef.current;
    const hoverPanelIdx = panelIdxFromEvent(e);
    if (hoverPanelIdx < 0) {
      scheduleCursorInfo(null);
      if (showLens) setLensPos(null);
    } else if (canvas && rawFrameDataRef.current) {
      const { imgRow, imgCol, panelIdx, panelCol } = screenToImg(e);
      const pixelDataCol = Math.floor(imgCol);
      const pixelPanelCol = Math.floor(panelCol);
      const pixelRow = Math.floor(imgRow);
      if (
        pixelDataCol >= 0 && pixelDataCol < width &&
        pixelPanelCol >= 0 && pixelPanelCol < sourcePanelWidth &&
        pixelRow >= 0 && pixelRow < height
      ) {
        const rawData = rawFrameDataRef.current;
        scheduleCursorInfo({
          row: pixelRow,
          col: pixelPanelCol,
          value: rawData[pixelRow * width + pixelDataCol],
          panelIdx,
        });
        if (showLens) setLensPos({ row: pixelRow, col: pixelDataCol });
      } else {
        scheduleCursorInfo(null);
        if (showLens) setLensPos(null);
      }
    }

    // Lens edge hover detection
    if (showLens) {
      const rect2 = canvasContainerRef.current?.getBoundingClientRect();
      if (rect2) {
        const cssX2 = e.clientX - rect2.left;
        const cssY2 = e.clientY - rect2.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        const inside = cssX2 >= lx && cssX2 <= lx + lensDisplaySize && cssY2 >= ly && cssY2 <= ly + lensDisplaySize;
        const edgeHit = 8;
        const nearEdge = inside && (cssX2 - lx < edgeHit || lx + lensDisplaySize - cssX2 < edgeHit ||
                                     cssY2 - ly < edgeHit || ly + lensDisplaySize - cssY2 < edgeHit);
        setIsHoveringLensEdge(nearEdge);
      }
    } else {
      setIsHoveringLensEdge(false);
    }

    // Lens drag
    if (isDraggingLens && lensDragStartRef.current) {
      const dx = e.clientX - lensDragStartRef.current.mx;
      const dy = e.clientY - lensDragStartRef.current.my;
      setLensAnchor({ x: lensDragStartRef.current.ax + dx, y: lensDragStartRef.current.ay + dy });
      return;
    }

    // Lens resize drag
    if (isResizingLens && lensResizeStartRef.current) {
      const dy = e.clientY - lensResizeStartRef.current.my;
      setLensDisplaySize(Math.max(64, Math.min(256, lensResizeStartRef.current.startSize + dy)));
      return;
    }

    if (profileActive && profilePoints.length === 2) {
      const { imgCol, imgRow, panelIdx } = screenToImg(e);
      const p0 = profilePoints[0];
      const p1 = profilePoints[1];
      const hitRadius = getImageHitRadius(profilePanelIdx);
      const sameProfilePanel = panelIdx === profilePanelIdx;
      const d0 = sameProfilePanel ? Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2) : Infinity;
      const d1 = sameProfilePanel ? Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2) : Infinity;
      if (draggingProfileEndpoint !== null) {
        if (!rawFrameDataRef.current || panelIdx !== profilePanelIdx) return;
        const clampedRow = Math.max(0, Math.min(height - 1, imgRow));
        const clampedCol = Math.max(0, Math.min(width - 1, imgCol));
        const next = [
          draggingProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
          draggingProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
        ];
        setProfileLine(next);
        setProfileData(sampleLineProfile(rawFrameDataRef.current, width, height, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth));
        return;
      }
      if (isDraggingProfileLine && profileDragStartRef.current) {
        if (!rawFrameDataRef.current || panelIdx !== profilePanelIdx) return;
        const drag = profileDragStartRef.current;
        let deltaRow = imgRow - drag.row;
        let deltaCol = imgCol - drag.col;
        const minRow = Math.min(drag.p0.row, drag.p1.row);
        const maxRow = Math.max(drag.p0.row, drag.p1.row);
        const minCol = Math.min(drag.p0.col, drag.p1.col);
        const maxCol = Math.max(drag.p0.col, drag.p1.col);
        deltaRow = Math.max(deltaRow, -minRow);
        deltaRow = Math.min(deltaRow, (height - 1) - maxRow);
        deltaCol = Math.max(deltaCol, -minCol);
        deltaCol = Math.min(deltaCol, (width - 1) - maxCol);
        const next = [
          { row: drag.p0.row + deltaRow, col: drag.p0.col + deltaCol },
          { row: drag.p1.row + deltaRow, col: drag.p1.col + deltaCol },
        ];
        setProfileLine(next);
        setProfileData(sampleLineProfile(rawFrameDataRef.current, width, height, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth));
        return;
      }
      const nextHoveredEndpoint: 0 | 1 | null = d0 <= hitRadius ? 0 : d1 <= hitRadius ? 1 : null;
      const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(imgCol, imgRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius;
      setHoveredProfileEndpoint(nextHoveredEndpoint);
      setIsHoveringProfileLine(nextHoverLine);
    } else {
      if (hoveredProfileEndpoint !== null) setHoveredProfileEndpoint(null);
      if (isHoveringProfileLine) setIsHoveringProfileLine(false);
    }

    // Resize handle dragging
    if (isDraggingResizeInner && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const newR = Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
      updateSelectedRoi({ radius_inner: Math.max(1, Math.min(selectedRoi.radius - 1, Math.round(newR))) });
      setShowRoiResizeHint(false);
      return;
    }
    if (isDraggingResize && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const shape = selectedRoi.shape || "circle";
      if (shape === "rectangle") {
        let newW = Math.max(2, Math.round(Math.abs(ic - selectedRoi.col) * 2));
        let newH = Math.max(2, Math.round(Math.abs(ir - selectedRoi.row) * 2));
        if (e.shiftKey && resizeAspectRef.current != null) {
          const aspect = resizeAspectRef.current;
          if (newW / newH > aspect) newH = Math.max(2, Math.round(newW / aspect));
          else newW = Math.max(2, Math.round(newH * aspect));
        }
        updateSelectedRoi({ width: newW, height: newH });
      } else {
        const newR = shape === "square"
          ? Math.max(Math.abs(ic - selectedRoi.col), Math.abs(ir - selectedRoi.row))
          : Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
        const minR = shape === "annular" ? selectedRoi.radius_inner + 1 : 1;
        updateSelectedRoi({ radius: Math.max(minR, Math.round(newR)) });
      }
      setShowRoiResizeHint(false);
      return;
    }

    // Hover state for resize handles
    if (effectiveRoiActive && !isDraggingROI && !isDraggingPan) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const hoveringInner = isNearResizeHandleInner(ic, ir);
      const hoveringOuter = isNearAnyEdge(ic, ir);
      setIsHoveringResizeInner(hoveringInner);
      setIsHoveringResize(hoveringOuter);
      if (hoveringInner || hoveringOuter) setShowRoiResizeHint(false);
    }

    if (isDraggingROI) {
      updateROI(e);
    }
  };

  const handleCanvasMouseUp = (e: React.MouseEvent) => {
    if (draggingProfileEndpoint !== null || isDraggingProfileLine) {
      setDraggingProfileEndpoint(null);
      setIsDraggingProfileLine(false);
      profileDragStartRef.current = null;
      clickStartRef.current = null;
      pendingRoiAddRef.current = null;
      setIsDraggingROI(false);
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      setIsDraggingLens(false);
      lensDragStartRef.current = null;
      setIsResizingLens(false);
      lensResizeStartRef.current = null;
      setIsDraggingPan(false);
      setPanStart(null);
      setHoveredProfileEndpoint(null);
      setIsHoveringProfileLine(false);
      return;
    }

    // Profile click capture
    if (profileActive && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        if (rawFrameDataRef.current) {
          const { imgCol, imgRow, panelIdx } = screenToImg(e);
          if (panelIdx >= 0 && imgCol >= 0 && imgCol < width && imgRow >= 0 && imgRow < height) {
            const pt = { row: imgRow, col: imgCol };
            if (profilePoints.length === 0 || profilePoints.length === 2 || panelIdx !== profilePanelIdx) {
              setProfilePanelIdx(panelIdx);
              setProfileLine([pt]);
              setProfileData(null);
            } else {
              const p0 = profilePoints[0];
              setProfileLine([p0, pt]);
              setProfileData(sampleLineProfile(rawFrameDataRef.current, width, height, p0.row, p0.col, pt.row, pt.col, profileWidth));
            }
          }
        }
      }
    }

    // ROI click-to-add (empty-area click)
    if (effectiveRoiActive && pendingRoiAddRef.current && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        addROIAt(pendingRoiAddRef.current.row, pendingRoiAddRef.current.col);
      }
    }
    clickStartRef.current = null;
    pendingRoiAddRef.current = null;
    if (isDraggingPan) commitLivePanelTransforms();
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingLens(false);
    lensDragStartRef.current = null;
    setIsResizingLens(false);
    lensResizeStartRef.current = null;
    setIsDraggingPan(false);
    setPanStart(null);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    profileDragStartRef.current = null;
  };

  const handleCanvasMouseLeave = () => {
    scheduleCursorInfo(null);
    // Lens persists at last position when cursor exits main canvas. Wiping on every
    // leave kills the inset whenever the user touches a slider, FFT panel, or any
    // sibling control - surprising "lens vanished" footgun. User explicitly turns
    // lens off via the Lens switch.
    pendingRoiAddRef.current = null;
    if (isDraggingPan) commitLivePanelTransforms();
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingLens(false);
    lensDragStartRef.current = null;
    setIsResizingLens(false);
    lensResizeStartRef.current = null;
    setIsHoveringLensEdge(false);
    setIsHoveringResize(false);
    setIsHoveringResizeInner(false);
    setIsDraggingPan(false);
    setPanStart(null);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    profileDragStartRef.current = null;
  };

  // FFT mouse handlers
  const [isFftDragging, setIsFftDragging] = React.useState(false);
  const [fftPanStart, setFftPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  const clampFftPan = React.useCallback((panX: number, panY: number, zoom: number, viewportW: number, viewportH: number) => {
    const clampAxis = (pan: number, viewport: number) => {
      if (zoom <= 1 || viewport <= 0) return 0;
      return Math.max(viewport * (1 - zoom), Math.min(0, pan));
    };
    return {
      panX: clampAxis(panX, viewportW),
      panY: clampAxis(panY, viewportH),
    };
  }, []);

  const zoomFftAtPoint = React.useCallback((anchorX: number, anchorY: number, deltaY: number, viewportW?: number, viewportH?: number) => {
    const currentBase = fftViewLiveRef.current;
    const current = !fftUserAdjustedViewRef.current && currentBase.zoom > 1 && viewportW != null && viewportH != null
      ? {
        zoom: currentBase.zoom,
        panX: viewportW * (1 - currentBase.zoom) / 2,
        panY: viewportH * (1 - currentBase.zoom) / 2,
      }
      : currentBase;
    fftUserAdjustedViewRef.current = true;
    fftOverlayInitialCenterPendingRef.current = false;
    fftViewCenterOnViewportRef.current = false;
    const zoomFactor = Math.max(0.75, Math.min(1.35, Math.exp(-deltaY * 0.002)));
    const minZoom = fftLayoutOverlay ? 1 : MIN_ZOOM;
    const newZoom = Math.max(minZoom, Math.min(MAX_ZOOM, current.zoom * zoomFactor));
    const zoomRatio = newZoom / Math.max(1e-6, current.zoom);
    const nextPanX = anchorX - (anchorX - current.panX) * zoomRatio;
    const nextPanY = anchorY - (anchorY - current.panY) * zoomRatio;
    const clamped = viewportW != null && viewportH != null
      ? clampFftPan(nextPanX, nextPanY, newZoom, viewportW, viewportH)
      : { panX: nextPanX, panY: nextPanY };
    scheduleFftViewState({ zoom: newZoom, panX: clamped.panX, panY: clamped.panY }, true, fftLayoutOverlay);
  }, [clampFftPan, fftLayoutOverlay, scheduleFftViewState]);

  fftInsetNativeWheelHandlerRef.current = (event: WheelEvent) => {
    const target = event.target;
    let inset: Element | null = target instanceof Element
      ? target.closest('[data-show3d-fft-inset="true"]')
      : null;
    if (!(inset instanceof HTMLElement)) {
      inset = document.elementsFromPoint(event.clientX, event.clientY)
        .find(el => el instanceof Element && el.closest('[data-show3d-fft-inset="true"]'))
        ?.closest('[data-show3d-fft-inset="true"]') ?? null;
    }
    if (!(inset instanceof HTMLElement)) {
      const root = rootRef.current;
      const hit = root
        ? Array.from(root.querySelectorAll<HTMLElement>('[data-show3d-fft-inset="true"]')).find(el => {
          const rect = el.getBoundingClientRect();
          return event.clientX >= rect.left && event.clientX <= rect.right
            && event.clientY >= rect.top && event.clientY <= rect.bottom;
        })
        : null;
      inset = hit ?? null;
    }
    if (!(inset instanceof HTMLElement)) return false;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    const rect = inset.getBoundingClientRect();
    zoomFftAtPoint(event.clientX - rect.left, event.clientY - rect.top, event.deltaY, rect.width, rect.height);
    return true;
  };

  const handleFftWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const canvas = fftCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const panelGrid = fftPanelGridRef.current;
    if (panelGrid) {
      for (let slot = 0; slot < panelGrid.count; slot++) {
        const dst = getFftSlot(slot, panelGrid.count, panelGrid.cols, panelGrid.rows);
        if (mouseX < dst.x || mouseX >= dst.x + dst.w || mouseY < dst.y || mouseY >= dst.y + dst.h) continue;
        const localX = mouseX - dst.x;
        const localY = mouseY - dst.y;
        zoomFftAtPoint(localX, localY, e.deltaY, dst.w, dst.h);
        return;
      }
    }
    zoomFftAtPoint(mouseX, mouseY, e.deltaY, canvas.width, canvas.height);
  };

  const handleFftInsetWheel = (e: React.WheelEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const localX = e.clientX - rect.left;
    const localY = e.clientY - rect.top;
    zoomFftAtPoint(localX, localY, e.deltaY, rect.width, rect.height);
  };

  const handleFftInsetPointerDown = (
    e: React.PointerEvent<HTMLElement>,
    panelLeft: number,
    panelTop: number,
    panelW: number,
    panelH: number,
    insetX: number,
    insetY: number,
    insetW: number,
    insetH: number,
  ) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    fftOverlayDragRef.current = {
      pointerId: e.pointerId,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startInsetX: insetX,
      startInsetY: insetY,
      panelLeft,
      panelTop,
      panelW,
      panelH,
      insetW,
      insetH,
      moved: false,
    };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const handleFftInsetPointerMove = (e: React.PointerEvent<HTMLElement>) => {
    const drag = fftOverlayDragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    if (Math.hypot(e.clientX - drag.startClientX, e.clientY - drag.startClientY) > 4) {
      drag.moved = true;
    }
    if (drag.moved) {
      const nextX = Math.max(drag.panelLeft, Math.min(drag.panelLeft + drag.panelW - drag.insetW, drag.startInsetX + e.clientX - drag.startClientX));
      const nextY = Math.max(drag.panelTop, Math.min(drag.panelTop + drag.panelH - drag.insetH, drag.startInsetY + e.clientY - drag.startClientY));
      setFftOverlayDragPreview({ x: nextX - drag.panelLeft, y: nextY - drag.panelTop });
    }
  };

  const handleFftInsetPointerUp = (e: React.PointerEvent<HTMLElement>) => {
    const drag = fftOverlayDragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    fftOverlayDragRef.current = null;
    setFftOverlayDragPreview(null);
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    if (!drag.moved) return;
    const centerX = drag.startInsetX + e.clientX - drag.startClientX - drag.panelLeft + drag.insetW / 2;
    const centerY = drag.startInsetY + e.clientY - drag.startClientY - drag.panelTop + drag.insetH / 2;
    const vertical = centerY < drag.panelH / 2 ? "top" : "bottom";
    const horizontal = centerX < drag.panelW / 2 ? "left" : "right";
    setFftOverlayPosition(`${vertical}-${horizontal}`);
  };

  const handleFftInsetMouseDown = (
    e: React.MouseEvent<HTMLElement>,
    panelLeft: number,
    panelTop: number,
    panelW: number,
    panelH: number,
    insetX: number,
    insetY: number,
    insetW: number,
    insetH: number,
  ) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const drag = {
      pointerId: -1,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startInsetX: insetX,
      startInsetY: insetY,
      panelLeft,
      panelTop,
      panelW,
      panelH,
      insetW,
      insetH,
      moved: false,
    };
    fftOverlayDragRef.current = drag;
    const onMove = (ev: MouseEvent) => {
      ev.preventDefault();
      if (Math.hypot(ev.clientX - drag.startClientX, ev.clientY - drag.startClientY) > 4) {
        drag.moved = true;
      }
      if (drag.moved) {
        const nextX = Math.max(drag.panelLeft, Math.min(drag.panelLeft + drag.panelW - drag.insetW, drag.startInsetX + ev.clientX - drag.startClientX));
        const nextY = Math.max(drag.panelTop, Math.min(drag.panelTop + drag.panelH - drag.insetH, drag.startInsetY + ev.clientY - drag.startClientY));
        setFftOverlayDragPreview({ x: nextX - drag.panelLeft, y: nextY - drag.panelTop });
      }
    };
    const onUp = (ev: MouseEvent) => {
      window.removeEventListener("mousemove", onMove, true);
      window.removeEventListener("mouseup", onUp, true);
      if (fftOverlayDragRef.current === drag) fftOverlayDragRef.current = null;
      setFftOverlayDragPreview(null);
      if (!drag.moved) return;
      const centerX = drag.startInsetX + ev.clientX - drag.startClientX - drag.panelLeft + drag.insetW / 2;
      const centerY = drag.startInsetY + ev.clientY - drag.startClientY - drag.panelTop + drag.insetH / 2;
      const vertical = centerY < drag.panelH / 2 ? "top" : "bottom";
      const horizontal = centerX < drag.panelW / 2 ? "left" : "right";
      setFftOverlayPosition(`${vertical}-${horizontal}`);
    };
    window.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);
  };

  const handleFftInsetPanMouseDown = (e: React.MouseEvent<HTMLElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const target = e.currentTarget;
    const rect = target.getBoundingClientRect();
    const viewportW = Math.max(1, rect.width);
    const viewportH = Math.max(1, rect.height);
    const startX = e.clientX;
    const startY = e.clientY;
    const current = fftViewLiveRef.current;
    const startView = !fftUserAdjustedViewRef.current && current.zoom > 1
      ? {
        zoom: current.zoom,
        panX: viewportW * (1 - current.zoom) / 2,
        panY: viewportH * (1 - current.zoom) / 2,
      }
      : current;
    if (!fftUserAdjustedViewRef.current) {
      scheduleFftViewState(startView, false, fftLayoutOverlay);
    }
    fftUserAdjustedViewRef.current = true;
    fftOverlayInitialCenterPendingRef.current = false;
    fftViewCenterOnViewportRef.current = false;
    const onMove = (ev: MouseEvent) => {
      ev.preventDefault();
      const clamped = clampFftPan(
        startView.panX + (ev.clientX - startX),
        startView.panY + (ev.clientY - startY),
        startView.zoom,
        viewportW,
        viewportH,
      );
      scheduleFftViewState({ zoom: startView.zoom, panX: clamped.panX, panY: clamped.panY }, false, fftLayoutOverlay);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove, true);
      window.removeEventListener("mouseup", onUp, true);
    };
    window.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);
  };

  React.useEffect(() => {
    if (!effectiveShowFft) return;
    const overlayCanvas = fftLayoutOverlay ? fftInsetLayerRef.current : null;
    const fftCanvas = fftCanvasRef.current;
    const panelGrid = fftPanelGridRef.current;
    const viewport = overlayCanvas
      ? (() => {
        const n = Math.max(1, visiblePanelCount || 1);
        const cols = panelColsForCount(n);
        const rows = Math.ceil(n / cols);
        const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
        const panelW = (canvasW - gap * (cols - 1)) / cols;
        const panelH = (canvasH - gap * (rows - 1)) / rows;
        const insetPad = Math.min(8, Math.max(3, panelW * 0.025));
        const insetMaxW = Math.max(24, panelW - insetPad * 2);
        const insetMaxH = Math.max(20, panelH - insetPad * 2);
        const insetBase = Math.min(insetMaxW, insetMaxH);
        return {
          w: Math.max(24, Math.min(insetMaxW, insetBase * resolvedFftOverlaySize)),
          h: Math.max(20, Math.min(insetMaxH, insetBase * resolvedFftOverlaySize)),
        };
      })()
      : fftCanvas
        ? panelGrid
          ? getFftSlot(0, panelGrid.count, panelGrid.cols, panelGrid.rows)
          : { w: fftCanvas.width, h: fftCanvas.height }
        : null;
    if (!viewport) return;
    const current = fftViewLiveRef.current;
    const centered = fftViewCenterOnViewportRef.current && current.zoom > 1
      ? {
        panX: viewport.w * (1 - current.zoom) / 2,
        panY: viewport.h * (1 - current.zoom) / 2,
      }
      : { panX: current.panX, panY: current.panY };
    fftViewCenterOnViewportRef.current = false;
    fftOverlayInitialCenterPendingRef.current = false;
    const clamped = clampFftPan(centered.panX, centered.panY, current.zoom, viewport.w, viewport.h);
    if (Math.abs(clamped.panX - current.panX) > 0.5 || Math.abs(clamped.panY - current.panY) > 0.5) {
      scheduleFftViewState({ zoom: current.zoom, panX: clamped.panX, panY: clamped.panY });
    }
  }, [clampFftPan, effectiveShowFft, fftLayoutOverlay, fftZoom, fftPanX, fftPanY, canvasW, canvasH, resolvedFftOverlaySize, visiblePanelCount, panelColsForCount, panelGapTrait, fftOffscreenVersion, scheduleFftViewState]);

  // Convert FFT canvas mouse position to FFT image pixel coordinates
  const fftScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    const panelGrid = fftPanelGridRef.current;
    if (panelGrid) {
      for (let slot = 0; slot < panelGrid.count; slot++) {
        const dst = getFftSlot(slot, panelGrid.count, panelGrid.cols, panelGrid.rows);
        if (mouseX < dst.x || mouseX >= dst.x + dst.w || mouseY < dst.y || mouseY >= dst.y + dst.h) continue;
        const localX = (mouseX - dst.x - fftPanX) / fftZoom;
        const localY = (mouseY - dst.y - fftPanY) / fftZoom;
        if (localX < 0 || localX >= dst.w || localY < 0 || localY >= dst.h) return null;
        const srcCol = slot % panelGrid.cols;
        const srcRow = Math.floor(slot / panelGrid.cols);
        const tileX = (localX / Math.max(1, dst.w)) * panelGrid.panelWidth;
        const tileY = (localY / Math.max(1, dst.h)) * panelGrid.panelHeight;
        return {
          col: srcCol * panelGrid.panelWidth + Math.max(0, Math.min(panelGrid.panelWidth - 1, tileX)),
          row: srcRow * panelGrid.panelHeight + Math.max(0, Math.min(panelGrid.panelHeight - 1, tileY)),
        };
      }
      return null;
    }
    const localX = (mouseX - fftPanX) / fftZoom;
    const localY = (mouseY - fftPanY) / fftZoom;
    const imgCol = localX / canvasW * fftW;
    const imgRow = localY / canvasH * fftH;
    if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
      return { col: imgCol, row: imgRow };
    }
    return null;
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsFftDragging(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: fftPanX, pY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent) => {
    if (isFftDragging && fftPanStart) {
      const canvas = fftCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - fftPanStart.x) * scaleX;
      const dy = (e.clientY - fftPanStart.y) * scaleY;
      const panelGrid = fftPanelGridRef.current;
      const viewport = panelGrid ? getFftSlot(0, panelGrid.count, panelGrid.cols, panelGrid.rows) : { w: canvas.width, h: canvas.height };
      const clamped = clampFftPan(fftPanStart.pX + dx, fftPanStart.pY + dy, fftZoom, viewport.w, viewport.h);
      setFftPanX(clamped.panX);
      setFftPanY(clamped.panY);
    }
  };

  const handleFftMouseUp = (e: React.MouseEvent) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const pos = fftScreenToImg(e);
        if (pos) {
          // Use crop dimensions when ROI FFT is active
          const fftW = fftCropDims?.fftWidth ?? width;
          const fftH = fftCropDims?.fftHeight ?? height;
          const panelGrid = fftPanelGridRef.current;
          let imgCol = pos.col;
          let imgRow = pos.row;
          if (fftMagCacheRef.current) {
            const bounds = panelGrid ? (() => {
              const panelCol = Math.max(0, Math.min(panelGrid.cols - 1, Math.floor(imgCol / panelGrid.panelWidth)));
              const panelRow = Math.max(0, Math.min(panelGrid.rows - 1, Math.floor(imgRow / panelGrid.panelHeight)));
              return {
                minCol: panelCol * panelGrid.panelWidth,
                maxCol: Math.min(fftW - 1, (panelCol + 1) * panelGrid.panelWidth - 1),
                minRow: panelRow * panelGrid.panelHeight,
                maxRow: Math.min(fftH - 1, (panelRow + 1) * panelGrid.panelHeight - 1),
              };
            })() : null;
            const snapped = bounds
              ? findFFTPeakInBounds(fftMagCacheRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS, bounds.minCol, bounds.maxCol, bounds.minRow, bounds.maxRow)
              : findFFTPeak(fftMagCacheRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
            imgCol = snapped.col;
            imgRow = snapped.row;
          }
          const local = panelGrid ? (() => {
            const panelCol = Math.max(0, Math.min(panelGrid.cols - 1, Math.floor(imgCol / panelGrid.panelWidth)));
            const panelRow = Math.max(0, Math.min(panelGrid.rows - 1, Math.floor(imgRow / panelGrid.panelHeight)));
            return {
              col: imgCol - panelCol * panelGrid.panelWidth,
              row: imgRow - panelRow * panelGrid.panelHeight,
              width: panelGrid.panelWidth,
              height: panelGrid.panelHeight,
            };
          })() : { col: imgCol, row: imgRow, width: fftW, height: fftH };
          const halfW = Math.floor(local.width / 2);
          const halfH = Math.floor(local.height / 2);
          const dcol = local.col - halfW;
          const drow = local.row - halfH;
          const distPx = Math.sqrt(dcol * dcol + drow * drow);
          if (distPx < 1) {
            setFftClickInfo(null);
          } else {
            let spatialFreq: number | null = null;
            let dSpacing: number | null = null;
            if (pixelSize > 0) {
              const paddedW = nextPow2(local.width);
              const paddedH = nextPow2(local.height);
              const binC = ((Math.round(local.col) - halfW) % local.width + local.width) % local.width;
              const binR = ((Math.round(local.row) - halfH) % local.height + local.height) % local.height;
              const freqC = binC <= paddedW / 2 ? binC / (paddedW * pixelSize) : (binC - paddedW) / (paddedW * pixelSize);
              const freqR = binR <= paddedH / 2 ? binR / (paddedH * pixelSize) : (binR - paddedH) / (paddedH * pixelSize);
              spatialFreq = Math.sqrt(freqC * freqC + freqR * freqR);
              dSpacing = spatialFreq > 0 ? 1 / spatialFreq : null;
            }
            setFftClickInfo({ row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
          }
        }
      }
      fftClickStartRef.current = null;
    }
    setIsFftDragging(false);
    setFftPanStart(null);
  };

  const handleFftTouchStart = (e: React.TouchEvent) => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return;
    const now = Date.now();
    const base = { zoom: fftZoom, panX: fftPanX, panY: fftPanY };
    if (e.touches.length === 1) {
      const lastTap = lastFftTapRef.current;
      if (lastTap && now - lastTap.time < 320) {
        e.preventDefault();
        handleFftReset();
        lastFftTapRef.current = null;
        fftTouchTransformRef.current = null;
        return;
      }
      lastFftTapRef.current = { time: now };
      const t = e.touches[0];
      fftTouchTransformRef.current = {
        mode: "pan",
        startX: t.clientX,
        startY: t.clientY,
        startDistance: 0,
        startMidX: t.clientX,
        startMidY: t.clientY,
        startState: base,
      };
      e.preventDefault();
      return;
    }
    if (e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      fftTouchTransformRef.current = {
        mode: "pinch",
        startX: mid.x,
        startY: mid.y,
        startDistance: Math.max(1, touchDistance(a, b)),
        startMidX: mid.x,
        startMidY: mid.y,
        startState: base,
      };
      e.preventDefault();
    }
  };

  const handleFftTouchMove = (e: React.TouchEvent) => {
    const start = fftTouchTransformRef.current;
    const canvas = fftCanvasRef.current;
    if (!start || !canvas) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const toCanvas = (clientX: number, clientY: number) => ({
      x: (clientX - rect.left) * (canvas.width / Math.max(1, rect.width)),
      y: (clientY - rect.top) * (canvas.height / Math.max(1, rect.height)),
    });
    const base = start.startState;
    if (start.mode === "pinch" && e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      const startCanvas = toCanvas(start.startMidX, start.startMidY);
      const currentCanvas = toCanvas(mid.x, mid.y);
      const imageX = (startCanvas.x - base.panX) / base.zoom;
      const imageY = (startCanvas.y - base.panY) / base.zoom;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base.zoom * (touchDistance(a, b) / start.startDistance)));
      setFftZoom(newZoom);
      setFftPanX(currentCanvas.x - imageX * newZoom);
      setFftPanY(currentCanvas.y - imageY * newZoom);
      return;
    }
    if (start.mode === "pan" && e.touches.length === 1) {
      const t = e.touches[0];
      const scaleX = canvas.width / Math.max(1, rect.width);
      const scaleY = canvas.height / Math.max(1, rect.height);
      setFftPanX(base.panX + (t.clientX - start.startX) * scaleX);
      setFftPanY(base.panY + (t.clientY - start.startY) * scaleY);
    }
  };

  const handleFftTouchEnd = (e: React.TouchEvent) => {
    if (e.touches.length > 0 || !fftTouchTransformRef.current) return;
    fftTouchTransformRef.current = null;
  };

  const handleFftReset = () => {
    const reset = { zoom: 1, panX: 0, panY: 0 };
    fftViewLiveRef.current = reset;
    fftViewCenterOnViewportRef.current = true;
    fftOverlayInitialCenterPendingRef.current = true;
    fftUserAdjustedViewRef.current = false;
    setFftZoom(reset.zoom);
    internalFftZoomSyncRef.current = true;
    setFftOverlayZoomTrait(1);
    setFftPanX(reset.panX);
    setFftPanY(reset.panY);
    setFftClickInfo(null);
  };

  // Kymograph mouse handlers (mirror FFT: wheel-zoom + pan-drag). Click readout
  // replaces the FFT d-spacing measurement (domain adaptation).
  const [isKymoDragging, setIsKymoDragging] = React.useState(false);
  const [kymoPanStart, setKymoPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  const handleKymoWheel = (e: React.WheelEvent) => {
    const canvas = kymoCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, kymoZoom * zoomFactor));
    const zoomRatio = newZoom / kymoZoom;
    setKymoZoom(newZoom);
    setKymoPanX(mouseX - (mouseX - kymoPanX) * zoomRatio);
    setKymoPanY(mouseY - (mouseY - kymoPanY) * zoomRatio);
  };

  // Convert kymograph canvas mouse position to (frame index, distance index).
  const kymoScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = kymoCanvasRef.current;
    const kymo = kymoDataRef.current;
    if (!canvas || !kymo) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;
    // The click already passed the canvas hit-test, so map into the image and
    // clamp - edge/last-row clicks must still yield a readout (a strict
    // `< nFrames` check silently dropped clicks on the bottom row).
    const imgCol = Math.max(0, Math.min(kymo.lineLen - 1, ((mouseX - kymoPanX) / kymoZoom) / canvasW * kymo.lineLen));
    const imgRow = Math.max(0, Math.min(kymo.nFrames - 1, ((mouseY - kymoPanY) / kymoZoom) / canvasH * kymo.nFrames));
    return { col: imgCol, row: imgRow };
  };

  const handleKymoMouseDown = (e: React.MouseEvent) => {
    kymoClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsKymoDragging(true);
    setKymoPanStart({ x: e.clientX, y: e.clientY, pX: kymoPanX, pY: kymoPanY });
  };

  const handleKymoMouseMove = (e: React.MouseEvent) => {
    if (isKymoDragging && kymoPanStart) {
      const canvas = kymoCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - kymoPanStart.x) * scaleX;
      const dy = (e.clientY - kymoPanStart.y) * scaleY;
      setKymoPanX(kymoPanStart.pX + dx);
      setKymoPanY(kymoPanStart.pY + dy);
    }
  };

  const handleKymoMouseUp = (e: React.MouseEvent) => {
    // Click detection for intensity readout at (time, distance).
    if (kymoClickStartRef.current) {
      const dx = e.clientX - kymoClickStartRef.current.x;
      const dy = e.clientY - kymoClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const pos = kymoScreenToImg(e);
        const kymo = kymoDataRef.current;
        if (pos && kymo) {
          const frame = Math.max(0, Math.min(kymo.nFrames - 1, Math.round(pos.row)));
          const dist = Math.max(0, Math.min(kymo.lineLen - 1, Math.round(pos.col)));
          const intensity = kymo.data[frame * kymo.lineLen + dist];
          const timeVal = dimSampling > 0 && dimUnit ? frame * dimSampling : frame;
          const timeUnit = dimSampling > 0 && dimUnit ? unitSymbol(dimUnit) : "frame";
          const distVal = pixelSize > 0 ? dist * pixelSize : dist;
          const distUnit = pixelSize > 0 ? unitSymbol(pixelUnit || "px") : "px";
          setKymoClickInfo({ timeVal, timeUnit, distVal, distUnit, intensity, col: dist, row: frame });
        } else {
          setKymoClickInfo(null);
        }
      }
      kymoClickStartRef.current = null;
    }
    setIsKymoDragging(false);
    setKymoPanStart(null);
  };

  const handleKymoTouchStart = (e: React.TouchEvent) => {
    const canvas = kymoCanvasRef.current;
    if (!canvas) return;
    const now = Date.now();
    const base = { zoom: kymoZoom, panX: kymoPanX, panY: kymoPanY };
    if (e.touches.length === 1) {
      const lastTap = lastKymoTapRef.current;
      if (lastTap && now - lastTap.time < 320) {
        e.preventDefault();
        handleKymoReset();
        lastKymoTapRef.current = null;
        kymoTouchTransformRef.current = null;
        return;
      }
      lastKymoTapRef.current = { time: now };
      const t = e.touches[0];
      kymoTouchTransformRef.current = {
        mode: "pan",
        startX: t.clientX,
        startY: t.clientY,
        startDistance: 0,
        startMidX: t.clientX,
        startMidY: t.clientY,
        startState: base,
      };
      e.preventDefault();
      return;
    }
    if (e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      kymoTouchTransformRef.current = {
        mode: "pinch",
        startX: mid.x,
        startY: mid.y,
        startDistance: Math.max(1, touchDistance(a, b)),
        startMidX: mid.x,
        startMidY: mid.y,
        startState: base,
      };
      e.preventDefault();
    }
  };

  const handleKymoTouchMove = (e: React.TouchEvent) => {
    const start = kymoTouchTransformRef.current;
    const canvas = kymoCanvasRef.current;
    if (!start || !canvas) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const toCanvas = (clientX: number, clientY: number) => ({
      x: (clientX - rect.left) * (canvas.width / Math.max(1, rect.width)),
      y: (clientY - rect.top) * (canvas.height / Math.max(1, rect.height)),
    });
    const base = start.startState;
    if (start.mode === "pinch" && e.touches.length >= 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      const mid = touchMidpoint(a, b);
      const startCanvas = toCanvas(start.startMidX, start.startMidY);
      const currentCanvas = toCanvas(mid.x, mid.y);
      const imageX = (startCanvas.x - base.panX) / base.zoom;
      const imageY = (startCanvas.y - base.panY) / base.zoom;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base.zoom * (touchDistance(a, b) / start.startDistance)));
      setKymoZoom(newZoom);
      setKymoPanX(currentCanvas.x - imageX * newZoom);
      setKymoPanY(currentCanvas.y - imageY * newZoom);
      return;
    }
    if (start.mode === "pan" && e.touches.length === 1) {
      const t = e.touches[0];
      const scaleX = canvas.width / Math.max(1, rect.width);
      const scaleY = canvas.height / Math.max(1, rect.height);
      setKymoPanX(base.panX + (t.clientX - start.startX) * scaleX);
      setKymoPanY(base.panY + (t.clientY - start.startY) * scaleY);
    }
  };

  const handleKymoTouchEnd = (e: React.TouchEvent) => {
    if (e.touches.length > 0 || !kymoTouchTransformRef.current) return;
    kymoTouchTransformRef.current = null;
  };

  const handleKymoReset = () => {
    setKymoZoom(1);
    setKymoPanX(0);
    setKymoPanY(0);
    setKymoClickInfo(null);
  };

  const kymoNeedsReset = kymoZoom !== 1 || kymoPanX !== 0 || kymoPanY !== 0;

  // Preview panel zoom/pan handlers
  const handlePreviewWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const canvas = previewCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const pw = previewCanvasDims.w;
    const ph = previewCanvasDims.h;
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = pw / 2;
    const cy = ph / 2;
    const mouseImageX = (mouseCanvasX - cx - previewZoom.panX) / previewZoom.zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - previewZoom.panY) / previewZoom.zoom + cy;
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, previewZoom.zoom * zoomFactor));
    const newPanX = mouseCanvasX - (mouseImageX - cx) * newZoom - cx;
    const newPanY = mouseCanvasY - (mouseImageY - cy) * newZoom - cy;
    setPreviewZoom({ zoom: newZoom, panX: newPanX, panY: newPanY });
  };

  const handlePreviewMouseDown = (e: React.MouseEvent) => {
    setIsDraggingPreviewPan(true);
    setPreviewPanStart({ x: e.clientX, y: e.clientY, pX: previewZoom.panX, pY: previewZoom.panY });
  };

  const handlePreviewMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingPreviewPan || !previewPanStart) return;
    const canvas = previewCanvasRef.current;
    const rect = canvas?.getBoundingClientRect();
    const scaleX = canvas && rect ? canvas.width / Math.max(1, rect.width) : 1;
    const scaleY = canvas && rect ? canvas.height / Math.max(1, rect.height) : 1;
    const dx = (e.clientX - previewPanStart.x) * scaleX;
    const dy = (e.clientY - previewPanStart.y) * scaleY;
    setPreviewZoom(prev => ({ ...prev, panX: previewPanStart.pX + dx, panY: previewPanStart.pY + dy }));
  };

  const handlePreviewMouseUp = () => {
    setIsDraggingPreviewPan(false);
    setPreviewPanStart(null);
  };

  const handlePreviewDoubleClick = () => {
    setPreviewZoom({ zoom: 1, panX: 0, panY: 0 });
  };

  // Resize handlers
  const handleMainResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    const rect = canvasContainerRef.current?.getBoundingClientRect();
    const startSize = rect && rect.width > 0 ? rect.width : mainCanvasSize;
    const startX = e.clientX;
    const startY = e.clientY;
    const visiblePanels = Math.max(1, visiblePanelCount || 1);
    let rafId = 0;
    let latestSize = startSize;
    const handleMouseMove = (e: MouseEvent) => {
      const delta = Math.max(e.clientX - startX, e.clientY - startY);
      const nextSize = startSize + delta;
      // Absolute minimum: 200 px per panel column. Lets reader shrink BELOW
      // the initial `size=` value (preset / kwarg) when their screen is small,
      // without collapsing the canvas to an unreadable sliver.
      const colsLocal = panelColsForCount(visiblePanels);
      const minSize = 200 * colsLocal;
      latestSize = Math.max(minSize, nextSize);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setMainCanvasSize(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      cancelAnimationFrame(rafId);
      setMainCanvasSize(latestSize);
      const colsLocal = panelColsForCount(visiblePanels);
      setCanvasSizeTrait(Math.round(latestSize / colsLocal));
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const clampSlice = (idx: number) => Math.max(0, Math.min(nSlices - 1, Math.round(idx)));
  const frameLabelForIndex = React.useCallback((idx: number): string => {
    const label = labels?.[idx];
    if (label == null) return "";
    const text = String(label).trim();
    if (!text || text === String(idx) || text === String(idx + 1)) return "";
    return text;
  }, [labels]);
  const panelFrameLabelForIndex = React.useCallback((panel: number, idx: number): string => {
    const panelLabels = panelFrameLabels?.[panel];
    const panelRealN = panelRealFrames?.[panel];
    const panelIdx = panelRealN ? Math.min(idx, Math.max(0, panelRealN - 1)) : idx;
    const label = panelLabels?.[panelIdx];
    if (label != null) {
      const text = String(label).trim();
      if (text && text !== String(panelIdx) && text !== String(panelIdx + 1)) return text;
    }
    return frameLabelForIndex(idx);
  }, [frameLabelForIndex, panelFrameLabels, panelRealFrames]);
  const formatFrameValueLabel = React.useCallback((idx: number) => {
    const rounded = clampSlice(idx);
    const label = frameLabelForIndex(rounded);
    return label ? `${rounded + 1}: ${label}` : `${rounded + 1}`;
  }, [frameLabelForIndex, nSlices]);
  const visibleSliceIdx = clampSlice(playing ? playbackUiSliceIdx : (offline ? liveSliceIdx : displaySliceIdx));
  React.useLayoutEffect(() => {
    updatePlaybackLiveControls(visibleSliceIdx);
  }, [updatePlaybackLiveControls, visibleSliceIdx]);
  const currentPlaybackIndex = () => (
    Number.isFinite(playbackIdxRef.current)
      ? playbackIdxRef.current
      : (Number.isFinite(displaySliceIdx) ? displaySliceIdx : sliceIdx)
  );
  const playFromCurrentFrame = (direction: 1 | -1 | null = null) => {
    const nextReverse = direction === null ? reverse : direction < 0;
    const rangeStart = loop ? Math.max(0, Math.min(loopStart, nSlices - 1)) : 0;
    const rangeEnd = loop ? Math.max(rangeStart, Math.min(effectiveLoopEnd, nSlices - 1)) : nSlices - 1;
    let start = Math.max(rangeStart, Math.min(rangeEnd, Math.round(currentPlaybackIndex())));
    if (!loop) {
      if (!nextReverse && start >= rangeEnd) start = rangeStart;
      if (nextReverse && start <= rangeStart) start = rangeEnd;
    }
    playbackIdxRef.current = start;
    setDisplaySliceIdx(start);
    setPlaybackUiSliceIdx(start);
    setLiveSliceIdx(start);
    setSliceIdx(start);
    if (direction !== null) setReverse(nextReverse);
    setPlaying(true);
  };
  const pausePlayback = () => {
    const current = clampSlice(currentPlaybackIndex());
    playbackIdxRef.current = current;
    setDisplaySliceIdx(current);
    setPlaybackUiSliceIdx(current);
    setLiveSliceIdx(current);
    setSliceIdx(current);
    setPlaying(false);
  };
  const stopPlayback = () => {
    const home = loop ? Math.max(0, Math.min(loopStart, nSlices - 1)) : 0;
    playbackIdxRef.current = home;
    setDisplaySliceIdx(home);
    setPlaybackUiSliceIdx(home);
    setLiveSliceIdx(home);
    setSliceIdx(home);
    setPlaying(false);
  };

  // Keyboard
  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (shouldIgnoreWidgetShortcut(e.target)) return;

    let handled = false;

    switch (e.key) {
        case " ":
          if (playing) pausePlayback();
          else playFromCurrentFrame();
          handled = true;
          break;
        case "ArrowLeft": {
          const lo = loop ? Math.max(0, loopStart) : 0;
          const candidate = hiddenSet.size ? nextVisible(sliceIdx, -1, false) : sliceIdx - 1;
          setSliceIdx(Math.max(lo, candidate));
          handled = true;
          break;
        }
        case "ArrowRight": {
          const hi = loop ? Math.min(effectiveLoopEnd, nSlices - 1) : nSlices - 1;
          const candidate = hiddenSet.size ? nextVisible(sliceIdx, 1, false) : sliceIdx + 1;
          setSliceIdx(Math.min(hi, candidate));
          handled = true;
          break;
        }
        case "Home":
          setSliceIdx(loop ? Math.max(0, loopStart) : 0);
          handled = true;
          break;
        case "End":
          setSliceIdx(loop ? Math.min(effectiveLoopEnd, nSlices - 1) : nSlices - 1);
          handled = true;
          break;
        case "r":
        case "R":
          handleDoubleClick();
          handled = true;
          break;
        case "c":
        case "C":
          if (cursorInfo && cursorReadoutVisible) {
            navigator.clipboard.writeText(`(${cursorInfo.row}, ${cursorInfo.col}, ${cursorInfo.value})`);
            handled = true;
          }
          break;
        case "Delete":
        case "Backspace":
          if (effectiveRoiActive && roiSelectedIdx >= 0) {
            deleteSelectedROI();
            handled = true;
          }
          break;
        case "d":
        case "D":
          if (effectiveRoiActive && roiSelectedIdx >= 0 && (e.metaKey || e.ctrlKey || e.shiftKey)) {
            duplicateSelectedROI();
            handled = true;
          }
          break;
        case "Escape":
          rootRef.current?.blur();
          handled = true;
          break;
      }
    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  // Check if view needs reset
  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;
  const scrubToSlice = (idx: number) => {
    const next = clampSlice(idx);
    if (playing) setPlaying(false);
    setPlaybackUiSliceIdx(next);
    const transformActive = diffMode !== "off" || Math.max(1, Math.round(avgWindow || 1)) > 1;
    if (!transformActive && renderGpuCachedSliceDirect(next)) return;
    setLiveSliceIdx(next);
    if (renderBufferedSlice(next)) return;
    if (!offline && frameServerUrl) {
      setDisplaySliceIdx(next);
      setPlaybackUiSliceIdx(next);
      void renderFetchedSlice(next);
      prefetchServerFrames(next, false, false);
      return;
    }
    setDisplaySliceIdx(next);
    setPlaybackUiSliceIdx(next);
    setSliceIdx(next);
  };
  const commitSlice = (idx: number) => {
    const next = clampSlice(idx);
    setLiveSliceIdx(next);
    setPlaybackUiSliceIdx(next);
    setSliceIdx(next);
  };
  const handleLoopSliderMouseDown = (e: React.MouseEvent<HTMLSpanElement>) => {
    const target = e.target as HTMLElement;
    if (target.closest(".MuiSlider-thumb")) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0;
    const next = clampSlice(pct * Math.max(0, nSlices - 1));
    e.preventDefault();
    e.stopPropagation();
    scrubToSlice(next);
    commitSlice(next);
  };
  const handleLoopSliderPointerDownCapture = (e: React.PointerEvent<HTMLSpanElement>) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    if (target.closest(".MuiSlider-thumb")) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const sliceFromClientX = (clientX: number) => {
      const pct = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
      return clampSlice(pct * Math.max(0, nSlices - 1));
    };
    const moveCurrent = (clientX: number, commit: boolean) => {
      const next = sliceFromClientX(clientX);
      scrubToSlice(next);
      if (commit) commitSlice(next);
    };
    e.preventDefault();
    e.stopPropagation();
    e.nativeEvent.stopImmediatePropagation();
    moveCurrent(e.clientX, false);
    const onMove = (ev: PointerEvent) => {
      ev.preventDefault();
      moveCurrent(ev.clientX, false);
    };
    const onUp = (ev: PointerEvent) => {
      ev.preventDefault();
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      moveCurrent(ev.clientX, true);
    };
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
  };
  const overlayCanvasVisible = effectiveRoiActive || profileActive;
  const lensCanvasVisible = showLens && lensPos !== null;
  const keyboardShortcutItems: [string, string][] = [
    ["Space", "Play / Pause"],
    ["← / →", `Prev / Next ${dimLabel.toLowerCase()}`],
    ["Home / End", `First / Last ${dimLabel.toLowerCase()}`],
    ["R", "Reset zoom"],
    ["C", "Copy cursor coords"],
    ...(roiAllowed ? [["Del", "Delete selected ROI"], ["Ctrl/⌘+D", "Duplicate selected ROI"]] as [string, string][] : []),
    ["Esc", "Release keyboard focus"],
    ["Scroll", "Zoom"],
    ["Dbl-click", "Reset view"],
  ];
  const webgpuStatusLabel =
    fftBackendInfo.webgpu === "ready" ? "available"
      : fftBackendInfo.webgpu === "software" ? "software adapter ignored"
        : fftBackendInfo.webgpu === "unavailable" ? "unavailable"
          : "checking";
  const fftSourceRaw = fftBackendInfo.source || "";
  const fftSourceCached = fftSourceRaw.endsWith("-cache");
  const fftSourceBase = fftSourceCached ? fftSourceRaw.slice(0, -6) : fftSourceRaw;
  const fftSourceLabel =
    fftSourceCached ? "Cached"
      : fftSourceBase === "webgpu-batch" || fftSourceBase === "webgpu" ? "WebGPU"
        : fftSourceBase ? "CPU fallback"
        : "not run yet";
  const fftSourceDetail =
    fftSourceBase === "cpu-sync-shifted" ? "offline CPU"
      : fftSourceBase === "worker-batch" || fftSourceBase === "worker" ? "CPU worker"
        : fftSourceBase || "";
  return (
    <Box
      ref={rootRef}
      className="show3d-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
      sx={{ ...container.root, width: "100%", maxWidth: "100%", boxSizing: "border-box", bgcolor: themeColors.bg, color: themeColors.text, outline: "none", "&:focus": { outline: "2px solid #0af", outlineOffset: 2 }, "& canvas": { display: "block" }, "@media (max-width: 700px)": { p: 0, ".jp-OutputArea-output &, .jp-OutputArea-child &": { width: "calc(100vw - 96px)", maxWidth: "calc(100vw - 96px)" } } }}
    >
      {!canRenderLive && hasSavedStaticFallback && (
        <Box sx={{ width: "100%", maxWidth: mainPanelWidth, boxSizing: "border-box" }}>
          <Box
            component="img"
            src={staticFallbackUrl}
            alt={`${title || "Show3D"} saved preview`}
            sx={{
              display: "block",
              width: "100%",
              maxWidth: mainPanelWidth,
              height: "auto",
              border: `1px solid ${themeColors.border}`,
              boxSizing: "border-box",
            }}
          />
        </Box>
      )}
      {(canRenderLive || !hasSavedStaticFallback) && (
      <>
      <Stack
        direction="row"
        spacing={`${SPACING.SM}px`}
        alignItems="flex-start"
        sx={{
          flexWrap: effectiveShowFft && fftLayoutBottom && (nPanels || 1) > 1 ? "wrap" : "nowrap",
          width: "100%",
          maxWidth: "100%",
          minWidth: 0,
          boxSizing: "border-box",
          "@media (max-width: 900px)": {
            flexDirection: "column",
            alignItems: "stretch",
            flexWrap: "nowrap",
            "& > :not(style) + :not(style)": {
              marginLeft: "0 !important",
              marginTop: `${SPACING.SM}px`,
            },
          },
        }}
      >
        <Box sx={{ width: mainPanelWidth, maxWidth: "100%", flexShrink: effectiveShowFft && fftLayoutBottom && (nPanels || 1) > 1 ? 0 : 1, boxSizing: "border-box" }}>
          {/* Title row */}
          {showTitle && <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block", height: 16, lineHeight: "16px", overflow: "hidden" }}>
            {title || "Image"}
            {diffMode !== "off" && (
              <Typography component="span" sx={{ fontSize: 9, fontWeight: "bold", color: "#fff", bgcolor: "#e65100", px: 0.5, py: 0.125, ml: 0.5, verticalAlign: "middle" }}>
                {diffMode === "previous" ? "\u0394-PREV" : "\u0394-FIRST"}
              </Typography>
            )}
	            {showControls && <InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <MetadataSection rows={[
                ["Shape", `${nSlices} x ${height} x ${width}`],
                ["Panels", nPanels > 1 ? `${nPanels} panels` : "single panel"],
                ["Frame axis", `${dimLabel || "Frame"}${dimSampling ? `, ${formatNumber(dimSampling)} ${dimUnit || ""}` : ""}`],
                ["Sampling", pixelSize > 0 ? `${formatNumber(pixelSize)} ${unitSymbol(pixelUnit || "px")}/px` : ""],
                ["Source", hasFrameServer ? "detail server" : "embedded stack"],
              ]} />
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum (Fourier transform) alongside image.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                FFT d-spacing uses the provided real-space sampling: Δk = 1 / (N × pixel_size), |g| = √(kx² + ky²), d = 1 / |g|. Current pixel_size: {pixelSize > 0 ? `${formatNumber(pixelSize)} ${unitSymbol(pixelUnit || "px")}/px` : "not set, so only pixel distances are shown"}.
              </Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Backend</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                WebGPU: {webgpuStatusLabel}{fftBackendInfo.adapter ? ` (${fftBackendInfo.adapter})` : ""}.
              </Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                FFT compute: {fftSourceLabel}{fftSourceDetail && fftSourceDetail !== fftSourceLabel ? ` (${fftSourceDetail})` : ""}
                {fftBackendInfo.ms != null ? `, ${fftBackendInfo.ms.toFixed(1)} ms` : ""}.
              </Typography>
              {fftBackendInfo.panels != null && (
                <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                  FFT panels: {fftBackendInfo.panels}{fftBackendInfo.grid ? `, grid ${fftBackendInfo.grid}` : ""}.
                </Typography>
              )}
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on image to draw a line intensity profile.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Lens: Magnifier inset that follows the cursor.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Scale: Linear or logarithmic intensity mapping.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Stack-wide percentile contrast for Show3D image panels. FFT Auto masks DC + clips to 99.9th.</Typography>
              {roiAllowed && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>ROI: Click empty image to add at cursor, click ROI to select, drag to move, hover edge to resize. Del removes selected; Ctrl/⌘+D duplicates.</Typography>}
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Cols / Panels: Change the panel grid or hide panels without changing the source stack.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Pinning: Click a panel to select or pin it for keyboard actions, per-panel zoom, ROI edits, and deletion shortcuts.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Pan: With Pan enabled, drag the image to move the zoomed view. With Link Zoom on, pan and zoom move together across panels.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop: Loop playback. Drag end markers on slider for loop range.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Bounce: Ping-pong playback - alternates forward and reverse.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Speed: Lower fps, increase avg only when needed, shorten the loop range, hide panels, or turn off FFT/Profile/Stats to reduce heavy-stack playback work.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT layout: Use side, bottom, or overlay mode. Overlay FFTs can be resized; wheel and drag over the overlay inspect FFT detail independently.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Export / Copy: Export HTML, GIF, or MP4 panel-only animations, or copy the current panel view from the toolbar.</Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
              <KeyboardShortcuts items={keyboardShortcutItems} />
	            </Box>} theme={themeInfo.theme} />}
	            {showControls && (
	              <Button
	                size="small"
	                sx={{
	                  ...compactButton,
	                  ml: 0.75,
	                  py: 0,
	                  px: 0.5,
	                  minHeight: 16,
	                  lineHeight: "16px",
	                  verticalAlign: "baseline",
	                }}
	                onClick={() => setControlsCollapsed(!controlsCollapsed)}
	                aria-label={controlsCollapsed ? "Show controls" : "Hide controls"}
	                aria-pressed={!controlsCollapsed}
	                title={controlsCollapsed ? "Show controls" : "Hide controls"}
	              >
	                Controls
	              </Button>
	            )}
	          </Typography>}
	          {/* Controls row */}
	          {controlsVisible && (
	          <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: "4px", mb: `${SPACING.XS}px`, minHeight: 28 }}>
            {isPaged && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px", flexShrink: 0 }}>Page</Typography>
                <Typography
                  title={currentPageStatus}
                  sx={{
                    ...typography.label,
                    fontSize: 10,
                    color: themeColors.accent,
                    flex: "0 1 14ch",
                    minWidth: "8ch",
                    maxWidth: { xs: "11ch", sm: "16ch", md: "20ch" },
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {currentPageStatus}
                </Typography>
                <Slider
                  value={currentPageIdx}
                  min={0}
                  max={Math.max(0, (nPages || 1) - 1)}
                  step={1}
                  onPointerDownCapture={() => setPagePlaying(false)}
                  onKeyDown={() => setPagePlaying(false)}
                  onChange={(_, value) => {
                    const raw = Array.isArray(value) ? value[0] : value;
                    setPagePlaying(false);
                    setPageIdx(Math.max(0, Math.min((nPages || 1) - 1, Math.round(Number(raw) || 0))));
                  }}
                  onChangeCommitted={() => setPagePlaying(false)}
                  size="small"
                  sx={{ ...sliderStyles.small, width: 120, flex: "0 0 120px", color: themeColors.accent, ml: "2px" }}
                  aria-label="Page"
                />
                <IconButton
                  size="small"
                  onClick={() => setPagePlaying((value) => !value)}
                  title={pagePlaying ? "Pause page playback" : "Play pages"}
                  aria-label={pagePlaying ? "Pause page playback" : "Play pages"}
                  sx={{ width: 24, height: 24, p: 0, color: themeColors.accent }}
                >
                  {pagePlaying ? <PauseIcon sx={{ fontSize: 16 }} /> : <PlayArrowIcon sx={{ fontSize: 16 }} />}
                </IconButton>
                <Select
                  value={String(pagePlayFps)}
                  onChange={(e) => setPagePlayFps(Number(e.target.value) || 2)}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 48, fontSize: 10 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Page playback frames per second" }}
                  title="Page playback speed"
                >
                  {PAGE_PLAY_FPS_OPTIONS.map((fps) => (
                    <MenuItem key={fps} value={String(fps)}>{fps} fps</MenuItem>
                  ))}
                </Select>
                <IconButton
                  size="small"
                  onClick={() => {
                    const next = Array.from({ length: Math.max(1, nPages || 1) }, (_, idx) => pageStarred?.[idx] ? 1 : 0);
                    next[currentPageIdx] = next[currentPageIdx] ? 0 : 1;
                    setPageStarred(next);
                  }}
                  title={(pageStarred?.[currentPageIdx] ? "Unstar " : "Star ") + currentPageLabel}
                  aria-label={(pageStarred?.[currentPageIdx] ? "Unstar " : "Star ") + currentPageLabel}
                  sx={{
                    width: 24,
                    height: 24,
                    p: 0,
                    color: pageStarred?.[currentPageIdx] ? "#ffc107" : themeColors.textMuted,
                    "&:hover": { color: pageStarred?.[currentPageIdx] ? "#ffc107" : themeColors.text },
                  }}
                >
                  {pageStarred?.[currentPageIdx] ? "★" : "☆"}
                </IconButton>
              </>
            )}
            {visiblePanelCount > 1 && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Cols</Typography>
                <Select
                  value={String(clampedMaxCols)}
                  onChange={(e) => {
                    const next = Math.max(1, Math.min(Number(e.target.value) || 1, visiblePanelCount || 1, MAX_PANEL_COLUMNS));
                    setMaxCols(next);
                  }}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 48, fontSize: 10 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Show3D panel columns" }}
                  title="Number of Show3D panel columns"
                >
                  {show3dColumnOptions.map((cols) => (
                    <MenuItem key={cols} value={String(cols)}>{cols}</MenuItem>
                  ))}
                </Select>
              </>
            )}
            {/* Kymograph toggle: HIDDEN until a profile line exists (not shown-
                but-disabled). Kymograph is a line-profile sub-feature, so the
                control only appears once there's a line to build it from. */}
            {/* Kymograph: appears only with a drawn profile line (canKymograph).
                Turning it on takes the side slot from FFT. */}
            {canKymograph && <>
              <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Kymo</Typography>
              <Switch checked={showKymograph} onChange={(e) => { const on = e.target.checked; setShowKymograph(on); if (on) setShowFft(false); }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle kymograph space-time panel" } }} />
            </>}
            {/* Profile and ROI are mutually exclusive line/region tools. Turning
                one on turns the other off. Kymograph rides on Profile. */}
            <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Profile</Typography>
            <Switch checked={profileActive} onChange={(e) => {
              const on = e.target.checked;
              setProfileActive(on);
              if (on) {
                setRoiActive(false); setRoiSelectedIdx(-1);
              } else {
                // Toggle OFF hides overlay + kymograph but keeps the line + data
                // so re-enable restores instantly. Use Clear to actively wipe.
                setShowKymograph(false);
                setHoveredProfileEndpoint(null); setIsHoveringProfileLine(false);
              }
            }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle line intensity profile tool" } }} />
            {profileActive && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "4px" }}>W</Typography>
                <Slider value={profileWidth} min={1} max={15} step={1} onChange={(_, v) => setProfileWidth(v as number)} size="small" valueLabelDisplay="auto" sx={{ width: 60, ml: "2px" }} aria-label={`Profile width ${profileWidth} px`} />
              </>
            )}
            {(nPanels || 1) === 1 && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Lens</Typography>
                <Switch
                  checked={showLens}
                  onChange={() => {
                    if (!showLens) { setShowLens(true); setLensPos({ row: Math.floor(height / 2), col: Math.floor(width / 2) }); }
                    else { setShowLens(false); setLensPos(null); }
                  }}
                  size="small"
                  sx={switchStyles.small}
                  slotProps={{ input: { "aria-label": "Toggle magnifier lens" } }}
                />
              </>
            )}
            {/* ROI hidden while kymograph is shown (roiAllowed already encodes
                single-panel && !showKymograph). */}
            {roiAllowed && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>ROI</Typography>
                <Switch checked={roiActive} onChange={(e) => {
                  const on = e.target.checked;
                  if (on) {
                    setRoiActive(true); setShowRoiResizeHint(true);
                    setProfileActive(false); setProfileLine([]); setProfileData(null); setHoveredProfileEndpoint(null); setIsHoveringProfileLine(false);
                  } else {
                    setRoiActive(false); setRoiSelectedIdx(-1); pendingRoiAddRef.current = null;
                  }
                }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle ROI selection tool" } }} />
              </>
            )}
            <>
              <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Stats</Typography>
              <Switch checked={showStats} onChange={(e) => setShowStats(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle statistics readout" } }} />
            </>
            {(nPanels || 1) > 1 && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Link</Typography>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Zoom</Typography>
                <Switch checked={linkPanels} onChange={(e) => setLinkPanels(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Link zoom and pan across panels" } }} />
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Contrast</Typography>
                <Switch checked={linkContrast} onChange={(e) => setLinkContrast(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Link contrast across panels" } }} />
              </>
            )}
            {fftAllowed && (
              <Box aria-hidden="true" sx={{ width: "1px", height: 20, flex: "0 0 1px", alignSelf: "center", mx: "4px", bgcolor: themeColors.border, opacity: 0.8 }} />
            )}
            {/* FFT can be shown below, beside, or as an inset over the image grid. */}
            {fftAllowed && <>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT</Typography>
              <Switch checked={showFft} onChange={(e) => { const on = e.target.checked; setShowFft(on); if (on) setShowKymograph(false); }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle FFT power spectrum panel" } }} />
              {showFft && (
                <Select
                  value={resolvedFftLayout}
                  onChange={(e) => setFftLayout(String(e.target.value))}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 78, fontSize: 10, ml: "2px" }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "FFT panel layout" }}
                >
                  <MenuItem value="bottom">Bottom</MenuItem>
                  <MenuItem value="right">Right</MenuItem>
                  <MenuItem value="overlay">Overlay</MenuItem>
                </Select>
              )}
              {showFft && fftLayoutOverlay && (
                <>
                  <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Size</Typography>
                  <Select
                    value={String(Math.round(resolvedFftOverlaySize * 100))}
                    onChange={(e) => setFftOverlaySize(Number(e.target.value) / 100)}
                    size="small"
                    sx={{ ...themedSelect, minWidth: 52, fontSize: 10, ml: "2px" }}
                    MenuProps={themedMenuProps}
                    inputProps={{ "aria-label": "FFT overlay size" }}
                  >
                    <MenuItem value="25">25%</MenuItem>
                    <MenuItem value="35">35%</MenuItem>
                    <MenuItem value="50">50%</MenuItem>
                    <MenuItem value="65">65%</MenuItem>
                  </Select>
                </>
              )}
            </>}
            <Box sx={{ flex: 1 }} />
            <Box sx={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <Button size="small" sx={compactButton} onClick={handleCopy} aria-label="Copy current frame to clipboard as PNG">Copy</Button>
              {(nPanels || 1) > 1 && (
                <>
                  {!isPaged && (
                    <Button
                      size="small"
                      sx={{
                        ...compactButton,
                        color: reorderMode ? themeColors.accent : themeColors.text,
                        "& .MuiButton-startIcon": { mr: 0.4 },
                      }}
                      startIcon={<DragIndicatorIcon sx={{ fontSize: 14 }} />}
                      onClick={() => setReorderMode((value) => !value)}
                      aria-pressed={reorderMode ? "true" : "false"}
                      aria-label={reorderMode ? "Finish reordering panels" : "Reorder panels"}
                      title={reorderMode ? "Finish reordering panels" : "Reorder panels"}
                    >
                      Reorder
                    </Button>
                  )}
                  <Button
                    size="small"
                    sx={{ ...compactButton, "& .MuiButton-startIcon": { mr: 0.4 } }}
                    startIcon={<VisibilityIcon sx={{ fontSize: 14 }} />}
                    onClick={(event) => setPanelMenuAnchor(event.currentTarget)}
                    aria-label="Choose visible panels"
                    aria-controls={panelMenuAnchor ? "show3d-panels-menu" : undefined}
                    aria-expanded={panelMenuAnchor ? "true" : undefined}
                    aria-haspopup="menu"
                  >
                    {visiblePanelCount === panelMenuTotal ? "Panels" : `Panels ${visiblePanelCount}/${panelMenuTotal}`}
                  </Button>
                  <Menu
                    id="show3d-panels-menu"
                    anchorEl={panelMenuAnchor}
                    open={Boolean(panelMenuAnchor)}
                    onClose={() => setPanelMenuAnchor(null)}
                    MenuListProps={{ "aria-label": "Panel visibility options" }}
                    {...themedMenuProps}
                  >
                    {orderedPanelIndices.map((panel) => {
                      const hidden = hiddenPanelSet.has(panel);
                      const disabled = !hidden && visiblePanelCount <= 1;
                      return (
                        <MenuItem
                          key={`panel-menu-${panel}`}
                          dense
                          disabled={disabled}
                          onClick={() => setPanelHidden(panel, !hidden)}
                          title={disabled ? "At least one panel must remain visible" : undefined}
                        >
                          {hidden
                            ? <VisibilityOffIcon sx={{ fontSize: 16, mr: 1, color: themeColors.textMuted }} />
                            : <VisibilityIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />}
                          <Typography sx={{ fontSize: 11, color: disabled ? themeColors.textMuted : themeColors.text }}>
                            {panelLabel(panel)}
                          </Typography>
                        </MenuItem>
                      );
                    })}
                    <MenuItem
                      dense
                      disabled={hiddenPanelSet.size === 0}
                      onClick={() => {
                        if (isPaged) setHiddenPageSlots([]);
                        setHiddenPanels([]);
                      }}
                    >
                      <VisibilityIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />
                      <Typography sx={{ fontSize: 11 }}>Show all panels</Typography>
                    </MenuItem>
                    {!isPaged && (
                      <MenuItem
                        dense
                        disabled={(panelOrder || []).length === 0}
                        onClick={resetPanelOrder}
                      >
                        <DragIndicatorIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />
                        <Typography sx={{ fontSize: 11 }}>Reset order</Typography>
                      </MenuItem>
                    )}
                  </Menu>
                </>
              )}
              {handoffEnabled && (
                <>
                  <Button
                    size="small"
                    sx={compactButton}
                    onClick={(event) => setViewMenuAnchor(event.currentTarget)}
                    aria-label="Open view options"
                    aria-controls={viewMenuAnchor ? "show3d-view-menu" : undefined}
                    aria-expanded={viewMenuAnchor ? "true" : undefined}
                    aria-haspopup="menu"
                    title={handoffStatus || "View options"}
                  >
                    View
                  </Button>
                  <Menu
                    id="show3d-view-menu"
                    anchorEl={viewMenuAnchor}
                    open={Boolean(viewMenuAnchor)}
                    onClose={() => setViewMenuAnchor(null)}
                    MenuListProps={{ "aria-label": "View options" }}
                    {...themedMenuProps}
                  >
                    <MenuItem onClick={handleHandoffToShow2D}>
                      View frame as 2D
                    </MenuItem>
                  </Menu>
                </>
              )}
              {exportEnabled && (
                <>
                  <Button
                    size="small"
                    sx={compactButton}
                    disabled={exportBusy}
                    onClick={handleExportMenuOpen}
                    aria-label="Export widget or animation"
                    aria-controls={exportMenuAnchor ? "show3d-export-menu" : undefined}
                    aria-expanded={exportMenuAnchor ? "true" : undefined}
                    aria-haspopup="menu"
                    title={localExportStatus || exportStatus || "Export HTML, GIF, or MP4 with a save dialog"}
                  >
                    {exportBusy ? "Exporting" : "Export"}
                  </Button>
                  <Menu
                    id="show3d-export-menu"
                    anchorEl={exportMenuAnchor}
                    open={Boolean(exportMenuAnchor)}
                    onClose={handleExportMenuClose}
                    MenuListProps={{ "aria-label": "Export options" }}
                    {...themedMenuProps}
                  >
                    <MenuItem onClick={() => handleExportSelect("exact")}>HTML exact float32 ({exactExportSize})</MenuItem>
                    <MenuItem onClick={() => handleExportSelect("quantized")}>HTML quantized uint8 ({quantizedExportSize})</MenuItem>
                    {height >= 2 && width >= 2 && <MenuItem onClick={() => handleExportSelect("quantized", "medium", 2)}>HTML quantized uint8, 2× binned ({quantizedExportSize2})</MenuItem>}
                    {height >= 4 && width >= 4 && <MenuItem onClick={() => handleExportSelect("quantized", "medium", 4)}>HTML quantized uint8, 4× binned ({quantizedExportSize4})</MenuItem>}
                    {height >= 8 && width >= 8 && <MenuItem onClick={() => handleExportSelect("quantized", "medium", 8)}>HTML quantized uint8, 8× binned ({quantizedExportSize8})</MenuItem>}
                    <MenuItem onClick={() => handleExportSelect("gif", "low")}>GIF low ({gifLowEstimate})</MenuItem>
                    <MenuItem onClick={() => handleExportSelect("gif", "medium")}>GIF medium ({gifMediumEstimate})</MenuItem>
                    <MenuItem onClick={() => handleExportSelect("gif", "high")}>GIF high ({gifHighEstimate})</MenuItem>
                    <MenuItem onClick={() => handleExportSelect("mp4", "low")}>MP4 low ({gifLowEstimate})</MenuItem>
                    <MenuItem onClick={() => handleExportSelect("mp4", "medium")}>MP4 medium ({gifMediumEstimate})</MenuItem>
                    <MenuItem onClick={() => handleExportSelect("mp4", "high")}>MP4 high ({gifHighEstimate})</MenuItem>
                  </Menu>
                </>
              )}
              {exportEnabled && (localExportStatus || exportStatus) && (
                <Typography
                  sx={{
                    ...typography.label,
                    fontSize: 10,
                    maxWidth: 120,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: (localExportStatus || exportStatus).startsWith("Export failed") ? "#d32f2f" : themeColors.textMuted,
                  }}
                  title={localExportStatus || exportStatus}
                >
                  {localExportStatus || exportStatus}
                </Typography>
              )}
              <Button size="small" sx={compactButton} disabled={!needsReset} onClick={handleDoubleClick} aria-label="Reset zoom and pan">Reset</Button>
              {handoffEnabled && handoffStatus && (
                <Typography
                  sx={{
                    ...typography.label,
                    fontSize: 10,
                    maxWidth: 140,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: handoffStatus.startsWith("View failed") ? "#d32f2f" : themeColors.textMuted,
                  }}
                  title={handoffStatus}
                >
                  {handoffStatus}
                </Typography>
              )}
	          </Box>
	          </Box>
	          )}
          <Box
            ref={canvasContainerRef}
            sx={{
              ...container.imageBox,
              width: "100%",
              maxWidth: canvasW,
              aspectRatio: mainPanelAspectRatio,
              height: "auto",
              overscrollBehavior: "contain",
              touchAction: "none",
              "&:hover .show3d-panel-hide-button, &:focus-within .show3d-panel-hide-button": {
                opacity: 1,
                pointerEvents: "auto",
                transform: "translateY(0)",
              },
              "@media (hover: none), (pointer: coarse)": {
                "& .show3d-panel-hide-button": {
                  display: "none",
                },
              },
              ...(reorderMode ? {
                "@keyframes show3d-reorder-jiggle": {
                  "0%": { rotate: "-0.45deg" },
                  "100%": { rotate: "0.45deg" },
                },
              } : {}),
              cursor: reorderMode
                ? "grab"
                : isHoveringLensEdge
                ? "nwse-resize"
                : (isHoveringResize || isDraggingResize || isHoveringResizeInner || isDraggingResizeInner)
                  ? "nwse-resize"
                  : (draggingProfileEndpoint !== null || isDraggingProfileLine)
                    ? "grabbing"
                    : (profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine))
                      ? "grab"
                      : (effectiveRoiActive || profileActive)
                        ? "crosshair"
                        : "grab",
            }}
            onMouseDown={reorderMode ? undefined : handleCanvasMouseDown}
            onMouseMove={reorderMode ? undefined : handleCanvasMouseMove}
            onMouseUp={reorderMode ? undefined : handleCanvasMouseUp}
            onMouseLeave={reorderMode ? undefined : handleCanvasMouseLeave}
            onDoubleClick={reorderMode ? undefined : handleDoubleClick}
          >
            <canvas
              ref={canvasRef}
              width={canvasW}
              height={canvasH}
              onTouchStart={reorderMode ? undefined : handleCanvasTouchStart}
              onTouchMove={reorderMode ? undefined : handleCanvasTouchMove}
              onTouchEnd={reorderMode ? undefined : handleCanvasTouchEnd}
              onTouchCancel={reorderMode ? undefined : handleCanvasTouchEnd}
              style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", opacity: gpuDisplayVisible ? 0 : 1, display: "block", touchAction: "none" }}
              role="img"
              aria-label={`Slice image ${visibleSliceIdx + 1} of ${nSlices}${title ? `: ${title}` : ""} (${width} by ${height} pixels). Use arrow keys to scrub frames.`}
            />
            <canvas ref={gpuCanvasRef} width={canvasW} height={canvasH} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", pointerEvents: "none", opacity: gpuDisplayVisible ? 1 : 0 }} aria-hidden="true" />
            <canvas ref={overlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", display: overlayCanvasVisible ? "block" : "none" }} aria-hidden="true" />
            <canvas ref={uiRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
            <canvas ref={lensCanvasRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", display: lensCanvasVisible ? "block" : "none" }} aria-hidden="true" />
            {effectiveShowFft && fftLayoutOverlay && (
              <canvas
                ref={fftInsetLayerRef}
                width={canvasW}
                height={canvasH}
                style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", pointerEvents: "none", zIndex: 7 }}
                aria-hidden="true"
              />
            )}
            {showPanelTitles !== false && (nPanels || 1) > 1 && visiblePanelIndices.map((panel, slot) => {
              const titleText = panelTitles?.[panel];
              if (!titleText) return null;
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const shownIdx = visibleSliceIdx;
              const realN = panelRealFrames?.[panel];
              const shown = realN ? Math.min(shownIdx + 1, realN) : shownIdx + 1;
              const total = realN || nSlices;
              const frameLabel = panelFrameLabelForIndex(panel, shownIdx);
              return (
                <Box
                  key={`panel-title-${panel}`}
                  sx={{
                    position: "absolute",
                    top: `${((panelTop + 6) / Math.max(1, canvasH)) * 100}%`,
                    left: `${(panelLeft / Math.max(1, canvasW)) * 100}%`,
                    width: `${(panelW / Math.max(1, canvasW)) * 100}%`,
                    px: 1,
                    boxSizing: "border-box",
                    color: "rgba(255, 255, 255, 0.95)",
                    fontFamily: UI_FONT,
                    fontSize: Math.max(8, panelTitleFontSize || 11),
                    fontWeight: 700,
                    lineHeight: 1.2,
                    textAlign: "center",
                    textShadow: "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px rgba(0,0,0,0.75)",
                    pointerEvents: "none",
                    userSelect: "none",
                    zIndex: 2,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {titleText}{frameLabel ? ` · ${frameLabel}` : ""} {shown}/{total}
                </Box>
              );
            })}
            {/* Per-panel "best frame" stars. One gold ★ button top-right of
                each panel. Click toggles the star on the currently displayed
                slice for THAT panel. Programmatic API: widget.star_panel(i). */}
	            {panelChromeVisible && visiblePanelIndices.map((i, slot) => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (Math.ceil(n / cols) - 1)) / Math.ceil(n / cols);
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const starredFrame = starred?.[i] ?? -1;
              const isStarredHere = starredFrame === visibleSliceIdx;
              const starElsewhere = starredFrame >= 0 && !isStarredHere;
              const tooltip = isStarredHere
                ? `★ Starred. Click to unstar frame ${visibleSliceIdx + 1}.`
                : starElsewhere
                  ? `Star is on frame ${starredFrame + 1}. Click to move it to frame ${visibleSliceIdx + 1}.`
                  : `Click to mark frame ${visibleSliceIdx + 1} as best for ${panelLabel(i)}.`;
              return (
                <button
                  key={`star-${i}`}
                  onMouseDown={(event) => event.stopPropagation()}
                  onClick={() => {
                    const cur = Array.from({ length: totalPanelCount }, (_, k) => starred?.[k] ?? -1);
                    cur[i] = isStarredHere ? -1 : visibleSliceIdx;
                    setStarred(cur);
                  }}
                  title={tooltip}
                  aria-label={tooltip}
                  style={{
                    position: "absolute",
                    top: `${((panelTop + 6) / Math.max(1, canvasH)) * 100}%`,
                    left: `calc(${((panelLeft + panelW) / Math.max(1, canvasW)) * 100}% - 26px)`,
                    width: 20, height: 20,
                    padding: 0,
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                    fontSize: 18,
                    lineHeight: "20px",
                    textAlign: "center",
                    color: isStarredHere
                      ? "#ffc107"  // bright gold: star IS on this frame
                      : starElsewhere
                        ? "rgba(255, 193, 7, 0.45)"  // faded gold: star elsewhere on this panel
                        : "rgba(255,255,255,0.5)",   // grey: no star on this panel
                    textShadow: "0 0 3px rgba(0,0,0,0.8)",
                    pointerEvents: "auto",
                    userSelect: "none",
                  }}
                >
                  {isStarredHere ? "★" : "☆"}
                </button>
              );
            })}
            {panelChromeVisible && (nPanels || 1) > 1 && visiblePanelIndices.map((panel, slot) => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (Math.ceil(n / cols) - 1)) / Math.ceil(n / cols);
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const disabled = visiblePanelCount <= 1;
              const label = disabled ? `Cannot hide the last visible panel` : `Hide ${panelLabel(panel)}`;
              return (
                <IconButton
                  key={`panel-hide-${panel}`}
                  className="show3d-panel-hide-button"
                  size="small"
                  disabled={disabled}
                  onMouseDown={(event) => event.stopPropagation()}
                  onClick={(event) => {
                    event.stopPropagation();
                    setPanelHidden(panel, true);
                  }}
                  aria-label={label}
                  title={label}
                  sx={{
                    position: "absolute",
                    top: `${((panelTop + 5) / Math.max(1, canvasH)) * 100}%`,
                    left: `${((panelLeft + 5) / Math.max(1, canvasW)) * 100}%`,
                    width: 22,
                    height: 22,
                    p: 0,
                    opacity: 0,
                    transform: "translateY(-3px)",
                    transition: "opacity 120ms ease, transform 120ms ease, background-color 120ms ease, color 120ms ease",
                    color: disabled ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.75)",
                    bgcolor: "rgba(0,0,0,0.22)",
                    pointerEvents: "none",
                    "&:hover, &:focus-visible": {
                      bgcolor: "rgba(0,0,0,0.42)",
                      color: "rgba(255,255,255,0.95)",
                    },
                  }}
                >
                  <VisibilityOffIcon sx={{ fontSize: 15 }} />
                </IconButton>
              );
            })}
            {panelChromeVisible && reorderMode && (nPanels || 1) > 1 && visiblePanelIndices.map((panel, slot) => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const active = dragOverPanel === panel;
              const draggingThisPanel = reorderDragVisual?.panel === panel;
              return (
                <Box
                  key={`panel-reorder-${panel}`}
                  draggable={reorderMode}
                  role="button"
                  data-show3d-reorder-panel={panel}
                  aria-label={`Move ${panelLabel(panel)}`}
                  title={`Drag to reorder ${panelLabel(panel)}`}
                  onDragStart={(event) => handlePanelDragStart(event, panel)}
                  onDragOver={(event) => handlePanelDragOver(event, panel)}
                  onDrop={handlePanelDrop}
                  onDragEnd={handlePanelDragEnd}
                  onPointerDown={(event) => handlePanelReorderPointerDown(event, panel)}
                  onPointerEnter={(event) => handlePanelReorderPointerEnter(event, panel)}
                  onPointerMove={handlePanelReorderPointerMove}
                  onPointerUp={handlePanelReorderPointerUp}
                  onPointerCancel={cancelPanelReorderPreview}
                  sx={{
                    position: "absolute",
                    top: `${(panelTop / Math.max(1, canvasH)) * 100}%`,
                    left: `${(panelLeft / Math.max(1, canvasW)) * 100}%`,
                    width: `${(panelW / Math.max(1, canvasW)) * 100}%`,
                    height: `${(panelH / Math.max(1, canvasH)) * 100}%`,
                    boxSizing: "border-box",
                    border: `2px solid ${active ? themeColors.accent : "rgba(255,255,255,0.48)"}`,
                    bgcolor: draggingThisPanel ? "rgba(0,0,0,0.28)" : active ? "rgba(79, 195, 247, 0.16)" : "rgba(0,0,0,0.04)",
                    outline: active ? `1px solid ${themeColors.accent}` : "none",
                    opacity: draggingThisPanel ? 0.38 : 1,
                    transform: active ? "translateY(-3px) scale(1.006)" : "translateY(0) scale(1)",
                    transition: "transform 110ms ease, opacity 110ms ease, background-color 110ms ease, border-color 110ms ease, box-shadow 110ms ease",
                    animation: "show3d-reorder-jiggle 220ms ease-in-out infinite alternate",
                    boxShadow: active ? `0 0 0 2px ${themeColors.accent}, 0 8px 18px rgba(0,0,0,0.20)` : "none",
                    cursor: draggedPanelRef.current === panel ? "grabbing" : "grab",
                    pointerEvents: "auto",
                    zIndex: 8,
                  }}
                >
                  <Box
                    sx={{
                      position: "absolute",
                      bottom: 6,
                      left: "50%",
                      transform: "translateX(-50%)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      width: 30,
                      height: 22,
                      borderRadius: 1,
                      bgcolor: "rgba(0,0,0,0.38)",
                      color: "rgba(255,255,255,0.92)",
                      pointerEvents: "none",
                    }}
                  >
                    <DragIndicatorIcon sx={{ fontSize: 18 }} />
                  </Box>
                </Box>
              );
            })}
            {panelChromeVisible && reorderMode && reorderDragVisual && (
              <Box
                ref={reorderGhostRef}
                data-show3d-reorder-ghost={reorderDragVisual.panel}
                aria-hidden="true"
                sx={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: `${reorderDragVisual.width}px`,
                  height: `${reorderDragVisual.height}px`,
                  transform: `translate3d(${reorderDragVisual.x}px, ${reorderDragVisual.y}px, 0)`,
                  boxSizing: "border-box",
                  overflow: "hidden",
                  border: `2px solid ${themeColors.accent}`,
                  bgcolor: reorderDragVisual.imageUrl ? "rgba(0,0,0,0.04)" : "rgba(25,25,25,0.68)",
                  boxShadow: `0 10px 24px rgba(0,0,0,0.32), 0 0 0 1px ${themeColors.accent}`,
                  opacity: 0.9,
                  pointerEvents: "none",
                  zIndex: 12,
                  willChange: "transform",
                }}
              >
                {reorderDragVisual.imageUrl && (
                  <Box
                    sx={{
                      position: "absolute",
                      inset: 0,
                      backgroundImage: `url(${reorderDragVisual.imageUrl})`,
                      backgroundSize: "100% 100%",
                      backgroundPosition: "center",
                      imageRendering: smooth ? "auto" : "pixelated",
                    }}
                  />
                )}
                <Box
                  sx={{
                    position: "absolute",
                    top: 6,
                    left: 8,
                    right: 8,
                    px: 0.75,
                    py: 0.25,
                    borderRadius: 0.75,
                    bgcolor: "rgba(0,0,0,0.48)",
                    color: "rgba(255,255,255,0.96)",
                    fontFamily: UI_FONT,
                    fontSize: Math.max(8, panelTitleFontSize || 11),
                    fontWeight: 700,
                    lineHeight: 1.2,
                    textAlign: "center",
                    textShadow: "0 1px 2px rgba(0,0,0,0.9)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {reorderDragVisual.label}
                </Box>
                <Box
                  sx={{
                    position: "absolute",
                    bottom: 8,
                    left: "50%",
                    transform: "translateX(-50%)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: 34,
                    height: 24,
                    borderRadius: 1,
                    bgcolor: "rgba(0,0,0,0.48)",
                    color: "rgba(255,255,255,0.95)",
                  }}
                >
                  <DragIndicatorIcon sx={{ fontSize: 19 }} />
                </Box>
              </Box>
            )}
            {/* Zoom indicator now drawn on the ui canvas in the scale-bar
                pass (Show2D-matching style: white, sans, Unicode ×). */}
            {/* Cursor readout overlay */}
	            {panelChromeVisible && cursorInfo && (() => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const slot = visiblePanelIndices.indexOf(cursorInfo.panelIdx);
              if (slot < 0) return null;
              const col = slot % cols;
              const row = Math.floor(slot / cols);
              const panelLeft = col * (panelW + gap);
              const panelTop = row * (panelH + gap);
              return (
                <Box className="show3d-cursor-readout" sx={{
                  position: "absolute",
                  top: `${((panelTop + 3) / Math.max(1, canvasH)) * 100}%`,
                  right: `calc(${((canvasW - (panelLeft + panelW)) / Math.max(1, canvasW)) * 100}% + 3px)`,
                  bgcolor: "rgba(0,0,0,0.35)",
                  px: 0.5,
                  py: 0.15,
                  opacity: cursorReadoutVisible ? 1 : 0,
                  transform: cursorReadoutVisible ? "translateY(0)" : "translateY(-2px)",
                  transition: "opacity 90ms ease, transform 90ms ease",
                  willChange: "opacity, transform",
                  pointerEvents: "none",
                  minWidth: 78,
                  maxWidth: `calc(${(panelW / Math.max(1, canvasW)) * 100}% - 6px)`,
                  textAlign: "right",
                }}>
                  <Typography sx={{ fontSize: 9, fontFamily: "monospace", fontVariantNumeric: "tabular-nums", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2, overflow: "hidden", textOverflow: "ellipsis" }}>
                    ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                  </Typography>
                </Box>
              );
            })()}
	            {panelChromeVisible && effectiveRoiActive && roiItems.length > 0 && showRoiResizeHint && (
              <Box sx={{ position: "absolute", left: 6, top: 6, px: 0.6, py: 0.25, bgcolor: "rgba(0,0,0,0.45)", pointerEvents: "none" }}>
                <Typography sx={{ fontSize: 9, color: "rgba(255,255,255,0.8)", lineHeight: 1.1 }}>
                  Hover ROI edge to resize
                </Typography>
              </Box>
            )}
            {/* Per-panel resize corner. Empty cells (partial last row) get
                no handle. Each handle scales the whole multi-panel canvas
                (linked behavior). Match Show2D gallery: 16x16, grey, 0.6/1.0.
                User trait `show_resize_handles` toggles visibility. */}
            {showResizeControls && (() => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const outPanelW = (canvasW - gap * (cols - 1)) / cols;
              const outPanelH = (canvasH - gap * (rows - 1)) / rows;
              return visiblePanelIndices.map((panel, slot) => {
                const col = slot % cols;
                const row = Math.floor(slot / cols);
                const slotX = col * (outPanelW + gap);
                const slotY = row * (outPanelH + gap);
                return (
                  <Box
                    key={`resize-${panel}`}
                    onMouseDown={handleMainResizeStart}
                    sx={{
                      position: "absolute",
                      left: `calc(${((slotX + outPanelW) / Math.max(1, canvasW)) * 100}% - 16px)`,
                      top: `calc(${((slotY + outPanelH) / Math.max(1, canvasH)) * 100}% - 16px)`,
                      width: 16,
                      height: 16,
                      cursor: "nwse-resize",
                      opacity: 0.6,
                      background: `linear-gradient(135deg, transparent 50%, ${themeColors.border} 50%)`,
                      borderRadius: "0 0 4px 0",
                      "&:hover": { opacity: 1 },
                    }}
                  />
                );
              });
            })()}
            {effectiveShowFft && fftLayoutOverlay && (() => {
              const n = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(n);
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              return visiblePanelIndices.map((panel, slot) => {
                const panelLeft = (slot % cols) * (panelW + gap);
                const panelTop = Math.floor(slot / cols) * (panelH + gap);
                const insetPad = Math.min(8, Math.max(3, panelW * 0.025));
                const insetMaxW = Math.max(24, panelW - insetPad * 2);
                const insetMaxH = Math.max(20, panelH - insetPad * 2);
                const insetBase = Math.min(insetMaxW, insetMaxH);
                const insetW = Math.max(24, Math.min(insetMaxW, insetBase * resolvedFftOverlaySize));
                const insetH = Math.max(20, Math.min(insetMaxH, insetBase * resolvedFftOverlaySize));
                const insetX = resolvedFftOverlayPosition.endsWith("right")
                  ? panelLeft + panelW - insetW - insetPad
                  : panelLeft + insetPad;
                const insetY = resolvedFftOverlayPosition.startsWith("bottom")
                  ? panelTop + panelH - insetH - insetPad
                  : panelTop + insetPad;
                const previewInsetX = fftOverlayDragPreview ? panelLeft + fftOverlayDragPreview.x : insetX;
                const previewInsetY = fftOverlayDragPreview ? panelTop + fftOverlayDragPreview.y : insetY;
                return (
                  <Box
                    key={`fft-overlay-inset-${panel}`}
                    data-show3d-fft-inset="true"
                    title="Drag to move FFT overlay; Shift-drag to pan FFT detail"
                    onWheel={handleFftInsetWheel}
                    onMouseDown={(e) => {
                      if (e.shiftKey) {
                        handleFftInsetPanMouseDown(e);
                      } else {
                        handleFftInsetMouseDown(e, panelLeft, panelTop, panelW, panelH, insetX, insetY, insetW, insetH);
                      }
                    }}
                    onDoubleClick={(e) => { e.preventDefault(); e.stopPropagation(); handleFftReset(); }}
                    role="img"
                    aria-label={`FFT power spectrum overlay for ${panelLabel(panel)}`}
                    sx={{
                      position: "absolute",
                      left: `${(previewInsetX / Math.max(1, canvasW)) * 100}%`,
                      top: `${(previewInsetY / Math.max(1, canvasH)) * 100}%`,
                      width: `${(insetW / Math.max(1, canvasW)) * 100}%`,
                      height: `${(insetH / Math.max(1, canvasH)) * 100}%`,
                      bgcolor: "transparent",
                      border: "1px solid transparent",
                      zIndex: 8,
                      overflow: "hidden",
                      pointerEvents: "auto",
                      cursor: "move",
                      touchAction: "none",
                    }}
                  >
                    <Box
                      data-show3d-fft-move-handle="true"
                      aria-label="Move FFT overlay; snaps to nearest corner"
                      onPointerDown={(e) => handleFftInsetPointerDown(e, panelLeft, panelTop, panelW, panelH, insetX, insetY, insetW, insetH)}
                      onPointerMove={handleFftInsetPointerMove}
                      onPointerUp={handleFftInsetPointerUp}
                      onPointerCancel={(e) => {
                        if (fftOverlayDragRef.current?.pointerId === e.pointerId) {
                          fftOverlayDragRef.current = null;
                          setFftOverlayDragPreview(null);
                        }
                      }}
                      onMouseDown={(e) => handleFftInsetMouseDown(e, panelLeft, panelTop, panelW, panelH, insetX, insetY, insetW, insetH)}
                      sx={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        right: 0,
                        height: Math.min(16, Math.max(10, insetH * 0.18)),
                        zIndex: 2,
                        cursor: "move",
                        background: "linear-gradient(180deg, rgba(0,0,0,0.38), rgba(0,0,0,0))",
                        opacity: 0.65,
                        touchAction: "none",
                        "&:hover": { opacity: 1 },
                      }}
                    />
                    {slot === 0 && fftMetricsEnabled && fftQuality && (
                      <Box
                        className="quantem-fft-quality-label"
                        aria-label={`FFT quality: ${formatFftQualityLabel(fftQuality)}`}
                        sx={{
                          position: "absolute",
                          top: 4,
                          left: 5,
                          right: 5,
                          color: "rgba(255,255,255,0.96)",
                          fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                          fontSize: 10,
                          fontWeight: 700,
                          lineHeight: 1.15,
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          textShadow: "1px 1px 0 rgba(0,0,0,0.9), 0 0 3px rgba(0,0,0,0.85)",
                          pointerEvents: "none",
                          userSelect: "none",
                          zIndex: 3,
                        }}
                      >
                        {formatFftQualityLabel(fftQuality)}
                      </Box>
                    )}
                  </Box>
                );
              });
            })()}
          </Box>
          {/* Panel titles render ON canvas inside drawMain - follows grid layout. */}
          {/* Statistics bar - right below the image. Multi-panel = one row per panel. */}
          {showStats && (
            (localPanelStats && (nPanels || 1) > 1) ? (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", flexDirection: "column", gap: 0.25, width: "100%", maxWidth: canvasW, boxSizing: "border-box", fontFamily: "ui-monospace, monospace" }}>
                {localPanelStats.map((st) => (
                  <Box key={st.panel} sx={{ display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap", maxWidth: "100%" }}>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted, minWidth: 80, fontFamily: "ui-monospace, monospace" }}>{panelLabel(st.panel)}</Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.mean)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.min)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.max)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.std)}</Box></Typography>
                  </Box>
                ))}
              </Box>
            ) : (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap", width: "100%", maxWidth: canvasW, boxSizing: "border-box" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.mean : statsMean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.min : statsMin)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.max : statsMax)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.std : statsStd)}</Box></Typography>
              </Box>
            )
          )}
          {/* Line profile sparkline */}
          {profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                onMouseMove={handleProfileMouseMove}
                onMouseLeave={handleProfileMouseLeave}
                style={{ width: "100%", height: profileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
                role="img"
                aria-label="Line intensity profile along the drawn line"
              />
              {showResizeControls && (
                <div
                  onMouseDown={(e) => { e.preventDefault(); setIsResizingProfile(true); setProfileResizeStart({ y: e.clientY, height: profileHeight }); }}
                  style={{ width: "100%", height: 4, cursor: "ns-resize", borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, background: `linear-gradient(to bottom, ${themeColors.border}, transparent)` }}
                />
              )}
            </Box>
          )}
          {/* ROI sparkline plot */}
          {effectiveRoiActive && showRoiPlot && roiPlotData && roiPlotData.byteLength >= 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, boxSizing: "border-box" }}>
              <canvas
                ref={roiPlotCanvasRef}
                style={{ width: "100%", height: 76, display: "block", border: `1px solid ${themeColors.border}` }}
                role="img"
                aria-label="ROI mean intensity over frames"
              />
            </Box>
          )}
          {/* Image controls stay content-sized so multi-panel stacks do not
              create a large empty gutter between display and playback rows. */}
	          {controlsVisible && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", columnGap: `${SPACING.SM}px`, rowGap: `${SPACING.XS}px`, alignItems: "flex-start", justifyContent: "flex-start", width: "fit-content", maxWidth: "100%", boxSizing: "border-box", flexWrap: "wrap" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: "0 0 auto", justifyContent: "center" }}>
                {/* Row 1: Scale + Auto + Color */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale</Typography>
                  <Select value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Intensity scale (linear or logarithmic)" }}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }} title={perPanelHistogramEnabled ? "Stack-wide auto contrast. Turn off for independent panel clips." : "Automatic percentile-based contrast."}>
                    {perPanelHistogramEnabled ? "Auto stack" : "Auto"}
                  </Typography>
                  <Switch checked={autoContrast} onChange={(e) => handleAutoContrastChange(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": perPanelHistogramEnabled ? "Toggle stack-wide automatic contrast" : "Toggle automatic percentile-based contrast" } }} />
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar</Typography>
                  <Switch checked={showColorbar} onChange={(e) => setShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle colorbar overlay" } }} />
                </Box>
                {/* Row 2: Color + Smooth + Diff + zoom indicator */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color</Typography>
                  <Select size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} inputProps={{ "aria-label": "Image colormap" }}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Smooth</Typography>
                  <Switch checked={smooth} onChange={(e) => setSmooth(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle bilinear smoothing" } }} />
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Diff</Typography>
                  <Select value={diffMode} onChange={(e) => setDiffMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Difference mode (off, previous frame, first frame)" }}>
                    <MenuItem value="off">Off</MenuItem>
                    <MenuItem value="previous">Prev</MenuItem>
                    <MenuItem value="first">First</MenuItem>
                  </Select>
                  {/* zoom indicator moved onto the canvas overlay */}
                </Box>
              </Box>
              {/* Playback: 2 rows side-by-side with Display + Histogram. */}
              {(() => { const activeIdx = visibleSliceIdx; return (
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: `${SPACING.XS}px`, flex: "0 1 auto", minWidth: 0, maxWidth: "100%", justifyContent: "center" }}>
                  <Box sx={{ ...controlRow, ...mobileControlRowSx, width: "fit-content", maxWidth: "100%", flexWrap: "nowrap", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, boxSizing: "border-box" }}>
                    <Stack direction="row" spacing={0} sx={{ flexShrink: 0, mr: 0.5 }}>
                      <IconButton size="small" onClick={() => playFromCurrentFrame(-1)} sx={{ color: reverse && playing ? themeColors.accent : themeColors.textMuted, p: 0.25 }} aria-label="Play in reverse" title="Play reverse">
                        <FastRewindIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                      <IconButton size="small" onClick={() => { if (playing) pausePlayback(); else playFromCurrentFrame(); }} sx={{ color: themeColors.accent, p: 0.25 }} aria-label={playing ? "Pause playback" : "Play"} title={playing ? "Pause (Space)" : "Play (Space)"}>
                        {playing ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
                      </IconButton>
                      <IconButton size="small" onClick={() => playFromCurrentFrame(1)} sx={{ color: !reverse && playing ? themeColors.accent : themeColors.textMuted, p: 0.25 }} aria-label="Play forward" title="Play forward">
                        <FastForwardIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                      <IconButton size="small" onClick={stopPlayback} sx={{ color: themeColors.textMuted, p: 0.25 }} aria-label="Stop and rewind to start" title="Stop">
                        <StopIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Stack>
                    {loop ? (
                      <Slider ref={playbackSliderRef} value={[loopStart, activeIdx, effectiveLoopEnd]} onMouseDown={handleLoopSliderMouseDown} onPointerDownCapture={handleLoopSliderPointerDownCapture} onChange={(_, v) => { const vals = v as number[]; setLoopStart(vals[0]); scrubToSlice(vals[1]); setLoopEnd(vals[2]); }} onChangeCommitted={(_, v) => { const vals = v as number[]; setLoopStart(vals[0]); commitSlice(vals[1]); setLoopEnd(vals[2]); }} disableSwap min={0} max={nSlices - 1} size="small" valueLabelDisplay="auto" valueLabelFormat={(v) => formatFrameValueLabel(v)} marks={bookmarkedFrames.map(f => ({ value: f }))} aria-label={`Loop range and current ${dimLabel.toLowerCase()} (frame ${activeIdx + 1} of ${nSlices}, loop ${loopStart + 1} to ${effectiveLoopEnd + 1})`} sx={{ ...sliderStyles.small, width: 150, flex: "0 1 150px", minWidth: 90, "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: themeColors.textMuted }, "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 }, "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: themeColors.textMuted }, "& .MuiSlider-mark": { bgcolor: themeColors.accent, width: 4, height: 4, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px", maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }} />
                    ) : (
                      <Slider ref={playbackSliderRef} value={activeIdx} onChange={(_, v) => scrubToSlice(v as number)} onChangeCommitted={(_, v) => commitSlice(v as number)} min={0} max={nSlices - 1} size="small" valueLabelDisplay="auto" valueLabelFormat={(v) => formatFrameValueLabel(v)} marks={bookmarkedFrames.map(f => ({ value: f }))} aria-label={`Current ${dimLabel.toLowerCase()} (${activeIdx + 1} of ${nSlices})`} sx={{ ...sliderStyles.small, width: 150, flex: "0 1 150px", minWidth: 90, "& .MuiSlider-mark": { bgcolor: themeColors.accent, width: 4, height: 4, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" }, "& .MuiSlider-valueLabel": { maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }} />
                    )}
                    <Typography ref={playbackLiveCountRef} sx={{ ...typography.value, color: themeColors.textMuted, minWidth: hiddenSet.size ? `${String(nSlices).length * 2 + String(visibleCount).length + 5}ch` : `${String(nSlices).length * 2 + 1}ch`, fontVariantNumeric: "tabular-nums", textAlign: "right", flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{hiddenSet.size ? `${activeIdx + 1}/${visibleCount} (${nSlices})` : `${activeIdx + 1}/${nSlices}`}</Typography>
                  </Box>
                  <Box sx={{ ...controlRow, ...mobileControlRowSx, width: "fit-content", maxWidth: "100%", flexWrap: "wrap", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, boxSizing: "border-box" }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>fps</Typography>
                      <Slider value={playbackFps} min={1} max={MAX_PLAYBACK_FPS} step={1} onChange={(_, v) => setPlaybackFps(v as number)} size="small" sx={{ ...sliderStyles.small, width: isMobileViewport ? 40 : 44, mx: isMobileViewport ? "3px" : 0, flexShrink: 0 }} aria-label="Playback frames per second" valueLabelDisplay="auto" />
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, minWidth: isMobileViewport ? 16 : 20, flexShrink: 0 }}>{Math.round(playbackFps)}</Typography>
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>avg</Typography>
                      {isMobileViewport ? (
                        <Select
                          value={String(Math.round(avgWindow || 1))}
                          onChange={(e) => setAvgWindow(Number(e.target.value) || 1)}
                          size="small"
                          sx={{ ...themedSelect, minWidth: 42, fontSize: 10 }}
                          MenuProps={themedMenuProps}
                          inputProps={{ "aria-label": "Moving average window" }}
                          title="Moving average window"
                        >
                          {AVG_WINDOW_OPTIONS.map((value) => (
                            <MenuItem key={value} value={String(value)}>{value}</MenuItem>
                          ))}
                        </Select>
                      ) : (
                        <>
                          <Slider value={avgWindow} min={1} max={15} step={1} onChange={(_, v) => setAvgWindow(v as number)} size="small" sx={{ ...sliderStyles.small, width: 44, flexShrink: 0 }} aria-label="Moving average window" valueLabelDisplay="auto" />
                          <Typography sx={{ ...typography.label, color: themeColors.textMuted, minWidth: 16, flexShrink: 0 }}>{Math.round(avgWindow || 1)}</Typography>
                        </>
                      )}
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>Loop</Typography>
                      <Switch size="small" checked={loop} onChange={() => setLoop(!loop)} sx={{ ...switchStyles.small, flexShrink: 0 }} slotProps={{ input: { "aria-label": "Toggle loop playback" } }} />
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>Bounce</Typography>
                      <Switch size="small" checked={boomerang} onChange={() => setBoomerang(!boomerang)} sx={{ ...switchStyles.small, flexShrink: 0 }} slotProps={{ input: { "aria-label": "Toggle bounce playback" } }} />
                    </Box>
                  </Box>
                </Box>
              ); })()}
              {(() => {
                // Global stack range from Python (data_min/data_max trait), not per-frame.
                // Log mode: log1p the range so bins line up with the log-scaled frame data.
                const { min: histMin, max: histMax } = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
                if (perPanelHistogramEnabled) {
                  const n = Math.max(1, visiblePanelCount || 1);
                  const cols = panelColsForCount(n);
                  // Match Show2D shell exactly (width=110, height=58, gap=15px)
                  // so the per-panel histogram strip is visually consistent
                  // across widgets.
                  const panelHistWidth = 110;
                  const panelHistGap = 15;
                  const panelHistMaxWidth = cols * panelHistWidth + Math.max(0, cols - 1) * panelHistGap;
                  return (
                    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start", gap: 0.5, opacity: 1, pointerEvents: "auto", maxWidth: "100%" }}>
                      <Box sx={{ display: "grid", gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${panelHistWidth}px), ${panelHistWidth}px))`, gap: `${panelHistGap}px`, width: "100%", maxWidth: panelHistMaxWidth, justifyContent: "start" }}>
                      {visiblePanelIndices.map((panel) => {
                        const state = panelStates[panel] || initialState;
                        // Per-panel histogram uses THIS panel's data range
                        // (not stack-wide histMin/histMax). Tight-range
                        // modalities (SSB phase) get a sensible slider
                        // space instead of being squashed by DF counts.
                        const pdr = panelDataRanges[panel];
                        const panelRange = (pdr && pdr.max > pdr.min) ? pdr : { min: histMin, max: histMax };
                        const vminPct = state.imageVminPct;
                        const vmaxPct = state.imageVmaxPct;
                        return (
                          <Histogram
                            key={`panel-hist-${panel}`}
                            data={panelHistogramData[panel] ?? null}
                            bins={null}
                            vminPct={vminPct}
                            vmaxPct={vmaxPct}
                            onRangeChange={(min, max) => {
                              updatePanelState(panel, { imageVminPct: min, imageVmaxPct: max });
                              setPanelRangeValues(panel, pctToValue(min, panelRange.min, panelRange.max), pctToValue(max, panelRange.min, panelRange.max));
                              if (autoContrast) {
                                restorePanelManualClipPcts();
                                manualImageRangeBeforeAutoRef.current = null;
                                setAutoContrast(false);
                              }
                            }}
                            width={110}
                            height={58}
                            theme={themeInfo.theme === "dark" ? "dark" : "light"}
                            dataMin={panelRange.min}
                            dataMax={panelRange.max}
                          />
                        );
                      })}
                      </Box>
                    </Box>
                  );
                }
                return (
                <Box sx={{
                  // Match Show2D histogram shell exactly so visual stays consistent
                  // across widgets. alignItems: flex-end (not stretch) prevents the
                  // inner Slider thumbs from overflowing onto the canvas, which was
                  // the source of the "2.8 tooltip overlaps bars" overlap bug.
                  display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start", gap: 0.5,
                }}>
                  <Histogram
                    data={imageHistogramData}
                    bins={imageHistogramBins}
                    vminPct={imageVminPct}
                    vmaxPct={imageVmaxPct}
                    onRangeChange={(min, max) => {
                      setImageVminPct(min);
                      setImageVmaxPct(max);
                      if (autoContrast) {
                        manualImageRangeBeforeAutoRef.current = null;
                        setAutoContrast(false);
                      }
                    }}
                    width={110}
                    height={58}
                    theme={themeInfo.theme === "dark" ? "dark" : "light"}
                    dataMin={histMin}
                    dataMax={histMax}
                  />
                </Box>
                );
              })()}
            </Box>
          )}
          {/* Lens settings row (when Lens is active) */}
          {showLens && (
            <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: "fit-content" }}>
              <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Lens {lensMag}×</Typography>
                <Slider value={lensMag} min={2} max={8} step={1} onChange={(_, v) => setLensMag(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} aria-label="Lens magnification" valueLabelDisplay="auto" />
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>{lensDisplaySize}px</Typography>
                <Slider value={lensDisplaySize} min={64} max={256} step={16} onChange={(_, v) => setLensDisplaySize(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} aria-label="Lens display size in pixels" valueLabelDisplay="auto" />
              </Box>
            </Box>
          )}
          {/* ROI settings row (when ROI is active) */}
          {effectiveRoiActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: "fit-content" }}>
              <Box sx={{ border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, px: 1, py: 0.5, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>
                {/* ROI: shape + add/duplicate + plot + dim */}
                <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>ROI</Typography>
                  <Select
                    size="small"
                    value={newRoiShape}
                    onChange={(e) => setNewRoiShape(e.target.value as "circle" | "square" | "rectangle" | "annular")}
                    MenuProps={themedMenuProps}
                    sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                    inputProps={{ "aria-label": "New ROI shape" }}
                  >
                    {(["square", "rectangle", "circle", "annular"] as const).map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                  </Select>
                  <Button size="small" sx={compactButton} onClick={() => addROIAt(height / 2, width / 2)} aria-label="Add ROI at image center">Add</Button>
                  <Button size="small" sx={compactButton} disabled={!selectedRoi} onClick={duplicateSelectedROI} aria-label="Duplicate selected ROI">Dup</Button>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Plot</Typography>
                  <Switch checked={showRoiPlot} onChange={(e) => setShowRoiPlot(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle ROI intensity plot" } }} />
                  <Box sx={{ flex: 1 }} />
                  <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 24, color: "#ef5350" }} disabled={!roiItems.length} onClick={() => { setRoiList([]); setRoiSelectedIdx(-1); }} aria-label="Clear all ROIs">Clear</Button>
                </Box>

                {/* Selected ROI details */}
                {selectedRoi && (
                  <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: `${SPACING.SM}px`, borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: selectedRoi.color }}>#{roiSelectedIdx + 1}/{roiItems.length}</Typography>
                    <Select
                      size="small"
                      value={selectedRoi.shape || "circle"}
                      onChange={(e) => updateSelectedRoi({ shape: String(e.target.value) })}
                      MenuProps={themedMenuProps}
                      sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                      inputProps={{ "aria-label": "Selected ROI shape" }}
                    >
                      {(["square", "rectangle", "circle", "annular"] as const).map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                    </Select>
                    {selectedRoi.shape === "rectangle" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>W</Typography>
                        <Slider value={selectedRoi.width} min={5} max={width} onChange={(_, v) => updateSelectedRoi({ width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="ROI width" valueLabelDisplay="auto" />
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>H</Typography>
                        <Slider value={selectedRoi.height} min={5} max={height} onChange={(_, v) => updateSelectedRoi({ height: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="ROI height" valueLabelDisplay="auto" />
                      </>
                    )}
                    {selectedRoi.shape === "annular" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Inner</Typography>
                        <Slider value={selectedRoi.radius_inner} min={1} max={Math.max(2, selectedRoi.radius - 1)} onChange={(_, v) => updateSelectedRoi({ radius_inner: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="Annular ROI inner radius" valueLabelDisplay="auto" />
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Outer</Typography>
                        <Slider value={selectedRoi.radius} min={selectedRoi.radius_inner + 1} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="Annular ROI outer radius" valueLabelDisplay="auto" />
                      </>
                    )}
                    {selectedRoi.shape !== "rectangle" && selectedRoi.shape !== "annular" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Size</Typography>
                        <Slider value={selectedRoi.radius} min={5} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 50 }} aria-label="ROI radius" valueLabelDisplay="auto" />
                      </>
                    )}
                    <Box sx={{ display: "flex", gap: "2px" }}>
                      {ROI_COLORS.map(c => (
                        <Box key={c} onClick={() => updateSelectedRoi({ color: c })} sx={{ width: 12, height: 12, bgcolor: c, cursor: "pointer", border: c === selectedRoi.color ? `2px solid ${themeColors.text}` : "1px solid transparent", "&:hover": { opacity: 0.8 } }} />
                      ))}
                    </Box>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Border</Typography>
                    <Slider value={selectedRoi.line_width} min={1} max={6} step={1} onChange={(_, v) => updateSelectedRoi({ line_width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 30 }} aria-label="ROI border line width" valueLabelDisplay="auto" />
                    <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 20, color: "#ef5350" }} onClick={deleteSelectedROI} aria-label="Delete selected ROI">&times;</Button>
                  </Box>
                )}

                {/* ROI list */}
                {roiItems.length > 0 && (
                  <Box sx={{ display: "flex", flexDirection: "column", borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                    {roiItems.map((roi, i) => {
                      const c = roi.color || ROI_COLORS[i % ROI_COLORS.length];
                      const isSelected = i === roiSelectedIdx;
                      const shapeLabel = roi.shape === "rectangle" ? `${roi.width}×${roi.height}` : roi.shape === "annular" ? `r${roi.radius_inner}-${roi.radius}` : `r${roi.radius}`;
                      return (
                        <Box key={i} onClick={() => setRoiSelectedIdx(i)} sx={{ display: "flex", alignItems: "center", gap: "3px", lineHeight: 1.6, cursor: "pointer", "&:hover .roi-delete": { opacity: 1 } }}>
                          <Box sx={{ width: 8, height: 8, borderRadius: roi.shape === "square" || roi.shape === "rectangle" ? 0 : "50%", bgcolor: c, border: isSelected ? "2px solid #fff" : "1px solid transparent", flexShrink: 0 }} />
                          <Typography component="span" sx={{ fontSize: 10, color: isSelected ? themeColors.text : themeColors.textMuted, fontWeight: isSelected ? "bold" : "normal" }}>
                            <Box component="span" sx={{ color: c }}>{i + 1}</Box>{" "}
                            {roi.shape} ({Math.round(roi.row)}, {Math.round(roi.col)}) {shapeLabel}
                          </Typography>
                          <Box
                            onClick={(e) => { e.stopPropagation(); const newList = roiItems.map((r, j) => ({ ...r, highlight: j === i ? !r.highlight : false })); setRoiList(newList); }}
                            sx={{ cursor: "pointer", fontSize: 10, color: roi.highlight ? themeColors.accentGreen : themeColors.textMuted, lineHeight: 1, opacity: roi.highlight ? 1 : 0.5, "&:hover": { opacity: 1 } }}
                            title="Focus (dim outside)"
                          >{roi.highlight ? "\u25C9" : "\u25CB"}</Box>
                          <Box
                            className="roi-delete"
                            onClick={(e) => { e.stopPropagation(); const newList = roiItems.filter((_, j) => j !== i); setRoiList(newList); setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1); }}
                            sx={{ opacity: 0, cursor: "pointer", fontSize: 10, color: themeColors.textMuted, ml: 0.5, lineHeight: 1, "&:hover": { color: "#f44336" } }}
                          >&times;</Box>
                        </Box>
                      );
                    })}
                  </Box>
                )}
              </Box>
            </Box>
          )}
        </Box>

        {/* Preview Panel - ROI crop at full resolution with aspect ratio */}
        {previewVisible && (
          <Box sx={{ width: "100%", maxWidth: canvasW, boxSizing: "border-box" }}>
            {/* Spacer - matches main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {/* Header row - matches main panel controls row height */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 28, height: "auto", flexWrap: "wrap", gap: `${SPACING.XS}px` }}>
              <Typography sx={{ ...typography.label, color: themeColors.accentGreen }}>
                Preview{previewCropDims ? ` (${previewCropDims.w}\u00d7${previewCropDims.h})` : ""}
              </Typography>
              <Button size="small" sx={compactButton} disabled={previewZoom.zoom === 1 && previewZoom.panX === 0 && previewZoom.panY === 0} onClick={handlePreviewDoubleClick} aria-label="Reset preview zoom and pan">Reset</Button>
            </Stack>
            <Box
              ref={previewContainerRef}
              sx={{
                position: "relative",
                bgcolor: "#000",
                border: `1px solid ${themeColors.border}`,
                cursor: "grab",
                width: "100%",
                maxWidth: previewCanvasDims.w,
                aspectRatio: `${Math.max(previewCanvasDims.w, 1)} / ${Math.max(previewCanvasDims.h, 1)}`,
                height: "auto",
              }}
              onWheel={handlePreviewWheel}
              onDoubleClick={handlePreviewDoubleClick}
              onMouseDown={handlePreviewMouseDown}
              onMouseMove={handlePreviewMouseMove}
              onMouseUp={handlePreviewMouseUp}
              onMouseLeave={handlePreviewMouseUp}
            >
              <canvas ref={previewCanvasRef} width={previewCanvasDims.w} height={previewCanvasDims.h} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: "pixelated" }} role="img" aria-label={`ROI preview crop${previewCropDims ? ` (${previewCropDims.w} by ${previewCropDims.h} pixels)` : ""}`} />
              <canvas ref={previewOverlayRef} width={Math.round(previewCanvasDims.w * DPR)} height={Math.round(previewCanvasDims.h * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
              {showResizeControls && (
                <Box onMouseDown={handleMainResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 28, height: 28, cursor: "nwse-resize", opacity: 0.95, background: `linear-gradient(135deg, transparent 50%, ${themeColors.border} 50%)`, "&:hover": { opacity: 1 } }} />
              )}
            </Box>
            {/* All-ROI Stats - one row per ROI, same style as main stats bar */}
            {showStats && allRoiStats.length > 0 && (
              <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: 0.5, width: "100%", maxWidth: previewCanvasDims.w, boxSizing: "border-box" }}>
                {allRoiStats.map((stats, i) => {
                  if (!stats) return null;
                  const color = roiItems[i]?.color || ROI_COLORS[i % ROI_COLORS.length];
                  const isSelected = i === roiSelectedIdx;
                  return (
                    <Box key={i} sx={{ px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap", border: isSelected ? `1px solid ${color}` : `1px solid transparent` }}>
                      <Box sx={{ width: 8, height: 8, bgcolor: color, borderRadius: "50%", flexShrink: 0 }} />
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color }}>{formatNumber(stats.mean)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color }}>{formatNumber(stats.min)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color }}>{formatNumber(stats.max)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color }}>{formatNumber(stats.std)}</Box></Typography>
                    </Box>
                  );
                })}
              </Box>
            )}
          </Box>
        )}

        {/* FFT Panel - same size as main image. Bottom stacks below; Right uses the side slot. */}
        {effectiveShowFft && !fftLayoutOverlay && (
          <Box sx={{
            width: "100%",
            maxWidth: fftLayoutBottom && (nPanels || 1) > 1 ? "100%" : canvasW,
            flex: fftLayoutBottom && (nPanels || 1) > 1 ? "1 0 100%" : `0 1 min(100%, ${canvasW}px)`,
            minWidth: fftLayoutBottom && (nPanels || 1) > 1 ? "100%" : undefined,
            ml: fftLayoutBottom && (nPanels || 1) > 1 ? "0 !important" : undefined,
            mt: fftLayoutBottom && (nPanels || 1) > 1 ? `${SPACING.SM}px !important` : undefined,
            boxSizing: "border-box",
          }}>
            {/* Spacer - matches main panel title row height for canvas alignment */}
            {(!fftLayoutBottom || (nPanels || 1) === 1) && <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />}
            {/* Controls row - matches main panel controls row height */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 28, height: "auto", flexWrap: "wrap", gap: `${SPACING.XS}px` }}>
              {roiFftActive && fftCropDims ? (
                <Typography sx={{ ...typography.label, color: themeColors.accentGreen }}>
                  ROI FFT ({fftCropDims.cropWidth}&times;{fftCropDims.cropHeight})
                </Typography>
              ) : <Box />}
            </Stack>
            {/* FFT Canvas - same size as main image */}
            <Box
              ref={fftContainerRef}
              sx={{
                ...container.imageBox,
                width: "100%",
                maxWidth: canvasW,
                aspectRatio: mainPanelAspectRatio,
                height: "auto",
                cursor: "grab",
                touchAction: "none",
              }}
              onMouseDown={handleFftMouseDown}
              onMouseMove={handleFftMouseMove}
              onMouseUp={handleFftMouseUp}
              onMouseLeave={() => { fftClickStartRef.current = null; setIsFftDragging(false); setFftPanStart(null); }}
              onWheel={handleFftWheel}
              onDoubleClick={handleFftReset}
              onTouchStart={handleFftTouchStart}
              onTouchMove={handleFftTouchMove}
              onTouchEnd={handleFftTouchEnd}
              onTouchCancel={handleFftTouchEnd}
            >
              <canvas ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", touchAction: "none" }} role="img" aria-label={roiFftActive && fftCropDims ? `FFT power spectrum of ROI crop (${fftCropDims.cropWidth} by ${fftCropDims.cropHeight} pixels)` : "FFT power spectrum of current frame"} />
              <canvas ref={fftOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
              {fftMetricsEnabled && fftQuality && (
                <Box
                  className="quantem-fft-quality-label"
                  aria-label={`FFT quality: ${formatFftQualityLabel(fftQuality)}`}
                  sx={{
                    position: "absolute",
                    top: 8,
                    left: 8,
                    maxWidth: "calc(100% - 16px)",
                    color: "rgba(255,255,255,0.96)",
                    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    fontSize: 11,
                    fontWeight: 700,
                    lineHeight: 1.2,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    textShadow: "1px 1px 0 rgba(0,0,0,0.9), 0 0 3px rgba(0,0,0,0.85)",
                    pointerEvents: "none",
                    userSelect: "none",
                    zIndex: 4,
                  }}
                >
                  {formatFftQualityLabel(fftQuality)}
                </Box>
              )}
              {showResizeControls && (() => {
                const n = Math.max(1, visiblePanelCount || 1);
                const cols = panelColsForCount(n);
                const rows = Math.ceil(n / cols);
                const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
                const outPanelW = (canvasW - gap * (cols - 1)) / cols;
                const outPanelH = (canvasH - gap * (rows - 1)) / rows;
                return visiblePanelIndices.map((panel, slot) => {
                  const col = slot % cols;
                  const row = Math.floor(slot / cols);
                  const slotX = col * (outPanelW + gap);
                  const slotY = row * (outPanelH + gap);
                  return (
                    <Box
                      key={`fft-resize-${panel}`}
                      onMouseDown={handleMainResizeStart}
                      sx={{
                        position: "absolute",
                        left: `calc(${((slotX + outPanelW) / Math.max(1, canvasW)) * 100}% - 16px)`,
                        top: `calc(${((slotY + outPanelH) / Math.max(1, canvasH)) * 100}% - 16px)`,
                        width: 16,
                        height: 16,
                        cursor: "nwse-resize",
                        opacity: 0.6,
                        background: `linear-gradient(135deg, transparent 50%, ${themeColors.border} 50%)`,
                        borderRadius: "0 0 4px 0",
                        zIndex: 3,
                        "&:hover": { opacity: 1 },
                      }}
                    />
                  );
                });
              })()}
            </Box>
            {/* FFT Statistics bar */}
            {showStats && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, flexWrap: "wrap" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.mean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.min)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.max)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.std)}</Box></Typography>
              </Box>
            )}
            {fftClickInfo && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, border: `1px solid ${themeColors.border}`, display: "flex", gap: 1.25, alignItems: "center", flexWrap: "wrap", width: "fit-content", maxWidth: canvasW, boxSizing: "border-box" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontWeight: 600 }}>FFT mark</Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                  {fftClickInfo.dSpacing != null ? (
                    <>d = <Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} Å`}</Box>{" | |g| = "}<Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.spatialFreq!.toFixed(4)} Å⁻¹</Box></>
                  ) : (
                    <>dist = <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                  )}
                </Typography>
              </Box>
            )}
            {/* FFT Controls - two rows with histogram on right (like Show4DSTEM) */}
	            {controlsVisible && <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", maxWidth: canvasW, boxSizing: "border-box", flexWrap: "wrap" }}>
              {/* Left: two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Auto */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale</Typography>
                  <Select value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT intensity scale (linear or logarithmic)" }}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto</Typography>
                  <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle automatic FFT contrast" } }} />
                  {roiFftActive && fftCropDims && (
                    <>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Win</Typography>
                      <Switch checked={fftWindow} onChange={(e) => setFftWindow(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle Hann windowing before FFT" } }} />
                    </>
                  )}
                </Box>
                {/* Row 2: Color + Colorbar */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT colormap" }}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar</Typography>
                  <Switch checked={fftShowColorbar} onChange={(e) => setFftShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle FFT colorbar overlay" } }} />
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                <Histogram
                  data={fftHistogramData}
                  vminPct={fftVminPct}
                  vmaxPct={fftVmaxPct}
                  onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }}
                  width={110}
                  height={58}
                  theme={themeInfo.theme}
                  dataMin={fftDataRange.min}
                  dataMax={fftDataRange.max}
                />
              </Box>
            </Box>}
          </Box>
        )}

        {/* Kymograph Panel - static space-time image (X = distance along line,
            Y = frame/time). Shares the side slot with FFT (mutually exclusive).
            Mirrors the FFT panel's adjustability (contrast, zoom/pan, colormap). */}
        {kymoReady && (
          <Box sx={{ width: "100%", maxWidth: canvasW, boxSizing: "border-box" }}>
            {/* Spacer - matches main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {/* Controls row - title on left, Reset on right */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 28, height: "auto", flexWrap: "wrap", gap: `${SPACING.XS}px` }}>
              <Typography sx={{ ...typography.label, color: themeColors.accentGreen }}>
                Kymograph ({kymoDataRef.current?.nFrames ?? nSlices} {dimUnit ? unitSymbol(dimUnit) : "frames"} &times; {kymoDataRef.current?.lineLen ?? 0} px)
              </Typography>
              <Button size="small" sx={compactButton} disabled={!kymoNeedsReset} onClick={handleKymoReset} aria-label="Reset kymograph zoom and pan">Reset</Button>
            </Stack>
            {/* Kymograph canvas - same size as main image */}
            <Box
              ref={kymoContainerRef}
              sx={{
                ...container.imageBox,
                width: "100%",
                maxWidth: canvasW,
                aspectRatio: mainPanelAspectRatio,
                height: "auto",
                cursor: "grab",
                position: "relative",
                touchAction: "none",
              }}
              onMouseDown={handleKymoMouseDown}
              onMouseMove={handleKymoMouseMove}
              onMouseUp={handleKymoMouseUp}
              onMouseLeave={() => { kymoClickStartRef.current = null; setIsKymoDragging(false); setKymoPanStart(null); }}
              onWheel={handleKymoWheel}
              onDoubleClick={handleKymoReset}
              onTouchStart={handleKymoTouchStart}
              onTouchMove={handleKymoTouchMove}
              onTouchEnd={handleKymoTouchEnd}
              onTouchCancel={handleKymoTouchEnd}
            >
              <canvas ref={kymoCanvasRef} width={canvasW} height={canvasH} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: "pixelated", touchAction: "none" }} role="img" aria-label="Kymograph space-time image: distance along profile line versus frame index" />
              <canvas ref={kymoOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
            </Box>
            {/* Axis labels - kymograph-specific footer */}
            <Box sx={{ display: "flex", justifyContent: "space-between", mt: 0.5, px: 0.5 }}>
              <Typography sx={{ fontSize: 9, color: themeColors.textMuted }}>
                {dimUnit ? `time (${unitSymbol(dimUnit)})${dimSampling && dimSampling !== 1 ? `, ${(dimSampling).toFixed(2)}/frame` : ""} ↓` : "frame ↓"}
              </Typography>
              <Typography sx={{ fontSize: 9, color: themeColors.textMuted }}>distance along line →</Typography>
            </Box>
            {/* Kymograph Statistics bar */}
            {showStats && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, flexWrap: "wrap" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.mean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.min)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.max)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.std)}</Box></Typography>
                {kymoClickInfo && (
                  <>
                    <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                      t = <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoClickInfo.timeVal)} {kymoClickInfo.timeUnit}</Box>{" | d = "}<Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoClickInfo.distVal)} {kymoClickInfo.distUnit}</Box>{" | I = "}<Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{formatNumber(kymoClickInfo.intensity)}</Box>
                    </Typography>
                  </>
                )}
              </Box>
            )}
            {/* Kymograph Controls - two rows with histogram on right (mirror FFT) */}
	            {controlsVisible && <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", maxWidth: canvasW, boxSizing: "border-box", flexWrap: "wrap" }}>
              {/* Left: two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Auto */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale</Typography>
                  <Select value={kymoLogScale ? "log" : "linear"} onChange={(e) => setKymoLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Kymograph intensity scale (linear or logarithmic)" }}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto</Typography>
                  <Switch checked={kymoAuto} onChange={(e) => setKymoAuto(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle automatic kymograph contrast" } }} />
                </Box>
                {/* Row 2: Color + Colorbar */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color</Typography>
                  <Select value={kymoColormap} onChange={(e) => setKymoColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Kymograph colormap" }}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar</Typography>
                  <Switch checked={kymoShowColorbar} onChange={(e) => setKymoShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle kymograph colorbar overlay" } }} />
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                <Histogram
                  data={kymoHistogramData}
                  vminPct={kymoVminPct}
                  vmaxPct={kymoVmaxPct}
                  onRangeChange={(min, max) => { setKymoVminPct(min); setKymoVmaxPct(max); }}
                  width={110}
                  height={58}
                  theme={themeInfo.theme}
                  dataMin={kymoDataRange.min}
                  dataMax={kymoDataRange.max}
                />
              </Box>
            </Box>}
          </Box>
        )}
      </Stack>
      {handoffEnabled && preparedViewWidget != null && (
        <EmbeddedWidgetView
          hostModel={model}
          widgetModel={preparedViewWidget}
          title="2D view"
          onClose={handleClosePreparedView}
          themeColors={themeColors}
          linkedTraits={SHOW3D_TO_SHOW2D_LINKED_TRAITS}
        />
      )}
      </>
      )}

    </Box>
  );
}

// anywidget v0.9+ deprecates `export render` in favor of `export default { render }`.
const render = createRender(Show3D);
export default { render };
