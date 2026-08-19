/// <reference types="@webgpu/types" />
import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import IconButton from "@mui/material/IconButton";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import StopIcon from "@mui/icons-material/Stop";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import FastForwardIcon from "@mui/icons-material/FastForward";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import DragIndicatorIcon from "@mui/icons-material/DragIndicator";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import { useTheme } from "../theme";
import { COLORMAPS, GPUColormapEngine, applyColormap } from "../colormaps";
import { WebGPUFFT, getWebGPUFFT, fft2dAsync, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2, applyHannWindow2D, reciprocalCoordinatesFromShiftedOffset } from "../fft";
import { findFFTPeakWebGPU, sampleLineProfileWebGPU } from "../geometry";
import {
  buildDetectorMask,
  buildFullDetectorMask,
  buildScanMask,
  DetectorCompute,
} from "../.generated/engine/detector/compute/webgpu/backend";
import { readH5MasterInfo, readH5Volume } from "../.generated/engine/io/backends/webgpu/h5reader";
import { decodeBslz4Batch, type Bslz4Spec } from "../.generated/engine/io/backends/webgpu/bslz4";
import {
  collectShow4DSTEMLocalH5Files,
  loadShow4DSTEMLocalH5MaskedSum,
  loadShow4DSTEMLocalH5Master,
  setShow4DSTEMLocalFiles,
  show4DSTEMHasLocalFiles,
} from "../.generated/engine/io/backends/webgpu/local-h5";
import { getGPUInfo, isSoftwareGPUAdapter } from "../.generated/engine/device/webgpu";
import { LazyShow4DSTEM } from "./lazy";
import { drawScaleBarHiDPI, drawColorbar, roundToNiceValue } from "../figure";
import { findDataRange, sliderRange, computeStats, computeHistogramFromBytes, percentileClip } from "../stats";
import { downloadBlob, extractBytes, formatNumber, preserveRestoredWidgetModelsOnSave } from "../format";
import { useHideStaticFallback } from "../staticFallback";
import { MetadataSection } from "../widgetInfo";
import { FolderWatchBadge, useFolderWatchModelLive } from "../folderWatchStatus";
import {
  beginPendingProgressiveComparePage,
  beginProgressiveComparePage,
  compareMessageGeneration,
  completeProgressiveComparePage,
  mergeProgressiveComparePanel,
  mergeProgressiveCompareCacheMetadata,
  freshVisibleComparePagePaintAck,
  progressiveCompareCacheBadge,
  progressiveComparePanelPresentation,
  recordComparePageClick,
  recordComparePageFirstPanelPaint,
  recordComparePageStaleDrop,
  recordComparePageVisiblePaint,
  reconcileCompletedCompareIndices,
  reconcileProgressiveComparePanels,
  retainCachedProgressiveComparePanels,
  shouldClearProgressiveComparePage,
  type ComparePageMessage,
  type ProgressiveComparePage,
} from "./progressiveCompare";

function normaliseViSource(value: unknown): string {
  const raw = String(value || "roi").trim();
  const key = raw.toLowerCase().replace(/[-\s]+/g, "_");
  if (["", "roi", "virtual", "virtual_image", "bf"].includes(key)) return "roi";
  if (["dpc_row", "dpc_com_row", "dpc_r", "dpcr"].includes(key)) return "DPC_row";
  if (["dpc_col", "dpc_com_col", "dpc_c", "dpcc"].includes(key)) return "DPC_col";
  if (["idpc", "integrated_dpc", "integrated_differential_phase_contrast"].includes(key)) return "iDPC";
  if (["ssb", "ssb_phase", "phase"].includes(key)) return "SSB";
  return raw;
}

function viSourceLabel(source: string): string {
  if (source === "roi") return "ROI";
  if (source === "DPC_row") return "DPC row";
  if (source === "DPC_col") return "DPC col";
  if (source === "iDPC") return "iDPC";
  if (source === "SSB") return "SSB";
  return source;
}

function ViSourceLabel({ source }: { source: string }) {
  return <>{viSourceLabel(source)}</>;
}

function viSourceUsesSymmetricRange(source: string): boolean {
  return source === "DPC_row" || source === "DPC_col" || source === "iDPC";
}

function publishShow4DSTEMViDisplay(detail: Record<string, unknown>) {
  try {
    const target = window as unknown as {
      __sh4dViDisplay?: Record<string, unknown>;
      __sh4dDpcDisplay?: Record<string, unknown>;
    };
    target.__sh4dViDisplay = detail;
    if (detail.source === "DPC_row" || detail.source === "DPC_col" || detail.source === "iDPC") {
      target.__sh4dDpcDisplay = detail;
    }
  } catch {
    // Diagnostics must not affect rendering.
  }
}

const VI_GPU_SLOT = 41;
const COMPARE_GPU_SLOT_BASE = 60;   // per-panel compare slots: 60, 61, 62, ...
type DpcGpuSource = "DPC_row" | "DPC_col" | "iDPC";
type ViGpuSource = "roi" | DpcGpuSource;
type ViGpuRangeMode = "cpu" | "gpu";
type ViGpuImage = {
  source: ViGpuSource;
  slot: number;
  width: number;
  height: number;
  rangeMode: ViGpuRangeMode;
  rawVersionAfter: number;
};

function isDpcGpuSource(source: string): source is DpcGpuSource {
  return source === "DPC_row" || source === "DPC_col" || source === "iDPC";
}

function viProductFrameView(
  model: any,
  scanRows: number,
  scanCols: number,
  sourceOverride?: string,
): DataView | null {
  const source = normaliseViSource(sourceOverride ?? model.get("vi_source"));
  if (source === "roi") return null;
  const labels = Array.isArray(model.get("vi_product_labels")) ? model.get("vi_product_labels") as string[] : [];
  const productIndex = labels.indexOf(source);
  if (productIndex < 0) return null;
  const bytes = model.get("vi_product_maps_bytes") as DataView | undefined;
  if (!bytes || bytes.byteLength === 0) return null;
  const frames = Math.max(1, Math.round(Number(model.get("vi_product_map_frames") || 1)));
  const pixels = Math.max(1, scanRows * scanCols);
  const frame = frames <= 1 ? 0 : Math.max(0, Math.min(frames - 1, Math.round(Number(model.get("frame_idx") || 0))));
  const start = ((productIndex * frames + frame) * pixels) * 4;
  const end = start + pixels * 4;
  if (end > bytes.byteLength) return null;
  return new DataView(bytes.buffer, bytes.byteOffset + start, pixels * 4);
}

function viProductStackForIndices(
  model: any,
  indices: number[],
  scanRows: number,
  scanCols: number,
): DataView | null {
  const source = normaliseViSource(model.get("vi_source"));
  if (source === "roi" || indices.length === 0) return null;
  const labels = Array.isArray(model.get("vi_product_labels")) ? model.get("vi_product_labels") as string[] : [];
  const productIndex = labels.indexOf(source);
  if (productIndex < 0) return null;
  const bytes = model.get("vi_product_maps_bytes") as DataView | undefined;
  if (!bytes || bytes.byteLength === 0) return null;
  const frames = Math.max(1, Math.round(Number(model.get("vi_product_map_frames") || 1)));
  const pixels = Math.max(1, scanRows * scanCols);
  const out = new Float32Array(indices.length * pixels);
  for (let slot = 0; slot < indices.length; slot++) {
    const frame = frames <= 1 ? 0 : Math.max(0, Math.min(frames - 1, Math.round(Number(indices[slot]) || 0)));
    const start = ((productIndex * frames + frame) * pixels) * 4;
    const end = start + pixels * 4;
    if (end > bytes.byteLength) return null;
    const src = new Float32Array(bytes.buffer, bytes.byteOffset + start, pixels);
    out.set(src, slot * pixels);
  }
  return new DataView(out.buffer);
}

function float32MapStackForLabel(
  model: any,
  labelsTrait: string,
  framesTrait: string,
  bytesTrait: string,
  label: string,
  indices: number[],
  scanRows: number,
  scanCols: number,
): DataView | null {
  if (!indices.length) return null;
  const labels = Array.isArray(model.get(labelsTrait)) ? model.get(labelsTrait) as string[] : [];
  const mapIndex = labels.map((value) => String(value).toUpperCase()).indexOf(label.toUpperCase());
  if (mapIndex < 0) return null;
  const bytes = model.get(bytesTrait) as DataView | undefined;
  if (!bytes || bytes.byteLength === 0) return null;
  const frames = Math.max(1, Math.round(Number(model.get(framesTrait) || 1)));
  const pixels = Math.max(1, scanRows * scanCols);
  const out = new Float32Array(indices.length * pixels);
  for (let slot = 0; slot < indices.length; slot++) {
    const frame = frames <= 1 ? 0 : Math.max(0, Math.min(frames - 1, Math.round(Number(indices[slot]) || 0)));
    const start = ((mapIndex * frames + frame) * pixels) * 4;
    const end = start + pixels * 4;
    if (end > bytes.byteLength) return null;
    const src = new Float32Array(bytes.buffer, bytes.byteOffset + start, pixels);
    out.set(src, slot * pixels);
  }
  return new DataView(out.buffer);
}

function viPresetLabelForCurrentRoi(model: any): string | null {
  const mode = String(model.get("roi_mode") || "circle").trim().toLowerCase();
  const bf = Math.max(1, Number(model.get("bf_radius") || 1));
  const centerRow = Number(model.get("center_row") || 0);
  const centerCol = Number(model.get("center_col") || 0);
  const roiCenterRow = Number(model.get("roi_center_row") || centerRow);
  const roiCenterCol = Number(model.get("roi_center_col") || centerCol);
  const radius = Number(model.get("roi_radius") || 0);
  const radiusInner = Number(model.get("roi_radius_inner") || 0);
  const close = (a: number, b: number, tol = 1.0) => Math.abs(a - b) <= tol;
  if (!close(roiCenterRow, centerRow) || !close(roiCenterCol, centerCol)) return null;
  if (mode === "circle" && close(radius, bf)) return "BF";
  if (mode === "annular" && close(radiusInner, bf * 0.5) && close(radius, bf)) return "ABF";
  if (mode === "annular" && close(radiusInner, bf) && close(radius, bf * 2.0)) return "ADF";
  if (mode === "annular" && close(radiusInner, bf * 2.0) && close(radius, bf * 4.0)) return "HAADF";
  return null;
}

function viPresetFrameView(
  model: any,
  scanRows: number,
  scanCols: number,
): DataView | null {
  const label = viPresetLabelForCurrentRoi(model);
  if (!label) return null;
  const frame = Math.round(Number(model.get("frame_idx") || 0));
  return float32MapStackForLabel(
    model,
    "vi_preset_labels",
    "vi_preset_map_frames",
    "vi_preset_maps_bytes",
    label,
    [frame],
    scanRows,
    scanCols,
  );
}

function viPresetStackForIndices(
  model: any,
  indices: number[],
  scanRows: number,
  scanCols: number,
): DataView | null {
  const label = viPresetLabelForCurrentRoi(model);
  if (!label) return null;
  return float32MapStackForLabel(
    model,
    "vi_preset_labels",
    "vi_preset_map_frames",
    "vi_preset_maps_bytes",
    label,
    indices,
    scanRows,
    scanCols,
  );
}

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

// ============================================================================
// UI Styles - component styling helpers
// ============================================================================
const SHOW4DSTEM_UI_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif";

const typography = {
  label: { fontSize: 11, fontFamily: SHOW4DSTEM_UI_FONT },
  labelSmall: { fontSize: 10, fontFamily: SHOW4DSTEM_UI_FONT },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const, fontFamily: SHOW4DSTEM_UI_FONT },
};

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

const switchStyles = {
  small: { '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
  medium: { '& .MuiSwitch-thumb': { width: 14, height: 14 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

// ============================================================================
// Layout Constants - consistent spacing throughout
// ============================================================================
const SPACING = {
  XS: 4,    // Extra small gap
  SM: 8,    // Small gap (default between elements)
  MD: 12,   // Medium gap (between control groups)
  LG: 16,   // Large gap (between major sections)
};

const CANVAS_SIZE = 480;  // Both DP and VI canvases
const MIN_CANVAS_SIZE = 240;
const COMPARE_GRID_DEFAULT_WIDTH = 980;
const MIN_COMPARE_GRID_WIDTH = 320;
const HTML_EXPORT_OVERHEAD_BYTES = 700_000;

function useDebugFps(enabled: boolean): number | null {
  const [fps, setFps] = React.useState<number | null>(null);

  React.useEffect(() => {
    if (!enabled) {
      setFps(null);
      return;
    }
    if (typeof window === "undefined" || typeof window.requestAnimationFrame !== "function") {
      return;
    }
    let disposed = false;
    let frameCount = 0;
    let last = window.performance.now();
    let raf = 0;
    const tick = (now: number) => {
      if (disposed) return;
      frameCount += 1;
      const elapsed = now - last;
      if (elapsed >= 500) {
        setFps(Math.round((frameCount * 1000) / Math.max(1, elapsed)));
        frameCount = 0;
        last = now;
      }
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => {
      disposed = true;
      window.cancelAnimationFrame(raf);
    };
  }, [enabled]);

  return fps;
}

function DebugPerfBadge({
  widget,
  fps,
  themeColors,
}: {
  widget: string;
  fps: number | null;
  themeColors: { accent: string };
}) {
  const fpsText = fps === null ? "--" : String(fps);
  return (
    <Box
      component="span"
      data-quantem-debug-badge={widget}
      title={`${widget} debug browser UI FPS`}
      sx={{
        ml: 0.75,
        px: 0.5,
        py: 0,
        borderRadius: "3px",
        border: `1px solid ${themeColors.accent}55`,
        bgcolor: themeColors.accent + "18",
        color: themeColors.accent,
        fontSize: 10,
        fontWeight: 600,
        fontVariantNumeric: "tabular-nums",
        whiteSpace: "nowrap",
        verticalAlign: "middle",
      }}
    >
      Debug UI FPS {fpsText}
    </Box>
  );
}

type Show4DSTEMWritableFile = {
  write: (data: BlobPart) => Promise<void>;
  close: () => Promise<void>;
};

type Show4DSTEMFileHandle = {
  createWritable: () => Promise<Show4DSTEMWritableFile>;
};

type Show4DSTEMSavePickerOptions = {
  suggestedName?: string;
  types?: { description: string; accept: Record<string, string[]> }[];
};

type Show4DSTEMWindow = Window & typeof globalThis & {
  showSaveFilePicker?: (options?: Show4DSTEMSavePickerOptions) => Promise<Show4DSTEMFileHandle>;
  showDirectoryPicker?: (options?: { mode?: "read" | "readwrite"; startIn?: string }) => Promise<unknown>;
};

function show4DSTEMGlobalInt(name: string, fallback: number, min: number, max: number): number {
  const value = (globalThis as Record<string, unknown>)[name];
  const raw = value === undefined || value === null || value === "" ? fallback : Number(value);
  const numeric = Number.isFinite(raw) ? Math.round(raw) : fallback;
  return Math.max(min, Math.min(max, numeric));
}

function show4DSTEMOptionalGlobalInt(name: string, min: number, max: number): number | undefined {
  const value = (globalThis as Record<string, unknown>)[name];
  if (value === undefined || value === null || value === "") return undefined;
  const raw = Number(value);
  if (!Number.isFinite(raw)) return undefined;
  return Math.max(min, Math.min(max, Math.round(raw)));
}

function show4DSTEMOptionalGlobalRegion(name: string): readonly [number, number, number, number] | undefined {
  const value = (globalThis as Record<string, unknown>)[name];
  if (value === undefined || value === null || value === "") return undefined;
  const raw = typeof value === "string" ? value.split(",").map((part) => Number(part.trim())) : value;
  if (!Array.isArray(raw) || raw.length !== 4) return undefined;
  const region = raw.map((part) => Math.round(Number(part)));
  if (!region.every((part) => Number.isFinite(part))) return undefined;
  return [region[0], region[1], region[2], region[3]];
}

type HtmlExportKind = "interactive" | "report";
type HtmlDatasetScope = "unhidden" | "current_page" | "starred" | "all";
type HtmlExportDtype = "uint8" | "uint16";
type HtmlInteractivePreset = {
  label: string;
  dtype: HtmlExportDtype;
  detBin: number;
  scanBin: number;
  estimatedBytes: number;
};

function makeHtmlExportFilename(
  title: string,
  nFrames: number,
  scanRows: number,
  scanCols: number,
  detRows: number,
  detCols: number,
  dtype: string,
  detBin: number,
  scanBin: number,
  exportKind: HtmlExportKind,
  datasetScope: HtmlDatasetScope,
): string {
  let slug = (title || "show4dstem")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  while (slug.includes("__")) slug = slug.replace(/__/g, "_");
  if (!slug) slug = "show4dstem";
  const binnedScanRows = Math.max(1, Math.floor(scanRows / scanBin));
  const binnedScanCols = Math.max(1, Math.floor(scanCols / scanBin));
  const binnedRows = Math.max(1, Math.floor(detRows / detBin));
  const binnedCols = Math.max(1, Math.floor(detCols / detBin));
  const shape = nFrames > 1
    ? `${nFrames}x${binnedScanRows}x${binnedScanCols}x${binnedRows}x${binnedCols}`
    : `${binnedScanRows}x${binnedScanCols}x${binnedRows}x${binnedCols}`;
  const prefix = exportKind === "report" ? `report_${datasetScope}` : dtype;
  return `${slug}_${shape}_${prefix}_rbin${scanBin}_kbin${detBin}.html`;
}

function formatSavedBytes(bytes: number): string {
  const mb = Math.max(0, bytes) / (1024 * 1024);
  if (mb >= 100) return `${Math.round(mb)} MB`;
  if (mb >= 10) return `${mb.toFixed(1)} MB`;
  return `${mb.toFixed(2)} MB`;
}

function formatEstimatedHtmlBytes(htmlBytes: number): string {
  const mb = htmlBytes / (1024 * 1024);
  if (mb >= 1000) return `~${(mb / 1024).toFixed(1)} GB`;
  if (mb >= 100) return `~${Math.round(mb)} MB`;
  if (mb >= 10) return `~${mb.toFixed(1)} MB`;
  return `~${mb.toFixed(2)} MB`;
}

function estimateInteractiveHtmlBytes(
  nFrames: number,
  scanRows: number,
  scanCols: number,
  detRows: number,
  detCols: number,
  dtype: HtmlExportDtype,
  detBin: number,
  scanBin: number,
): number {
  const binnedScanRows = Math.max(1, Math.floor(scanRows / scanBin));
  const binnedScanCols = Math.max(1, Math.floor(scanCols / scanBin));
  const binnedRows = Math.max(1, Math.floor(detRows / detBin));
  const binnedCols = Math.max(1, Math.floor(detCols / detBin));
  const bytesPerPixel = dtype === "uint16" ? 2 : 1;
  const payloadBytes = Math.max(0, nFrames) * binnedScanRows * binnedScanCols * binnedRows * binnedCols * bytesPerPixel;
  return Math.max(0, payloadBytes) * 4 / 3 + HTML_EXPORT_OVERHEAD_BYTES;
}

function formatEstimatedHtmlSize(payloadBytes: number): string {
  return formatEstimatedHtmlBytes(Math.max(0, payloadBytes) * 4 / 3 + HTML_EXPORT_OVERHEAD_BYTES);
}

function isAbortLikeError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

// Theme-aware ROI colors for DP detector overlay
interface RoiColors {
  stroke: string;
  strokeDragging: string;
  fill: string;
  fillDragging: string;
  handleFill: string;
  innerStroke: string;
  innerStrokeDragging: string;
  innerHandleFill: string;
  textColor: string;
}
const DARK_ROI_COLORS: RoiColors = {
  stroke: "rgba(0, 255, 0, 0.9)",
  strokeDragging: "rgba(255, 255, 0, 0.9)",
  fill: "rgba(0, 255, 0, 0.12)",
  fillDragging: "rgba(255, 255, 0, 0.12)",
  handleFill: "rgba(0, 255, 0, 0.8)",
  innerStroke: "rgba(0, 220, 255, 0.9)",
  innerStrokeDragging: "rgba(255, 200, 0, 0.9)",
  innerHandleFill: "rgba(0, 220, 255, 0.8)",
  textColor: "#0f0",
};
const LIGHT_ROI_COLORS: RoiColors = {
  stroke: "rgba(0, 140, 0, 0.9)",
  strokeDragging: "rgba(200, 160, 0, 0.9)",
  fill: "rgba(0, 140, 0, 0.15)",
  fillDragging: "rgba(200, 160, 0, 0.15)",
  handleFill: "rgba(0, 140, 0, 0.85)",
  innerStroke: "rgba(0, 160, 200, 0.9)",
  innerStrokeDragging: "rgba(200, 160, 0, 0.9)",
  innerHandleFill: "rgba(0, 160, 200, 0.85)",
  textColor: "#0a0",
};

const VI_SOURCE_COLORS = {
  bf: { dark: DARK_ROI_COLORS.textColor, light: LIGHT_ROI_COLORS.textColor },
  abf: { dark: "#44aaff", light: "#1769aa" },
  adf: { dark: "#ffaa44", light: "#9a5a00" },
  DPC_row: { dark: "#38bdf8", light: "#0369a1" },
  DPC_col: { dark: "#a78bfa", light: "#6d28d9" },
  iDPC: { dark: "#2dd4bf", light: "#0f766e" },
  SSB: { dark: "#f472b6", light: "#be185d" },
} as const;

function viSourceColorKey(source: string): keyof typeof VI_SOURCE_COLORS | null {
  const normalised = normaliseViSource(source);
  if (normalised === "DPC_row" || normalised === "DPC_col" || normalised === "iDPC" || normalised === "SSB") {
    return normalised;
  }

  const key = String(source || "").trim().toLowerCase().replace(/[-\s]+/g, "_");
  if (key === "bf" || key === "roi") return "bf";
  if (key === "abf") return "abf";
  if (key === "adf") return "adf";
  return null;
}

function viSourceDisplayColor(source: string, themeName: string): string | null {
  const key = viSourceColorKey(source);
  if (!key) return null;
  const palette = VI_SOURCE_COLORS[key];
  return themeName === "light" ? palette.light : palette.dark;
}

// Interaction constants
const RESIZE_HIT_AREA_PX = 10;
const CIRCLE_HANDLE_ANGLE = 0.707;  // cos(45°)
// Compact button style for Reset/Export
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

// Control row style — bordered container per row.
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
};

/** Format stat value for display (compact scientific notation for small values) */
function formatStat(value: number): string {
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 0.001 || abs >= 10000) {
    return value.toExponential(2);
  }
  if (abs < 0.01) return value.toFixed(4);
  if (abs < 1) return value.toFixed(3);
  return value.toFixed(2);
}


// ============================================================================
// FFT peak finder (snap to Bragg spot with sub-pixel centroid refinement)
// ============================================================================
const FFT_SNAP_RADIUS = 5;

/**
 * Draw VI crosshair on high-DPI canvas (crisp regardless of image resolution)
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawViPositionMarker(
  canvas: HTMLCanvasElement,
  dpr: number,
  posRow: number,  // Position in image coordinates
  posCol: number,
  zoom: number,
  panX: number,
  panY: number,
  imageWidth: number,
  imageHeight: number,
  isDragging: boolean,
  showLabel: boolean = true,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const scaleY = cssHeight / imageHeight;

  // posRow/posCol are integer scan indices. Center the crosshair on the SAMPLED
  // pixel (+0.5) so it sits in the middle of the scan position the CBED came from,
  // not at the pixel corner - otherwise on a zoomed coarse grid it reads as
  // ambiguous between two adjacent positions.
  const cellRow = Math.round(posRow);
  const cellCol = Math.round(posCol);
  const screenX = (cellCol + 0.5) * zoom * scaleX + panX * scaleX;
  const screenY = (cellRow + 0.5) * zoom * scaleY + panY * scaleY;

  // Simple crosshair (no circle)
  const crosshairSize = 12;
  const lineWidth = 1.5;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(255, 100, 100, 0.9)";
  ctx.lineWidth = lineWidth;

  // Draw crosshair lines only
  ctx.beginPath();
  ctx.moveTo(screenX - crosshairSize, screenY);
  ctx.lineTo(screenX + crosshairSize, screenY);
  ctx.moveTo(screenX, screenY - crosshairSize);
  ctx.lineTo(screenX, screenY + crosshairSize);
  ctx.stroke();

  if (showLabel) {
    // Label the exact scan position (row, col) so the scientist knows which
    // position the diffraction pattern was sampled from.
    const label = `(${cellRow}, ${cellCol})`;
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.font = "11px monospace";
    ctx.textBaseline = "bottom";
    const textW = ctx.measureText(label).width;
    const labelX = Math.min(cssWidth - textW - 4, screenX + crosshairSize + 4);
    const labelY = Math.max(13, screenY - 4);
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillRect(labelX - 2, labelY - 12, textW + 4, 13);
    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.95)" : "rgba(255, 160, 160, 0.95)";
    ctx.fillText(label, labelX, labelY);
  }

  ctx.restore();
}

/**
 * Draw VI ROI overlay on high-DPI canvas for real-space region selection
 * Note: Does NOT clear canvas - should be called after drawViPositionMarker
 */
function drawViRoiOverlayHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  roiMode: string,
  centerRow: number,
  centerCol: number,
  radius: number,
  roiWidth: number,
  roiHeight: number,
  zoom: number,
  panX: number,
  panY: number,
  imageWidth: number,
  imageHeight: number,
  isDragging: boolean,
  isDraggingResize: boolean,
  isHoveringResize: boolean
) {
  if (roiMode === "off") return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const scaleY = cssHeight / imageHeight;

  // Convert image coordinates to screen coordinates (row→screenY, col→screenX)
  const screenX = centerCol * zoom * scaleX + panX * scaleX;
  const screenY = centerRow * zoom * scaleY + panY * scaleY;

  const lineWidth = 2.5;
  const crosshairSize = 10;
  const handleRadius = 6;

  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  // Helper to draw resize handle (purple color for VI ROI to differentiate from DP)
  const drawResizeHandle = (handleX: number, handleY: number) => {
    let handleFill: string;
    let handleStroke: string;

    if (isDraggingResize) {
      handleFill = "rgba(180, 100, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else if (isHoveringResize) {
      handleFill = "rgba(220, 150, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else {
      handleFill = "rgba(160, 80, 255, 0.8)";
      handleStroke = "rgba(255, 255, 255, 0.8)";
    }
    ctx.beginPath();
    ctx.arc(handleX, handleY, handleRadius, 0, 2 * Math.PI);
    ctx.fillStyle = handleFill;
    ctx.fill();
    ctx.strokeStyle = handleStroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };

  // Helper to draw center crosshair (purple/magenta for VI ROI)
  const drawCenterCrosshair = () => {
    ctx.strokeStyle = isDragging ? "rgba(255, 200, 0, 0.9)" : "rgba(180, 80, 255, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(screenX - crosshairSize, screenY);
    ctx.lineTo(screenX + crosshairSize, screenY);
    ctx.moveTo(screenX, screenY - crosshairSize);
    ctx.lineTo(screenX, screenY + crosshairSize);
    ctx.stroke();
  };

  // Purple/magenta color for VI ROI to differentiate from green DP detector
  const strokeColor = isDragging ? "rgba(255, 200, 0, 0.9)" : "rgba(180, 80, 255, 0.9)";
  const fillColor = isDragging ? "rgba(255, 200, 0, 0.15)" : "rgba(180, 80, 255, 0.15)";

  if (roiMode === "circle" && radius > 0) {
    const screenRadiusX = radius * zoom * scaleX;
    const screenRadiusY = radius * zoom * scaleY;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusX, screenRadiusY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();

    // Resize handle at 45° diagonal
    const handleOffsetX = screenRadiusX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetY = screenRadiusY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetX, screenY + handleOffsetY);

  } else if (roiMode === "square" && radius > 0) {
    // Square uses radius as half-size
    const screenHalfW = radius * zoom * scaleX;
    const screenHalfH = radius * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "rect" && roiWidth > 0 && roiHeight > 0) {
    const screenHalfW = (roiWidth / 2) * zoom * scaleX;
    const screenHalfH = (roiHeight / 2) * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);
  }

  ctx.restore();
}

/**
 * Draw DP crosshair on high-DPI canvas (crisp regardless of detector resolution)
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawDpCrosshairHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  kCol: number,  // Column position in detector coordinates
  kRow: number,  // Row position in detector coordinates
  zoom: number,
  panX: number,
  panY: number,
  detWidth: number,
  detHeight: number,
  isDragging: boolean,
  roiColors: RoiColors = DARK_ROI_COLORS
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  // Use separate X/Y scale factors (canvas stretches to fill container)
  const scaleX = cssWidth / detWidth;
  const scaleY = cssHeight / detHeight;

  // Convert detector coordinates to CSS pixel coordinates
  const screenX = kCol * zoom * scaleX + panX * scaleX;
  const screenY = kRow * zoom * scaleY + panY * scaleY;
  
  // Fixed UI sizes in CSS pixels (consistent with VI crosshair)
  const crosshairSize = 18;
  const lineWidth = 3;
  const dotRadius = 6;
  
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
  ctx.lineWidth = lineWidth;
  
  // Draw crosshair
  ctx.beginPath();
  ctx.moveTo(screenX - crosshairSize, screenY);
  ctx.lineTo(screenX + crosshairSize, screenY);
  ctx.moveTo(screenX, screenY - crosshairSize);
  ctx.lineTo(screenX, screenY + crosshairSize);
  ctx.stroke();
  
  // Draw center dot
  ctx.beginPath();
  ctx.arc(screenX, screenY, dotRadius, 0, 2 * Math.PI);
  ctx.stroke();
  
  ctx.restore();
}

/**
 * Draw ROI overlay (circle, square, rect, annular) on high-DPI canvas
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawRoiOverlayHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  roiMode: string,
  centerCol: number,
  centerRow: number,
  radius: number,
  radiusInner: number,
  roiWidth: number,
  roiHeight: number,
  zoom: number,
  panX: number,
  panY: number,
  detWidth: number,
  detHeight: number,
  isDragging: boolean,
  isDraggingResize: boolean,
  isDraggingResizeInner: boolean,
  isHoveringResize: boolean,
  isHoveringResizeInner: boolean,
  roiColors: RoiColors = DARK_ROI_COLORS
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  // Use separate X/Y scale factors (canvas stretches to fill container)
  const scaleX = cssWidth / detWidth;
  const scaleY = cssHeight / detHeight;

  // Convert detector coordinates to CSS pixel coordinates
  const screenX = centerCol * zoom * scaleX + panX * scaleX;
  const screenY = centerRow * zoom * scaleY + panY * scaleY;
  
  // Fixed UI sizes in CSS pixels
  const lineWidth = 2.5;
  const crosshairSizeSmall = 10;
  const handleRadius = 6;
  
  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  // Helper to draw resize handle
  const drawResizeHandle = (handleX: number, handleY: number, isInner: boolean = false) => {
    let handleFill: string;
    let handleStroke: string;
    const dragging = isInner ? isDraggingResizeInner : isDraggingResize;
    const hovering = isInner ? isHoveringResizeInner : isHoveringResize;
    
    if (dragging) {
      handleFill = "rgba(0, 200, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else if (hovering) {
      handleFill = "rgba(255, 100, 100, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else {
      handleFill = isInner ? roiColors.innerHandleFill : roiColors.handleFill;
      handleStroke = "rgba(255, 255, 255, 0.8)";
    }
    ctx.beginPath();
    ctx.arc(handleX, handleY, handleRadius, 0, 2 * Math.PI);
    ctx.fillStyle = handleFill;
    ctx.fill();
    ctx.strokeStyle = handleStroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };
  
  // Helper to draw center crosshair
  const drawCenterCrosshair = () => {
    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(screenX - crosshairSizeSmall, screenY);
    ctx.lineTo(screenX + crosshairSizeSmall, screenY);
    ctx.moveTo(screenX, screenY - crosshairSizeSmall);
    ctx.lineTo(screenX, screenY + crosshairSizeSmall);
    ctx.stroke();
  };
  
  if (roiMode === "circle" && radius > 0) {
    // Use separate X/Y radii for ellipse (handles non-square detectors)
    const screenRadiusX = radius * zoom * scaleX;
    const screenRadiusY = radius * zoom * scaleY;

    // Draw ellipse (becomes circle if scaleX === scaleY)
    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusX, screenRadiusY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Semi-transparent fill
    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
    ctx.fill();

    drawCenterCrosshair();

    // Resize handle at 45° diagonal
    const handleOffsetX = screenRadiusX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetY = screenRadiusY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetX, screenY + handleOffsetY);

  } else if (roiMode === "square" && radius > 0) {
    // Square in detector space uses same half-size in both dimensions
    const screenHalfW = radius * zoom * scaleX;
    const screenHalfH = radius * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "rect" && roiWidth > 0 && roiHeight > 0) {
    const screenHalfW = (roiWidth / 2) * zoom * scaleX;
    const screenHalfH = (roiHeight / 2) * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "annular" && radius > 0) {
    // Use separate X/Y radii for ellipses
    const screenRadiusOuterX = radius * zoom * scaleX;
    const screenRadiusOuterY = radius * zoom * scaleY;
    const screenRadiusInnerX = (radiusInner || 0) * zoom * scaleX;
    const screenRadiusInnerY = (radiusInner || 0) * zoom * scaleY;

    // Outer ellipse
    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusOuterX, screenRadiusOuterY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Inner ellipse
    ctx.strokeStyle = isDragging ? roiColors.innerStrokeDragging : roiColors.innerStroke;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusInnerX, screenRadiusInnerY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Fill annular region
    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusOuterX, screenRadiusOuterY, 0, 0, 2 * Math.PI);
    ctx.ellipse(screenX, screenY, screenRadiusInnerX, screenRadiusInnerY, 0, 0, 2 * Math.PI, true);
    ctx.fill();

    drawCenterCrosshair();

    // Outer handle at 45° diagonal
    const handleOffsetOuterX = screenRadiusOuterX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetOuterY = screenRadiusOuterY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetOuterX, screenY + handleOffsetOuterY);

    // Inner handle at 45° diagonal
    const handleOffsetInnerX = screenRadiusInnerX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetInnerY = screenRadiusInnerY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetInnerX, screenY + handleOffsetInnerY, true);
  }
  
  ctx.restore();
}

// ============================================================================
// Histogram Component
// ============================================================================

interface HistogramProps {
  data: Float32Array | null;
  bins?: number[] | Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
}

/**
 * Info tooltip component - small ⓘ icon with hover tooltip
 */
function InfoTooltip({ text, theme = "dark" }: { text: React.ReactNode; theme?: "light" | "dark" }) {
  const isDark = theme === "dark";
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  return (
    <Tooltip
      title={content}
      arrow
      placement="bottom"
      componentsProps={{
        tooltip: {
          sx: {
            bgcolor: isDark ? "#333" : "#fff",
            color: isDark ? "#ddd" : "#333",
            border: `1px solid ${isDark ? "#555" : "#ccc"}`,
            maxWidth: 280,
            p: 1,
          },
        },
        arrow: {
          sx: {
            color: isDark ? "#333" : "#fff",
            "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` },
          },
        },
      }}
    >
      <Typography
        component="span"
        sx={{
          fontSize: 12,
          color: isDark ? "#888" : "#666",
          cursor: "help",
          ml: 0.5,
          "&:hover": { color: isDark ? "#aaa" : "#444" },
        }}
      >
        ⓘ
      </Typography>
    </Tooltip>
  );
}

function KeyboardShortcuts({ items }: { items: [string, string][] }) {
  return (
    <Box component="table" sx={{ borderCollapse: "collapse", "& td": { py: 0.25, fontSize: 11, lineHeight: 1.3, verticalAlign: "top" }, "& td:first-of-type": { pr: 1.5, opacity: 0.7, fontFamily: "monospace", fontSize: 10, whiteSpace: "nowrap" } }}>
      <tbody>
        {items.map(([key, desc], i) => (
          <tr key={i}><td>{key}</td><td>{desc}</td></tr>
        ))}
      </tbody>
    </Box>
  );
}

/**
 * Histogram component with integrated vmin/vmax slider and statistics.
 * Shows data distribution with adjustable clipping.
 */
function Histogram({
  data,
  bins: precomputedBins = null,
  vminPct,
  vmaxPct,
  onRangeChange,
  width = 120,
  height = 40,
  theme = "dark",
  dataMin = 0,
  dataMax = 1,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const sliderRef = React.useRef<HTMLDivElement | null>(null);
  const minLabelRef = React.useRef<HTMLElement | null>(null);
  const maxLabelRef = React.useRef<HTMLElement | null>(null);
  const onRangeChangeRef = React.useRef(onRangeChange);
  const pendingRangeRef = React.useRef<[number, number] | null>(null);
  const rangeRafRef = React.useRef<number | null>(null);
  const bins = React.useMemo(
    () => precomputedBins ? Array.from(precomputedBins) : computeHistogramFromBytes(data),
    [data, precomputedBins],
  );

  // Theme-aware colors
  const colors = theme === "dark" ? {
    bg: "#1a1a1a",
    barActive: "#888",
    barInactive: "#444",
    border: "#333",
  } : {
    bg: "#f0f0f0",
    barActive: "#666",
    barInactive: "#bbb",
    border: "#ccc",
  };

  const formatValue = React.useCallback((pct: number) => {
    const val = dataMin + (pct / 100) * (dataMax - dataMin);
    return val >= 1000 ? val.toExponential(1) : val.toFixed(1);
  }, [dataMax, dataMin]);

  // Draw histogram (vertical gray bars)
  const drawHistogram = React.useCallback((loPct: number, hiPct: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear with theme background
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);

    // Reduce to fewer bins for cleaner display
    const displayBins = 64;
    const binRatio = Math.floor(bins.length / displayBins);
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) {
        sum += bins[i * binRatio + j] || 0;
      }
      reducedBins.push(sum / binRatio);
    }

    // Normalize
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;

    // Calculate which bins are in the clipped range
    const vminBin = Math.floor((loPct / 100) * displayBins);
    const vmaxBin = Math.floor((hiPct / 100) * displayBins);

    // Draw histogram bars
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;

      // Bars inside range are highlighted, outside are dimmed
      const inRange = i >= vminBin && i <= vmaxBin;
      ctx.fillStyle = inRange ? colors.barActive : colors.barInactive;
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

  const sliderInset = 6;
  const sliderWidth = Math.max(1, width - sliderInset * 2);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0, width, overflow: "visible" }}>
      <Box sx={{ position: "relative", width, height: height + 6, overflow: "visible" }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}`, display: "block" }}
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
          min={0}
          max={100}
          size="small"
          valueLabelDisplay="auto"
          valueLabelFormat={formatValue}
          sx={{
            width: sliderWidth,
            py: 0,
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
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}><Typography ref={minLabelRef} sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{formatValue(vminPct)}</Typography><Typography ref={maxLabelRef} sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{formatValue(vmaxPct)}</Typography></Box>
    </Box>
  );
}

// ============================================================================
// Line Profile Sampling
// ============================================================================

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
// Crop single-mode ROI region from raw float32 data for ROI-scoped FFT
// ============================================================================
function cropSingleROI(
  data: Float32Array, imgW: number, imgH: number,
  mode: string, centerRow: number, centerCol: number,
  radius: number, roiW: number, roiH: number,
): { cropped: Float32Array; cropW: number; cropH: number } | null {
  if (mode === "off") return null;
  let x0: number, y0: number, x1: number, y1: number;

  if (mode === "rect") {
    const hw = roiW / 2, hh = roiH / 2;
    x0 = Math.max(0, Math.floor(centerCol - hw));
    y0 = Math.max(0, Math.floor(centerRow - hh));
    x1 = Math.min(imgW, Math.ceil(centerCol + hw));
    y1 = Math.min(imgH, Math.ceil(centerRow + hh));
  } else {
    x0 = Math.max(0, Math.floor(centerCol - radius));
    y0 = Math.max(0, Math.floor(centerRow - radius));
    x1 = Math.min(imgW, Math.ceil(centerCol + radius));
    y1 = Math.min(imgH, Math.ceil(centerRow + radius));
  }

  const cropW = x1 - x0, cropH = y1 - y0;
  if (cropW < 2 || cropH < 2) return null;

  const cropped = new Float32Array(cropW * cropH);
  if (mode === "circle") {
    const rSq = radius * radius;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const ix = x0 + dx, iy = y0 + dy;
        const distSq = (ix - centerCol) * (ix - centerCol) + (iy - centerRow) * (iy - centerRow);
        cropped[dy * cropW + dx] = distSq <= rSq ? data[iy * imgW + ix] : 0;
      }
    }
  } else {
    for (let dy = 0; dy < cropH; dy++) {
      const srcOff = (y0 + dy) * imgW + x0;
      cropped.set(data.subarray(srcOff, srcOff + cropW), dy * cropW);
    }
  }
  return { cropped, cropW, cropH };
}

interface CompareVirtualGridProps {
  bytes: DataView | null | undefined;
  count: number;
  indices: number[];
  // GPU-resident panels: frame -> engine colormap slot, painted with a GPU range
  // through each tile's visible WebGPU canvas; bytes stay the settle/export fallback.
  gpuSlots?: Map<number, number> | null;
  gpuVersion?: number;
  gpuEngine?: GPUColormapEngine | null;
  progressivePage?: ProgressiveComparePage | null;
  labels: string[];
  activeIdx: number;
  shapeRows: number;
  shapeCols: number;
  cols: number;
  colormap: string;
  scaleMode: "linear" | "log";
  vminPct: number;
  vmaxPct: number;
  autoContrast: boolean;
  smooth: boolean;
  cursorRow: number;
  cursorCol: number;
  status: string;
  themeColors: ReturnType<typeof useTheme>["colors"];
  panelChromeVisible: boolean;
  showScaleBar: boolean;
  pixelSize: number;
  pixelUnit: string;
  panelOrder: number[];
  hidden: number[];
  starred: number[];
  reorderMode: boolean;
  draggingFrame: number | null;
  pendingMoveFrame: number | null;
  maxWidthPx: number;
  panelGapPx: number;
  onResizeStart?: (event: React.PointerEvent<HTMLElement>, panelScale?: number) => void;
  onSelect: (idx: number) => void;
  onToggleStar: (idx: number) => void;
  onHide: (idx: number) => void;
  onReorderFrame: (dragFrame: number, targetFrame: number) => void;
  onDragFrameChange: (idx: number | null) => void;
  onPendingMoveFrameChange: (idx: number | null) => void;
  onPositionChange: (row: number, col: number, commit?: boolean) => void;
  onFreshVisiblePaint?: (
    page: ProgressiveComparePage,
    paintedIndices: number[],
  ) => void;
  onGpuPaint?: (panelCount: number) => void;
  onGpuRendererReady?: (renderNow: (() => number) | null) => void;
}

function CompareVirtualGrid({
  bytes,
  count,
  indices,
  gpuSlots,
  gpuVersion,
  gpuEngine,
  progressivePage,
  labels,
  activeIdx,
  shapeRows,
  shapeCols,
  cols,
  colormap,
  scaleMode,
  vminPct,
  vmaxPct,
  autoContrast,
  smooth,
  cursorRow,
  cursorCol,
  status,
  themeColors,
  panelChromeVisible,
  showScaleBar,
  pixelSize,
  pixelUnit,
  panelOrder,
  hidden,
  starred,
  reorderMode,
  draggingFrame,
  pendingMoveFrame,
  maxWidthPx,
  panelGapPx,
  onResizeStart,
  onSelect,
  onToggleStar,
  onHide,
  onReorderFrame,
  onDragFrameChange,
  onPendingMoveFrameChange,
  onPositionChange,
  onFreshVisiblePaint,
  onGpuPaint,
  onGpuRendererReady,
}: CompareVirtualGridProps) {
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const gpuCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const gpuCanvasContextsRef = React.useRef<(GPUCanvasContext | null)[]>([]);
  const canvasDrawCacheRef = React.useRef(new Map<number, {
    canvas: HTMLCanvasElement;
    panel: Float32Array;
    styleKey: string;
  }>());
  const progressivePageForPaintRef = React.useRef<ProgressiveComparePage | null>(progressivePage ?? null);
  const visiblePanelByFrameRef = React.useRef(new Map<number, Float32Array>());
  const visiblePaintRafRef = React.useRef({ beforePaint: 0, afterPaint: 0 });
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const tileRefs = React.useRef<(HTMLDivElement | null)[]>([]);
  const isDraggingPositionRef = React.useRef(false);
  const [isDraggingPosition, setIsDraggingPosition] = React.useState(false);
  const [overlayVersion, setOverlayVersion] = React.useState(0);
  const [compareZoom, setCompareZoom] = React.useState(1);
  const [comparePanX, setComparePanX] = React.useState(0);
  const [comparePanY, setComparePanY] = React.useState(0);
  const compareViewRef = React.useRef({ zoom: 1, panX: 0, panY: 0, raf: 0 });
  const panelPixels = Math.max(1, shapeRows * shapeCols);
  const panels = React.useMemo(() => {
    if (!bytes || count <= 0 || bytes.byteLength < panelPixels * count * 4) {
      return [] as Float32Array[];
    }
    const raw = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
    return Array.from({ length: count }, (_, idx) => {
      const start = idx * panelPixels;
      return raw.slice(start, start + panelPixels);
    });
  }, [bytes, count, panelPixels]);
  const sourceIndices = progressivePage ? progressivePage.expectedIndices : (indices || []);
  const [previewIndices, setPreviewIndices] = React.useState<number[] | null>(null);
  const panelByFrame = React.useMemo(
    () => reconcileProgressiveComparePanels(
      progressivePage ?? null,
      indices || [],
      panels,
    ),
    [indices, panels, progressivePage],
  );
  progressivePageForPaintRef.current = progressivePage ?? null;
  visiblePanelByFrameRef.current = panelByFrame;
  const cacheBadge = React.useMemo(
    () => progressiveCompareCacheBadge(progressivePage ?? null, panelByFrame),
    [panelByFrame, progressivePage],
  );
  const displayIndices = React.useMemo(() => {
    const available = new Set(sourceIndices);
    const hiddenSet = new Set((hidden || []).filter((idx) => Number.isInteger(idx) && available.has(idx)));
    const ordered: number[] = [];
    const seen = new Set<number>();
    (panelOrder || []).forEach((idx) => {
      if (available.has(idx) && !hiddenSet.has(idx) && !seen.has(idx)) {
        ordered.push(idx);
        seen.add(idx);
      }
    });
    sourceIndices.forEach((idx) => {
      if (!hiddenSet.has(idx) && !seen.has(idx)) ordered.push(idx);
    });
    return ordered;
  }, [hidden, panelOrder, sourceIndices]);
  const orderKey = displayIndices.join("|");

  React.useEffect(() => {
    setPreviewIndices(null);
  }, [orderKey, reorderMode]);

  const renderIndices = (
    reorderMode && previewIndices && previewIndices.length === displayIndices.length
      ? previewIndices
      : displayIndices
  );
  const renderEntries = React.useMemo(() => {
    return (renderIndices || [])
      .map((frame) => ({
        frame,
        panel: panelByFrame.get(frame),
        gpuLoaded: Boolean(gpuSlots?.has(frame) && gpuEngine),
      }))
      .filter((entry) => Boolean(progressivePage) || entry.panel !== undefined || entry.gpuLoaded);
  }, [gpuEngine, gpuSlots, gpuVersion, panelByFrame, progressivePage, renderIndices, scaleMode]);

  const renderGpuSlotsNow = React.useCallback((): number => {
    if (!gpuEngine || !gpuSlots) return 0;
    const lut = COLORMAPS[colormap] || COLORMAPS.inferno;
    gpuEngine.uploadLUT(colormap, lut);
    let painted = 0;
    renderEntries.forEach((entry, localIdx) => {
      const slot = gpuSlots.get(entry.frame);
      const canvas = gpuCanvasRefs.current[localIdx];
      if (slot === undefined || !canvas) return;
      if (
        canvas.width !== shapeCols
        || canvas.height !== shapeRows
        || !gpuCanvasContextsRef.current[localIdx]
      ) {
        gpuCanvasContextsRef.current[localIdx] = gpuEngine.configureCanvas(canvas, shapeCols, shapeRows);
      }
      const ctx = gpuCanvasContextsRef.current[localIdx];
      if (!ctx) return;
      const ok = gpuEngine.renderSlotDirectWithGpuRangeToCanvas(
        slot,
        vminPct,
        vmaxPct,
        scaleMode === "log",
        ctx,
        {
          width: shapeCols,
          height: shapeRows,
          bgRgb: 0,
          transform: { zoom: compareZoom, panX: comparePanX, panY: comparePanY },
          smooth,
        },
      );
      if (ok) painted++;
    });
    if (painted > 0) onGpuPaint?.(painted);
    return painted;
  }, [colormap, comparePanX, comparePanY, compareZoom, gpuEngine, gpuSlots, onGpuPaint, renderEntries, scaleMode, shapeCols, shapeRows, smooth, vmaxPct, vminPct]);

  React.useEffect(() => {
    onGpuRendererReady?.(renderGpuSlotsNow);
    return () => onGpuRendererReady?.(null);
  }, [onGpuRendererReady, renderGpuSlotsNow]);

  React.useEffect(() => {
    renderGpuSlotsNow();
  }, [gpuVersion, renderGpuSlotsNow]);

  const movePreviewFrame = React.useCallback((dragFrame: number, targetFrame: number) => {
    if (dragFrame === targetFrame) return;
    setPreviewIndices((current) => {
      const base = current && current.length === displayIndices.length ? current : [...displayIndices];
      if (!base.includes(dragFrame) || !base.includes(targetFrame)) return base;
      const next = base.filter((frame) => frame !== dragFrame);
      const targetPos = next.indexOf(targetFrame);
      next.splice(targetPos < 0 ? next.length : targetPos, 0, dragFrame);
      return next;
    });
  }, [displayIndices]);

  React.useEffect(() => {
    const lut = COLORMAPS[colormap] || COLORMAPS.inferno;
    if (gpuEngine) gpuEngine.uploadLUT(colormap, lut);
    const styleKey = [
      colormap,
      scaleMode,
      vminPct,
      vmaxPct,
      autoContrast ? 1 : 0,
      smooth ? 1 : 0,
      shapeRows,
      shapeCols,
    ].join("|");
    const visibleFrames = new Set(renderEntries.map(({ frame }) => frame));
    canvasDrawCacheRef.current.forEach((_, frame) => {
      if (!visibleFrames.has(frame)) canvasDrawCacheRef.current.delete(frame);
    });
    renderEntries.forEach(({ frame, panel }, idx) => {
      const canvas = canvasRefs.current[idx];
      if (!canvas) return;
      const resized = canvas.width !== shapeCols || canvas.height !== shapeRows;
      if (canvas.width !== shapeCols) canvas.width = shapeCols;
      if (canvas.height !== shapeRows) canvas.height = shapeRows;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      // GPU-resident path: adopted slot -> visible WebGPU canvas. This mirrors
      // the single-panel VI path and avoids readback or CPU colormap work.
      const gpuSlot = gpuEngine && gpuSlots ? gpuSlots.get(frame) : undefined;
      if (gpuSlot !== undefined && gpuEngine) {
        canvasDrawCacheRef.current.delete(frame);
        return;
      }
      if (!panel) {
        ctx.clearRect(0, 0, shapeCols, shapeRows);
        canvasDrawCacheRef.current.delete(frame);
        return;
      }
      const previous = canvasDrawCacheRef.current.get(frame);
      if (!resized && previous?.canvas === canvas && previous.panel === panel && previous.styleKey === styleKey) return;
      ctx.imageSmoothingEnabled = smooth;
      if (smooth) ctx.imageSmoothingQuality = "high";

      let scaled = panel;
      if (scaleMode === "log") {
        scaled = new Float32Array(panel.length);
        for (let i = 0; i < panel.length; i++) {
          scaled[i] = Math.log1p(Math.max(0, panel[i]));
        }
      }
      const { min, max } = findDataRange(scaled);
      let vmin: number;
      let vmax: number;
      if (autoContrast) {
        ({ vmin, vmax } = percentileClip(scaled, 1, 99));
      } else {
        ({ vmin, vmax } = sliderRange(min, max, vminPct, vmaxPct));
      }
      const imageData = ctx.createImageData(shapeCols, shapeRows);
      applyColormap(scaled, imageData.data, lut, vmin, vmax);
      ctx.putImageData(imageData, 0, 0);
      canvasDrawCacheRef.current.set(frame, { canvas, panel, styleKey });
    });
    const expected = progressivePage?.expectedIndices ?? [];
    const drawnExpectedIndices = expected.filter((frame) => {
      const currentPanel = panelByFrame.get(frame);
      const drawn = canvasDrawCacheRef.current.get(frame);
      return Boolean(
        (currentPanel && drawn?.panel === currentPanel && drawn.canvas.isConnected)
        || (gpuEngine && gpuSlots?.has(frame)),
      );
    });
    const paintRaf = visiblePaintRafRef.current;
    if (drawnExpectedIndices.length > 0 && paintRaf.beforePaint === 0 && paintRaf.afterPaint === 0) {
      paintRaf.beforePaint = requestAnimationFrame(() => {
        paintRaf.beforePaint = 0;
        paintRaf.afterPaint = requestAnimationFrame(() => {
          paintRaf.afterPaint = 0;
          const currentPage = progressivePageForPaintRef.current;
          if (!currentPage) return;
          const currentPanels = visiblePanelByFrameRef.current;
          const paintedIndices = currentPage.expectedIndices.filter((frame) => {
            const currentPanel = currentPanels.get(frame);
            const drawn = canvasDrawCacheRef.current.get(frame);
            return Boolean(
              (currentPanel && drawn?.panel === currentPanel && drawn.canvas.isConnected)
              || (gpuEngine && gpuSlots?.has(frame)),
            );
          });
          recordComparePageFirstPanelPaint(currentPage, paintedIndices);
          // Performance telemetry is optional and may not be initialized when
          // a notebook view reconnects from trait state. Scientific paint
          // acknowledgement must therefore be validated independently.
          recordComparePageVisiblePaint(currentPage, paintedIndices);
          onFreshVisiblePaint?.(currentPage, paintedIndices);
        });
      });
    }
  }, [autoContrast, colormap, gpuEngine, gpuSlots, gpuVersion, onFreshVisiblePaint, renderEntries, scaleMode, shapeCols, shapeRows, smooth, vmaxPct, vminPct]);

  React.useEffect(() => {
    if (!gpuEngine || !gpuSlots) return;
    const gpuEntries = renderEntries
      .map((entry, localIdx) => ({
        frame: entry.frame,
        localIdx,
        slot: gpuSlots.get(entry.frame),
      }))
      .filter((entry): entry is { frame: number; localIdx: number; slot: number } => entry.slot !== undefined);
    if (gpuEntries.length === 0) return;

    const lut = COLORMAPS[colormap] || COLORMAPS.inferno;
    gpuEngine.uploadLUT(colormap, lut);

    gpuEntries.forEach(({ localIdx }) => {
      const canvas = canvasRefs.current[localIdx];
      if (!canvas) return;
      if (canvas.width !== shapeCols) canvas.width = shapeCols;
      if (canvas.height !== shapeRows) canvas.height = shapeRows;
    });

    void (async () => {
      const bitmaps = await gpuEngine.renderSlotsWithComputedGpuRangeAsync(
        gpuEntries.map((entry) => entry.slot),
        gpuEntries.map(() => vminPct),
        gpuEntries.map(() => vmaxPct),
        scaleMode === "log",
      );
      if (!bitmaps) return;

      let painted = 0;
      bitmaps.forEach((bitmap, i) => {
        if (!bitmap) return;
        const entry = gpuEntries[i];
        const canvas = canvasRefs.current[entry.localIdx];
        const ctx = canvas?.getContext("2d");
        if (!canvas || !ctx) {
          bitmap.close?.();
          return;
        }
        if (canvas.width !== shapeCols) canvas.width = shapeCols;
        if (canvas.height !== shapeRows) canvas.height = shapeRows;
        ctx.clearRect(0, 0, shapeCols, shapeRows);
        ctx.imageSmoothingEnabled = smooth;
        if (smooth) ctx.imageSmoothingQuality = "high";
        ctx.drawImage(bitmap, 0, 0, shapeCols, shapeRows);
        bitmap.close?.();
        canvasDrawCacheRef.current.delete(entry.frame);
        painted++;
      });
      if (painted > 0) onGpuPaint?.(painted);
    })();
  }, [
    colormap,
    gpuEngine,
    gpuSlots,
    onGpuPaint,
    renderEntries,
    scaleMode,
    shapeCols,
    shapeRows,
    smooth,
    vmaxPct,
    vminPct,
  ]);

  React.useEffect(() => {
    return () => {
      const paintRaf = visiblePaintRafRef.current;
      if (paintRaf.beforePaint) cancelAnimationFrame(paintRaf.beforePaint);
      if (paintRaf.afterPaint) cancelAnimationFrame(paintRaf.afterPaint);
      paintRaf.beforePaint = 0;
      paintRaf.afterPaint = 0;
    };
  }, []);

  const displayCount = Math.max(1, renderEntries.length);
  const autoCols = displayCount >= 8 ? 4 : displayCount >= 5 ? 3 : displayCount >= 2 ? 2 : 1;
  const requestedMaxCols = cols > 0 ? Math.max(1, Math.floor(cols)) : autoCols;
  const gridCols = Math.max(1, Math.min(displayCount, requestedMaxCols));
  const mobileGridCols = Math.max(1, Math.min(gridCols, 2));
  const gridGapPx = Math.max(0, Math.floor(Number.isFinite(panelGapPx) ? panelGapPx : 0));
  const resizeGripSx = React.useMemo(() => ({
    position: "absolute",
    bottom: 0,
    right: 0,
    width: 16,
    height: 16,
    cursor: "nwse-resize",
    opacity: 0.6,
    pointerEvents: "auto",
    background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
    touchAction: "none",
    zIndex: 5,
    "&:hover": { opacity: 1 },
  }), [themeColors.accent]);
  const imageLeft = `${(comparePanX / Math.max(1, shapeCols)) * 100}%`;
  const imageTop = `${(comparePanY / Math.max(1, shapeRows)) * 100}%`;
  const imageWidth = `${compareZoom * 100}%`;
  const imageHeight = `${compareZoom * 100}%`;
  React.useEffect(() => {
    const view = compareViewRef.current;
    view.zoom = compareZoom;
    view.panX = comparePanX;
    view.panY = comparePanY;
  }, [compareZoom, comparePanX, comparePanY]);

  React.useEffect(() => {
    const view = compareViewRef.current;
    view.zoom = 1;
    view.panX = 0;
    view.panY = 0;
    setCompareZoom(1);
    setComparePanX(0);
    setComparePanY(0);
  }, [shapeCols, shapeRows]);

  React.useEffect(() => {
    return () => {
      const raf = compareViewRef.current.raf;
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  const zoomCompareAt = React.useCallback((tile: HTMLDivElement, clientX: number, clientY: number, deltaY: number) => {
    const rect = tile.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const mouseX = ((clientX - rect.left) / rect.width) * shapeCols;
    const mouseY = ((clientY - rect.top) / rect.height) * shapeRows;
    const view = compareViewRef.current;
    const zoomFactor = deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, view.zoom * zoomFactor));
    const zoomRatio = newZoom / view.zoom;
    view.zoom = newZoom;
    view.panX = mouseX - (mouseX - view.panX) * zoomRatio;
    view.panY = mouseY - (mouseY - view.panY) * zoomRatio;
    if (view.raf === 0) {
      view.raf = requestAnimationFrame(() => {
        view.raf = 0;
        setCompareZoom(view.zoom);
        setComparePanX(view.panX);
        setComparePanY(view.panY);
      });
    }
  }, [shapeCols, shapeRows]);

  React.useEffect(() => {
    const listeners: Array<[HTMLDivElement, (event: WheelEvent) => void]> = [];
    tileRefs.current.forEach((node) => {
      if (!node) return;
      const listener = (event: WheelEvent) => {
        event.preventDefault();
        event.stopPropagation();
        zoomCompareAt(node, event.clientX, event.clientY, event.deltaY);
      };
      node.addEventListener("wheel", listener, { passive: false });
      listeners.push([node, listener]);
    });
    return () => {
      listeners.forEach(([node, listener]) => node.removeEventListener("wheel", listener));
    };
  }, [orderKey, renderEntries.length, zoomCompareAt]);

  const handleCompareDoubleClick = React.useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const view = compareViewRef.current;
    view.zoom = 1;
    view.panX = 0;
    view.panY = 0;
    setCompareZoom(1);
    setComparePanX(0);
    setComparePanY(0);
  }, []);

  const updatePositionFromPointer = React.useCallback((
    tile: HTMLDivElement,
    clientX: number,
    clientY: number,
    commit = false,
  ) => {
    const rect = tile.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const tileX = ((clientX - rect.left) / rect.width) * shapeCols;
    const tileY = ((clientY - rect.top) / rect.height) * shapeRows;
    const col = Math.round(Math.max(0, Math.min(shapeCols - 1, (tileX - comparePanX) / compareZoom)));
    const row = Math.round(Math.max(0, Math.min(shapeRows - 1, (tileY - comparePanY) / compareZoom)));
    onPositionChange(row, col, commit);
  }, [comparePanX, comparePanY, compareZoom, onPositionChange, shapeCols, shapeRows]);

  React.useLayoutEffect(() => {
    const bump = () => setOverlayVersion((value) => value + 1);
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(bump) : null;
    tileRefs.current.forEach((node) => {
      if (node) observer?.observe(node);
    });
    bump();
    return () => observer?.disconnect();
  }, [gridCols, mobileGridCols, renderEntries.length]);

  React.useEffect(() => {
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    renderEntries.forEach(({ panel, gpuLoaded }, idx) => {
      const overlay = overlayRefs.current[idx];
      const tile = tileRefs.current[idx];
      if (!overlay || !tile) return;
      const cssWidth = Math.max(1, Math.round(tile.clientWidth));
      const cssHeight = Math.max(1, Math.round(tile.clientHeight));
      const width = Math.max(1, Math.round(cssWidth * dpr));
      const height = Math.max(1, Math.round(cssHeight * dpr));
      if (overlay.width !== width) overlay.width = width;
      if (overlay.height !== height) overlay.height = height;
      const ctx = overlay.getContext("2d");
      ctx?.clearRect(0, 0, overlay.width, overlay.height);
      if (!panel && !gpuLoaded) return;
      if (showScaleBar) {
        const unit = pixelSize > 0 ? pixelUnit || "px" : "px";
        const pxSize = pixelSize > 0 ? pixelSize : 1;
        drawScaleBarHiDPI(overlay, dpr, compareZoom, pxSize, unit, shapeCols);
      }
      drawViPositionMarker(
        overlay,
        dpr,
        cursorRow,
        cursorCol,
        compareZoom,
        comparePanX,
        comparePanY,
        shapeCols,
        shapeRows,
        isDraggingPosition,
        idx === 0,
      );
    });
  }, [comparePanX, comparePanY, compareZoom, cursorCol, cursorRow, isDraggingPosition, overlayVersion, pixelSize, pixelUnit, renderEntries, shapeCols, shapeRows, showScaleBar]);

  if (renderEntries.length === 0) {
    return (
      <Box sx={{ border: `1px solid ${themeColors.border}`, bgcolor: themeColors.bgAlt, px: 1, py: 2 }}>
        <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
          {status || "Multiple grid is waiting for multiple frames or datasets."}
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: "100%", maxWidth: maxWidthPx > 0 ? `${maxWidthPx}px` : "100%", position: "relative", "@media (max-width: 700px)": { maxWidth: "100%" } }}>
      {cacheBadge && (
        <Box
          role="status"
          aria-live="polite"
          data-testid="show4dstem-compare-cache-status"
          data-show4dstem-cache-tone={cacheBadge.tone}
          sx={{
            display: "inline-flex",
            alignItems: "center",
            minHeight: 20,
            mb: 0.5,
            px: 0.75,
            py: 0.25,
            border: `1px solid ${cacheBadge.tone === "warning" ? "#d97706" : cacheBadge.tone === "fresh" ? "#16a34a" : themeColors.border}`,
            borderRadius: "10px",
            bgcolor: cacheBadge.tone === "warning"
              ? "rgba(217,119,6,0.12)"
              : cacheBadge.tone === "fresh"
                ? "rgba(22,163,74,0.1)"
                : themeColors.controlBg,
            color: cacheBadge.tone === "warning"
              ? "#d97706"
              : cacheBadge.tone === "fresh"
                ? "#16a34a"
                : themeColors.textMuted,
            fontSize: 10,
            fontWeight: 600,
            lineHeight: 1.2,
          }}
        >
          {cacheBadge.label}
        </Box>
      )}
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: `repeat(${gridCols}, minmax(128px, 1fr))`,
          gap: `${gridGapPx}px`,
          maxWidth: "100%",
          "@media (max-width: 700px)": {
            gridTemplateColumns: `repeat(${mobileGridCols}, minmax(0, 1fr))`,
            gap: `${gridGapPx}px`,
          },
        }}
      >
        {renderEntries.map(({ frame, panel, gpuLoaded }, localIdx) => {
          const loaded = panel !== undefined || gpuLoaded;
          const panelPresentation = progressiveComparePanelPresentation(
            progressivePage ?? null,
            frame,
            loaded,
          );
          const waiting = !loaded && (progressivePage?.loading ?? false);
          const placeholderText = waiting ? "Loading" : "Unavailable";
          const active = frame === activeIdx;
          const label = labels && labels.length > frame ? labels[frame] : `Dataset ${frame + 1}`;
          const isStarred = (starred || []).includes(frame);
          const isDragging = draggingFrame === frame;
          const isPendingMove = pendingMoveFrame === frame;
          const tileRing = isPendingMove
            ? "inset 0 0 0 2px #facc15, inset 0 0 0 3px rgba(0,0,0,0.75)"
            : active
              ? `inset 0 0 0 2px ${themeColors.accent}, inset 0 0 0 3px rgba(255,255,255,0.72)`
              : "none";
          return (
            <Box
              key={`${frame}-${localIdx}`}
              ref={(node: HTMLDivElement | null) => { tileRefs.current[localIdx] = node; }}
              role="button"
              aria-label={`Show4DSTEM multiple panel ${frame + 1}${panelPresentation.labelSuffix}`}
              aria-busy={panelPresentation.busy}
              aria-disabled={panelPresentation.disabled}
              data-show4dstem-panel-cache={panelPresentation.cached ? "cached" : loaded ? "fresh" : "empty"}
              tabIndex={0}
              draggable={loaded && reorderMode}
              onDoubleClick={handleCompareDoubleClick}
              onPointerDown={(event) => {
                const target = event.target instanceof Element ? event.target : null;
                if (!loaded || reorderMode || target?.closest("button")) return;
                try { event.currentTarget.setPointerCapture(event.pointerId); } catch {}
                isDraggingPositionRef.current = true;
                setIsDraggingPosition(true);
                updatePositionFromPointer(event.currentTarget, event.clientX, event.clientY);
                onSelect(frame);
              }}
              onPointerMove={(event) => {
                if (!isDraggingPositionRef.current || reorderMode) return;
                event.preventDefault();
                updatePositionFromPointer(event.currentTarget, event.clientX, event.clientY);
              }}
              onPointerUp={(event) => {
                if (!isDraggingPositionRef.current) return;
                updatePositionFromPointer(event.currentTarget, event.clientX, event.clientY, true);
                isDraggingPositionRef.current = false;
                setIsDraggingPosition(false);
              }}
              onPointerCancel={(event) => {
                if (!isDraggingPositionRef.current) return;
                updatePositionFromPointer(event.currentTarget, event.clientX, event.clientY, true);
                isDraggingPositionRef.current = false;
                setIsDraggingPosition(false);
              }}
              onClick={() => {
                if (!loaded) return;
                if (!reorderMode) {
                  onSelect(frame);
                  return;
                }
                if (pendingMoveFrame == null) {
                  onPendingMoveFrameChange(frame);
                  return;
                }
                if (pendingMoveFrame === frame) {
                  onPendingMoveFrameChange(null);
                  return;
                }
                onReorderFrame(pendingMoveFrame, frame);
                onPendingMoveFrameChange(null);
              }}
              onKeyDown={(event) => {
                if (!loaded) return;
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  if (reorderMode) {
                    if (pendingMoveFrame == null) {
                      onPendingMoveFrameChange(frame);
                    } else if (pendingMoveFrame === frame) {
                      onPendingMoveFrameChange(null);
                    } else {
                      onReorderFrame(pendingMoveFrame, frame);
                      onPendingMoveFrameChange(null);
                    }
                  } else {
                    onSelect(frame);
                  }
                }
              }}
              onDragStart={(event) => {
                if (!loaded || !reorderMode) return;
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData("text/plain", String(frame));
                setPreviewIndices([...displayIndices]);
                onDragFrameChange(frame);
              }}
              onDragEnter={(event) => {
                if (!loaded || !reorderMode || draggingFrame == null || draggingFrame === frame) return;
                event.preventDefault();
                movePreviewFrame(draggingFrame, frame);
              }}
              onDragOver={(event) => {
                if (!loaded || !reorderMode) return;
                event.preventDefault();
                event.dataTransfer.dropEffect = "move";
              }}
              onDrop={(event) => {
                if (!loaded || !reorderMode) return;
                event.preventDefault();
                const rawFrame = event.dataTransfer.getData("text/plain");
                const dragFrame = rawFrame ? Number(rawFrame) : draggingFrame;
                if (typeof dragFrame === "number" && Number.isInteger(dragFrame) && dragFrame !== frame) {
                  onReorderFrame(dragFrame, frame);
                }
                setPreviewIndices(null);
                onDragFrameChange(null);
                onPendingMoveFrameChange(null);
              }}
              onDragEnd={() => {
                setPreviewIndices(null);
                onDragFrameChange(null);
              }}
              sx={{
                position: "relative",
                bgcolor: "#000",
                border: "none",
                boxSizing: "border-box",
                outline: "none",
                cursor: loaded ? (reorderMode ? "grab" : "crosshair") : waiting ? "progress" : "default",
                overflow: "hidden",
                touchAction: reorderMode ? "auto" : "none",
                opacity: isDragging ? 0.45 : 1,
                transform: isPendingMove ? "translateY(-2px)" : "translateY(0)",
                transition: "transform 120ms ease, opacity 120ms ease",
                aspectRatio: `${shapeCols} / ${shapeRows}`,
                "&::after": {
                  content: '""',
                  position: "absolute",
                  inset: 0,
                  pointerEvents: "none",
                  boxShadow: tileRing,
                  transition: "box-shadow 120ms ease",
                  zIndex: 4,
                },
                "&:focus-visible::after": {
                  boxShadow: `inset 0 0 0 2px ${themeColors.accent}, inset 0 0 0 4px rgba(255,255,255,0.82)`,
                },
                "&:hover .show4dstem-compare-hide-button, &:focus-within .show4dstem-compare-hide-button": {
                  opacity: 1,
                  pointerEvents: "auto",
                  transform: "translateY(0)",
                },
                "&:hover .show4dstem-compare-star-button, &:focus-within .show4dstem-compare-star-button": {
                  opacity: 1,
                  pointerEvents: "auto",
                  transform: "translateY(0)",
                },
                "@media (hover: none), (pointer: coarse)": {
                  "& .show4dstem-compare-hide-button": { display: "none" },
                  "& .show4dstem-compare-star-button": { opacity: 1, pointerEvents: "auto", transform: "translateY(0)" },
                },
                ...(reorderMode ? {
                  "@keyframes show4dstem-compare-reorder-jiggle": {
                    "0%": { rotate: "-0.25deg" },
                    "100%": { rotate: "0.25deg" },
                  },
                  animation: "show4dstem-compare-reorder-jiggle 180ms ease-in-out infinite alternate",
                } : {}),
              }}
            >
              <canvas
                data-quantem-scientific-output={`show4dstem-compare-${frame}`}
                ref={(node) => { canvasRefs.current[localIdx] = node; }}
                width={shapeCols}
                height={shapeRows}
                style={{
                  position: "absolute",
                  left: imageLeft,
                  top: imageTop,
                  width: imageWidth,
                  height: imageHeight,
                  imageRendering: smooth ? "auto" : "pixelated",
                  pointerEvents: "none",
                  opacity: panel || gpuLoaded ? 1 : 0,
                  transition: "opacity 160ms ease",
                }}
              />
              <canvas
                ref={(node) => {
                  gpuCanvasRefs.current[localIdx] = node;
                  if (!node) gpuCanvasContextsRef.current[localIdx] = null;
                }}
                width={shapeCols}
                height={shapeRows}
                style={{
                  position: "absolute",
                  left: imageLeft,
                  top: imageTop,
                  width: imageWidth,
                  height: imageHeight,
                  imageRendering: smooth ? "auto" : "pixelated",
                  pointerEvents: "none",
                  opacity: gpuLoaded ? 1 : 0,
                  zIndex: 2,
                }}
              />
              <Box
                aria-hidden="true"
                data-show4dstem-panel-loading={loaded ? "false" : "true"}
                sx={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  bgcolor: themeColors.bgAlt,
                  color: themeColors.textMuted,
                  fontSize: 10,
                  letterSpacing: "0.02em",
                  opacity: loaded ? 0 : 1,
                  transition: "opacity 160ms ease",
                  pointerEvents: "none",
                  zIndex: 1,
                }}
              >
                {loaded ? "" : placeholderText}
              </Box>
              <canvas
                ref={(node) => { overlayRefs.current[localIdx] = node; }}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "none",
                  zIndex: 2,
                }}
              />
              <Box
                sx={{
                  position: "absolute",
                  top: 6,
                  left: 28,
                  right: 28,
                  px: 0.5,
                  color: "rgba(255,255,255,0.95)",
                  fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                  fontSize: 11,
                  fontWeight: 700,
                  lineHeight: 1.2,
                  textAlign: "center",
                  textShadow: "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px rgba(0,0,0,0.75)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  pointerEvents: "none",
                  userSelect: "none",
                  zIndex: 2,
                }}
                title={label}
              >
                {label}
              </Box>
              {loaded && panelChromeVisible && reorderMode && (
                <Box
                  sx={{
                    position: "absolute",
                    bottom: 6,
                    left: "50%",
                    transform: "translateX(-50%)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: 28,
                    height: 20,
                    borderRadius: 1,
                    bgcolor: "rgba(0,0,0,0.35)",
                    color: "rgba(255,255,255,0.9)",
                    pointerEvents: "none",
                    zIndex: 3,
                  }}
                >
                  <DragIndicatorIcon sx={{ fontSize: 18 }} />
                </Box>
              )}
              {loaded && panelChromeVisible && (
                <Tooltip title={(isStarred ? "Unstar " : "Star ") + label}>
                  <IconButton
                    size="small"
                    aria-label={`${isStarred ? "Unstar" : "Star"} Show4DSTEM multiple panel ${frame + 1}`}
                    className="show4dstem-compare-star-button"
                    data-frame={frame}
                    onPointerDown={(event) => event.stopPropagation()}
                    onMouseDown={(event) => event.stopPropagation()}
                    onMouseUp={(event) => event.stopPropagation()}
                    onClick={(event) => {
                      event.stopPropagation();
                      onToggleStar(frame);
                    }}
                    sx={{
                      position: "absolute",
                      top: 5,
                      right: 5,
                      width: 22,
                      height: 22,
                      p: 0,
                      border: "none",
                      bgcolor: "transparent",
                      cursor: "pointer",
                      fontSize: 18,
                      lineHeight: "20px",
                      textAlign: "center",
                      color: isStarred ? "#ffc107" : "rgba(255,255,255,0.58)",
                      textShadow: "0 0 3px rgba(0,0,0,0.8)",
                      opacity: isStarred ? 1 : 0,
                      pointerEvents: "auto",
                      transform: isStarred ? "translateY(0)" : "translateY(-3px)",
                      transition: "opacity 120ms ease, transform 120ms ease, background-color 120ms ease, color 120ms ease",
                      userSelect: "none",
                      zIndex: 3,
                      "&:hover, &:focus-visible": {
                        bgcolor: "rgba(0,0,0,0.22)",
                        color: isStarred ? "#ffc107" : "rgba(255,255,255,0.9)",
                      },
                    }}
                  >
                    {isStarred ? "★" : "☆"}
                  </IconButton>
                </Tooltip>
              )}
              {loaded && panelChromeVisible && (
                <Tooltip title={renderEntries.length <= 1 ? "Cannot hide the last visible panel" : `Hide ${label}`}>
                  <IconButton
                    size="small"
                    disabled={renderEntries.length <= 1}
                    aria-label={renderEntries.length <= 1 ? "Cannot hide the last visible panel" : `Hide Show4DSTEM multiple panel ${frame + 1}`}
                    className="show4dstem-compare-hide-button"
                    data-frame={frame}
                    onPointerDown={(event) => event.stopPropagation()}
                    onMouseDown={(event) => event.stopPropagation()}
                    onClick={(event) => {
                      event.stopPropagation();
                      if (renderEntries.length > 1) onHide(frame);
                    }}
                    sx={{
                      position: "absolute",
                      top: 5,
                      left: 5,
                      width: 22,
                      height: 22,
                      p: 0,
                      opacity: 0,
                      transform: "translateY(-3px)",
                      transition: "opacity 120ms ease, transform 120ms ease, background-color 120ms ease, color 120ms ease",
                      color: renderEntries.length <= 1 ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.75)",
                      bgcolor: "rgba(0,0,0,0.22)",
                      pointerEvents: "none",
                      zIndex: 3,
                      "&:hover, &:focus-visible": {
                        bgcolor: "rgba(0,0,0,0.42)",
                        color: "rgba(255,255,255,0.95)",
                      },
                    }}
                  >
                    <VisibilityOffIcon sx={{ fontSize: 15 }} />
                  </IconButton>
                </Tooltip>
              )}
              {panelChromeVisible && onResizeStart && !reorderMode && (
                <Box
                  onPointerDown={(event) => {
                    const view = event.currentTarget.ownerDocument.defaultView;
                    const activeGridCols = view && view.innerWidth <= 700 ? mobileGridCols : gridCols;
                    onResizeStart(event, activeGridCols);
                  }}
                  aria-label={`Resize Show4DSTEM multiple panel ${frame + 1}`}
                  role="button"
                  tabIndex={-1}
                  className="show4dstem-compare-panel-resize"
                  title="Resize panels"
                  sx={resizeGripSx}
                />
              )}
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}

// ============================================================================
// Main Component
// ============================================================================
function Show4DSTEM() {
  // Direct model access for batched updates
  const model = useModel();
  const folderWatchLive = useFolderWatchModelLive(model);
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);

  // ─────────────────────────────────────────────────────────────────────────
  // Model State (synced with Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [shapeRows] = useModelState<number>("shape_rows");
  const [shapeCols] = useModelState<number>("shape_cols");
  const [detRows] = useModelState<number>("det_rows");
  const [detCols] = useModelState<number>("det_cols");

  const [posRow, setPosRow] = useModelState<number>("pos_row");
  const [posCol, setPosCol] = useModelState<number>("pos_col");
  const [roiCenterCol, setRoiCenterCol] = useModelState<number>("roi_center_col");
  const [roiCenterRow, setRoiCenterRow] = useModelState<number>("roi_center_row");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [kPixelSize] = useModelState<number>("k_pixel_size");
  const [kPixelUnit] = useModelState<string>("k_pixel_unit");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");
  const [title] = useModelState<string>("title");
  const [showTitle] = useModelState<boolean>("show_title");
  const [folderWatchState] = useModelState<string>("folder_watch_state");
  const [folderWatchDetail] = useModelState<string>("folder_watch_detail");
  const [gpuMemoryLabel] = useModelState<string>("gpu_memory_label");
  const [memoryWarning] = useModelState<string>("memory_warning");

  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [virtualImageBytes, setVirtualImageBytes] = useModelState<DataView>("virtual_image_bytes");
  const [frontendVirtualImageBytes, setFrontendVirtualImageBytes] = React.useState<DataView | null>(null);
  const [viSource, setViSourceModel] = useModelState<string>("vi_source");
  const [viProductLabels] = useModelState<string[]>("vi_product_labels");
  const [viProductMapFrames] = useModelState<number>("vi_product_map_frames");
  const [viProductMapsBytes] = useModelState<DataView>("vi_product_maps_bytes");
  const [, setSsbComputeRequest] = useModelState<string>("ssb_compute_request");
  const [ssbComputeStatus] = useModelState<string>("ssb_compute_status");
  const [ssbComputeBusy] = useModelState<boolean>("ssb_compute_busy");
  const [ssbComputeEnabled] = useModelState<boolean>("ssb_compute_enabled");
  const [ssbComputeNTrials, setSsbComputeNTrials] = useModelState<number>("ssb_compute_n_trials");
  const [ssbComputeRefine, setSsbComputeRefine] = useModelState<boolean>("ssb_compute_refine");
  const [ssbComputeLockC10, setSsbComputeLockC10] = useModelState<boolean>("ssb_compute_lock_c10");
  const [ssbComputeLockC12, setSsbComputeLockC12] = useModelState<boolean>("ssb_compute_lock_c12");
  const [ssbComputeBfSubsample, setSsbComputeBfSubsample] = useModelState<number>("ssb_compute_bf_subsample");
  const [ssbComputeBfPixels] = useModelState<number>("ssb_compute_bf_pixels");
  const [ssbComputeBfSelectedPixels] = useModelState<number>("ssb_compute_bf_selected_pixels");
  const [ssbComputeC10Nm, setSsbComputeC10Nm] = useModelState<number>("ssb_compute_c10_nm");
  const [ssbComputeC12Nm, setSsbComputeC12Nm] = useModelState<number>("ssb_compute_c12_nm");
  const [ssbComputePhi12Deg, setSsbComputePhi12Deg] = useModelState<number>("ssb_compute_phi12_deg");
  const [ssbComputeRotationDeg, setSsbComputeRotationDeg] = useModelState<number>("ssb_compute_rotation_angle_deg");
  const [ssbComputeCalibrationJson] = useModelState<string>("ssb_compute_calibration_json");
  const [ssbComputeCalibrationFilename] = useModelState<string>("ssb_compute_calibration_filename");

  // ROI state
  const [roiRadiusModel, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiRadiusInner, setRoiRadiusInner] = useModelState<number>("roi_radius_inner");
  const [roiMode, setRoiMode] = useModelState<string>("roi_mode");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");

  // Global min/max for DP normalization (from Python)
  const [dpGlobalMin] = useModelState<number>("dp_global_min");
  const [dpGlobalMax] = useModelState<number>("dp_global_max");

  // VI min/max for normalization (from Python)
  // viDataMin/viDataMax are derived JS-side from virtual_image_bytes (computed below).
  // Keeping them out of Python traits avoids a comm-message ordering race where
  // bytes from click N arrive with min/max from click N-1.

  // Detector calibration (for presets)
  const [centerCol] = useModelState<number>("center_col");
  const [centerRow] = useModelState<number>("center_row");

  // Path animation state
  const [pathPlaying, setPathPlaying] = useModelState<boolean>("path_playing");
  const [pathIndex, setPathIndex] = useModelState<number>("path_index");
  const [pathLength] = useModelState<number>("path_length");
  const [pathIntervalMs] = useModelState<number>("path_interval_ms");
  const [pathLoop] = useModelState<boolean>("path_loop");

  // Frame animation state (5D time/tilt series)
  const [frameIdx, setFrameIdx] = useModelState<number>("frame_idx");
  const [nFrames] = useModelState<number>("n_frames");
  const [frameDimLabel] = useModelState<string>("frame_dim_label");
  const [frameLabels] = useModelState<string[]>("frame_labels");
  const [framePlaying, setFramePlaying] = useModelState<boolean>("frame_playing");
  const [frameLoop, setFrameLoop] = useModelState<boolean>("frame_loop");
  const [frameFps, setFrameFps] = useModelState<number>("frame_fps");
  const [frameReverse, setFrameReverse] = useModelState<boolean>("frame_reverse");
  const [frameBoomerang, setFrameBoomerang] = useModelState<boolean>("frame_boomerang");
  const [viewMode, setViewMode] = useModelState<string>("view_mode");
  const [compareLayout] = useModelState<string>("compare_layout");
  const [compareCols, setCompareCols] = useModelState<number>("compare_cols");
  const [compareVirtualImageBytes] = useModelState<DataView>("compare_virtual_image_bytes");
  const [comparePanelCount] = useModelState<number>("compare_panel_count");
  const [comparePanelIndices] = useModelState<number[]>("compare_panel_indices");
  const [compareStatus] = useModelState<string>("compare_status");
  const [compareDpMode, setCompareDpMode] = useModelState<string>("compare_dp_mode");
  const [compareGroupMode, setCompareGroupMode] = useModelState<string>("compare_group_mode");
  const [comparePageIdx, setComparePageIdx] = useModelState<number>("compare_page_idx");
  const [comparePageCount] = useModelState<number>("compare_page_count");
  const [comparePanelGapPx] = useModelState<number>("compare_panel_gap_px");
  const [compareMaxPanels] = useModelState<number>("compare_max_panels");
  const [comparePanelOrder, setComparePanelOrder] = useModelState<number[]>("compare_panel_order");
  const [compareHiddenPanels, setCompareHiddenPanels] = useModelState<number[]>("compare_hidden_panels");
  const [compareStarredPanels, setCompareStarredPanels] = useModelState<number[]>("compare_starred_panels");
  const [comparePageProgressiveEnabled] = useModelState<boolean>("compare_page_progressive_enabled");
  const [comparePageExpectedIndices] = useModelState<number[]>("compare_page_expected_indices");
  const [comparePageLoading] = useModelState<boolean>("compare_page_loading");
  const [comparePageGeneration] = useModelState<number>("compare_page_generation");
  const [comparePagePanelBytes] = useModelState<DataView>("compare_page_panel_bytes");
  const [comparePagePanelFrameIdx] = useModelState<number>("compare_page_panel_frame_idx");
  const [comparePagePanelSlot] = useModelState<number>("compare_page_panel_slot");
  const [comparePagePanelSequence] = useModelState<number>("compare_page_panel_sequence");
  const [comparePagePanelCached] = useModelState<boolean>("compare_page_panel_cached");
  const [comparePageCachedIndices] = useModelState<number[]>("compare_page_cached_indices");
  const [comparePageCacheState] = useModelState<string>("compare_page_cache_state");
  const [progressiveComparePage, setProgressiveComparePage] = React.useState<ProgressiveComparePage | null>(null);
  const progressiveCompareGenerationRef = React.useRef<string | null>(null);
  const progressiveCompareLastNumericGenerationRef = React.useRef<number | null>(null);
  const progressiveComparePendingGenerationRef = React.useRef(0);
  const comparePagePaintAckKeyRef = React.useRef<string | null>(null);
  const comparePagePaintClientIdRef = React.useRef(
    `show4dstem-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`,
  );

  React.useEffect(() => {
    try {
      model.send({
        type: "show4dstem_frontend_ready",
        version: 1,
      });
    } catch {
      // A closing notebook comm has no mounted UI left to initialize.
    }
    try {
      model.send({
        type: "compare_page_paint_capability",
        version: 1,
        active: true,
        client_id: comparePagePaintClientIdRef.current,
      });
    } catch {
      // A closing notebook comm has no mounted UI left to acknowledge.
    }
    return () => {
      try {
        model.send({
          type: "compare_page_paint_capability",
          version: 1,
          active: false,
          client_id: comparePagePaintClientIdRef.current,
        });
      } catch {
        // The comm can already be gone during notebook teardown.
      }
    };
  }, [model]);

  const acknowledgeFreshComparePagePaint = React.useCallback((
    page: ProgressiveComparePage,
    paintedIndices: number[],
  ) => {
    const acknowledgement = freshVisibleComparePagePaintAck(
      page,
      paintedIndices,
      comparePagePaintAckKeyRef.current,
    );
    if (!acknowledgement) return;
    try {
      model.send(acknowledgement.message);
      comparePagePaintAckKeyRef.current = acknowledgement.key;
    } catch {
      // The delayed after-paint callback may outlive a closing notebook comm.
    }
  }, [model]);

  React.useEffect(() => {
    const handler = (
      content: ComparePageMessage,
      buffers?: Array<DataView | ArrayBuffer | Uint8Array>,
    ) => {
      const type = String(content?.type || "");
      if (type === "compare_page_start") {
        const incomingGeneration = compareMessageGeneration(content);
        if (incomingGeneration === null) return;
        const incomingPageValue = content.page_idx ?? content.page;
        const incomingPage = Number(incomingPageValue);
        const currentPage = Math.max(0, Math.round(Number(model.get("compare_page_idx")) || 0));
        if (incomingPageValue !== undefined && Number.isFinite(incomingPage) && Math.round(incomingPage) !== currentPage) {
          recordComparePageStaleDrop();
          return;
        }
        const nextNumber = Number(incomingGeneration);
        const lastNumber = progressiveCompareLastNumericGenerationRef.current;
        if (
          Number.isFinite(nextNumber)
          && lastNumber !== null
          && nextNumber <= lastNumber
        ) {
          recordComparePageStaleDrop();
          return;
        }
        const next = beginProgressiveComparePage(content);
        if (!next) return;
        progressiveCompareGenerationRef.current = next.generation;
        if (Number.isFinite(nextNumber)) progressiveCompareLastNumericGenerationRef.current = nextNumber;
        setProgressiveComparePage((current) => retainCachedProgressiveComparePanels(next, current));
        return;
      }

      if (type !== "compare_panel" && type !== "compare_page_complete") return;
      const generation = compareMessageGeneration(content);
      if (generation === null || generation !== progressiveCompareGenerationRef.current) {
        recordComparePageStaleDrop();
        return;
      }
      setProgressiveComparePage((current) => {
        if (!current || current.generation !== generation) return current;
        if (type === "compare_panel") {
          return mergeProgressiveComparePanel(
            current,
            content,
            buffers,
            Math.max(1, shapeRows * shapeCols),
          ) ?? current;
        }
        return completeProgressiveComparePage(current, content) ?? current;
      });
    };
    model.on("msg:custom", handler);
    return () => model.off("msg:custom", handler);
  }, [model, shapeCols, shapeRows]);

  React.useEffect(() => {
    const sequence = Math.max(0, Math.round(Number(comparePagePanelSequence) || 0));
    const frame = Math.round(Number(comparePagePanelFrameIdx));
    if (sequence <= 0 || frame < 0 || !comparePagePanelBytes || comparePagePanelBytes.byteLength === 0) return;
    const generation = String(Math.round(Number(comparePageGeneration) || 0));
    const page = Math.max(0, Math.round(Number(model.get("compare_page_idx")) || 0));
    setProgressiveComparePage((current) => {
      const expected = Array.isArray(comparePageExpectedIndices) ? comparePageExpectedIndices : [];
      const base = current && current.generation === generation
        ? current
        : {
            generation,
            page,
            expectedIndices: [...expected],
            panels: new Map<number, Float32Array>(),
            cachedIndices: new Set<number>(),
            cacheState: "off" as const,
            loading: true,
            complete: false,
          };
      const withCacheMetadata = mergeProgressiveCompareCacheMetadata(
        base,
        comparePageCachedIndices,
        comparePageCacheState,
      );
      return mergeProgressiveComparePanel(
        withCacheMetadata,
        {
          type: "compare_panel",
          generation,
          page_idx: page,
          frame_idx: frame,
          slot: Math.round(Number(comparePagePanelSlot) || 0),
          cached: Boolean(comparePagePanelCached),
        },
        [comparePagePanelBytes],
        Math.max(1, shapeRows * shapeCols),
      ) ?? withCacheMetadata;
    });
  }, [
    comparePageCachedIndices,
    comparePageCacheState,
    comparePageExpectedIndices,
    comparePageGeneration,
    comparePagePanelBytes,
    comparePagePanelCached,
    comparePagePanelFrameIdx,
    comparePagePanelSequence,
    comparePagePanelSlot,
    model,
    shapeCols,
    shapeRows,
  ]);

  React.useEffect(() => {
    if (!comparePageProgressiveEnabled) {
      progressiveCompareGenerationRef.current = null;
      setProgressiveComparePage(null);
    }
  }, [comparePageProgressiveEnabled]);

  React.useEffect(() => {
    if (!comparePageProgressiveEnabled) return;
    const generation = String(Math.round(Number(comparePageGeneration) || 0));
    const expected = Array.isArray(comparePageExpectedIndices) ? comparePageExpectedIndices : [];
    const page = Math.max(0, Math.round(Number(model.get("compare_page_idx")) || 0));
    setProgressiveComparePage((current) => {
      if (current?.generation.startsWith("pending:") && current.generation !== generation) {
        return current;
      }
      if (!current && expected.length === 0) return current;
      const base = current && current.generation === generation
        ? {
            ...current,
            expectedIndices: expected.length > 0 ? [...expected] : current.expectedIndices,
          }
        : {
            generation,
            page,
            expectedIndices: [...expected],
            panels: new Map<number, Float32Array>(),
            cachedIndices: new Set<number>(),
            cacheState: "off" as const,
            loading: Boolean(comparePageLoading),
            complete: false,
          };
      progressiveCompareGenerationRef.current = generation;
      return mergeProgressiveCompareCacheMetadata(
        base,
        comparePageCachedIndices,
        comparePageCacheState,
      );
    });
  }, [
    comparePageCachedIndices,
    comparePageCacheState,
    comparePageExpectedIndices,
    comparePageGeneration,
    comparePageLoading,
    comparePageProgressiveEnabled,
    model,
  ]);

  React.useEffect(() => {
    if (comparePageLoading) return;
    const generation = String(Math.round(Number(comparePageGeneration) || 0));
    const expected = Array.isArray(comparePageExpectedIndices) ? comparePageExpectedIndices : [];
    const durable = Array.isArray(comparePanelIndices) ? comparePanelIndices : [];
    if (shouldClearProgressiveComparePage(false, expected, durable)) {
      progressiveCompareGenerationRef.current = null;
      setProgressiveComparePage(null);
      return;
    }
    setProgressiveComparePage((current) => {
      if (!current || current.generation !== generation || current.complete) return current;
      return {
        ...mergeProgressiveCompareCacheMetadata(
          current,
          comparePageCachedIndices,
          comparePageCacheState,
        ),
        expectedIndices: reconcileCompletedCompareIndices(
          current.expectedIndices,
          durable,
        ),
        loading: false,
        complete: true,
      };
    });
  }, [
    comparePageCachedIndices,
    comparePageCacheState,
    comparePageExpectedIndices,
    comparePageGeneration,
    comparePageLoading,
    comparePanelIndices,
  ]);

  // Profile line state (synced with Python)
  const [profileLine, setProfileLine] = useModelState<{row: number; col: number}[]>("profile_line");
  const [profileWidth] = useModelState<number>("profile_width");

  // Auto-detection trigger
  // ─────────────────────────────────────────────────────────────────────────
  // Local State (UI-only, not synced to Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [localKCol, setLocalKCol] = React.useState(roiCenterCol);
  const [localKRow, setLocalKRow] = React.useState(roiCenterRow);
  const [localPosRow, setLocalPosRow] = React.useState(posRow);
  const [localPosCol, setLocalPosCol] = React.useState(posCol);
  const scanPositionPendingRef = React.useRef<[number, number] | null>(null);
  const scanPositionRafRef = React.useRef<number | null>(null);
  const scanPositionTimerRef = React.useRef<number | null>(null);
  const scanPositionLastSyncRef = React.useRef(0);
  const scanPositionOptimisticRef = React.useRef<[number, number] | null>(null);
  const scanPositionCurrentRef = React.useRef<[number, number]>([Math.round(posRow), Math.round(posCol)]);
  const writeQueuedScanPosition = React.useCallback(() => {
    const pending = scanPositionPendingRef.current;
    if (!pending) return;
    const [row, col] = pending;
    scanPositionPendingRef.current = null;
    scanPositionCurrentRef.current = [row, col];
    model.set("pos_row", row);
    model.set("pos_col", col);
    model.save_changes();
    scanPositionLastSyncRef.current = performance.now();
  }, [model]);
  const writeRoiCenterModel = React.useCallback((row: number, col: number) => {
    model.set("roi_center_row", row);
    model.set("roi_center_col", col);
    model.set("roi_center", [row, col]);
    model.save_changes();
  }, [model]);
  const queueScanPosition = React.useCallback((row: number, col: number) => {
    const current = scanPositionPendingRef.current
      ?? scanPositionOptimisticRef.current
      ?? scanPositionCurrentRef.current;
    if (current[0] === row && current[1] === col) return;
    scanPositionPendingRef.current = [row, col];
    scanPositionOptimisticRef.current = [row, col];
    if (scanPositionTimerRef.current === null && scanPositionRafRef.current === null) {
      const elapsed = performance.now() - scanPositionLastSyncRef.current;
      const delay = Math.max(0, 33 - elapsed);
      scanPositionTimerRef.current = window.setTimeout(() => {
        scanPositionTimerRef.current = null;
        scanPositionRafRef.current = requestAnimationFrame(() => {
          scanPositionRafRef.current = null;
          writeQueuedScanPosition();
        });
      }, delay);
    }
  }, [writeQueuedScanPosition]);
  const flushScanPosition = React.useCallback(() => {
    if (scanPositionTimerRef.current !== null) {
      window.clearTimeout(scanPositionTimerRef.current);
      scanPositionTimerRef.current = null;
    }
    if (scanPositionRafRef.current !== null) {
      cancelAnimationFrame(scanPositionRafRef.current);
      scanPositionRafRef.current = null;
    }
    writeQueuedScanPosition();
  }, [writeQueuedScanPosition]);
  React.useEffect(() => {
    return () => {
      if (scanPositionTimerRef.current !== null) {
        window.clearTimeout(scanPositionTimerRef.current);
      }
      if (scanPositionRafRef.current !== null) {
        cancelAnimationFrame(scanPositionRafRef.current);
        scanPositionRafRef.current = null;
      }
    };
  }, []);
  const [isDraggingDP, setIsDraggingDP] = React.useState(false);
  // rAF coalescing for ROI drag: collapse rapid mousemove events into ≤1
  // Python comm message per animation frame. Without this, drag fires 60+
  // events/sec at >100ms Python compute each → queue piles up → laggy UX.
  const roiCenterPendingRef = React.useRef<[number, number] | null>(null);
  const roiCenterRafRef = React.useRef<number | null>(null);
  const flushRoiCenter = React.useCallback(() => {
    if (roiCenterPendingRef.current) {
      const [r, c] = roiCenterPendingRef.current;
      writeRoiCenterModel(r, c);
      roiCenterPendingRef.current = null;
    }
    roiCenterRafRef.current = null;
  }, [writeRoiCenterModel]);
  const queueRoiCenter = React.useCallback((row: number, col: number) => {
    roiCenterPendingRef.current = [row, col];
    if (roiCenterRafRef.current === null) {
      roiCenterRafRef.current = requestAnimationFrame(flushRoiCenter);
    }
  }, [flushRoiCenter]);
  // rAF coalescing for ROI RADIUS drag — same reason as center: a no-bin BF/DF
  // recompute is ~100ms in Python, and a resize-drag fires 60+ mousemoves/sec.
  // Without coalescing every move becomes a queued comm message + recompute, so
  // the image lags ~1s behind the cursor. Collapse to <=1 radius per frame.
  // Local radius drives the ring/handle render INSTANTLY during a resize drag, so
  // the ring tracks the cursor with no snap-back, while the model trait (which
  // triggers the ~100ms Python recompute) is sent at most once per recompute.
  const [localRoiRadius, setLocalRoiRadius] = React.useState<number | null>(null);
  // Effective radius used by ALL render/hit-test code below: the live local value
  // while dragging, else the model value. Keeps the ring glued to the cursor.
  const roiRadius = localRoiRadius != null ? localRoiRadius : roiRadiusModel;
  // Coalesce radius writes with requestAnimationFrame, always flushing the LATEST
  // radius (issue #751). Do NOT gate sends on virtual_image_bytes: the old guard
  // waited for the VI bytes to change before sending the next radius, so if a send
  // didn't land changed bytes the final drag value stayed local and Python never
  // recomputed — the hand-drag resize silently did nothing. rAF flush is robust:
  // one Python recompute per frame, last-value-wins, no stuck in-flight guard.
  const roiRadiusPendingRef = React.useRef<number | null>(null);
  const roiRadiusRafRef = React.useRef<number | null>(null);
  const flushRoiRadius = React.useCallback(() => {
    if (roiRadiusPendingRef.current !== null) {
      const r = roiRadiusPendingRef.current;
      roiRadiusPendingRef.current = null;
      model.set("roi_radius", r);
      model.save_changes();
    }
    roiRadiusRafRef.current = null;
  }, [model]);
  const sendRoiRadius = React.useCallback((radius: number) => {
    roiRadiusPendingRef.current = radius;
    if (roiRadiusRafRef.current === null) {
      roiRadiusRafRef.current = requestAnimationFrame(flushRoiRadius);
    }
  }, [flushRoiRadius]);
  const dpRoiInteractiveRef = React.useRef(false);
  const requestViFinalizeRef = React.useRef<(() => void) | null>(null);
  const requestCompareViLiveRef = React.useRef<(() => void) | null>(null);
  const compareViLiveRafRef = React.useRef<number | null>(null);
  const compareViLiveInFlightRef = React.useRef(false);
  const compareViLivePendingRef = React.useRef(false);
  const requestDpFrameLiveRef = React.useRef<(() => void) | null>(null);
  const dpFrameLiveRafRef = React.useRef<number | null>(null);
  const requestCompareViLive = React.useCallback(() => {
    if (compareViLiveRafRef.current !== null) return;
    compareViLiveRafRef.current = requestAnimationFrame(() => {
      compareViLiveRafRef.current = null;
      requestCompareViLiveRef.current?.();
    });
  }, []);
  const requestDpFrameLive = React.useCallback(() => {
    if (dpFrameLiveRafRef.current !== null) return;
    dpFrameLiveRafRef.current = requestAnimationFrame(() => {
      dpFrameLiveRafRef.current = null;
      requestDpFrameLiveRef.current?.();
    });
  }, []);
  React.useEffect(() => () => {
    if (compareViLiveRafRef.current !== null) {
      cancelAnimationFrame(compareViLiveRafRef.current);
      compareViLiveRafRef.current = null;
    }
    if (dpFrameLiveRafRef.current !== null) {
      cancelAnimationFrame(dpFrameLiveRafRef.current);
      dpFrameLiveRafRef.current = null;
    }
  }, []);
  const finishDpRoiInteraction = React.useCallback(() => {
    const wasInteractive = dpRoiInteractiveRef.current;
    dpRoiInteractiveRef.current = false;
    if (compareViLiveRafRef.current !== null) {
      cancelAnimationFrame(compareViLiveRafRef.current);
      compareViLiveRafRef.current = null;
    }
    compareViLivePendingRef.current = false;
    flushRoiCenter();
    flushRoiRadius();
    if (wasInteractive) {
      requestAnimationFrame(() => {
        requestViFinalizeRef.current?.();
      });
    }
  }, [flushRoiCenter, flushRoiRadius]);
  const [isDraggingVI, setIsDraggingVI] = React.useState(false);
  const [isDraggingFFT, setIsDraggingFFT] = React.useState(false);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number, y: number, panX: number, panY: number } | null>(null);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false); // For annular inner handle
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const resizeAspectRef = React.useRef<number | null>(null);
  // VI ROI drag/resize states (same pattern as DP)
  const [isDraggingViRoi, setIsDraggingViRoi] = React.useState(false);
  const [isDraggingViRoiResize, setIsDraggingViRoiResize] = React.useState(false);
  const [isHoveringViRoiResize, setIsHoveringViRoiResize] = React.useState(false);
  // Independent colormaps for DP and VI panels
  const [showDpColorbar, setShowDpColorbar] = useModelState<boolean>("dp_show_colorbar");
  const [dpColormap, setDpColormap] = useModelState<string>("dp_colormap");
  const [viColormap, setViColormap] = useModelState<string>("vi_colormap");
  // vmin/vmax percentile clipping (0-100)
  const [dpVminPct, setDpVminPct] = useModelState<number>("dp_vmin_pct");
  const [dpVmaxPct, setDpVmaxPct] = useModelState<number>("dp_vmax_pct");
  const [viVminPct, setViVminPct] = useModelState<number>("vi_vmin_pct");
  const [viVmaxPct, setViVmaxPct] = useModelState<number>("vi_vmax_pct");
  // Absolute intensity bounds (override percentile sliders when both set)
  const [traitDpVmin] = useModelState<number | null>("dp_vmin");
  const [traitDpVmax] = useModelState<number | null>("dp_vmax");
  const [traitViVmin] = useModelState<number | null>("vi_vmin");
  const [traitViVmax] = useModelState<number | null>("vi_vmax");
  // Scale mode: "linear" | "log"
  const [dpScaleMode, setDpScaleMode] = useModelState<"linear" | "log">("dp_scale_mode");
  const [viScaleMode, setViScaleMode] = useModelState<"linear" | "log">("vi_scale_mode");
  // VI auto-contrast (1st/99th percentile clip) + Smooth (CSS bilinear blit).
  // DP doesn't need them — Bragg spots read best with the slider's percentile
  // range and nearest-neighbor blit.
  const [viAutoContrast, setViAutoContrast] = useModelState<boolean>("vi_auto_contrast");
  const [viSmooth, setViSmooth] = useModelState<boolean>("vi_smooth");
  const viPreAutoPctRef = React.useRef<[number, number] | null>(null);
  const toggleViAutoContrast = React.useCallback((on: boolean) => {
    if (on) {
      viPreAutoPctRef.current = [viVminPct, viVmaxPct];
    } else if (viPreAutoPctRef.current) {
      const [vmn, vmx] = viPreAutoPctRef.current;
      setViVminPct(vmn);
      setViVmaxPct(vmx);
      viPreAutoPctRef.current = null;
    }
    setViAutoContrast(on);
  }, [setViAutoContrast, setViVmaxPct, setViVminPct, viVmaxPct, viVminPct]);

  // VI ROI state (real-space region selection for summed DP) - synced with Python
  const [viRoiMode, setViRoiMode] = useModelState<string>("vi_roi_mode");
  const [viRoiCenterRow, setViRoiCenterRow] = useModelState<number>("vi_roi_center_row");
  const [viRoiCenterCol, setViRoiCenterCol] = useModelState<number>("vi_roi_center_col");
  const [viRoiRadius, setViRoiRadius] = useModelState<number>("vi_roi_radius");
  const [viRoiWidth, setViRoiWidth] = useModelState<number>("vi_roi_width");
  const [viRoiHeight, setViRoiHeight] = useModelState<number>("vi_roi_height");
  // Local VI ROI center for smooth dragging
  const [localViRoiCenterRow, setLocalViRoiCenterRow] = React.useState(viRoiCenterRow || 0);
  const [localViRoiCenterCol, setLocalViRoiCenterCol] = React.useState(viRoiCenterCol || 0);
  const [viRoiDpBytes] = useModelState<DataView>("vi_roi_dp_bytes");
  const [viRoiReduce, setViRoiReduce] = useModelState<string>("vi_roi_reduce");
  const [webgpuDpcReady, setWebgpuDpcReady] = React.useState(false);
  const [viGpuVersion, setViGpuVersion] = React.useState(0);
  const [viGpuRetainedReady, setViGpuRetainedReady] = React.useState(false);
  const viGpuImageRef = React.useRef<ViGpuImage | null>(null);
  // GPU-resident compare panels: frame index -> engine colormap slot. Written by
  // the interactive compare recompute (no readback), consumed by the grid painter.
  const [compareGpuVersion, setCompareGpuVersion] = React.useState(0);
  const compareGpuSlotsRef = React.useRef(new Map<number, number>());
  const compareGpuHistogramGenRef = React.useRef(0);
  const compareGpuRenderNowRef = React.useRef<(() => number) | null>(null);
  const compareIncrementalRef = React.useRef<{
    mask: Uint32Array;
    buffers: Map<number, GPUBuffer>;
    indicesKey: string;
  } | null>(null);
  const liveCompareViStatsRef = React.useRef({
    computeTimes: [] as number[],
    paintTimes: [] as number[],
    lastComputeMs: 0,
    lastPaintMs: 0,
    lastAdoptedPanels: 0,
    lastRequestedPanels: 0,
    lastPaintedPanels: 0,
  });
  const publishLiveCompareViStats = React.useCallback((
    event: string,
    detail: { ms?: number; adoptedPanels?: number; requestedPanels?: number; paintedPanels?: number; addedPixels?: number; removedPixels?: number },
  ) => {
    const now = performance.now();
    const stats = liveCompareViStatsRef.current;
    if (event !== "paint") {
      if ((detail.adoptedPanels ?? 0) > 0) stats.computeTimes.push(now);
      stats.lastComputeMs = detail.ms ?? 0;
      stats.lastAdoptedPanels = detail.adoptedPanels ?? 0;
      stats.lastRequestedPanels = detail.requestedPanels ?? 0;
    } else {
      stats.paintTimes.push(now);
      stats.lastPaintMs = now;
      stats.lastPaintedPanels = detail.paintedPanels ?? 0;
    }
    const cutoff = now - 1000;
    while (stats.computeTimes.length && stats.computeTimes[0] < cutoff) stats.computeTimes.shift();
    while (stats.paintTimes.length && stats.paintTimes[0] < cutoff) stats.paintTimes.shift();
    const recentCompute = stats.computeTimes.length;
    const recentPaint = stats.paintTimes.length;
    const payload = {
      event,
      gpuOnlyHotPath: true,
      computeFps: Math.round(recentCompute * 10) / 10,
      paintFps: Math.round(recentPaint * 10) / 10,
      lastComputeMs: Math.round(stats.lastComputeMs * 10) / 10,
      lastAdoptedPanels: stats.lastAdoptedPanels,
      lastRequestedPanels: stats.lastRequestedPanels,
      lastPaintedPanels: stats.lastPaintedPanels,
      addedPixels: detail.addedPixels ?? 0,
      removedPixels: detail.removedPixels ?? 0,
      updatedAtMs: Math.round(now),
      note: "computeFps counts fresh GPU virtual-image buffers only; paintFps counts WebGPU canvas presents, including repeated presents of the latest buffer.",
    };
    const statsWindow = window as unknown as {
      __sh4dLiveViStats?: Record<string, unknown>;
      __sh4dLiveCompareStats?: Record<string, unknown>;
      __sh4dLiveCompareHistory?: Record<string, unknown>[];
    };
    statsWindow.__sh4dLiveViStats = payload;
    statsWindow.__sh4dLiveCompareStats = payload;
    const history = statsWindow.__sh4dLiveCompareHistory ?? [];
    history.push(payload);
    if (history.length > 240) history.splice(0, history.length - 240);
    statsWindow.__sh4dLiveCompareHistory = history;
  }, []);
  const rawVirtualImageVersionRef = React.useRef(0);
  const viGpuColormapRef = React.useRef<GPUColormapEngine | null>(null);
  const viGpuColormapDeviceRef = React.useRef<GPUDevice | null>(null);
  const ensureViGpuColormap = React.useCallback((
    backend: DetectorCompute,
  ): GPUColormapEngine | null => {
    const maybeDevice = backend as unknown as { getDevice?: () => GPUDevice };
    const device = typeof maybeDevice.getDevice === "function" ? maybeDevice.getDevice() : null;
    if (!device) return null;
    if (!viGpuColormapRef.current || viGpuColormapDeviceRef.current !== device) {
      viGpuColormapRef.current?.destroy();
      viGpuColormapRef.current = new GPUColormapEngine(device);
      viGpuColormapDeviceRef.current = device;
    }
    return viGpuColormapRef.current;
  }, []);
  const clearViGpuDisplay = React.useCallback(() => {
    if (!viGpuImageRef.current) return;
    viGpuImageRef.current = null;
    publishShow4DSTEMViDisplay({ gpuBufferToDisplay: false, rendered: false });
    setViGpuVersion(v => v + 1);
  }, []);

  React.useEffect(() => () => {
    viGpuImageRef.current = null;
    viGpuColormapRef.current?.destroy();
    viGpuColormapRef.current = null;
    viGpuColormapDeviceRef.current = null;
  }, []);

  // ── Offline WebGPU compute backend ──────────────────────────────────────
  // Small datasets ship the full uint16 stack (the `_offline_stack` trait); we
  // run the virtual-image and DP-from-ROI reductions in WebGPU right here, with
  // NO Python kernel. We play Python's role: on any detector/ROI trait change we
  // recompute and set `virtual_image_bytes` / `vi_roi_dp_bytes` on the model, so
  // every existing render effect works unchanged. Detector counts are integers,
  // so the browser masked-sum (u32 accumulate) is bit-exact to the kernel.
  const [offline] = useModelState<boolean>("offline");
  const [offlineBackendLoading, setOfflineBackendLoading] = React.useState(false);
  const [offlineBackendStatus, setOfflineBackendStatus] = React.useState("");
  const [offlineBackendError, setOfflineBackendError] = React.useState("");
  const h5SourceAvailable = Boolean(
    model.get("_h5_url")
    || model.get("_h5_urls")
    || model.get("_lazy_url")
    || model.get("_lazy_urls")
  );
  const [h5LocalFilesGranted, setH5LocalFilesGranted] = React.useState(show4DSTEMHasLocalFiles());
  const [h5LocalSourceStatus, setH5LocalSourceStatus] = React.useState("");
  const h5LocalInputRef = React.useRef<HTMLInputElement | null>(null);
  const requireLocalH5Files = (globalThis as { __QT_REQUIRE_LOCAL_H5_FILES?: boolean })
    .__QT_REQUIRE_LOCAL_H5_FILES === true;
  const localH5FolderName = React.useMemo(() => {
    try {
      const parts = decodeURIComponent(globalThis.location?.pathname || "").split("/").filter(Boolean);
      const viewerIdx = parts.lastIndexOf(".viewer");
      if (viewerIdx > 0) return parts[viewerIdx - 1];
      return parts.length >= 2 ? parts[parts.length - 2] : "";
    } catch {
      return "";
    }
  }, []);
  const onH5LocalInput = React.useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files ? Array.from(event.target.files) : [];
    if (!files.length) return;
    setShow4DSTEMLocalFiles(files);
    setH5LocalFilesGranted(true);
    setH5LocalSourceStatus(`${files.length} local HDF5 file${files.length === 1 ? "" : "s"} granted`);
  }, []);
  const grantH5LocalFiles = React.useCallback(async () => {
    setH5LocalSourceStatus("");
    const picker = (globalThis as Show4DSTEMWindow).showDirectoryPicker;
    if (picker) {
      try {
        const handle = await picker.call(globalThis, { mode: "read", startIn: "downloads" });
        const files = await collectShow4DSTEMLocalH5Files(
          handle as Parameters<typeof collectShow4DSTEMLocalH5Files>[0],
        );
        if (files.length) {
          setShow4DSTEMLocalFiles(files);
          setH5LocalFilesGranted(true);
          setH5LocalSourceStatus(`${files.length} local HDF5 file${files.length === 1 ? "" : "s"} granted`);
          return;
        }
        setH5LocalSourceStatus("No HDF5 files found in that folder");
      } catch (err) {
        const name = err instanceof DOMException ? err.name : "";
        if (name !== "AbortError") console.warn("Could not use directory picker for local HDF5 files", err);
      }
    }
    h5LocalInputRef.current?.click();
  }, []);
  React.useEffect(() => {
    if (!offline) {
      setWebgpuDpcReady(false);
      setOfflineBackendLoading(false);
      setOfflineBackendStatus("");
      setOfflineBackendError("");
      return;
    }
    let disposed = false;
    let detach: (() => void) | null = null;
    setOfflineBackendLoading(true);
    setOfflineBackendError("");
    setOfflineBackendStatus(h5SourceAvailable ? "Loading WebGPU source" : "Loading offline 4D-STEM data");
    (async () => {
      const scanRows = model.get("shape_rows"), scanCols = model.get("shape_cols");
      const detR = model.get("det_rows"), detC = model.get("det_cols");
      // Companion mode: fetch the stack from a sibling file (mount already happened
      // on the tiny widget-state JSON, and the inline initial virtual image is
      // already painted - so this runs in the background). Inline mode: read the
      // embedded bytes. Either way, create() infers uint8 vs uint16 from length.
      const offlineUrl = model.get("_offline_url") as string | undefined;
      const chunksMeta = model.get("_offline_chunks") as string | undefined;
      const bslz4Meta = model.get("_offline_bslz4") as string | undefined;
      const gunzip = async (b: Uint8Array) => new Uint8Array(await new Response(new Blob([b as BlobPart]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
      let compute: DetectorCompute | null = null;
      let cpuStack: Uint8Array | null = null;  // full decompressed stack for the per-frame probe (single-chunk only)
      // Multi-VOLUME (5D): several datasets, decoded LAZILY (decode-on-scrub) with a
      // small LRU of resident volumes. Only the viewed dataset (plus a few recent)
      // lives in VRAM, so it runs on a laptop regardless of how many h5 files - and
      // first paint is one decode, not N. The frame slider picks the active dataset.
      let computes: (DetectorCompute)[] = [];   // resident set for single / non-lazy paths
      let volMetas: any[] = [];                  // multi-volume descriptors (lazy)
      const volCache = new Map<number, DetectorCompute>();   // LRU: idx -> decoded volume
      const volLoadPromises = new Map<number, Promise<DetectorCompute | null>>();
      const inlineVolCache = new Map<number, DetectorCompute>(); // LRU for inline gzip 5D exports
      const compareResidentTarget = Math.max(3, Math.min(12, Math.max(1, Number(model.get("compare_max_panels") || 3))));
      const MAX_RESIDENT = compareResidentTarget; // recent / compare volumes kept hot for instant scrub and detector drags
      let latestResidentVolumeIndex: number | null = null;
      let volumeCount = 0;
      let getVol: ((idx: number) => Promise<DetectorCompute | null>) | null = null;
      let initialVolumeLoad: Promise<DetectorCompute | null> | null = null;
      let h5VolumePreload: Promise<void> | null = null;
      let h5VolumePreloadDone = false;
      // Browser HDF5 source: normal CLI exports use H5 URLs below. Lazy URLs are
      // an explicit internal source path, not the default CLI launch path.
      const lazyUrl = model.get("_lazy_url") as string | undefined;
      const lazyUrlsJson = model.get("_lazy_urls") as string | undefined;
      const lazyUrls = (() => {
        if (!lazyUrlsJson) return [] as string[];
        try {
          const parsed = JSON.parse(lazyUrlsJson);
          return Array.isArray(parsed) ? parsed.map((value) => String(value)).filter(Boolean) : [];
        } catch {
          return [] as string[];
        }
      })();
      const h5Url = model.get("_h5_url") as string | undefined;
      const h5UrlsJson = model.get("_h5_urls") as string | undefined;
      const h5Urls = (() => {
        if (!h5UrlsJson) return [] as string[];
        try {
          const parsed = JSON.parse(h5UrlsJson);
          return Array.isArray(parsed) ? parsed.map((value) => String(value)).filter(Boolean) : [];
        } catch {
          return [] as string[];
        }
      })();
      // Native uint16 HDF5 datasets are intentionally decoded one at a time so
      // the viewer never holds the full collection in VRAM. That is a
      // residency limit, not a preload target: the background loader still
      // visits every visible dataset sequentially and publishes each BF/DF panel
      // as soon as its volume is ready.
      const h5DecodeDtype = String(
        (globalThis as { __QT_H5_DECODE_DTYPE?: unknown }).__QT_H5_DECODE_DTYPE || "",
      ).toLowerCase();
      const h5UsesNativeU16 = h5DecodeDtype === "u2"
        || h5DecodeDtype === "uint16"
        || h5DecodeDtype === "native";
      const h5AllowU16MultiResident = (globalThis as { __QT_H5_ALLOW_U16_MULTI_PRELOAD?: boolean })
        .__QT_H5_ALLOW_U16_MULTI_PRELOAD === true;
      const h5RequestedResidentLimit = show4DSTEMGlobalInt(
        "__QT_H5_MAX_RESIDENT",
        MAX_RESIDENT,
        1,
        MAX_RESIDENT,
      );
      const h5ResidentLimit = h5UsesNativeU16 && !h5AllowU16MultiResident
        ? 1
        : h5RequestedResidentLimit;
      if (requireLocalH5Files && (h5Url || h5Urls.length) && !show4DSTEMHasLocalFiles()) {
        if (!disposed) {
          setOfflineBackendStatus("Waiting for local HDF5 files");
          setOfflineBackendLoading(false);
        }
        return;
      }
      const hasInlineViMaps = () => {
        const preset = model.get("vi_preset_maps_bytes") as DataView | undefined;
        const product = model.get("vi_product_maps_bytes") as DataView | undefined;
        return Boolean((preset && preset.byteLength > 0) || (product && product.byteLength > 0));
      };
      const h5FamilyBase = (sourceUrl: string): string =>
        sourceUrl.replace(/_master\.h5(?:[?#].*)?$/, "");
      const h5DataFileUrl = (sourceUrl: string, n: number): string =>
        `${h5FamilyBase(sourceUrl)}_data_${String(n).padStart(6, "0")}.h5`;
      const h5RawFileCache = new Map<string, Promise<ArrayBuffer | null>>();
      const h5FetchCached = (url: string): Promise<ArrayBuffer | null> => {
        const existing = h5RawFileCache.get(url);
        if (existing) return existing;
        const promise = fetch(url).then((resp) => resp.ok ? resp.arrayBuffer() : null);
        h5RawFileCache.set(url, promise);
        return promise;
      };
      const h5MasterInfoCache = new Map<string, Promise<ReturnType<typeof readH5MasterInfo> | null>>();
      const readH5MasterInfoCached = (sourceUrl: string, name = "master"): Promise<ReturnType<typeof readH5MasterInfo> | null> => {
        const existing = h5MasterInfoCache.get(sourceUrl);
        if (existing) return existing;
        const promise = h5FetchCached(sourceUrl).then((buffer) => buffer ? readH5MasterInfo(buffer, name) : null);
        h5MasterInfoCache.set(sourceUrl, promise);
        return promise;
      };
      const loadH5Compute = async (sourceUrl: string, label = "HDF5 source"): Promise<DetectorCompute | null> => {
        if (!disposed) setOfflineBackendStatus(`Loading ${label}`);
        let h5BadPx = new Uint32Array(0);
        const embeddedBadPxJson = model.get("_offline_bad_px") as string | undefined;
        const h5Uint8Lossless = Boolean(model.get("_h5_uint8_lossless"));
        const low8Only = h5Uint8Lossless ||
          (globalThis as { __QT_H5_FORCE_LOW8?: boolean }).__QT_H5_FORCE_LOW8 === true;
        if (low8Only) {
          (globalThis as { __BSLZ4_LOW8_ONLY?: boolean }).__BSLZ4_LOW8_ONLY = true;
        } else {
          (globalThis as { __BSLZ4_LOW8_ONLY?: boolean }).__BSLZ4_LOW8_ONLY = false;
        }
        let hasEmbeddedBadPx = false;
        if (embeddedBadPxJson) {
          try {
            const parsed = JSON.parse(embeddedBadPxJson) as number[];
            h5BadPx = new Uint32Array(parsed);
            hasEmbeddedBadPx = true;
          } catch (e) {
            console.warn("Could not parse embedded HDF5 hot-pixel mask; falling back to master HDF5 metadata", e);
          }
        }
        if (show4DSTEMHasLocalFiles() && /_master\.h5(?:[?#].*)?$/.test(sourceUrl)) {
          try {
            if (!disposed) setOfflineBackendStatus(`Loading local ${label}`);
            const sourceScanRows = show4DSTEMOptionalGlobalInt("__QT_H5_SOURCE_SCAN_ROWS", 1, 100000) ?? scanRows;
            const sourceScanCols = show4DSTEMOptionalGlobalInt("__QT_H5_SOURCE_SCAN_COLS", 1, 100000) ?? scanCols;
            const scanRegion = show4DSTEMOptionalGlobalRegion("__QT_H5_SCAN_REGION");
            const local = await loadShow4DSTEMLocalH5Master(sourceUrl, {
              scanRows: sourceScanRows,
              scanCols: sourceScanCols,
              scanRegion,
              embeddedBadPixelsJson: embeddedBadPxJson,
              decodeBatch: show4DSTEMOptionalGlobalInt("__QT_H5_DECODE_BATCH", 1, 16),
              groupSize: show4DSTEMOptionalGlobalInt("__QT_H5_LOCAL_GROUP", 1, 16),
              workerCount: show4DSTEMOptionalGlobalInt("__QT_H5_LOCAL_WORKERS", 0, 8),
              detBin: show4DSTEMOptionalGlobalInt("__QT_H5_DET_BIN", 1, 16),
              // Lossless decode override ("u2"/"uint16"/"native"): routes to the fused
              // native-uint16 kernel so counts above 255 survive (the u8 default wraps).
              decodeDtype: (globalThis as { __QT_H5_DECODE_DTYPE?: unknown }).__QT_H5_DECODE_DTYPE as
                Parameters<typeof loadShow4DSTEMLocalH5Master>[1] extends infer O
                  ? O extends { decodeDtype?: infer D } ? D : undefined
                  : undefined,
            });
            if (local) {
              const bytesPerPixel = local.mode === 2 ? 4 : local.mode === 0 ? 2 : 1;
              const decodedGB = local.profile.frames * local.detSize * bytesPerPixel / 1e9;
              (window as unknown as { __loadprof: unknown }).__loadprof = {
                ...local.profile,
                fetchedCompressedGB: local.profile.compressedGB,
                decodedGB: +decodedGB.toFixed(1),
                localFiles: true,
                decodeIncludesUpload: true,
                decompressGBps: +(decodedGB / Math.max(0.001, local.profile.decompressMs / 1000)).toFixed(2),
                targetFrames: local.scanCount,
              };
              if (!disposed) setH5LocalSourceStatus(`Local HDF5 ${local.profile.totalMs} ms`);
              if (!disposed) setOfflineBackendStatus(`Local ${label} ready in ${local.profile.totalMs} ms`);
              const created = DetectorCompute.fromGpuChunks(
                local.device,
                local.chunks,
                local.scanCount,
                local.detSize,
                local.mode,
              );
              if (local.badPixels.length) created.badPx = local.badPixels;
              return created;
            }
            if (!disposed) setH5LocalSourceStatus("Selected local HDF5 files did not match this source");
            if (requireLocalH5Files) {
              throw new Error(`Selected local HDF5 files did not match ${sourceUrl}.`);
            }
          } catch (e) {
            if (requireLocalH5Files) throw e;
            if (!disposed) setH5LocalSourceStatus("Local HDF5 load failed; using URL fallback");
            if (!disposed) setOfflineBackendStatus(`Local ${label} failed; using URL fallback`);
            console.warn("Local HDF5 WebGPU load failed; falling back to URL fetch", e);
          }
        }
        if (/_master\.h5(?:[?#].*)?$/.test(sourceUrl)) {
          if (!disposed) setOfflineBackendStatus(`Reading ${label}`);
          const fetchWindow = show4DSTEMGlobalInt("__QT_H5_FETCH_WINDOW", 8, 4, 24);
          const fetchOne = async (n: number): Promise<ArrayBuffer | null> => {
            return await h5FetchCached(h5DataFileUrl(sourceUrl, n));
          };
          const inflight = new Map<number, Promise<ArrayBuffer | null>>();
          let next = 1;
          const gpuChunks: { buffer: GPUBuffer; startScan: number; nScan: number }[] = [];
          let startScan = 0, ds = 0, computeMode = 1, decodedBytes = 0;
          let maxDataFiles = Number.POSITIVE_INFINITY;
          let h5TotalFrames = hasEmbeddedBadPx ? scanRows * scanCols : 0;
          let dev: GPUDevice | null = null;
          // Honor the lossless decode override on the HTTP/streamed path too, so a
          // served bundle can request native uint16 exactly like the local-file path.
          const httpDecodeOverride = String((globalThis as { __QT_H5_DECODE_DTYPE?: unknown }).__QT_H5_DECODE_DTYPE || "").toLowerCase();
          const wantU16 = httpDecodeOverride === "u2" || httpDecodeOverride === "uint16" || httpDecodeOverride === "native";
          let decodeDtype: "uint8" | "uint16" | "float32" = "uint8";
          let sourceDtype: "unknown" | "uint8" | "uint16" | "uint32" | "float32" = "unknown";
          type QueuedBslz4Spec = Bslz4Spec & {
            startScan: number;
            nScan: number;
            decodeDtype: "uint8" | "uint16" | "float32";
            sourceDtype: "uint8" | "uint16" | "uint32" | "float32";
          };
          const decodeQueue: QueuedBslz4Spec[] = [];
          const decodeBatch = show4DSTEMGlobalInt("__QT_H5_DECODE_BATCH", 4, 1, 16);
          const decodeQueueTarget = show4DSTEMGlobalInt(
            "__QT_H5_DECODE_QUEUE",
            Math.max(decodeBatch * 2, fetchWindow),
            decodeBatch,
            16,
          );
          const __t0 = performance.now(); let __decMs = 0, __parseMs = 0, __fetchBytes = 0, __waitMs = 0, __masterMs = 0;
          let __uploadMs = 0, __buildMs = 0, __gpuWaitMs = 0, __decodeProfileMs = 0, __decodeCompressedMB = 0;
          let decodeDone = false;
          let decodeWake: (() => void) | null = null;
          let decodeSpaceWake: (() => void) | null = null;
          let maxDecodeQueue = 0;
          const wakeDecode = () => {
            const wake = decodeWake;
            decodeWake = null;
            if (wake) wake();
          };
          const wakeDecodeSpace = () => {
            const wake = decodeSpaceWake;
            decodeSpaceWake = null;
            if (wake) wake();
          };
          const nextDecodeGroup = async (): Promise<QueuedBslz4Spec[] | null> => {
            while (!decodeDone && decodeQueue.length < decodeBatch) {
              await new Promise<void>((resolve) => { decodeWake = resolve; });
            }
            if (!decodeQueue.length) return null;
            const group = decodeQueue.splice(0, Math.min(decodeBatch, decodeQueue.length));
            wakeDecodeSpace();
            return group;
          };
          const waitForDecodeSpace = async (): Promise<void> => {
            while (!decodeDone && decodeQueue.length >= decodeQueueTarget) {
              await new Promise<void>((resolve) => { decodeSpaceWake = resolve; });
            }
          };
          const decodeWorker = async (): Promise<boolean> => {
            while (true) {
              const group = await nextDecodeGroup();
              if (!group) return true;
              const groupDecodeDtype = group[0].decodeDtype;
              const groupSourceDtype = group[0].sourceDtype;
              const __dt = performance.now();
              const decoded = await decodeBslz4Batch(group, groupDecodeDtype, groupSourceDtype, decodeBatch);
              __decMs += performance.now() - __dt;
              if (!decoded) return false;
              dev = decoded.device;
              computeMode = decoded.mode;
              __uploadMs += decoded.profile.uploadMs;
              __buildMs += decoded.profile.buildMs;
              __gpuWaitMs += decoded.profile.gpuWaitMs;
              __decodeProfileMs += decoded.profile.totalMs;
              __decodeCompressedMB += decoded.profile.compressedMB;
              decoded.buffers.forEach((buffer, i) => {
                const spec = group[i];
                gpuChunks.push({ buffer, startScan: spec.startScan, nScan: spec.nScan });
              });
            }
          };
          try {
            const decodePromise = decodeWorker();
            try {
              const __mt = performance.now();
              const masterInfo = await readH5MasterInfoCached(sourceUrl, "master");
              __masterMs += performance.now() - __mt;
              if (masterInfo) {
                if (!hasEmbeddedBadPx && masterInfo.badPixels.length) h5BadPx = new Uint32Array(masterInfo.badPixels);
                h5TotalFrames = Math.max(0, Math.round(Number(masterInfo.totalFrames || h5TotalFrames || 0)));
                if (Number.isFinite(masterInfo.dataFileCount) && Number(masterInfo.dataFileCount) > 0) {
                  maxDataFiles = Math.round(Number(masterInfo.dataFileCount));
                }
              }
            } catch (e) {
              console.warn("Could not read HDF5 master metadata; continuing without detector mask/file-count bounds", e);
            }
            const initialFetchLimit = Number.isFinite(maxDataFiles)
              ? Math.min(fetchWindow, maxDataFiles)
              : fetchWindow;
            for (; next <= initialFetchLimit; next++) {
              inflight.set(next, fetchOne(next));
            }
            for (let n = 1; !disposed; n++) {
              const p = inflight.get(n);
              if (!p) break;
              inflight.delete(n);
              const __wt = performance.now();
              const buf = await p;
              __waitMs += performance.now() - __wt;
              h5RawFileCache.delete(h5DataFileUrl(sourceUrl, n));
              if (!buf) break;
              if (!disposed) setOfflineBackendStatus(`Reading ${label}: data file ${n}`);
              __fetchBytes += buf.byteLength;
              const __pt = performance.now();
              const vol = readH5Volume(buf, "merged");
              __parseMs += performance.now() - __pt;
              if (!Number.isFinite(maxDataFiles)) {
                maxDataFiles = Math.ceil((h5TotalFrames || scanRows * scanCols) / Math.max(1, vol.nFrames));
              }
              ds = vol.detSize;
              if (sourceDtype !== "unknown" && sourceDtype !== vol.srcDtype) {
                throw new Error(`Mixed HDF5 source dtypes are not supported in one browser load: ${sourceDtype} and ${vol.srcDtype}.`);
              }
              sourceDtype = vol.srcDtype;
              decodeDtype = vol.srcDtype === "float32" ? "float32" : (wantU16 && vol.srcDtype === "uint16") ? "uint16" : "uint8";
              decodedBytes += vol.nFrames * vol.detSize * (decodeDtype === "float32" ? 4 : decodeDtype === "uint16" ? 2 : 1);
              {
                // Every chunk of this file, each at its own scan offset. Taking only
                // chunks[0] silently dropped all later chunks of multi-chunk files,
                // leaving the virtual image black outside the first chunk's scan rows.
                let chunkStart = startScan;
                for (const chunk of vol.chunks) {
                  await waitForDecodeSpace();
                  decodeQueue.push({
                    ...chunk,
                    startScan: chunkStart,
                    nScan: chunk.nFrames,
                    decodeDtype,
                    sourceDtype: vol.srcDtype,
                  });
                  chunkStart += chunk.nFrames;
                }
                maxDecodeQueue = Math.max(maxDecodeQueue, decodeQueue.length);
                wakeDecode();
              }
              startScan += vol.nFrames;
              if (next <= maxDataFiles) {
                inflight.set(next, fetchOne(next));
                next++;
              }
            }
            decodeDone = true;
            wakeDecode();
            if (!(await decodePromise)) {
              throw new Error("HDF5 BSLZ4 decode failed.");
            }
          } catch (e) {
            decodeDone = true;
            wakeDecode();
            wakeDecodeSpace();
            gpuChunks.forEach((c) => c.buffer.destroy());
            throw e;
          }
          const __decGB = decodedBytes / 1e9;
          const profileDevice = dev as GPUDevice | null;
          (window as unknown as { __loadprof: unknown }).__loadprof = { totalMs: Math.round(performance.now() - __t0),
            fetchedCompressedGB: +(__fetchBytes / 1e9).toFixed(1), decodedGB: +__decGB.toFixed(1),
            sourceDtype, decodeDtype, chunks: gpuChunks.length, frames: startScan,
            badPixels: h5BadPx.length,
            adapterInfo: getGPUInfo(), softwareAdapter: isSoftwareGPUAdapter(),
            timestampQuery: Boolean(profileDevice?.features.has("timestamp-query")),
            subgroups: Boolean(profileDevice?.features.has("subgroups" as GPUFeatureName)),
            maxBufferGB: profileDevice ? +(Number(profileDevice.limits.maxBufferSize || 0) / 1e9).toFixed(2) : null,
            maxStorageBufferGB: profileDevice ? +(Number(profileDevice.limits.maxStorageBufferBindingSize || 0) / 1e9).toFixed(2) : null,
            dataFilesExpected: Number.isFinite(maxDataFiles) ? maxDataFiles : null,
            decodeBatch,
            fetchWindow,
            decodeQueueTarget,
            pipelineMode: "fetch-parse-decode-queue",
            maxDecodeQueue,
            targetFrames: h5TotalFrames || scanRows * scanCols,
            fetchWaitMs: Math.round(__waitMs), masterFetchMs: Math.round(__masterMs),
            decompressMs: Math.round(__decMs), parseMs: Math.round(__parseMs),
            uploadMs: Math.round(__uploadMs), decodeBuildMs: Math.round(__buildMs),
            gpuWaitMs: Math.round(__gpuWaitMs), decodeProfileMs: Math.round(__decodeProfileMs),
            decodeCompressedMB: Math.round(__decodeCompressedMB),
            decodeIncludesUpload: true,
            decompressGBps: +(__decGB / Math.max(0.001, __decMs / 1000)).toFixed(2) };
          const created = dev ? DetectorCompute.fromGpuChunks(dev, gpuChunks, scanRows * scanCols, ds, computeMode) : null;
          if (created && h5BadPx.length) created.badPx = h5BadPx;
          if (!disposed) setOfflineBackendStatus(`${label} ready`);
          return created;
        }
        if (!disposed) setOfflineBackendStatus(`Reading ${label}`);
        const h5Buffer = await (await fetch(sourceUrl)).arrayBuffer();
        try {
          const masterInfo = readH5MasterInfo(h5Buffer, "merged");
          if (masterInfo.badPixels.length) h5BadPx = new Uint32Array(masterInfo.badPixels);
        } catch (e) {
          console.warn("Could not read HDF5 hot-pixel mask; continuing without detector mask", e);
        }
        const vol = readH5Volume(h5Buffer, "merged");
        const decodeDtype = vol.srcDtype === "float32" ? "float32" : "uint8";
        let mergedStart = 0;
        const mergedSpecs = vol.chunks.map((chunk) => {
          const spec = { ...chunk, startScan: mergedStart, nScan: chunk.nFrames };
          mergedStart += chunk.nFrames;
          return spec;
        });
        const created = await DetectorCompute.createFromBslz4Chunked(mergedSpecs, scanRows * scanCols, vol.detSize, decodeDtype, vol.srcDtype);
        if (created && h5BadPx.length) created.badPx = h5BadPx;
        if (!disposed) setOfflineBackendStatus(`${label} ready`);
        return created;
      };
      if (lazyUrl || lazyUrls.length) {
        const urls = lazyUrls.length ? lazyUrls : [lazyUrl!];
        volumeCount = urls.length;
        getVol = async (idx: number) => {
          const clamped = Math.max(0, Math.min(urls.length - 1, idx));
          if (volCache.has(clamped)) return volCache.get(clamped)!;
          const existingLoad = volLoadPromises.get(clamped);
          if (existingLoad) return await existingLoad;
          const loadPromise = (async () => {
            const cc = await LazyShow4DSTEM.create(urls[clamped]) as unknown as DetectorCompute | null;
            if (cc) {
              volCache.set(clamped, cc);
              latestResidentVolumeIndex = clamped;
              const activeFrame = Math.max(0, Math.min(urls.length - 1, model.get("frame_idx") | 0));
              while (volCache.size > MAX_RESIDENT) {
                const old = [...volCache.keys()].find((k) => k !== clamped && k !== activeFrame);
                if (old === undefined) break;
                volCache.get(old)!.dispose(); volCache.delete(old);
              }
            }
            return cc;
          })().finally(() => { volLoadPromises.delete(clamped); });
          volLoadPromises.set(clamped, loadPromise);
          return await loadPromise;
        };
        const initialIdx = Math.max(0, Math.min(urls.length - 1, model.get("frame_idx") | 0));
        const lz = await getVol(initialIdx);
        compute = lz as unknown as DetectorCompute;
      } else if (h5Urls.length) {
        volumeCount = h5Urls.length;
        getVol = async (idx: number) => {
          const clamped = Math.max(0, Math.min(h5Urls.length - 1, idx));
          if (volCache.has(clamped)) return volCache.get(clamped)!;
          const existingLoad = volLoadPromises.get(clamped);
          if (existingLoad) return await existingLoad;
          const loadPromise = (async () => {
            const cc = await loadH5Compute(h5Urls[clamped], `dataset ${clamped + 1}/${h5Urls.length}`);
            if (cc) {
              volCache.set(clamped, cc);
              latestResidentVolumeIndex = clamped;
              const activeFrame = Math.max(0, Math.min(h5Urls.length - 1, model.get("frame_idx") | 0));
              while (volCache.size > h5ResidentLimit) {
                const old = [...volCache.keys()].find((k) => k !== clamped && k !== activeFrame);
                if (old === undefined) break;
                volCache.get(old)!.dispose(); volCache.delete(old);
              }
            }
            return cc;
          })().finally(() => { volLoadPromises.delete(clamped); });
          volLoadPromises.set(clamped, loadPromise);
          return await loadPromise;
        };
        const initialIdx = Math.max(0, Math.min(h5Urls.length - 1, model.get("frame_idx") | 0));
        const startH5Preloads = (): Promise<void> => {
          if (!getVol) return Promise.resolve();
          const preloadWantsU16 = h5UsesNativeU16;
          const allowU16MultiPreload = h5AllowU16MultiResident;
          const residentPreloadLimit = h5ResidentLimit;
          // Visit every dataset in the background. For uint16 this remains
          // serial (preloadWindow=1) and only one detector volume is resident,
          // but its BF/DF result is retained in a display slot before moving on.
          const maxPreload = Math.max(1, h5Urls.length);
          const defaultPreload = h5Urls.length;
          const preloadCount = show4DSTEMGlobalInt("__QT_H5_PRELOAD_VOLUMES", defaultPreload, 1, maxPreload);
          const preloadWindow = show4DSTEMGlobalInt(
            "__QT_H5_PRELOAD_WINDOW",
            1,
            1,
            Math.min(4, preloadCount),
          );
          const prefetchNext = !preloadWantsU16 &&
            (globalThis as { __QT_H5_PREFETCH_NEXT?: boolean }).__QT_H5_PREFETCH_NEXT !== false;
          const prefetchWindow = show4DSTEMGlobalInt("__QT_H5_PREFETCH_WINDOW", 2, 1, 8);
          const maxPrefetchFiles = show4DSTEMGlobalInt("__QT_H5_PREFETCH_FILES", 32, 1, 64);
          const order = [
            initialIdx,
            ...Array.from({ length: h5Urls.length }, (_v, i) => i).filter((i) => i !== initialIdx),
          ].slice(0, preloadCount);
          const startedAt = performance.now();
          const profile = {
            source: "h5_urls",
            volumeCount: h5Urls.length,
            requested: preloadCount,
            preloadWindow,
            decodeDtype: preloadWantsU16 ? "uint16" : "uint8",
            u16MultiPreloadAllowed: allowU16MultiPreload,
            residentPreloadLimit,
            prefetchNext,
            prefetchWindow,
            maxPrefetchFiles,
            prefetched: [] as { index: number; files: number; elapsedMs: number }[],
            order,
            completed: 0,
            failed: 0,
            volumes: [] as { index: number; elapsedMs: number; ok: boolean; error?: string }[],
            totalMs: 0,
          };
          (window as unknown as { __show4dstemH5PreloadProfile?: unknown }).__show4dstemH5PreloadProfile = profile;
          const prefetchStarted = new Set<number>();
          const prefetchVolume = async (index: number): Promise<void> => {
            if (!prefetchNext || prefetchStarted.has(index) || !/_master\.h5(?:[?#].*)?$/.test(h5Urls[index])) return;
            prefetchStarted.add(index);
            const t = performance.now();
            let files = 0;
            let fileLimit = maxPrefetchFiles;
            try {
              const masterInfo = await readH5MasterInfoCached(h5Urls[index], `prefetch-${index}`);
              if (Number.isFinite(masterInfo?.dataFileCount) && Number(masterInfo?.dataFileCount) > 0) {
                fileLimit = Math.min(fileLimit, Math.round(Number(masterInfo?.dataFileCount)));
              }
            } catch (error) {
              console.warn("Show4DSTEM HDF5 prefetch could not read master metadata", error);
            }
            const inflight = new Map<number, Promise<ArrayBuffer | null>>();
            let nextFile = 1;
            for (; nextFile <= Math.min(prefetchWindow, fileLimit); nextFile++) {
              inflight.set(nextFile, h5FetchCached(h5DataFileUrl(h5Urls[index], nextFile)));
            }
            for (let n = 1; n <= fileLimit && !disposed; n++) {
              const p = inflight.get(n);
              if (!p) break;
              inflight.delete(n);
              const buf = await p.catch(() => null);
              if (!buf) break;
              files += 1;
              if (nextFile <= fileLimit) {
                inflight.set(nextFile, h5FetchCached(h5DataFileUrl(h5Urls[index], nextFile)));
                nextFile += 1;
              }
            }
            profile.prefetched.push({ index, files, elapsedMs: Math.round(performance.now() - t) });
          };
          let next = 0;
          const worker = async () => {
            while (!disposed) {
              const at = next++;
              if (at >= order.length) return;
              const index = order[at];
              if (prefetchNext && preloadWindow === 1 && at + 1 < order.length) {
                void prefetchVolume(order[at + 1]).catch((error) => {
                  console.warn("Show4DSTEM HDF5 compressed prefetch failed", error);
                });
              }
              const t = performance.now();
              try {
                const cc = await getVol!(index);
                const elapsedMs = Math.round(performance.now() - t);
                if (cc) profile.completed += 1;
                else profile.failed += 1;
                profile.volumes.push({ index, elapsedMs, ok: Boolean(cc) });
                if (cc && !disposed) {
                  latestResidentVolumeIndex = index;
                  requestCompareViLive();
                  requestDpFrameLive();
                }
              } catch (error) {
                profile.failed += 1;
                profile.volumes.push({
                  index,
                  elapsedMs: Math.round(performance.now() - t),
                  ok: false,
                  error: error instanceof Error ? error.message : String(error),
                });
              } finally {
                profile.totalMs = Math.round(performance.now() - startedAt);
              }
            }
          };
          return Promise.all(Array.from({ length: Math.min(preloadWindow, order.length) }, worker)).then(() => undefined);
        };
        h5VolumePreload = startH5Preloads().finally(() => {
          h5VolumePreloadDone = true;
        });
        if (hasInlineViMaps()) {
          initialVolumeLoad = getVol(initialIdx);
        } else {
          compute = await getVol(initialIdx);
        }
        void h5VolumePreload.catch((error) => {
          console.warn("Show4DSTEM HDF5 preload failed", error);
        });
      } else if (h5Url) {
        compute = await loadH5Compute(h5Url, "dataset 1/1");
        if (compute) computes.push(compute);
      } else if (bslz4Meta) {
        // bslz4 mode: ship native HDF5 bitshuffle+LZ4 bytes (~6x smaller than uint16),
        // decompress on the GPU into a uint8 stack. The meta JSON is single
        // (chunked: {base, chunks}), or multi-volume ({volumes:[{base,chunks,badPx}]}).
        const m = JSON.parse(bslz4Meta) as any;
        const srcDtype = (m.srcDtype === "uint8" ? "uint8" : "uint16") as "uint8" | "uint16";  // 8-plane fast path if uint8-encoded
        const fetchU8 = async (u: string) => new Uint8Array(await (await fetch(u)).arrayBuffer());
        const fetchU32 = async (u: string) => new Uint32Array(await (await fetch(u)).arrayBuffer());
        const decodeVol = async (v: any) => {
          const specs: (Bslz4Spec & { startScan: number; nScan: number })[] = [];
          for (const c of v.chunks) specs.push({ compressed: await fetchU8(v.base + c.bin), blockMeta: await fetchU32(v.base + c.meta),
            nFrames: c.nScan, nBlocksPerFrame: c.nBlocksPerFrame, blockElems: c.blockElems,
            detSize: detR * detC, startScan: c.startScan, nScan: c.nScan });
          const cc = await DetectorCompute.createFromBslz4Chunked(specs, scanRows * scanCols, detR * detC, "uint8", srcDtype);
          if (cc && v.badPx) cc.badPx = new Uint32Array(v.badPx);
          return cc;
        };
        if (Array.isArray(m.volumes)) {
          volMetas = m.volumes;
          volumeCount = volMetas.length;
          getVol = async (idx: number) => {
            if (volCache.has(idx)) return volCache.get(idx)!;
            const existingLoad = volLoadPromises.get(idx);
            if (existingLoad) return await existingLoad;
            const loadPromise = (async () => {
              const cc = await decodeVol(volMetas[idx]);
              if (cc) {
                volCache.set(idx, cc);
                latestResidentVolumeIndex = idx;
                const activeFrame = Math.max(0, Math.min(volMetas.length - 1, model.get("frame_idx") | 0));
                while (volCache.size > MAX_RESIDENT) {           // evict the oldest non-active volume
                  const old = [...volCache.keys()].find((k) => k !== idx && k !== activeFrame);
                  if (old === undefined) break;
                  volCache.get(old)!.dispose(); volCache.delete(old);
                }
              }
              return cc;
            })().finally(() => { volLoadPromises.delete(idx); });
            volLoadPromises.set(idx, loadPromise);
            return await loadPromise;
          };
          compute = await getVol(Math.max(0, Math.min(volMetas.length - 1, model.get("frame_idx") | 0)));
        } else if (Array.isArray(m.chunks)) {
          const c = await decodeVol(m); if (c) computes.push(c); compute = c;
        } else {
          const raw = await fetchU8(offlineUrl!);
          compute = await DetectorCompute.createFromBslz4({ compressed: raw, blockMeta: new Uint32Array(m.blockMeta),
            nFrames: m.nFrames, nBlocksPerFrame: m.nBlocksPerFrame, blockElems: m.blockElems, detSize: detR * detC }, "uint8");
          if (compute) computes.push(compute);
        }
      } else if (chunksMeta && offlineUrl) {
        // Chunked companion: one gzip blob with N chunks; stream each into its own
        // GPU buffer (handles stacks far bigger than one buffer / one ArrayBuffer).
        const blob = new Uint8Array(await (await fetch(offlineUrl)).arrayBuffer());
        const meta = JSON.parse(chunksMeta) as { coff: number; clen: number; startScan: number; nScan: number }[];
        const specs = [];
        for (const m of meta) {
          const bytes = await gunzip(blob.subarray(m.coff, m.coff + m.clen));
          specs.push({ bytes, startScan: m.startScan, nScan: m.nScan });
        }
        compute = await DetectorCompute.createChunked(specs, scanRows * scanCols, detR * detC);
      } else {
        // Single stack: companion fetch or inline, then inflate (gzip, lossless).
        let stack: Uint8Array;
        if (offlineUrl) {
          stack = new Uint8Array(await (await fetch(offlineUrl)).arrayBuffer());
        } else {
          const stackView = model.get("_offline_stack") as DataView | undefined;
          if (!stackView || stackView.byteLength === 0) return;
          stack = new Uint8Array(stackView.buffer, stackView.byteOffset, stackView.byteLength);
        }
        if (model.get("_offline_gzip")) stack = await gunzip(stack);
        const widgetFrames = Math.max(1, model.get("n_frames") | 0);
        const scanCount = scanRows * scanCols;
        const detSize = detR * detC;
        const expectedU8 = widgetFrames * scanCount * detSize;
        const expectedU16 = expectedU8 * 2;
        if (widgetFrames > 1 && (stack.byteLength === expectedU8 || stack.byteLength === expectedU16)) {
          const volumeBytes = stack.byteLength / widgetFrames;
          volumeCount = widgetFrames;
          const MAX_INLINE_RESIDENT = Math.max(3, Math.min(widgetFrames, compareResidentTarget));
          getVol = async (idx: number) => {
            if (inlineVolCache.has(idx)) return inlineVolCache.get(idx)!;
            const start = idx * volumeBytes;
            const bytes = stack.subarray(start, start + volumeBytes);
            const cc = await DetectorCompute.create(bytes, scanCount, detSize);
            if (cc) {
              inlineVolCache.set(idx, cc);
              const activeFrame = Math.max(0, Math.min(widgetFrames - 1, model.get("frame_idx") | 0));
              while (inlineVolCache.size > MAX_INLINE_RESIDENT) {
                const old = [...inlineVolCache.keys()].find((k) => k !== idx && k !== activeFrame);
                if (old === undefined) break;
                inlineVolCache.get(old)!.dispose();
                inlineVolCache.delete(old);
              }
            }
            return cc;
          };
          compute = await getVol(Math.max(0, Math.min(widgetFrames - 1, model.get("frame_idx") | 0)));
        } else {
          cpuStack = stack;  // keep for the per-frame probe (single-chunk only)
          compute = await DetectorCompute.create(stack, scanRows * scanCols, detR * detC);
        }
      }
      const publishComputeDpcReady = (backend: DetectorCompute) => {
        setWebgpuDpcReady(Boolean(
          (backend as unknown as { maskedDpcBuffer?: unknown }).maskedDpcBuffer
          || (backend as unknown as { maskedIDpcBuffer?: unknown }).maskedIDpcBuffer
          || (backend as unknown as { maskedDpc?: unknown }).maskedDpc
          || (backend as unknown as { maskedCoM?: unknown }).maskedCoM,
        ));
      };
      if ((!compute && !initialVolumeLoad) || disposed) {
        compute?.dispose();
        if (!disposed) {
          setOfflineBackendError(
            "WebGPU is unavailable. Show4DSTEM scientific compute requires a GPU; no CPU fallback is used."
          );
          setOfflineBackendStatus("");
          setOfflineBackendLoading(false);
        }
        return;
      }
      if (compute) publishComputeDpcReady(compute);
      // Auto-filter hot/dead detector pixels (from the HDF5 pixel_mask) so the
      // offline result matches CUDA's apply_mask path - no manual masking needed.
      const badPxJson = model.get("_offline_bad_px") as string | undefined;
      if (badPxJson && compute && compute.detSize === detR * detC) {
        compute.badPx = new Uint32Array(JSON.parse(badPxJson) as number[]);
      }
      const dpcMask = buildFullDetectorMask(detR, detC);
      const computeDpcImage = async (
        backend: DetectorCompute,
        source: DpcGpuSource,
      ): Promise<Float32Array | null> => {
        if (source === "iDPC") {
          const maybeIDpc = backend as unknown as {
            maskedIDpc?: (
              mask: Uint32Array,
              detCols: number,
              scanRows: number,
              scanCols: number,
              rotationDeg?: number,
              useTranspose?: boolean,
            ) => Promise<Float32Array>;
          };
          if (typeof maybeIDpc.maskedIDpc !== "function") return null;
          return await maybeIDpc.maskedIDpc(dpcMask, detC, scanRows, scanCols, 0, false);
        }
        const component = source === "DPC_row" ? "row" : "col";
        const maybeDpc = backend as unknown as {
          maskedDpc?: (mask: Uint32Array, detCols: number, component: "row" | "col") => Promise<Float32Array>;
          maskedCoM?: (mask: Uint32Array, detCols: number) => Promise<{ comY: Float32Array; comX: Float32Array }>;
        };
        if (typeof maybeDpc.maskedDpc === "function") {
          return await maybeDpc.maskedDpc(dpcMask, detC, component);
        }
        if (typeof maybeDpc.maskedCoM !== "function") return null;
        const { comY, comX } = await maybeDpc.maskedCoM(dpcMask, detC);
        const values = component === "row" ? comY : comX;
        let mean = 0;
        for (let i = 0; i < values.length; i++) mean += values[i];
        mean /= Math.max(1, values.length);
        const out = new Float32Array(values.length);
        for (let i = 0; i < values.length; i++) out[i] = values[i] - mean;
        return out;
      };
      type WarmRoiPresetName = "bf" | "abf" | "adf";
      type WarmRoiGeometry = {
        mode: "circle" | "annular";
        centerRow: number;
        centerCol: number;
        radius: number;
        radiusInner: number;
      };
      type WarmViCacheEntry = {
        source: ViGpuSource;
        label: string;
        data: Float32Array;
        key: string;
        kind: string;
        computedMs: number;
      };
      const viWarmCache = new Map<string, WarmViCacheEntry>();
      let viWarmupStarted = false;
      let viWarmupGeneration = 0;
      let viWarmupStatus: "idle" | "warming" | "ready" | "failed" = "idle";
      let suppressViTraitRecompute = false;
      const roundedCacheValue = (value: unknown): string => {
        const n = Number(value);
        if (!Number.isFinite(n)) return "0";
        return String(Math.round(n * 1000) / 1000);
      };
      const activeVolumeCacheKey = () => `vol:${Math.max(0, Math.round(Number(model.get("frame_idx") || 0)))}`;
      const dpcWarmCacheKey = (source: DpcGpuSource) => [
        activeVolumeCacheKey(),
        "dpc",
        source,
        `scan:${scanRows}x${scanCols}`,
        `det:${detR}x${detC}`,
      ].join("|");
      const normalizedRadiusInner = (geometry: WarmRoiGeometry) =>
        geometry.mode === "annular" ? Math.max(0, geometry.radiusInner) : 0;
      const roiWarmCacheKey = (geometry: WarmRoiGeometry) => [
        activeVolumeCacheKey(),
        "roi",
        geometry.mode,
        `cr:${roundedCacheValue(geometry.centerRow)}`,
        `cc:${roundedCacheValue(geometry.centerCol)}`,
        `r:${roundedCacheValue(geometry.radius)}`,
        `ri:${roundedCacheValue(normalizedRadiusInner(geometry))}`,
        `scan:${scanRows}x${scanCols}`,
        `det:${detR}x${detC}`,
      ].join("|");
      const currentRoiGeometry = (): WarmRoiGeometry => {
        const mode = String(model.get("roi_mode") || "circle") === "annular" ? "annular" : "circle";
        return {
          mode,
          centerRow: Number(model.get("roi_center_row") || model.get("center_row") || detR / 2),
          centerCol: Number(model.get("roi_center_col") || model.get("center_col") || detC / 2),
          radius: Math.max(1, Number(model.get("roi_radius") || model.get("bf_radius") || 1)),
          radiusInner: mode === "annular" ? Math.max(0, Number(model.get("roi_radius_inner") || 0)) : 0,
        };
      };
      const presetRoiGeometry = (name: WarmRoiPresetName): WarmRoiGeometry => {
        const bf = Math.max(1, Number(model.get("bf_radius") || 1));
        const centerRow = Number(model.get("center_row") || model.get("roi_center_row") || detR / 2);
        const centerCol = Number(model.get("center_col") || model.get("roi_center_col") || detC / 2);
        if (name === "abf") {
          return {
            mode: "annular",
            centerRow,
            centerCol,
            radius: bf,
            radiusInner: Math.max(0.5, bf * 0.5),
          };
        }
        if (name === "adf") {
          return {
            mode: "annular",
            centerRow,
            centerCol,
            radius: bf * 2,
            radiusInner: bf,
          };
        }
        return {
          mode: "circle",
          centerRow,
          centerCol,
          radius: bf,
          radiusInner: 0,
        };
      };
      const maskForRoiGeometry = (geometry: WarmRoiGeometry): Uint32Array => buildDetectorMask({
        get: (name: string) => {
          if (name === "roi_center_row") return geometry.centerRow;
          if (name === "roi_center_col") return geometry.centerCol;
          if (name === "roi_mode") return geometry.mode;
          if (name === "roi_radius") return geometry.radius;
          if (name === "roi_radius_inner") return geometry.radiusInner;
          if (name === "roi_width" || name === "roi_height") return 0;
          return model.get(name);
        },
      }, detR, detC);
      const warmCacheSummary = () => ({
        status: viWarmupStatus,
        generation: viWarmupGeneration,
        count: viWarmCache.size,
        keys: Array.from(viWarmCache.keys()),
        entries: Array.from(viWarmCache.values()).map((entry) => ({
          label: entry.label,
          source: entry.source,
          kind: entry.kind,
          pixels: entry.data.length,
          computedMs: entry.computedMs,
        })),
      });
      const publishWarmCacheSummary = (extra: Record<string, unknown> = {}) => {
        try {
          (window as unknown as { __sh4dWarmCache?: unknown }).__sh4dWarmCache = {
            ...warmCacheSummary(),
            ...extra,
          };
        } catch {
          // Diagnostics must not affect interaction.
        }
      };
      const dataViewForFloat32 = (data: Float32Array): DataView => (
        new DataView(data.buffer as ArrayBuffer, data.byteOffset, data.byteLength)
      );
      const setWarmCacheEntry = (entry: WarmViCacheEntry) => {
        viWarmCache.set(entry.key, entry);
        publishWarmCacheSummary({ lastStored: entry.label });
      };
      const serveWarmCacheEntry = (
        key: string,
        source: ViGpuSource,
        startedAt: number,
        generation: number,
      ): boolean => {
        if (dpRoiInteractiveRef.current) return false;
        const cached = viWarmCache.get(key);
        if (!cached || cached.source !== source || cached.data.length !== scanRows * scanCols) {
          return false;
        }
        clearViGpuDisplay();
        publishVirtualImageBytes(dataViewForFloat32(cached.data));
        recordViProfile(source, `${cached.kind}_hit`, startedAt, generation);
        publishWarmCacheSummary({ lastHit: cached.label });
        return true;
      };
      let dpcBufferQueue: Promise<void> = Promise.resolve();
      const computeDpcBufferImage = async (
        backend: DetectorCompute,
        source: DpcGpuSource,
      ): Promise<boolean> => {
        const run = async (): Promise<boolean> => {
          const engine = ensureViGpuColormap(backend);
          if (!engine) {
            return false;
          }
          await engine.getDevice().queue.onSubmittedWorkDone().catch(() => {});
          let result: { buffer: GPUBuffer; n: number; cleanup?: () => void } | null = null;
          if (source === "iDPC") {
            const maybeIDpc = backend as unknown as {
              maskedIDpcBuffer?: (
                mask: Uint32Array,
                detCols: number,
                scanRows: number,
                scanCols: number,
                rotationDeg?: number,
                useTranspose?: boolean,
              ) => Promise<{ buffer: GPUBuffer; n: number; cleanup?: () => void }>;
            };
            if (typeof maybeIDpc.maskedIDpcBuffer !== "function") return false;
            result = await maybeIDpc.maskedIDpcBuffer(dpcMask, detC, scanRows, scanCols, 0, false);
          } else {
            const component = source === "DPC_row" ? "row" : "col";
            const maybeDpc = backend as unknown as {
              maskedDpcBuffer?: (mask: Uint32Array, detCols: number, component: "row" | "col") => { buffer: GPUBuffer; n: number; cleanup?: () => void };
            };
            if (typeof maybeDpc.maskedDpcBuffer !== "function") return false;
            result = maybeDpc.maskedDpcBuffer(dpcMask, detC, component);
          }
          if (!result) return false;
          const { buffer, n, cleanup } = result;
          await engine.getDevice().queue.onSubmittedWorkDone().catch(() => {});
          cleanup?.();
          if (n === 0) {
            buffer.destroy();
            return false;
          }
          engine.adoptBuffer(VI_GPU_SLOT, buffer, scanCols, scanRows);
          viGpuImageRef.current = {
            source,
            slot: VI_GPU_SLOT,
            width: scanCols,
            height: scanRows,
            rangeMode: "gpu",
            rawVersionAfter: rawVirtualImageVersionRef.current + 1,
          };
          setViGpuVersion(v => v + 1);
          publishShow4DSTEMViDisplay({
            source,
            gpuBufferToDisplay: true,
            rendered: false,
            pixels: scanRows * scanCols,
            slot: VI_GPU_SLOT,
            rangeMode: "gpu",
          });
          return true;
        };
        const queued = dpcBufferQueue.then(run, run);
        dpcBufferQueue = queued.then(() => undefined, () => undefined);
        return await queued;
      };
      const computeRoiBufferImage = (
        backend: DetectorCompute,
        mask: Uint32Array,
      ): boolean => {
        const engine = ensureViGpuColormap(backend);
        const maybeVi = backend as unknown as {
          maskedSumBuffer?: (mask: Uint32Array) => { buffer: GPUBuffer; n: number };
        };
        if (!engine || typeof maybeVi.maskedSumBuffer !== "function") {
          return false;
        }
        const t0 = performance.now();
        const { buffer, n } = maybeVi.maskedSumBuffer(mask);
        if (n === 0) {
          buffer.destroy();
          return false;
        }
        engine.adoptBuffer(VI_GPU_SLOT, buffer, scanCols, scanRows);
        viGpuImageRef.current = {
          source: "roi",
          slot: VI_GPU_SLOT,
          width: scanCols,
          height: scanRows,
          rangeMode: "gpu",
          rawVersionAfter: rawVirtualImageVersionRef.current + 1,
        };
        setViGpuVersion(v => v + 1);
        publishShow4DSTEMViDisplay({
          source: "roi",
          gpuBufferToDisplay: true,
          rendered: false,
          pixels: scanRows * scanCols,
          maskPixels: n,
          slot: VI_GPU_SLOT,
          rangeMode: "gpu",
          submitMs: performance.now() - t0,
        });
        return true;
      };
      const readGpuFloatBuffer = async (
        device: GPUDevice,
        buffer: GPUBuffer,
        n: number,
      ): Promise<Float32Array> => {
        const readback = device.createBuffer({
          size: Math.max(4, n * 4),
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, readback, 0, n * 4);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        readback.destroy();
        return out;
      };
      const h5ProductFirstSourceUrl = (): string | null => {
        const candidates = h5Urls.length ? h5Urls : h5Url ? [h5Url] : [];
        if (candidates.length !== 1) return null;
        const sourceUrl = candidates[0];
        return /_master\.h5(?:[?#].*)?$/.test(sourceUrl) ? sourceUrl : null;
      };
      const computeH5ProductFirstRoi = async (
        mask: Uint32Array,
        generation: number,
      ): Promise<{ data: Float32Array; displayed: boolean; profile: unknown } | null> => {
        const sourceUrl = h5ProductFirstSourceUrl();
        if (!sourceUrl || !show4DSTEMHasLocalFiles()) return null;
        const productBatch = show4DSTEMOptionalGlobalInt("__QT_H5_PRODUCT_BATCH", 1, 16);
        const product = await loadShow4DSTEMLocalH5MaskedSum(sourceUrl, {
          scanRows,
          scanCols,
          embeddedBadPixelsJson: model.get("_offline_bad_px") as string | undefined,
          mask,
          productBatch,
        });
        if (!product) return null;
        if (generation !== viRecomputeGen) {
          product.buffer.destroy();
          return null;
        }
        const engine = ensureViGpuColormap({
          getDevice: () => product.device,
        } as unknown as DetectorCompute);
        let displayed = false;
        if (engine) {
          engine.adoptBuffer(VI_GPU_SLOT, product.buffer, product.scanCols, product.scanRows);
          viGpuImageRef.current = {
            source: "roi",
            slot: VI_GPU_SLOT,
            width: product.scanCols,
            height: product.scanRows,
            rangeMode: "gpu",
            rawVersionAfter: rawVirtualImageVersionRef.current + 1,
          };
          setViGpuVersion(v => v + 1);
          publishShow4DSTEMViDisplay({
            source: "roi",
            gpuBufferToDisplay: true,
            rendered: false,
            pixels: product.scanRows * product.scanCols,
            slot: VI_GPU_SLOT,
            rangeMode: "gpu",
            productFirstH5: true,
            profile: product.profile,
          });
          displayed = true;
        }
        const data = await readGpuFloatBuffer(product.device, product.buffer, product.scanRows * product.scanCols);
        if (!displayed) product.buffer.destroy();
        return { data, displayed, profile: product.profile };
      };
      let viRecomputeGen = 0;
      const recordViProfile = (
        source: string,
        kind: string,
        startedAt: number,
        generation: number,
      ) => {
        try {
          const profile = {
            source,
            kind,
            generation,
            ms: Math.round((performance.now() - startedAt) * 10) / 10,
            roiMode: String(model.get("roi_mode") || ""),
            roiRadius: Number(model.get("roi_radius") || 0),
            roiRadiusInner: Number(model.get("roi_radius_inner") || 0),
            timestamp: Math.round(performance.now()),
          };
          const win = window as unknown as {
            __sh4dViProfile?: unknown;
            __sh4dViHistory?: unknown[];
          };
          const history = Array.isArray(win.__sh4dViHistory) ? win.__sh4dViHistory : [];
          history.push(profile);
          if (history.length > 80) history.splice(0, history.length - 80);
          win.__sh4dViProfile = profile;
          win.__sh4dViHistory = history;
        } catch {
          // Diagnostics must not affect interaction.
        }
      };
      const recomputeVI = async () => {
        const generation = ++viRecomputeGen;
        const startedAt = performance.now();
        const source = normaliseViSource(model.get("vi_source"));
        const product = viProductFrameView(model, scanRows, scanCols, source);
        if (product) {
          if (generation !== viRecomputeGen) return;
          clearViGpuDisplay();
          publishVirtualImageBytes(product);
          recordViProfile(source, "product", startedAt, generation);
          return;
        }
        if (source === "roi") {
          const preset = viPresetFrameView(model, scanRows, scanCols);
          if (preset) {
            if (generation !== viRecomputeGen) return;
            clearViGpuDisplay();
            publishVirtualImageBytes(preset);
            recordViProfile(source, "preset_product", startedAt, generation);
            return;
          }
        }
        if (isDpcGpuSource(source)) {
          if (serveWarmCacheEntry(dpcWarmCacheKey(source), source, startedAt, generation)) {
            return;
          }
          if (!compute) return;
          const displayed = await computeDpcBufferImage(compute!, source);
          if (generation !== viRecomputeGen) return;
          if (!displayed) {
            clearViGpuDisplay();
          } else {
            recordViProfile(source, "dpc_gpu_display", startedAt, generation);
            void (async () => {
              const dpc = await computeDpcImage(compute!, source);
              if (generation !== viRecomputeGen || !dpc) return;
              setWarmCacheEntry({
                source,
                label: viSourceLabel(source),
                data: dpc,
                key: dpcWarmCacheKey(source),
                kind: "dpc_gpu_display_warm_cache",
                computedMs: Math.round((performance.now() - startedAt) * 10) / 10,
              });
              publishVirtualImageBytes(new DataView(dpc.buffer));
            })();
            return;
          }
          const dpc = await computeDpcImage(compute!, source);
          if (generation !== viRecomputeGen) return;
          if (dpc) {
            setWarmCacheEntry({
              source,
              label: viSourceLabel(source),
              data: dpc,
              key: dpcWarmCacheKey(source),
              kind: displayed ? "dpc_gpu_display_warm_cache" : "dpc_warm_cache",
              computedMs: Math.round((performance.now() - startedAt) * 10) / 10,
            });
            publishVirtualImageBytes(new DataView(dpc.buffer));
            recordViProfile(source, displayed ? "dpc_gpu_display" : "dpc", startedAt, generation);
            return;
          }
          return;
        }
        const mask = buildDetectorMask(model, detR, detC);
        const roiKey = roiWarmCacheKey(currentRoiGeometry());
        if (serveWarmCacheEntry(roiKey, "roi", startedAt, generation)) {
          return;
        }
        const preferH5ProductFirst = (globalThis as { __QT_H5_PRODUCT_FIRST_VI?: unknown }).__QT_H5_PRODUCT_FIRST_VI === true;
        if (preferH5ProductFirst || !compute) {
          const h5Product = await computeH5ProductFirstRoi(mask, generation);
          if (generation !== viRecomputeGen) return;
          if (h5Product) {
            setWarmCacheEntry({
              source: "roi",
              label: String(model.get("roi_mode") || "ROI"),
              data: h5Product.data,
              key: roiKey,
              kind: h5Product.displayed ? "h5_product_first_gpu_display_warm_cache" : "h5_product_first_warm_cache",
              computedMs: Math.round((performance.now() - startedAt) * 10) / 10,
            });
            publishVirtualImageBytes(new DataView(h5Product.data.buffer));
            recordViProfile(source, h5Product.displayed ? "h5_product_first_gpu_display" : "h5_product_first", startedAt, generation);
            return;
          }
        }
        if (!compute) return;
        const displayed = computeRoiBufferImage(compute!, mask);
        if (!displayed) {
          clearViGpuDisplay();
        }
        if (generation !== viRecomputeGen) return;
        if (displayed && dpRoiInteractiveRef.current) {
          recordViProfile(source, "masked_sum_gpu_display_interactive", startedAt, generation);
          return;
        }
        const vi = await compute!.maskedSum(mask);
        if (generation !== viRecomputeGen) return;
        setWarmCacheEntry({
          source: "roi",
          label: String(model.get("roi_mode") || "ROI"),
          data: vi,
          key: roiKey,
          kind: displayed ? "masked_sum_gpu_display_warm_cache" : "masked_sum_warm_cache",
          computedMs: Math.round((performance.now() - startedAt) * 10) / 10,
        });
        publishVirtualImageBytes(new DataView(vi.buffer));
        recordViProfile(source, displayed ? "masked_sum_gpu_display" : "masked_sum", startedAt, generation);
      };
      const warmStandardViCache = async () => {
        if (viWarmupStarted || disposed || !compute) {
          return warmCacheSummary();
        }
        viWarmupStarted = true;
        viWarmupStatus = "warming";
        const warmGeneration = ++viWarmupGeneration;
        const startedAt = performance.now();
        publishWarmCacheSummary({ status: "warming" });
        try {
          for (const preset of ["bf", "abf", "adf"] as WarmRoiPresetName[]) {
            if (disposed || warmGeneration !== viWarmupGeneration || !compute) {
              return warmCacheSummary();
            }
            const geometry = presetRoiGeometry(preset);
            const key = roiWarmCacheKey(geometry);
            if (viWarmCache.has(key)) continue;
            const t0 = performance.now();
            const data = await compute.maskedSum(maskForRoiGeometry(geometry));
            if (disposed || warmGeneration !== viWarmupGeneration) {
              return warmCacheSummary();
            }
            setWarmCacheEntry({
              source: "roi",
              label: preset.toUpperCase(),
              data,
              key,
              kind: "launch_warm_cache",
              computedMs: Math.round((performance.now() - t0) * 10) / 10,
            });
          }
          for (const source of ["DPC_row", "DPC_col"] as DpcGpuSource[]) {
            if (disposed || warmGeneration !== viWarmupGeneration || !compute) {
              return warmCacheSummary();
            }
            const key = dpcWarmCacheKey(source);
            if (viWarmCache.has(key)) continue;
            const t0 = performance.now();
            const data = await computeDpcImage(compute, source);
            if (!data || disposed || warmGeneration !== viWarmupGeneration) {
              continue;
            }
            setWarmCacheEntry({
              source,
              label: viSourceLabel(source),
              data,
              key,
              kind: "launch_warm_cache",
              computedMs: Math.round((performance.now() - t0) * 10) / 10,
            });
          }
          publishWarmCacheSummary({
            status: "ready",
            warmMs: Math.round((performance.now() - startedAt) * 10) / 10,
          });
          viWarmupStatus = "ready";
        } catch (error) {
          viWarmupStatus = "failed";
          publishWarmCacheSummary({
            status: "failed",
            error: error instanceof Error ? error.message : String(error),
          });
        }
        return warmCacheSummary();
      };
      const resetWarmViCache = (reason: string) => {
        viWarmupGeneration += 1;
        viWarmupStarted = false;
        viWarmupStatus = "idle";
        viWarmCache.clear();
        publishWarmCacheSummary({ status: "idle", reason });
      };
      const scheduleWarmStandardViCache = () => {
        if (viWarmupStarted || disposed) return;
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            if (!disposed) {
              void warmStandardViCache();
            }
          });
        });
      };
      let compareViGen = 0;
      const comparePageState = () => {
        const total = Math.max(0, Number(model.get("n_frames") || 0));
        const mode = String(model.get("view_mode") || "single");
        if (total <= 1 || (mode !== "multiple" && mode !== "compare")) {
          return { visible: [] as number[], page: [] as number[] };
        }
        const maxPanels = Math.max(1, Number(model.get("compare_max_panels") || total));
        const natural = Array.from({ length: total }, (_, idx) => idx);
        const rawOrder = Array.isArray(model.get("compare_panel_order")) ? model.get("compare_panel_order") as number[] : [];
        let ordered = natural;
        if (
          rawOrder.length === total
          && rawOrder.every((idx) => Number.isInteger(idx) && idx >= 0 && idx < total)
          && new Set(rawOrder).size === total
        ) {
          ordered = rawOrder.map((idx) => Number(idx));
        }
        const hidden = new Set(
          (Array.isArray(model.get("compare_hidden_panels")) ? model.get("compare_hidden_panels") as number[] : [])
            .filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < total)
            .map((idx) => Number(idx)),
        );
        const visible = ordered.filter((idx) => !hidden.has(idx));
        const pageCount = Math.max(1, Math.ceil(ordered.length / maxPanels));
        const rawPage = Math.round(Number(model.get("compare_page_idx") || 0));
        const pageIdx = Math.max(0, Math.min(pageCount - 1, rawPage));
        if (Number(model.get("compare_page_count") || 1) !== pageCount) model.set("compare_page_count", pageCount);
        if (rawPage !== pageIdx) model.set("compare_page_idx", pageIdx);
        const start = pageIdx * maxPanels;
        return {
          visible,
          page: ordered.slice(start, start + maxPanels).filter((idx) => !hidden.has(idx)),
        };
      };
      const compareVisibleIndices = () => {
        const state = comparePageState();
        return String(model.get("compare_group_mode") || "paged") === "all" ? state.visible : state.page;
      };
      const compareAverageDpIndices = () => {
        return comparePageState().page;
      };
      // Interactive drags must never PAGE a volume: with more panels than the
      // resident LRU, recomputing every panel per drag step forces a full-volume
      // decode + eviction each time (seconds per step, permanent thrash). During
      // a drag only resident volumes update; skipped panels keep their previous
      // image (persistent stack) and catch up on the mouseup finalize.
      let comparePersistentStack: Float32Array | null = null;
      let compareLastInteractiveMs = 0;
      // Fresh settled bytes supersede the drag-time GPU slots: clear them so the
      // grid falls back to the exact (bad-px-corrected, mask-normalised) images.
      // Coalesce GPU-slot version bumps to one React commit per animation frame:
      // mouse moves arrive faster than paints, and each bump re-renders the grid.
      let compareGpuRafHandle = 0;
      const bumpCompareGpuVersion = () => {
        if (compareGpuRafHandle) return;
        compareGpuRafHandle = requestAnimationFrame(() => {
          compareGpuRafHandle = 0;
          setCompareGpuVersion(v => v + 1);
        });
      };
      const settleCompareGpuSlots = () => {
        compareIncrementalRef.current = null;
        if (compareGpuSlotsRef.current.size) {
          compareGpuSlotsRef.current.clear();
          setCompareGpuVersion(v => v + 1);
        }
      };
      const volIsResident = (idx: number): boolean =>
        !getVol || volCache.has(idx) || inlineVolCache.has(idx);
      const publishDirectCompareStack = (
        bytes: DataView,
        count: number,
        indices: number[],
      ) => {
        progressiveCompareGenerationRef.current = null;
        setProgressiveComparePage(null);
        model.set("compare_virtual_image_bytes", bytes);
        model.set("compare_panel_count", count);
        model.set("compare_panel_indices", indices);
      };
      const recomputeCompareVI = async () => {
        const indices = compareVisibleIndices();
        if (!indices.length) return;
        const source = normaliseViSource(model.get("vi_source"));
        const interactiveDrag = dpRoiInteractiveRef.current;
        // ROI compare panels render through GPU-resident slots both for the
        // initial/settled image and for drag updates. The initial frame uses a
        // full exact masked sum; subsequent geometry changes use exact deltas.
        const updateRoiCompareGpuSlots = async (): Promise<boolean> => {
          if (source !== "roi") return false;
          const engine0 = compute ? ensureViGpuColormap(compute) : null;
          if (!engine0) return false;
          const computeStartedAt = performance.now();
          const mask0 = buildDetectorMask(model, detR, detC);
          let slotCursor = 0;
          const batchComputes: DetectorCompute[] = [];
          const batchSlots: number[] = [];
          const batchFrames: number[] = [];
          for (const idx of indices) {
            const slot = COMPARE_GPU_SLOT_BASE + slotCursor++;
            // Do not turn a progressive refresh into an all-volume decode.
            // A completed panel keeps its GPU display slot even after the
            // native uint16 source volume has been evicted.
            if (getVol && !volIsResident(idx)) continue;
            const panelCompute = getVol ? await getVol(idx) : compute;
            if (!(panelCompute instanceof DetectorCompute)) continue;
            batchComputes.push(panelCompute);
            batchSlots.push(slot);
            batchFrames.push(idx);
            compareGpuSlotsRef.current.set(idx, slot);
          }
          let adopted = 0;
          if (batchComputes.length) {
            const indicesKey = batchFrames.join(",");
            const previous = compareIncrementalRef.current;
            let buffers: GPUBuffer[];
            let path: string;
            let addedPixels = 0;
            let removedPixels = 0;
            if (
              interactiveDrag
              && previous
              && previous.indicesKey === indicesKey
              && previous.mask.length === mask0.length
              && batchFrames.every((frame) => previous.buffers.has(frame))
            ) {
              const addedMask = new Uint32Array(mask0.length);
              const removedMask = new Uint32Array(mask0.length);
              for (let i = 0; i < mask0.length; i++) {
                const next = mask0[i] !== 0;
                const prev = previous.mask[i] !== 0;
                if (next && !prev) { addedMask[i] = 1; addedPixels++; }
                else if (!next && prev) { removedMask[i] = 1; removedPixels++; }
              }
              if (addedPixels === 0 && removedPixels === 0) {
                publishLiveCompareViStats("delta-skip", {
                  ms: performance.now() - computeStartedAt,
                  adoptedPanels: 0,
                  requestedPanels: indices.length,
                });
                return true;
              }
              const prevBuffers = batchFrames.map((frame) => previous.buffers.get(frame)!);
              const delta = DetectorCompute.maskedSumDeltaBuffersBatch(batchComputes, prevBuffers, addedMask, removedMask);
              buffers = delta.buffers;
              path = delta.path;
              addedPixels = delta.addedPixels;
              removedPixels = delta.removedPixels;
            } else {
              const full = DetectorCompute.maskedSumBuffersBatch(batchComputes, mask0);
              buffers = full.buffers;
              path = full.path;
            }
            const nextBuffers = new Map<number, GPUBuffer>();
            for (let i = 0; i < buffers.length; i++) {
              engine0.adoptBuffer(batchSlots[i], buffers[i], scanCols, scanRows);
              nextBuffers.set(batchFrames[i], buffers[i]);
              adopted++;
            }
            compareIncrementalRef.current = {
              mask: new Uint32Array(mask0),
              buffers: nextBuffers,
              indicesKey,
            };
            const paintedNow = interactiveDrag ? (compareGpuRenderNowRef.current?.() ?? 0) : 0;
            if (interactiveDrag) {
              await engine0.getDevice().queue.onSubmittedWorkDone().catch(() => {});
            }
            publishLiveCompareViStats(path, {
              ms: performance.now() - computeStartedAt,
              adoptedPanels: adopted,
              requestedPanels: indices.length,
              addedPixels,
              removedPixels,
              paintedPanels: paintedNow,
            });
          }
          if (adopted) bumpCompareGpuVersion();
          if (batchFrames.length) {
            progressiveCompareGenerationRef.current = null;
            setProgressiveComparePage(null);
            model.set("compare_panel_count", indices.length);
            model.set("compare_panel_indices", indices);
          }
          return adopted > 0 || interactiveDrag;
        };
        if (await updateRoiCompareGpuSlots()) return;
        if (interactiveDrag) {
          // No engine (CPU compute fallback): keep the old throttled bytes path.
          const now = performance.now();
          if (now - compareLastInteractiveMs < 150) return;
          compareLastInteractiveMs = now;
        }
        const productStack = viProductStackForIndices(model, indices, scanRows, scanCols);
        if (productStack) {
          settleCompareGpuSlots();
          publishDirectCompareStack(productStack, indices.length, indices);
          return;
        }
        const presetStack = source === "roi"
          ? viPresetStackForIndices(model, indices, scanRows, scanCols)
          : null;
        if (presetStack) {
          settleCompareGpuSlots();
          publishDirectCompareStack(presetStack, indices.length, indices);
          return;
        }
        const gen = ++compareViGen;
        const panelPixels = scanRows * scanCols;
        const stackLength = indices.length * panelPixels;
        if (!comparePersistentStack || comparePersistentStack.length !== stackLength) {
          comparePersistentStack = new Float32Array(stackLength);
        }
        const stack = comparePersistentStack;
        if (isDpcGpuSource(source)) {
          for (let slot = 0; slot < indices.length; slot++) {
            const idx = indices[slot];
            if (interactiveDrag && !volIsResident(idx)) continue;   // keep previous pixels
            const panelCompute = getVol ? await getVol(idx) : compute;
            if (gen !== compareViGen || !panelCompute) return;
            const dpc = await computeDpcImage(panelCompute, source);
            if (gen !== compareViGen || !dpc) return;
            stack.set(dpc, slot * panelPixels);
          }
          settleCompareGpuSlots();
          // fresh copy: reusing the persistent stack's ArrayBuffer identity makes this
          // model.set a silent no-op (no change event -> stats/export/save-state stale)
          publishDirectCompareStack(new DataView(stack.slice().buffer), indices.length, indices);
          return;
        }
        const mask = buildDetectorMask(model, detR, detC);
        let maskArea = 0;
        for (let i = 0; i < mask.length; i++) maskArea += mask[i] ? 1 : 0;
        maskArea = Math.max(1, maskArea);
        for (let slot = 0; slot < indices.length; slot++) {
          const idx = indices[slot];
          if (interactiveDrag && !volIsResident(idx)) continue;   // keep previous pixels
          const panelCompute = getVol ? await getVol(idx) : compute;
          if (gen !== compareViGen || !panelCompute) return;
          const vi = await panelCompute.maskedSum(mask);
          if (gen !== compareViGen) return;
          for (let p = 0; p < panelPixels; p++) {
            stack[slot * panelPixels + p] = vi[p] / maskArea;
          }
        }
        settleCompareGpuSlots();
          // fresh copy: reusing the persistent stack's ArrayBuffer identity makes this
          // model.set a silent no-op (no change event -> stats/export/save-state stale)
        publishDirectCompareStack(new DataView(stack.slice().buffer), indices.length, indices);
      };
      (window as unknown as { __sh4d: unknown }).__sh4d = { model, recomputeVI, recomputeCompareVI,
        detMask: () => buildDetectorMask(model, detR, detC),
        deriveOnly: async () => { const vi = await compute!.maskedSum(buildDetectorMask(model, detR, detC)); return vi.length; },
        rawChecksums: async (scanIndices: number[] = [0, Math.floor((scanRows * scanCols) / 2), scanRows * scanCols - 1]) => {
          const checksum = (compute as unknown as { checksumFrames?: (indices: number[]) => Promise<unknown> })?.checksumFrames;
          if (typeof checksum !== "function") return null;
          const result = await checksum.call(compute, scanIndices);
          (window as unknown as { __sh4dRawChecksums?: unknown }).__sh4dRawChecksums = result;
          return result;
        },
        warmStandardViCache,
        warmCache: () => warmCacheSummary(),
        prepareDetectorMajor: async (options?: { maxVolumes?: number }) => {
          const indices = compareVisibleIndices();
          const maxVolumes = Math.max(1, Math.min(indices.length, Math.round(Number(options?.maxVolumes ?? indices.length))));
          const loaded = [] as DetectorCompute[];
          for (const idx of indices) {
            if (loaded.length >= maxVolumes) break;
            const panelCompute = getVol ? await getVol(idx) : compute;
            if (panelCompute instanceof DetectorCompute) loaded.push(panelCompute);
          }
          const device = loaded[0]?.getDevice();
          if (!device || !loaded.length) return { available: false, reason: "no loaded WebGPU volumes" };
          const startedAt = performance.now();
          const result = DetectorCompute.prepareU8WordMajorBatch(loaded);
          await device.queue.onSubmittedWorkDone().catch(() => {});
          return {
            ...result,
            loaded: loaded.length,
            elapsedMs: Math.round((performance.now() - startedAt) * 10) / 10,
          };
        },
        compareGpuBench: async (options?: {
          mode?: string;
          centerRow?: number;
          centerCol?: number;
          radius?: number;
          innerRadius?: number;
          iterations?: number;
          render?: boolean;
          logScale?: boolean;
        }) => {
          const indices = compareVisibleIndices();
          const engine = compute ? ensureViGpuColormap(compute) : null;
          const device = engine?.getDevice();
          if (!engine || !device || !indices.length) {
            return { available: false, reason: "compare GPU engine or visible indices unavailable" };
          }
          const mode = String(options?.mode ?? model.get("roi_mode") ?? "circle");
          const centerRow = Number(options?.centerRow ?? model.get("roi_center_row") ?? detR / 2);
          const centerCol = Number(options?.centerCol ?? model.get("roi_center_col") ?? detC / 2);
          const radius = Number(options?.radius ?? model.get("roi_radius") ?? 1);
          const innerRadius = Number(options?.innerRadius ?? model.get("roi_radius_inner") ?? 0);
          const iterations = Math.max(1, Math.min(20, Math.round(Number(options?.iterations ?? 3))));
          const render = options?.render !== false;
          const logScale = options?.logScale ?? String(model.get("vi_scale_mode") || "linear") === "log";
          model.set("vi_source", "roi");
          model.set("roi_active", true);
          model.set("roi_mode", mode);
          model.set("roi_center_row", centerRow);
          model.set("roi_center_col", centerCol);
          model.set("roi_center", [centerRow, centerCol]);
          model.set("roi_radius", radius);
          model.set("roi_radius_inner", innerRadius);
          model.set("compare_max_panels", Math.max(indices.length, Number(model.get("compare_max_panels") || 0)));
          model.set("compare_group_mode", "all");
          model.set("vi_scale_mode", logScale ? "log" : "linear");
          model.save_changes();

          const mask = buildDetectorMask(model, detR, detC);
          let maskPixels = 0;
          for (let i = 0; i < mask.length; i++) if (mask[i]) maskPixels++;
          const prepStartedAt = performance.now();
          const loaded = [] as DetectorCompute[];
          for (const idx of indices) {
            const panelCompute = getVol ? await getVol(idx) : compute;
            if (panelCompute instanceof DetectorCompute) loaded.push(panelCompute);
          }
          await device.queue.onSubmittedWorkDone().catch(() => {});
          const prepMs = performance.now() - prepStartedAt;

          const lut = COLORMAPS[String(model.get("vi_colormap") || "inferno")] || COLORMAPS.inferno;
          engine.uploadLUT(String(model.get("vi_colormap") || "inferno"), lut);
          const benchWindow = window as unknown as {
            __sh4dBenchCanvases?: HTMLCanvasElement[];
            __sh4dBenchContexts?: (GPUCanvasContext | null)[];
          };
          if (render && (!benchWindow.__sh4dBenchCanvases || benchWindow.__sh4dBenchCanvases.length < loaded.length)) {
            benchWindow.__sh4dBenchCanvases = Array.from({ length: loaded.length }, () => {
              const canvas = document.createElement("canvas");
              canvas.style.position = "fixed";
              canvas.style.left = "-10000px";
              canvas.style.top = "0";
              canvas.style.width = "32px";
              canvas.style.height = "32px";
              canvas.width = scanCols;
              canvas.height = scanRows;
              document.body.appendChild(canvas);
              return canvas;
            });
            benchWindow.__sh4dBenchContexts = benchWindow.__sh4dBenchCanvases.map((canvas) =>
              engine.configureCanvas(canvas, scanCols, scanRows),
            );
          }

          const results = [] as Record<string, unknown>[];
          for (let iter = 0; iter < iterations; iter++) {
            const slots = [] as number[];
            const computeSubmitStartedAt = performance.now();
            let adopted = 0;
            const { buffers, path } = DetectorCompute.maskedSumBuffersBatch(loaded, mask);
            for (let i = 0; i < buffers.length; i++) {
              const slot = COMPARE_GPU_SLOT_BASE + i;
              engine.adoptBuffer(slot, buffers[i], scanCols, scanRows);
              slots.push(slot);
              adopted++;
            }
            const computeSubmitMs = performance.now() - computeSubmitStartedAt;
            const computeWaitStartedAt = performance.now();
            await device.queue.onSubmittedWorkDone().catch(() => {});
            const computeGpuDoneMs = performance.now() - computeWaitStartedAt;
            const computeTotalMs = performance.now() - computeSubmitStartedAt;

            let renderSubmitMs = 0;
            let renderGpuDoneMs = 0;
            let rendered = 0;
            if (render && slots.length) {
              const renderSubmitStartedAt = performance.now();
              const contexts = benchWindow.__sh4dBenchContexts || [];
              for (let i = 0; i < slots.length; i++) {
                const ctx = contexts[i];
                if (!ctx) continue;
                const ok = engine.renderSlotDirectWithGpuRangeToCanvas(
                  slots[i],
                  Number(model.get("vi_vmin_pct") ?? 0),
                  Number(model.get("vi_vmax_pct") ?? 100),
                  logScale,
                  ctx,
                  {
                    width: scanCols,
                    height: scanRows,
                    bgRgb: 0,
                    transform: { zoom: 1, panX: 0, panY: 0 },
                    smooth: Boolean(model.get("vi_smooth")),
                  },
                );
                if (ok) rendered++;
              }
              renderSubmitMs = performance.now() - renderSubmitStartedAt;
              const renderWaitStartedAt = performance.now();
              await device.queue.onSubmittedWorkDone().catch(() => {});
              renderGpuDoneMs = performance.now() - renderWaitStartedAt;
            }
            results.push({
              iter,
              adopted,
              rendered,
              path,
              computeSubmitMs: Math.round(computeSubmitMs * 10) / 10,
              computeGpuDoneMs: Math.round(computeGpuDoneMs * 10) / 10,
              computeTotalMs: Math.round(computeTotalMs * 10) / 10,
              renderSubmitMs: Math.round(renderSubmitMs * 10) / 10,
              renderGpuDoneMs: Math.round(renderGpuDoneMs * 10) / 10,
            });
          }
          const payload = {
            available: true,
            indices,
            loaded: loaded.length,
            maskPixels,
            mode,
            centerRow,
            centerCol,
            radius,
            innerRadius,
            logScale,
            render,
            batch: true,
            maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
            prepMs: Math.round(prepMs * 10) / 10,
            results,
            adapterInfo: getGPUInfo(),
            softwareAdapter: isSoftwareGPUAdapter(),
          };
          (window as unknown as { __sh4dCompareGpuBench?: unknown }).__sh4dCompareGpuBench = payload;
          return payload;
        },
        compareIncrementalBench: async (options?: {
          mode?: string;
          centerRow?: number;
          centerCol?: number;
          radius?: number;
          innerRadius?: number;
          steps?: Array<[number, number]>;
          render?: boolean;
          logScale?: boolean;
        }) => {
          const indices = compareVisibleIndices();
          const engine = compute ? ensureViGpuColormap(compute) : null;
          const device = engine?.getDevice();
          if (!engine || !device || !indices.length) {
            return { available: false, reason: "compare GPU engine or visible indices unavailable" };
          }
          const mode = String(options?.mode ?? model.get("roi_mode") ?? "annular");
          const centerRow = Number(options?.centerRow ?? model.get("roi_center_row") ?? detR / 2);
          const centerCol = Number(options?.centerCol ?? model.get("roi_center_col") ?? detC / 2);
          const radius = Number(options?.radius ?? model.get("roi_radius") ?? 1);
          const innerRadius = Number(options?.innerRadius ?? model.get("roi_radius_inner") ?? 0);
          const render = options?.render !== false;
          const logScale = options?.logScale ?? String(model.get("vi_scale_mode") || "linear") === "log";
          const centers = options?.steps?.length
            ? options.steps
            : Array.from({ length: 8 }, (_, i): [number, number] => [centerRow + (i > 3 ? 1 : 0), centerCol + i]);
          const waitForFrame = () => new Promise<void>((resolve) => {
            requestAnimationFrame(() => resolve());
          });

          const previousInteractive = dpRoiInteractiveRef.current;
          compareIncrementalRef.current = null;
          suppressViTraitRecompute = true;
          try {
            model.set("vi_source", "roi");
            model.set("roi_active", true);
            model.set("roi_mode", mode);
            model.set("roi_radius", radius);
            model.set("roi_radius_inner", innerRadius);
            model.set("compare_max_panels", Math.max(indices.length, Number(model.get("compare_max_panels") || 0)));
            model.set("compare_group_mode", "all");
            model.set("vi_scale_mode", logScale ? "log" : "linear");
            model.save_changes();
          } finally {
            suppressViTraitRecompute = false;
          }
          await device.queue.onSubmittedWorkDone().catch(() => {});

          const results = [] as Record<string, unknown>[];
          try {
            dpRoiInteractiveRef.current = true;
            for (let iter = 0; iter < centers.length; iter++) {
              const [row, col] = centers[iter];
              const startedAt = performance.now();
              suppressViTraitRecompute = true;
              try {
                writeRoiCenterModel(row, col);
                await recomputeCompareVI();
              } finally {
                suppressViTraitRecompute = false;
              }
              const computeStats = ((window as unknown as {
                __sh4dLiveCompareStats?: Record<string, unknown>;
              }).__sh4dLiveCompareStats) || {};
              const computeSubmittedMs = performance.now() - startedAt;
              const waitStartedAt = performance.now();
              await device.queue.onSubmittedWorkDone().catch(() => {});
              const computeGpuDoneMs = performance.now() - waitStartedAt;
              let paintStats = null as Record<string, unknown> | null;
              if (render) {
                await waitForFrame();
                await waitForFrame();
                await device.queue.onSubmittedWorkDone().catch(() => {});
                paintStats = ((window as unknown as {
                  __sh4dLiveCompareStats?: Record<string, unknown>;
                }).__sh4dLiveCompareStats) || null;
              }
              const totalMs = performance.now() - startedAt;
              results.push({
                iter,
                row,
                col,
                event: computeStats.event,
                addedPixels: computeStats.addedPixels,
                removedPixels: computeStats.removedPixels,
                computeSubmitMs: Math.round(computeSubmittedMs * 10) / 10,
                computeGpuDoneMs: Math.round(computeGpuDoneMs * 10) / 10,
                totalMs: Math.round(totalMs * 10) / 10,
                fps: totalMs > 0 ? Math.round((1000 / totalMs) * 10) / 10 : null,
                liveComputeFps: computeStats.computeFps,
                livePaintFps: paintStats?.paintFps ?? computeStats.paintFps,
                paintedPanels: paintStats?.lastPaintedPanels ?? computeStats.lastPaintedPanels,
              });
            }
          } finally {
            dpRoiInteractiveRef.current = previousInteractive;
          }
          const deltaResults = results.filter((result) => String(result.event || "").startsWith("delta"));
          const deltaTotalMs = deltaResults.map((result) => Number(result.totalMs)).filter(Number.isFinite);
          const meanDeltaMs = deltaTotalMs.length
            ? deltaTotalMs.reduce((sum, ms) => sum + ms, 0) / deltaTotalMs.length
            : null;
          const payload = {
            available: true,
            indices,
            loaded: indices.length,
            mode,
            radius,
            innerRadius,
            logScale,
            render,
            centers,
            firstFullMs: results.length ? results[0].totalMs : null,
            meanDeltaMs: meanDeltaMs == null ? null : Math.round(meanDeltaMs * 10) / 10,
            meanDeltaFps: meanDeltaMs && meanDeltaMs > 0 ? Math.round((1000 / meanDeltaMs) * 10) / 10 : null,
            results,
            adapterInfo: getGPUInfo(),
            softwareAdapter: isSoftwareGPUAdapter(),
          };
          (window as unknown as { __sh4dCompareIncrementalBench?: unknown }).__sh4dCompareIncrementalBench = payload;
          return payload;
        },
        compareIncrementalKernelBench: async (options?: {
          mode?: string;
          centerRow?: number;
          centerCol?: number;
          radius?: number;
          innerRadius?: number;
          steps?: Array<[number, number]>;
          render?: boolean;
          logScale?: boolean;
        }) => {
          const indices = compareVisibleIndices();
          const engine = compute ? ensureViGpuColormap(compute) : null;
          const device = engine?.getDevice();
          if (!engine || !device || !indices.length) {
            return { available: false, reason: "compare GPU engine or visible indices unavailable" };
          }
          const mode = String(options?.mode ?? model.get("roi_mode") ?? "annular");
          const centerRow = Number(options?.centerRow ?? model.get("roi_center_row") ?? detR / 2);
          const centerCol = Number(options?.centerCol ?? model.get("roi_center_col") ?? detC / 2);
          const radius = Number(options?.radius ?? model.get("roi_radius") ?? 1);
          const innerRadius = Number(options?.innerRadius ?? model.get("roi_radius_inner") ?? 0);
          const render = options?.render !== false;
          const logScale = options?.logScale ?? String(model.get("vi_scale_mode") || "linear") === "log";
          const centers = options?.steps?.length
            ? options.steps
            : Array.from({ length: 8 }, (_, i): [number, number] => [centerRow + (i > 3 ? 1 : 0), centerCol + i]);
          const loaded = [] as DetectorCompute[];
          const loadedIndices = [] as number[];
          for (const idx of indices) {
            const panelCompute = getVol ? await getVol(idx) : compute;
            if (panelCompute instanceof DetectorCompute) {
              loaded.push(panelCompute);
              loadedIndices.push(idx);
            }
          }
          await device.queue.onSubmittedWorkDone().catch(() => {});
          const lut = COLORMAPS[String(model.get("vi_colormap") || "inferno")] || COLORMAPS.inferno;
          engine.uploadLUT(String(model.get("vi_colormap") || "inferno"), lut);
          const benchWindow = window as unknown as {
            __sh4dKernelBenchCanvases?: HTMLCanvasElement[];
            __sh4dKernelBenchContexts?: (GPUCanvasContext | null)[];
          };
          if (render && (!benchWindow.__sh4dKernelBenchCanvases || benchWindow.__sh4dKernelBenchCanvases.length < loaded.length)) {
            benchWindow.__sh4dKernelBenchCanvases = Array.from({ length: loaded.length }, () => {
              const canvas = document.createElement("canvas");
              canvas.style.position = "fixed";
              canvas.style.left = "-10000px";
              canvas.style.top = "40px";
              canvas.style.width = "32px";
              canvas.style.height = "32px";
              canvas.width = scanCols;
              canvas.height = scanRows;
              document.body.appendChild(canvas);
              return canvas;
            });
            benchWindow.__sh4dKernelBenchContexts = benchWindow.__sh4dKernelBenchCanvases.map((canvas) =>
              engine.configureCanvas(canvas, scanCols, scanRows),
            );
          }
          const maskForCenter = (row: number, col: number) => buildDetectorMask({
            get: (name: string) => {
              if (name === "roi_mode") return mode;
              if (name === "roi_center_row") return row;
              if (name === "roi_center_col") return col;
              if (name === "roi_radius") return radius;
              if (name === "roi_radius_inner") return innerRadius;
              if (name === "roi_width" || name === "roi_height") return 0;
              return model.get(name);
            },
          }, detR, detC);
          let previousMask: Uint32Array | null = null;
          let previousBuffers: GPUBuffer[] | null = null;
          const results = [] as Record<string, unknown>[];
          for (let iter = 0; iter < centers.length; iter++) {
            const [row, col] = centers[iter];
            const mask = maskForCenter(row, col);
            let addedPixels = 0;
            let removedPixels = 0;
            let buffers: GPUBuffer[];
            let path = "full";
            const startedAt = performance.now();
            if (previousMask && previousBuffers) {
              const addedMask = new Uint32Array(mask.length);
              const removedMask = new Uint32Array(mask.length);
              for (let i = 0; i < mask.length; i++) {
                const next = mask[i] !== 0;
                const prev = previousMask[i] !== 0;
                if (next && !prev) { addedMask[i] = 1; addedPixels++; }
                else if (!next && prev) { removedMask[i] = 1; removedPixels++; }
              }
              const delta = DetectorCompute.maskedSumDeltaBuffersBatch(loaded, previousBuffers, addedMask, removedMask);
              buffers = delta.buffers;
              path = delta.path;
              addedPixels = delta.addedPixels;
              removedPixels = delta.removedPixels;
            } else {
              const full = DetectorCompute.maskedSumBuffersBatch(loaded, mask);
              buffers = full.buffers;
              path = full.path;
            }
            const submitMs = performance.now() - startedAt;
            const waitStartedAt = performance.now();
            await device.queue.onSubmittedWorkDone().catch(() => {});
            const computeGpuDoneMs = performance.now() - waitStartedAt;
            let renderSubmitMs = 0;
            let renderGpuDoneMs = 0;
            let rendered = 0;
            if (render) {
              const contexts = benchWindow.__sh4dKernelBenchContexts || [];
              const renderStartedAt = performance.now();
              for (let i = 0; i < buffers.length; i++) {
                const slot = COMPARE_GPU_SLOT_BASE + 128 + i;
                engine.adoptBuffer(slot, buffers[i], scanCols, scanRows);
                const ctx = contexts[i];
                if (!ctx) continue;
                const ok = engine.renderSlotDirectWithGpuRangeToCanvas(
                  slot,
                  Number(model.get("vi_vmin_pct") ?? 0),
                  Number(model.get("vi_vmax_pct") ?? 100),
                  logScale,
                  ctx,
                  {
                    width: scanCols,
                    height: scanRows,
                    bgRgb: 0,
                    transform: { zoom: 1, panX: 0, panY: 0 },
                    smooth: Boolean(model.get("vi_smooth")),
                  },
                );
                if (ok) rendered++;
              }
              renderSubmitMs = performance.now() - renderStartedAt;
              const renderWaitStartedAt = performance.now();
              await device.queue.onSubmittedWorkDone().catch(() => {});
              renderGpuDoneMs = performance.now() - renderWaitStartedAt;
            }
            previousMask = new Uint32Array(mask);
            previousBuffers = buffers;
            const totalMs = performance.now() - startedAt;
            results.push({
              iter,
              row,
              col,
              path,
              addedPixels,
              removedPixels,
              submitMs: Math.round(submitMs * 10) / 10,
              computeGpuDoneMs: Math.round(computeGpuDoneMs * 10) / 10,
              renderSubmitMs: Math.round(renderSubmitMs * 10) / 10,
              renderGpuDoneMs: Math.round(renderGpuDoneMs * 10) / 10,
              rendered,
              totalMs: Math.round(totalMs * 10) / 10,
              fps: totalMs > 0 ? Math.round((1000 / totalMs) * 10) / 10 : null,
            });
          }
          const deltaMs = results.slice(1).map((result) => Number(result.totalMs)).filter(Number.isFinite);
          const meanDeltaMs = deltaMs.length ? deltaMs.reduce((sum, ms) => sum + ms, 0) / deltaMs.length : null;
          const payload = {
            available: true,
            indices: loadedIndices,
            loaded: loaded.length,
            mode,
            radius,
            innerRadius,
            logScale,
            render,
            firstFullMs: results.length ? results[0].totalMs : null,
            meanDeltaMs: meanDeltaMs == null ? null : Math.round(meanDeltaMs * 10) / 10,
            meanDeltaFps: meanDeltaMs && meanDeltaMs > 0 ? Math.round((1000 / meanDeltaMs) * 10) / 10 : null,
            results,
            adapterInfo: getGPUInfo(),
            softwareAdapter: isSoftwareGPUAdapter(),
          };
          (window as unknown as { __sh4dCompareIncrementalKernelBench?: unknown }).__sh4dCompareIncrementalKernelBench = payload;
          return payload;
        },
        roiBufferOnly: async () => {
          const mask = buildDetectorMask(model, detR, detC);
          const t0 = performance.now();
          const displayed = computeRoiBufferImage(compute!, mask);
          await ensureViGpuColormap(compute!)?.getDevice().queue.onSubmittedWorkDone().catch(() => {});
          return {
            available: displayed,
            displayed: Boolean(viGpuImageRef.current && viGpuImageRef.current.source === "roi"),
            length: scanRows * scanCols,
            elapsedMs: performance.now() - t0,
            detail: (window as unknown as { __sh4dViDisplay?: Record<string, unknown> }).__sh4dViDisplay || {},
          };
        },
        h5ProductFirstRoi: async () => {
          const mask = buildDetectorMask(model, detR, detC);
          const generation = ++viRecomputeGen;
          const t0 = performance.now();
          const result = await computeH5ProductFirstRoi(mask, generation);
          if (!result) return { available: false, elapsedMs: performance.now() - t0 };
          publishVirtualImageBytes(new DataView(result.data.buffer));
          return {
            available: true,
            displayed: result.displayed,
            length: result.data.length,
            elapsedMs: performance.now() - t0,
            profile: result.profile,
            detail: (window as unknown as { __sh4dViDisplay?: Record<string, unknown> }).__sh4dViDisplay || {},
          };
        },
        dpcOnly: async (source: DpcGpuSource = "DPC_row") => {
          const normalSource = isDpcGpuSource(source) ? source : "DPC_row";
          const dpc = await computeDpcImage(compute!, normalSource);
          let sum = 0;
          if (dpc) for (let i = 0; i < dpc.length; i++) sum += dpc[i];
          return { source: normalSource, length: dpc?.length ?? 0, sum };
        },
        dpcCompareReference: async (source: DpcGpuSource = "DPC_row", referenceUrl: string) => {
          const normalSource = isDpcGpuSource(source) ? source : "DPC_row";
          const dpc = await computeDpcImage(compute!, normalSource);
          if (!dpc) return { source: normalSource, available: false, error: "DPC source unavailable" };
          const response = await fetch(referenceUrl);
          if (!response.ok) return { source: normalSource, available: false, error: `reference fetch failed ${response.status}` };
          const ref = new Float32Array(await response.arrayBuffer());
          if (ref.length !== dpc.length) {
            return { source: normalSource, available: false, error: `reference length ${ref.length} != ${dpc.length}` };
          }
          let maxAbsErr = 0;
          let sumAbsErr = 0;
          let sumSqErr = 0;
          let maxAbsRef = 0;
          for (let i = 0; i < dpc.length; i++) {
            const err = Math.abs(dpc[i] - ref[i]);
            if (err > maxAbsErr) maxAbsErr = err;
            sumAbsErr += err;
            sumSqErr += err * err;
            const refAbs = Math.abs(ref[i]);
            if (refAbs > maxAbsRef) maxAbsRef = refAbs;
          }
          return {
            source: normalSource,
            available: true,
            length: dpc.length,
            maxAbsErr,
            meanAbsErr: sumAbsErr / Math.max(1, dpc.length),
            rmsErr: Math.sqrt(sumSqErr / Math.max(1, dpc.length)),
            maxAbsRef,
          };
        },
        dpcBufferOnly: async (source: DpcGpuSource = "DPC_row") => {
          const normalSource = isDpcGpuSource(source) ? source : "DPC_row";
          const displayed = await computeDpcBufferImage(compute!, normalSource);
          return {
            source: normalSource,
            available: displayed,
            displayed: Boolean(viGpuImageRef.current),
            length: scanRows * scanCols,
            detail: (window as unknown as { __sh4dDpcDisplay?: Record<string, unknown> }).__sh4dDpcDisplay || {},
          };
        },
        dpcDisplayOnly: async (source: DpcGpuSource = "DPC_row") => {
          const normalSource = isDpcGpuSource(source) ? source : "DPC_row";
          const displayed = await computeDpcBufferImage(compute!, normalSource);
          return {
            source: normalSource,
            available: displayed,
            displayed: Boolean(viGpuImageRef.current),
            length: scanRows * scanCols,
            detail: (window as unknown as { __sh4dDpcDisplay?: Record<string, unknown> }).__sh4dDpcDisplay || {},
          };
        },
        comLen: () => { const c = compute as unknown as { com?: Float32Array | null }; return c && c.com ? c.com.length : -1; },
        rd: () => ({ mode: model.get("roi_mode"), r: model.get("roi_radius"), ri: model.get("roi_radius_inner"),
          cr: model.get("roi_center_row"), cc: model.get("roi_center_col"), active: model.get("roi_active") }) };
      requestCompareViLiveRef.current = () => {
        if (compareViLiveInFlightRef.current) {
          compareViLivePendingRef.current = true;
          return;
        }
        flushRoiCenter();
        flushRoiRadius();
        compareViLiveInFlightRef.current = true;
        void (async () => {
          await recomputeVI();
          await recomputeCompareVI();
        })().finally(() => {
          compareViLiveInFlightRef.current = false;
          if (compareViLivePendingRef.current && dpRoiInteractiveRef.current) {
            compareViLivePendingRef.current = false;
            requestCompareViLive();
          } else {
            compareViLivePendingRef.current = false;
          }
        });
      };
      requestViFinalizeRef.current = () => {
        void recomputeVI();
        void recomputeCompareVI();
      };
      const recomputeDP = async () => {
        const mode = model.get("vi_roi_mode");
        if (!mode || mode === "off") { model.set("vi_roi_dp_bytes", new DataView(new ArrayBuffer(0))); return; }
        if (!compute) { model.set("vi_roi_dp_bytes", new DataView(new ArrayBuffer(0))); return; }
        const dp = await compute!.reduceFrames(buildScanMask(model, scanRows, scanCols), model.get("vi_roi_reduce") !== "sum");
        model.set("vi_roi_dp_bytes", new DataView(dp.buffer));
      };
      // Pointing at a scan position normally asks the kernel for that position's raw
      // diffraction pattern (frame_bytes). With no kernel we slice it straight out of
      // the offline stack, so the DP follows the probe offline too.
      const detSize = detR * detC;
      const sample = (gp: number) => compute!.mode === 1 ? cpuStack![gp] : (cpuStack![gp * 2] | (cpuStack![gp * 2 + 1] << 8));
      const recomputeFrame = async () => {
        const pr = Math.max(0, Math.min(scanRows - 1, model.get("pos_row") | 0));
        const pc = Math.max(0, Math.min(scanCols - 1, model.get("pos_col") | 0));
        const scanIdx = pr * scanCols + pc;
        const mode = String(model.get("view_mode") || "single");
        const dpMode = String(model.get("compare_dp_mode") || "average");
        if ((mode === "multiple" || mode === "compare") && dpMode !== "selected" && getVol) {
          const indices = compareAverageDpIndices();
          if (indices.length) {
            const latestLoaded = latestResidentVolumeIndex != null && volIsResident(latestResidentVolumeIndex)
              ? latestResidentVolumeIndex
              : null;
            const averageIndices = h5VolumePreloadDone
              ? indices
              : latestLoaded != null
                ? [latestLoaded]
                : indices.filter((idx) => volIsResident(idx));
            const averaged = new Float32Array(detSize);
            let count = 0;
            for (const idx of averageIndices) {
              const source = await getVol(idx);
              if (!source) continue;
              const frame = await source.frameAt(scanIdx);
              for (let k = 0; k < detSize; k++) averaged[k] += frame[k];
              count += 1;
            }
            if (count > 0) {
              for (let k = 0; k < detSize; k++) averaged[k] /= count;
              model.set("frame_bytes", new DataView(averaged.buffer));
              (window as unknown as { __show4dstemLatestDp?: unknown }).__show4dstemLatestDp = {
                scanIdx,
                source: h5VolumePreloadDone ? "average-loaded" : "latest-resident",
                indices: averageIndices,
                count,
                updatedAt: Math.round(performance.now()),
              };
              model.save_changes();
              return;
            }
          }
        }
        // bslz4 / chunked stacks have no CPU copy -> extract the frame on the GPU.
        if (!compute) return;
        const frame = cpuStack
          ? (() => { const f = new Float32Array(detSize); const base = scanIdx * detSize; for (let k = 0; k < detSize; k++) f[k] = sample(base + k); return f; })()
          : await compute!.frameAt(scanIdx);
        model.set("frame_bytes", new DataView(frame.buffer)); model.save_changes();
      };
      requestDpFrameLiveRef.current = () => {
        void recomputeFrame();
      };
      if (h5VolumePreload) {
        void h5VolumePreload.then(() => {
          if (disposed) return;
          void recomputeCompareVI();
          void recomputeFrame();
        }).catch((error) => {
          console.warn("Show4DSTEM HDF5 volume preload refresh failed", error);
        });
      }
      let splittingRoiCenter = false;
      let splittingViCenter = false;
      const onVI = () => {
        if (splittingRoiCenter || suppressViTraitRecompute) return;
        if (dpRoiInteractiveRef.current) {
          requestCompareViLive();
          return;
        }
        void recomputeVI(); void recomputeCompareVI();
      };
      const onDP = () => {
        if (splittingViCenter) return;
        void recomputeDP();
      };
      const onPos = () => { void recomputeFrame(); };
      const onCompareFrameSource = () => { void recomputeFrame(); };
      const onCompareGridSource = () => { void recomputeCompareVI(); void recomputeFrame(); };
      const activateCurrentVolume = async () => {
        if (!getVol) return true;
        const nVolumes = volumeCount || volMetas.length || 1;
        const v = Math.max(0, Math.min(nVolumes - 1, model.get("frame_idx") | 0));
        const cc = await getVol(v);
        if (!cc) return false;
        compute = cc;
        publishComputeDpcReady(cc);
        return true;
      };
      if (initialVolumeLoad) {
        void initialVolumeLoad.then((cc) => {
          if (!cc || disposed) return;
          compute = cc;
          publishComputeDpcReady(cc);
          void (async () => {
            if (!disposed) setOfflineBackendStatus("Rendering first diffraction pattern and virtual image");
            await recomputeFrame();
            await recomputeVI();
            if (!disposed) {
              setOfflineBackendStatus("");
              setOfflineBackendLoading(false);
            }
            scheduleWarmStandardViCache();
          })();
        }).catch((error) => {
          console.warn("Show4DSTEM background H5 load failed", error);
          if (!disposed) {
            setOfflineBackendError(error instanceof Error ? error.message : String(error));
            setOfflineBackendStatus("");
            setOfflineBackendLoading(false);
          }
        });
      }
      const recomputeActiveView = async () => {
        const ready = await activateCurrentVolume();
        if (!ready || disposed) return;
        void recomputeVI();
        void recomputeCompareVI();
        void recomputeDP();
        void recomputeFrame();
      };
      // 5D multi-volume: the slider picks the active dataset; decode-on-scrub (LRU).
      let frameGen = 0;
      const onFrame = async () => {
        const gen = ++frameGen;                  // ignore a stale decode if the user keeps scrubbing
        const ready = await activateCurrentVolume();
        if (gen !== frameGen || !ready) return;   // a newer scroll superseded this one
        resetWarmViCache("frame");
        void recomputeVI(); void recomputeCompareVI(); void recomputeDP(); void recomputeFrame();
        scheduleWarmStandardViCache();
      };
      if (getVol) model.on("change:frame_idx", onFrame);
      model.on("change:view_mode", recomputeActiveView);
      await recomputeFrame();  // initial DP at mount (so the panel isn't blank)
      // BF/ABF/ADF/HAADF presets normally route through the Python kernel
      // (_preset_request -> apply_preset). With no kernel we translate them into
      // the same detector-ROI geometry here so the buttons work offline too.
      const onPreset = () => {
        const name = String(model.get("_preset_request") || "").toLowerCase();
        if (!name) return;
        const bf = model.get("bf_radius") || 1;
        suppressViTraitRecompute = true;
        try {
          model.set("roi_active", true);
          model.set("vi_source", "roi");
          model.set("roi_center_row", model.get("center_row"));
          model.set("roi_center_col", model.get("center_col"));
          if (name === "bf") { model.set("roi_mode", "circle"); model.set("roi_radius_inner", 0); model.set("roi_radius", Math.max(1, bf)); }
          else if (name === "abf") { model.set("roi_mode", "annular"); model.set("roi_radius_inner", Math.max(0.5, bf * 0.5)); model.set("roi_radius", Math.max(1, bf)); }
          else if (name === "adf") { model.set("roi_mode", "annular"); model.set("roi_radius_inner", bf); model.set("roi_radius", bf * 2); }
          else if (name === "haadf") { model.set("roi_mode", "annular"); model.set("roi_radius_inner", bf * 2); model.set("roi_radius", bf * 4); }
          model.set("_preset_request", "");  // consume so the same preset can fire again
        } finally {
          suppressViTraitRecompute = false;
        }
        void recomputeVI(); void recomputeCompareVI();
      };
      // Dragging the aperture sets the COMPOUND roi_center [row, col]; the kernel
      // normally splits it into roi_center_row/col. With no kernel we split it
      // ourselves so the mask sees the dragged center (else only presets/sliders,
      // which write the scalars directly, would move the detector). Same for the
      // real-space vi_roi_center drag.
      const onRoiCenter = () => {
        const rc = model.get("roi_center");
        if (Array.isArray(rc) && rc.length === 2) {
          splittingRoiCenter = true;
          try {
            model.set("roi_center_row", rc[0]);
            model.set("roi_center_col", rc[1]);
          } finally {
            splittingRoiCenter = false;
          }
        }
        if (suppressViTraitRecompute) return;
        if (dpRoiInteractiveRef.current) {
          requestCompareViLive();
          return;
        }
        void recomputeVI(); void recomputeCompareVI();
      };
      const onViCenter = () => {
        const rc = model.get("vi_roi_center");
        if (Array.isArray(rc) && rc.length === 2) {
          splittingViCenter = true;
          try {
            model.set("vi_roi_center_row", rc[0]);
            model.set("vi_roi_center_col", rc[1]);
          } finally {
            splittingViCenter = false;
          }
        }
        void recomputeDP();
      };
      const viTraits = ["roi_center_row", "roi_center_col", "roi_radius", "roi_radius_inner", "roi_mode", "roi_width", "roi_height"];
      const dpTraits = ["vi_roi_center_row", "vi_roi_center_col", "vi_roi_radius", "vi_roi_mode", "vi_roi_width", "vi_roi_height", "vi_roi_reduce"];
      viTraits.forEach((t) => model.on("change:" + t, onVI));
      dpTraits.forEach((t) => model.on("change:" + t, onDP));
      model.on("change:vi_source", onVI);
      model.on("change:roi_center", onRoiCenter);
      model.on("change:vi_roi_center", onViCenter);
      model.on("change:_preset_request", onPreset);
      model.on("change:pos_row", onPos);
      model.on("change:pos_col", onPos);
      model.on("change:compare_dp_mode", onCompareFrameSource);
      model.on("change:compare_max_panels", onCompareGridSource);
      model.on("change:compare_group_mode", onCompareGridSource);
      model.on("change:compare_page_idx", onCompareGridSource);
      model.on("change:compare_panel_order", onCompareGridSource);
      model.on("change:compare_hidden_panels", onCompareGridSource);
      detach = () => {
        viTraits.forEach((t) => model.off("change:" + t, onVI));
        dpTraits.forEach((t) => model.off("change:" + t, onDP));
        model.off("change:vi_source", onVI);
        model.off("change:roi_center", onRoiCenter);
        model.off("change:vi_roi_center", onViCenter);
        model.off("change:_preset_request", onPreset);
        model.off("change:pos_row", onPos);
        model.off("change:pos_col", onPos);
        model.off("change:compare_dp_mode", onCompareFrameSource);
        model.off("change:compare_max_panels", onCompareGridSource);
        model.off("change:compare_group_mode", onCompareGridSource);
        model.off("change:compare_page_idx", onCompareGridSource);
        model.off("change:compare_panel_order", onCompareGridSource);
        model.off("change:compare_hidden_panels", onCompareGridSource);
        model.off("change:frame_idx", onFrame);
        model.off("change:view_mode", recomputeActiveView);
        computes.forEach((c) => c.dispose());          // single / non-lazy resident set
        volCache.forEach((c) => c.dispose()); volCache.clear();  // every cached lazy volume
        inlineVolCache.forEach((c) => c.dispose()); inlineVolCache.clear();
      };
      await recomputeVI();  // initial virtual image, no interaction needed
      await recomputeCompareVI();
      if (!initialVolumeLoad && !disposed) {
        setOfflineBackendStatus("");
        setOfflineBackendLoading(false);
      }
      // Fit the BF disk from the mean diffraction pattern before the presets warm.
      // On the H5/WebGPU path Python never holds the pixels, so bf_radius keeps the
      // det_size/8 guess and every BF/ABF/ADF preset samples the wrong disk: on real
      // Arina data the true radius was 54 px against a 24 px guess, so "ADF" at twice
      // bf_radius still sat inside the bright field. The pixels only exist in the
      // browser, so the fit has to happen here.
      const fitBfDiskFromMeanDp = async (): Promise<void> => {
        if (!compute || disposed) return;
        // Only override the ratio guess; an explicit user bf_radius must win.
        const current = Number(model.get("bf_radius") || 0);
        const ratioGuess = Math.min(detR, detC) * 0.125;
        if (Math.abs(current - ratioGuess) > 0.51) return;
        // A few thousand scan positions fix the disk edge; reducing all of them would
        // read the whole stack and delay first paint for no extra accuracy.
        const scanCount = scanRows * scanCols;
        const stride = Math.max(1, Math.floor(scanCount / 16384));
        const scanMask = new Uint32Array(scanCount);
        for (let i = 0; i < scanCount; i += stride) scanMask[i] = 1;
        const dp = await compute.reduceFrames(scanMask, true);
        if (disposed || !dp || dp.length !== detR * detC) return;
        let peak = -Infinity;
        for (let i = 0; i < dp.length; i++) if (dp[i] > peak) peak = dp[i];
        const median = Float32Array.from(dp).sort()[dp.length >> 1];
        const threshold = 0.5 * (peak + median);
        // Intensity-weighted centroid of the disk interior gives a sub-pixel centre;
        // the beam is not exactly on the detector centre (measured 94.4, 96.6).
        let weight = 0, rowSum = 0, colSum = 0;
        for (let row = 0; row < detR; row++) {
          for (let col = 0; col < detC; col++) {
            const value = dp[row * detC + col];
            if (value >= threshold) { weight += value; rowSum += row * value; colSum += col * value; }
          }
        }
        if (!(weight > 0)) return;
        const centerRow = rowSum / weight, centerCol = colSum / weight;
        // Radial profile, then the half-max crossing: the disk is flat inside and
        // falls off a cliff at the edge, so half-max is stable against hot pixels.
        const maxRadius = Math.ceil(Math.hypot(
          Math.max(centerRow, detR - centerRow), Math.max(centerCol, detC - centerCol)));
        const radialSum = new Float64Array(maxRadius + 1);
        const radialCount = new Float64Array(maxRadius + 1);
        for (let row = 0; row < detR; row++) {
          for (let col = 0; col < detC; col++) {
            const radius = Math.round(Math.hypot(row - centerRow, col - centerCol));
            if (radius <= maxRadius) { radialSum[radius] += dp[row * detC + col]; radialCount[radius] += 1; }
          }
        }
        const profile = new Float64Array(maxRadius + 1);
        for (let i = 0; i <= maxRadius; i++) profile[i] = radialCount[i] ? radialSum[i] / radialCount[i] : 0;
        // Only radii that actually contain detector pixels carry a profile value. A
        // sub-pixel centre usually leaves the radius-0 bin empty, and an empty bin
        // reads as zero, which would otherwise look like the disk edge at r=0.
        const filled: number[] = [];
        for (let i = 0; i <= maxRadius; i++) if (radialCount[i] > 0) filled.push(i);
        if (filled.length < 8) return;
        const plateau = (profile[filled[0]] + profile[filled[1]] + profile[filled[2]]) / 3;
        let background = 0, backgroundCount = 0;
        for (const i of filled.slice(-10)) { background += profile[i]; backgroundCount++; }
        background = backgroundCount ? background / backgroundCount : 0;
        const halfMax = 0.5 * (plateau + background);
        let edge = 0;
        for (const i of filled) { if (i >= 2 && profile[i] < halfMax) { edge = i; break; } }
        if (edge <= 1 || edge >= maxRadius) return;
        const previousBf = current;
        model.set("center_row", centerRow);
        model.set("center_col", centerCol);
        model.set("bf_radius", edge);
        // roi_radius mirrors bf_radius at construction; keep it on the fitted disk
        // unless the user has already moved it.
        if (Math.abs(Number(model.get("roi_radius") || 0) - previousBf) < 0.51) {
          model.set("roi_radius", edge);
          model.set("roi_center_row", centerRow);
          model.set("roi_center_col", centerCol);
        }
      };
      await fitBfDiskFromMeanDp().catch((error) => {
        console.warn("Show4DSTEM BF disk fit failed; keeping the default bf_radius", error);
      });
      scheduleWarmStandardViCache();
      // Safety re-run: at first mount the offline stack / roi-detector traits can
      // still be settling, so the very first maskedSum can return an empty (zero)
      // virtual image - leaving the panel blank until the user nudges the detector.
      // A deferred recompute guarantees the BF image appears with no interaction.
      requestAnimationFrame(() => { if (!disposed) { void recomputeVI(); void recomputeCompareVI(); } });
      setTimeout(() => { if (!disposed) { void recomputeVI(); void recomputeCompareVI(); scheduleWarmStandardViCache(); } }, 200);
    })().catch((error) => {
      console.error("Show4DSTEM offline WebGPU initialization failed", error);
      if (!disposed) {
        setOfflineBackendError(error instanceof Error ? error.message : String(error));
        setOfflineBackendStatus("");
        setOfflineBackendLoading(false);
      }
    });
    return () => {
      disposed = true;
      requestViFinalizeRef.current = null;
      requestCompareViLiveRef.current = null;
      requestDpFrameLiveRef.current = null;
      setWebgpuDpcReady(false);
      setOfflineBackendLoading(false);
      setOfflineBackendStatus("");
      setOfflineBackendError("");
      clearViGpuDisplay();
      detach?.();
    };
  }, [clearViGpuDisplay, ensureViGpuColormap, h5LocalFilesGranted, h5SourceAvailable, offline, requestCompareViLive, requestDpFrameLive, requireLocalH5Files]);
  // dp_stats are computed in JS from frameBytes (Python side no longer
  // syncs a dp_stats trait — saves 4 trait sync round-trips per click).
  const [viStats, setViStats] = React.useState<number[]>([0, 0, 0, 0]);
  const [viDataMin, setViDataMin] = React.useState<number>(0);
  const [viDataMax, setViDataMax] = React.useState<number>(1);
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const [showControls] = useModelState<boolean>("show_controls");
  const [controlsCollapsed] = useModelState<boolean>("controls_collapsed");
  const [debug] = useModelState<boolean>("debug");
  const controlsVisible = showControls && !controlsCollapsed;
  const panelChromeVisible = controlsVisible;
  const debugFps = useDebugFps(Boolean(debug));
  const [showStats] = useModelState<boolean>("show_stats");
  const [showScaleBar] = useModelState<boolean>("show_scale_bar");
  const [mobileDpOptionsOpen, setMobileDpOptionsOpen] = React.useState(false);
  const [mobileViOptionsOpen, setMobileViOptionsOpen] = React.useState(false);
  const [mobileFftOptionsOpen, setMobileFftOptionsOpen] = React.useState(false);
  const [compareReorderMode, setCompareReorderMode] = React.useState(false);
  const [compareDraggingFrame, setCompareDraggingFrame] = React.useState<number | null>(null);
  const [comparePendingMoveFrame, setComparePendingMoveFrame] = React.useState<number | null>(null);
  const [panelWidthPx, setPanelWidthPx] = useModelState<number>("panel_width_px");
  const [compareGridWidthPx, setCompareGridWidthPx] = useModelState<number>("compare_grid_width_px");
  const [compareGridPreviewWidth, setCompareGridPreviewWidth] = React.useState<number | null>(null);
  const compareGridResizeCleanupRef = React.useRef<(() => void) | null>(null);
  const [compareHiddenMenuAnchor, setCompareHiddenMenuAnchor] = React.useState<HTMLElement | null>(null);

  const effectiveShowFft = showFft;
  const displayViewMode = viewMode === "compare" ? "multiple" : viewMode === "temporal" ? "single" : (viewMode || "single");
  const compareMode = (displayViewMode === "multiple" || viewMode === "compare") && nFrames > 1;
  const viProductSourceOptions = React.useMemo(() => {
    const seen = new Set<string>();
    const out: string[] = [];
    const add = (source: string) => {
      if (!["DPC_row", "DPC_col", "iDPC", "SSB"].includes(source) || seen.has(source)) return;
      seen.add(source);
      out.push(source);
    };
    (Array.isArray(viProductLabels) ? viProductLabels : []).forEach((label) => {
      add(normaliseViSource(label));
    });
    if (webgpuDpcReady) {
      add("DPC_row");
      add("DPC_col");
      add("iDPC");
    }
    return out;
  }, [viProductLabels, webgpuDpcReady]);
  const activeViSource = React.useMemo(() => {
    const source = normaliseViSource(viSource);
    return source === "roi" || viProductSourceOptions.includes(source) ? source : "roi";
  }, [viProductSourceOptions, viSource]);
  const hasViProductSources = viProductSourceOptions.length > 0;
  const roiVirtualDetectorActive = activeViSource === "roi";
  const saveChangesIfLiveComm = React.useCallback(() => {
    const liveModel = model as unknown as { save_changes?: () => void };
    if (typeof liveModel.save_changes !== "function") return;
    requestAnimationFrame(() => {
      window.setTimeout(() => {
        try {
          liveModel.save_changes?.();
        } catch (error) {
          console.warn("Show4DSTEM could not sync virtual detector state", error);
        }
      }, 0);
    });
  }, [model]);
  const publishVirtualImageBytes = React.useCallback((bytes: DataView) => {
    setFrontendVirtualImageBytes(bytes);
    setVirtualImageBytes(bytes);
  }, [setVirtualImageBytes]);
  React.useEffect(() => {
    if (virtualImageBytes) setFrontendVirtualImageBytes(virtualImageBytes);
  }, [virtualImageBytes]);
  const requestViPreset = React.useCallback((preset: "bf" | "abf" | "adf") => {
    model.set("_preset_request", preset);
    saveChangesIfLiveComm();
  }, [model, saveChangesIfLiveComm]);
  const setViSource = React.useCallback((nextSource: string) => {
    const source = normaliseViSource(nextSource);
    setViSourceModel(source);
    model.set("vi_source", source);
    saveChangesIfLiveComm();
  }, [model, saveChangesIfLiveComm, setViSourceModel]);
  const displayedVirtualImageBytes = React.useMemo(() => {
    const roiBytes = frontendVirtualImageBytes ?? virtualImageBytes;
    if (activeViSource === "roi") return roiBytes;
    return viProductFrameView(model, shapeRows, shapeCols, activeViSource) ?? roiBytes;
  }, [
    activeViSource,
    frontendVirtualImageBytes,
    frameIdx,
    model,
    shapeCols,
    shapeRows,
    viProductLabels,
    viProductMapFrames,
    viProductMapsBytes,
    virtualImageBytes,
  ]);
  const compareAllGroups = String(compareGroupMode || "paged") === "all";
  const activeComparePageCount = Math.max(1, Math.round(Number(comparePageCount || 1)));
  const activeComparePageIdx = Math.max(0, Math.min(activeComparePageCount - 1, Math.round(Number(comparePageIdx || 0))));
  const comparePageStatus = compareAllGroups ? "All groups" : `${activeComparePageIdx + 1}/${activeComparePageCount}`;
  const displayedCompareVirtualImageBytes = React.useMemo(() => {
    if (activeViSource === "roi") return compareVirtualImageBytes;
    const indices = Array.isArray(comparePanelIndices) ? comparePanelIndices : [];
    return viProductStackForIndices(model, indices, shapeRows, shapeCols) ?? compareVirtualImageBytes;
  }, [
    activeViSource,
    comparePanelIndices,
    compareVirtualImageBytes,
    model,
    shapeCols,
    shapeRows,
    viProductLabels,
    viProductMapFrames,
    viProductMapsBytes,
  ]);
  const comparePageButtonItems = React.useMemo<(number | "gap")[]>(() => {
    if (activeComparePageCount <= 8) {
      return Array.from({ length: activeComparePageCount }, (_, idx) => idx);
    }
    const pages = Array.from(new Set([
      0,
      activeComparePageIdx - 1,
      activeComparePageIdx,
      activeComparePageIdx + 1,
      activeComparePageCount - 1,
    ].filter((idx) => idx >= 0 && idx < activeComparePageCount))).sort((a, b) => a - b);
    const items: (number | "gap")[] = [];
    pages.forEach((page, idx) => {
      if (idx > 0 && page - pages[idx - 1] > 1) items.push("gap");
      items.push(page);
    });
    return items;
  }, [activeComparePageCount, activeComparePageIdx]);
  const frameSliderLabel = compareMode ? "Panel" : frameDimLabel;
  const frameSliderAriaLabel = compareMode ? "Show4DSTEM active multiple panel" : `Show4DSTEM ${frameDimLabel.toLowerCase()}`;
  const [comparePagePlaying, setComparePagePlaying] = React.useState(false);
  const compareGridWidth = compareGridPreviewWidth ?? (compareGridWidthPx > 0 ? compareGridWidthPx : COMPARE_GRID_DEFAULT_WIDTH);
  React.useEffect(() => {
    if (!compareMode || compareAllGroups) {
      setCompareReorderMode(false);
      setCompareDraggingFrame(null);
      setComparePendingMoveFrame(null);
      setCompareGridPreviewWidth(null);
      setComparePagePlaying(false);
      compareGridResizeCleanupRef.current?.();
    }
  }, [compareAllGroups, compareMode]);
  React.useEffect(() => {
    if (activeComparePageCount <= 1 || compareAllGroups) setComparePagePlaying(false);
  }, [activeComparePageCount, compareAllGroups]);
  const compareHiddenCount = React.useMemo(() => {
    const seen = new Set<number>();
    (compareHiddenPanels || []).forEach((idx) => {
      if (Number.isInteger(idx) && idx >= 0 && idx < nFrames) seen.add(idx);
    });
    return seen.size;
  }, [compareHiddenPanels, nFrames]);
  const normalizedCompareOrder = React.useCallback(() => {
    const natural = Array.from({ length: Math.max(0, nFrames) }, (_, idx) => idx);
    const order = Array.isArray(comparePanelOrder) ? comparePanelOrder : [];
    if (order.length !== nFrames) return natural;
    const seen = new Set<number>();
    for (const idx of order) {
      if (!Number.isInteger(idx) || idx < 0 || idx >= nFrames || seen.has(idx)) return natural;
      seen.add(idx);
    }
    return [...order];
  }, [comparePanelOrder, nFrames]);
  const visibleCompareHistogramFrames = React.useMemo(() => {
    const source = progressiveComparePage?.expectedIndices?.length
      ? progressiveComparePage.expectedIndices
      : Array.isArray(comparePanelIndices)
        ? comparePanelIndices
        : [];
    if (!source.length) return [] as number[];
    const available = new Set(source);
    const hidden = new Set(
      (compareHiddenPanels || []).filter((idx) => Number.isInteger(idx) && available.has(idx)),
    );
    const ordered: number[] = [];
    const seen = new Set<number>();
    normalizedCompareOrder().forEach((idx) => {
      if (available.has(idx) && !hidden.has(idx) && !seen.has(idx)) {
        ordered.push(idx);
        seen.add(idx);
      }
    });
    source.forEach((idx) => {
      if (!hidden.has(idx) && !seen.has(idx)) ordered.push(idx);
    });
    return ordered;
  }, [compareHiddenPanels, comparePanelIndices, normalizedCompareOrder, progressiveComparePage]);
  const requestComparePage = React.useCallback((page: number) => {
    const next = Math.max(0, Math.min(activeComparePageCount - 1, Math.round(Number(page) || 0)));
    if (next === activeComparePageIdx) return;
    const hidden = new Set(
      (compareHiddenPanels || []).filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < nFrames),
    );
    const ordered = normalizedCompareOrder();
    const pageSize = Math.max(1, Math.round(Number(compareMaxPanels || ordered.length || 1)));
    const pendingGeneration = comparePageProgressiveEnabled
      ? `pending:${++progressiveComparePendingGenerationRef.current}`
      : "";
    const pending = beginPendingProgressiveComparePage(
      Boolean(comparePageProgressiveEnabled),
      pendingGeneration,
      next,
      ordered,
      [...hidden],
      pageSize,
    );
    progressiveCompareGenerationRef.current = pending?.generation ?? null;
    setProgressiveComparePage(pending);
    if (pending) recordComparePageClick(next);
    setComparePageIdx(next);
  }, [
    activeComparePageCount,
    activeComparePageIdx,
    compareHiddenPanels,
    compareMaxPanels,
    comparePageProgressiveEnabled,
    nFrames,
    normalizedCompareOrder,
    setComparePageIdx,
  ]);
  const comparePanelLabel = React.useCallback((idx: number) => {
    return frameLabels && frameLabels.length > idx && frameLabels[idx]
      ? frameLabels[idx]
      : `${frameDimLabel} ${idx + 1}`;
  }, [frameDimLabel, frameLabels]);
  const compareHiddenPanelItems = React.useMemo(() => {
    const hidden = new Set<number>(
      (compareHiddenPanels || []).filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < nFrames),
    );
    return normalizedCompareOrder()
      .filter((idx) => hidden.has(idx))
      .map((idx) => ({ idx, label: comparePanelLabel(idx) }));
  }, [compareHiddenPanels, comparePanelLabel, nFrames, normalizedCompareOrder]);
  const moveCompareFrame = React.useCallback((dragFrame: number, targetFrame: number) => {
    if (!Number.isInteger(dragFrame) || !Number.isInteger(targetFrame) || dragFrame === targetFrame) return;
    const order = normalizedCompareOrder();
    if (!order.includes(dragFrame) || !order.includes(targetFrame)) return;
    const next = order.filter((idx) => idx !== dragFrame);
    const targetPos = next.indexOf(targetFrame);
    next.splice(targetPos < 0 ? next.length : targetPos, 0, dragFrame);
    setComparePanelOrder(next);
    setFramePlaying(false);
  }, [normalizedCompareOrder, setComparePanelOrder, setFramePlaying]);
  const toggleCompareStar = React.useCallback((frame: number) => {
    if (!Number.isInteger(frame) || frame < 0 || frame >= nFrames) return;
    const next = new Set<number>((compareStarredPanels || []).filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < nFrames));
    if (next.has(frame)) next.delete(frame);
    else next.add(frame);
    setCompareStarredPanels([...next].sort((a, b) => a - b));
  }, [compareStarredPanels, nFrames, setCompareStarredPanels]);
  const showCompareFrame = React.useCallback((frame: number) => {
    if (!Number.isInteger(frame) || frame < 0 || frame >= nFrames) return;
    const next = (compareHiddenPanels || [])
      .filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < nFrames && idx !== frame);
    setCompareHiddenPanels([...new Set(next)].sort((a, b) => a - b));
    setCompareHiddenMenuAnchor(null);
  }, [compareHiddenPanels, nFrames, setCompareHiddenPanels]);
  const hideCompareFrame = React.useCallback((frame: number) => {
    if (!Number.isInteger(frame) || frame < 0 || frame >= nFrames) return;
    const next = new Set<number>((compareHiddenPanels || []).filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < nFrames));
    if (next.size >= Math.max(0, nFrames - 1) && !next.has(frame)) return;
    next.add(frame);
    if (next.size < nFrames) setCompareHiddenPanels([...next].sort((a, b) => a - b));
    if (comparePendingMoveFrame === frame) setComparePendingMoveFrame(null);
  }, [compareHiddenPanels, comparePendingMoveFrame, nFrames, setCompareHiddenPanels]);
  const resetComparePanelState = React.useCallback(() => {
    setComparePanelOrder([]);
    setCompareHiddenPanels([]);
    setCompareStarredPanels([]);
    setCompareGroupMode("paged");
    setComparePageIdx(0);
    setComparePagePlaying(false);
    setComparePendingMoveFrame(null);
    setCompareDraggingFrame(null);
    setCompareHiddenMenuAnchor(null);
  }, [setCompareGroupMode, setCompareHiddenPanels, setComparePageIdx, setComparePanelOrder, setCompareStarredPanels]);

  // ROI FFT state (VI ROI crops virtual image for FFT)
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);
  const roiFftActive = effectiveShowFft && viRoiMode !== "off";

  // Canvas resize state
  const initialCanvasSize = panelWidthPx > 0 ? panelWidthPx : CANVAS_SIZE;
  const [canvasSize, setCanvasSize] = React.useState(initialCanvasSize);
  React.useEffect(() => {
    if (panelWidthPx > 0) setCanvasSize(panelWidthPx);
  }, [panelWidthPx]);
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeCanvasStart, setResizeCanvasStart] = React.useState<{ x: number; y: number; size: number } | null>(null);

  // Export
  const [dpExportAnchor, setDpExportAnchor] = React.useState<HTMLElement | null>(null);
  const [dpMoreAnchor, setDpMoreAnchor] = React.useState<HTMLElement | null>(null);
  const [ssbCalOpen, setSsbCalOpen] = React.useState(false);
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportPayloadFilename] = useModelState<string>("export_filename");
  const [htmlExportBusy, setHtmlExportBusy] = React.useState(false);
  const [localHtmlExportStatus, setLocalHtmlExportStatus] = React.useState("");
  const pendingHtmlExportRef = React.useRef<{
    id: string;
    filename: string;
    mode: string;
    handle: Show4DSTEMFileHandle | null;
  } | null>(null);
  React.useEffect(() => {
    if (!exportStatus) return;
    const preparing = exportStatus.startsWith("Preparing ") || exportStatus.startsWith("Exporting ");
    if (preparing) {
      setHtmlExportBusy(true);
    } else if (!pendingHtmlExportRef.current) {
      setHtmlExportBusy(false);
    }
  }, [exportStatus]);
  const requestSsbCompute = React.useCallback((options?: {
    manualAberrations?: boolean;
    closeMenu?: boolean;
    c10Nm?: number;
    c12Nm?: number;
    phi12Deg?: number;
    rotationDeg?: number;
  }) => {
    const nTrials = Math.max(0, Math.round(Number(ssbComputeNTrials ?? 200)));
    const bfSubsample = Math.max(0.01, Math.min(1, Number(ssbComputeBfSubsample ?? 1)));
    const manualAberrations = Boolean(options?.manualAberrations);
    const c10Nm = Number(options?.c10Nm ?? ssbComputeC10Nm ?? 0);
    const c12Nm = Number(options?.c12Nm ?? ssbComputeC12Nm ?? 0);
    const phi12Deg = Number(options?.phi12Deg ?? ssbComputePhi12Deg ?? 0);
    const rotationDeg = Number(options?.rotationDeg ?? ssbComputeRotationDeg ?? 0);
    const payload = JSON.stringify({
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      action: "compute_ssb",
      n_trials: nTrials,
      refine: Boolean(ssbComputeRefine),
      bf_subsample: bfSubsample,
      lock_c10: Boolean(ssbComputeLockC10),
      lock_c12: Boolean(ssbComputeLockC12),
      manual_aberrations: manualAberrations,
      lock_aberrations: manualAberrations,
      ...(manualAberrations ? {
        c10_nm: c10Nm,
        c12_nm: c12Nm,
        phi12_deg: phi12Deg,
        rotation_angle_deg: rotationDeg,
      } : {}),
    });
    if (options?.closeMenu !== false) setDpMoreAnchor(null);
    setSsbComputeRequest(payload);
    model.set("ssb_compute_request", payload);
    model.save_changes();
  }, [
    model,
    setSsbComputeRequest,
    ssbComputeBfSubsample,
    ssbComputeC10Nm,
    ssbComputeC12Nm,
    ssbComputeLockC10,
    ssbComputeLockC12,
    ssbComputeNTrials,
    ssbComputePhi12Deg,
    ssbComputeRefine,
    ssbComputeRotationDeg,
  ]);
  const requestSsbManualReconstruct = React.useCallback((values?: {
    c10Nm?: number;
    c12Nm?: number;
    phi12Deg?: number;
    rotationDeg?: number;
  }) => {
    requestSsbCompute({
      manualAberrations: true,
      closeMenu: false,
      ...values,
    });
  }, [requestSsbCompute]);
  const downloadSsbCalibration = React.useCallback(() => {
    const text = String(ssbComputeCalibrationJson || "").trim();
    if (!text) return;
    const filename = String(ssbComputeCalibrationFilename || "").trim() || "show4dstem_ssb_calibration.json";
    downloadBlob(new Blob([text], { type: "application/json;charset=utf-8" }), filename);
    setDpMoreAnchor(null);
  }, [ssbComputeCalibrationFilename, ssbComputeCalibrationJson]);
  const reportDatasetCount = React.useCallback((datasetScope: HtmlDatasetScope) => {
    const hidden = new Set((compareHiddenPanels || []).filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < nFrames));
    const unhidden = Math.max(1, nFrames - hidden.size);
    if (datasetScope === "all") return Math.max(1, nFrames);
    if (datasetScope === "starred") {
      return (compareStarredPanels || []).filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < nFrames && !hidden.has(idx)).length;
    }
    if (datasetScope === "current_page") {
      const pageSize = Math.max(1, Math.round(Number(compareMaxPanels || comparePanelCount || unhidden || 1)));
      const start = Math.max(0, Math.round(Number(comparePageIdx || 0))) * pageSize;
      return normalizedCompareOrder()
        .slice(start, start + pageSize)
        .filter((idx) => !hidden.has(idx)).length;
    }
    return unhidden;
  }, [compareHiddenPanels, compareMaxPanels, comparePageIdx, comparePanelCount, compareStarredPanels, nFrames, normalizedCompareOrder]);

  const estimateHtmlExportSize = React.useCallback((
    exportKind: HtmlExportKind,
    dtype: string,
    detBin: number,
    scanBin: number,
    datasetScope: HtmlDatasetScope = "unhidden",
  ) => {
    const binnedScanRows = Math.max(1, Math.floor(shapeRows / scanBin));
    const binnedScanCols = Math.max(1, Math.floor(shapeCols / scanBin));
    if (exportKind === "report") {
      const datasetCount = reportDatasetCount(datasetScope);
      const presetCount = 4;
      const rgbBytes = datasetCount * presetCount * binnedScanRows * binnedScanCols * 3;
      return formatEstimatedHtmlSize(rgbBytes);
    }
    const binnedRows = Math.max(1, Math.floor(detRows / detBin));
    const binnedCols = Math.max(1, Math.floor(detCols / detBin));
    const bytesPerPixel = dtype === "uint16" ? 2 : 1;
    const payloadBytes = Math.max(0, nFrames) * binnedScanRows * binnedScanCols * binnedRows * binnedCols * bytesPerPixel;
    return formatEstimatedHtmlSize(payloadBytes);
  }, [detCols, detRows, nFrames, reportDatasetCount, shapeCols, shapeRows]);

  const handleHtmlExportSelect = async (
    exportKind: HtmlExportKind,
    dtype: string,
    detBin: number,
    scanBin: number,
    datasetScope: HtmlDatasetScope = "unhidden",
  ) => {
    setDpExportAnchor(null);
    if (!["uint8", "uint16"].includes(dtype) || ![1, 2, 4, 8].includes(detBin) || ![1, 2, 4, 8].includes(scanBin)) return;
    if (detRows % detBin !== 0 || detCols % detBin !== 0 || shapeRows % scanBin !== 0 || shapeCols % scanBin !== 0) return;
    const mode = `${dtype}-bin${detBin}`;
    const filename = makeHtmlExportFilename(
      title,
      nFrames,
      shapeRows,
      shapeCols,
      detRows,
      detCols,
      dtype,
      detBin,
      scanBin,
      exportKind,
      datasetScope,
    );
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setHtmlExportBusy(true);
    setLocalHtmlExportStatus("Choose export location...");
    const picker = (window as Show4DSTEMWindow).showSaveFilePicker;
    let handle: Show4DSTEMFileHandle | null = null;
    if (picker) {
      try {
        handle = await picker({
          suggestedName: filename,
          types: [{ description: "Standalone HTML", accept: { "text/html": [".html"] } }],
        });
      } catch (err) {
        if (isAbortLikeError(err)) {
          setHtmlExportBusy(false);
          setLocalHtmlExportStatus("Export canceled");
          return;
        }
        setHtmlExportBusy(false);
        setLocalHtmlExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
    }
    pendingHtmlExportRef.current = { id, filename, mode, handle };
    setLocalHtmlExportStatus(`Preparing ${filename}...`);
    setExportRequest(JSON.stringify({
      export_kind: exportKind,
      mode,
      dtype,
      det_bin: detBin,
      scan_bin: scanBin,
      dataset_scope: datasetScope,
      id,
      filename,
      download: true,
    }));
  };

  const reportScanBin = React.useMemo(() => (
    [4, 2, 1].find((bin) => shapeRows % bin === 0 && shapeCols % bin === 0) || 1
  ), [shapeCols, shapeRows]);
  const detailedReportScanBin = React.useMemo(() => (
    [2, 1].find((bin) => shapeRows % bin === 0 && shapeCols % bin === 0) || 1
  ), [shapeCols, shapeRows]);
  const reportDetBin = React.useMemo(() => (
    [8, 4, 2, 1].find((bin) => detRows % bin === 0 && detCols % bin === 0) || 1
  ), [detCols, detRows]);
  const interactiveHtmlPresets = React.useMemo<HtmlInteractivePreset[]>(() => {
    const desired: Array<Omit<HtmlInteractivePreset, "estimatedBytes">> = [
      { label: "Tiny preview", dtype: "uint8", scanBin: 8, detBin: 8 },
      { label: "Small preview", dtype: "uint8", scanBin: 4, detBin: 8 },
      { label: "Compact", dtype: "uint8", scanBin: 4, detBin: 4 },
      { label: "Balanced", dtype: "uint8", scanBin: 2, detBin: 4 },
      { label: "Detector detail", dtype: "uint8", scanBin: 1, detBin: 8 },
      { label: "Detailed", dtype: "uint8", scanBin: 2, detBin: 2 },
      { label: "Fine detector", dtype: "uint8", scanBin: 1, detBin: 2 },
      { label: "Full uint8", dtype: "uint8", scanBin: 1, detBin: 1 },
      { label: "Exact raw", dtype: "uint16", scanBin: 1, detBin: 1 },
    ];
    const out: HtmlInteractivePreset[] = [];
    const seen = new Set<string>();
    const add = (preset: Omit<HtmlInteractivePreset, "estimatedBytes">) => {
      if (![1, 2, 4, 8].includes(preset.scanBin) || ![1, 2, 4, 8].includes(preset.detBin)) return;
      if (shapeRows % preset.scanBin !== 0 || shapeCols % preset.scanBin !== 0) return;
      if (detRows % preset.detBin !== 0 || detCols % preset.detBin !== 0) return;
      const key = `${preset.dtype}:${preset.scanBin}:${preset.detBin}`;
      if (seen.has(key)) return;
      seen.add(key);
      out.push({
        ...preset,
        estimatedBytes: estimateInteractiveHtmlBytes(
          nFrames,
          shapeRows,
          shapeCols,
          detRows,
          detCols,
          preset.dtype,
          preset.detBin,
          preset.scanBin,
        ),
      });
    };
    desired.forEach(add);
    if (out.length < 9) {
      const fallback: Array<Omit<HtmlInteractivePreset, "estimatedBytes">> = [];
      (["uint8", "uint16"] as HtmlExportDtype[]).forEach((dtype) => {
        [8, 4, 2, 1].forEach((scanBin) => {
          [8, 4, 2, 1].forEach((detBin) => {
            fallback.push({
              label: dtype === "uint16" ? "16-bit option" : "8-bit option",
              dtype,
              scanBin,
              detBin,
            });
          });
        });
      });
      fallback
        .map((preset) => ({
          ...preset,
          estimatedBytes: estimateInteractiveHtmlBytes(
            nFrames,
            shapeRows,
            shapeCols,
            detRows,
            detCols,
            preset.dtype,
            preset.detBin,
            preset.scanBin,
          ),
        }))
        .sort((a, b) => a.estimatedBytes - b.estimatedBytes)
        .forEach((preset) => {
          if (out.length < 9) add(preset);
        });
    }
    return out.sort((a, b) => a.estimatedBytes - b.estimatedBytes);
  }, [detCols, detRows, nFrames, shapeCols, shapeRows]);
  const starredReportCount = reportDatasetCount("starred");
  const currentPageReportCount = reportDatasetCount("current_page");

  React.useEffect(() => {
    const pending = pendingHtmlExportRef.current;
    if (!pending || exportPayloadId !== pending.id) return;
    const bytes = extractBytes(exportPayload);
    if (bytes.length === 0) return;
    let canceled = false;
    const save = async () => {
      const payload = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? bytes
        : bytes.slice();
      const filename = exportPayloadFilename || pending.filename;
      const blob = new Blob([payload as BlobPart], { type: "text/html;charset=utf-8" });
      try {
        if (pending.handle) {
          setLocalHtmlExportStatus(`Saving ${filename}...`);
          const writable = await pending.handle.createWritable();
          await writable.write(blob);
          await writable.close();
        } else {
          downloadBlob(blob, filename);
        }
        if (canceled) return;
        pendingHtmlExportRef.current = null;
        setHtmlExportBusy(false);
        setLocalHtmlExportStatus(`Saved ${filename} (${formatSavedBytes(bytes.byteLength)})`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      } catch (err) {
        if (canceled) return;
        pendingHtmlExportRef.current = null;
        setHtmlExportBusy(false);
        setLocalHtmlExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      }
    };
    void save();
    return () => { canceled = true; };
  }, [exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest]);

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number; panel: string } | null>(null);

  // DP Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const profileResizeStart = React.useRef<{ startY: number; startHeight: number } | null>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);
  const profilePoints = profileLine || [];
  const rawDpDataRef = React.useRef<Float32Array | null>(null);
  const dpClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [draggingDpProfileEndpoint, setDraggingDpProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingDpProfileLine, setIsDraggingDpProfileLine] = React.useState(false);
  const [hoveredDpProfileEndpoint, setHoveredDpProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringDpProfileLine, setIsHoveringDpProfileLine] = React.useState(false);
  const dpProfileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);
  const dpDragOffsetRef = React.useRef<{ dRow: number; dCol: number }>({ dRow: 0, dCol: 0 });

  // VI Line profile state
  const [viProfileActive, setViProfileActive] = React.useState(false);
  const [viProfileData, setViProfileData] = React.useState<Float32Array | null>(null);
  const [viProfilePoints, setViProfilePoints] = React.useState<Array<{ row: number; col: number }>>([]);
  const [viProfileHeight, setViProfileHeight] = React.useState(76);
  const [isResizingViProfile, setIsResizingViProfile] = React.useState(false);
  const viProfileResizeStart = React.useRef<{ startY: number; startHeight: number } | null>(null);
  const viProfileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const viProfileBaseImageRef = React.useRef<ImageData | null>(null);
  const viProfileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);
  const rawViDataRef = React.useRef<Float32Array | null>(null);
  const viClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [draggingViProfileEndpoint, setDraggingViProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingViProfileLine, setIsDraggingViProfileLine] = React.useState(false);
  const [hoveredViProfileEndpoint, setHoveredViProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringViProfileLine, setIsHoveringViProfileLine] = React.useState(false);
  const viProfileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);
  const viRoiDragOffsetRef = React.useRef<{ dRow: number; dCol: number }>({ dRow: 0, dCol: 0 });

  // Theme detection
  const { themeInfo, colors: themeColors } = useTheme();
  const roiColors = themeInfo.theme === "dark" ? DARK_ROI_COLORS : LIGHT_ROI_COLORS;
  const accentGreen = themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a";

  // Themed typography — applies theme colors to module-level font sizes
  const typo = React.useMemo(() => ({
    label: { ...typography.label, color: themeColors.textMuted },
    labelSmall: { ...typography.labelSmall, color: themeColors.textMuted },
    value: { ...typography.value, color: themeColors.textMuted },
    title: { ...typography.title, color: themeColors.accent },
  }), [themeColors]);

  // Compute VI canvas dimensions to respect aspect ratio of rectangular scans
  const viCanvasWidth = shapeRows > shapeCols ? Math.round(canvasSize * (shapeCols / shapeRows)) : canvasSize;
  const viCanvasHeight = shapeCols > shapeRows ? Math.round(canvasSize * (shapeRows / shapeCols)) : canvasSize;

  // Histogram data - use state to ensure re-renders (both are Float32Array now)
  const [dpHistogramData, setDpHistogramData] = React.useState<Float32Array | null>(null);
  const [viHistogramData, setViHistogramData] = React.useState<Float32Array | null>(null);
  const [viHistogramBins, setViHistogramBins] = React.useState<Float32Array | null>(null);

  // DP stats computed JS-side from frame_bytes (was Python trait pre-refactor;
  // moving to JS skips 4 sync trait round-trips per scan-position click).
  const [dpStats, setDpStats] = React.useState<number[]>([0, 0, 0, 0]);

  const usesViRoiDp = viRoiMode && viRoiMode !== "off" && viRoiDpBytes && viRoiDpBytes.byteLength > 0;
  const displayedDpBytes = usesViRoiDp ? viRoiDpBytes : frameBytes;

  // Parse displayed DP bytes for stats/histogram. When a VI ROI is active, the
  // DP panel shows the ROI-reduced DP, so its stats must use the same bytes.
  React.useEffect(() => {
    if (!displayedDpBytes) return;
    // Parse as Float32Array since Python now sends raw float32
    const rawData = new Float32Array(displayedDpBytes.buffer, displayedDpBytes.byteOffset, displayedDpBytes.byteLength / 4);
    // Store raw data for profile sampling
    if (!rawDpDataRef.current || rawDpDataRef.current.length !== rawData.length) {
      rawDpDataRef.current = new Float32Array(rawData.length);
    }
    rawDpDataRef.current.set(rawData);
    // Compute stats JS-side (replaces removed Python dp_stats trait)
    const s = computeStats(rawData);
    setDpStats([s.mean, s.min, s.max, s.std]);
    // Apply scale transformation for histogram display
    const scaledData = new Float32Array(rawData.length);
    if (dpScaleMode === "log") {
      for (let i = 0; i < rawData.length; i++) {
        scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
      }
    } else {
      scaledData.set(rawData);
    }
    setDpHistogramData(scaledData);
  }, [displayedDpBytes, dpScaleMode]);

  // GPU FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);

  // Path animation timer
  React.useEffect(() => {
    if (!pathPlaying || pathLength === 0) return;

    const timer = setInterval(() => {
      setPathIndex((prev: number) => {
        const next = prev + 1;
        if (next >= pathLength) {
          if (pathLoop) {
            return 0;  // Loop back to start
          } else {
            setPathPlaying(false);  // Stop at end
            return prev;
          }
        }
        return next;
      });
    }, pathIntervalMs);

    return () => clearInterval(timer);
  }, [pathPlaying, pathLength, pathIntervalMs, pathLoop, setPathIndex, setPathPlaying]);

  // Frame animation timer (5D time/tilt series)
  const frameBounceDir = React.useRef(1);
  React.useEffect(() => {
    frameBounceDir.current = frameReverse ? -1 : 1;
  }, [frameReverse]);

  React.useEffect(() => {
    if (!framePlaying || nFrames <= 1) return;

    const intervalMs = 1000 / Math.max(0.1, frameFps);
    const timer = setInterval(() => {
      setFrameIdx((prev: number) => {
        let next: number;
        if (frameBoomerang) {
          next = prev + frameBounceDir.current;
          if (next >= nFrames) { frameBounceDir.current = -1; next = nFrames - 2; }
          if (next < 0) { frameBounceDir.current = 1; next = 1; }
          next = Math.max(0, Math.min(nFrames - 1, next));
        } else {
          next = prev + (frameReverse ? -1 : 1);
          if (next >= nFrames) {
            if (frameLoop) return 0;
            setFramePlaying(false);
            return prev;
          }
          if (next < 0) {
            if (frameLoop) return nFrames - 1;
            setFramePlaying(false);
            return prev;
          }
        }
        return next;
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [framePlaying, nFrames, frameFps, frameLoop, frameReverse, frameBoomerang, setFrameIdx, setFramePlaying]);

  React.useEffect(() => {
    if (!comparePagePlaying || activeComparePageCount <= 1) return;
    const timer = setInterval(() => {
      setComparePageIdx((prev: number) => {
        const current = Math.max(0, Math.min(activeComparePageCount - 1, Math.round(Number(prev) || 0)));
        const next = current + 1;
        if (next >= activeComparePageCount) {
          setComparePagePlaying(false);
          return current;
        }
        return next;
      });
    }, 700);

    return () => clearInterval(timer);
  }, [activeComparePageCount, comparePagePlaying, setComparePageIdx]);

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
      }
    });
  }, []);

  // Root element ref (theme-aware styling handled via CSS variables)
  const rootRef = React.useRef<HTMLDivElement>(null);
  useHideStaticFallback(model, rootRef);

  // Zoom state
  const [dpZoom, setDpZoom] = React.useState(1);
  const [dpPanX, setDpPanX] = React.useState(0);
  const [dpPanY, setDpPanY] = React.useState(0);
  const [viZoom, setViZoom] = React.useState(1);
  const [viPanX, setViPanX] = React.useState(0);
  const [viPanY, setViPanY] = React.useState(0);
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  // Live view refs for rAF-coalesced wheel zoom. A Mac trackpad fires MANY wheel
  // events per frame; without coalescing each one triggers a full re-render of
  // this large component and zoom feels laggy. The handler accumulates against
  // the ref (synchronous, accurate) and flushes to React state once per frame.
  const dpViewRef = React.useRef({ zoom: 1, panX: 0, panY: 0, raf: 0 });
  const viViewRef = React.useRef({ zoom: 1, panX: 0, panY: 0, raf: 0 });
  const fftViewRef = React.useRef({ zoom: 1, panX: 0, panY: 0, raf: 0 });
  React.useEffect(() => { const r = dpViewRef.current; r.zoom = dpZoom; r.panX = dpPanX; r.panY = dpPanY; }, [dpZoom, dpPanX, dpPanY]);
  React.useEffect(() => { const r = viViewRef.current; r.zoom = viZoom; r.panX = viPanX; r.panY = viPanY; }, [viZoom, viPanX, viPanY]);
  React.useEffect(() => { const r = fftViewRef.current; r.zoom = fftZoom; r.panX = fftPanX; r.panY = fftPanY; }, [fftZoom, fftPanX, fftPanY]);
  const [fftScaleMode, setFftScaleMode] = useModelState<"linear" | "log">("fft_scale_mode");
  const [fftColormap, setFftColormap] = useModelState<string>("fft_colormap");
  const [fftAuto, setFftAuto] = useModelState<boolean>("fft_auto");
  const [fftVminPct, setFftVminPct] = useModelState<number>("fft_vmin_pct");
  const [fftVmaxPct, setFftVmaxPct] = useModelState<number>("fft_vmax_pct");
  // Remember the manual histogram thumbs from BEFORE Auto was switched on, so
  // switching Auto back off restores the user's previous range instead of
  // leaving whatever the auto pass (or a mid-auto thumb drag) left behind.
  const fftPreAutoPctRef = React.useRef<[number, number] | null>(null);
  const toggleFftAuto = React.useCallback((on: boolean) => {
    if (on) {
      fftPreAutoPctRef.current = [fftVminPct, fftVmaxPct];
    } else if (fftPreAutoPctRef.current) {
      const [vmn, vmx] = fftPreAutoPctRef.current;
      setFftVminPct(vmn); setFftVmaxPct(vmx);
      fftPreAutoPctRef.current = null;
    }
    setFftAuto(on);
  }, [fftVminPct, fftVmaxPct, setFftAuto, setFftVminPct, setFftVmaxPct]);
  const [fftStats, setFftStats] = React.useState<number[] | null>(null);  // [mean, min, max, std]
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataMin, setFftDataMin] = React.useState(0);
  const [fftDataMax, setFftDataMax] = React.useState(1);
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);

  const isTypingTarget = React.useCallback((target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) return false;
    if (target.isContentEditable) return true;
    return target.closest("input, textarea, select, [role='textbox'], [contenteditable='true']") !== null;
  }, []);

  const handleRootMouseDownCapture = React.useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement | null;
    if (target?.closest("canvas")) rootRef.current?.focus();
  }, []);

  const handleKeyDown = React.useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    if (isTypingTarget(e.target)) return;

    const step = e.shiftKey ? 10 : 1;
    let handled = false;

    switch (e.key) {
        case "ArrowUp":
          setPosRow(Math.max(0, posRow - step));
          handled = true;
          break;
        case "ArrowDown":
          setPosRow(Math.min(shapeRows - 1, posRow + step));
          handled = true;
          break;
        case "ArrowLeft":
          setPosCol(Math.max(0, posCol - step));
          handled = true;
          break;
        case "ArrowRight":
          setPosCol(Math.min(shapeCols - 1, posCol + step));
          handled = true;
          break;
        case " ": // Space bar
          if (pathLength > 0) {
            setPathPlaying(!pathPlaying);
            handled = true;
          }
          break;
        case "r":
        case "R":
          setDpZoom(1); setDpPanX(0); setDpPanY(0);
          setViZoom(1); setViPanX(0); setViPanY(0);
          setFftZoom(1); setFftPanX(0); setFftPanY(0);
          handled = true;
          break;
        case "[":
          if (nFrames > 1) {
            setFrameIdx(Math.max(0, frameIdx - 1));
            handled = true;
          }
          break;
        case "]":
          if (nFrames > 1) {
            setFrameIdx(Math.min(nFrames - 1, frameIdx + 1));
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
  }, [
    frameIdx, isTypingTarget, nFrames, pathLength,
    pathPlaying, posCol, posRow, setFrameIdx, setPathPlaying, setPosCol, setPosRow, shapeCols, shapeRows,
  ]);

  // Sync local state
  React.useEffect(() => {
    if (!isDraggingDP && !isDraggingResize) { setLocalKCol(roiCenterCol); setLocalKRow(roiCenterRow); }
  }, [roiCenterCol, roiCenterRow, isDraggingDP, isDraggingResize]);

  React.useEffect(() => {
    scanPositionCurrentRef.current = [Math.round(posRow), Math.round(posCol)];
    if (isDraggingVI) return;
    const optimistic = scanPositionOptimisticRef.current;
    if (optimistic) {
      if (Math.round(posRow) !== optimistic[0] || Math.round(posCol) !== optimistic[1]) return;
      scanPositionOptimisticRef.current = null;
    }
    setLocalPosRow(posRow);
    setLocalPosCol(posCol);
  }, [posRow, posCol, isDraggingVI]);

  const updateScanPosition = React.useCallback((row: number, col: number, commit = false) => {
    setLocalPosRow(row);
    setLocalPosCol(col);
    queueScanPosition(row, col);
    if (commit) flushScanPosition();
  }, [flushScanPosition, queueScanPosition]);

  // Sync VI ROI local state
  React.useEffect(() => {
    if (!isDraggingViRoi && !isDraggingViRoiResize) {
      setLocalViRoiCenterRow(viRoiCenterRow || shapeRows / 2);
      setLocalViRoiCenterCol(viRoiCenterCol || shapeCols / 2);
    }
  }, [viRoiCenterRow, viRoiCenterCol, isDraggingViRoi, isDraggingViRoiResize, shapeRows, shapeCols]);

  // Canvas refs
  const dpCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const dpOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const dpUiRef = React.useRef<HTMLCanvasElement>(null);  // High-DPI UI overlay for scale bar
  const dpOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const dpImageDataRef = React.useRef<ImageData | null>(null);
  const virtualGpuSnapshotSerialRef = React.useRef(0);
  const virtualCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const virtualOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const viUiRef = React.useRef<HTMLCanvasElement>(null);  // High-DPI UI overlay for scale bar
  const viOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const viImageDataRef = React.useRef<ImageData | null>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const fftImageDataRef = React.useRef<ImageData | null>(null);

  type TouchPanelKind = "dp" | "vi" | "fft";
  type TouchTransformState = {
    kind: TouchPanelKind;
    mode: "pan" | "pinch";
    startX: number;
    startY: number;
    startDistance: number;
    startMidX: number;
    startMidY: number;
    startZoom: number;
    startPanX: number;
    startPanY: number;
  };
  const touchTransformRef = React.useRef<TouchTransformState | null>(null);
  const lastTapRef = React.useRef<{ kind: TouchPanelKind; time: number } | null>(null);

  // Offscreen version counters — bump when colormap/data changes, cheap draw effects depend on these
  const [dpOffscreenVersion, setDpOffscreenVersion] = React.useState(0);
  const [viOffscreenVersion, setViOffscreenVersion] = React.useState(0);
  const [fftOffscreenVersion, setFftOffscreenVersion] = React.useState(0);

  // Cached colorbar vmin/vmax — computed in expensive DP effect, reused in UI overlay without recomputing
  const dpColorbarVminRef = React.useRef(0);
  const dpColorbarVmaxRef = React.useRef(1);

  // Device pixel ratio for high-DPI UI overlays
  const DPR = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  // ─────────────────────────────────────────────────────────────────────────
  // Effects: Canvas Rendering & Animation
  // ─────────────────────────────────────────────────────────────────────────

  // Prevent page scroll when scrolling on canvases
  // Re-run when showFft changes since FFT canvas is conditionally rendered
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const overlays = [dpOverlayRef.current, virtualOverlayRef.current, fftOverlayRef.current];
    overlays.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => overlays.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, [effectiveShowFft]);

  // Store raw data for filtering/FFT
  const rawVirtualImageRef = React.useRef<Float32Array | null>(null);
  const fftMagnitudeRef = React.useRef<Float32Array | null>(null);
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);

  // Parse displayed image bytes into Float32Array and apply scale for histogram.
  // For DPC/SSB product maps this is a view into vi_product_maps_bytes, so source
  // switching reuses the static map payload instead of asking Python to resend it.
  React.useEffect(() => {
    if (!displayedVirtualImageBytes) return;
    // Parse as Float32Array
    const numFloats = displayedVirtualImageBytes.byteLength / 4;
    const rawData = new Float32Array(displayedVirtualImageBytes.buffer, displayedVirtualImageBytes.byteOffset, numFloats);

    // Store a copy for filtering/FFT (rawData is a view, we need a copy)
    let storedData = rawVirtualImageRef.current;
    if (!storedData || storedData.length !== numFloats) {
      storedData = new Float32Array(numFloats);
      rawVirtualImageRef.current = storedData;
    }
    storedData.set(rawData);
    rawVirtualImageVersionRef.current += 1;

    // Also store for VI profile sampling
    if (!rawViDataRef.current || rawViDataRef.current.length !== numFloats) {
      rawViDataRef.current = new Float32Array(numFloats);
    }
    rawViDataRef.current.set(rawData);

    // Compute stats + min/max JS-side (replaces removed Python vi_stats / vi_data_min / vi_data_max traits).
    // Python sending bytes + 4 separate stat traits caused a comm-message ordering race on rapid
    // preset clicks: bytes from click N could arrive with min/max from click N-1, normalizing
    // the colormap to the wrong range and producing a uniform-color VI flash.
    if (!compareMode) {
      const s = computeStats(rawData);
      setViStats([s.mean, s.min, s.max, s.std]);
      if (viSourceUsesSymmetricRange(activeViSource)) {
        const span = Math.max(Math.abs(s.min), Math.abs(s.max), 1e-12);
        setViDataMin(-span);
        setViDataMax(span);
      } else {
        setViDataMin(s.min);
        setViDataMax(s.max);
      }
    }

    // Apply scale transformation for histogram display
    if (!compareMode) {
      const scaledData = new Float32Array(numFloats);
      if (viScaleMode === "log") {
        for (let i = 0; i < numFloats; i++) {
          scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
        }
      } else {
        scaledData.set(rawData);
      }
      setViHistogramBins(null);
      setViHistogramData(scaledData);
    }
  }, [activeViSource, compareMode, displayedVirtualImageBytes, viScaleMode]);

  React.useEffect(() => {
    if (!compareMode) return;
    const expectedFloats = Math.max(0, (comparePanelCount || 0) * shapeRows * shapeCols);
    if (!displayedCompareVirtualImageBytes || expectedFloats === 0 || displayedCompareVirtualImageBytes.byteLength < expectedFloats * 4) {
      return;
    }
    const rawData = new Float32Array(
      displayedCompareVirtualImageBytes.buffer,
      displayedCompareVirtualImageBytes.byteOffset,
      expectedFloats,
    );
    const s = computeStats(rawData);
    setViStats([s.mean, s.min, s.max, s.std]);
    if (viSourceUsesSymmetricRange(activeViSource)) {
      const span = Math.max(Math.abs(s.min), Math.abs(s.max), 1e-12);
      setViDataMin(-span);
      setViDataMax(span);
    } else {
      setViDataMin(s.min);
      setViDataMax(s.max);
    }

    const scaledData = new Float32Array(expectedFloats);
    if (viScaleMode === "log") {
      for (let i = 0; i < expectedFloats; i++) {
        scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
      }
    } else {
      scaledData.set(rawData);
    }
    setViHistogramBins(null);
    setViHistogramData(scaledData);
  }, [activeViSource, compareMode, comparePanelCount, displayedCompareVirtualImageBytes, shapeCols, shapeRows, viScaleMode]);

  React.useEffect(() => {
    if (!compareMode || activeViSource !== "roi") return;
    const engine = viGpuColormapRef.current;
    if (!engine) return;
    const slotIndices = visibleCompareHistogramFrames
      .map((frame) => compareGpuSlotsRef.current.get(frame))
      .filter((slot): slot is number => slot !== undefined);
    if (slotIndices.length === 0) return;

    let cancelled = false;
    const generation = ++compareGpuHistogramGenRef.current;
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const rawRanges = await engine.computeRangeBatch(slotIndices);
          if (cancelled || generation !== compareGpuHistogramGenRef.current) return;
          const logScale = viScaleMode === "log";
          const ranges = rawRanges
            .map((range) => {
              if (!Number.isFinite(range.min) || !Number.isFinite(range.max)) return null;
              if (!logScale) return range;
              return {
                min: Math.log1p(Math.max(0, range.min)),
                max: Math.log1p(Math.max(0, range.max)),
              };
            })
            .filter((range): range is { min: number; max: number } => Boolean(range));
          if (ranges.length === 0) return;
          let dmin = Number.POSITIVE_INFINITY;
          let dmax = Number.NEGATIVE_INFINITY;
          ranges.forEach((range) => {
            if (range.min < dmin) dmin = range.min;
            if (range.max > dmax) dmax = range.max;
          });
          if (!Number.isFinite(dmin) || !Number.isFinite(dmax)) return;
          if (dmax <= dmin) dmax = dmin + 1e-12;
          const histograms = await engine.computeHistogramBatch(
            slotIndices,
            slotIndices.map(() => ({ min: dmin, max: dmax })),
            logScale,
          );
          if (cancelled || generation !== compareGpuHistogramGenRef.current || histograms.length === 0) return;
          const merged = new Float32Array(256);
          histograms.forEach((histogram) => {
            for (let i = 0; i < Math.min(256, histogram.length); i++) {
              merged[i] += Number(histogram[i]) || 0;
            }
          });
          setViDataMin(dmin);
          setViDataMax(dmax);
          setViHistogramData(null);
          setViHistogramBins(merged);
        } catch (error) {
          console.warn("Show4DSTEM multiple histogram update failed", error);
        }
      })();
    }, 120);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [activeViSource, compareGpuVersion, compareMode, viScaleMode, visibleCompareHistogramFrames]);

  // Render DP with zoom (use summed DP when VI ROI is active)
  // Expensive: colormap + data processing → cached offscreen canvas
  React.useEffect(() => {
    const sourceBytes = displayedDpBytes;
    if (!sourceBytes) return;

    const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;

    // Parse raw float32 data and apply scale transformation
    const rawData = new Float32Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength / 4);
    let scaled: Float32Array;
    if (dpScaleMode === "log") {
      scaled = new Float32Array(rawData.length);
      for (let i = 0; i < rawData.length; i++) {
        scaled[i] = Math.log1p(Math.max(0, rawData[i]));
      }
    } else {
      scaled = rawData;
    }

    const { min: dataMin, max: dataMax } = findDataRange(scaled);

    let vmin: number, vmax: number;
    if (traitDpVmin != null && traitDpVmax != null) {
      if (dpScaleMode === "log") {
        vmin = Math.log1p(Math.max(traitDpVmin, 0));
        vmax = Math.log1p(Math.max(traitDpVmax, 0));
      } else {
        vmin = traitDpVmin;
        vmax = traitDpVmax;
      }
    } else {
      ({ vmin, vmax } = sliderRange(dataMin, dataMax, dpVminPct, dpVmaxPct));
    }

    let offscreen = dpOffscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      dpOffscreenRef.current = offscreen;
    }
    const sizeChanged = offscreen.width !== detCols || offscreen.height !== detRows;
    if (sizeChanged) {
      offscreen.width = detCols;
      offscreen.height = detRows;
      dpImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    let imgData = dpImageDataRef.current;
    if (!imgData) {
      imgData = offCtx.createImageData(detCols, detRows);
      dpImageDataRef.current = imgData;
    }
    applyColormap(scaled, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);
    // Cache colorbar range for the UI overlay (avoids recomputing findDataRange on every zoom/pan)
    dpColorbarVminRef.current = vmin;
    dpColorbarVmaxRef.current = vmax;
    setDpOffscreenVersion(v => v + 1);
  }, [displayedDpBytes, detRows, detCols, dpColormap, dpVminPct, dpVmaxPct, dpScaleMode, traitDpVmin, traitDpVmax]);

  // Cheap: zoom/pan redraw — just drawImage from cached offscreen
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  React.useLayoutEffect(() => {
    const offscreen = dpOffscreenRef.current;
    if (!offscreen || !dpCanvasRef.current) return;
    const canvas = dpCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(dpPanX, dpPanY);
    ctx.scale(dpZoom, dpZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [dpOffscreenVersion, dpZoom, dpPanX, dpPanY]);

  // Render DP overlay - just clear (ROI shapes now drawn on high-DPI UI canvas)
  React.useEffect(() => {
    if (!dpOverlayRef.current) return;
    const canvas = dpOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // All visual overlays (crosshair, ROI shapes, scale bar) are now on dpUiRef for crisp rendering
  }, [localKCol, localKRow, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner, dpZoom, dpPanX, dpPanY, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, detRows, detCols]);

  // Expensive: VI colormap + data processing → cached offscreen canvas
  React.useEffect(() => {
    if (!rawVirtualImageRef.current) return;
    if (
      !compareMode
      && viGpuImageRef.current
      && activeViSource === viGpuImageRef.current.source
    ) {
      setViGpuRetainedReady(false);
      return;
    }

    const width = shapeCols;
    const height = shapeRows;
    const filtered = rawVirtualImageRef.current;

    let scaled = filtered;
    if (viScaleMode === "log") {
      scaled = new Float32Array(filtered.length);
      for (let i = 0; i < filtered.length; i++) {
        scaled[i] = Math.log1p(Math.max(0, filtered[i]));
      }
    }

    // Compute min/max from the data we just received. Do NOT use Python's
    // viDataMin/viDataMax traits here: they arrive as separate comm messages
    // and can be stale on rapid preset clicks (BF↔ABF), causing the render
    // to apply the WRONG normalization range and produce a uniform white/black
    // VI panel until comm catches up. findDataRange on a scan-shape buffer
    // (~64K-256K floats) is sub-millisecond.
    const r = findDataRange(scaled);
    let dataMin = r.min;
    let dataMax = r.max;
    if (viSourceUsesSymmetricRange(activeViSource)) {
      const span = Math.max(Math.abs(r.min), Math.abs(r.max), 1e-12);
      dataMin = -span;
      dataMax = span;
    }

    // Apply absolute bounds or percentile clipping
    let vmin: number, vmax: number;
    if (traitViVmin != null && traitViVmax != null) {
      if (viScaleMode === "log") {
        vmin = Math.log1p(Math.max(traitViVmin, 0));
        vmax = Math.log1p(Math.max(traitViVmax, 0));
      } else {
        vmin = traitViVmin;
        vmax = traitViVmax;
      }
    } else if (viAutoContrast) {
      ({ vmin, vmax } = percentileClip(scaled, 1, 99));
      const span = dataMax - dataMin;
      if (span > 0) {
        const lo = Math.max(0, Math.min(100, ((vmin - dataMin) / span) * 100));
        const hi = Math.max(0, Math.min(100, ((vmax - dataMin) / span) * 100));
        if (Math.abs(lo - viVminPct) > 0.5) setViVminPct(lo);
        if (Math.abs(hi - viVmaxPct) > 0.5) setViVmaxPct(hi);
      }
    } else {
      ({ vmin, vmax } = sliderRange(dataMin, dataMax, viVminPct, viVmaxPct));
    }

    const lut = COLORMAPS[viColormap] || COLORMAPS.inferno;
    let offscreen = viOffscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      viOffscreenRef.current = offscreen;
    }
    const sizeChanged = offscreen.width !== width || offscreen.height !== height;
    if (sizeChanged) {
      offscreen.width = width;
      offscreen.height = height;
      viImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    let imageData = viImageDataRef.current;
    if (!imageData) {
      imageData = offCtx.createImageData(width, height);
      viImageDataRef.current = imageData;
    }
    applyColormap(scaled, imageData.data, lut, vmin, vmax);
    offCtx.putImageData(imageData, 0, 0);
    setViOffscreenVersion(v => v + 1);
  }, [activeViSource, compareMode, displayedVirtualImageBytes, shapeRows, shapeCols, viGpuVersion, viColormap, viVminPct, viVmaxPct, viScaleMode, traitViVmin, traitViVmax, viAutoContrast]);

  // WebGPU virtual-image display path: the reduction output stays as a GPUBuffer
  // and the colormap shader renders it to a dedicated canvas layer. The
  // virtual_image_bytes trait is still populated afterward so stats, FFT,
  // profile, COPY fallback, and non-WebGPU paths keep working.
  React.useEffect(() => {
    const gpuImage = viGpuImageRef.current;
    const engine = viGpuColormapRef.current;
    const raw = rawVirtualImageRef.current;
    const currentSource = normaliseViSource(model.get("vi_source"));
    if (
      !gpuImage
      || !engine
      || compareMode
      || (activeViSource !== gpuImage.source && currentSource !== gpuImage.source)
    ) {
      return;
    }

    const expectedPixels = gpuImage.width * gpuImage.height;
    let vmin: number | null = null;
    let vmax: number | null = null;
    const rawReady = Boolean(
      raw
      && raw.length === expectedPixels
      && rawVirtualImageVersionRef.current >= gpuImage.rawVersionAfter,
    );
    if (rawReady && raw) {
      let scaled = raw;
      if (viScaleMode === "log") {
        scaled = new Float32Array(raw.length);
        for (let i = 0; i < raw.length; i++) {
          scaled[i] = Math.log1p(Math.max(0, raw[i]));
        }
      }

      const r = findDataRange(scaled);
      let dataMin = r.min;
      let dataMax = r.max;
      if (viSourceUsesSymmetricRange(gpuImage.source)) {
        const span = Math.max(Math.abs(r.min), Math.abs(r.max), 1e-12);
        dataMin = -span;
        dataMax = span;
      }

      if (traitViVmin != null && traitViVmax != null) {
        if (viScaleMode === "log") {
          vmin = Math.log1p(Math.max(traitViVmin, 0));
          vmax = Math.log1p(Math.max(traitViVmax, 0));
        } else {
          vmin = traitViVmin;
          vmax = traitViVmax;
        }
      } else if (viAutoContrast) {
        ({ vmin, vmax } = percentileClip(scaled, 1, 99));
        const span = dataMax - dataMin;
        if (span > 0) {
          const lo = Math.max(0, Math.min(100, ((vmin - dataMin) / span) * 100));
          const hi = Math.max(0, Math.min(100, ((vmax - dataMin) / span) * 100));
          if (Math.abs(lo - viVminPct) > 0.5) setViVminPct(lo);
          if (Math.abs(hi - viVmaxPct) > 0.5) setViVmaxPct(hi);
        }
      } else {
        ({ vmin, vmax } = sliderRange(dataMin, dataMax, viVminPct, viVmaxPct));
      }
    } else if (traitViVmin != null && traitViVmax != null && viScaleMode !== "log") {
      vmin = traitViVmin;
      vmax = traitViVmax;
    }

    const lut = COLORMAPS[viColormap] || COLORMAPS.inferno;
    engine.uploadLUT(viColormap, lut);
    const renderStart = performance.now();
    let displayRange = "cpu";
    let durableFrame: Promise<ImageBitmap | null> | null = null;
    if (vmin != null && vmax != null) {
      durableFrame = engine.renderPanelSlotsToImageBitmapAsync(
        [gpuImage.slot],
        { vmin, vmax },
        viScaleMode === "log",
        {
          width: shapeCols,
          height: shapeRows,
          panelCount: 1,
          cols: 1,
          rows: 1,
          gap: 0,
          bgRgb: 0,
          transforms: [{ zoom: viZoom, panX: viPanX, panY: viPanY }],
          smooth: viSmooth,
        },
      );
    } else if (gpuImage.rangeMode === "gpu") {
      displayRange = "gpu";
      const renderOpts = {
        width: shapeCols,
        height: shapeRows,
        bgRgb: 0,
        transform: { zoom: viZoom, panX: viPanX, panY: viPanY },
        smooth: viSmooth,
      };
      durableFrame = engine.renderSlotDirectWithGpuRangeToImageBitmapAsync(
        gpuImage.slot,
        viVminPct,
        viVmaxPct,
        viScaleMode === "log",
        renderOpts,
      );
    }
    if (durableFrame) {
      const snapshotSerial = ++virtualGpuSnapshotSerialRef.current;
      void durableFrame.then(bitmap => {
        if (!bitmap) return;
        try {
          if (snapshotSerial !== virtualGpuSnapshotSerialRef.current) return;
          const retained = virtualCanvasRef.current;
          const retainedCtx = retained?.getContext("2d");
          if (!retained || !retainedCtx) return;
          retainedCtx.clearRect(0, 0, retained.width, retained.height);
          retainedCtx.drawImage(bitmap, 0, 0, retained.width, retained.height);
          setViGpuRetainedReady(true);
        } finally {
          bitmap.close();
        }
      }).catch(error => {
        setViGpuRetainedReady(false);
        console.warn("[Show4DSTEM] Could not retain the presented WebGPU virtual image", error);
      });
      publishShow4DSTEMViDisplay({
        source: gpuImage.source,
        gpuBufferToDisplay: true,
        rendered: true,
        width: shapeCols,
        height: shapeRows,
        slot: gpuImage.slot,
        rangeMode: displayRange,
        rawReady,
        rawVersion: rawVirtualImageVersionRef.current,
        rawVersionAfter: gpuImage.rawVersionAfter,
        renderSubmitMs: performance.now() - renderStart,
      });
    }
  }, [
    activeViSource,
    compareMode,
    displayedVirtualImageBytes,
    shapeRows,
    shapeCols,
    viGpuVersion,
    viColormap,
    viVminPct,
    viVmaxPct,
    viScaleMode,
    traitViVmin,
    traitViVmax,
    viAutoContrast,
    viZoom,
    viPanX,
    viPanY,
    viSmooth,
    model,
    setViVminPct,
    setViVmaxPct,
  ]);

  // Cheap: VI zoom/pan redraw — just drawImage from cached offscreen
  React.useLayoutEffect(() => {
    const offscreen = viOffscreenRef.current;
    if (!offscreen || !virtualCanvasRef.current) return;
    const canvas = virtualCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = viSmooth;
    if (viSmooth) ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(viPanX, viPanY);
    ctx.scale(viZoom, viZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [compareMode, viOffscreenVersion, viZoom, viPanX, viPanY, viSmooth]);

  // Render virtual image overlay (just clear - crosshair drawn on high-DPI UI canvas)
  React.useEffect(() => {
    if (!virtualOverlayRef.current) return;
    const canvas = virtualOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Crosshair and scale bar now drawn on high-DPI UI canvas (viUiRef)
  }, [localPosRow, localPosCol, isDraggingVI, viZoom, viPanX, viPanY, pixelSize, shapeRows, shapeCols]);

  // Compute FFT (expensive, async — only re-run on data/GPU changes)
  const fftRealRef = React.useRef<Float32Array | null>(null);
  const fftImagRef = React.useRef<Float32Array | null>(null);
  const fftRunSeqRef = React.useRef(0);
  const [fftVersion, setFftVersion] = React.useState(0);

  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !effectiveShowFft) { setFftCropDims(null); return; }
    const runSeq = ++fftRunSeqRef.current;
    let cancelled = false;
    let width = shapeCols;
    let height = shapeRows;
    let sourceData = rawVirtualImageRef.current;
    let origCropW = 0, origCropH = 0;

    // ROI FFT: crop virtual image to VI ROI region and pre-pad to power-of-2.
    // Use localViRoiCenter* (updated immediately on drag) instead of the synced
    // model traits, which lag by one comm roundtrip after a compound trait write.
    // Without this, FFT visibly stalls during rapid VI ROI drag.
    if (roiFftActive) {
      const cRow = localViRoiCenterRow ?? viRoiCenterRow;
      const cCol = localViRoiCenterCol ?? viRoiCenterCol;
      const crop = cropSingleROI(sourceData, shapeCols, shapeRows, viRoiMode, cRow, cCol, viRoiRadius, viRoiWidth, viRoiHeight);
      if (crop) {
        origCropW = crop.cropW;
        origCropH = crop.cropH;
        // Apply Hann window to crop at native dimensions BEFORE zero-padding
        if (fftWindow) applyHannWindow2D(crop.cropped, crop.cropW, crop.cropH);
        const padW = nextPow2(crop.cropW);
        const padH = nextPow2(crop.cropH);
        const padded = new Float32Array(padW * padH);
        for (let y = 0; y < crop.cropH; y++) {
          for (let x = 0; x < crop.cropW; x++) {
            padded[y * padW + x] = crop.cropped[y * crop.cropW + x];
          }
        }
        sourceData = padded;
        width = padW;
        height = padH;
      }
    }

    // Pre-pad non-power-of-2 full images so fft2d doesn't truncate frequency data
    if (!roiFftActive) {
      const padW = nextPow2(width);
      const padH = nextPow2(height);
      if (padW !== width || padH !== height) {
        const padded = new Float32Array(padW * padH);
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            padded[y * padW + x] = sourceData[y * width + x];
          }
        }
        sourceData = padded;
        width = padW;
        height = padH;
      }
    }

    const fftW = width, fftH = height;
    const commitFft = (
      real: Float32Array,
      imag: Float32Array,
      startedAt: number,
      mode: "webgpu" | "worker",
    ) => {
      if (cancelled || runSeq !== fftRunSeqRef.current) return;
      fftRealRef.current = real;
      fftImagRef.current = imag;
      if (origCropW > 0) {
        setFftCropDims({ cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH });
      } else if (fftW !== shapeCols || fftH !== shapeRows) {
        setFftCropDims({ cropWidth: shapeCols, cropHeight: shapeRows, fftWidth: fftW, fftHeight: fftH });
      } else {
        setFftCropDims(null);
      }
      const profile = {
        mode,
        width: fftW,
        height: fftH,
        ms: Math.round(performance.now() - startedAt),
        seq: runSeq,
      };
      const win = window as unknown as {
        __show4dstemFftProfile?: unknown;
        __show4dstemFftHistory?: unknown[];
      };
      const history = Array.isArray(win.__show4dstemFftHistory) ? win.__show4dstemFftHistory : [];
      history.push(profile);
      if (history.length > 60) history.splice(0, history.length - 60);
      win.__show4dstemFftProfile = profile;
      win.__show4dstemFftHistory = history;
      setFftVersion(v => v + 1);
    };
    const timer = window.setTimeout(() => {
      if (gpuFFTRef.current && gpuReady) {
        const runGpuFFT = async () => {
          const startedAt = performance.now();
          const real = sourceData.slice();
          const imag = new Float32Array(real.length);
          const { real: fReal, imag: fImag } = await gpuFFTRef.current!.fft2D(real, imag, fftW, fftH, false);
          if (cancelled || runSeq !== fftRunSeqRef.current) return;
          fftshift(fReal, fftW, fftH);
          fftshift(fImag, fftW, fftH);
          commitFft(fReal, fImag, startedAt, "webgpu");
        };
        runGpuFFT();
      } else {
        const runWorkerFFT = async () => {
          const startedAt = performance.now();
          const real = sourceData.slice();
          const imag = new Float32Array(real.length);
          const result = await fft2dAsync(real, imag, fftW, fftH, false);
          commitFft(result.real, result.imag, startedAt, "worker");
        };
        runWorkerFFT();
      }
    }, 16);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [displayedVirtualImageBytes, shapeRows, shapeCols, gpuReady, effectiveShowFft, roiFftActive, viRoiMode, viRoiCenterRow, viRoiCenterCol, localViRoiCenterRow, localViRoiCenterCol, viRoiRadius, viRoiWidth, viRoiHeight, fftWindow]);

  // Expensive: FFT magnitude + histogram + colormap → cached offscreen canvas
  React.useEffect(() => {
    if (!fftRealRef.current || !fftImagRef.current) return;
    if (!effectiveShowFft) return;

    const width = fftCropDims?.fftWidth ?? shapeCols;
    const height = fftCropDims?.fftHeight ?? shapeRows;
    const real = fftRealRef.current;
    const imag = fftImagRef.current;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;

    // Compute magnitude with scale mode
    let magnitude = fftMagnitudeRef.current;
    if (!magnitude || magnitude.length !== real.length) {
      magnitude = new Float32Array(real.length);
      fftMagnitudeRef.current = magnitude;
    }
    // Cache raw magnitude for peak-snap before applying scale transform
    let rawMag = fftMagCacheRef.current;
    if (!rawMag || rawMag.length !== real.length) {
      rawMag = new Float32Array(real.length);
      fftMagCacheRef.current = rawMag;
    }
    computeMagnitude(real, imag, rawMag);
    for (let i = 0; i < rawMag.length; i++) {
      magnitude[i] = fftScaleMode === "log" ? Math.log1p(rawMag[i]) : rawMag[i];
    }

    let displayMin: number, displayMax: number;
    if (fftAuto) {
      ({ min: displayMin, max: displayMax } = autoEnhanceFFT(magnitude, width, height));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(magnitude));
    }
    setFftDataMin(displayMin);
    setFftDataMax(displayMax);
    const magStats = computeStats(magnitude);
    setFftStats([magStats.mean, displayMin, displayMax, magStats.std]);
    setFftHistogramData(magnitude.slice());

    // Render to offscreen canvas
    let offscreen = fftOffscreenRef.current;
    if (!offscreen) { offscreen = document.createElement("canvas"); fftOffscreenRef.current = offscreen; }
    if (offscreen.width !== width || offscreen.height !== height) {
      offscreen.width = width; offscreen.height = height; fftImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;
    let imgData = fftImageDataRef.current;
    if (!imgData) { imgData = offCtx.createImageData(width, height); fftImageDataRef.current = imgData; }

    const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);
    applyColormap(magnitude, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);
    setFftOffscreenVersion(v => v + 1);
  }, [effectiveShowFft, fftVersion, fftScaleMode, fftAuto, fftVminPct, fftVmaxPct, fftColormap, shapeRows, shapeCols, fftCropDims]);

  // Cheap: FFT zoom/pan redraw — just drawImage from cached offscreen
  React.useLayoutEffect(() => {
    if (!fftCanvasRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const offscreen = fftOffscreenRef.current;
    if (!offscreen || !effectiveShowFft) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
    const fftW = offscreen.width;
    const fftH = offscreen.height;
    const canvasW = canvas.width;
    const canvasH = canvas.height;
    // Use bilinear smoothing when FFT dims differ from canvas (non-pow2 padding or ROI crop).
    // Stretch offscreen to fill canvas via the 9-arg drawImage form: ROI FFT crops produce a
    // small offscreen (e.g. 64×64) that would otherwise blit at native size in the corner.
    ctx.imageSmoothingEnabled = fftW !== canvasW || fftH !== canvasH;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(fftPanX, fftPanY);
    ctx.scale(fftZoom, fftZoom);
    ctx.drawImage(offscreen, 0, 0, fftW, fftH, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [fftOffscreenVersion, fftZoom, fftPanX, fftPanY, effectiveShowFft]);

  // Render FFT overlay with d-spacing crosshair marker
  React.useEffect(() => {
    if (!fftOverlayRef.current) return;
    const canvas = fftOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // D-spacing crosshair marker
    if (fftClickInfo && effectiveShowFft) {
      const fftW = fftCropDims?.fftWidth ?? shapeCols;
      const fftH = fftCropDims?.fftHeight ?? shapeRows;
      ctx.save();
      // Forward mapping: image col/row → canvas x/y (matches stretched drawImage).
      const screenX = fftPanX + fftZoom * (fftClickInfo.col * canvas.width / fftW);
      const screenY = fftPanY + fftZoom * (fftClickInfo.row * canvas.height / fftH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
      ctx.shadowBlur = 2;
      ctx.lineWidth = 1.5;
      // Scale crosshair size relative to canvas (not zoom-dependent)
      const r = 8 * Math.max(fftW, fftH) / 450;
      const gap = 3 * Math.max(fftW, fftH) / 450;
      const dotR = 4 * Math.max(fftW, fftH) / 450;
      ctx.beginPath();
      ctx.moveTo(screenX - r, screenY); ctx.lineTo(screenX - gap, screenY);
      ctx.moveTo(screenX + gap, screenY); ctx.lineTo(screenX + r, screenY);
      ctx.moveTo(screenX, screenY - r); ctx.lineTo(screenX, screenY - gap);
      ctx.moveTo(screenX, screenY + gap); ctx.lineTo(screenX, screenY + r);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(screenX, screenY, dotR, 0, Math.PI * 2);
      ctx.stroke();
      if (fftClickInfo.dSpacing != null) {
        const d = fftClickInfo.dSpacing;
        const label = d >= 10 ? `d = ${(d / 10).toFixed(2)} nm` : `d = ${d.toFixed(2)} \u00C5`;
        const fontSize = Math.max(10, Math.round(11 * Math.max(fftW, fftH) / 450));
        ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
        ctx.fillStyle = "white";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(label, screenX + r + 4, screenY - gap);
      }
      ctx.restore();
    }
  }, [fftZoom, fftPanX, fftPanY, effectiveShowFft, fftClickInfo, shapeCols, shapeRows, fftCropDims]);

  // Clear FFT click info when virtual image changes (scan position, VI ROI, etc.)
  React.useEffect(() => {
    setFftClickInfo(null);
  }, [displayedVirtualImageBytes]);

  // ─────────────────────────────────────────────────────────────────────────
  // High-DPI Scale Bar UI Overlays
  // ─────────────────────────────────────────────────────────────────────────
  
  // DP scale bar + crosshair + ROI overlay + profile line (high-DPI)
  React.useEffect(() => {
    if (!dpUiRef.current) return;
    const canvas = dpUiRef.current;
    const ctx = canvas.getContext("2d");
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
    // Draw scale bar first when enabled.
    const kUnit = kCalibrated ? kPixelUnit : "px";
    if (showScaleBar) drawScaleBarHiDPI(canvas, DPR, dpZoom, kPixelSize || 1, kUnit, detCols);
    // Draw detector ROI only when the displayed virtual image is produced from
    // the live detector mask. Precomputed DPC/SSB maps are static products, so
    // showing the BF/ADF circle there is misleading.
    if (roiVirtualDetectorActive) {
      if (roiMode === "point") {
        drawDpCrosshairHiDPI(dpUiRef.current, DPR, localKCol, localKRow, dpZoom, dpPanX, dpPanY, detCols, detRows, isDraggingDP, roiColors);
      } else {
        drawRoiOverlayHiDPI(
          dpUiRef.current, DPR, roiMode,
          localKCol, localKRow, roiRadius, roiRadiusInner, roiWidth, roiHeight,
          dpZoom, dpPanX, dpPanY, detCols, detRows,
          isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner,
          roiColors
        );
      }
    }

    // Profile line overlay
    if (profileActive && profilePoints.length > 0) {
        const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.save();
        ctx.scale(DPR, DPR);
        const cssW = canvas.width / DPR;
        const cssH = canvas.height / DPR;
        const scaleX = cssW / detCols;
        const scaleY = cssH / detRows;
        const toScreenX = (col: number) => col * dpZoom * scaleX + dpPanX * scaleX;
        const toScreenY = (row: number) => row * dpZoom * scaleY + dpPanY * scaleY;

        // Draw point A
        const ax = toScreenX(profilePoints[0].col);
        const ay = toScreenY(profilePoints[0].row);
        ctx.fillStyle = themeColors.accent;
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();

        if (profilePoints.length === 2) {
          const bx = toScreenX(profilePoints[1].col);
          const by = toScreenY(profilePoints[1].row);

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
              ctx.lineWidth = 1;
              ctx.setLineDash([3, 3]);
              ctx.beginPath();
              ctx.moveTo(toScreenX(profilePoints[0].col + perpC), toScreenY(profilePoints[0].row + perpR));
              ctx.lineTo(toScreenX(profilePoints[1].col + perpC), toScreenY(profilePoints[1].row + perpR));
              ctx.lineTo(toScreenX(profilePoints[1].col - perpC), toScreenY(profilePoints[1].row - perpR));
              ctx.lineTo(toScreenX(profilePoints[0].col - perpC), toScreenY(profilePoints[0].row - perpR));
              ctx.closePath();
              ctx.fill();
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }

          // Draw line A->B
          ctx.strokeStyle = themeColors.accent;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();

          // Draw point B
          ctx.fillStyle = themeColors.accent;
          ctx.beginPath();
          ctx.arc(bx, by, 4, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }
    }

    // Colorbar overlay — uses cached vmin/vmax from the expensive DP offscreen effect
    if (showDpColorbar) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.save();
        ctx.scale(DPR, DPR);
        const cssW = canvas.width / DPR;
        const cssH = canvas.height / DPR;
        const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;
        drawColorbar(ctx, cssW, cssH, lut, dpColorbarVminRef.current, dpColorbarVmaxRef.current, dpScaleMode === "log");
        ctx.restore();
      }
    }
  }, [roiVirtualDetectorActive, dpZoom, dpPanX, dpPanY, kPixelSize, kPixelUnit, kCalibrated, detRows, detCols, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, localKCol, localKRow, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner,
      profileActive, profilePoints, profileWidth, themeColors, showDpColorbar, showScaleBar, dpColormap, dpScaleMode, dpVminPct, dpVmaxPct, canvasSize, roiColors]);
  
  // VI scale bar + crosshair + ROI + profile lines (high-DPI)
  React.useEffect(() => {
    if (!viUiRef.current) return;
    const canvas = viUiRef.current;
    const ctx = canvas.getContext("2d");
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
    // Draw scale bar first when enabled.
    if (showScaleBar) drawScaleBarHiDPI(canvas, DPR, viZoom, pixelSize || 1, pixelUnit || "px", shapeCols);
    // Draw crosshair only when ROI is off (ROI replaces the crosshair)
    if (!viRoiMode || viRoiMode === "off") {
      drawViPositionMarker(viUiRef.current, DPR, localPosRow, localPosCol, viZoom, viPanX, viPanY, shapeCols, shapeRows, isDraggingVI);
    } else {
      // Draw VI ROI instead of crosshair
      drawViRoiOverlayHiDPI(
        viUiRef.current, DPR, viRoiMode,
        localViRoiCenterRow, localViRoiCenterCol, viRoiRadius || 5, viRoiWidth || 10, viRoiHeight || 10,
        viZoom, viPanX, viPanY, shapeCols, shapeRows,
        isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize
      );
    }
    // Draw VI profile lines
    if (viProfileActive && viProfilePoints.length > 0) {
      const canvas = viUiRef.current;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const cssW = canvas.width / DPR;
        const cssH = canvas.height / DPR;
        const scaleX = cssW / shapeCols;
        const scaleY = cssH / shapeRows;
        ctx.save();
        ctx.scale(DPR, DPR);
        ctx.strokeStyle = "#a0f";
        ctx.lineWidth = 2;
        ctx.shadowColor = "rgba(0,0,0,0.5)";
        ctx.shadowBlur = 2;
        if (viProfilePoints.length >= 1) {
          const p0 = viProfilePoints[0];
          const x0 = p0.col * viZoom * scaleX + viPanX * scaleX;
          const y0 = p0.row * viZoom * scaleY + viPanY * scaleY;
          ctx.beginPath();
          ctx.arc(x0, y0, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#fff";
          ctx.fillText("1", x0 + 6, y0 - 6);
        }
        if (viProfilePoints.length === 2) {
          const p0 = viProfilePoints[0], p1 = viProfilePoints[1];
          const x0 = p0.col * viZoom * scaleX + viPanX * scaleX;
          const y0 = p0.row * viZoom * scaleY + viPanY * scaleY;
          const x1 = p1.col * viZoom * scaleX + viPanX * scaleX;
          const y1 = p1.row * viZoom * scaleY + viPanY * scaleY;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(x1, y1, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#fff";
          ctx.fillText("2", x1 + 6, y1 - 6);
        }
        ctx.restore();
      }
    }
  }, [compareMode, viZoom, viPanX, viPanY, pixelSize, pixelUnit, showScaleBar, shapeRows, shapeCols, localPosRow, localPosCol, isDraggingVI,
      viRoiMode, localViRoiCenterRow, localViRoiCenterCol, viRoiRadius, viRoiWidth, viRoiHeight,
      isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize, canvasSize, viProfileActive, viProfilePoints]);

  // ── DP Profile computation ──
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawDpDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      void sampleLineProfileWebGPU(rawDpDataRef.current, detCols, detRows, p0.row, p0.col, p1.row, p1.col, profileWidth)
        .then(setProfileData)
        .catch(error => console.error("[Show4DSTEM] WebGPU DP profile failed", error));
      if (!profileActive) setProfileActive(true);
    } else {
      setProfileData(null);
    }
  }, [profilePoints, profileWidth, frameBytes]);

  // ── VI Profile computation ──
  React.useEffect(() => {
    if (viProfilePoints.length === 2 && rawViDataRef.current && shapeCols > 0 && shapeRows > 0) {
      const p0 = viProfilePoints[0], p1 = viProfilePoints[1];
      void sampleLineProfileWebGPU(rawViDataRef.current, shapeCols, shapeRows, p0.row, p0.col, p1.row, p1.col, 1)
        .then(setViProfileData)
        .catch(error => console.error("[Show4DSTEM] WebGPU VI profile failed", error));
    } else {
      setViProfileData(null);
    }
  }, [viProfilePoints, displayedVirtualImageBytes, shapeCols, shapeRows]);

  // ── Profile sparkline rendering ──
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasSize;
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
      ctx.fillText("Click two points on the DP to draw a profile", cssW / 2, cssH / 2);
      profileBaseImageRef.current = null;
      profileLayoutRef.current = null;
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

    // X-axis: calibrated distance
    let totalDist = profileData.length - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const dx = profilePoints[1].col - profilePoints[0].col;
      const dy = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      if (kCalibrated && kPixelSize > 0) {
        totalDist = distPx * kPixelSize;
        xUnit = kPixelUnit;
      } else {
        totalDist = distPx;
      }
    }

    // Draw axes
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Draw profile curve
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
      const label = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${label} ${xUnit}` : label, x, tickY + 4);
    }

    // Y-axis min/max labels (left margin)
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), 2, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), 2, padTop + plotH);

    // Save base image and layout for hover
    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [profileData, profilePoints, kPixelSize, kCalibrated, themeInfo.theme, themeColors.accent, canvasSize, profileHeight]);

  // DP Profile hover handlers
  const handleProfileMouseMove = React.useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    const layout = profileLayoutRef.current;
    if (!canvas || !base || !layout || !profileData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit } = layout;
    const range = gMax - gMin || 1;

    // Restore base image
    ctx.putImageData(base, 0, 0);

    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Vertical crosshair
    const isDark = themeInfo.theme === "dark";
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on curve + value
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

    ctx.restore();
  }, [profileData, themeInfo.theme, themeColors.accent]);

  const handleProfileMouseLeave = React.useCallback(() => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  }, []);

  // DP Profile resize handlers
  React.useEffect(() => {
    if (!isResizingProfile) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!profileResizeStart.current) return;
      const deltaY = e.clientY - profileResizeStart.current.startY;
      const newHeight = Math.max(40, Math.min(300, profileResizeStart.current.startHeight + deltaY));
      setProfileHeight(newHeight);
    };
    const handleMouseUp = () => {
      setIsResizingProfile(false);
      profileResizeStart.current = null;
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingProfile]);

  // ── VI Profile sparkline rendering ──
  React.useEffect(() => {
    const canvas = viProfileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = viCanvasWidth;
    const cssH = viProfileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, cssW, cssH);

    if (!viProfileData || viProfileData.length < 2) {
      ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = isDark ? "#555" : "#999";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Click two points on the VI to draw a profile", cssW / 2, cssH / 2);
      viProfileBaseImageRef.current = null;
      viProfileLayoutRef.current = null;
      return;
    }

    const padLeft = 40;
    const padRight = 8;
    const padTop = 6;
    const padBottom = 18;
    const plotW = cssW - padLeft - padRight;
    const plotH = cssH - padTop - padBottom;

    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < viProfileData.length; i++) {
      if (viProfileData[i] < gMin) gMin = viProfileData[i];
      if (viProfileData[i] > gMax) gMax = viProfileData[i];
    }
    const range = gMax - gMin || 1;

    // X-axis: calibrated distance
    let totalDist = viProfileData.length - 1;
    let xUnit = "px";
    if (viProfilePoints.length === 2 && pixelSize > 0) {
      const dx = viProfilePoints[1].col - viProfilePoints[0].col;
      const dy = viProfilePoints[1].row - viProfilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      totalDist = distPx * pixelSize;
      xUnit = pixelUnit;
    }

    // Draw axes
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Draw profile curve
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < viProfileData.length; i++) {
      const x = padLeft + (i / (viProfileData.length - 1)) * plotW;
      const y = padTop + plotH - ((viProfileData[i] - gMin) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

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
      const label = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${label} ${xUnit}` : label, x, tickY + 4);
    }

    // Y-axis min/max labels (left margin)
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), 2, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), 2, padTop + plotH);

    // Save base image and layout for hover
    viProfileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    viProfileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [viProfileData, viProfilePoints, pixelSize, themeInfo.theme, themeColors.accent, viCanvasWidth, viProfileHeight]);

  // VI Profile hover handlers
  const handleViProfileMouseMove = React.useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = viProfileCanvasRef.current;
    const base = viProfileBaseImageRef.current;
    const layout = viProfileLayoutRef.current;
    if (!canvas || !base || !layout || !viProfileData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit } = layout;
    const range = gMax - gMin || 1;

    // Restore base image
    ctx.putImageData(base, 0, 0);

    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Vertical crosshair
    const isDark = themeInfo.theme === "dark";
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on curve + value
    const dataIdx = Math.min(viProfileData.length - 1, Math.max(0, Math.round(frac * (viProfileData.length - 1))));
    const val = viProfileData[dataIdx];
    const y = padTop + plotH - ((val - gMin) / range) * plotH;
    ctx.fillStyle = themeColors.accent;
    ctx.beginPath();
    ctx.arc(cssX, y, 3, 0, Math.PI * 2);
    ctx.fill();

    // Value readout label
    const dist = frac * totalDist;
    const label = `${formatNumber(val)}  @  ${dist.toFixed(1)} ${xUnit}`;
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

    ctx.restore();
  }, [viProfileData, themeInfo.theme, themeColors.accent]);

  const handleViProfileMouseLeave = React.useCallback(() => {
    const canvas = viProfileCanvasRef.current;
    const base = viProfileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  }, []);

  // VI Profile resize handlers
  React.useEffect(() => {
    if (!isResizingViProfile) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!viProfileResizeStart.current) return;
      const deltaY = e.clientY - viProfileResizeStart.current.startY;
      const newHeight = Math.max(40, Math.min(300, viProfileResizeStart.current.startHeight + deltaY));
      setViProfileHeight(newHeight);
    };
    const handleMouseUp = () => {
      setIsResizingViProfile(false);
      viProfileResizeStart.current = null;
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingViProfile]);

  // Generic zoom handler
  const createZoomHandler = (
    setZoom: React.Dispatch<React.SetStateAction<number>>,
    setPanX: React.Dispatch<React.SetStateAction<number>>,
    setPanY: React.Dispatch<React.SetStateAction<number>>,
    viewRef: React.RefObject<{ zoom: number; panX: number; panY: number; raf: number }>,
    canvasRef: React.RefObject<HTMLCanvasElement | null>,
  ) => (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.stopPropagation();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const v = viewRef.current;
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, v.zoom * zoomFactor));
    const zoomRatio = newZoom / v.zoom;
    // Accumulate synchronously against the live ref (handles a burst of trackpad
    // wheel events within one frame correctly), flush to React state once per rAF.
    v.zoom = newZoom;
    v.panX = mouseX - (mouseX - v.panX) * zoomRatio;
    v.panY = mouseY - (mouseY - v.panY) * zoomRatio;
    if (v.raf === 0) {
      v.raf = requestAnimationFrame(() => {
        v.raf = 0;
        setZoom(v.zoom); setPanX(v.panX); setPanY(v.panY);
      });
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Mouse Handlers
  // ─────────────────────────────────────────────────────────────────────────

  // Helper: convert screen-pixel hit radius to image-pixel radius
  // handleRadius=6 CSS px drawn, hit area ~10 CSS px → convert to image coords
  const dpHitRadius = RESIZE_HIT_AREA_PX * Math.max(detCols, detRows) / canvasSize / dpZoom;
  const activeRoiCenterCol = Number.isFinite(localKCol) ? localKCol : roiCenterCol;
  const activeRoiCenterRow = Number.isFinite(localKRow) ? localKRow : roiCenterRow;

  const isInResizableRadiusBand = (distance: number, radius: number, innerRadius: number = 0): boolean => {
    if (!Number.isFinite(distance) || !Number.isFinite(radius) || radius <= 0) return false;
    const ringWidth = Math.max(radius - innerRadius, 1);
    const edgePad = Math.max(dpHitRadius, Math.min(radius * 0.35, 8), ringWidth * 0.35);
    if (innerRadius > 0) {
      return distance >= Math.max(0, innerRadius - edgePad) && distance <= radius + edgePad;
    }
    return distance >= Math.max(0, radius * 0.45 - edgePad) && distance <= radius + edgePad;
  };

  // Helper: check if point is near the outer resize handle.
  // The drawn handle is a tiny 6px dot - too small to grab by hand, especially
  // on a binned detector where the whole pattern is ~48px. So we accept a click
  // anywhere on the ROI's EDGE (circle perimeter / square border), not just the
  // 45-deg handle dot. Dragging the edge is the natural "resize" gesture.
  const isNearResizeHandle = (imgX: number, imgY: number): boolean => {
    if (!roiVirtualDetectorActive) return false;
    if (roiMode === "rect") {
      const handleX = activeRoiCenterCol + roiWidth / 2;
      const handleY = activeRoiCenterRow + roiHeight / 2;
      if (Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2) < dpHitRadius) return true;
      const dx = Math.abs(imgX - activeRoiCenterCol), dy = Math.abs(imgY - activeRoiCenterRow);
      const onVert = Math.abs(dx - roiWidth / 2) < dpHitRadius && dy <= roiHeight / 2 + dpHitRadius;
      const onHorz = Math.abs(dy - roiHeight / 2) < dpHitRadius && dx <= roiWidth / 2 + dpHitRadius;
      return onVert || onHorz;
    }
    if ((roiMode !== "circle" && roiMode !== "square" && roiMode !== "annular") || !roiRadius) return false;
    const offset = roiMode === "square" ? roiRadius : roiRadius * CIRCLE_HANDLE_ANGLE;
    const handleX = activeRoiCenterCol + offset;
    const handleY = activeRoiCenterRow + offset;
    if (Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2) < dpHitRadius) return true;
    const dx = imgX - activeRoiCenterCol, dy = imgY - activeRoiCenterRow;
    // GENEROUS grab: a hand can't hit a thin ring. Treat the OUTER HALF of the
    // ROI (and just outside it) as the resize zone; the inner half is the move
    // zone. So grabbing anywhere near the rim resizes - no pixel precision needed.
    if (roiMode === "square") {
      const cheb = Math.max(Math.abs(dx), Math.abs(dy));
      return isInResizableRadiusBand(cheb, roiRadius);
    }
    const distFromCenter = Math.sqrt(dx ** 2 + dy ** 2);
    if (roiMode === "annular") {
      return isInResizableRadiusBand(distFromCenter, roiRadius, roiRadiusInner || 0);
    }
    return isInResizableRadiusBand(distFromCenter, roiRadius);
  };

  // Helper: check if point is near the inner resize handle (annular mode only)
  const isNearResizeHandleInner = (imgX: number, imgY: number): boolean => {
    if (!roiVirtualDetectorActive) return false;
    if (roiMode !== "annular" || !roiRadiusInner) return false;
    const offset = roiRadiusInner * CIRCLE_HANDLE_ANGLE;
    const handleX = activeRoiCenterCol + offset;
    const handleY = activeRoiCenterRow + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < dpHitRadius;
  };

  // Helper: check if point is near VI ROI resize handle (same logic as DP)
  // Hit area is capped to avoid overlap with center for small ROIs
  const viHitRadius = RESIZE_HIT_AREA_PX * Math.max(shapeRows, shapeCols) / canvasSize / viZoom;
  const isNearViRoiResizeHandle = (imgX: number, imgY: number): boolean => {
    if (!viRoiMode || viRoiMode === "off") return false;
    if (viRoiMode === "rect") {
      const halfH = (viRoiHeight || 10) / 2;
      const halfW = (viRoiWidth || 10) / 2;
      const handleX = localViRoiCenterRow + halfH;
      const handleY = localViRoiCenterCol + halfW;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      const cornerDist = Math.sqrt(halfW ** 2 + halfH ** 2);
      const hitArea = Math.min(viHitRadius, cornerDist * 0.5);
      return dist < hitArea;
    }
    if (viRoiMode === "circle" || viRoiMode === "square") {
      const radius = viRoiRadius || 5;
      const offset = viRoiMode === "square" ? radius : radius * CIRCLE_HANDLE_ANGLE;
      const handleX = localViRoiCenterRow + offset;
      const handleY = localViRoiCenterCol + offset;
      const hitArea = Math.min(viHitRadius, radius * 0.5);
      if (Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2) < hitArea) return true;
      // GENEROUS grab: outer half of the ROI (and just outside) resizes; inner
      // half moves. No pixel precision needed to grab the rim by hand.
      const dx = imgX - localViRoiCenterRow, dy = imgY - localViRoiCenterCol;
      if (viRoiMode === "square") {
        const cheb = Math.max(Math.abs(dx), Math.abs(dy));
        return cheb >= radius * 0.5 && cheb <= radius * 1.8 + viHitRadius;
      }
      return Math.sqrt(dx ** 2 + dy ** 2) >= radius * 0.5 && Math.sqrt(dx ** 2 + dy ** 2) <= radius * 1.8 + viHitRadius;
    }
    return false;
  };

  // Helper: check if point is inside the DP ROI area
  const isInsideDpRoi = (imgX: number, imgY: number): boolean => {
    if (!roiVirtualDetectorActive) return false;
    if (roiMode === "point") return false;
    const dx = imgX - activeRoiCenterCol;
    const dy = imgY - activeRoiCenterRow;
    if (roiMode === "circle") return Math.sqrt(dx * dx + dy * dy) <= (roiRadius || 5);
    if (roiMode === "square") return Math.abs(dx) <= (roiRadius || 5) && Math.abs(dy) <= (roiRadius || 5);
    if (roiMode === "annular") { const d = Math.sqrt(dx * dx + dy * dy); return d <= (roiRadius || 20) && d >= (roiRadiusInner || 5); }
    if (roiMode === "rect") return Math.abs(dx) <= (roiWidth || 10) / 2 && Math.abs(dy) <= (roiHeight || 10) / 2;
    return false;
  };

  // Helper: check if point is inside the VI ROI area
  const isInsideViRoi = (imgX: number, imgY: number): boolean => {
    if (!viRoiMode || viRoiMode === "off") return false;
    const dx = imgY - localViRoiCenterCol;
    const dy = imgX - localViRoiCenterRow;
    if (viRoiMode === "circle") return Math.sqrt(dx * dx + dy * dy) <= (viRoiRadius || 5);
    if (viRoiMode === "square") return Math.abs(dx) <= (viRoiRadius || 5) && Math.abs(dy) <= (viRoiRadius || 5);
    if (viRoiMode === "rect") return Math.abs(dx) <= (viRoiWidth || 10) / 2 && Math.abs(dy) <= (viRoiHeight || 10) / 2;
    return false;
  };

  // Mouse handlers
  const getDpImageCoordsFromClient = React.useCallback((clientX: number, clientY: number): { imgX: number; imgY: number } | null => {
    const canvas = dpOverlayRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const screenX = (clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (clientY - rect.top) * (canvas.height / rect.height);
    return {
      imgX: (screenX - dpPanX) / dpZoom,
      imgY: (screenY - dpPanY) / dpZoom,
    };
  }, [dpPanX, dpPanY, dpZoom]);

  const resizeDpRoiFromImagePoint = React.useCallback((imgX: number, imgY: number, shiftKey: boolean = false): boolean => {
    if (isDraggingResizeInner) {
      const dx = Math.abs(imgX - activeRoiCenterCol);
      const dy = Math.abs(imgY - activeRoiCenterRow);
      const newRadius = Math.sqrt(dx ** 2 + dy ** 2);
      setRoiRadiusInner(Math.max(1, Math.min(roiRadius - 1, Math.round(newRadius))));
      requestCompareViLive();
      return true;
    }

    if (isDraggingResize) {
      const dx = Math.abs(imgX - activeRoiCenterCol);
      const dy = Math.abs(imgY - activeRoiCenterRow);
      if (roiMode === "rect") {
        let newW = Math.max(2, Math.round(dx * 2));
        let newH = Math.max(2, Math.round(dy * 2));
        if (shiftKey && resizeAspectRef.current != null) {
          const aspect = resizeAspectRef.current;
          if (newW / newH > aspect) newH = Math.max(2, Math.round(newW / aspect));
          else newW = Math.max(2, Math.round(newH * aspect));
        }
        setRoiWidth(newW);
        setRoiHeight(newH);
      } else {
        const newRadius = roiMode === "square" ? Math.max(dx, dy) : Math.sqrt(dx ** 2 + dy ** 2);
        const minRadius = roiMode === "annular" ? (roiRadiusInner || 0) + 1 : 1;
        const rad = Math.max(minRadius, Math.round(newRadius));
        setLocalRoiRadius(rad);
        sendRoiRadius(rad);
      }
      requestCompareViLive();
      return true;
    }

    return false;
  }, [
    activeRoiCenterCol, activeRoiCenterRow, isDraggingResize, isDraggingResizeInner,
    requestCompareViLive, roiMode, roiRadius, roiRadiusInner, sendRoiRadius, setRoiHeight, setRoiRadiusInner, setRoiWidth
  ]);

  React.useEffect(() => {
    if (!isDraggingResize && !isDraggingResizeInner) return;

    const onMove = (event: MouseEvent | PointerEvent) => {
      const coords = getDpImageCoordsFromClient(event.clientX, event.clientY);
      if (!coords) return;
      if (resizeDpRoiFromImagePoint(coords.imgX, coords.imgY, event.shiftKey)) {
        event.preventDefault();
      }
    };
    const onUp = () => {
      finishDpRoiInteraction();
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      setLocalRoiRadius(null);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp); window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp); window.removeEventListener("pointercancel", onUp);
    };
  }, [finishDpRoiInteraction, getDpImageCoordsFromClient, isDraggingResize, isDraggingResizeInner, resizeDpRoiFromImagePoint]);

  const handleDpMouseDown = (e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
    // Capture the pointer so a fast edge-drag resize keeps receiving move/up
    // events even when the cursor leaves the canvas (#751). Without capture the
    // window listener can miss events and the radius never updates.
    if ("pointerId" in e) {
      try { e.currentTarget.setPointerCapture(e.pointerId); } catch {}
    }
    dpClickStartRef.current = { x: e.clientX, y: e.clientY };
    const coords = getDpImageCoordsFromClient(e.clientX, e.clientY);
    if (!coords) return;
    const { imgX, imgY } = coords;

    // When profile mode is active, use profile interactions only
    if (profileActive) {
      if (profilePoints.length === 2) {
        const p0 = profilePoints[0];
        const p1 = profilePoints[1];
        const hitRadius = 10 / dpZoom;
        const d0 = Math.sqrt((imgX - p0.col) ** 2 + (imgY - p0.row) ** 2);
        const d1 = Math.sqrt((imgX - p1.col) ** 2 + (imgY - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingDpProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingDP(false);
          return;
        }
        if (pointToSegmentDistance(imgX, imgY, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
          setIsDraggingDpProfileLine(true);
          dpProfileDragStartRef.current = {
            row: imgY,
            col: imgX,
            p0: { row: p0.row, col: p0.col },
            p1: { row: p1.row, col: p1.col },
          };
          setIsDraggingDP(false);
          return;
        }
      }
      setIsDraggingDP(false);
      return;
    }

    if (!roiVirtualDetectorActive) {
      setIsDraggingDP(false);
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      setIsHoveringResize(false);
      setIsHoveringResizeInner(false);
      return;
    }

    dpRoiInteractiveRef.current = true;

    // Check if clicking on resize handle (inner first, then outer)
    if (isNearResizeHandleInner(imgX, imgY)) {
      setIsDraggingResizeInner(true);
      return;
    }
    if (isNearResizeHandle(imgX, imgY)) {
      e.preventDefault();
      resizeAspectRef.current = roiMode === "rect" && roiWidth > 0 && roiHeight > 0 ? roiWidth / roiHeight : null;
      setIsDraggingResize(true);
      return;
    }

    setIsDraggingDP(true);
    // If clicking inside the ROI, drag with offset (grab-and-drag)
    if (roiMode !== "off" && roiMode !== "point" && isInsideDpRoi(imgX, imgY)) {
      dpDragOffsetRef.current = { dRow: imgY - activeRoiCenterRow, dCol: imgX - activeRoiCenterCol };
      return;
    }
    // Clicking outside ROI — teleport center to click position
    dpDragOffsetRef.current = { dRow: 0, dCol: 0 };
    setLocalKCol(imgX); setLocalKRow(imgY);
    // Use compound roi_center trait [row, col] - single observer fires in Python
    const newCol = Math.round(Math.max(0, Math.min(detCols - 1, imgX)));
    const newRow = Math.round(Math.max(0, Math.min(detRows - 1, imgY)));
    model.set("roi_active", true);
    writeRoiCenterModel(newRow, newCol);
    requestCompareViLive();
  };

  const handleDpMouseMove = (e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
    const coords = getDpImageCoordsFromClient(e.clientX, e.clientY);
    if (!coords) return;
    const { imgX, imgY } = coords;

    // Fast path: skip cursor readout during any active drag — avoids setCursorInfo re-renders
    const anyDrag = isDraggingDP || isDraggingResize || isDraggingResizeInner
      || draggingDpProfileEndpoint !== null || isDraggingDpProfileLine;

    // Cursor readout: look up raw DP value at pixel position
    if (!anyDrag) {
      const pxCol = Math.floor(imgX);
      const pxRow = Math.floor(imgY);
      if (pxCol >= 0 && pxCol < detCols && pxRow >= 0 && pxRow < detRows && frameBytes) {
        const usesViRoiDp = viRoiMode && viRoiMode !== "off" && viRoiDpBytes && viRoiDpBytes.byteLength > 0;
        const sourceBytes = usesViRoiDp ? viRoiDpBytes : frameBytes;
        const raw = new Float32Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength / 4);
        setCursorInfo({ row: pxRow, col: pxCol, value: raw[pxRow * detCols + pxCol], panel: "DP" });
      } else {
        setCursorInfo(null);
      }
    }

    if (profileActive && profilePoints.length === 2) {
      const p0 = profilePoints[0];
      const p1 = profilePoints[1];
      const hitRadius = 10 / dpZoom;
      const d0 = Math.sqrt((imgX - p0.col) ** 2 + (imgY - p0.row) ** 2);
      const d1 = Math.sqrt((imgX - p1.col) ** 2 + (imgY - p1.row) ** 2);
      if (draggingDpProfileEndpoint !== null) {
        if (!rawDpDataRef.current) return;
        const clampedRow = Math.max(0, Math.min(detRows - 1, imgY));
        const clampedCol = Math.max(0, Math.min(detCols - 1, imgX));
        const next = [
          draggingDpProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
          draggingDpProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
        ];
        setProfileLine(next);
        void sampleLineProfileWebGPU(rawDpDataRef.current, detCols, detRows, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth)
          .then(setProfileData)
          .catch(error => console.error("[Show4DSTEM] WebGPU DP profile failed", error));
        return;
      }
      if (isDraggingDpProfileLine && dpProfileDragStartRef.current) {
        if (!rawDpDataRef.current) return;
        const drag = dpProfileDragStartRef.current;
        let deltaRow = imgY - drag.row;
        let deltaCol = imgX - drag.col;
        const minRow = Math.min(drag.p0.row, drag.p1.row);
        const maxRow = Math.max(drag.p0.row, drag.p1.row);
        const minCol = Math.min(drag.p0.col, drag.p1.col);
        const maxCol = Math.max(drag.p0.col, drag.p1.col);
        deltaRow = Math.max(deltaRow, -minRow);
        deltaRow = Math.min(deltaRow, (detRows - 1) - maxRow);
        deltaCol = Math.max(deltaCol, -minCol);
        deltaCol = Math.min(deltaCol, (detCols - 1) - maxCol);
        const next = [
          { row: drag.p0.row + deltaRow, col: drag.p0.col + deltaCol },
          { row: drag.p1.row + deltaRow, col: drag.p1.col + deltaCol },
        ];
        setProfileLine(next);
        void sampleLineProfileWebGPU(rawDpDataRef.current, detCols, detRows, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth)
          .then(setProfileData)
          .catch(error => console.error("[Show4DSTEM] WebGPU DP profile failed", error));
        return;
      }
      const nextHoveredEndpoint: 0 | 1 | null = d0 <= hitRadius ? 0 : d1 <= hitRadius ? 1 : null;
      const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(imgX, imgY, p0.col, p0.row, p1.col, p1.row) <= hitRadius;
      setHoveredDpProfileEndpoint(nextHoveredEndpoint);
      setIsHoveringDpProfileLine(nextHoverLine);
      return;
    } else {
      if (hoveredDpProfileEndpoint !== null) setHoveredDpProfileEndpoint(null);
      if (isHoveringDpProfileLine) setIsHoveringDpProfileLine(false);
    }

    // Handle inner resize dragging (annular mode)
    if (roiVirtualDetectorActive && resizeDpRoiFromImagePoint(imgX, imgY, e.shiftKey)) {
      return;
    }

    if (!roiVirtualDetectorActive) {
      if (isHoveringResize) setIsHoveringResize(false);
      if (isHoveringResizeInner) setIsHoveringResizeInner(false);
      return;
    }

    // Check hover state for resize handles
    if (!isDraggingDP) {
      setIsHoveringResizeInner(isNearResizeHandleInner(imgX, imgY));
      setIsHoveringResize(isNearResizeHandle(imgX, imgY));
      return;
    }

    const centerCol = imgX - dpDragOffsetRef.current.dCol;
    const centerRow = imgY - dpDragOffsetRef.current.dRow;
    setLocalKCol(centerCol); setLocalKRow(centerRow);
    // rAF-coalesced — sends only the latest roi_center per frame.
    const newCol = Math.round(Math.max(0, Math.min(detCols - 1, centerCol)));
    const newRow = Math.round(Math.max(0, Math.min(detRows - 1, centerRow)));
    queueRoiCenter(newRow, newCol);
    requestCompareViLive();
  };

  const handleDpMouseUp = (e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
    finishDpRoiInteraction();
    if (draggingDpProfileEndpoint !== null || isDraggingDpProfileLine) {
      setDraggingDpProfileEndpoint(null);
      setIsDraggingDpProfileLine(false);
      dpProfileDragStartRef.current = null;
      dpClickStartRef.current = null;
      setIsDraggingDP(false);
      setIsDraggingResize(false);
      setLocalRoiRadius(null);  // revert ring to committed model radius on release
      setIsDraggingResizeInner(false);
      setLocalRoiRadius(null);
      setHoveredDpProfileEndpoint(null);
      setIsHoveringDpProfileLine(false);
      return;
    }

    // Profile click capture
    if (profileActive && dpClickStartRef.current) {
      const dx = e.clientX - dpClickStartRef.current.x;
      const dy = e.clientY - dpClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = dpOverlayRef.current;
        if (canvas && rawDpDataRef.current) {
          const rect = canvas.getBoundingClientRect();
          const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const imgCol = (screenX - dpPanX) / dpZoom;
          const imgRow = (screenY - dpPanY) / dpZoom;
          if (imgCol >= 0 && imgCol < detCols && imgRow >= 0 && imgRow < detRows) {
            const pt = { row: imgRow, col: imgCol };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              setProfileLine([pt]);
              setProfileData(null);
            } else {
              const p0 = profilePoints[0];
              setProfileLine([p0, pt]);
              void sampleLineProfileWebGPU(rawDpDataRef.current, detCols, detRows, p0.row, p0.col, pt.row, pt.col, profileWidth)
                .then(setProfileData)
                .catch(error => console.error("[Show4DSTEM] WebGPU DP profile failed", error));
            }
          }
        }
      }
    }
    dpClickStartRef.current = null;
    setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false);
    setLocalRoiRadius(null);
    setDraggingDpProfileEndpoint(null);
    setIsDraggingDpProfileLine(false);
    setHoveredDpProfileEndpoint(null);
    setIsHoveringDpProfileLine(false);
    dpProfileDragStartRef.current = null;
  };
  const handleDpMouseLeave = () => {
    dpClickStartRef.current = null;
    finishDpRoiInteraction();
    setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false);
    setLocalRoiRadius(null);
    setDraggingDpProfileEndpoint(null);
    setIsDraggingDpProfileLine(false);
    setHoveredDpProfileEndpoint(null);
    setIsHoveringDpProfileLine(false);
    dpProfileDragStartRef.current = null;
    setIsHoveringResize(false); setIsHoveringResizeInner(false);
    setCursorInfo(prev => prev?.panel === "DP" ? null : prev);
  };
  const handleDpDoubleClick = () => {
    dpViewRef.current.zoom = 1;
    dpViewRef.current.panX = 0;
    dpViewRef.current.panY = 0;
    setDpZoom(1);
    setDpPanX(0);
    setDpPanY(0);
  };

  const handleViMouseDown = (e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
    // Capture the pointer so a touch/mouse probe-drag keeps receiving move/up
    // events even when the finger leaves the small canvas. Needed for mobile
    // parity: touchscreens deliver these as pointer events (mirrors DP #751).
    if ("pointerId" in e) {
      try { e.currentTarget.setPointerCapture(e.pointerId); } catch {}
    }
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;

    // VI Profile mode - click to set points
    if (viProfileActive) {
      viClickStartRef.current = { x: screenX, y: screenY };
      if (viProfilePoints.length === 2) {
        const p0 = viProfilePoints[0];
        const p1 = viProfilePoints[1];
        const hitRadius = 10 / viZoom;
        const d0 = Math.sqrt((imgY - p0.col) ** 2 + (imgX - p0.row) ** 2);
        const d1 = Math.sqrt((imgY - p1.col) ** 2 + (imgX - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingViProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingVI(false);
          return;
        }
        if (pointToSegmentDistance(imgY, imgX, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
          setIsDraggingViProfileLine(true);
          viProfileDragStartRef.current = {
            row: imgX,
            col: imgY,
            p0: { row: p0.row, col: p0.col },
            p1: { row: p1.row, col: p1.col },
          };
          setIsDraggingVI(false);
          return;
        }
      }
      return;
    }

    // Check if VI ROI mode is active - same logic as DP
    if (viRoiMode && viRoiMode !== "off") {
      // Check if clicking on resize handle
      if (isNearViRoiResizeHandle(imgX, imgY)) {
        setIsDraggingViRoiResize(true);
        return;
      }

      // Grab-and-drag if clicking inside VI ROI, otherwise teleport
      setIsDraggingViRoi(true);
      if (isInsideViRoi(imgX, imgY)) {
        viRoiDragOffsetRef.current = { dRow: imgX - localViRoiCenterRow, dCol: imgY - localViRoiCenterCol };
      } else {
        viRoiDragOffsetRef.current = { dRow: 0, dCol: 0 };
        setLocalViRoiCenterRow(imgX);
        setLocalViRoiCenterCol(imgY);
        setViRoiCenterRow(Math.round(Math.max(0, Math.min(shapeRows - 1, imgX))));
        setViRoiCenterCol(Math.round(Math.max(0, Math.min(shapeCols - 1, imgY))));
      }
      return;
    }

    // Regular position selection (when ROI is off)
    setIsDraggingVI(true);
    // Snap to the integer scan index so the crosshair marks the exact pixel the
    // CBED is sampled from (not the fractional cursor position).
    const newX = Math.round(Math.max(0, Math.min(shapeRows - 1, imgX)));
    const newY = Math.round(Math.max(0, Math.min(shapeCols - 1, imgY)));
    updateScanPosition(newX, newY);
  };

  const handleViMouseMove = (e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;

    // Fast path: skip cursor readout during any active drag — avoids setCursorInfo re-renders
    const anyViDrag = isDraggingVI || isDraggingViRoi || isDraggingViRoiResize
      || draggingViProfileEndpoint !== null || isDraggingViProfileLine;

    // Cursor readout: look up raw VI value at pixel position
    // imgX = row, imgY = col (swapped coordinate convention)
    if (!anyViDrag) {
      const pxRow = Math.floor(imgX);
      const pxCol = Math.floor(imgY);
      if (pxRow >= 0 && pxRow < shapeRows && pxCol >= 0 && pxCol < shapeCols && rawVirtualImageRef.current) {
        const raw = rawVirtualImageRef.current;
        setCursorInfo({ row: pxRow, col: pxCol, value: raw[pxRow * shapeCols + pxCol], panel: "VI" });
      } else {
        setCursorInfo(prev => prev?.panel === "VI" ? null : prev);
      }
    }

    if (viProfileActive && viProfilePoints.length === 2) {
      const p0 = viProfilePoints[0];
      const p1 = viProfilePoints[1];
      const hitRadius = 10 / viZoom;
      const d0 = Math.sqrt((imgY - p0.col) ** 2 + (imgX - p0.row) ** 2);
      const d1 = Math.sqrt((imgY - p1.col) ** 2 + (imgX - p1.row) ** 2);
      if (draggingViProfileEndpoint !== null) {
        const clampedRow = Math.max(0, Math.min(shapeRows - 1, imgX));
        const clampedCol = Math.max(0, Math.min(shapeCols - 1, imgY));
        const next = [
          draggingViProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : viProfilePoints[0],
          draggingViProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : viProfilePoints[1],
        ];
        setViProfilePoints(next);
        return;
      }
      if (isDraggingViProfileLine && viProfileDragStartRef.current) {
        const drag = viProfileDragStartRef.current;
        let deltaRow = imgX - drag.row;
        let deltaCol = imgY - drag.col;
        const minRow = Math.min(drag.p0.row, drag.p1.row);
        const maxRow = Math.max(drag.p0.row, drag.p1.row);
        const minCol = Math.min(drag.p0.col, drag.p1.col);
        const maxCol = Math.max(drag.p0.col, drag.p1.col);
        deltaRow = Math.max(deltaRow, -minRow);
        deltaRow = Math.min(deltaRow, (shapeRows - 1) - maxRow);
        deltaCol = Math.max(deltaCol, -minCol);
        deltaCol = Math.min(deltaCol, (shapeCols - 1) - maxCol);
        const next = [
          { row: drag.p0.row + deltaRow, col: drag.p0.col + deltaCol },
          { row: drag.p1.row + deltaRow, col: drag.p1.col + deltaCol },
        ];
        setViProfilePoints(next);
        return;
      }
      const nextHoveredEndpoint: 0 | 1 | null = d0 <= hitRadius ? 0 : d1 <= hitRadius ? 1 : null;
      const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(imgY, imgX, p0.col, p0.row, p1.col, p1.row) <= hitRadius;
      setHoveredViProfileEndpoint(nextHoveredEndpoint);
      setIsHoveringViProfileLine(nextHoverLine);
      return;
    } else {
      if (hoveredViProfileEndpoint !== null) setHoveredViProfileEndpoint(null);
      if (isHoveringViProfileLine) setIsHoveringViProfileLine(false);
    }

    // Handle VI ROI resize dragging (same pattern as DP)
    if (isDraggingViRoiResize) {
      const dx = Math.abs(imgX - localViRoiCenterRow);
      const dy = Math.abs(imgY - localViRoiCenterCol);
      if (viRoiMode === "rect") {
        setViRoiWidth(Math.max(2, Math.round(dy * 2)));
        setViRoiHeight(Math.max(2, Math.round(dx * 2)));
      } else if (viRoiMode === "square") {
        const newHalfSize = Math.max(dx, dy);
        setViRoiRadius(Math.max(1, Math.round(newHalfSize)));
      } else {
        // circle
        const newRadius = Math.sqrt(dx ** 2 + dy ** 2);
        setViRoiRadius(Math.max(1, Math.round(newRadius)));
      }
      return;
    }

    // Check hover state for resize handles (same as DP)
    if (!isDraggingViRoi) {
      setIsHoveringViRoiResize(isNearViRoiResizeHandle(imgX, imgY));
      if (viRoiMode && viRoiMode !== "off") return;  // Don't update position when ROI active
    }

    // Handle VI ROI center dragging (same as DP — with offset)
    if (isDraggingViRoi) {
      const centerRow = imgX - viRoiDragOffsetRef.current.dRow;
      const centerCol = imgY - viRoiDragOffsetRef.current.dCol;
      setLocalViRoiCenterRow(centerRow);
      setLocalViRoiCenterCol(centerCol);
      // Compound trait update — single observer fires Python-side; reduced DP is
      // never computed against split-trait state (old col + new row, or vice versa).
      const newViX = Math.round(Math.max(0, Math.min(shapeRows - 1, centerRow)));
      const newViY = Math.round(Math.max(0, Math.min(shapeCols - 1, centerCol)));
      model.set("vi_roi_center", [newViX, newViY]);
      model.save_changes();
      return;
    }

    // Handle regular position dragging (when ROI is off)
    if (!isDraggingVI) return;
    // Snap to the integer scan index so the crosshair tracks discrete sampled
    // positions, matching the CBED actually shown.
    const newX = Math.round(Math.max(0, Math.min(shapeRows - 1, imgX)));
    const newY = Math.round(Math.max(0, Math.min(shapeCols - 1, imgY)));
    updateScanPosition(newX, newY);
  };

  const handleViMouseUp = (e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
    flushScanPosition();
    if (draggingViProfileEndpoint !== null || isDraggingViProfileLine) {
      setDraggingViProfileEndpoint(null);
      setIsDraggingViProfileLine(false);
      viProfileDragStartRef.current = null;
      viClickStartRef.current = null;
      setIsDraggingVI(false);
      setIsDraggingViRoi(false);
      setIsDraggingViRoiResize(false);
      setHoveredViProfileEndpoint(null);
      setIsHoveringViProfileLine(false);
      return;
    }

    // VI Profile mode - complete point selection
    if (viProfileActive && viClickStartRef.current) {
      const canvas = virtualOverlayRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const endX = (e.clientX - rect.left) * (canvas.width / rect.width);
        const endY = (e.clientY - rect.top) * (canvas.height / rect.height);
        const dx = endX - viClickStartRef.current.x;
        const dy = endY - viClickStartRef.current.y;
        const wasDrag = Math.sqrt(dx * dx + dy * dy) > 3;

        if (!wasDrag) {
          // Click to add point
          const imgX = (endY - viPanY) / viZoom;
          const imgY = (endX - viPanX) / viZoom;
          const pt = { row: Math.round(Math.max(0, Math.min(shapeRows - 1, imgX))), col: Math.round(Math.max(0, Math.min(shapeCols - 1, imgY))) };
          if (viProfilePoints.length < 2) {
            setViProfilePoints([...viProfilePoints, pt]);
          } else {
            setViProfilePoints([pt]);
          }
        }
      }
      viClickStartRef.current = null;
    }

    setDraggingViProfileEndpoint(null);
    setIsDraggingViProfileLine(false);
    setHoveredViProfileEndpoint(null);
    setIsHoveringViProfileLine(false);
    viProfileDragStartRef.current = null;
    setIsDraggingVI(false);
    setIsDraggingViRoi(false);
    setIsDraggingViRoiResize(false);
  };
  const handleViMouseLeave = () => {
    flushScanPosition();
    viClickStartRef.current = null;
    setDraggingViProfileEndpoint(null);
    setIsDraggingViProfileLine(false);
    setHoveredViProfileEndpoint(null);
    setIsHoveringViProfileLine(false);
    viProfileDragStartRef.current = null;
    setIsDraggingVI(false);
    setIsDraggingViRoi(false);
    setIsDraggingViRoiResize(false);
    setIsHoveringViRoiResize(false);
    setCursorInfo(prev => prev?.panel === "VI" ? null : prev);
  };
  const handleViDoubleClick = () => {
    viViewRef.current.zoom = 1;
    viViewRef.current.panX = 0;
    viViewRef.current.panY = 0;
    setViZoom(1);
    setViPanX(0);
    setViPanY(0);
  };
  const handleFftDoubleClick = () => {
    fftViewRef.current.zoom = 1;
    fftViewRef.current.panX = 0;
    fftViewRef.current.panY = 0;
    setFftZoom(1);
    setFftPanX(0);
    setFftPanY(0);
    setFftClickInfo(null);
  };

  const touchDistance = (a: React.Touch, b: React.Touch): number => {
    return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
  };

  const touchMidpoint = (a: React.Touch, b: React.Touch): { x: number; y: number } => {
    return { x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 };
  };

  const canvasPointFromClient = (
    canvas: HTMLCanvasElement,
    clientX: number,
    clientY: number,
  ): { x: number; y: number } => {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return { x: 0, y: 0 };
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  const getTouchPanelRefs = (kind: TouchPanelKind) => {
    if (kind === "dp") {
      return {
        canvasRef: dpOverlayRef,
        viewRef: dpViewRef,
        setZoom: setDpZoom,
        setPanX: setDpPanX,
        setPanY: setDpPanY,
        reset: handleDpDoubleClick,
      };
    }
    if (kind === "vi") {
      return {
        canvasRef: virtualOverlayRef,
        viewRef: viViewRef,
        setZoom: setViZoom,
        setPanX: setViPanX,
        setPanY: setViPanY,
        reset: handleViDoubleClick,
      };
    }
    return {
      canvasRef: fftOverlayRef,
      viewRef: fftViewRef,
      setZoom: setFftZoom,
      setPanX: setFftPanX,
      setPanY: setFftPanY,
      reset: handleFftDoubleClick,
    };
  };

  const setTouchView = (
    viewRef: React.RefObject<{ zoom: number; panX: number; panY: number; raf: number }>,
    setZoom: React.Dispatch<React.SetStateAction<number>>,
    setPanX: React.Dispatch<React.SetStateAction<number>>,
    setPanY: React.Dispatch<React.SetStateAction<number>>,
    zoom: number,
    panX: number,
    panY: number,
  ) => {
    const view = viewRef.current;
    view.zoom = zoom;
    view.panX = panX;
    view.panY = panY;
    setZoom(zoom);
    setPanX(panX);
    setPanY(panY);
  };

  const handlePanelTouchStart = (kind: TouchPanelKind) => (e: React.TouchEvent<HTMLCanvasElement>) => {
    const refs = getTouchPanelRefs(kind);
    const canvas = refs.canvasRef.current;
    if (!canvas) return;

    if (e.touches.length === 1) {
      const now = window.performance.now();
      const previousTap = lastTapRef.current;
      lastTapRef.current = { kind, time: now };
      if (previousTap && previousTap.kind === kind && now - previousTap.time < 320) {
        e.preventDefault();
        refs.reset();
        touchTransformRef.current = null;
        return;
      }

      if (kind !== "fft" && refs.viewRef.current.zoom <= 1) {
        touchTransformRef.current = null;
        return;
      }

      const touch = e.touches[0];
      touchTransformRef.current = {
        kind,
        mode: "pan",
        startX: touch.clientX,
        startY: touch.clientY,
        startDistance: 0,
        startMidX: touch.clientX,
        startMidY: touch.clientY,
        startZoom: refs.viewRef.current.zoom,
        startPanX: refs.viewRef.current.panX,
        startPanY: refs.viewRef.current.panY,
      };
      e.preventDefault();
      return;
    }

    if (e.touches.length >= 2) {
      const first = e.touches[0];
      const second = e.touches[1];
      const midpoint = touchMidpoint(first, second);
      touchTransformRef.current = {
        kind,
        mode: "pinch",
        startX: midpoint.x,
        startY: midpoint.y,
        startDistance: touchDistance(first, second),
        startMidX: midpoint.x,
        startMidY: midpoint.y,
        startZoom: refs.viewRef.current.zoom,
        startPanX: refs.viewRef.current.panX,
        startPanY: refs.viewRef.current.panY,
      };
      e.preventDefault();
    }
  };

  const handlePanelTouchMove = (kind: TouchPanelKind) => (e: React.TouchEvent<HTMLCanvasElement>) => {
    const state = touchTransformRef.current;
    if (!state || state.kind !== kind) return;
    const refs = getTouchPanelRefs(kind);
    const canvas = refs.canvasRef.current;
    if (!canvas) return;

    if (state.mode === "pan" && e.touches.length === 1) {
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      const dx = (touch.clientX - state.startX) * (canvas.width / rect.width);
      const dy = (touch.clientY - state.startY) * (canvas.height / rect.height);
      setTouchView(
        refs.viewRef,
        refs.setZoom,
        refs.setPanX,
        refs.setPanY,
        state.startZoom,
        state.startPanX + dx,
        state.startPanY + dy,
      );
      e.preventDefault();
      return;
    }

    if (state.mode === "pinch" && e.touches.length >= 2) {
      const first = e.touches[0];
      const second = e.touches[1];
      const midpoint = touchMidpoint(first, second);
      const startCanvasPoint = canvasPointFromClient(canvas, state.startMidX, state.startMidY);
      const currentCanvasPoint = canvasPointFromClient(canvas, midpoint.x, midpoint.y);
      const ratio = state.startDistance > 0 ? touchDistance(first, second) / state.startDistance : 1;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, state.startZoom * ratio));
      const imageX = (startCanvasPoint.x - state.startPanX) / state.startZoom;
      const imageY = (startCanvasPoint.y - state.startPanY) / state.startZoom;
      setTouchView(
        refs.viewRef,
        refs.setZoom,
        refs.setPanX,
        refs.setPanY,
        newZoom,
        currentCanvasPoint.x - imageX * newZoom,
        currentCanvasPoint.y - imageY * newZoom,
      );
      e.preventDefault();
    }
  };

  const handlePanelTouchEnd = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (e.touches.length === 0) {
      touchTransformRef.current = null;
    }
  };

  // FFT drag-to-pan handlers
  const handleFftMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsDraggingFFT(true);
    setFftDragStart({ x: e.clientX, y: e.clientY, panX: fftPanX, panY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDraggingFFT || !fftDragStart) return;
    const canvas = fftOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const dx = (e.clientX - fftDragStart.x) * scaleX;
    const dy = (e.clientY - fftDragStart.y) * scaleY;
    setFftPanX(fftDragStart.panX + dx);
    setFftPanY(fftDragStart.panY + dy);
  };

  const handleFftMouseUp = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        // Convert screen coords to FFT image coords
        const canvas = fftOverlayRef.current;
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const scaleX = canvas.width / rect.width;
          const scaleY = canvas.height / rect.height;
          const canvasX = (e.clientX - rect.left) * scaleX;
          const canvasY = (e.clientY - rect.top) * scaleY;
          const fftW = fftCropDims?.fftWidth ?? shapeCols;
          const fftH = fftCropDims?.fftHeight ?? shapeRows;
          // Reverse the render transform: canvas coords -> image coords.
          // Render: translate(panX, panY); scale(zoom); drawImage(offscreen, 0,0,fftW,fftH, 0,0,canvasW,canvasH)
          // So: canvasX = panX + zoom * (imgCol * canvasW / fftW)  →  imgCol = (canvasX - panX) / zoom * fftW / canvasW
          let imgCol = ((canvasX - fftPanX) / fftZoom) * (fftW / canvas.width);
          let imgRow = ((canvasY - fftPanY) / fftZoom) * (fftH / canvas.height);
          // Bounds check
          if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
            // Snap to nearest peak in FFT magnitude
            if (fftMagCacheRef.current) {
              let snapped: { row: number; col: number };
              try {
                snapped = await findFFTPeakWebGPU(fftMagCacheRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
              } catch (error) {
                console.error("[Show4DSTEM] WebGPU FFT peak refinement failed", error);
                return;
              }
              imgCol = snapped.col;
              imgRow = snapped.row;
            }
            const halfW = Math.floor(fftW / 2);
            const halfH = Math.floor(fftH / 2);
            const dcol = imgCol - halfW;
            const drow = imgRow - halfH;
            const distPx = Math.sqrt(dcol * dcol + drow * drow);
            if (distPx < 1) {
              setFftClickInfo(null); // Clicked on DC center
            } else {
              let spatialFreq: number | null = null;
              let dSpacing: number | null = null;
              if (pixelSize > 0) {
                const paddedW = nextPow2(fftW);
                const paddedH = nextPow2(fftH);
                ({ spatialFrequency: spatialFreq, dSpacing } = reciprocalCoordinatesFromShiftedOffset(
                  Math.round(imgRow) - halfH,
                  Math.round(imgCol) - halfW,
                  paddedH,
                  paddedW,
                  pixelSize,
                  pixelSize,
                ));
              }
              setFftClickInfo({ row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
            }
          }
        }
      }
      fftClickStartRef.current = null;
    }
    setIsDraggingFFT(false);
    setFftDragStart(null);
  };
  const handleFftMouseLeave = () => { fftClickStartRef.current = null; setIsDraggingFFT(false); setFftDragStart(null); };

  // ── Canvas resize handlers ──
  const handleCanvasResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizingCanvas(true);
    setResizeCanvasStart({ x: e.clientX, y: e.clientY, size: canvasSize });
  };

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    let rafId = 0;
    let latestSize = resizeCanvasStart ? resizeCanvasStart.size : canvasSize;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeCanvasStart) return;
      const delta = Math.max(e.clientX - resizeCanvasStart.x, e.clientY - resizeCanvasStart.y);
      const minCanvasSize = MIN_CANVAS_SIZE;
      latestSize = Math.max(minCanvasSize, resizeCanvasStart.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setCanvasSize(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      cancelAnimationFrame(rafId);
      setCanvasSize(latestSize);
      setPanelWidthPx(Math.round(latestSize));
      setIsResizingCanvas(false);
      setResizeCanvasStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingCanvas, resizeCanvasStart, panelWidthPx, setPanelWidthPx]);

  const handleCompareGridResizeStart = (e: React.PointerEvent<HTMLElement>, panelScale = 1) => {
    e.stopPropagation();
    e.preventDefault();
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch {}
    compareGridResizeCleanupRef.current?.();
    const startX = e.clientX;
    const startY = e.clientY;
    const startWidth = compareGridWidth;
    const resizeScale = Math.max(1, Number.isFinite(panelScale) ? panelScale : 1);
    let rafId = 0;
    let latestWidth = startWidth;
    const handlePointerMove = (e: PointerEvent) => {
      const delta = Math.max(e.clientX - startX, e.clientY - startY);
      latestWidth = Math.max(MIN_COMPARE_GRID_WIDTH, startWidth + delta * resizeScale);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setCompareGridPreviewWidth(latestWidth);
        });
      }
      e.preventDefault();
    };
    const handlePointerUp = () => {
      cancelAnimationFrame(rafId);
      setCompareGridWidthPx(Math.round(latestWidth));
      setCompareGridPreviewWidth(null);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
      compareGridResizeCleanupRef.current = null;
    };
    compareGridResizeCleanupRef.current = handlePointerUp;
    window.addEventListener("pointermove", handlePointerMove, { passive: false });
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);
  };

  React.useEffect(() => {
    return () => {
      compareGridResizeCleanupRef.current?.();
    };
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  // Theme-aware select style
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
  const statsBarSx = {
    mt: `${SPACING.XS}px`,
    px: 1,
    py: 0.5,
    height: 28,
    minHeight: 28,
    bgcolor: themeColors.bgAlt,
    display: "flex",
    columnGap: 1.25,
    alignItems: "center",
    flexWrap: "nowrap",
    maxWidth: "100%",
    overflow: "hidden",
    boxSizing: "border-box",
    "@media (max-width: 700px)": {
      mt: 0,
      px: 0.5,
      py: 0.25,
      height: 24,
      minHeight: 24,
      columnGap: "6px",
    },
  };
  const statsTextSx = {
    fontSize: 11,
    lineHeight: 1.4,
    color: themeColors.textMuted,
    whiteSpace: "nowrap",
    flexShrink: 0,
  };
  const viSourceButtonSx = (source: string, active = false) => {
    const color = viSourceDisplayColor(source, themeInfo.theme) ?? themeColors.textMuted;
    return {
      ...statsTextSx,
      color,
      fontWeight: active ? 800 : 700,
      opacity: active ? 1 : 0.9,
      cursor: "pointer",
      display: "inline-flex",
      alignItems: "center",
      whiteSpace: "nowrap",
      "&:hover": { color, opacity: 1, textDecoration: "underline" },
    };
  };
  const statsValueSx = { color: themeColors.accent };
  const memoryWarningSx = {
    mb: `${SPACING.SM}px`,
    px: 1,
    py: 0.75,
    border: `1px solid ${themeColors.border}`,
    borderLeft: `3px solid ${themeColors.accent}`,
    bgcolor: themeInfo.theme === "dark" ? "rgba(245, 158, 11, 0.14)" : "rgba(245, 158, 11, 0.12)",
    color: themeColors.text,
    fontSize: 11,
    lineHeight: 1.35,
    maxWidth: "100%",
    boxSizing: "border-box",
    "@media (max-width: 700px)": {
      mb: "2px",
      px: 0.75,
      py: 0.5,
      fontSize: 10,
    },
  };

  const keyboardShortcutItems: [string, string][] = [
    ["↑ / ↓", "Move scan row"],
    ["← / →", "Move scan col"],
    ["Shift+Arrows", "Move ×10"],
    ...(nFrames > 1 ? [["[ / ]", `Prev / next ${frameDimLabel.toLowerCase()}`] as [string, string]] : []),
    ["Space", "Play / pause"],
    ["R", "Reset all zoom/pan"],
    ["Esc", "Release keyboard focus"],
    ["Scroll", "Zoom"],
    ["Dbl-click", "Reset view"],
  ];
  const squarePanelWidth = `min(${canvasSize}px, 100%)`;
  const viPanelWidth = compareMode ? `min(${compareGridWidth}px, 100%)` : `min(${viCanvasWidth}px, 100%)`;
  const mobileTightLayout = nFrames > 1;
  const mobilePanelSx = {
    "@media (max-width: 700px)": {
      width: "100%",
      maxWidth: "100%",
      minWidth: 0,
    },
  };
  const mobileImageBoxSx = {
    "@media (max-width: 700px)": {
      maxWidth: "100%",
    },
  };
  const panelHeaderSx = {
    mb: `${SPACING.XS}px`,
    minHeight: 28,
    height: "auto",
    flexWrap: "wrap",
    gap: `${SPACING.XS}px`,
    "@media (max-width: 700px)": {
      mb: mobileTightLayout ? 0 : "1px",
      minHeight: mobileTightLayout ? 18 : 22,
      rowGap: "1px",
    },
  };
  const hideBetweenPanelsOnMobileSx = mobileTightLayout
    ? { "@media (max-width: 700px)": { display: "none" } }
    : {};
  const mainStackDirection = compareMode && compareLayout === "top" ? "column" : "row";
  const optionLabel = (value: string | undefined | null): string => {
    if (!value) return "";
    return value.charAt(0).toUpperCase() + value.slice(1);
  };
  const mobileOptionToggleSx = {
    ...compactButton,
    display: "none",
    mt: `${SPACING.XS}px`,
    width: "100%",
    justifyContent: "space-between",
    border: `1px solid ${themeColors.border}`,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    textTransform: "none",
    "@media (max-width: 700px)": {
      display: "flex",
      mt: "2px",
      minHeight: 22,
      px: 0.5,
      py: 0,
      fontSize: 10,
      lineHeight: "18px",
      "& .MuiButton-endIcon": { ml: 0.25, mr: 0 },
      "& .MuiSvgIcon-root": { fontSize: 16 },
    },
  };
  const mobileOptionSummarySx = {
    ml: 1,
    color: themeColors.textMuted,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    minWidth: 0,
    flex: 1,
    textAlign: "right",
  };
  const mobileOptionsPanelSx = (open: boolean) => ({
    mt: `${SPACING.SM}px`,
    display: "grid",
    gridTemplateRows: "1fr",
    opacity: 1,
    transition: "grid-template-rows 180ms ease, opacity 160ms ease",
    "@media (max-width: 700px)": {
      mt: open ? "2px" : 0,
      gridTemplateRows: open ? "1fr" : "0fr",
      opacity: open ? 1 : 0,
      pointerEvents: open ? "auto" : "none",
    },
  });
  const mobileOptionsContentSx = {
    minHeight: 0,
    overflow: "hidden",
    display: "flex",
    gap: `${SPACING.SM}px`,
    width: "100%",
    maxWidth: "100%",
    boxSizing: "border-box",
    flexWrap: "wrap",
  };
  const dpOptionSummary = `${optionLabel(roiMode)}${roiMode === "annular" ? ` ${Math.round(roiRadiusInner)}-${Math.round(roiRadius)}px` : roiMode !== "point" ? ` ${Math.round(roiRadius)}px` : ""} | ${optionLabel(dpColormap)} | ${dpScaleMode === "log" ? "Log" : "Lin"}`;
  const viOptionSummary = `${viSourceLabel(activeViSource)} | ${viRoiMode === "off" ? "ROI off" : `${optionLabel(viRoiMode)} ${Math.round(viRoiRadius || 5)}px`} | ${optionLabel(viColormap)} | ${viScaleMode === "log" ? "Log" : "Lin"}`;
  const fftOptionSummary = `${fftScaleMode === "log" ? "Log" : "Lin"} | ${optionLabel(fftColormap)}${fftAuto ? " | Auto" : ""}`;
  const activeSsbBfSubsample = Math.max(0.01, Math.min(1, Number(ssbComputeBfSubsample ?? 1)));
  const ssbBfCountText = Number(ssbComputeBfPixels || 0) > 0
    ? `${Math.max(0, Math.round(Number(ssbComputeBfSelectedPixels || 0)))} / ${Math.max(0, Math.round(Number(ssbComputeBfPixels || 0)))} BF px`
    : "BF count appears after first run";
  const hasSsbCalibrationDownload = String(ssbComputeCalibrationJson || "").trim().length > 0;
  const ssbProgressText = String(ssbComputeStatus || "").trim()
    || (ssbComputeBusy ? "Running SSB..." : "");
  const ssbDisplayStatus = ssbProgressText;
  const ssbStatusIsFailure = ssbDisplayStatus.startsWith("SSB failed");
  const hasSsbProductMap = viProductSourceOptions.includes("SSB");
  const showSsbCalibrationPanel = controlsVisible
    && ssbComputeEnabled
    && hasSsbProductMap
    && hasSsbCalibrationDownload
    && !compareMode;
  const ssbC10Limit = Math.max(100, Math.ceil(Math.abs(Number(ssbComputeC10Nm ?? 0)) * 2 / 25) * 25);
  const ssbC12Limit = Math.max(100, Math.ceil(Math.abs(Number(ssbComputeC12Nm ?? 0)) * 2 / 25) * 25);
  const ssbTuneSliderSx = {
    ...sliderStyles.small,
    minWidth: 115,
    flex: 1,
    mx: 0.75,
  };
  const ssbCalSummary = `C10 ${Number(ssbComputeC10Nm ?? 0).toFixed(0)} nm | C12 ${Number(ssbComputeC12Nm ?? 0).toFixed(0)} nm | φ12 ${Number(ssbComputePhi12Deg ?? 0).toFixed(0)}° | rot ${Number(ssbComputeRotationDeg ?? 0).toFixed(1)}°`;
  const ssbCalToggleSx = {
    ...compactButton,
    display: "flex",
    width: "100%",
    justifyContent: "space-between",
    border: `1px solid ${themeColors.border}`,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    textTransform: "none",
    minHeight: 22,
    px: 0.75,
    py: 0,
    fontSize: 10,
    lineHeight: "20px",
    "& .MuiButton-endIcon": { ml: 0.25, mr: 0 },
    "& .MuiSvgIcon-root": { fontSize: 16 },
  };
  const ssbTuneCommit = React.useCallback((values?: {
    c10Nm?: number;
    c12Nm?: number;
    phi12Deg?: number;
    rotationDeg?: number;
  }) => {
    requestSsbManualReconstruct({
      c10Nm: Number(values?.c10Nm ?? ssbComputeC10Nm ?? 0),
      c12Nm: Number(values?.c12Nm ?? ssbComputeC12Nm ?? 0),
      phi12Deg: Number(values?.phi12Deg ?? ssbComputePhi12Deg ?? 0),
      rotationDeg: Number(values?.rotationDeg ?? ssbComputeRotationDeg ?? 0),
    });
  }, [
    requestSsbManualReconstruct,
    ssbComputeC10Nm,
    ssbComputeC12Nm,
    ssbComputePhi12Deg,
    ssbComputeRotationDeg,
  ]);
  const ssbTuneDebounceRef = React.useRef<number | null>(null);
  const scheduleSsbTuneCommit = React.useCallback((values?: {
    c10Nm?: number;
    c12Nm?: number;
    phi12Deg?: number;
    rotationDeg?: number;
  }) => {
    if (ssbComputeBusy) return;
    if (ssbTuneDebounceRef.current !== null) {
      window.clearTimeout(ssbTuneDebounceRef.current);
    }
    ssbTuneDebounceRef.current = window.setTimeout(() => {
      ssbTuneDebounceRef.current = null;
      ssbTuneCommit(values);
    }, 350);
  }, [ssbComputeBusy, ssbTuneCommit]);
  const commitSsbTuneNow = React.useCallback((values?: {
    c10Nm?: number;
    c12Nm?: number;
    phi12Deg?: number;
    rotationDeg?: number;
  }) => {
    if (ssbTuneDebounceRef.current !== null) {
      window.clearTimeout(ssbTuneDebounceRef.current);
      ssbTuneDebounceRef.current = null;
    }
    ssbTuneCommit(values);
  }, [ssbTuneCommit]);
  React.useEffect(() => () => {
    if (ssbTuneDebounceRef.current !== null) {
      window.clearTimeout(ssbTuneDebounceRef.current);
      ssbTuneDebounceRef.current = null;
    }
  }, []);
  const currentViSource = normaliseViSource(model.get("vi_source"));
  const viGpuVisible = Boolean(
    viGpuRetainedReady
    &&
    viGpuVersion >= 0
    && !compareMode
    && viGpuImageRef.current
    && (
      activeViSource === viGpuImageRef.current.source
      || currentViSource === viGpuImageRef.current.source
    ),
  );
  const getActiveViCanvas = React.useCallback((): HTMLCanvasElement | null => {
    return virtualCanvasRef.current;
  }, []);
  const panelLoadingOverlaySx = React.useMemo(() => ({
    position: "absolute",
    inset: 0,
    zIndex: 8,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    bgcolor: themeInfo.theme === "dark" ? "rgba(15,18,22,0.88)" : "rgba(247,249,252,0.92)",
    color: themeColors.textMuted,
    pointerEvents: "none",
    backdropFilter: "blur(1px)",
  }), [themeColors.textMuted, themeInfo.theme]);
  const panelLoadingTextSx = React.useMemo(() => ({
    px: 1,
    py: 0.5,
    borderRadius: "4px",
    bgcolor: themeInfo.theme === "dark" ? "rgba(0,0,0,0.35)" : "rgba(255,255,255,0.78)",
    color: themeColors.textMuted,
    fontFamily: SHOW4DSTEM_UI_FONT,
    fontSize: 12,
    fontWeight: 600,
    letterSpacing: 0,
  }), [themeColors.textMuted, themeInfo.theme]);
  const renderPanelLoadingOverlay = React.useCallback((label: string, detail = "") => (
    <Box
      data-show4dstem-panel-loading="true"
      data-quantem-load-error={/\bfailed\b/i.test(label) ? "true" : undefined}
      sx={panelLoadingOverlaySx}
    >
      <Typography sx={panelLoadingTextSx}>
        {label}
        {detail && <Box component="span" sx={{ display: "block", mt: 0.25, fontSize: 10, fontWeight: 500, maxWidth: 220, overflowWrap: "anywhere" }}>{detail}</Box>}
      </Typography>
    </Box>
  ), [panelLoadingOverlaySx, panelLoadingTextSx]);
  const dpPanelReady = Boolean(displayedDpBytes && displayedDpBytes.byteLength >= detRows * detCols * 4);
  const viPanelReady = Boolean(
    viGpuVisible
    || (displayedVirtualImageBytes && displayedVirtualImageBytes.byteLength >= shapeRows * shapeCols * 4),
  );
  const dpPanelLoading = offlineBackendLoading && !dpPanelReady;
  const viPanelLoading = offlineBackendLoading && !compareMode && !viPanelReady;
  const fftPanelLoading = offlineBackendLoading && effectiveShowFft;
  const offlineStatusText = offlineBackendError || offlineBackendStatus;
  const offlineStatusIsError = Boolean(offlineBackendError);
  const offlineStatusIsReady = !offlineStatusIsError && /\bready\b/i.test(offlineStatusText);
  const showOfflineStatus = offline && Boolean(offlineStatusText) && !offlineStatusIsReady;
  const showLocalH5GrantBanner = offline && h5SourceAvailable && requireLocalH5Files && !h5LocalFilesGranted;

  return (
    <Box
      ref={rootRef}
      className="show4dstem-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
      sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, outline: "none", borderRadius: "2px", width: "100%", maxWidth: "100%", boxSizing: "border-box", "@media (max-width: 700px)": { p: 0, overflowX: "hidden", ".jp-OutputArea-output &, .jp-OutputArea-child &": { width: "calc(100vw - 96px)", maxWidth: "calc(100vw - 96px)" } } }}
    >
      <FolderWatchBadge
        state={folderWatchState}
        detail={folderWatchDetail}
        live={folderWatchLive}
      />
      <input
        ref={h5LocalInputRef}
        type="file"
        multiple
        accept=".h5,.hdf5"
        onChange={onH5LocalInput}
        style={{ display: "none" }}
        {...({ webkitdirectory: "", directory: "" } as object)}
      />
      {/* HEADER */}
      {showTitle && <Typography variant="h6" sx={{ ...typo.title, mb: `${SPACING.SM}px` }}>
        {title || "4D-STEM Explorer"}
        {nFrames > 1 && <span style={{ fontWeight: "normal", fontSize: 13, marginLeft: 8, opacity: 0.7 }}>({frameLabels && frameLabels.length > frameIdx ? frameLabels[frameIdx] : `${frameDimLabel} ${frameIdx + 1}/${nFrames}`})</span>}
        {gpuMemoryLabel && <span style={{ fontWeight: 500, fontSize: 11, marginLeft: 10, opacity: 0.66 }} title="Current Python GPU memory">
          {gpuMemoryLabel}
        </span>}
        {debug && <DebugPerfBadge widget="Show4DSTEM" fps={debugFps} themeColors={themeColors} />}
        {panelChromeVisible && <InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
          <MetadataSection rows={[
            ["Scan", `${shapeRows} x ${shapeCols}`],
            ["Detector", `${detRows} x ${detCols}`],
            ["Frames", nFrames > 1 ? `${nFrames} ${frameDimLabel}` : "single frame"],
            ["Real space", pixelSize > 0 ? `${formatNumber(pixelSize)} ${pixelUnit || "px"}/px` : ""],
            ["Diffraction", kCalibrated && kPixelSize > 0 ? `${formatNumber(kPixelSize)} ${kPixelUnit || "px"}/px` : "detector pixels"],
          ]} />
          <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>DP: Diffraction pattern I(kx,ky) at scan position. Drag to move ROI center.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Detector: ROI mask shape defines which DP pixels are integrated for the virtual image.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>BF/ABF/ADF: Preset detector configurations (bright-field, annular bright-field, annular dark-field).</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Image: Virtual image, integrated intensity within detector ROI at each scan position.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Spatial frequency content of the virtual image. Auto masks DC and clips to the 99.9th percentile.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Smooth: CSS bilinear blit on the VI canvas. No data change; browser smooths the upscale visually. Off = nearest-neighbor.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile contrast (1st-99th). Clips outliers automatically.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on DP to draw a line intensity profile.</Typography>
          {nFrames > 1 && <>
            <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Frame playback ({frameDimLabel})</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop: Loop playback. Bounce: Ping-pong, alternates forward and reverse.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FPS: Adjust playback speed (1-30 frames per second).</Typography>
          </>}
          <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
          <KeyboardShortcuts items={keyboardShortcutItems} />
        </Box>} theme={themeInfo.theme} />}
      </Typography>}
      {memoryWarning && (
        <Box role="status" data-testid="show4dstem-memory-warning" sx={memoryWarningSx}>
          {memoryWarning}
        </Box>
      )}
      {showOfflineStatus && (
        <Box
          role="status"
          data-testid="show4dstem-offline-status"
          data-quantem-load-error={offlineStatusIsError ? "true" : undefined}
          sx={{
            mb: `${SPACING.SM}px`,
            px: 1,
            py: 0.25,
            border: `1px solid ${offlineStatusIsError ? "#d32f2f" : themeColors.border}`,
            bgcolor: themeColors.controlBg,
            ...typo.label,
            color: offlineStatusIsError ? "#d32f2f" : themeColors.textMuted,
            width: "fit-content",
            maxWidth: "100%",
            lineHeight: 1.35,
            overflowWrap: "anywhere",
          }}
        >
          {offlineStatusIsError ? `Show4DSTEM load failed: ${offlineStatusText}` : offlineStatusText}
        </Box>
      )}
      {showLocalH5GrantBanner && (
        <Box
          role="status"
          data-testid="show4dstem-local-h5-grant"
          sx={{
            mb: `${SPACING.SM}px`,
            px: 1,
            py: 0.75,
            border: `1px solid ${themeColors.border}`,
            bgcolor: themeColors.controlBg,
            color: themeColors.text,
            display: "flex",
            alignItems: "center",
            gap: 1,
            flexWrap: "wrap",
            maxWidth: "100%",
            boxSizing: "border-box",
          }}
        >
          <Typography sx={{ ...typo.label, color: themeColors.text }}>
            {localH5FolderName
              ? `No server needed - click Open data folder, then select "${localH5FolderName}".`
              : "No server needed - grant this page access to its exported HDF5 folder."}
          </Typography>
          {h5LocalSourceStatus && (
            <Typography sx={{ ...typo.label, color: h5LocalSourceStatus.includes("No ") ? "#d32f2f" : themeColors.textMuted }}>
              {h5LocalSourceStatus}
            </Typography>
          )}
          <Typography sx={{ ...typo.label, color: themeColors.textMuted, fontSize: 11 }}>
            Alternative: double-click Show4DSTEM.command.
          </Typography>
          <Button
            size="small"
            variant="outlined"
            onClick={grantH5LocalFiles}
            sx={{ ...compactButton, color: themeColors.accent }}
            data-show4dstem-open-folder
          >
            Open data folder
          </Button>
        </Box>
      )}
      {/* MAIN CONTENT: DP | VI | FFT (three columns when FFT shown) */}
      <Stack
        direction={mainStackDirection}
        sx={{
          gap: `${SPACING.LG}px`,
          flexWrap: "wrap",
          alignItems: "flex-start",
          maxWidth: "100%",
          overflowX: "hidden",
          "@media (max-width: 700px)": {
            flexDirection: "column",
            alignItems: "stretch",
            gap: mobileTightLayout ? 0 : "4px",
            "& > :not(style) + :not(style)": {
              marginLeft: "0 !important",
              marginTop: 0,
            },
          },
        }}
      >
        {/* LEFT COLUMN: DP Panel */}
        <Box sx={{ width: squarePanelWidth, maxWidth: "100%", ...mobilePanelSx }}>
          {/* DP Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={panelHeaderSx}>
            <Typography variant="caption" sx={{ ...typo.label }}>
              DP at ({Math.round(localPosRow)}, {Math.round(localPosCol)})
              <span style={{ color: roiColors.textColor, marginLeft: SPACING.SM }}>k: ({Math.round(localKRow)}, {Math.round(localKCol)})</span>
            </Typography>
            {controlsVisible && <Stack
              direction="row"
              spacing={`${SPACING.SM}px`}
              alignItems="center"
              justifyContent="flex-end"
              sx={{ flexWrap: "wrap", rowGap: 0.5 }}
            >
              <Typography sx={{ ...typo.label, fontSize: 10 }}>Profile</Typography>
              <Switch checked={profileActive} onChange={(e) => {
                const on = e.target.checked;
                setProfileActive(on);
                if (!on) {
                  setProfileLine([]);
                  setProfileData(null);
                  setHoveredDpProfileEndpoint(null);
                  setIsHoveringDpProfileLine(false);
                }
              }} size="small" sx={switchStyles.small} />
              <Button size="small" sx={compactButton} disabled={dpZoom === 1 && dpPanX === 0 && dpPanY === 0 && roiCenterCol === centerCol && roiCenterRow === centerRow} onClick={() => { setDpZoom(1); setDpPanX(0); setDpPanY(0); setRoiCenterCol(centerCol); setRoiCenterRow(centerRow); }}>Reset</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={async () => {
                if (!dpCanvasRef.current) return;
                try {
                  const blob = await new Promise<Blob | null>(resolve => dpCanvasRef.current!.toBlob(resolve, "image/png"));
                  if (!blob) return;
                  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                } catch {
                  dpCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4dstem_dp.png"); }, "image/png");
                }
              }}>Copy</Button>
              {offline && h5SourceAvailable && <Button
                size="small"
                sx={{ ...compactButton, color: h5LocalFilesGranted ? themeColors.accent : themeColors.textMuted }}
                onClick={grantH5LocalFiles}
                title={h5LocalSourceStatus || "Grant local HDF5 master/data files for browser WebGPU load"}
              >
                Local H5
              </Button>}
              {exportEnabled && <Button
                size="small"
                sx={{ ...compactButton, color: themeColors.accent }}
                onClick={(e) => setDpExportAnchor(e.currentTarget)}
                disabled={htmlExportBusy}
                title={localHtmlExportStatus || exportStatus || "Export standalone HTML"}
              >
                {htmlExportBusy ? "..." : "HTML"}
              </Button>}
              {exportEnabled && <Menu anchorEl={dpExportAnchor} open={Boolean(dpExportAnchor)} onClose={() => setDpExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <Box sx={{ px: 1.5, pt: 1, pb: 0.25, fontSize: 11, color: themeColors.textMuted, fontWeight: 700 }}>
                  HTML report: static PNG, no raw 4D
                </Box>
                <MenuItem onClick={() => handleHtmlExportSelect("report", "uint8", reportDetBin, reportScanBin, "unhidden")} sx={{ fontSize: 12 }}>
                    Unhidden · rbin {reportScanBin} · DP kbin {reportDetBin} ({estimateHtmlExportSize("report", "uint8", reportDetBin, reportScanBin, "unhidden")})
                </MenuItem>
                {currentPageReportCount > 0 && (
                  <MenuItem onClick={() => handleHtmlExportSelect("report", "uint8", reportDetBin, detailedReportScanBin, "current_page")} sx={{ fontSize: 12 }}>
                    Current page · rbin {detailedReportScanBin} · DP kbin {reportDetBin} ({estimateHtmlExportSize("report", "uint8", reportDetBin, detailedReportScanBin, "current_page")})
                  </MenuItem>
                )}
                {starredReportCount > 0 && (
                  <MenuItem onClick={() => handleHtmlExportSelect("report", "uint8", reportDetBin, detailedReportScanBin, "starred")} sx={{ fontSize: 12 }}>
                    Starred · rbin {detailedReportScanBin} · DP kbin {reportDetBin} ({estimateHtmlExportSize("report", "uint8", reportDetBin, detailedReportScanBin, "starred")})
                  </MenuItem>
                )}
                <Box sx={{ px: 1.5, pt: 1, pb: 0.25, fontSize: 11, color: themeColors.textMuted, fontWeight: 700 }}>
                  HTML interactive raw 4D
                </Box>
                {interactiveHtmlPresets.map((preset) => (
                  <MenuItem
                    key={`${preset.dtype}-${preset.scanBin}-${preset.detBin}`}
                    onClick={() => handleHtmlExportSelect("interactive", preset.dtype, preset.detBin, preset.scanBin, "unhidden")}
                    sx={{ fontSize: 12 }}
                  >
                    {preset.label} · {preset.dtype} · rbin {preset.scanBin} · kbin {preset.detBin} ({formatEstimatedHtmlBytes(preset.estimatedBytes)})
                  </MenuItem>
                ))}
              </Menu>}
              {ssbComputeEnabled && <Button
                size="small"
                sx={compactButton}
                onClick={(e) => setDpMoreAnchor(e.currentTarget)}
                title="More actions"
              >
                More
              </Button>}
              {ssbComputeEnabled && <Menu
                anchorEl={dpMoreAnchor}
                open={Boolean(dpMoreAnchor)}
                onClose={() => setDpMoreAnchor(null)}
                anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
                transformOrigin={{ vertical: "top", horizontal: "left" }}
                sx={{ zIndex: 9999 }}
                PaperProps={{ sx: { bgcolor: themeColors.bgAlt, backgroundImage: "none", color: themeColors.text, border: `1px solid ${themeColors.border}` } }}
              >
                <Box sx={{ px: 1.5, py: 1, width: 242, boxSizing: "border-box" }}>
                  <Stack direction="row" alignItems="center" sx={{ mb: 0.75, gap: 0.25 }}>
                    <Typography sx={{ ...typo.label, color: themeColors.text }}>
                      SSB
                    </Typography>
                    <InfoTooltip
                      theme={themeInfo.theme}
                      text={
                        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                          <Typography sx={{ fontSize: 11, lineHeight: 1.35 }}>
                            SSB (single-sideband ptychography) computes a phase image from the 4D-STEM diffraction stack using the live backend.
                          </Typography>
                          <Typography sx={{ fontSize: 11, lineHeight: 1.35 }}>
                            Trials controls the aberration search; Refine runs a final local fit. Lock C10 or C12 to pin a coefficient at its slider value during the search.
                          </Typography>
                          <Typography sx={{ fontSize: 11, lineHeight: 1.35 }}>
                            BF ratio uses a uniform subset of detected BF pixels for optimize/refine. Calibration sliders appear below the image after the phase is ready.
                          </Typography>
                        </Box>
                      }
                    />
                  </Stack>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.75, gap: 1 }}>
                    <Typography sx={typo.label}>Trials</Typography>
                    <Select
                      value={Math.max(0, Math.round(Number(ssbComputeNTrials ?? 200)))}
                      onChange={(e) => setSsbComputeNTrials(Number(e.target.value))}
                      size="small"
                      disabled={ssbComputeBusy}
                      sx={{ ...themedSelect, minWidth: 82, fontSize: 10 }}
                      MenuProps={themedMenuProps}
                    >
                      <MenuItem value={0}>0</MenuItem>
                      <MenuItem value={20}>20</MenuItem>
                      <MenuItem value={50}>50</MenuItem>
                      <MenuItem value={100}>100</MenuItem>
                      <MenuItem value={200}>200</MenuItem>
                    </Select>
                  </Stack>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5, gap: 1 }}>
                    <Typography sx={typo.label}>Refine</Typography>
                    <Switch
                      checked={Boolean(ssbComputeRefine)}
                      onChange={(e) => setSsbComputeRefine(e.target.checked)}
                      disabled={ssbComputeBusy}
                      size="small"
                      sx={switchStyles.small}
                    />
                  </Stack>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5, gap: 1 }}>
                    <Typography sx={typo.label}>BF ratio</Typography>
                    <Select
                      value={activeSsbBfSubsample}
                      onChange={(e) => setSsbComputeBfSubsample(Number(e.target.value))}
                      size="small"
                      disabled={ssbComputeBusy}
                      sx={{ ...themedSelect, minWidth: 82, fontSize: 10 }}
                      MenuProps={themedMenuProps}
                    >
                      <MenuItem value={0.3}>0.3</MenuItem>
                      <MenuItem value={0.5}>0.5</MenuItem>
                      <MenuItem value={1}>1.0</MenuItem>
                    </Select>
                  </Stack>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5, gap: 1 }}>
                    <Typography sx={typo.label} title="Pin C10 at its slider value during the aberration search">
                      Lock C10 <Box component="span" sx={{ color: themeColors.textMuted }}>{Number(ssbComputeC10Nm ?? 0).toFixed(0)} nm</Box>
                    </Typography>
                    <Switch
                      checked={Boolean(ssbComputeLockC10)}
                      onChange={(e) => setSsbComputeLockC10(e.target.checked)}
                      disabled={ssbComputeBusy}
                      size="small"
                      sx={switchStyles.small}
                    />
                  </Stack>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5, gap: 1 }}>
                    <Typography sx={typo.label} title="Pin C12 and φ12 at their slider values during the aberration search">
                      Lock C12 <Box component="span" sx={{ color: themeColors.textMuted }}>{Number(ssbComputeC12Nm ?? 0).toFixed(0)} nm</Box>
                    </Typography>
                    <Switch
                      checked={Boolean(ssbComputeLockC12)}
                      onChange={(e) => setSsbComputeLockC12(e.target.checked)}
                      disabled={ssbComputeBusy}
                      size="small"
                      sx={switchStyles.small}
                    />
                  </Stack>
                  <Typography sx={{ ...typo.label, color: themeColors.textMuted, whiteSpace: "normal", lineHeight: 1.35, mb: 0.75 }}>
                    {ssbBfCountText}
                  </Typography>
                  {ssbProgressText && (
                    <Typography
                      role="status"
                      sx={{
                        ...typo.label,
                        color: ssbStatusIsFailure
                          ? "#d32f2f"
                          : ssbComputeBusy
                            ? themeColors.accent
                            : themeColors.textMuted,
                        whiteSpace: "normal",
                        lineHeight: 1.35,
                        mb: 0.75,
                      }}
                    >
                      {ssbDisplayStatus}
                    </Typography>
                  )}
                  <Typography sx={{ ...typo.label, color: themeColors.textMuted, whiteSpace: "normal", lineHeight: 1.35 }}>
                    Default is 200 trials, refine on, BF ratio 1.0. Full 512 scans can take seconds to about a minute.
                  </Typography>
                </Box>
                <MenuItem onClick={() => requestSsbCompute()} disabled={ssbComputeBusy} sx={{ fontSize: 12 }}>
                  Calculate Phase
                </MenuItem>
                {hasSsbCalibrationDownload && (
                  <MenuItem onClick={downloadSsbCalibration} disabled={ssbComputeBusy} sx={{ fontSize: 12 }}>
                    Download calibration JSON
                  </MenuItem>
                )}
              </Menu>}
              {exportEnabled && (localHtmlExportStatus || exportStatus) && (
                <Typography
                  sx={{
                    ...typo.label,
                    maxWidth: 120,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: (localHtmlExportStatus || exportStatus).startsWith("Export failed") ? "#d32f2f" : themeColors.textMuted,
                  }}
                  title={localHtmlExportStatus || exportStatus}
                >
                  {localHtmlExportStatus || exportStatus}
                </Typography>
              )}
              {h5LocalSourceStatus && (
                <Typography
                  sx={{
                    ...typo.label,
                    maxWidth: 140,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: h5LocalSourceStatus.includes("failed") || h5LocalSourceStatus.includes("No ")
                      ? "#d32f2f"
                      : themeColors.textMuted,
                  }}
                  title={h5LocalSourceStatus}
                >
                  {h5LocalSourceStatus}
                </Typography>
              )}
              {(ssbComputeStatus || ssbComputeBusy) && (
                <Typography
                  sx={{
                    ...typo.label,
                    maxWidth: 140,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: ssbStatusIsFailure
                      ? "#d32f2f"
                      : ssbComputeBusy
                        ? themeColors.accent
                        : themeColors.textMuted,
                  }}
                  title={ssbDisplayStatus || "Running SSB..."}
                >
                  {ssbDisplayStatus || "Running SSB..."}
                </Typography>
              )}
            </Stack>}
          </Stack>

          {/* DP Canvas */}
          <Box sx={{ ...container.imageBox, width: "100%", maxWidth: canvasSize, aspectRatio: "1 / 1", height: "auto", touchAction: "none", ...mobileImageBoxSx }}>
            <canvas data-quantem-scientific-output="show4dstem-diffraction-pattern" ref={dpCanvasRef} width={detCols} height={detRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={dpOverlayRef} width={detCols} height={detRows}
              onPointerDown={handleDpMouseDown} onPointerMove={handleDpMouseMove}
              onPointerUp={handleDpMouseUp} onPointerCancel={handleDpMouseUp} onMouseLeave={handleDpMouseLeave}
              onWheel={createZoomHandler(setDpZoom, setDpPanX, setDpPanY, dpViewRef, dpOverlayRef)}
              onDoubleClick={handleDpDoubleClick}
              onTouchStart={handlePanelTouchStart("dp")}
              onTouchMove={handlePanelTouchMove("dp")}
              onTouchEnd={handlePanelTouchEnd}
              onTouchCancel={handlePanelTouchEnd}
              style={{
                position: "absolute",
                width: "100%",
                height: "100%",
                touchAction: "none",
                cursor: (draggingDpProfileEndpoint !== null || isDraggingDpProfileLine)
                  ? "grabbing"
                  : (profileActive && (hoveredDpProfileEndpoint !== null || isHoveringDpProfileLine))
                    ? "grab"
                    : isHoveringResize || isDraggingResize
                      ? "nwse-resize"
                      : "crosshair",
              }}
            />
            <canvas ref={dpUiRef} width={canvasSize * DPR} height={canvasSize * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
            {(dpPanelLoading || offlineBackendError) && renderPanelLoadingOverlay(
              offlineBackendError ? "Show4DSTEM load failed" : "Loading DP",
            )}
            {panelChromeVisible && cursorInfo && cursorInfo.panel === "DP" && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            {panelChromeVisible && <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />}
          </Box>

          {/* DP Stats Bar */}
          {showStats && !dpPanelLoading && dpStats && dpStats.length === 4 && (
            <Box sx={{ ...statsBarSx, ...hideBetweenPanelsOnMobileSx }}>
              <Typography sx={statsTextSx}>Mean <Box component="span" sx={statsValueSx}>{formatStat(dpStats[0])}</Box></Typography>
              <Typography sx={statsTextSx}>Min <Box component="span" sx={statsValueSx}>{formatStat(dpStats[1])}</Box></Typography>
              <Typography sx={statsTextSx}>Max <Box component="span" sx={statsValueSx}>{formatStat(dpStats[2])}</Box></Typography>
              <Typography sx={statsTextSx}>Std <Box component="span" sx={statsValueSx}>{formatStat(dpStats[3])}</Box></Typography>
              {controlsVisible && <>
                <Box sx={{ flex: 1, minWidth: 4, "@media (max-width: 700px)": { display: "none" } }} />
                <Typography component="span" onClick={() => requestViPreset("bf")} sx={viSourceButtonSx("bf", activeViSource === "roi")}>BF</Typography>
                <Typography component="span" onClick={() => requestViPreset("abf")} sx={viSourceButtonSx("abf")}>ABF</Typography>
                <Typography component="span" onClick={() => requestViPreset("adf")} sx={viSourceButtonSx("adf")}>ADF</Typography>
                {hasViProductSources && viProductSourceOptions.map((source) => {
                  const active = activeViSource === source;
                  return (
                    <Typography
                      key={source}
                      component="span"
                      aria-label={`Show ${viSourceLabel(source)} virtual detector`}
                      aria-pressed={active}
                      onClick={() => setViSource(source)}
                      sx={viSourceButtonSx(source, active)}
                    >
                      <ViSourceLabel source={source} />
                    </Typography>
                  );
                })}
              </>}
            </Box>
          )}

          {/* Profile sparkline */}
          {profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, width: "100%", maxWidth: canvasSize, boxSizing: "border-box", ...mobileImageBoxSx }}>
              <canvas
                ref={profileCanvasRef}
                onMouseMove={handleProfileMouseMove}
                onMouseLeave={handleProfileMouseLeave}
                style={{ width: "100%", height: profileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
              />
              <Box
                onMouseDown={(e) => {
                  setIsResizingProfile(true);
                  profileResizeStart.current = { startY: e.clientY, startHeight: profileHeight };
                }}
                sx={{ width: "100%", height: 4, cursor: "ns-resize", borderTop: `1px solid ${themeColors.border}`, borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, "&:hover": { bgcolor: themeColors.accent } }}
              />
            </Box>
          )}

          {/* DP Controls - two rows with histogram on right */}
          {controlsVisible && (
            <>
              <Button
                size="small"
                onClick={() => setMobileDpOptionsOpen(v => !v)}
                sx={mobileOptionToggleSx}
                endIcon={mobileDpOptionsOpen ? <KeyboardArrowUpIcon fontSize="small" /> : <KeyboardArrowDownIcon fontSize="small" />}
              >
                <Box component="span">Detector options</Box>
                <Box component="span" sx={mobileOptionSummarySx}>{dpOptionSummary}</Box>
              </Button>
              <Box sx={mobileOptionsPanelSx(mobileDpOptionsOpen)}>
                <Box sx={mobileOptionsContentSx}>
                  {/* Left: two rows of controls */}
                  <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: "1 1 220px", minWidth: 0, justifyContent: "center" }}>
                    {/* Row 1: Detector + slider */}
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                      <Typography sx={{ ...typo.label, fontSize: 10 }}>Detector</Typography>
                      <Select value={roiMode || "point"} onChange={(e) => setRoiMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="point">Point</MenuItem>
                        <MenuItem value="circle">Circle</MenuItem>
                        <MenuItem value="square">Square</MenuItem>
                        <MenuItem value="rect">Rect</MenuItem>
                        <MenuItem value="annular">Annular</MenuItem>
                      </Select>
                      {(roiMode === "circle" || roiMode === "square" || roiMode === "annular") && (
                        <>
                          <Slider
                            value={roiMode === "annular" ? [roiRadiusInner, roiRadius] : [roiRadius]}
                            onChange={(_, v) => {
                              dpRoiInteractiveRef.current = true;
                              if (roiMode === "annular") {
                                const [inner, outer] = v as number[];
                                setRoiRadiusInner(Math.min(inner, outer - 1));
                                setRoiRadius(Math.max(outer, inner + 1));
                              } else {
                                const next = Array.isArray(v) ? v[0] : v;
                                setRoiRadius(next);
                              }
                              requestCompareViLive();
                            }}
                            onChangeCommitted={finishDpRoiInteraction}
                            min={1}
                            max={Math.min(detRows, detCols) / 2}
                            size="small"
                            sx={{ ...sliderStyles.small, width: roiMode === "annular" ? 67 : 47, mx: 1 }}
                          />
                          <Typography sx={{ ...typo.label, fontSize: 10 }}>
                            {roiMode === "annular" ? `${Math.round(roiRadiusInner)}-${Math.round(roiRadius)}px` : `${Math.round(roiRadius)}px`}
                          </Typography>
                        </>
                      )}
                    </Box>
                    {/* Row 2: Color + Scale + Colorbar */}
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                      <Typography sx={{ ...typo.label, fontSize: 10 }}>Color</Typography>
                      <Select value={dpColormap} onChange={(e) => setDpColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="inferno">Inferno</MenuItem>
                        <MenuItem value="viridis">Viridis</MenuItem>
                        <MenuItem value="plasma">Plasma</MenuItem>
                        <MenuItem value="magma">Magma</MenuItem>
                        <MenuItem value="hot">Hot</MenuItem>
                        <MenuItem value="RdBu_r">RdBu</MenuItem>
                        <MenuItem value="twilight_shifted">Twilight</MenuItem>
                        <MenuItem value="gray">Gray</MenuItem>
                      </Select>
                      <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale</Typography>
                      <Select value={dpScaleMode} onChange={(e) => setDpScaleMode(e.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="linear">Lin</MenuItem>
                        <MenuItem value="log">Log</MenuItem>

                      </Select>
                      <Typography sx={{ ...typo.label, fontSize: 10 }}>Colorbar</Typography>
                      <Switch checked={showDpColorbar} onChange={(e) => setShowDpColorbar(e.target.checked)} size="small" sx={switchStyles.small} />
                    </Box>
                  </Box>
                  {/* Right: Histogram spanning both rows */}
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", justifyContent: "center", flex: "0 0 auto", maxWidth: "100%" }}>
                    <Histogram data={dpHistogramData} vminPct={dpVminPct} vmaxPct={dpVmaxPct} onRangeChange={(min, max) => { setDpVminPct(min); setDpVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={dpGlobalMin} dataMax={dpGlobalMax} />
                  </Box>
                </Box>
              </Box>
            </>
          )}
        </Box>

        {/* SECOND COLUMN: VI Panel */}
        <Box sx={{ width: viPanelWidth, maxWidth: "100%", ...mobilePanelSx }}>
          {/* VI Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ ...panelHeaderSx, ...hideBetweenPanelsOnMobileSx }}>
            <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} sx={{ minWidth: 0, flexWrap: "wrap", rowGap: 0.5 }}>
              <Typography sx={{ ...typo.label, color: themeColors.textMuted, flexShrink: 0 }}>
                {compareMode ? "Multiple " : ""}<ViSourceLabel source={activeViSource} />{compareMode ? ` | ${shapeRows}×${shapeCols}` : ` | ${shapeRows}×${shapeCols} | ${detRows}×${detCols}`}
              </Typography>
              {controlsVisible && compareMode && activeComparePageCount > 1 && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.35, flexShrink: 0 }}>
                  <Typography sx={{ ...typo.label, fontSize: 10, flexShrink: 0 }}>Group</Typography>
                  <Box
                    role="group"
                    aria-label="Show4DSTEM multiple group mode"
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      border: `1px solid ${themeColors.border}`,
                      bgcolor: themeColors.controlBg,
                      height: 22,
                      overflow: "hidden",
                    }}
                  >
                    {[
                      ["paged", "Paged"],
                      ["all", "All"],
                    ].map(([value, label]) => {
                      const active = compareAllGroups ? value === "all" : value === "paged";
                      return (
                        <Button
                          key={value}
                          size="small"
                          aria-label={`Use ${label.toLowerCase()} Show4DSTEM multiple groups`}
                          aria-pressed={active}
                          onClick={() => {
                            setComparePagePlaying(false);
                            setCompareGroupMode(value);
                          }}
                          sx={{
                            ...compactButton,
                            minWidth: 38,
                            height: 20,
                            px: 0.5,
                            borderRadius: 0,
                            color: active ? "#fff" : themeColors.textMuted,
                            bgcolor: active ? themeColors.accent : "transparent",
                            "&:hover": { bgcolor: active ? themeColors.accent : themeColors.bgAlt },
                          }}
                        >
                          {label}
                        </Button>
                      );
                    })}
                  </Box>
                  {!compareAllGroups && <>
                  <IconButton
                    size="small"
                    aria-label="Previous Show4DSTEM multiple group"
                    disabled={activeComparePageIdx <= 0}
                    onClick={() => {
                      setComparePagePlaying(false);
                      requestComparePage(activeComparePageIdx - 1);
                    }}
                    sx={{ color: activeComparePageIdx <= 0 ? themeColors.textMuted : themeColors.accent, p: 0.2 }}
                  >
                    <FastRewindIcon sx={{ fontSize: 15 }} />
                  </IconButton>
                  <IconButton
                    size="small"
                    aria-label={comparePagePlaying ? "Pause Show4DSTEM multiple groups" : "Play Show4DSTEM multiple groups"}
                    onClick={() => {
                      if (comparePagePlaying) {
                        setComparePagePlaying(false);
                        return;
                      }
                      if (activeComparePageIdx >= activeComparePageCount - 1) {
                        requestComparePage(0);
                      }
                      setComparePagePlaying(true);
                    }}
                    sx={{ color: comparePagePlaying ? themeColors.accent : themeColors.textMuted, p: 0.2 }}
                  >
                    {comparePagePlaying ? <PauseIcon sx={{ fontSize: 15 }} /> : <PlayArrowIcon sx={{ fontSize: 15 }} />}
                  </IconButton>
                  <Box
                    role="group"
                    aria-label="Show4DSTEM multiple groups"
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      border: `1px solid ${themeColors.border}`,
                      bgcolor: themeColors.controlBg,
                      height: 22,
                      overflow: "hidden",
                    }}
                  >
                    {comparePageButtonItems.map((item, idx) => {
                      if (item === "gap") {
                        return (
                          <Typography
                            key={`gap-${idx}`}
                            sx={{ ...typo.value, width: 16, textAlign: "center", color: themeColors.textMuted, lineHeight: "20px" }}
                          >
                            …
                          </Typography>
                        );
                      }
                      const active = item === activeComparePageIdx;
                      return (
                        <Button
                          key={item}
                          size="small"
                          aria-label={`Show Show4DSTEM multiple group ${item + 1}`}
                          aria-pressed={active}
                          onClick={() => {
                            setComparePagePlaying(false);
                            requestComparePage(item);
                          }}
                          sx={{
                            ...compactButton,
                            minWidth: 23,
                            height: 20,
                            px: 0.4,
                            borderRadius: 0,
                            color: active ? "#fff" : themeColors.textMuted,
                            bgcolor: active ? themeColors.accent : "transparent",
                            "&:hover": { bgcolor: active ? themeColors.accent : themeColors.bgAlt },
                          }}
                        >
                          {item + 1}
                        </Button>
                      );
                    })}
                  </Box>
                  <IconButton
                    size="small"
                    aria-label="Next Show4DSTEM multiple group"
                    disabled={activeComparePageIdx >= activeComparePageCount - 1}
                    onClick={() => {
                      setComparePagePlaying(false);
                      requestComparePage(activeComparePageIdx + 1);
                    }}
                    sx={{ color: activeComparePageIdx >= activeComparePageCount - 1 ? themeColors.textMuted : themeColors.accent, p: 0.2 }}
                  >
                    <FastForwardIcon sx={{ fontSize: 15 }} />
                  </IconButton>
                  </>}
                  <Typography sx={{ ...typo.value, minWidth: compareAllGroups ? 58 : activeComparePageCount > 99 ? 52 : 34, textAlign: "left", flexShrink: 0 }}>{comparePageStatus}</Typography>
                </Box>
              )}
            </Stack>
            {controlsVisible && <Stack
              direction="row"
              spacing={`${SPACING.SM}px`}
              alignItems="center"
              justifyContent="flex-end"
              sx={{ flexWrap: "wrap", rowGap: 0.5 }}
            >
              {compareMode && <>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Cols</Typography>
                <Select
                  value={compareCols || 0}
                  onChange={(e) => setCompareCols(Number(e.target.value))}
                  size="small"
                  inputProps={{ "aria-label": "Show4DSTEM multiple columns" }}
                  sx={{ ...themedSelect, minWidth: 54, fontSize: 10 }}
                  MenuProps={themedMenuProps}
                >
                  <MenuItem value={0}>Auto</MenuItem>
                  <MenuItem value={2}>2</MenuItem>
                  <MenuItem value={3}>3</MenuItem>
                  <MenuItem value={4}>4</MenuItem>
                  <MenuItem value={5}>5</MenuItem>
                </Select>
                <Tooltip title={compareHiddenCount > 0 ? `${compareHiddenCount} hidden panel${compareHiddenCount === 1 ? "" : "s"}` : "No hidden panels"}>
                  <Button
                    size="small"
                    aria-label="Show4DSTEM hidden multiple panels"
                    className="show4dstem-compare-hidden-menu"
                    onClick={(event) => setCompareHiddenMenuAnchor(event.currentTarget)}
                    startIcon={<VisibilityOffIcon sx={{ fontSize: 14 }} />}
                    sx={{
                      ...compactButton,
                      minWidth: 64,
                      px: 0.75,
                      color: compareHiddenCount > 0 ? themeColors.accent : themeColors.textMuted,
                      "& .MuiButton-startIcon": { mr: 0.25, ml: 0 },
                    }}
                  >
                    {compareHiddenCount > 0 ? `Hidden ${compareHiddenCount}` : "Hidden"}
                  </Button>
                </Tooltip>
                <Menu
                  anchorEl={compareHiddenMenuAnchor}
                  open={Boolean(compareHiddenMenuAnchor)}
                  onClose={() => setCompareHiddenMenuAnchor(null)}
                  MenuListProps={{ "aria-label": "Show4DSTEM hidden multiple panels menu" }}
                  {...themedMenuProps}
                >
                  {compareHiddenPanelItems.length === 0 ? (
                    <MenuItem disabled>No hidden panels</MenuItem>
                  ) : (
                    compareHiddenPanelItems.map(({ idx, label }) => (
                      <MenuItem
                        key={idx}
                        aria-label={`Show Show4DSTEM multiple panel ${idx + 1}`}
                        onClick={() => showCompareFrame(idx)}
                      >
                        Show {label}
                      </MenuItem>
                    ))
                  )}
                  {compareHiddenPanelItems.length > 1 && (
                    <MenuItem
                      aria-label="Show all Show4DSTEM multiple panels"
                      onClick={() => {
                        setCompareHiddenPanels([]);
                        setCompareHiddenMenuAnchor(null);
                      }}
                    >
                      Show all
                    </MenuItem>
                  )}
                </Menu>
              </>}
              <Typography sx={{ ...typo.label, fontSize: 10 }}>FFT</Typography>
              <Switch checked={effectiveShowFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
              {!compareMode && <>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Profile</Typography>
                <Switch checked={viProfileActive} onChange={(e) => {
                  const on = e.target.checked;
                  setViProfileActive(on);
                  if (!on) {
                    setViProfilePoints([]);
                    setHoveredViProfileEndpoint(null);
                    setIsHoveringViProfileLine(false);
                  }
                }} size="small" sx={switchStyles.small} />
                <Button size="small" sx={compactButton} disabled={viZoom === 1 && viPanX === 0 && viPanY === 0} onClick={() => { setViZoom(1); setViPanX(0); setViPanY(0); }}>Reset</Button>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={async () => {
                  const canvas = getActiveViCanvas();
                  if (!canvas) return;
                  try {
                    const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/png"));
                    if (!blob) return;
                    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                  } catch {
                    canvas.toBlob((b) => { if (b) downloadBlob(b, "show4dstem_vi.png"); }, "image/png");
                  }
                }}>Copy</Button>
              </>}
            </Stack>}
          </Stack>

          {/* VI Canvas */}
          {compareMode ? (
            <CompareVirtualGrid
              bytes={displayedCompareVirtualImageBytes}
              count={comparePanelCount || 0}
              indices={comparePanelIndices || []}
              gpuSlots={compareGpuSlotsRef.current}
              gpuVersion={compareGpuVersion}
              gpuEngine={viGpuColormapRef.current}
              progressivePage={progressiveComparePage}
              labels={frameLabels || []}
              activeIdx={frameIdx}
              shapeRows={shapeRows}
              shapeCols={shapeCols}
              cols={compareCols || 0}
              colormap={viColormap}
              scaleMode={viScaleMode}
              vminPct={viVminPct}
              vmaxPct={viVmaxPct}
              autoContrast={viAutoContrast}
              smooth={viSmooth}
              cursorRow={localPosRow}
              cursorCol={localPosCol}
              status={compareStatus}
              themeColors={themeColors}
              panelChromeVisible={panelChromeVisible}
              showScaleBar={showScaleBar}
              pixelSize={pixelSize}
              pixelUnit={pixelUnit}
              panelOrder={comparePanelOrder || []}
              hidden={compareHiddenPanels || []}
              starred={compareStarredPanels || []}
              reorderMode={compareReorderMode}
              draggingFrame={compareDraggingFrame}
              pendingMoveFrame={comparePendingMoveFrame}
              maxWidthPx={compareGridWidth}
              panelGapPx={comparePanelGapPx}
              onResizeStart={handleCompareGridResizeStart}
              onSelect={(idx) => {
                setFrameIdx(Math.max(0, Math.min(nFrames - 1, idx)));
              }}
              onToggleStar={toggleCompareStar}
              onHide={hideCompareFrame}
              onReorderFrame={moveCompareFrame}
              onDragFrameChange={setCompareDraggingFrame}
              onPendingMoveFrameChange={setComparePendingMoveFrame}
              onPositionChange={updateScanPosition}
              onFreshVisiblePaint={acknowledgeFreshComparePagePaint}
              onGpuPaint={(panelCount) => publishLiveCompareViStats("paint", { paintedPanels: panelCount })}
              onGpuRendererReady={(renderNow) => {
                compareGpuRenderNowRef.current = renderNow;
              }}
            />
          ) : (
            <Box sx={{ ...container.imageBox, width: "100%", maxWidth: viCanvasWidth, aspectRatio: `${shapeCols} / ${shapeRows}`, height: "auto", touchAction: "none", ...mobileImageBoxSx }}>
              <canvas
                data-quantem-scientific-output="show4dstem-virtual-image"
                ref={virtualCanvasRef}
                width={shapeCols}
                height={shapeRows}
                style={{
                  position: "absolute",
                  width: "100%",
                  height: "100%",
                  imageRendering: "pixelated",
                  display: "block",
                }}
              />
              <canvas
                ref={virtualOverlayRef} width={shapeCols} height={shapeRows}
                onPointerDown={handleViMouseDown} onPointerMove={handleViMouseMove}
                onPointerUp={handleViMouseUp} onPointerCancel={handleViMouseUp} onMouseLeave={handleViMouseLeave}
                onWheel={createZoomHandler(setViZoom, setViPanX, setViPanY, viViewRef, virtualOverlayRef)}
                onDoubleClick={handleViDoubleClick}
                onTouchStart={handlePanelTouchStart("vi")}
                onTouchMove={handlePanelTouchMove("vi")}
                onTouchEnd={handlePanelTouchEnd}
                onTouchCancel={handlePanelTouchEnd}
                style={{
                  position: "absolute",
                  width: "100%",
                  height: "100%",
                  touchAction: "none",
                  cursor: (draggingViProfileEndpoint !== null || isDraggingViProfileLine)
                    ? "grabbing"
                    : (viProfileActive && (hoveredViProfileEndpoint !== null || isHoveringViProfileLine))
                      ? "grab"
                      : "crosshair",
                }}
              />
              <canvas ref={viUiRef} width={viCanvasWidth * DPR} height={viCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
              {(viPanelLoading || offlineBackendError) && renderPanelLoadingOverlay(
                offlineBackendError ? "Show4DSTEM load failed" : "Loading virtual image",
              )}
              {panelChromeVisible && cursorInfo && cursorInfo.panel === "VI" && (
                <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                  <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                    ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                  </Typography>
                </Box>
              )}
              {panelChromeVisible && <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />}
            </Box>
          )}

          {/* VI Stats Bar — stats on left, Auto/Smooth toggles on right edge */}
          {showStats && !viPanelLoading && viStats && viStats.length === 4 && (
            <Box sx={statsBarSx}>
              <Typography sx={statsTextSx}>Mean <Box component="span" sx={statsValueSx}>{formatStat(viStats[0])}</Box></Typography>
              <Typography sx={statsTextSx}>Min <Box component="span" sx={statsValueSx}>{formatStat(viStats[1])}</Box></Typography>
              <Typography sx={statsTextSx}>Max <Box component="span" sx={statsValueSx}>{formatStat(viStats[2])}</Box></Typography>
              <Typography sx={statsTextSx}>Std <Box component="span" sx={statsValueSx}>{formatStat(viStats[3])}</Box></Typography>
              {controlsVisible && <Box sx={{ ml: "auto", display: "flex", alignItems: "center", gap: "2px", flexWrap: "nowrap", whiteSpace: "nowrap", flexShrink: 0 }}>
                <Typography sx={{ ...typo.label, fontSize: 10, lineHeight: "20px" }}>Auto</Typography>
                <Switch checked={viAutoContrast} onChange={(e) => toggleViAutoContrast(e.target.checked)} size="small" sx={switchStyles.small} />
                <Typography sx={{ ...typo.label, fontSize: 10, lineHeight: "20px" }} title="CSS bilinear interpolation. Same data, browser smooths visually.">Smooth</Typography>
                <Switch checked={viSmooth} onChange={(e) => setViSmooth(e.target.checked)} size="small" sx={switchStyles.small} />
              </Box>}
            </Box>
          )}

          {/* VI Profile sparkline */}
          {!compareMode && viProfileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, width: "100%", maxWidth: viCanvasWidth, boxSizing: "border-box", ...mobileImageBoxSx }}>
              <canvas
                ref={viProfileCanvasRef}
                onMouseMove={handleViProfileMouseMove}
                onMouseLeave={handleViProfileMouseLeave}
                style={{ width: "100%", height: viProfileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
              />
              <Box
                onMouseDown={(e) => {
                  setIsResizingViProfile(true);
                  viProfileResizeStart.current = { startY: e.clientY, startHeight: viProfileHeight };
                }}
                sx={{ width: "100%", height: 4, cursor: "ns-resize", borderTop: `1px solid ${themeColors.border}`, borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, "&:hover": { bgcolor: themeColors.accent } }}
              />
            </Box>
          )}

          {/* VI Controls - Two rows with histogram on right */}
          {controlsVisible && (
            <>
              <Button
                size="small"
                onClick={() => setMobileViOptionsOpen(v => !v)}
                sx={mobileOptionToggleSx}
                endIcon={mobileViOptionsOpen ? <KeyboardArrowUpIcon fontSize="small" /> : <KeyboardArrowDownIcon fontSize="small" />}
              >
                <Box component="span">Image options</Box>
                <Box component="span" sx={mobileOptionSummarySx}>{viOptionSummary}</Box>
              </Button>
              <Box sx={mobileOptionsPanelSx(mobileViOptionsOpen)}>
                <Box sx={mobileOptionsContentSx}>
                  {/* Left: Two rows of controls */}
                  <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: "1 1 220px", minWidth: 0, justifyContent: "center" }}>
                    {/* Row 1: ROI selector */}
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                      <Typography sx={{ ...typo.label, fontSize: 10 }}>ROI</Typography>
                      <Select value={viRoiMode || "off"} onChange={(e) => setViRoiMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="off">Off</MenuItem>
                        <MenuItem value="circle">Circle</MenuItem>
                        <MenuItem value="square">Square</MenuItem>
                        <MenuItem value="rect">Rect</MenuItem>
                      </Select>
                      {viRoiMode && viRoiMode !== "off" && (
                        <>
                          {(viRoiMode === "circle" || viRoiMode === "square") && (
                            <>
                              <Slider
                                value={viRoiRadius || 5}
                                onChange={(_, v) => setViRoiRadius(v as number)}
                                min={1}
                                max={Math.min(shapeRows, shapeCols) / 2}
                                size="small"
                                sx={{ ...sliderStyles.small, width: 53, mx: 1 }}
                              />
                              <Typography sx={{ ...typo.value, fontSize: 10, minWidth: 30 }}>
                                {Math.round(viRoiRadius || 5)}px
                              </Typography>
                            </>
                          )}
                          <Select value={viRoiReduce || "mean"} onChange={(e) => setViRoiReduce(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps}>
                            <MenuItem value="mean">Mean</MenuItem>
                            <MenuItem value="sum">Sum</MenuItem>
                            <MenuItem value="max">Max</MenuItem>
                          </Select>
                        </>
                      )}
                    </Box>
                    {/* Row 2: Color + Scale */}
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                      <Typography sx={{ ...typo.label, fontSize: 10 }}>Color</Typography>
                      <Select value={viColormap} onChange={(e) => setViColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="inferno">Inferno</MenuItem>
                        <MenuItem value="viridis">Viridis</MenuItem>
                        <MenuItem value="plasma">Plasma</MenuItem>
                        <MenuItem value="magma">Magma</MenuItem>
                        <MenuItem value="hot">Hot</MenuItem>
                        <MenuItem value="RdBu_r">RdBu</MenuItem>
                        <MenuItem value="twilight_shifted">Twilight</MenuItem>
                        <MenuItem value="gray">Gray</MenuItem>
                      </Select>
                      <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale</Typography>
                      <Select value={viScaleMode} onChange={(e) => setViScaleMode(e.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="linear">Lin</MenuItem>
                        <MenuItem value="log">Log</MenuItem>
                      </Select>
                    </Box>
                  </Box>
                  {/* Right: Histogram spanning both rows */}
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", justifyContent: "center", flex: "0 0 auto", maxWidth: "100%" }}>
                    <Histogram data={viHistogramData} bins={viHistogramBins} vminPct={viVminPct} vmaxPct={viVmaxPct} onRangeChange={(min, max) => { if (viAutoContrast) { viPreAutoPctRef.current = null; setViAutoContrast(false); } setViVminPct(min); setViVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={viDataMin} dataMax={viDataMax} />
                  </Box>
                </Box>
              </Box>
              {showSsbCalibrationPanel && (
                <Box sx={{ mt: `${SPACING.XS}px`, width: "100%", maxWidth: viCanvasWidth, boxSizing: "border-box" }}>
                  <Button
                    size="small"
                    onClick={() => setSsbCalOpen(!ssbCalOpen)}
                    sx={ssbCalToggleSx}
                    endIcon={ssbCalOpen ? <KeyboardArrowUpIcon fontSize="small" /> : <KeyboardArrowDownIcon fontSize="small" />}
                    aria-expanded={ssbCalOpen}
                  >
                    <Box component="span">SSB calibration</Box>
                    <Box component="span" sx={mobileOptionSummarySx}>{ssbCalSummary}</Box>
                  </Button>
                  {ssbCalOpen && (
                  <Box
                    sx={{
                      border: `1px solid ${themeColors.border}`,
                      borderTop: "none",
                      bgcolor: themeColors.controlBg,
                      px: 1,
                      py: 0.75,
                      display: "flex",
                      flexDirection: "column",
                      gap: `${SPACING.XS}px`,
                      boxSizing: "border-box",
                    }}
                  >
                  <Stack direction="row" alignItems="center" sx={{ gap: 0.75, flexWrap: "wrap" }}>
                    {ssbDisplayStatus && (
                      <Typography
                        role="status"
                        sx={{
                          ...typo.label,
                          color: ssbStatusIsFailure
                            ? "#d32f2f"
                            : ssbComputeBusy
                              ? themeColors.accent
                              : themeColors.textMuted,
                          minWidth: 0,
                          flex: "1 1 120px",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                        title={ssbDisplayStatus}
                      >
                        {ssbDisplayStatus}
                      </Typography>
                    )}
                    {hasSsbCalibrationDownload && (
                      <Button
                        size="small"
                        onClick={downloadSsbCalibration}
                        disabled={ssbComputeBusy}
                        sx={{ ...compactButton, color: themeColors.accent, ml: "auto" }}
                      >
                        Download JSON
                      </Button>
                    )}
                  </Stack>
                  <Box
                    sx={{
                      display: "grid",
                      gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                      gap: "4px 12px",
                      "@media (max-width: 700px)": {
                        gridTemplateColumns: "1fr",
                      },
                    }}
                  >
                    {([
                      {
                        key: "c10",
                        label: "C10",
                        unit: "nm",
                        value: Number(ssbComputeC10Nm ?? 0),
                        min: -ssbC10Limit,
                        max: ssbC10Limit,
                        step: 1,
                        precision: 0,
                        setValue: setSsbComputeC10Nm,
                        schedule: (value: number) => scheduleSsbTuneCommit({ c10Nm: value }),
                        commit: (value: number) => commitSsbTuneNow({ c10Nm: value }),
                        title: "Defocus in nanometers. Release the slider to reconstruct.",
                      },
                      {
                        key: "c12",
                        label: "C12",
                        unit: "nm",
                        value: Number(ssbComputeC12Nm ?? 0),
                        min: 0,
                        max: ssbC12Limit,
                        step: 1,
                        precision: 0,
                        setValue: setSsbComputeC12Nm,
                        schedule: (value: number) => scheduleSsbTuneCommit({ c12Nm: value }),
                        commit: (value: number) => commitSsbTuneNow({ c12Nm: value }),
                        title: "Two-fold astigmatism magnitude in nanometers. Release the slider to reconstruct.",
                      },
                      {
                        key: "phi12",
                        label: "φ12",
                        unit: "°",
                        value: Number(ssbComputePhi12Deg ?? 0),
                        min: -180,
                        max: 180,
                        step: 1,
                        precision: 0,
                        setValue: setSsbComputePhi12Deg,
                        schedule: (value: number) => scheduleSsbTuneCommit({ phi12Deg: value }),
                        commit: (value: number) => commitSsbTuneNow({ phi12Deg: value }),
                        title: "Two-fold astigmatism angle. Release the slider to reconstruct.",
                      },
                      {
                        key: "rotation",
                        label: "Rotation",
                        unit: "°",
                        value: Number(ssbComputeRotationDeg ?? 0),
                        min: -180,
                        max: 180,
                        step: 0.1,
                        precision: 1,
                        setValue: setSsbComputeRotationDeg,
                        schedule: (value: number) => scheduleSsbTuneCommit({ rotationDeg: value }),
                        commit: (value: number) => commitSsbTuneNow({ rotationDeg: value }),
                        title: "Scan-detector rotation angle. Release the slider to reconstruct.",
                      },
                    ] as const).map((cfg) => (
                      <Box key={cfg.key} sx={{ minWidth: 0 }}>
                        <Tooltip title={cfg.title} placement="top" arrow>
                          <Typography sx={{ ...typo.label, color: themeColors.textMuted, cursor: "help", mb: -0.5 }}>
                            {cfg.label} <Box component="span" sx={{ color: themeColors.accent }}>{cfg.value.toFixed(cfg.precision)}</Box> {cfg.unit}
                          </Typography>
                        </Tooltip>
                        <Slider
                          value={cfg.value}
                          min={cfg.min}
                          max={cfg.max}
                          step={cfg.step}
                          disabled={ssbComputeBusy}
                          onChange={(_, value) => {
                            const next = Number(Array.isArray(value) ? value[0] : value);
                            cfg.setValue(next);
                            cfg.schedule(next);
                          }}
                          onChangeCommitted={(_, value) => cfg.commit(Number(Array.isArray(value) ? value[0] : value))}
                          size="small"
                          valueLabelDisplay="auto"
                          valueLabelFormat={(value) => `${Number(value).toFixed(cfg.precision)} ${cfg.unit}`}
                          sx={ssbTuneSliderSx}
                        />
                      </Box>
                    ))}
                  </Box>
                  </Box>
                  )}
                </Box>
              )}
            </>
          )}
        </Box>

        {/* THIRD COLUMN: FFT Panel (conditionally shown) */}
        {effectiveShowFft && (
          <Box sx={{ width: viPanelWidth, maxWidth: "100%" }}>
            {/* FFT Header */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={panelHeaderSx}>
              <Typography variant="caption" sx={{ ...typo.label, color: roiFftActive && fftCropDims ? accentGreen : themeColors.textMuted }}>{roiFftActive && fftCropDims ? `ROI FFT (${fftCropDims.cropWidth}\u00D7${fftCropDims.cropHeight})` : "FFT"}</Typography>
              {controlsVisible && <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
                <Button size="small" sx={compactButton} disabled={fftZoom === 1 && fftPanX === 0 && fftPanY === 0} onClick={() => { setFftZoom(1); setFftPanX(0); setFftPanY(0); }}>Reset</Button>
              </Stack>}
            </Stack>

            {/* FFT Canvas */}
            <Box sx={{ ...container.imageBox, width: "100%", maxWidth: viCanvasWidth, aspectRatio: `${shapeCols} / ${shapeRows}`, height: "auto", touchAction: "none", ...mobileImageBoxSx }}>
              <canvas data-quantem-scientific-output="show4dstem-fft" ref={fftCanvasRef} width={shapeCols} height={shapeRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
              <canvas
                ref={fftOverlayRef} width={shapeCols} height={shapeRows}
                onMouseDown={handleFftMouseDown} onMouseMove={handleFftMouseMove}
                onMouseUp={handleFftMouseUp} onMouseLeave={handleFftMouseLeave}
                onWheel={createZoomHandler(setFftZoom, setFftPanX, setFftPanY, fftViewRef, fftOverlayRef)}
                onDoubleClick={handleFftDoubleClick}
                onTouchStart={handlePanelTouchStart("fft")}
                onTouchMove={handlePanelTouchMove("fft")}
                onTouchEnd={handlePanelTouchEnd}
                onTouchCancel={handlePanelTouchEnd}
                style={{ position: "absolute", width: "100%", height: "100%", touchAction: "none", cursor: isDraggingFFT ? "grabbing" : "grab" }}
              />
              {fftPanelLoading && renderPanelLoadingOverlay("Loading FFT")}
              {panelChromeVisible && <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />}
            </Box>

            {/* FFT Stats Bar */}
            {showStats && !fftPanelLoading && fftStats && fftStats.length === 4 && (
              <Box sx={statsBarSx}>
                <Typography sx={statsTextSx}>Mean <Box component="span" sx={statsValueSx}>{formatStat(fftStats[0])}</Box></Typography>
                <Typography sx={statsTextSx}>Min <Box component="span" sx={statsValueSx}>{formatStat(fftStats[1])}</Box></Typography>
                <Typography sx={statsTextSx}>Max <Box component="span" sx={statsValueSx}>{formatStat(fftStats[2])}</Box></Typography>
                <Typography sx={statsTextSx}>Std <Box component="span" sx={statsValueSx}>{formatStat(fftStats[3])}</Box></Typography>
              </Box>
            )}

            {/* FFT D-spacing readout */}
            {fftClickInfo && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap", maxWidth: "100%", boxSizing: "border-box" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                  Spot <Box component="span" sx={{ color: themeColors.accent }}>({fftClickInfo.row.toFixed(1)}, {fftClickInfo.col.toFixed(1)})</Box>
                </Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                  dist <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box>
                </Typography>
                {fftClickInfo.dSpacing != null && (
                  <Typography sx={{ fontSize: 11, fontWeight: "bold", color: themeColors.accent }}>
                    d = {fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} \u00C5`}
                  </Typography>
                )}
                {fftClickInfo.spatialFreq != null && (
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                    q = <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.spatialFreq.toFixed(4)} {"\u00C5\u207B\u00B9"}</Box>
                  </Typography>
                )}
              </Box>
            )}

            {/* FFT Controls - Two rows with histogram on right */}
            {controlsVisible && (
              <>
                <Button
                  size="small"
                  onClick={() => setMobileFftOptionsOpen(v => !v)}
                  sx={mobileOptionToggleSx}
                  endIcon={mobileFftOptionsOpen ? <KeyboardArrowUpIcon fontSize="small" /> : <KeyboardArrowDownIcon fontSize="small" />}
                >
                  <Box component="span">FFT options</Box>
                  <Box component="span" sx={mobileOptionSummarySx}>{fftOptionSummary}</Box>
                </Button>
                <Box sx={mobileOptionsPanelSx(mobileFftOptionsOpen)}>
                  <Box sx={mobileOptionsContentSx}>
                    {/* Left: Two rows of controls */}
                    <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: "1 1 220px", minWidth: 0, justifyContent: "center" }}>
                      {/* Row 1: Scale + Clip */}
                      <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                        <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale</Typography>
                        <Select value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                          <MenuItem value="linear">Lin</MenuItem>
                          <MenuItem value="log">Log</MenuItem>

                        </Select>
                        <Typography sx={{ ...typo.label, fontSize: 10 }}>Auto</Typography>
                        <Switch checked={fftAuto} onChange={(e) => toggleFftAuto(e.target.checked)} size="small" sx={switchStyles.small} />
                        {fftCropDims && (
                          <>
                            <Typography sx={{ ...typo.label, fontSize: 10 }}>Win</Typography>
                            <Switch checked={fftWindow} onChange={(e) => setFftWindow(e.target.checked)} size="small" sx={switchStyles.small} />
                          </>
                        )}
                      </Box>
                      {/* Row 2: Color */}
                      <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                        <Typography sx={{ ...typo.label, fontSize: 10 }}>Color</Typography>
                        <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                          <MenuItem value="inferno">Inferno</MenuItem>
                          <MenuItem value="viridis">Viridis</MenuItem>
                          <MenuItem value="plasma">Plasma</MenuItem>
                          <MenuItem value="magma">Magma</MenuItem>
                          <MenuItem value="hot">Hot</MenuItem>
                          <MenuItem value="gray">Gray</MenuItem>
                        </Select>
                      </Box>
                    </Box>
                    {/* Right: Histogram spanning both rows */}
                    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", justifyContent: "center", flex: "0 0 auto", maxWidth: "100%" }}>
                      {fftHistogramData && (
                        <Histogram data={fftHistogramData} vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={fftDataMin} dataMax={fftDataMax} />
                      )}
                    </Box>
                  </Box>
                </Box>
              </>
            )}
          </Box>
        )}
      </Stack>

      {/* BOTTOM CONTROLS */}

      {/* Frame controls (5D time/tilt series) — matches Show3D playback */}
      {controlsVisible && nFrames > 1 && (<>
        <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
          <Typography sx={{ ...typo.label, fontSize: 10, flexShrink: 0 }}>View</Typography>
          <Select
            value={displayViewMode}
            onChange={(e) => setViewMode(String(e.target.value))}
            size="small"
            inputProps={{ "aria-label": "Show4DSTEM view mode" }}
            sx={{ ...themedSelect, minWidth: 82, fontSize: 10 }}
            MenuProps={themedMenuProps}
          >
            <MenuItem value="single">Single</MenuItem>
            <MenuItem value="multiple">Multiple</MenuItem>
          </Select>
          {compareMode && (
            <>
              <Typography sx={{ ...typo.label, fontSize: 10, flexShrink: 0 }}>DP</Typography>
              <Select
                value={compareDpMode || "average"}
                onChange={(e) => setCompareDpMode(String(e.target.value))}
                size="small"
                inputProps={{ "aria-label": "Show4DSTEM multiple DP source" }}
                sx={{ ...themedSelect, minWidth: 82, fontSize: 10 }}
                MenuProps={themedMenuProps}
              >
                <MenuItem value="average">Average</MenuItem>
                <MenuItem value="selected">Selected</MenuItem>
              </Select>
              <Tooltip title={compareAllGroups ? "Switch to Paged to reorder panels" : compareReorderMode ? "Finish reordering" : "Reorder multiple panels"}>
                <IconButton
                  size="small"
                  aria-label="Show4DSTEM multiple reorder"
                  className="show4dstem-compare-reorder"
                  disabled={compareAllGroups}
                  onClick={() => {
                    setCompareReorderMode((value) => !value);
                    setComparePendingMoveFrame(null);
                    setCompareDraggingFrame(null);
                  }}
                  sx={{ color: compareAllGroups ? themeColors.textMuted : compareReorderMode ? themeColors.accent : themeColors.textMuted, p: 0.25 }}
                >
                  <DragIndicatorIcon sx={{ fontSize: 17 }} />
                </IconButton>
              </Tooltip>
              <Button
                size="small"
                sx={compactButton}
                className="show4dstem-compare-reset"
                disabled={
                  !(comparePanelOrder || []).length
                  && !(compareHiddenPanels || []).length
                  && !(compareStarredPanels || []).length
                  && !compareAllGroups
                  && activeComparePageIdx === 0
                }
                onClick={resetComparePanelState}
              >
                Reset
              </Button>
            </>
          )}
          <Typography sx={{ ...typo.label, fontSize: 10, flexShrink: 0 }}>{frameSliderLabel}:</Typography>
          <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
            <IconButton size="small" aria-label="Show4DSTEM play frames backward" onClick={() => { setFrameReverse(true); setFramePlaying(true); }} sx={{ color: frameReverse && framePlaying ? themeColors.accent : themeColors.textMuted, p: 0.25 }}>
              <FastRewindIcon sx={{ fontSize: 18 }} />
            </IconButton>
            <IconButton size="small" aria-label={framePlaying ? "Show4DSTEM pause frames" : "Show4DSTEM play frames"} onClick={() => setFramePlaying(!framePlaying)} sx={{ color: themeColors.accent, p: 0.25 }}>
              {framePlaying ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
            </IconButton>
            <IconButton size="small" aria-label="Show4DSTEM play frames forward" onClick={() => { setFrameReverse(false); setFramePlaying(true); }} sx={{ color: !frameReverse && framePlaying ? themeColors.accent : themeColors.textMuted, p: 0.25 }}>
              <FastForwardIcon sx={{ fontSize: 18 }} />
            </IconButton>
            <IconButton size="small" aria-label="Show4DSTEM stop frames" onClick={() => { setFramePlaying(false); setFrameIdx(0); }} sx={{ color: themeColors.textMuted, p: 0.25 }}>
              <StopIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Stack>
          <Slider value={frameIdx} onChange={(_, v) => { setFramePlaying(false); setFrameIdx(v as number); }} min={0} max={Math.max(0, nFrames - 1)} size="small" aria-label={frameSliderAriaLabel} sx={{ flex: 1, minWidth: 60, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
          <Typography sx={{ ...typo.value, minWidth: 50, textAlign: "right", flexShrink: 0 }}>{frameLabels && frameLabels.length > frameIdx ? frameLabels[frameIdx] : `${frameIdx + 1}/${nFrames}`}</Typography>
        </Box>
        <Box sx={{ ...controlRow, mt: `${SPACING.XS}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
          <Typography sx={{ ...typo.label, fontSize: 10, color: themeColors.textMuted, flexShrink: 0 }}>fps</Typography>
          <Slider value={frameFps} min={1} max={30} step={1} onChange={(_, v) => setFrameFps(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35, flexShrink: 0 }} />
          <Typography sx={{ ...typo.label, fontSize: 10, color: themeColors.textMuted, minWidth: 14, flexShrink: 0 }}>{Math.round(frameFps)}</Typography>
          <Typography sx={{ ...typo.label, fontSize: 10, color: themeColors.textMuted, flexShrink: 0 }}>Loop</Typography>
          <Switch size="small" checked={frameLoop} onChange={() => setFrameLoop(!frameLoop)} sx={{ ...switchStyles.small, flexShrink: 0 }} />
          <Typography sx={{ ...typo.label, fontSize: 10, color: themeColors.textMuted, flexShrink: 0 }}>Bounce</Typography>
          <Switch size="small" checked={frameBoomerang} onChange={() => setFrameBoomerang(!frameBoomerang)} sx={{ ...switchStyles.small, flexShrink: 0 }} />
        </Box>
      </>)}

      {/* Path animation slider */}
      {controlsVisible && pathLength > 0 && (
        <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
          <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
            <IconButton size="small" onClick={() => setPathPlaying(!pathPlaying)} sx={{ color: themeColors.accent, p: 0.25 }}>
              {pathPlaying ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
            </IconButton>
            <IconButton size="small" onClick={() => { setPathPlaying(false); setPathIndex(0); }} sx={{ color: themeColors.textMuted, p: 0.25 }}>
              <StopIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Stack>
          <Slider value={pathIndex} onChange={(_, v) => { setPathPlaying(false); setPathIndex(v as number); }} min={0} max={Math.max(0, pathLength - 1)} size="small" sx={{ flex: 1, minWidth: 60, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
          <Typography sx={{ ...typo.value, minWidth: 50, textAlign: "right", flexShrink: 0 }}>{pathIndex + 1}/{pathLength}</Typography>
          <Typography sx={{ ...typo.label, fontSize: 10 }}>Loop</Typography>
          <Switch checked={pathLoop} onChange={(_, v) => { model.set("path_loop", v); model.save_changes(); }} size="small" sx={switchStyles.small} />
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(Show4DSTEM);
