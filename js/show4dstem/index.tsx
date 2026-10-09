/// <reference types="@webgpu/types" />
import { captureGpuCanvas } from "./captureGpuCanvas";
import { createLatestFrameQueue } from "./latestFrameQueue";
import { createDpPointerOwner } from "./dpPointerOwner";
import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import { readSettledCompareHistogram } from "./settledHistogram";
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
import { COLORMAPS, GPUColormapEngine, applyColormap } from "../display/colormaps";
import { DisplayFFT, getDisplayFFT, fft2dAsync, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2, applyHannWindow2D, reciprocalCoordinatesFromShiftedOffset } from "../display/fft";
import { findFFTPeakBrowser, sampleLineProfileBrowser } from "../display/geometry";
import {
  buildDetectorMask,
  buildFullDetectorMask,
  buildScanMask,
  DetectorCompute,
} from "../.generated/engine/detector/webgpu/backend";
import { readH5MasterInfo, readH5Volume } from "../.generated/engine/io/hdf5/webgpu/h5reader";
import { decodeBslz4Batch, type Bslz4Spec } from "../.generated/engine/io/hdf5/webgpu/bslz4";
import {
  collectShow4DSTEMLocalH5Files,
  loadShow4DSTEMLocalH5MaskedSum,
  loadShow4DSTEMLocalH5Master,
  setShow4DSTEMLocalFiles,
  show4DSTEMHasLocalFiles,
} from "../.generated/engine/io/hdf5/webgpu/local-h5";
import { AllDiffractionGrid } from "./AllDiffractionGrid";
import { latestRegion } from "./latestRegion";
import { drawScaleBarHiDPI, drawColorbar, roundToNiceValue } from "../figure";
import { findDataRange, sliderRange, computeStats, percentileClip } from "../display/stats";
import { downloadBlob, extractBytes, formatNumber, preserveRestoredWidgetModelsOnSave } from "../format";
import { useHideStaticFallback } from "../staticFallback";
import { MetadataSection } from "../widgetInfo";
import { FolderWatchBadge, useFolderWatchModelLive } from "../folderWatchStatus";
import { InfoTooltip } from "../shared/InfoTooltip";
import { RenderPathBadge } from "../shared/RenderPathBadge";
import { KeyboardShortcuts } from "../shared/KeyboardShortcuts";
import { HTML_EXPORT_OVERHEAD_BYTES, exportTitleSlug, formatSavedBytes, isAbortLikeError } from "../shared/exportFormat";
import { pointToSegmentDistance } from "../shared/geometry";
import { Histogram } from "../shared/Histogram";
import {
  canvasToMask,
  clampDetectorCenter,
  grabDetector,
  maskToCanvas,
  resizeDetectorFromPointer,
  type DetectorGrab,
  type DetectorResizeStart,
  type DetectorRoiMode,
} from "./detectorInteraction";

// The exported page reduces the 4D counts on WebGPU (no JavaScript reducer);
// without an adapter the saved views stay visible and the page says why.
const OFFLINE_NEEDS_WEBGPU =
  "Needs WebGPU: this page computes virtual images and diffraction patterns from the 4D data on WebGPU, " +
  "which this browser does not provide. The images shown were saved at export and do not follow the detector. " +
  "Open the page in a browser with WebGPU (Chrome, Edge or Safari 26) to explore.";

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

// DPC products are signed center-of-mass shifts: they share the WebGPU DPC
// path and display on a range symmetric about zero.
function isDpcGpuSource(source: string): source is DpcGpuSource {
  return source === "DPC_row" || source === "DPC_col" || source === "iDPC";
}

type Show4DSTEMModel = ReturnType<typeof useModel>;

type ViProductMaps = { bytes: DataView; productIndex: number; frames: number; pixels: number };

/**
 * Locate the precomputed product maps (DPC, iDPC, SSB) Python sent for one VI
 * source: vi_product_maps_bytes holds float32 [label, frame, scan row, scan col].
 * Null when that source has no map, so the caller falls back to the ROI image.
 */
function viProductMaps(model: Show4DSTEMModel, source: string, scanRows: number, scanCols: number): ViProductMaps | null {
  const labels = Array.isArray(model.get("vi_product_labels")) ? model.get("vi_product_labels") as string[] : [];
  const productIndex = labels.indexOf(source);
  if (productIndex < 0) return null;
  const bytes = model.get("vi_product_maps_bytes") as DataView | undefined;
  if (!bytes || bytes.byteLength === 0) return null;
  const frames = Math.max(1, Math.round(Number(model.get("vi_product_map_frames") || 1)));
  return { bytes, productIndex, frames, pixels: Math.max(1, scanRows * scanCols) };
}

/** Byte offset of one frame's map (frame clamped to the stored frames), or null when the payload is too short. */
function viProductMapOffset(maps: ViProductMaps, frame: number): number | null {
  const clampedFrame = maps.frames <= 1 ? 0 : Math.max(0, Math.min(maps.frames - 1, frame));
  const start = ((maps.productIndex * maps.frames + clampedFrame) * maps.pixels) * 4;
  return start + maps.pixels * 4 > maps.bytes.byteLength ? null : start;
}

/** The current frame's product map for the active (or given) VI source, as a view into the shared payload. */
function viProductFrameView(
  model: Show4DSTEMModel,
  scanRows: number,
  scanCols: number,
  sourceOverride?: string,
): DataView | null {
  const source = normaliseViSource(sourceOverride ?? model.get("vi_source"));
  if (source === "roi") return null;
  const maps = viProductMaps(model, source, scanRows, scanCols);
  if (!maps) return null;
  const start = viProductMapOffset(maps, Math.round(Number(model.get("frame_idx") || 0)));
  if (start === null) return null;
  return new DataView(maps.bytes.buffer, maps.bytes.byteOffset + start, maps.pixels * 4);
}

/** The active source's product maps for several frames, stacked in panel order for the compare grid. */
function viProductStackForIndices(
  model: Show4DSTEMModel,
  indices: number[],
  scanRows: number,
  scanCols: number,
): DataView | null {
  const source = normaliseViSource(model.get("vi_source"));
  if (source === "roi" || indices.length === 0) return null;
  const maps = viProductMaps(model, source, scanRows, scanCols);
  if (!maps) return null;
  const stack = new Float32Array(indices.length * maps.pixels);
  for (let slot = 0; slot < indices.length; slot++) {
    const start = viProductMapOffset(maps, Math.round(Number(indices[slot]) || 0));
    if (start === null) return null;
    stack.set(new Float32Array(maps.bytes.buffer, maps.bytes.byteOffset + start, maps.pixels), slot * maps.pixels);
  }
  return new DataView(stack.buffer);
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
  const slug = exportTitleSlug(title, "show4dstem");
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

// Control row style: bordered container per row.
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


// Search radius (FFT pixels) that snaps a click to the nearest Bragg spot,
// refined to a sub-pixel centroid.
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
    const textWidth = ctx.measureText(label).width;
    const labelX = Math.min(cssWidth - textWidth - 4, screenX + crosshairSize + 4);
    const labelY = Math.max(13, screenY - 4);
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillRect(labelX - 2, labelY - 12, textWidth + 4, 13);
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

  // Scan-ROI center to CSS pixels, at the center of the scan pixel it selects (row→screenY, col→screenX)
  const screenX = maskToCanvas(centerCol, zoom, panX) * scaleX;
  const screenY = maskToCanvas(centerRow, zoom, panY) * scaleY;

  const lineWidth = 2.5;
  const crosshairSize = 10;
  const handleRadius = 6;

  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  // Purple handles keep the scan ROI distinct from the green detector ROI.
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

  // Detector coordinates to CSS pixels, at the center of the pixel the mask selects
  const screenX = maskToCanvas(kCol, zoom, panX) * scaleX;
  const screenY = maskToCanvas(kRow, zoom, panY) * scaleY;
  
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
  
  ctx.beginPath();
  ctx.moveTo(screenX - crosshairSize, screenY);
  ctx.lineTo(screenX + crosshairSize, screenY);
  ctx.moveTo(screenX, screenY - crosshairSize);
  ctx.lineTo(screenX, screenY + crosshairSize);
  ctx.stroke();
  
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

  // Detector coordinates to CSS pixels: the mask centers pixel i at i, the image draws it over [i, i + 1)
  const screenX = maskToCanvas(centerCol, zoom, panX) * scaleX;
  const screenY = maskToCanvas(centerRow, zoom, panY) * scaleY;
  
  // Fixed UI sizes in CSS pixels
  const lineWidth = 2.5;
  const crosshairSizeSmall = 10;
  const handleRadius = 6;
  
  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
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

    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusOuterX, screenRadiusOuterY, 0, 0, 2 * Math.PI);
    ctx.stroke();

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

/**
 * Crop the scan-ROI bounding box out of a float32 virtual image for the
 * ROI-scoped FFT. A circle ROI zeroes the pixels outside its radius so the
 * spectrum covers only the drawn region; square and rect keep the whole box.
 * Returns null when the crop is smaller than 2x2.
 */
function cropSingleROI(
  data: Float32Array, imageWidth: number, imageHeight: number,
  mode: string, centerRow: number, centerCol: number,
  radius: number, roiWidth: number, roiHeight: number,
): { cropped: Float32Array; cropW: number; cropH: number } | null {
  if (mode === "off") return null;
  let colStart: number, rowStart: number, colEnd: number, rowEnd: number;

  if (mode === "rect") {
    const halfWidth = roiWidth / 2, halfHeight = roiHeight / 2;
    colStart = Math.max(0, Math.floor(centerCol - halfWidth));
    rowStart = Math.max(0, Math.floor(centerRow - halfHeight));
    colEnd = Math.min(imageWidth, Math.ceil(centerCol + halfWidth));
    rowEnd = Math.min(imageHeight, Math.ceil(centerRow + halfHeight));
  } else {
    colStart = Math.max(0, Math.floor(centerCol - radius));
    rowStart = Math.max(0, Math.floor(centerRow - radius));
    colEnd = Math.min(imageWidth, Math.ceil(centerCol + radius));
    rowEnd = Math.min(imageHeight, Math.ceil(centerRow + radius));
  }

  const cropW = colEnd - colStart, cropH = rowEnd - rowStart;
  if (cropW < 2 || cropH < 2) return null;

  const cropped = new Float32Array(cropW * cropH);
  if (mode === "circle") {
    const radiusSq = radius * radius;
    for (let cropRow = 0; cropRow < cropH; cropRow++) {
      for (let cropCol = 0; cropCol < cropW; cropCol++) {
        const col = colStart + cropCol, row = rowStart + cropRow;
        const distSq = (col - centerCol) * (col - centerCol) + (row - centerRow) * (row - centerRow);
        cropped[cropRow * cropW + cropCol] = distSq <= radiusSq ? data[row * imageWidth + col] : 0;
      }
    }
  } else {
    for (let cropRow = 0; cropRow < cropH; cropRow++) {
      const sourceOffset = (rowStart + cropRow) * imageWidth + colStart;
      cropped.set(data.subarray(sourceOffset, sourceOffset + cropW), cropRow * cropW);
    }
  }
  return { cropped, cropW, cropH };
}

/**
 * Copy a rows x cols image into the top-left corner of a zeroed padRows x
 * padCols buffer. fft2d needs power-of-two sides and would otherwise truncate
 * the frequency content of the image.
 */
function zeroPadImage(image: Float32Array, rows: number, cols: number, padRows: number, padCols: number): Float32Array {
  const padded = new Float32Array(padRows * padCols);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      padded[row * padCols + col] = image[row * cols + col];
    }
  }
  return padded;
}

/**
 * Dispose the oldest resident detector volumes until at most `limit` remain,
 * never the one just loaded nor the one on screen. Map order is load order, so
 * the earliest-loaded other volume goes first; this bounds GPU memory when
 * scrubbing through more datasets than fit at once.
 */
function evictOldVolumes(cache: Map<number, DetectorCompute>, limit: number, loaded: number, active: number) {
  while (cache.size > limit) {
    const oldest = [...cache.keys()].find((index) => index !== loaded && index !== active);
    if (oldest === undefined) break;
    cache.get(oldest)!.dispose();
    cache.delete(oldest);
  }
}

/** A profile-line drag: the pointer at the press and both endpoints at that moment. */
type ProfileLineDrag = { row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } };

/**
 * Move both profile endpoints by the pointer's offset from the press, clamped
 * so the whole line stays inside a rows x cols image; otherwise a drag past the
 * edge would push an endpoint off the data the sampler reads.
 */
function dragProfileLine(drag: ProfileLineDrag, pointerRow: number, pointerCol: number, rows: number, cols: number) {
  let deltaRow = pointerRow - drag.row;
  let deltaCol = pointerCol - drag.col;
  const minRow = Math.min(drag.p0.row, drag.p1.row);
  const maxRow = Math.max(drag.p0.row, drag.p1.row);
  const minCol = Math.min(drag.p0.col, drag.p1.col);
  const maxCol = Math.max(drag.p0.col, drag.p1.col);
  deltaRow = Math.max(deltaRow, -minRow);
  deltaRow = Math.min(deltaRow, (rows - 1) - maxRow);
  deltaCol = Math.max(deltaCol, -minCol);
  deltaCol = Math.min(deltaCol, (cols - 1) - maxCol);
  return [
    { row: drag.p0.row + deltaRow, col: drag.p0.col + deltaCol },
    { row: drag.p1.row + deltaRow, col: drag.p1.col + deltaCol },
  ];
}

type ProfileLayout = {
  padLeft: number; plotW: number; padTop: number; plotH: number;
  valueMin: number; valueMax: number; totalDist: number; xUnit: string;
};

/**
 * Paint a line-profile sparkline (axes, curve, distance ticks, min/max labels)
 * on a HiDPI canvas. Returns the painted image and plot layout that the hover
 * readout restores and maps the pointer through, or null after painting the
 * empty-state hint when there are fewer than two samples. `distance` is the
 * line length with its unit; null labels the axis in samples. The DP and VI
 * profiles share it so both panels read the same.
 */
function drawProfileSparkline(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  cssW: number,
  cssH: number,
  data: Float32Array | null,
  distance: { total: number; unit: string } | null,
  emptyHint: string,
  isDark: boolean,
  accent: string,
): { base: ImageData; layout: ProfileLayout } | null {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.scale(dpr, dpr);

  ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
  ctx.fillRect(0, 0, cssW, cssH);

  if (!data || data.length < 2) {
    ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#555" : "#999";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(emptyHint, cssW / 2, cssH / 2);
    return null;
  }

  const padLeft = 40;
  const padRight = 8;
  const padTop = 6;
  const padBottom = 18;
  const plotW = cssW - padLeft - padRight;
  const plotH = cssH - padTop - padBottom;

  let valueMin = Infinity, valueMax = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < valueMin) valueMin = data[i];
    if (data[i] > valueMax) valueMax = data[i];
  }
  const range = valueMax - valueMin || 1;

  // X-axis: calibrated line length, or sample count when the caller has none
  const totalDist = distance ? distance.total : data.length - 1;
  const xUnit = distance ? distance.unit : "px";

  ctx.strokeStyle = isDark ? "#555" : "#bbb";
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop);
  ctx.lineTo(padLeft, padTop + plotH);
  ctx.lineTo(padLeft + plotW, padTop + plotH);
  ctx.stroke();

  ctx.strokeStyle = accent;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = padLeft + (i / (data.length - 1)) * plotW;
    const y = padTop + plotH - ((data[i] - valueMin) / range) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  const tickY = padTop + plotH;
  ctx.strokeStyle = isDark ? "#555" : "#bbb";
  ctx.lineWidth = 0.5;
  const idealTicks = Math.max(2, Math.floor(plotW / 70));
  const tickStep = roundToNiceValue(totalDist / idealTicks);
  ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.fillStyle = isDark ? "#888" : "#666";
  ctx.textBaseline = "top";
  const ticks: number[] = [];
  for (let tick = 0; tick <= totalDist + tickStep * 0.01; tick += tickStep) {
    if (tick > totalDist * 1.001) break;
    ticks.push(tick);
  }
  for (let i = 0; i < ticks.length; i++) {
    const tick = ticks[i];
    const frac = totalDist > 0 ? tick / totalDist : 0;
    const x = padLeft + frac * plotW;
    ctx.beginPath(); ctx.moveTo(x, tickY); ctx.lineTo(x, tickY + 3); ctx.stroke();
    ctx.textAlign = frac < 0.05 ? "left" : frac > 0.95 ? "right" : "center";
    const label = tick % 1 === 0 ? tick.toFixed(0) : tick.toFixed(1);
    ctx.fillText(i === ticks.length - 1 ? `${label} ${xUnit}` : label, x, tickY + 4);
  }

  // Y-axis min/max labels (left margin)
  ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.fillStyle = isDark ? "#888" : "#666";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(formatNumber(valueMax), 2, padTop);
  ctx.textBaseline = "bottom";
  ctx.fillText(formatNumber(valueMin), 2, padTop + plotH);

  return {
    base: ctx.getImageData(0, 0, canvas.width, canvas.height),
    layout: { padLeft, plotW, padTop, plotH, valueMin, valueMax, totalDist, xUnit },
  };
}

/**
 * Redraw a profile sparkline's base image, then the hover crosshair, the dot on
 * the curve and the value/distance readout at CSS x position `cssX`. Outside the
 * plot area only the base image is restored, which clears a stale readout.
 */
function drawProfileHover(
  ctx: CanvasRenderingContext2D,
  base: ImageData,
  layout: ProfileLayout,
  data: Float32Array,
  cssX: number,
  isDark: boolean,
  accent: string,
) {
  const { padLeft, plotW, padTop, plotH, valueMin, valueMax, totalDist, xUnit } = layout;
  const range = valueMax - valueMin || 1;

  ctx.putImageData(base, 0, 0);

  if (cssX < padLeft || cssX > padLeft + plotW) return;
  const frac = (cssX - padLeft) / plotW;

  const dpr = window.devicePixelRatio || 1;
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.strokeStyle = isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  ctx.beginPath();
  ctx.moveTo(cssX, padTop);
  ctx.lineTo(cssX, padTop + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  const sampleIndex = Math.min(data.length - 1, Math.max(0, Math.round(frac * (data.length - 1))));
  const value = data[sampleIndex];
  const y = padTop + plotH - ((value - valueMin) / range) * plotH;
  ctx.fillStyle = accent;
  ctx.beginPath();
  ctx.arc(cssX, y, 3, 0, Math.PI * 2);
  ctx.fill();

  const hoverDistance = frac * totalDist;
  const label = `${formatNumber(value)}  @  ${hoverDistance.toFixed(1)} ${xUnit}`;
  ctx.font = "bold 9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  const textWidth = ctx.measureText(label).width;
  const labelX = Math.min(cssX + 6, padLeft + plotW - textWidth - 2);
  const labelY = padTop + 2;
  ctx.fillStyle = isDark ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.8)";
  ctx.fillRect(labelX - 2, labelY - 1, textWidth + 4, 11);
  ctx.fillStyle = isDark ? "#fff" : "#000";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(label, labelX, labelY);

  ctx.restore();
}

/** Restore a profile sparkline's base image so the hover readout disappears when the pointer leaves. */
function restoreProfileBase(canvas: HTMLCanvasElement | null, base: ImageData | null) {
  if (!canvas || !base) return;
  canvas.getContext("2d")?.putImageData(base, 0, 0);
}

interface CompareVirtualGridProps {
  kind?: "virtual" | "diffraction";
  scanRegion?: {mode: string; row: number; col: number; radius: number; width: number; height: number};
  bytes: DataView | null | undefined;
  count: number;
  indices: number[];
  // GPU-resident panels: frame -> engine colormap slot, painted with a GPU range
  // through each tile's visible WebGPU canvas; bytes stay the settle/export fallback.
  gpuSlots?: Map<number, number> | null;
  gpuRanges?: Map<number, { min: number; max: number }> | null;
  gpuVersion?: number;
  gpuEngine?: GPUColormapEngine | null;
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
  onGpuRendererReady?: (renderNow: (() => number) | null) => void;
}

const CompareVirtualGrid = React.memo(function CompareVirtualGrid({
  kind = "virtual",
  scanRegion,
  bytes,
  count,
  indices,
  gpuSlots,
  gpuRanges,
  gpuVersion,
  gpuEngine,
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
  onGpuRendererReady,
}: CompareVirtualGridProps) {
  const positionRafRef = React.useRef(0);
  const panDragRef = React.useRef<{clientX:number; clientY:number; panX:number; panY:number} | null>(null);
  const pendingPositionRef = React.useRef<[number, number]>([0, 0]);
  React.useEffect(() => () => cancelAnimationFrame(positionRafRef.current), []);
  const batchModel = useModel();
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const gpuCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const gpuRenderGenerationRef = React.useRef(0);
  const canvasDrawCacheRef = React.useRef(new Map<number, {
    canvas: HTMLCanvasElement;
    panel: Float32Array;
    styleKey: string;
  }>());
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
    const raw = new Float32Array(bytes.buffer, bytes.byteOffset, panelPixels * count);
    return Array.from({ length: count }, (_, idx) => {
      const start = idx * panelPixels;
      return raw.subarray(start, start + panelPixels);
    });
  }, [bytes, count, panelPixels]);
  const sourceIndices = indices || [];
  const [previewIndices, setPreviewIndices] = React.useState<number[] | null>(null);
  const panelByFrame = React.useMemo(() => {
    const byFrame = new Map<number, Float32Array>();
    (indices || []).forEach((frame, index) => {
      const panel = panels[index];
      if (panel) byFrame.set(frame, panel);
    });
    return byFrame;
  }, [indices, panels]);
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
        gpuLoaded: Boolean(gpuSlots?.has(frame) && gpuEngine && gpuRanges?.has(frame)),
      }))
      .filter((entry) => entry.panel !== undefined || entry.gpuLoaded);
  }, [gpuEngine, gpuRanges, gpuSlots, gpuVersion, panelByFrame, renderIndices, scaleMode]);

  const renderGpuSlotsNow = React.useCallback((): number => {
    if (!gpuEngine || !gpuSlots) return 0;
    const lut = COLORMAPS[colormap] || COLORMAPS.inferno;
    gpuEngine.uploadLUT(colormap, lut);
    const generation = ++gpuRenderGenerationRef.current;
    const panels: {
      canvas: HTMLCanvasElement;
      range: { vmin: number; vmax: number };
      slot: number;
    }[] = [];
    renderEntries.forEach((entry, localIdx) => {
      const slot = gpuSlots.get(entry.frame);
      const rawRange = gpuRanges?.get(entry.frame);
      const canvas = gpuCanvasRefs.current[localIdx];
      if (slot === undefined || !rawRange || !canvas) return;
      const transformRangeValue = (value: number) => scaleMode === "log"
        ? (value >= 0 ? Math.log1p(value) : -Math.log1p(-value))
        : value;
      const rangeMin = transformRangeValue(rawRange.min);
      const rangeMax = transformRangeValue(rawRange.max);
      const span = Math.max(0, rangeMax - rangeMin);
      const displayRange = {
        vmin: rangeMin + span * Math.max(0, Math.min(100, vminPct)) / 100,
        vmax: rangeMin + span * Math.max(0, Math.min(100, vmaxPct)) / 100,
      };
      panels.push({ canvas, range: displayRange, slot });
    });
    if (!panels.length) return 0;
    void (async () => {
      const bitmap = await gpuEngine.renderPanelSlotsToImageBitmapAsync(
        panels.map((panel) => panel.slot),
        panels.map((panel) => panel.range),
        panels.map(() => scaleMode === "log"),
        {
          width: shapeCols * panels.length,
          height: shapeRows,
          panelCount: panels.length,
          cols: panels.length,
          rows: 1,
          gap: 0,
          bgRgb: 0,
          transforms: panels.map(() => ({
            zoom: 1,
            panX: 0,
            panY: 0,
          })),
          smooth,
        },
      );
      if (!bitmap || generation !== gpuRenderGenerationRef.current) {
        bitmap?.close();
        return;
      }
      panels.forEach((panel, index) => {
        if (!panel.canvas.isConnected) return;
        if (panel.canvas.width !== shapeCols) panel.canvas.width = shapeCols;
        if (panel.canvas.height !== shapeRows) panel.canvas.height = shapeRows;
        const context = panel.canvas.getContext("2d");
        if (!context) return;
        context.imageSmoothingEnabled = false;
        context.clearRect(0, 0, shapeCols, shapeRows);
        context.drawImage(
          bitmap,
          index * shapeCols,
          0,
          shapeCols,
          shapeRows,
          0,
          0,
          shapeCols,
          shapeRows,
        );
      });
      bitmap.close();
    })();
    return panels.length;
  }, [colormap, comparePanX, comparePanY, compareZoom, gpuEngine, gpuRanges, gpuSlots, renderEntries, scaleMode, shapeCols, shapeRows, smooth, vmaxPct, vminPct]);

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
  }, [autoContrast, colormap, gpuEngine, gpuSlots, gpuVersion, renderEntries, scaleMode, shapeCols, shapeRows, smooth, vmaxPct, vminPct]);

  React.useEffect(() => {
    if (!gpuEngine || !gpuSlots || kind === "diffraction") return;
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
      });
    })();
  }, [
    kind,
    colormap,
    gpuEngine,
    gpuSlots,
    renderEntries,
    scaleMode,
    shapeCols,
    shapeRows,
    smooth,
    vmaxPct,
    vminPct,
  ]);

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

  // Identity view for a new image shape, a double-click and the Reset button.
  const resetCompareView = React.useCallback(() => {
    const view = compareViewRef.current;
    view.zoom = 1;
    view.panX = 0;
    view.panY = 0;
    setCompareZoom(1);
    setComparePanX(0);
    setComparePanY(0);
  }, []);

  React.useEffect(() => {
    resetCompareView();
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
    resetCompareView();
  }, [resetCompareView]);

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
    const col = Math.round(Math.max(0, Math.min(shapeCols - 1, canvasToMask(tileX, compareZoom, comparePanX))));
    const row = Math.round(Math.max(0, Math.min(shapeRows - 1, canvasToMask(tileY, compareZoom, comparePanY))));
    pendingPositionRef.current = [row, col];
    if (commit) {
      cancelAnimationFrame(positionRafRef.current);
      positionRafRef.current = 0;
      onPositionChange(row, col, true);
    } else if (!positionRafRef.current) {
      positionRafRef.current = requestAnimationFrame(() => {
        positionRafRef.current = 0;
        onPositionChange(...pendingPositionRef.current, false);
      });
    }
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
    let raf = 0;
    const paint = () => {
    raf = 0;
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
      if (showScaleBar && cssWidth >= 96) {
        const unit = pixelSize > 0 ? pixelUnit || "px" : "px";
        const pxSize = pixelSize > 0 ? pixelSize : 1;
        drawScaleBarHiDPI(overlay, dpr, compareZoom, pxSize, unit, shapeCols);
      }
      drawViPositionMarker(
        overlay,
        dpr,
        kind === "diffraction" ? cursorRow : Number(batchModel.get("pos_row")),
        kind === "diffraction" ? cursorCol : Number(batchModel.get("pos_col")),
        compareZoom,
        comparePanX,
        comparePanY,
        shapeCols,
        shapeRows,
        isDraggingPosition,
        idx === 0 && cssWidth >= 96,
      );
      if (scanRegion && scanRegion.mode !== "off") {
        ctx?.clearRect(0, 0, overlay.width, overlay.height);
        if (showScaleBar) drawScaleBarHiDPI(overlay, dpr, compareZoom, pixelSize || 1, pixelUnit || "px", shapeCols);
        drawViRoiOverlayHiDPI(overlay, dpr, scanRegion.mode,
          scanRegion.row, scanRegion.col, scanRegion.radius, scanRegion.width, scanRegion.height,
          compareZoom, comparePanX, comparePanY, shapeCols, shapeRows, isDraggingPosition, false, false);
      }
    });
    };
    const changed = () => { if (!raf) raf = requestAnimationFrame(paint); };
    paint();
    batchModel.on("change:pos_row", changed); batchModel.on("change:pos_col", changed);
    return () => { if (raf) cancelAnimationFrame(raf); batchModel.off("change:pos_row", changed); batchModel.off("change:pos_col", changed); };
  }, [batchModel, kind, comparePanX, comparePanY, compareZoom, cursorCol, cursorRow, isDraggingPosition, overlayVersion, pixelSize, pixelUnit, renderEntries, shapeCols, shapeRows, showScaleBar, scanRegion]);

  const panelTiles = React.useMemo(() => renderEntries.map(({ frame, gpuLoaded }, localIdx) => {
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
              aria-label={`Show4DSTEM ${kind === "diffraction" ? "diffraction" : "multiple"} panel ${frame + 1}`}
              data-comparison-zoom={compareZoom}
              data-comparison-pan={`${comparePanX},${comparePanY}`}
              tabIndex={0}
              draggable={reorderMode}
              onDoubleClick={handleCompareDoubleClick}
              onPointerDown={(event) => {
                const target = event.target instanceof Element ? event.target : null;
                if (reorderMode || target?.closest("button")) return;
                try { event.currentTarget.setPointerCapture(event.pointerId); } catch {}
                if (event.shiftKey) {
                  panDragRef.current={clientX:event.clientX,clientY:event.clientY,panX:comparePanX,panY:comparePanY};
                  return;
                }
                isDraggingPositionRef.current = true;
                setIsDraggingPosition(true);
                updatePositionFromPointer(event.currentTarget, event.clientX, event.clientY);
                onSelect(frame);
              }}
              onPointerMove={(event) => {
                const pan = panDragRef.current;
                if (pan) {
                  event.preventDefault();
                  const rect=event.currentTarget.getBoundingClientRect();
                  const view=compareViewRef.current;
                  view.panX=pan.panX+(event.clientX-pan.clientX)*shapeCols/rect.width;
                  view.panY=pan.panY+(event.clientY-pan.clientY)*shapeRows/rect.height;
                  if (!view.raf) view.raf=requestAnimationFrame(() => {
                    view.raf=0;setComparePanX(view.panX);setComparePanY(view.panY);
                  });
                  return;
                }
                if (!isDraggingPositionRef.current || reorderMode) return;
                event.preventDefault();
                updatePositionFromPointer(event.currentTarget, event.clientX, event.clientY);
              }}
              onPointerUp={(event) => {
                if (panDragRef.current) {panDragRef.current=null;return;}
                if (!isDraggingPositionRef.current) return;
                updatePositionFromPointer(event.currentTarget, event.clientX, event.clientY, true);
                isDraggingPositionRef.current = false;
                setIsDraggingPosition(false);
              }}
              onPointerCancel={(event) => {
                if (panDragRef.current) {panDragRef.current=null;return;}
                if (!isDraggingPositionRef.current) return;
                updatePositionFromPointer(event.currentTarget, event.clientX, event.clientY, true);
                isDraggingPositionRef.current = false;
                setIsDraggingPosition(false);
              }}
              onClick={() => {
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
                if (!reorderMode) return;
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData("text/plain", String(frame));
                setPreviewIndices([...displayIndices]);
                onDragFrameChange(frame);
              }}
              onDragEnter={(event) => {
                if (!reorderMode || draggingFrame == null || draggingFrame === frame) return;
                event.preventDefault();
                movePreviewFrame(draggingFrame, frame);
              }}
              onDragOver={(event) => {
                if (!reorderMode) return;
                event.preventDefault();
                event.dataTransfer.dropEffect = "move";
              }}
              onDrop={(event) => {
                if (!reorderMode) return;
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
                containerType: "inline-size",
                border: "none",
                boxSizing: "border-box",
                outline: "none",
                cursor: reorderMode ? "grab" : "crosshair",
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
                  transition: "opacity 160ms ease",
                }}
              />
              <canvas
                data-quantem-scientific-output={kind === "diffraction" ? "show4dstem-comparison-diffraction" : undefined}
                ref={(node) => {
                  gpuCanvasRefs.current[localIdx] = node;
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
                  left: "min(28px, 10%)",
                  right: "min(28px, 10%)",
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
                  "& .show4dstem-full-tile-label": { display: "inline" },
                  "& .show4dstem-compact-tile-label": { display: "none" },
                  "@container (max-width: 96px)": {
                    top: 3,
                    fontSize: 9,
                    "& .show4dstem-full-tile-label": { display: "none" },
                    "& .show4dstem-compact-tile-label": { display: "inline" },
                  },
                }}
                title={label}
              >
                <span className="show4dstem-full-tile-label">{label}</span>
                <span className="show4dstem-compact-tile-label">{frame + 1}</span>
              </Box>
              {panelChromeVisible && reorderMode && (
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
              {panelChromeVisible && (
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
              {panelChromeVisible && (
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
        }), [
    renderEntries, activeIdx, labels, starred, draggingFrame,
    pendingMoveFrame, themeColors, reorderMode, handleCompareDoubleClick,
    updatePositionFromPointer, onSelect, onPendingMoveFrameChange, onReorderFrame,
    displayIndices, onDragFrameChange, movePreviewFrame, shapeCols, shapeRows,
    imageLeft, imageTop, imageWidth, imageHeight, smooth, panelChromeVisible,
    onToggleStar, onHide, onResizeStart, mobileGridCols, gridCols, resizeGripSx, kind, compareZoom, comparePanX, comparePanY,
  ]);

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
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, color: themeColors.textMuted, fontSize: 10 }}>
        <span>Scroll to zoom · Shift-drag to pan</span>
        <Button size="small" sx={{ fontSize: 10, minWidth: 0 }} onClick={resetCompareView}>Reset</Button>
        <span>{compareZoom.toFixed(1)}×</span>
      </Box>
      <Box
        sx={{
          position: "relative",
          display: "grid",
          gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))`,
          gap: `${gridGapPx}px`,
          maxWidth: "100%",
          "@media (max-width: 700px)": {
            gridTemplateColumns: `repeat(${mobileGridCols}, minmax(0, 1fr))`,
            gap: `${gridGapPx}px`,
          },
        }}
      >
        {panelTiles}
      </Box>
    </Box>
  );
});

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
  const [ssbComputeBfPixels] = useModelState<number>("ssb_compute_bf_pixels");
  const [ssbComputeC10Nm, setSsbComputeC10Nm] = useModelState<number>("ssb_compute_c10_nm");
  const [ssbComputeC12Nm, setSsbComputeC12Nm] = useModelState<number>("ssb_compute_c12_nm");
  const [ssbComputePhi12Deg, setSsbComputePhi12Deg] = useModelState<number>("ssb_compute_phi12_deg");
  const [ssbComputeRotationDeg, setSsbComputeRotationDeg] = useModelState<number>("ssb_compute_rotation_angle_deg");
  const [ssbComputeCalibrationJson] = useModelState<string>("ssb_compute_calibration_json");
  const [ssbComputeCalibrationFilename] = useModelState<string>("ssb_compute_calibration_filename");

  // ROI state
  const [roiRadius, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiRadiusInner, setRoiRadiusInner] = useModelState<number>("roi_radius_inner");
  const [roiMode, setRoiMode] = useModelState<string>("roi_mode");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");

  // Global min/max for DP normalization (from Python)
  const [dpGlobalMin] = useModelState<number>("dp_global_min");
  const [dpGlobalMax] = useModelState<number>("dp_global_max");

  // VI min/max (viDataMin/viDataMax) are derived JS-side from virtual_image_bytes below.
  // Keeping them out of Python traits avoids a comm-message ordering race where
  // bytes from click N arrive with min/max from click N-1.

  // Detector calibration (for presets)
  const [centerCol] = useModelState<number>("center_col");
  const [centerRow] = useModelState<number>("center_row");

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
  const [compareCols, setCompareCols] = useModelState<number>("compare_cols");
  const [compareVirtualImageBytes] = useModelState<DataView>("compare_virtual_image_bytes");
  const [comparePanelCount] = useModelState<number>("compare_panel_count");
  const [comparePanelIndices] = useModelState<number[]>("compare_panel_indices");
  const [compareStatus] = useModelState<string>("compare_status");
  const [compareDpMode, setCompareDpMode] = useModelState<string>("compare_dp_mode");
  const [compareDiffractionBytes] = useModelState<DataView>("compare_diffraction_bytes");
  const [compareDiffractionIndices] = useModelState<number[]>("compare_diffraction_indices");
  const [compareGroupMode, setCompareGroupMode] = useModelState<string>("compare_group_mode");
  const [comparePageIdx, setComparePageIdx] = useModelState<number>("compare_page_idx");
  const [comparePageCount] = useModelState<number>("compare_page_count");
  const [compareMaxPanels] = useModelState<number>("compare_max_panels");
  const [comparePanelOrder, setComparePanelOrder] = useModelState<number[]>("compare_panel_order");
  const [compareHiddenPanels, setCompareHiddenPanels] = useModelState<number[]>("compare_hidden_panels");
  const [compareStarredPanels, setCompareStarredPanels] = useModelState<number[]>("compare_starred_panels");

  React.useEffect(() => {
    try {
      model.send({
        type: "show4dstem_frontend_ready",
        version: 1,
      });
    } catch {
      // A closing notebook comm has no mounted UI left to initialize.
    }
  }, [model]);

  // Profile line state (synced with Python)
  const [profileLine, setProfileLine] = useModelState<{row: number; col: number}[]>("profile_line");
  const [profileWidth] = useModelState<number>("profile_width");

  // ─────────────────────────────────────────────────────────────────────────
  // Local State (UI-only, not synced to Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [localKCol, setLocalKCol] = React.useState(roiCenterCol);
  const [localKRow, setLocalKRow] = React.useState(roiCenterRow);
  const [localPosRow, setLocalPosRow] = React.useState(posRow);
  const [localPosCol, setLocalPosCol] = React.useState(posCol);
  const scanPositionPendingRef = React.useRef<[number, number] | null>(null);
  const scanPositionRafRef = React.useRef<number | null>(null);
  const scanPositionOptimisticRef = React.useRef<[number, number] | null>(null);
  const scanPositionCurrentRef = React.useRef<[number, number]>([Math.round(posRow), Math.round(posCol)]);
  const writeQueuedScanPosition = React.useCallback(() => {
    const pending = scanPositionPendingRef.current;
    if (!pending) return;
    const [row, col] = pending;
    scanPositionPendingRef.current = null;
    scanPositionCurrentRef.current = [row, col];
    setLocalPosRow(row);
    setLocalPosCol(col);
    model.set("pos_row", row);
    model.set("pos_col", col);
    model.save_changes();
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
    if (scanPositionRafRef.current === null) {
      scanPositionRafRef.current = requestAnimationFrame(() => {
        scanPositionRafRef.current = null;
        writeQueuedScanPosition();
      });
    }
  }, [writeQueuedScanPosition]);
  const flushScanPosition = React.useCallback(() => {
    if (scanPositionRafRef.current !== null) {
      cancelAnimationFrame(scanPositionRafRef.current);
      scanPositionRafRef.current = null;
    }
    writeQueuedScanPosition();
  }, [writeQueuedScanPosition]);
  React.useEffect(() => {
    return () => {
      if (scanPositionRafRef.current !== null) {
        cancelAnimationFrame(scanPositionRafRef.current);
        scanPositionRafRef.current = null;
      }
    };
  }, []);
  const [isDraggingDP, setIsDraggingDP] = React.useState(false);
  const dpPointerOwner = React.useMemo(() => createDpPointerOwner(), []);
  // rAF coalescing for ROI drag: collapse rapid mousemove events into ≤1
  // Python comm message per animation frame. Without this, drag fires 60+
  // events/sec at >100ms Python compute each → queue piles up → laggy UX.
  const roiCenterPendingRef = React.useRef<[number, number] | null>(null);
  const roiCenterRafRef = React.useRef<number | null>(null);
  const flushRoiCenter = React.useCallback((paint = true) => {
    if (roiCenterPendingRef.current) {
      const [row, col] = roiCenterPendingRef.current;
      writeRoiCenterModel(row, col);
      roiCenterPendingRef.current = null;
      setLocalKRow(row);
      setLocalKCol(col);
    }
    if (paint) roiCenterRafRef.current = null;
  }, [writeRoiCenterModel]);
  const queueRoiCenter = React.useCallback((row: number, col: number) => {
    roiCenterPendingRef.current = [row, col];
    if (roiCenterRafRef.current === null) {
      roiCenterRafRef.current = requestAnimationFrame(() => flushRoiCenter());
    }
  }, [flushRoiCenter]);
  // Radius drags keep the latest geometry in refs (read by the DP overlay and
  // the live recompute) and write the model at most once per animation frame.
  const roiRadiusPendingRef = React.useRef<number | null>(null);
  const roiRadiusInnerPendingRef = React.useRef<number | null>(null);
  const roiRadiusRafRef = React.useRef<number | null>(null);
  const flushRoiRadius = React.useCallback((paint = true) => {
    if (paint && roiRadiusRafRef.current !== null) cancelAnimationFrame(roiRadiusRafRef.current);
    const updates: Record<string, number> = {};
    if (roiRadiusPendingRef.current !== null) updates.roi_radius = roiRadiusPendingRef.current;
    if (roiRadiusInnerPendingRef.current !== null) updates.roi_radius_inner = roiRadiusInnerPendingRef.current;
    roiRadiusPendingRef.current = null;
    roiRadiusInnerPendingRef.current = null;
    if (Object.keys(updates).length) {
      for (const [name, radius] of Object.entries(updates)) model.set(name, radius);
      model.save_changes();
    }
    if (paint) roiRadiusRafRef.current = null;
  }, [model]);
  const sendRoiRadius = React.useCallback((radius: number, boundary: "inner" | "outer" = "outer") => {
    if (boundary === "inner") roiRadiusInnerPendingRef.current = radius;
    else roiRadiusPendingRef.current = radius;
    if (roiRadiusRafRef.current === null) roiRadiusRafRef.current = requestAnimationFrame(() => flushRoiRadius());
  }, [flushRoiRadius]);
  React.useEffect(() => () => {
    if (roiRadiusRafRef.current !== null) cancelAnimationFrame(roiRadiusRafRef.current);
    roiRadiusPendingRef.current = null;
    roiRadiusInnerPendingRef.current = null;
  }, []);
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
  const resizeStartRef = React.useRef<DetectorResizeStart | null>(null);
  // VI ROI drag/resize states (same pattern as DP)
  const [isDraggingViRoi, setIsDraggingViRoi] = React.useState(false);
  const [isDraggingViRoiResize, setIsDraggingViRoiResize] = React.useState(false);
  const [isHoveringViRoiResize, setIsHoveringViRoiResize] = React.useState(false);
  const viResizeStartRef = React.useRef<DetectorResizeStart | null>(null);
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
  // DP doesn't need them: Bragg spots read best with the slider's percentile
  // range and nearest-neighbor blit.
  const [viAutoContrast, setViAutoContrast] = useModelState<boolean>("vi_auto_contrast");
  const [viSmooth, setViSmooth] = useModelState<boolean>("vi_smooth");
  const viPreAutoPctRef = React.useRef<[number, number] | null>(null);
  const toggleViAutoContrast = React.useCallback((on: boolean) => {
    if (on) {
      viPreAutoPctRef.current = [viVminPct, viVmaxPct];
    } else if (viPreAutoPctRef.current) {
      const [savedVminPct, savedVmaxPct] = viPreAutoPctRef.current;
      setViVminPct(savedVminPct);
      setViVmaxPct(savedVmaxPct);
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
  const regionRequests = React.useMemo(() => latestRegion((row, col) => {
    model.set("vi_roi_center", [row, col]);
    model.save_changes();
  }, model.get("vi_roi_center") || []), [model]);
  React.useEffect(() => {
    const acknowledge = () => regionRequests.acknowledge();
    model.on("change:vi_roi_receipt", acknowledge);
    return () => { model.off("change:vi_roi_receipt", acknowledge); regionRequests.clear(); };
  }, [model, regionRequests]);
  const [webgpuDpcReady, setWebgpuDpcReady] = React.useState(false);
  const [viGpuVersion, setViGpuVersion] = React.useState(0);
  const [viGpuRetainedReady, setViGpuRetainedReady] = React.useState(false);
  const viGpuImageRef = React.useRef<ViGpuImage | null>(null);
  // GPU-resident compare panels: frame index -> engine colormap slot. Written by
  // the interactive compare recompute (no readback), consumed by the grid painter.
  const [compareGpuVersion, setCompareGpuVersion] = React.useState(0);
  const compareGpuSlotsRef = React.useRef(new Map<number, number>());
  const compareGpuRangesRef = React.useRef(new Map<number, { min: number; max: number }>());
  const compareGpuHistogramGenRef = React.useRef(0);
  const compareHistogramPendingSettleRef = React.useRef(false);
  const beginDpRoiInteraction = React.useCallback(() => {
    if (!dpRoiInteractiveRef.current) {
      compareGpuHistogramGenRef.current++;
      compareHistogramPendingSettleRef.current = true;
    }
    dpRoiInteractiveRef.current = true;
  }, []);
  const compareGpuRenderNowRef = React.useRef<(() => number) | null>(null);
  const compareIncrementalRef = React.useRef<{
    mask: Uint32Array;
    buffers: Map<number, GPUBuffer>;
    indicesKey: string;
  } | null>(null);
  const rawVirtualImageVersionRef = React.useRef(0);
  const viGpuColormapRef = React.useRef<GPUColormapEngine | null>(null);
  const viGpuColormapDeviceRef = React.useRef<GPUDevice | null>(null);
  const ensureViGpuColormap = React.useCallback((
    backend: DetectorCompute,
  ): GPUColormapEngine => {
    const device = backend.getDevice();
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
    setViGpuVersion(version => version + 1);
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
  const h5SourceAvailable = Boolean(model.get("_h5_urls"));
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
      } catch (error) {
        const name = error instanceof DOMException ? error.name : "";
        if (name !== "AbortError") console.warn("Could not use directory picker for local HDF5 files", error);
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
      const detectorRows = model.get("det_rows"), detectorCols = model.get("det_cols");
      // Inline mode: the embedded gzip stack. Browser HDF5 sources load below.
      const gunzip = async (compressed: Uint8Array) => new Uint8Array(await new Response(new Blob([compressed as BlobPart]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
      let compute: DetectorCompute | null = null;
      let cpuStack: Uint8Array | null = null;  // full decompressed stack for the per-frame probe (single-chunk only)
      // Multi-VOLUME (5D): several datasets, decoded LAZILY (decode-on-scrub) with a
      // small LRU of resident volumes. Only the viewed dataset (plus a few recent)
      // lives in VRAM, so it runs on a laptop regardless of how many h5 files - and
      // first paint is one decode, not N. The frame slider picks the active dataset.
      const volCache = new Map<number, DetectorCompute>();   // LRU: idx -> decoded volume
      const volLoadPromises = new Map<number, Promise<DetectorCompute | null>>();
      const inlineVolCache = new Map<number, DetectorCompute>(); // LRU for inline gzip 5D exports
      // Recent and compared volumes kept hot for instant scrub and detector drags.
      const compareResidentTarget = Math.max(3, Math.min(12, Math.max(1, Number(model.get("compare_max_panels") || 3))));
      let latestResidentVolumeIndex: number | null = null;
      let volumeCount = 0;
      let getVol: ((idx: number) => Promise<DetectorCompute | null>) | null = null;
      let initialVolumeLoad: Promise<DetectorCompute | null> | null = null;
      let h5VolumePreload: Promise<void> | null = null;
      let h5VolumePreloadDone = false;
      // Browser HDF5 source: the CLI WebGPU export lists its linked masters here.
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
        compareResidentTarget,
        1,
        compareResidentTarget,
      );
      const h5ResidentLimit = h5UsesNativeU16 && !h5AllowU16MultiResident
        ? 1
        : h5RequestedResidentLimit;
      if (requireLocalH5Files && h5Urls.length && !show4DSTEMHasLocalFiles()) {
        if (!disposed) {
          setOfflineBackendStatus("Waiting for local HDF5 files");
          setOfflineBackendLoading(false);
        }
        return;
      }
      const hasInlineViMaps = () => {
        const product = model.get("vi_product_maps_bytes") as DataView | undefined;
        return Boolean(product && product.byteLength > 0);
      };
      const h5FamilyBase = (sourceUrl: string): string =>
        sourceUrl.replace(/_master\.h5(?:[?#].*)?$/, "");
      const h5DataFileUrl = (sourceUrl: string, fileNumber: number): string =>
        `${h5FamilyBase(sourceUrl)}_data_${String(fileNumber).padStart(6, "0")}.h5`;
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
        const low8Only = (globalThis as { __QT_H5_FORCE_LOW8?: boolean }).__QT_H5_FORCE_LOW8 === true;
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
          } catch (event) {
            console.warn("Could not parse embedded HDF5 hot-pixel mask; falling back to master HDF5 metadata", event);
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
          } catch (event) {
            if (requireLocalH5Files) throw event;
            if (!disposed) setH5LocalSourceStatus("Local HDF5 load failed; using URL fallback");
            if (!disposed) setOfflineBackendStatus(`Local ${label} failed; using URL fallback`);
            console.warn("Local HDF5 WebGPU load failed; falling back to URL fetch", event);
          }
        }
        if (/_master\.h5(?:[?#].*)?$/.test(sourceUrl)) {
          if (!disposed) setOfflineBackendStatus(`Reading ${label}`);
          const fetchWindow = show4DSTEMGlobalInt("__QT_H5_FETCH_WINDOW", 8, 4, 24);
          const fetchOne = async (fileNumber: number): Promise<ArrayBuffer | null> => {
            return await h5FetchCached(h5DataFileUrl(sourceUrl, fileNumber));
          };
          const inflight = new Map<number, Promise<ArrayBuffer | null>>();
          let next = 1;
          const gpuChunks: { buffer: GPUBuffer; startScan: number; nScan: number }[] = [];
          let startScan = 0, detectorSize = 0, computeMode = 1;
          let maxDataFiles = Number.POSITIVE_INFINITY;
          let h5TotalFrames = hasEmbeddedBadPx ? scanRows * scanCols : 0;
          let device: GPUDevice | null = null;
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
          let decodeDone = false;
          let decodeWake: (() => void) | null = null;
          let decodeSpaceWake: (() => void) | null = null;
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
              const decoded = await decodeBslz4Batch(group, groupDecodeDtype, groupSourceDtype, decodeBatch);
              if (!decoded) return false;
              device = decoded.device;
              computeMode = decoded.mode;
              decoded.buffers.forEach((buffer, i) => {
                const spec = group[i];
                gpuChunks.push({ buffer, startScan: spec.startScan, nScan: spec.nScan });
              });
            }
          };
          try {
            const decodePromise = decodeWorker();
            try {
              const masterInfo = await readH5MasterInfoCached(sourceUrl, "master");
              if (masterInfo) {
                if (!hasEmbeddedBadPx && masterInfo.badPixels.length) h5BadPx = new Uint32Array(masterInfo.badPixels);
                h5TotalFrames = Math.max(0, Math.round(Number(masterInfo.totalFrames || h5TotalFrames || 0)));
                if (Number.isFinite(masterInfo.dataFileCount) && Number(masterInfo.dataFileCount) > 0) {
                  maxDataFiles = Math.round(Number(masterInfo.dataFileCount));
                }
              }
            } catch (event) {
              console.warn("Could not read HDF5 master metadata; continuing without detector mask/file-count bounds", event);
            }
            const initialFetchLimit = Number.isFinite(maxDataFiles)
              ? Math.min(fetchWindow, maxDataFiles)
              : fetchWindow;
            for (; next <= initialFetchLimit; next++) {
              inflight.set(next, fetchOne(next));
            }
            for (let n = 1; !disposed; n++) {
              const pending = inflight.get(n);
              if (!pending) break;
              inflight.delete(n);
              const fileBuffer = await pending;
              h5RawFileCache.delete(h5DataFileUrl(sourceUrl, n));
              if (!fileBuffer) break;
              if (!disposed) setOfflineBackendStatus(`Reading ${label}: data file ${n}`);
              const fileVolume = readH5Volume(fileBuffer, "merged");
              if (!Number.isFinite(maxDataFiles)) {
                maxDataFiles = Math.ceil((h5TotalFrames || scanRows * scanCols) / Math.max(1, fileVolume.nFrames));
              }
              detectorSize = fileVolume.detSize;
              if (sourceDtype !== "unknown" && sourceDtype !== fileVolume.srcDtype) {
                throw new Error(`Mixed HDF5 source dtypes are not supported in one browser load: ${sourceDtype} and ${fileVolume.srcDtype}.`);
              }
              sourceDtype = fileVolume.srcDtype;
              decodeDtype = fileVolume.srcDtype === "float32" ? "float32" : (wantU16 && fileVolume.srcDtype === "uint16") ? "uint16" : "uint8";
              {
                // Every chunk of this file, each at its own scan offset. Taking only
                // chunks[0] silently dropped all later chunks of multi-chunk files,
                // leaving the virtual image black outside the first chunk's scan rows.
                let chunkStart = startScan;
                for (const chunk of fileVolume.chunks) {
                  await waitForDecodeSpace();
                  decodeQueue.push({
                    ...chunk,
                    startScan: chunkStart,
                    nScan: chunk.nFrames,
                    decodeDtype,
                    sourceDtype: fileVolume.srcDtype,
                  });
                  chunkStart += chunk.nFrames;
                }
                wakeDecode();
              }
              startScan += fileVolume.nFrames;
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
          } catch (event) {
            decodeDone = true;
            wakeDecode();
            wakeDecodeSpace();
            gpuChunks.forEach((chunk) => chunk.buffer.destroy());
            throw event;
          }
          const created = device ? DetectorCompute.fromGpuChunks(device, gpuChunks, scanRows * scanCols, detectorSize, computeMode) : null;
          if (created && h5BadPx.length) created.badPx = h5BadPx;
          if (!disposed) setOfflineBackendStatus(`${label} ready`);
          return created;
        }
        if (!disposed) setOfflineBackendStatus(`Reading ${label}`);
        const h5Buffer = await (await fetch(sourceUrl)).arrayBuffer();
        try {
          const masterInfo = readH5MasterInfo(h5Buffer, "merged");
          if (masterInfo.badPixels.length) h5BadPx = new Uint32Array(masterInfo.badPixels);
        } catch (event) {
          console.warn("Could not read HDF5 hot-pixel mask; continuing without detector mask", event);
        }
        const mergedVolume = readH5Volume(h5Buffer, "merged");
        const decodeDtype = mergedVolume.srcDtype === "float32" ? "float32" : "uint8";
        let mergedStart = 0;
        const mergedSpecs = mergedVolume.chunks.map((chunk) => {
          const spec = { ...chunk, startScan: mergedStart, nScan: chunk.nFrames };
          mergedStart += chunk.nFrames;
          return spec;
        });
        const created = await DetectorCompute.createFromBslz4Chunked(mergedSpecs, scanRows * scanCols, mergedVolume.detSize, decodeDtype, mergedVolume.srcDtype);
        if (created && h5BadPx.length) created.badPx = h5BadPx;
        if (!disposed) setOfflineBackendStatus(`${label} ready`);
        return created;
      };
      if (h5Urls.length) {
        volumeCount = h5Urls.length;
        getVol = async (idx: number) => {
          const clamped = Math.max(0, Math.min(h5Urls.length - 1, idx));
          if (volCache.has(clamped)) return volCache.get(clamped)!;
          const existingLoad = volLoadPromises.get(clamped);
          if (existingLoad) return await existingLoad;
          const loadPromise = (async () => {
            const volume = await loadH5Compute(h5Urls[clamped], `dataset ${clamped + 1}/${h5Urls.length}`);
            if (volume) {
              volCache.set(clamped, volume);
              latestResidentVolumeIndex = clamped;
              const activeFrame = Math.max(0, Math.min(h5Urls.length - 1, model.get("frame_idx") | 0));
              evictOldVolumes(volCache, h5ResidentLimit, clamped, activeFrame);
            }
            return volume;
          })().finally(() => { volLoadPromises.delete(clamped); });
          volLoadPromises.set(clamped, loadPromise);
          return await loadPromise;
        };
        const initialIdx = Math.max(0, Math.min(h5Urls.length - 1, model.get("frame_idx") | 0));
        const startH5Preloads = (): Promise<void> => {
          if (!getVol) return Promise.resolve();
          // Visit every dataset in the background. For uint16 this remains
          // serial (preloadWindow=1) and only one detector volume is resident,
          // but its BF/DF result is retained in a display slot before moving on.
          const maxPreload = Math.max(1, h5Urls.length);
          const preloadCount = show4DSTEMGlobalInt("__QT_H5_PRELOAD_VOLUMES", h5Urls.length, 1, maxPreload);
          const preloadWindow = show4DSTEMGlobalInt(
            "__QT_H5_PRELOAD_WINDOW",
            1,
            1,
            Math.min(4, preloadCount),
          );
          const prefetchNext = !h5UsesNativeU16 &&
            (globalThis as { __QT_H5_PREFETCH_NEXT?: boolean }).__QT_H5_PREFETCH_NEXT !== false;
          const prefetchWindow = show4DSTEMGlobalInt("__QT_H5_PREFETCH_WINDOW", 2, 1, 8);
          const maxPrefetchFiles = show4DSTEMGlobalInt("__QT_H5_PREFETCH_FILES", 32, 1, 64);
          const order = [
            initialIdx,
            ...Array.from({ length: h5Urls.length }, (_v, i) => i).filter((i) => i !== initialIdx),
          ].slice(0, preloadCount);
          const prefetchStarted = new Set<number>();
          const prefetchVolume = async (index: number): Promise<void> => {
            if (!prefetchNext || prefetchStarted.has(index) || !/_master\.h5(?:[?#].*)?$/.test(h5Urls[index])) return;
            prefetchStarted.add(index);
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
              const pending = inflight.get(n);
              if (!pending) break;
              inflight.delete(n);
              const fileBuffer = await pending.catch(() => null);
              if (!fileBuffer) break;
              if (nextFile <= fileLimit) {
                inflight.set(nextFile, h5FetchCached(h5DataFileUrl(h5Urls[index], nextFile)));
                nextFile += 1;
              }
            }
          };
          let next = 0;
          const worker = async () => {
            while (!disposed) {
              const orderIndex = next++;
              if (orderIndex >= order.length) return;
              const index = order[orderIndex];
              if (prefetchNext && preloadWindow === 1 && orderIndex + 1 < order.length) {
                void prefetchVolume(order[orderIndex + 1]).catch((error) => {
                  console.warn("Show4DSTEM HDF5 compressed prefetch failed", error);
                });
              }
              try {
                const volume = await getVol!(index);
                if (volume && !disposed) {
                  latestResidentVolumeIndex = index;
                  requestCompareViLive();
                  requestDpFrameLive();
                }
              } catch (error) {
                // a background preload; the dataset loads again when the user selects it
                console.warn(`Show4DSTEM HDF5 preload of dataset ${index} failed`, error);
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
      } else {
        // Inline stack: inflate the gzip bytes (lossless); create() infers uint8 vs uint16 from the length.
        const stackView = model.get("_offline_stack") as DataView | undefined;
        if (!stackView || stackView.byteLength === 0) return;
        let stack = new Uint8Array(stackView.buffer, stackView.byteOffset, stackView.byteLength);
        stack = await gunzip(stack);
        const widgetFrames = Math.max(1, model.get("n_frames") | 0);
        const scanCount = scanRows * scanCols;
        const detSize = detectorRows * detectorCols;
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
            const volume = await DetectorCompute.create(bytes, scanCount, detSize);
            if (volume) {
              inlineVolCache.set(idx, volume);
              const activeFrame = Math.max(0, Math.min(widgetFrames - 1, model.get("frame_idx") | 0));
              evictOldVolumes(inlineVolCache, MAX_INLINE_RESIDENT, idx, activeFrame);
            }
            return volume;
          };
          compute = await getVol(Math.max(0, Math.min(widgetFrames - 1, model.get("frame_idx") | 0)));
        } else {
          cpuStack = stack;  // keep for the per-frame probe (single-chunk only)
          compute = await DetectorCompute.create(stack, scanRows * scanCols, detectorRows * detectorCols);
        }
      }
      if ((!compute && !initialVolumeLoad) || disposed) {
        compute?.dispose();
        if (!disposed) {
          setOfflineBackendError(OFFLINE_NEEDS_WEBGPU);
          setOfflineBackendStatus("");
          setOfflineBackendLoading(false);
        }
        return;
      }
      // Every DetectorCompute carries the DPC and iDPC kernels, so a loaded volume
      // makes the DPC sources available.
      if (compute) setWebgpuDpcReady(true);
      // Auto-filter hot/dead detector pixels (from the HDF5 pixel_mask) so the
      // offline result matches CUDA's apply_mask path - no manual masking needed.
      const badPxJson = model.get("_offline_bad_px") as string | undefined;
      if (badPxJson && compute && compute.detSize === detectorRows * detectorCols) {
        compute.badPx = new Uint32Array(JSON.parse(badPxJson) as number[]);
      }
      const dpcMask = buildFullDetectorMask(detectorRows, detectorCols);
      const computeDpcImage = async (
        backend: DetectorCompute,
        source: DpcGpuSource,
      ): Promise<Float32Array> => {
        if (source === "iDPC") return await backend.maskedIDpc(dpcMask, detectorCols, scanRows, scanCols, 0, false);
        return await backend.maskedDpc(dpcMask, detectorCols, source === "DPC_row" ? "row" : "col");
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
        data: Float32Array;
        key: string;
      };
      const viWarmCache = new Map<string, WarmViCacheEntry>();
      let viWarmupStarted = false;
      let viWarmupGeneration = 0;
      let suppressViTraitRecompute = false;
      const roundedCacheValue = (value: unknown): string => {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) return "0";
        return String(Math.round(numeric * 1000) / 1000);
      };
      const activeVolumeCacheKey = () => `vol:${Math.max(0, Math.round(Number(model.get("frame_idx") || 0)))}`;
      const dpcWarmCacheKey = (source: DpcGpuSource) => [
        activeVolumeCacheKey(),
        "dpc",
        source,
        `scan:${scanRows}x${scanCols}`,
        `det:${detectorRows}x${detectorCols}`,
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
        `det:${detectorRows}x${detectorCols}`,
      ].join("|");
      const currentRoiGeometry = (): WarmRoiGeometry => {
        const mode = String(model.get("roi_mode") || "circle") === "annular" ? "annular" : "circle";
        return {
          mode,
          centerRow: Number(model.get("roi_center_row") || model.get("center_row") || detectorRows / 2),
          centerCol: Number(model.get("roi_center_col") || model.get("center_col") || detectorCols / 2),
          radius: Math.max(1, Number(model.get("roi_radius") || model.get("bf_radius") || 1)),
          radiusInner: mode === "annular" ? Math.max(0, Number(model.get("roi_radius_inner") || 0)) : 0,
        };
      };
      const presetRoiGeometry = (name: WarmRoiPresetName): WarmRoiGeometry => {
        const bfRadius = Math.max(1, Number(model.get("bf_radius") || 1));
        const centerRow = Number(model.get("center_row") || model.get("roi_center_row") || detectorRows / 2);
        const centerCol = Number(model.get("center_col") || model.get("roi_center_col") || detectorCols / 2);
        if (name === "abf") {
          return {
            mode: "annular",
            centerRow,
            centerCol,
            radius: bfRadius,
            radiusInner: Math.max(0.5, bfRadius * 0.5),
          };
        }
        if (name === "adf") {
          return {
            mode: "annular",
            centerRow,
            centerCol,
            radius: bfRadius * 2,
            radiusInner: bfRadius,
          };
        }
        return {
          mode: "circle",
          centerRow,
          centerCol,
          radius: bfRadius,
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
      }, detectorRows, detectorCols);
      const dataViewForFloat32 = (data: Float32Array): DataView => (
        new DataView(data.buffer as ArrayBuffer, data.byteOffset, data.byteLength)
      );
      const setWarmCacheEntry = (entry: WarmViCacheEntry) => {
        viWarmCache.set(entry.key, entry);
      };
      const serveWarmCacheEntry = (key: string, source: ViGpuSource): boolean => {
        if (dpRoiInteractiveRef.current) return false;
        const cached = viWarmCache.get(key);
        if (!cached || cached.source !== source || cached.data.length !== scanRows * scanCols) {
          return false;
        }
        clearViGpuDisplay();
        publishVirtualImageBytes(dataViewForFloat32(cached.data));
        return true;
      };
      let dpcBufferQueue: Promise<void> = Promise.resolve();
      const computeDpcBufferImage = async (
        backend: DetectorCompute,
        source: DpcGpuSource,
      ): Promise<boolean> => {
        const run = async (): Promise<boolean> => {
          const engine = ensureViGpuColormap(backend);
          await engine.getDevice().queue.onSubmittedWorkDone().catch(() => {});
          const result: { buffer: GPUBuffer; n: number; cleanup?: () => void } = source === "iDPC"
            ? await backend.maskedIDpcBuffer(dpcMask, detectorCols, scanRows, scanCols, 0, false)
            : backend.maskedDpcBuffer(dpcMask, detectorCols, source === "DPC_row" ? "row" : "col");
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
          setViGpuVersion(version => version + 1);
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
        const { buffer, n } = backend.maskedSumBuffer(mask);
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
        setViGpuVersion(version => version + 1);
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
        if (h5Urls.length !== 1) return null;
        const sourceUrl = h5Urls[0];
        return /_master\.h5(?:[?#].*)?$/.test(sourceUrl) ? sourceUrl : null;
      };
      const computeH5ProductFirstRoi = async (
        mask: Uint32Array,
        generation: number,
      ): Promise<{ data: Float32Array; displayed: boolean } | null> => {
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
        if (disposed || generation !== viRecomputeGen) {
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
          setViGpuVersion(version => version + 1);
          displayed = true;
        }
        const data = await readGpuFloatBuffer(product.device, product.buffer, product.scanRows * product.scanCols);
        if (!displayed) product.buffer.destroy();
        return { data, displayed };
      };
      let viRecomputeGen = 0;
      const recomputeVI = async () => {
        const generation = ++viRecomputeGen;
        const source = normaliseViSource(model.get("vi_source"));
        const product = viProductFrameView(model, scanRows, scanCols, source);
        if (product) {
          if (disposed || generation !== viRecomputeGen) return;
          clearViGpuDisplay();
          publishVirtualImageBytes(product);
          return;
        }
        if (isDpcGpuSource(source)) {
          if (serveWarmCacheEntry(dpcWarmCacheKey(source), source)) {
            return;
          }
          if (!compute) return;
          const displayed = await computeDpcBufferImage(compute!, source);
          if (disposed || generation !== viRecomputeGen) return;
          if (!displayed) {
            clearViGpuDisplay();
          } else {
            void (async () => {
              const dpc = await computeDpcImage(compute!, source);
              if (disposed || generation !== viRecomputeGen || !dpc) return;
              setWarmCacheEntry({
                source,
                data: dpc,
                key: dpcWarmCacheKey(source),
              });
              publishVirtualImageBytes(new DataView(dpc.buffer));
            })();
            return;
          }
          const dpc = await computeDpcImage(compute!, source);
          if (disposed || generation !== viRecomputeGen) return;
          if (dpc) {
            setWarmCacheEntry({
              source,
              data: dpc,
              key: dpcWarmCacheKey(source),
            });
            publishVirtualImageBytes(new DataView(dpc.buffer));
            return;
          }
          return;
        }
        const mask = buildDetectorMask(model, detectorRows, detectorCols);
        const roiKey = roiWarmCacheKey(currentRoiGeometry());
        if (serveWarmCacheEntry(roiKey, "roi")) {
          return;
        }
        const preferH5ProductFirst = (globalThis as { __QT_H5_PRODUCT_FIRST_VI?: unknown }).__QT_H5_PRODUCT_FIRST_VI === true;
        if (preferH5ProductFirst || !compute) {
          const h5Product = await computeH5ProductFirstRoi(mask, generation);
          if (disposed || generation !== viRecomputeGen) return;
          if (h5Product) {
            setWarmCacheEntry({
              source: "roi",
              data: h5Product.data,
              key: roiKey,
            });
            publishVirtualImageBytes(new DataView(h5Product.data.buffer));
            return;
          }
        }
        if (!compute) return;
        const displayed = computeRoiBufferImage(compute!, mask);
        if (!displayed) {
          clearViGpuDisplay();
        }
        if (disposed || generation !== viRecomputeGen) return;
        if (displayed && dpRoiInteractiveRef.current) {
          return;
        }
        const virtualImage = await compute!.maskedSum(mask);
        if (disposed || generation !== viRecomputeGen) return;
        setWarmCacheEntry({
          source: "roi",
          data: virtualImage,
          key: roiKey,
        });
        publishVirtualImageBytes(new DataView(virtualImage.buffer));
      };
      const warmStandardViCache = async () => {
        if (viWarmupStarted || disposed || !compute) {
          return;
        }
        viWarmupStarted = true;
        const warmGeneration = ++viWarmupGeneration;
        try {
          for (const preset of ["bf", "abf", "adf"] as WarmRoiPresetName[]) {
            if (disposed || warmGeneration !== viWarmupGeneration || !compute) {
              return;
            }
            const geometry = presetRoiGeometry(preset);
            const key = roiWarmCacheKey(geometry);
            if (viWarmCache.has(key)) continue;
            const data = await compute.maskedSum(maskForRoiGeometry(geometry));
            if (disposed || warmGeneration !== viWarmupGeneration) {
              return;
            }
            setWarmCacheEntry({
              source: "roi",
              data,
              key,
            });
          }
          for (const source of ["DPC_row", "DPC_col"] as DpcGpuSource[]) {
            if (disposed || warmGeneration !== viWarmupGeneration || !compute) {
              return;
            }
            const key = dpcWarmCacheKey(source);
            if (viWarmCache.has(key)) continue;
            const data = await computeDpcImage(compute, source);
            if (!data || disposed || warmGeneration !== viWarmupGeneration) {
              continue;
            }
            setWarmCacheEntry({
              source,
              data,
              key,
            });
          }
        } catch (error) {
          // the cache only pre-computes the presets; without it they are computed when picked
          console.warn("Show4DSTEM could not pre-compute the BF, ABF, ADF and DPC images", error);
        }
      };
      const resetWarmViCache = () => {
        viWarmupGeneration += 1;
        viWarmupStarted = false;
        viWarmCache.clear();
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
      let compareGpuCompletion: Promise<void> = Promise.resolve();
      let compareGpuInFlight = 0;
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
      // Interactive drags must never PAGE a volume: with more panels than the
      // resident LRU, recomputing every panel per drag step forces a full-volume
      // decode + eviction each time (seconds per step, permanent thrash). During
      // a drag only resident volumes update; skipped panels keep their previous
      // image (persistent stack) and catch up on the mouseup finalize.
      let comparePersistentStack: Float32Array | null = null;
      let compareLastInteractiveMs = 0;
      // Coalesce GPU-slot version bumps to one React commit per animation frame:
      // mouse moves arrive faster than paints, and each bump re-renders the grid.
      let compareGpuRafHandle = 0;
      const bumpCompareGpuVersion = () => {
        if (compareGpuRafHandle) return;
        compareGpuRafHandle = requestAnimationFrame(() => {
          compareGpuRafHandle = 0;
          setCompareGpuVersion(version => version + 1);
        });
      };
      // Fresh settled bytes supersede the drag-time GPU slots: clear them so the
      // grid falls back to the exact (bad-px-corrected, mask-normalised) images.
      const settleCompareGpuSlots = () => {
        compareIncrementalRef.current = null;
        if (compareGpuSlotsRef.current.size) {
          compareGpuSlotsRef.current.clear();
          compareGpuRangesRef.current.clear();
          setCompareGpuVersion(version => version + 1);
        }
      };
      const volIsResident = (idx: number): boolean =>
        !getVol || volCache.has(idx) || inlineVolCache.has(idx);
      const publishDirectCompareStack = (
        bytes: DataView,
        count: number,
        indices: number[],
      ) => {
        model.set("compare_virtual_image_bytes", bytes);
        model.set("compare_panel_count", count);
        model.set("compare_panel_indices", indices);
      };
      const recomputeCompareVI = async () => {
        if (disposed) return;
        const generation = ++compareViGen;
        const indices = compareVisibleIndices();
        if (!indices.length) return;
        const source = normaliseViSource(model.get("vi_source"));
        const interactiveDrag = dpRoiInteractiveRef.current;
        // ROI compare panels render through GPU-resident slots both for the
        // initial/settled image and for drag updates. The initial frame uses a
        // full exact masked sum; subsequent geometry changes use exact deltas.
        const updateRoiCompareGpuSlots = async (): Promise<boolean> => {
          if (source !== "roi") return false;
          const engine = compute ? ensureViGpuColormap(compute) : null;
          if (!engine) return false;
          const detectorMask = buildDetectorMask(model, detectorRows, detectorCols);
          let slotCursor = 0;
          const batchComputes: DetectorCompute[] = [];
          const batchSlots: number[] = [];
          const batchFrames: number[] = [];
          for (const idx of indices) {
            const slot = COMPARE_GPU_SLOT_BASE + slotCursor++;
            // Do not turn a progressive refresh into an all-volume decode.
            // A completed panel keeps its GPU display slot even after the
            // native uint16 source volume has been evicted.
            // A settled/initial refresh must materialise every visible panel so
            // later detector drags can update the whole comparison grid live.
            // During the drag itself, keep the existing no-paging rule and only
            // touch volumes that are already resident.
            if (interactiveDrag && getVol && !volIsResident(idx)) continue;
            const panelCompute = getVol ? await getVol(idx) : compute;
            if (disposed || generation !== compareViGen) return false;
            if (!panelCompute || !(panelCompute instanceof DetectorCompute)) continue;
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
            let addedPixels = 0;
            let removedPixels = 0;
            if (
              interactiveDrag
              && previous
              && previous.indicesKey === indicesKey
              && previous.mask.length === detectorMask.length
              && batchFrames.every((frame) => previous.buffers.has(frame))
            ) {
              const addedMask = new Uint32Array(detectorMask.length);
              const removedMask = new Uint32Array(detectorMask.length);
              for (let i = 0; i < detectorMask.length; i++) {
                const next = detectorMask[i] !== 0;
                const prev = previous.mask[i] !== 0;
                if (next && !prev) { addedMask[i] = 1; addedPixels++; }
                else if (!next && prev) { removedMask[i] = 1; removedPixels++; }
              }
              if (addedPixels === 0 && removedPixels === 0) return true;
              const prevBuffers = batchFrames.map((frame) => previous.buffers.get(frame)!);
              buffers = DetectorCompute.maskedSumDeltaBuffersBatch(batchComputes, prevBuffers, addedMask, removedMask).buffers;
            } else {
              buffers = DetectorCompute.maskedSumBuffersBatch(batchComputes, detectorMask).buffers;
            }
            const nextBuffers = new Map<number, GPUBuffer>();
            for (let i = 0; i < buffers.length; i++) {
              engine.adoptBuffer(batchSlots[i], buffers[i], scanCols, scanRows);
              nextBuffers.set(batchFrames[i], buffers[i]);
              adopted++;
            }
            const rangesReady = batchFrames.every((frame) => compareGpuRangesRef.current.has(frame));
            if (!interactiveDrag || !rangesReady) {
              const ranges = await engine.computeRangeBatch(batchSlots);
              ranges.forEach((range, index) => {
                const frame = batchFrames[index];
                if (frame !== undefined) compareGpuRangesRef.current.set(frame, range);
              });
            }
            compareIncrementalRef.current = {
              mask: new Uint32Array(detectorMask),
              buffers: nextBuffers,
              indicesKey,
            };
            if (interactiveDrag) compareGpuRenderNowRef.current?.();
            compareGpuCompletion = engine.getDevice().queue.onSubmittedWorkDone();
            if (interactiveDrag) await compareGpuCompletion;
            if (disposed || generation !== compareViGen) return false;
          }
          if (adopted && !interactiveDrag) {
            // A quick release alone does not make float slots current. Their
            // settled normalization must be queued before histograms resume.
            if (!disposed && generation === compareViGen && !dpRoiInteractiveRef.current)
              compareHistogramPendingSettleRef.current = false;
            bumpCompareGpuVersion();
          }
          if (batchFrames.length && !interactiveDrag) {
            model.set("compare_panel_count", indices.length);
            model.set("compare_panel_indices", indices);
          }
          return adopted > 0 || interactiveDrag;
        };
        if (await updateRoiCompareGpuSlots()) return;
        if (disposed || generation !== compareViGen) return;
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
            if (disposed || generation !== compareViGen || !panelCompute) return;
            const dpc = await computeDpcImage(panelCompute, source);
            if (disposed || generation !== compareViGen || !dpc) return;
            stack.set(dpc, slot * panelPixels);
          }
          settleCompareGpuSlots();
          // fresh copy: reusing the persistent stack's ArrayBuffer identity makes this
          // model.set a silent no-op (no change event -> stats/export/save-state stale)
          publishDirectCompareStack(new DataView(stack.slice().buffer), indices.length, indices);
          return;
        }
        const mask = buildDetectorMask(model, detectorRows, detectorCols);
        let maskArea = 0;
        for (let i = 0; i < mask.length; i++) maskArea += mask[i] ? 1 : 0;
        maskArea = Math.max(1, maskArea);
        for (let slot = 0; slot < indices.length; slot++) {
          const idx = indices[slot];
          if (interactiveDrag && !volIsResident(idx)) continue;   // keep previous pixels
          const panelCompute = getVol ? await getVol(idx) : compute;
          if (disposed || generation !== compareViGen || !panelCompute) return;
          const virtualImage = await panelCompute.maskedSum(mask);
          if (disposed || generation !== compareViGen) return;
          for (let pixel = 0; pixel < panelPixels; pixel++) {
            stack[slot * panelPixels + pixel] = virtualImage[pixel] / maskArea;
          }
        }
        settleCompareGpuSlots();
        // fresh copy: reusing the persistent stack's ArrayBuffer identity makes this
        // model.set a silent no-op (no change event -> stats/export/save-state stale)
        publishDirectCompareStack(new DataView(stack.slice().buffer), indices.length, indices);
      };
      const recomputeVisibleVirtualImages = async () => {
        const mode = String(model.get("view_mode") || "single");
        if (mode === "multiple" || mode === "compare") {
          await recomputeCompareVI();
          return;
        }
        await recomputeVI();
      };
      requestCompareViLiveRef.current = () => {
        if (compareViLiveInFlightRef.current || compareGpuInFlight >= 2) {
          compareViLivePendingRef.current = true;
          return;
        }
        compareViLivePendingRef.current = false;
        flushRoiCenter(false);
        // Keep scientific geometry current without repainting between display frames.
        flushRoiRadius(false);
        compareViLiveInFlightRef.current = true;
        void (async () => {
          await recomputeVisibleVirtualImages();
        })().catch(error => {
          if (!disposed) setOfflineBackendError(String(error));
        }).finally(() => {
          if (disposed) return;
          compareViLiveInFlightRef.current = false;
          if (compareViLivePendingRef.current && dpRoiInteractiveRef.current && compareGpuInFlight < 2) {
            compareViLivePendingRef.current = false;
            requestCompareViLive();
          } else if (!dpRoiInteractiveRef.current) {
            compareViLivePendingRef.current = false;
          }
        });
      };
      requestViFinalizeRef.current = () => {
        void (async () => {
          await compareGpuCompletion;
          if (disposed || dpRoiInteractiveRef.current) return;
          await recomputeVisibleVirtualImages();
        })().catch(error => { if (!disposed) setOfflineBackendError(String(error)); });
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
      const detSize = detectorRows * detectorCols;
      const sample = (pixelIndex: number) => compute!.mode === 1 ? cpuStack![pixelIndex] : (cpuStack![pixelIndex * 2] | (cpuStack![pixelIndex * 2 + 1] << 8));
      const computeFrame = async (isCurrent: () => boolean) => {
        const scanRow = Math.max(0, Math.min(scanRows - 1, model.get("pos_row") | 0));
        const scanCol = Math.max(0, Math.min(scanCols - 1, model.get("pos_col") | 0));
        const scanIdx = scanRow * scanCols + scanCol;
        const mode = String(model.get("view_mode") || "single");
        const dpMode = String(model.get("compare_dp_mode") || "average");
        if ((mode === "multiple" || mode === "compare") && dpMode !== "selected" && getVol) {
          const indices = comparePageState().page;
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
              if (!isCurrent()) return;
              model.set("frame_bytes", new DataView(averaged.buffer));
              model.save_changes();
              return;
            }
          }
        }
        // bslz4 / chunked stacks have no CPU copy -> extract the frame on the GPU.
        if (!compute) return;
        const frameIndex = Math.max(0, Math.min((volumeCount || 1) - 1, model.get("frame_idx") | 0));
        const frameSource = getVol ? await getVol(frameIndex) : compute;
        if (!frameSource || !isCurrent()) return;
        const frame = cpuStack
          ? (() => { const values = new Float32Array(detSize); const base = scanIdx * detSize; for (let k = 0; k < detSize; k++) values[k] = sample(base + k); return values; })()
          : await frameSource.frameAt(scanIdx);
        if (!isCurrent()) return;
        model.set("frame_bytes", new DataView(frame.buffer)); model.save_changes();
      };
      const frameQueue = createLatestFrameQueue(computeFrame, error => {
        if (!disposed) setOfflineBackendError(String(error));
      });
      const recomputeFrame = frameQueue.request;
      requestDpFrameLiveRef.current = () => {
        void recomputeFrame();
      };
      if (h5VolumePreload) {
        void h5VolumePreload.then(() => {
          if (disposed) return;
          frameQueue.invalidate();
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
        void recomputeFrame();
      };
      const onPos = () => { void recomputeFrame(); };
      const onCompareFrameSource = () => { frameQueue.invalidate(); void recomputeFrame(); };
      const onCompareGridSource = () => { frameQueue.invalidate(); void recomputeCompareVI(); void recomputeFrame(); };
      const activateCurrentVolume = async () => {
        if (!getVol) return true;
        const nVolumes = volumeCount || 1;
        const volumeIndex = Math.max(0, Math.min(nVolumes - 1, model.get("frame_idx") | 0));
        const volume = await getVol(volumeIndex);
        if (!volume) return false;
        compute = volume;
        setWebgpuDpcReady(true);
        return true;
      };
      if (initialVolumeLoad) {
        void initialVolumeLoad.then((volume) => {
          if (!volume || disposed) return;
          frameQueue.invalidate();
          compute = volume;
          setWebgpuDpcReady(true);
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
        frameQueue.invalidate();
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
        frameQueue.invalidate();
        const generation = ++frameGen;                  // ignore a stale decode if the user keeps scrubbing
        const ready = await activateCurrentVolume();
        if (generation !== frameGen || !ready) return;   // a newer scroll superseded this one
        resetWarmViCache();
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
        const bfRadius = model.get("bf_radius") || 1;
        suppressViTraitRecompute = true;
        try {
          model.set("roi_active", true);
          model.set("vi_source", "roi");
          model.set("roi_center_row", model.get("center_row"));
          model.set("roi_center_col", model.get("center_col"));
          if (name === "bf") { model.set("roi_mode", "circle"); model.set("roi_radius_inner", 0); model.set("roi_radius", Math.max(1, bfRadius)); }
          else if (name === "abf") { model.set("roi_mode", "annular"); model.set("roi_radius_inner", Math.max(0.5, bfRadius * 0.5)); model.set("roi_radius", Math.max(1, bfRadius)); }
          else if (name === "adf") { model.set("roi_mode", "annular"); model.set("roi_radius_inner", bfRadius); model.set("roi_radius", bfRadius * 2); }
          else if (name === "haadf") { model.set("roi_mode", "annular"); model.set("roi_radius_inner", bfRadius * 2); model.set("roi_radius", bfRadius * 4); }
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
        const center = model.get("roi_center");
        if (Array.isArray(center) && center.length === 2) {
          splittingRoiCenter = true;
          try {
            model.set("roi_center_row", center[0]);
            model.set("roi_center_col", center[1]);
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
        const center = model.get("vi_roi_center");
        if (Array.isArray(center) && center.length === 2) {
          splittingViCenter = true;
          try {
            model.set("vi_roi_center_row", center[0]);
            model.set("vi_roi_center_col", center[1]);
          } finally {
            splittingViCenter = false;
          }
        }
        void recomputeDP();
      };
      const viTraits = ["roi_center_row", "roi_center_col", "roi_radius", "roi_radius_inner", "roi_mode", "roi_width", "roi_height"];
      const dpTraits = ["vi_roi_center_row", "vi_roi_center_col", "vi_roi_radius", "vi_roi_mode", "vi_roi_width", "vi_roi_height", "vi_roi_reduce"];
      viTraits.forEach((trait) => model.on("change:" + trait, onVI));
      dpTraits.forEach((trait) => model.on("change:" + trait, onDP));
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
        viTraits.forEach((trait) => model.off("change:" + trait, onVI));
        dpTraits.forEach((trait) => model.off("change:" + trait, onDP));
        model.off("change:vi_source", onVI);
        model.off("change:roi_center", onRoiCenter);
        model.off("change:vi_roi_center", onViCenter);
        model.off("change:_preset_request", onPreset);
        frameQueue.close();
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
        volCache.forEach((volume) => volume.dispose()); volCache.clear();  // every cached lazy volume
        inlineVolCache.forEach((volume) => volume.dispose()); inlineVolCache.clear();
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
        const ratioGuess = Math.min(detectorRows, detectorCols) * 0.125;
        if (Math.abs(current - ratioGuess) > 0.51) return;
        // A few thousand scan positions fix the disk edge; reducing all of them would
        // read the whole stack and delay first paint for no extra accuracy.
        const scanCount = scanRows * scanCols;
        const stride = Math.max(1, Math.floor(scanCount / 16384));
        const scanMask = new Uint32Array(scanCount);
        for (let i = 0; i < scanCount; i += stride) scanMask[i] = 1;
        const dp = await compute.reduceFrames(scanMask, true);
        if (disposed || !dp || dp.length !== detectorRows * detectorCols) return;
        let peak = -Infinity;
        for (let i = 0; i < dp.length; i++) if (dp[i] > peak) peak = dp[i];
        const median = Float32Array.from(dp).sort()[dp.length >> 1];
        const threshold = 0.5 * (peak + median);
        // Intensity-weighted centroid of the disk interior gives a sub-pixel center;
        // the beam is not exactly on the detector center (measured 94.4, 96.6).
        let weight = 0, rowSum = 0, colSum = 0;
        for (let row = 0; row < detectorRows; row++) {
          for (let col = 0; col < detectorCols; col++) {
            const value = dp[row * detectorCols + col];
            if (value >= threshold) { weight += value; rowSum += row * value; colSum += col * value; }
          }
        }
        if (!(weight > 0)) return;
        const centerRow = rowSum / weight, centerCol = colSum / weight;
        // Radial profile, then the half-max crossing: the disk is flat inside and
        // falls off a cliff at the edge, so half-max is stable against hot pixels.
        const maxRadius = Math.ceil(Math.hypot(
          Math.max(centerRow, detectorRows - centerRow), Math.max(centerCol, detectorCols - centerCol)));
        const radialSum = new Float64Array(maxRadius + 1);
        const radialCount = new Float64Array(maxRadius + 1);
        for (let row = 0; row < detectorRows; row++) {
          for (let col = 0; col < detectorCols; col++) {
            const radius = Math.round(Math.hypot(row - centerRow, col - centerCol));
            if (radius <= maxRadius) { radialSum[radius] += dp[row * detectorCols + col]; radialCount[radius] += 1; }
          }
        }
        const profile = new Float64Array(maxRadius + 1);
        for (let i = 0; i <= maxRadius; i++) profile[i] = radialCount[i] ? radialSum[i] / radialCount[i] : 0;
        // Only radii that actually contain detector pixels carry a profile value. A
        // sub-pixel center usually leaves the radius-0 bin empty, and an empty bin
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
      compareGpuHistogramGenRef.current++;
      compareHistogramPendingSettleRef.current = true;
      if (roiRadiusRafRef.current !== null) cancelAnimationFrame(roiRadiusRafRef.current);
      roiRadiusRafRef.current = null;
      roiRadiusPendingRef.current = null;
      roiRadiusInnerPendingRef.current = null;
      requestViFinalizeRef.current = null;
      requestCompareViLiveRef.current = null;
      compareViLiveInFlightRef.current = false;
      compareViLivePendingRef.current = false;
      if (compareViLiveRafRef.current !== null) {
        cancelAnimationFrame(compareViLiveRafRef.current);
        compareViLiveRafRef.current = null;
      }
      requestDpFrameLiveRef.current = null;
      setWebgpuDpcReady(false);
      setOfflineBackendLoading(false);
      setOfflineBackendStatus("");
      setOfflineBackendError("");
      clearViGpuDisplay();
      detach?.();
    };
  }, [beginDpRoiInteraction, clearViGpuDisplay, ensureViGpuColormap, h5LocalFilesGranted, h5SourceAvailable, offline, requestCompareViLive, requestDpFrameLive, requireLocalH5Files]);
  const [viStats, setViStats] = React.useState<number[]>([0, 0, 0, 0]);
  const [viDataMin, setViDataMin] = React.useState<number>(0);
  const [viDataMax, setViDataMax] = React.useState<number>(1);
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const [showControls] = useModelState<boolean>("show_controls");
  const [controlsCollapsed] = useModelState<boolean>("controls_collapsed");
  const controlsVisible = showControls && !controlsCollapsed;
  const panelChromeVisible = controlsVisible;
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
  const frameSliderAriaLabel = `Show4DSTEM ${frameDimLabel.toLowerCase()}`;
  const compareGridWidth = compareGridPreviewWidth ?? (compareGridWidthPx > 0 ? compareGridWidthPx : COMPARE_GRID_DEFAULT_WIDTH);
  React.useEffect(() => {
    if (!compareMode || compareAllGroups) {
      setCompareReorderMode(false);
      setCompareDraggingFrame(null);
      setComparePendingMoveFrame(null);
      setCompareGridPreviewWidth(null);
      compareGridResizeCleanupRef.current?.();
    }
  }, [compareAllGroups, compareMode]);
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
    const source = Array.isArray(comparePanelIndices) ? comparePanelIndices : [];
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
  }, [compareHiddenPanels, comparePanelIndices, normalizedCompareOrder]);
  const requestComparePage = React.useCallback((page: number) => {
    const next = Math.max(0, Math.min(activeComparePageCount - 1, Math.round(Number(page) || 0)));
    if (next === activeComparePageIdx) return;
    setComparePageIdx(next);
  }, [activeComparePageCount, activeComparePageIdx, setComparePageIdx]);
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
    setComparePendingMoveFrame(null);
    setCompareDraggingFrame(null);
    setCompareHiddenMenuAnchor(null);
  }, [setCompareGroupMode, setCompareHiddenPanels, setComparePageIdx, setComparePanelOrder, setCompareStarredPanels]);

  // ROI FFT state (VI ROI crops virtual image for FFT)
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);
  const roiFftActive = showFft && viRoiMode !== "off";

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

  // A static report embeds one RGB image per preset (BF, ABF, ADF, HAADF) per
  // dataset at the binned scan size; the interactive menu sizes its presets itself.
  const estimateReportHtmlSize = React.useCallback((scanBin: number, datasetScope: HtmlDatasetScope) => {
    const binnedScanRows = Math.max(1, Math.floor(shapeRows / scanBin));
    const binnedScanCols = Math.max(1, Math.floor(shapeCols / scanBin));
    const datasetCount = reportDatasetCount(datasetScope);
    const presetCount = 4;
    const rgbBytes = datasetCount * presetCount * binnedScanRows * binnedScanCols * 3;
    return formatEstimatedHtmlSize(rgbBytes);
  }, [reportDatasetCount, shapeCols, shapeRows]);

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
      } catch (error) {
        if (isAbortLikeError(error)) {
          setHtmlExportBusy(false);
          setLocalHtmlExportStatus("Export canceled");
          return;
        }
        setHtmlExportBusy(false);
        setLocalHtmlExportStatus(`Export failed: ${error instanceof Error ? error.message : String(error)}`);
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
      } catch (error) {
        if (canceled) return;
        pendingHtmlExportRef.current = null;
        setHtmlExportBusy(false);
        setLocalHtmlExportStatus(`Export failed: ${error instanceof Error ? error.message : String(error)}`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      }
    };
    void save();
    return () => { canceled = true; };
  }, [exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest]);

  // Cursor readout state
  const [cursorInfo, commitCursorInfo] = React.useState<{ row: number; col: number; value: number; panel: string } | null>(null);
  const pendingCursorInfoRef = React.useRef<typeof cursorInfo>(null);
  const cursorRafRef = React.useRef<number | null>(null);
  const setCursorInfo = React.useCallback((next: React.SetStateAction<typeof cursorInfo>) => {
    pendingCursorInfoRef.current = typeof next === "function" ? next(pendingCursorInfoRef.current) : next;
    if (cursorRafRef.current !== null) return;
    cursorRafRef.current = requestAnimationFrame(() => {
      cursorRafRef.current = null;
      const pending = pendingCursorInfoRef.current;
      commitCursorInfo(previous => previous === pending || (previous && pending &&
        previous.row === pending.row && previous.col === pending.col &&
        previous.value === pending.value && previous.panel === pending.panel) ? previous : pending);
    });
  }, []);
  React.useEffect(() => () => {
    if (cursorRafRef.current !== null) cancelAnimationFrame(cursorRafRef.current);
  }, []);

  // DP Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const profileResizeStart = React.useRef<{ startY: number; startHeight: number } | null>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<ProfileLayout | null>(null);
  const profilePoints = profileLine || [];
  const rawDpDataRef = React.useRef<Float32Array | null>(null);
  const dpClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [draggingDpProfileEndpoint, setDraggingDpProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingDpProfileLine, setIsDraggingDpProfileLine] = React.useState(false);
  const [hoveredDpProfileEndpoint, setHoveredDpProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringDpProfileLine, setIsHoveringDpProfileLine] = React.useState(false);
  const dpProfileDragStartRef = React.useRef<ProfileLineDrag | null>(null);
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
  const viProfileLayoutRef = React.useRef<ProfileLayout | null>(null);
  const rawViDataRef = React.useRef<Float32Array | null>(null);

  const viClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [draggingViProfileEndpoint, setDraggingViProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingViProfileLine, setIsDraggingViProfileLine] = React.useState(false);
  const [hoveredViProfileEndpoint, setHoveredViProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringViProfileLine, setIsHoveringViProfileLine] = React.useState(false);
  const viProfileDragStartRef = React.useRef<ProfileLineDrag | null>(null);
  const viRoiDragOffsetRef = React.useRef<{ dRow: number; dCol: number }>({ dRow: 0, dCol: 0 });

  // Theme detection
  const { themeInfo, colors: themeColors } = useTheme();
  const roiColors = themeInfo.theme === "dark" ? DARK_ROI_COLORS : LIGHT_ROI_COLORS;
  const accentGreen = themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a";

  // Themed typography: module-level font sizes in theme colors
  const themedTypography = React.useMemo(() => ({
    label: { ...typography.label, color: themeColors.textMuted },
    labelSmall: { ...typography.labelSmall, color: themeColors.textMuted },
    value: { ...typography.value, color: themeColors.textMuted },
    title: { ...typography.title, color: themeColors.accent },
  }), [themeColors]);

  // Compute VI canvas dimensions to respect aspect ratio of rectangular scans
  const viCanvasWidth = shapeRows > shapeCols ? Math.round(canvasSize * (shapeCols / shapeRows)) : canvasSize;
  const viCanvasHeight = shapeCols > shapeRows ? Math.round(canvasSize * (shapeRows / shapeCols)) : canvasSize;

  // Histogram inputs live in state so a new array re-renders the Histogram
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
    const rawData = new Float32Array(displayedDpBytes.buffer, displayedDpBytes.byteOffset, displayedDpBytes.byteLength / 4);
    // Store raw data for profile sampling
    if (!rawDpDataRef.current || rawDpDataRef.current.length !== rawData.length) {
      rawDpDataRef.current = new Float32Array(rawData.length);
    }
    rawDpDataRef.current.set(rawData);
    const stats = computeStats(rawData);
    setDpStats([stats.mean, stats.min, stats.max, stats.std]);
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
  const gpuFFTRef = React.useRef<DisplayFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  // Frame animation timer (5D time/tilt series)
  const frameBounceDir = React.useRef(1);
  React.useEffect(() => {
    frameBounceDir.current = frameReverse ? -1 : 1;
  }, [frameReverse]);

  React.useEffect(() => {
    if (compareMode) {
      if (framePlaying) setFramePlaying(false);
      return;
    }
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
  }, [compareMode, framePlaying, nFrames, frameFps, frameLoop, frameReverse, frameBoomerang, setFrameIdx, setFramePlaying]);

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getDisplayFFT().then(fft => {
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
  React.useEffect(() => { const view = dpViewRef.current; view.zoom = dpZoom; view.panX = dpPanX; view.panY = dpPanY; }, [dpZoom, dpPanX, dpPanY]);
  React.useEffect(() => { const view = viViewRef.current; view.zoom = viZoom; view.panX = viPanX; view.panY = viPanY; }, [viZoom, viPanX, viPanY]);
  React.useEffect(() => { const view = fftViewRef.current; view.zoom = fftZoom; view.panX = fftPanX; view.panY = fftPanY; }, [fftZoom, fftPanX, fftPanY]);
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
      const [savedVminPct, savedVmaxPct] = fftPreAutoPctRef.current;
      setFftVminPct(savedVminPct); setFftVmaxPct(savedVmaxPct);
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

  const handleRootMouseDownCapture = React.useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement | null;
    if (target?.closest("canvas")) rootRef.current?.focus({ preventScroll: true });
  }, []);

  const handleKeyDown = React.useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    if (isTypingTarget(event.target)) return;

    const step = event.shiftKey ? 10 : 1;
    let handled = false;

    switch (event.key) {
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
      event.preventDefault();
      event.stopPropagation();
    }
  }, [
    compareMode, frameIdx, isTypingTarget, nFrames,
    posCol, posRow, setFrameIdx, setPosCol, setPosRow, shapeCols, shapeRows,
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
    queueScanPosition(row, col);
    if (commit) flushScanPosition();
  }, [flushScanPosition, queueScanPosition]);

  // Sync VI ROI local state
  React.useEffect(() => {
    if (!isDraggingViRoi && !isDraggingViRoiResize && !regionRequests.isPending()) {
      setLocalViRoiCenterRow(viRoiCenterRow ?? shapeRows / 2);
      setLocalViRoiCenterCol(viRoiCenterCol ?? shapeCols / 2);
    }
  }, [viRoiCenterRow, viRoiCenterCol, isDraggingViRoi, isDraggingViRoiResize, shapeRows, shapeCols]);

  // Canvas refs
  const dpCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const dpOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const dpUiRef = React.useRef<HTMLCanvasElement>(null);  // High-DPI UI overlay for scale bar
  const dpOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const dpImageDataRef = React.useRef<ImageData | null>(null);
  const virtualGpuCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const virtualGpuCaptureRef = React.useRef<(() => Promise<HTMLCanvasElement>) | null>(null);
  const virtualGpuContextRef = React.useRef<{ engine: GPUColormapEngine; context: GPUCanvasContext } | null>(null);
  const attachVirtualGpuCanvas = React.useCallback((canvas: HTMLCanvasElement | null) => {
    if (virtualGpuCanvasRef.current !== canvas) {
      virtualGpuContextRef.current?.context.unconfigure();
      virtualGpuContextRef.current = null;
      virtualGpuCaptureRef.current = null;
      virtualGpuCanvasRef.current = canvas;
    }
  }, []);
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

  // Offscreen version counters: bumped when colormap/data change, the cheap draw effects depend on them
  const [dpOffscreenVersion, setDpOffscreenVersion] = React.useState(0);
  const [viOffscreenVersion, setViOffscreenVersion] = React.useState(0);
  const [fftOffscreenVersion, setFftOffscreenVersion] = React.useState(0);

  // Cached colorbar vmin/vmax: computed in the expensive DP effect, reused by the UI overlay without recomputing
  const dpColorbarVminRef = React.useRef(0);
  const dpColorbarVmaxRef = React.useRef(1);

  // Device pixel ratio for high-DPI UI overlays
  const DPR = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  // ─────────────────────────────────────────────────────────────────────────
  // Effects: Canvas Rendering & Animation
  // ─────────────────────────────────────────────────────────────────────────

  // Prevent page scroll when scrolling on canvases
  // Re-run when showFft or the compare grid changes: both remount the overlays
  React.useEffect(() => {
    const preventDefault = (event: WheelEvent) => event.preventDefault();
    const overlays = [dpOverlayRef.current, virtualOverlayRef.current, fftOverlayRef.current];
    overlays.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => overlays.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, [showFft, compareMode]);

  // Store raw data for filtering/FFT
  const rawVirtualImageRef = React.useRef<Float32Array | null>(null);
  const fftMagnitudeRef = React.useRef<Float32Array | null>(null);
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);

  // Parse displayed image bytes into Float32Array and apply scale for histogram.
  // For DPC/SSB product maps this is a view into vi_product_maps_bytes, so source
  // switching reuses the static map payload instead of asking Python to resend it.
  React.useEffect(() => {
    if (!displayedVirtualImageBytes) return;
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
      const stats = computeStats(rawData);
      setViStats([stats.mean, stats.min, stats.max, stats.std]);
      if (isDpcGpuSource(activeViSource)) {
        const span = Math.max(Math.abs(stats.min), Math.abs(stats.max), 1e-12);
        setViDataMin(-span);
        setViDataMax(span);
      } else {
        setViDataMin(stats.min);
        setViDataMax(stats.max);
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
    const stats = computeStats(rawData);
    setViStats([stats.mean, stats.min, stats.max, stats.std]);
    if (isDpcGpuSource(activeViSource)) {
      const span = Math.max(Math.abs(stats.min), Math.abs(stats.max), 1e-12);
      setViDataMin(-span);
      setViDataMax(span);
    } else {
      setViDataMin(stats.min);
      setViDataMax(stats.max);
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
    if (dpRoiInteractiveRef.current || compareHistogramPendingSettleRef.current) return;
    const engine = viGpuColormapRef.current;
    if (!engine) return;
    const slotIndices = visibleCompareHistogramFrames
      .map((frame) => compareGpuSlotsRef.current.get(frame))
      .filter((slot): slot is number => slot !== undefined);
    if (slotIndices.length === 0) return;

    let cancelled = false;
    const generation = ++compareGpuHistogramGenRef.current;
    const isCurrent = () => !cancelled
      && generation === compareGpuHistogramGenRef.current
      && !dpRoiInteractiveRef.current && !compareHistogramPendingSettleRef.current;
    const timer = window.setTimeout(() => {
      void readSettledCompareHistogram(engine, slotIndices, viScaleMode === "log", isCurrent)
        .then(summary => {
          if (!summary || !isCurrent()) return;
          setViDataMin(summary.min);
          setViDataMax(summary.max);
          setViHistogramData(null);
          setViHistogramBins(summary.bins);
        })
        .catch(error => {
          if (isCurrent()) console.warn("Show4DSTEM multiple histogram update failed", error);
        });
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
    // These small CPU-colored canvases need CPU backing: on the verified
    // Chrome/NVIDIA raster path, default 2D writes can stay fully transparent
    // while the context reports no loss. Select the backing on first use.
    const offCtx = offscreen.getContext("2d", { willReadFrequently: true });
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
    setDpOffscreenVersion(version => version + 1);
  }, [displayedDpBytes, detRows, detCols, dpColormap, dpVminPct, dpVmaxPct, dpScaleMode, traitDpVmin, traitDpVmax]);

  // Cheap: zoom/pan redraw is one drawImage from the cached offscreen canvas.
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  React.useLayoutEffect(() => {
    const offscreen = dpOffscreenRef.current;
    if (!offscreen || !dpCanvasRef.current) return;
    const canvas = dpCanvasRef.current;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const identity = dpZoom === 1 && dpPanX === 0 && dpPanY === 0;
    const pixels = dpImageDataRef.current;
    if (identity && pixels && pixels.width === canvas.width && pixels.height === canvas.height) {
      // Exact pixel copy at identity view avoids unnecessary resampling.
      ctx.putImageData(pixels, 0, 0);
    } else {
      ctx.save();
      ctx.translate(dpPanX, dpPanY);
      ctx.scale(dpZoom, dpZoom);
      ctx.drawImage(offscreen, 0, 0);
      ctx.restore();
    }
  }, [dpOffscreenVersion, dpZoom, dpPanX, dpPanY]);

  // The DP overlay canvas only takes pointer events: crosshair, ROI shapes and
  // scale bar draw on the HiDPI dpUiRef canvas for crisp rendering.
  React.useEffect(() => {
    if (!dpOverlayRef.current) return;
    const canvas = dpOverlayRef.current;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, [localKCol, localKRow, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner, dpZoom, dpPanX, dpPanY, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, detRows, detCols]);

  // Expensive: VI colormap + data processing → cached offscreen canvas
  React.useEffect(() => {
    if (!rawVirtualImageRef.current) return;
    if (
      !compareMode
      && viGpuImageRef.current
      && activeViSource === viGpuImageRef.current.source
    ) {
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
    const range = findDataRange(scaled);
    let dataMin = range.min;
    let dataMax = range.max;
    if (isDpcGpuSource(activeViSource)) {
      const span = Math.max(Math.abs(range.min), Math.abs(range.max), 1e-12);
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
    setViOffscreenVersion(version => version + 1);
  }, [activeViSource, compareMode, displayedVirtualImageBytes, shapeRows, shapeCols, viGpuVersion, viColormap, viVminPct, viVmaxPct, viScaleMode, traitViVmin, traitViVmax, viAutoContrast]);

  // Present resident reductions directly, including while the pointer is held.
  // Offscreen bitmap snapshots can be black and can lag the latest detector. The
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

      const range = findDataRange(scaled);
      let dataMin = range.min;
      let dataMax = range.max;
      if (isDpcGpuSource(gpuImage.source)) {
        const span = Math.max(Math.abs(range.min), Math.abs(range.max), 1e-12);
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
    const canvas = virtualGpuCanvasRef.current;
    if (!canvas) return;
    let target = virtualGpuContextRef.current;
    if (!target || target.engine !== engine || canvas.width !== shapeCols || canvas.height !== shapeRows) {
      target?.context.unconfigure();
      const context = engine.configureCanvas(canvas, shapeCols, shapeRows);
      if (!context) { setViGpuRetainedReady(false); return; }
      context.configure({device: engine.getDevice(), format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "opaque", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC});
      target = { engine, context };
      virtualGpuContextRef.current = target;
    }
    const render = () => {
      let rendered = false;
      if (vmin != null && vmax != null) {
        rendered = engine.renderPanelSlotsDirectToCanvas(
          [gpuImage.slot], { vmin, vmax }, viScaleMode === "log", target.context,
          { width: shapeCols, height: shapeRows, panelCount: 1, cols: 1, rows: 1,
            gap: 0, bgRgb: 0,
            transforms: [{ zoom: viZoom, panX: viPanX, panY: viPanY }], smooth: viSmooth },
        );
      } else if (gpuImage.rangeMode === "gpu") {
        rendered = engine.renderSlotDirectWithGpuRangeToCanvas(
          gpuImage.slot, viVminPct, viVmaxPct, viScaleMode === "log", target.context,
          { width: shapeCols, height: shapeRows, bgRgb: 0,
            transform: { zoom: viZoom, panX: viPanX, panY: viPanY }, smooth: viSmooth },
        );
      }
      return rendered;
    };
    virtualGpuCaptureRef.current = () => captureGpuCanvas(engine.getDevice(), target.context, render);
    const rendered = render();
    setViGpuRetainedReady(rendered);
    if (rendered) {
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

  // Cheap: VI zoom/pan redraw is one drawImage from the cached offscreen canvas
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

  // The VI overlay canvas only takes pointer events: crosshair and scale bar
  // draw on the HiDPI viUiRef canvas.
  React.useEffect(() => {
    if (!virtualOverlayRef.current) return;
    const canvas = virtualOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, [localPosRow, localPosCol, isDraggingVI, viZoom, viPanX, viPanY, pixelSize, shapeRows, shapeCols]);

  // Compute FFT (expensive and async: only re-run on data/GPU changes)
  const fftRealRef = React.useRef<Float32Array | null>(null);
  const fftImagRef = React.useRef<Float32Array | null>(null);
  const fftRunSeqRef = React.useRef(0);
  const [fftVersion, setFftVersion] = React.useState(0);

  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !showFft) { setFftCropDims(null); return; }
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
      const cropCenterRow = localViRoiCenterRow ?? viRoiCenterRow;
      const cropCenterCol = localViRoiCenterCol ?? viRoiCenterCol;
      const crop = cropSingleROI(sourceData, shapeCols, shapeRows, viRoiMode, cropCenterRow, cropCenterCol, viRoiRadius, viRoiWidth, viRoiHeight);
      if (crop) {
        origCropW = crop.cropW;
        origCropH = crop.cropH;
        // Apply Hann window to crop at native dimensions BEFORE zero-padding
        if (fftWindow) applyHannWindow2D(crop.cropped, crop.cropW, crop.cropH);
        const padW = nextPow2(crop.cropW);
        const padH = nextPow2(crop.cropH);
        sourceData = zeroPadImage(crop.cropped, crop.cropH, crop.cropW, padH, padW);
        width = padW;
        height = padH;
      }
    }

    // Pre-pad non-power-of-2 full images so fft2d doesn't truncate frequency data
    if (!roiFftActive) {
      const padW = nextPow2(width);
      const padH = nextPow2(height);
      if (padW !== width || padH !== height) {
        sourceData = zeroPadImage(sourceData, height, width, padH, padW);
        width = padW;
        height = padH;
      }
    }

    const fftW = width, fftH = height;
    const commitFft = (real: Float32Array, imag: Float32Array) => {
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
      setFftVersion(version => version + 1);
    };
    const timer = window.setTimeout(() => {
      if (gpuFFTRef.current && gpuReady) {
        const runGpuFFT = async () => {
          const real = sourceData.slice();
          const imag = new Float32Array(real.length);
          const { real: spectrumReal, imag: spectrumImag } = await gpuFFTRef.current!.fft2D(real, imag, fftW, fftH, false);
          if (cancelled || runSeq !== fftRunSeqRef.current) return;
          fftshift(spectrumReal, fftW, fftH);
          fftshift(spectrumImag, fftW, fftH);
          commitFft(spectrumReal, spectrumImag);
        };
        runGpuFFT();
      } else {
        const runWorkerFFT = async () => {
          const real = sourceData.slice();
          const imag = new Float32Array(real.length);
          const result = await fft2dAsync(real, imag, fftW, fftH, false);
          commitFft(result.real, result.imag);
        };
        runWorkerFFT();
      }
    }, 16);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [displayedVirtualImageBytes, shapeRows, shapeCols, gpuReady, showFft, roiFftActive, viRoiMode, viRoiCenterRow, viRoiCenterCol, localViRoiCenterRow, localViRoiCenterCol, viRoiRadius, viRoiWidth, viRoiHeight, fftWindow]);

  // Expensive: FFT magnitude + histogram + colormap → cached offscreen canvas
  React.useEffect(() => {
    if (!fftRealRef.current || !fftImagRef.current) return;
    if (!showFft) return;

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
    setFftOffscreenVersion(version => version + 1);
  }, [showFft, fftVersion, fftScaleMode, fftAuto, fftVminPct, fftVmaxPct, fftColormap, shapeRows, shapeCols, fftCropDims]);

  // Cheap: FFT zoom/pan redraw is one drawImage from the cached offscreen canvas
  React.useLayoutEffect(() => {
    if (!fftCanvasRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const offscreen = fftOffscreenRef.current;
    if (!offscreen || !showFft) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
    const fftW = offscreen.width;
    const fftH = offscreen.height;
    const canvasW = canvas.width;
    const canvasH = canvas.height;
    // Stretch offscreen to fill canvas via the 9-arg drawImage form: ROI FFT crops produce a
    // small offscreen (e.g. 64×64) that would otherwise blit at native size in the corner.
    // Nearest neighbour keeps each FFT pixel a sharp block, as the pixelated canvas style does.
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(fftPanX, fftPanY);
    ctx.scale(fftZoom, fftZoom);
    ctx.drawImage(offscreen, 0, 0, fftW, fftH, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [fftOffscreenVersion, fftZoom, fftPanX, fftPanY, showFft]);

  // Render FFT overlay with d-spacing crosshair marker
  React.useEffect(() => {
    if (!fftOverlayRef.current) return;
    const canvas = fftOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // D-spacing crosshair marker
    if (fftClickInfo && showFft) {
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
      const armLength = 8 * Math.max(fftW, fftH) / 450;
      const gap = 3 * Math.max(fftW, fftH) / 450;
      const dotRadius = 4 * Math.max(fftW, fftH) / 450;
      ctx.beginPath();
      ctx.moveTo(screenX - armLength, screenY); ctx.lineTo(screenX - gap, screenY);
      ctx.moveTo(screenX + gap, screenY); ctx.lineTo(screenX + armLength, screenY);
      ctx.moveTo(screenX, screenY - armLength); ctx.lineTo(screenX, screenY - gap);
      ctx.moveTo(screenX, screenY + gap); ctx.lineTo(screenX, screenY + armLength);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(screenX, screenY, dotRadius, 0, Math.PI * 2);
      ctx.stroke();
      if (fftClickInfo.dSpacing != null) {
        const dSpacing = fftClickInfo.dSpacing;
        const label = dSpacing >= 10 ? `d = ${(dSpacing / 10).toFixed(2)} nm` : `d = ${dSpacing.toFixed(2)} \u00C5`;
        const fontSize = Math.max(10, Math.round(11 * Math.max(fftW, fftH) / 450));
        ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
        ctx.fillStyle = "white";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(label, screenX + armLength + 4, screenY - gap);
      }
      ctx.restore();
    }
  }, [fftZoom, fftPanX, fftPanY, showFft, fftClickInfo, shapeCols, shapeRows, fftCropDims]);

  // Clear FFT click info when virtual image changes (scan position, VI ROI, etc.)
  React.useEffect(() => {
    setFftClickInfo(null);
  }, [displayedVirtualImageBytes]);

  // ─────────────────────────────────────────────────────────────────────────
  // High-DPI Scale Bar UI Overlays
  // ─────────────────────────────────────────────────────────────────────────
  
  // DP scale bar + crosshair + ROI overlay + profile line (high-DPI)
  const drawDpUi = React.useCallback((center?: [number, number]) => {
    center ??= dpRoiInteractiveRef.current ? roiCenterPendingRef.current ?? undefined : undefined;
    // Read the same pending radii as mask construction, including repaints
    // caused by zoom/exposure while a handle is held. React commits on release.
    const outer = dpRoiInteractiveRef.current ? roiRadiusPendingRef.current ?? roiRadius : roiRadius;
    const inner = dpRoiInteractiveRef.current ? roiRadiusInnerPendingRef.current ?? roiRadiusInner : roiRadiusInner;
    if (!dpUiRef.current) return;
    const canvas = dpUiRef.current;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
    // Draw scale bar first when enabled.
    const kUnit = kCalibrated ? kPixelUnit : "px";
    if (showScaleBar) drawScaleBarHiDPI(canvas, DPR, dpZoom, kPixelSize || 1, kUnit, detCols);
    // Draw detector ROI only when the displayed virtual image is produced from
    // the live detector mask. Precomputed DPC/SSB maps are static products, so
    // showing the BF/ADF circle there is misleading.
    if (roiVirtualDetectorActive) {
      if (roiMode === "point") {
        drawDpCrosshairHiDPI(dpUiRef.current, DPR, center?.[1] ?? localKCol, center?.[0] ?? localKRow, dpZoom, dpPanX, dpPanY, detCols, detRows, isDraggingDP, roiColors);
      } else {
        drawRoiOverlayHiDPI(
          dpUiRef.current, DPR, roiMode,
          center?.[1] ?? localKCol, center?.[0] ?? localKRow, outer, inner, roiWidth, roiHeight,
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
        // profile points center pixel i at i (the sampler's convention); draw them on that pixel's center
        const toScreenX = (col: number) => maskToCanvas(col, dpZoom, dpPanX) * scaleX;
        const toScreenY = (row: number) => maskToCanvas(row, dpZoom, dpPanY) * scaleY;

        const ax = toScreenX(profilePoints[0].col);
        const ay = toScreenY(profilePoints[0].row);
        ctx.fillStyle = themeColors.accent;
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();

        if (profilePoints.length === 2) {
          const bx = toScreenX(profilePoints[1].col);
          const by = toScreenY(profilePoints[1].row);

          // A profile wider than one pixel averages a band; outline the averaged area.
          if (profileWidth > 1) {
            const colDelta = profilePoints[1].col - profilePoints[0].col;
            const rowDelta = profilePoints[1].row - profilePoints[0].row;
            const lineLength = Math.sqrt(colDelta * colDelta + rowDelta * rowDelta);
            if (lineLength > 0) {
              const halfWidth = (profileWidth - 1) / 2;
              const perpRow = -colDelta / lineLength * halfWidth;
              const perpCol = rowDelta / lineLength * halfWidth;
              ctx.fillStyle = themeColors.accent + "20";
              ctx.strokeStyle = themeColors.accent;
              ctx.lineWidth = 1;
              ctx.setLineDash([3, 3]);
              ctx.beginPath();
              ctx.moveTo(toScreenX(profilePoints[0].col + perpCol), toScreenY(profilePoints[0].row + perpRow));
              ctx.lineTo(toScreenX(profilePoints[1].col + perpCol), toScreenY(profilePoints[1].row + perpRow));
              ctx.lineTo(toScreenX(profilePoints[1].col - perpCol), toScreenY(profilePoints[1].row - perpRow));
              ctx.lineTo(toScreenX(profilePoints[0].col - perpCol), toScreenY(profilePoints[0].row - perpRow));
              ctx.closePath();
              ctx.fill();
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }

          ctx.strokeStyle = themeColors.accent;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();

          ctx.fillStyle = themeColors.accent;
          ctx.beginPath();
          ctx.arc(bx, by, 4, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }
    }

    // Colorbar overlay: uses cached vmin/vmax from the expensive DP offscreen effect
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
  React.useEffect(() => drawDpUi(), [drawDpUi]);
  
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
          const x0 = maskToCanvas(p0.col, viZoom, viPanX) * scaleX;
          const y0 = maskToCanvas(p0.row, viZoom, viPanY) * scaleY;
          ctx.beginPath();
          ctx.arc(x0, y0, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#fff";
          ctx.fillText("1", x0 + 6, y0 - 6);
        }
        if (viProfilePoints.length === 2) {
          const p0 = viProfilePoints[0], p1 = viProfilePoints[1];
          const x0 = maskToCanvas(p0.col, viZoom, viPanX) * scaleX;
          const y0 = maskToCanvas(p0.row, viZoom, viPanY) * scaleY;
          const x1 = maskToCanvas(p1.col, viZoom, viPanX) * scaleX;
          const y1 = maskToCanvas(p1.row, viZoom, viPanY) * scaleY;
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
      void sampleLineProfileBrowser(rawDpDataRef.current, detCols, detRows, p0.row, p0.col, p1.row, p1.col, profileWidth)
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
      void sampleLineProfileBrowser(rawViDataRef.current, shapeCols, shapeRows, p0.row, p0.col, p1.row, p1.col, 1)
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
    let distance: { total: number; unit: string } | null = null;
    if (profilePoints.length === 2) {
      const colDelta = profilePoints[1].col - profilePoints[0].col;
      const rowDelta = profilePoints[1].row - profilePoints[0].row;
      const lengthPx = Math.sqrt(colDelta * colDelta + rowDelta * rowDelta);
      distance = kCalibrated && kPixelSize > 0
        ? { total: lengthPx * kPixelSize, unit: kPixelUnit }
        : { total: lengthPx, unit: "px" };
    }
    const painted = drawProfileSparkline(canvas, ctx, canvasSize, profileHeight, profileData, distance,
      "Click two points on the DP to draw a profile", themeInfo.theme === "dark", themeColors.accent);
    profileBaseImageRef.current = painted?.base ?? null;
    profileLayoutRef.current = painted?.layout ?? null;
  }, [profileData, profilePoints, kPixelSize, kCalibrated, themeInfo.theme, themeColors.accent, canvasSize, profileHeight]);

  // DP Profile hover handlers
  const handleProfileMouseMove = React.useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    const layout = profileLayoutRef.current;
    if (!canvas || !base || !layout || !profileData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawProfileHover(ctx, base, layout, profileData, event.clientX - canvas.getBoundingClientRect().left,
      themeInfo.theme === "dark", themeColors.accent);
  }, [profileData, themeInfo.theme, themeColors.accent]);

  const handleProfileMouseLeave = React.useCallback(() => {
    restoreProfileBase(profileCanvasRef.current, profileBaseImageRef.current);
  }, []);

  // DP Profile resize handlers
  React.useEffect(() => {
    if (!isResizingProfile) return;
    const handleMouseMove = (event: MouseEvent) => {
      if (!profileResizeStart.current) return;
      const deltaY = event.clientY - profileResizeStart.current.startY;
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
    let distance: { total: number; unit: string } | null = null;
    if (viProfilePoints.length === 2 && pixelSize > 0) {
      const colDelta = viProfilePoints[1].col - viProfilePoints[0].col;
      const rowDelta = viProfilePoints[1].row - viProfilePoints[0].row;
      const lengthPx = Math.sqrt(colDelta * colDelta + rowDelta * rowDelta);
      distance = { total: lengthPx * pixelSize, unit: pixelUnit };
    }
    const painted = drawProfileSparkline(canvas, ctx, viCanvasWidth, viProfileHeight, viProfileData, distance,
      "Click two points on the VI to draw a profile", themeInfo.theme === "dark", themeColors.accent);
    viProfileBaseImageRef.current = painted?.base ?? null;
    viProfileLayoutRef.current = painted?.layout ?? null;
  }, [viProfileData, viProfilePoints, pixelSize, themeInfo.theme, themeColors.accent, viCanvasWidth, viProfileHeight]);

  // VI Profile hover handlers
  const handleViProfileMouseMove = React.useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = viProfileCanvasRef.current;
    const base = viProfileBaseImageRef.current;
    const layout = viProfileLayoutRef.current;
    if (!canvas || !base || !layout || !viProfileData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawProfileHover(ctx, base, layout, viProfileData, event.clientX - canvas.getBoundingClientRect().left,
      themeInfo.theme === "dark", themeColors.accent);
  }, [viProfileData, themeInfo.theme, themeColors.accent]);

  const handleViProfileMouseLeave = React.useCallback(() => {
    restoreProfileBase(viProfileCanvasRef.current, viProfileBaseImageRef.current);
  }, []);

  // VI Profile resize handlers
  React.useEffect(() => {
    if (!isResizingViProfile) return;
    const handleMouseMove = (event: MouseEvent) => {
      if (!viProfileResizeStart.current) return;
      const deltaY = event.clientY - viProfileResizeStart.current.startY;
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
  ) => (event: React.WheelEvent<HTMLCanvasElement>) => {
    event.stopPropagation();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (event.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (event.clientY - rect.top) * (canvas.height / rect.height);
    const view = viewRef.current;
    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, view.zoom * zoomFactor));
    const zoomRatio = newZoom / view.zoom;
    // Accumulate synchronously against the live ref (handles a burst of trackpad
    // wheel events within one frame correctly), flush to React state once per rAF.
    view.zoom = newZoom;
    view.panX = mouseX - (mouseX - view.panX) * zoomRatio;
    view.panY = mouseY - (mouseY - view.panY) * zoomRatio;
    if (view.raf === 0) {
      view.raf = requestAnimationFrame(() => {
        view.raf = 0;
        setZoom(view.zoom); setPanX(view.panX); setPanY(view.panY);
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

  // What a press grabs on the detector ROI: the inner half moves it, the outer
  // half and a hit margin past the edge resize it (grabDetector). The drawn
  // handle is a 6 px dot, too small to catch by hand on a binned detector.
  const grabDpRoi = (col: number, row: number): DetectorGrab => grabDetector({
    mode: roiMode as DetectorRoiMode,
    centerRow: activeRoiCenterRow,
    centerCol: activeRoiCenterCol,
    pointerRow: row,
    pointerCol: col,
    radius: Number(roiRadius) || 0,
    radiusInner: Number(roiRadiusInner) || 0,
    width: Number(roiWidth) || 0,
    height: Number(roiHeight) || 0,
    hitMargin: dpHitRadius,
  });
  const isNearResizeHandle = (col: number, row: number): boolean =>
    roiVirtualDetectorActive && grabDpRoi(col, row).action === "resize";

  // Helper: check if point is near the inner resize handle (annular mode only)
  const isNearResizeHandleInner = (col: number, row: number): boolean => {
    if (!roiVirtualDetectorActive) return false;
    if (roiMode !== "annular" || !roiRadiusInner) return false;
    const offset = roiRadiusInner * CIRCLE_HANDLE_ANGLE;
    const handleCol = activeRoiCenterCol + offset;
    const handleRow = activeRoiCenterRow + offset;
    const distance = Math.sqrt((col - handleCol) ** 2 + (row - handleRow) ** 2);
    return distance < dpHitRadius;
  };

  // What a press grabs on the scan ROI, with the detector's rules (grabDetector): the
  // inner half moves it, the outer half and a hit margin past the edge resize it.
  const viHitRadius = RESIZE_HIT_AREA_PX * Math.max(shapeRows, shapeCols) / canvasSize / viZoom;
  const grabViRoi = (row: number, col: number): DetectorGrab => grabDetector({
    mode: (viRoiMode || "off") as DetectorRoiMode,
    centerRow: localViRoiCenterRow,
    centerCol: localViRoiCenterCol,
    pointerRow: row,
    pointerCol: col,
    radius: viRoiRadius || 5,
    radiusInner: 0,
    width: viRoiWidth || 10,
    height: viRoiHeight || 10,
    hitMargin: viHitRadius,
  });

  // Mouse handlers. imageCol/imageRow are image coordinates where pixel i spans [i, i + 1)
  // (value readout); maskCol/maskRow are the detector mask's, where pixel i is
  // centered at i (ROIs and profile points, which the bilinear sampler reads with
  // the same convention), so a ROI or profile point dropped on a pixel centers on it.
  const getDpImageCoordsFromClient = React.useCallback((clientX: number, clientY: number): { imageCol: number; imageRow: number; maskCol: number; maskRow: number } | null => {
    const canvas = dpOverlayRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const screenX = (clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (clientY - rect.top) * (canvas.height / rect.height);
    return {
      imageCol: (screenX - dpPanX) / dpZoom,
      imageRow: (screenY - dpPanY) / dpZoom,
      maskCol: canvasToMask(screenX, dpZoom, dpPanX),
      maskRow: canvasToMask(screenY, dpZoom, dpPanY),
    };
  }, [dpPanX, dpPanY, dpZoom]);

  const resizeDpRoiFromImagePoint = React.useCallback((col: number, row: number, shiftKey: boolean = false): boolean => {
    const start = resizeStartRef.current;
    if (isDraggingResizeInner && start) {
      const geometry = resizeDetectorFromPointer({
        mode: roiMode as DetectorRoiMode,
        centerRow: activeRoiCenterRow,
        centerCol: activeRoiCenterCol,
        pointerRow: row,
        pointerCol: col,
        start,
        resizeInner: true,
      });
      if (geometry?.radiusInner === undefined) return false;
      sendRoiRadius(geometry.radiusInner, "inner");
      requestCompareViLive();
      return true;
    }

    if (isDraggingResize && start) {
      const geometry = resizeDetectorFromPointer({
        mode: roiMode as DetectorRoiMode,
        centerRow: activeRoiCenterRow,
        centerCol: activeRoiCenterCol,
        pointerRow: row,
        pointerCol: col,
        start,
        preserveAspect: shiftKey,
      });
      if (!geometry) return false;
      if (roiMode === "rect") {
        setRoiWidth(geometry.width!);
        setRoiHeight(geometry.height!);
      } else {
        sendRoiRadius(geometry.radius!);
      }
      requestCompareViLive();
      return true;
    }

    return false;
  }, [
    activeRoiCenterCol, activeRoiCenterRow, isDraggingResize, isDraggingResizeInner,
    model, requestCompareViLive, roiMode, sendRoiRadius, setRoiHeight, setRoiWidth
  ]);

  const handleDpMouseDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const coords = getDpImageCoordsFromClient(event.clientX, event.clientY);
    if (!coords || !dpPointerOwner.start(event)) return;
    const { maskCol, maskRow } = coords;
    // Capture the pointer so a fast edge-drag resize keeps receiving move/up
    // events even when the cursor leaves the canvas (#751). Without capture the
    // window listener can miss events and the radius never updates.
    if ("pointerId" in event) {
      try { event.currentTarget.setPointerCapture(event.pointerId); } catch {}
    }
    dpClickStartRef.current = { x: event.clientX, y: event.clientY };

    // When profile mode is active, use profile interactions only
    if (profileActive) {
      if (profilePoints.length === 2) {
        const p0 = profilePoints[0];
        const p1 = profilePoints[1];
        const hitRadius = 10 / dpZoom;
        const d0 = Math.sqrt((maskCol - p0.col) ** 2 + (maskRow - p0.row) ** 2);
        const d1 = Math.sqrt((maskCol - p1.col) ** 2 + (maskRow - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingDpProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingDP(false);
          return;
        }
        if (pointToSegmentDistance(maskCol, maskRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
          setIsDraggingDpProfileLine(true);
          dpProfileDragStartRef.current = {
            row: maskRow,
            col: maskCol,
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

    beginDpRoiInteraction();

    // Inner handle first, then the outer resize band. A resize remembers the
    // press so the edge follows the pointer from where it was grabbed.
    const pressed = {
      pointerRow: maskRow,
      pointerCol: maskCol,
      radius: roiRadiusPendingRef.current ?? Number(model.get("roi_radius")),
      radiusInner: roiRadiusInnerPendingRef.current ?? Number(model.get("roi_radius_inner") || 0),
      width: Number(roiWidth) || 0,
      height: Number(roiHeight) || 0,
    };
    if (isNearResizeHandleInner(maskCol, maskRow)) {
      resizeStartRef.current = { ...pressed, rows: true, cols: true };
      setIsDraggingResizeInner(true);
      return;
    }
    const grab = grabDpRoi(maskCol, maskRow);
    if (grab.action === "resize") {
      event.preventDefault();
      resizeStartRef.current = { ...pressed, rows: grab.rows, cols: grab.cols };
      setIsDraggingResize(true);
      return;
    }

    setIsDraggingDP(true);
    // If clicking inside the ROI's move zone, drag with offset (grab-and-drag)
    if (grab.action === "move") {
      dpDragOffsetRef.current = { dRow: maskRow - activeRoiCenterRow, dCol: maskCol - activeRoiCenterCol };
      return;
    }
    // Clicking outside the ROI teleports its center to the click position
    dpDragOffsetRef.current = { dRow: 0, dCol: 0 };
    setLocalKCol(maskCol); setLocalKRow(maskRow);
    // Use compound roi_center trait [row, col] - single observer fires in Python
    const { row: newRow, col: newCol } = clampDetectorCenter(
      maskRow,
      maskCol,
      detRows,
      detCols,
    );
    model.set("roi_active", true);
    writeRoiCenterModel(newRow, newCol);
    requestCompareViLive();
  };

  const handleDpMouseMove = (event: PointerEvent | React.PointerEvent<HTMLCanvasElement>) => {
    const pointerMove = dpPointerOwner.move(event);
    if (pointerMove === "ignore") return;
    if (pointerMove === "release") { handleDpMouseUp(event); return; }
    const coords = getDpImageCoordsFromClient(event.clientX, event.clientY);
    if (!coords) return;
    const { imageCol, imageRow, maskCol, maskRow } = coords;

    // Fast path: skip cursor readout during any active drag, which avoids setCursorInfo re-renders
    const anyDrag = isDraggingDP || isDraggingResize || isDraggingResizeInner
      || draggingDpProfileEndpoint !== null || isDraggingDpProfileLine;
    if (anyDrag && pointerMove !== "drag") return;

    // Cursor readout: look up raw DP value at pixel position
    if (!anyDrag) {
      const pxCol = Math.floor(imageCol);
      const pxRow = Math.floor(imageRow);
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
      const d0 = Math.sqrt((maskCol - p0.col) ** 2 + (maskRow - p0.row) ** 2);
      const d1 = Math.sqrt((maskCol - p1.col) ** 2 + (maskRow - p1.row) ** 2);
      if (draggingDpProfileEndpoint !== null) {
        if (!rawDpDataRef.current) return;
        const clampedRow = Math.max(0, Math.min(detRows - 1, maskRow));
        const clampedCol = Math.max(0, Math.min(detCols - 1, maskCol));
        const next = [
          draggingDpProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
          draggingDpProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
        ];
        setProfileLine(next);
        void sampleLineProfileBrowser(rawDpDataRef.current, detCols, detRows, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth)
          .then(setProfileData)
          .catch(error => console.error("[Show4DSTEM] WebGPU DP profile failed", error));
        return;
      }
      if (isDraggingDpProfileLine && dpProfileDragStartRef.current) {
        if (!rawDpDataRef.current) return;
        const next = dragProfileLine(dpProfileDragStartRef.current, maskRow, maskCol, detRows, detCols);
        setProfileLine(next);
        void sampleLineProfileBrowser(rawDpDataRef.current, detCols, detRows, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth)
          .then(setProfileData)
          .catch(error => console.error("[Show4DSTEM] WebGPU DP profile failed", error));
        return;
      }
      const nextHoveredEndpoint: 0 | 1 | null = d0 <= hitRadius ? 0 : d1 <= hitRadius ? 1 : null;
      const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(maskCol, maskRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius;
      setHoveredDpProfileEndpoint(nextHoveredEndpoint);
      setIsHoveringDpProfileLine(nextHoverLine);
      return;
    } else {
      if (hoveredDpProfileEndpoint !== null) setHoveredDpProfileEndpoint(null);
      if (isHoveringDpProfileLine) setIsHoveringDpProfileLine(false);
    }

    // Handle inner resize dragging (annular mode)
    if (roiVirtualDetectorActive && resizeDpRoiFromImagePoint(maskCol, maskRow, event.shiftKey)) {
      return;
    }

    if (!roiVirtualDetectorActive) {
      if (isHoveringResize) setIsHoveringResize(false);
      if (isHoveringResizeInner) setIsHoveringResizeInner(false);
      return;
    }

    // Check hover state for resize handles
    if (!isDraggingDP) {
      setIsHoveringResizeInner(isNearResizeHandleInner(maskCol, maskRow));
      setIsHoveringResize(isNearResizeHandle(maskCol, maskRow));
      return;
    }

    const centerCol = maskCol - dpDragOffsetRef.current.dCol;
    const centerRow = maskRow - dpDragOffsetRef.current.dRow;
    setLocalKCol(centerCol); setLocalKRow(centerRow);
    // rAF-coalesced: sends only the latest roi_center per frame.
    const boundedCol = Math.max(0, Math.min(detCols - 1, centerCol));
    const boundedRow = Math.max(0, Math.min(detRows - 1, centerRow));
    // Area detectors retain subpixel centers; point detectors select one pixel.
    const newCol = roiMode === "point" ? Math.round(boundedCol) : boundedCol;
    const newRow = roiMode === "point" ? Math.round(boundedRow) : boundedRow;
    queueRoiCenter(newRow, newCol);
    requestCompareViLive();
  };

  const handleDpMouseUp = (event: PointerEvent | React.PointerEvent<HTMLCanvasElement>) => {
    if (!dpPointerOwner.release(event)) return;
    finishDpRoiInteraction();
    if (draggingDpProfileEndpoint !== null || isDraggingDpProfileLine) {
      setDraggingDpProfileEndpoint(null);
      setIsDraggingDpProfileLine(false);
      dpProfileDragStartRef.current = null;
      dpClickStartRef.current = null;
      setIsDraggingDP(false);
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      setHoveredDpProfileEndpoint(null);
      setIsHoveringDpProfileLine(false);
      return;
    }

    // Profile click capture
    if (event.type === "pointerup" && profileActive && dpClickStartRef.current) {
      const dx = event.clientX - dpClickStartRef.current.x;
      const dy = event.clientY - dpClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = dpOverlayRef.current;
        if (canvas && rawDpDataRef.current) {
          const rect = canvas.getBoundingClientRect();
          const screenX = (event.clientX - rect.left) * (canvas.width / rect.width);
          const screenY = (event.clientY - rect.top) * (canvas.height / rect.height);
          const imgCol = (screenX - dpPanX) / dpZoom;
          const imgRow = (screenY - dpPanY) / dpZoom;
          if (imgCol >= 0 && imgCol < detCols && imgRow >= 0 && imgRow < detRows) {
            // profile points use the mask convention the sampler reads: a click on a pixel center is that pixel's index
            const point = {
              row: Math.max(0, Math.min(detRows - 1, canvasToMask(screenY, dpZoom, dpPanY))),
              col: Math.max(0, Math.min(detCols - 1, canvasToMask(screenX, dpZoom, dpPanX))),
            };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              setProfileLine([point]);
              setProfileData(null);
            } else {
              const p0 = profilePoints[0];
              setProfileLine([p0, point]);
              void sampleLineProfileBrowser(rawDpDataRef.current, detCols, detRows, p0.row, p0.col, point.row, point.col, profileWidth)
                .then(setProfileData)
                .catch(error => console.error("[Show4DSTEM] WebGPU DP profile failed", error));
            }
          }
        }
      }
    }
    dpClickStartRef.current = null;
    setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false);
    setDraggingDpProfileEndpoint(null);
    setIsDraggingDpProfileLine(false);
    setHoveredDpProfileEndpoint(null);
    setIsHoveringDpProfileLine(false);
    dpProfileDragStartRef.current = null;
  };
  const handleDpMouseLeave = () => {
    // Capture can be lost when the canvas is replaced. The owner-filtered
    // window handlers continue the gesture outside it until real up/cancel.
    if (dpPointerOwner.active) return;
    dpClickStartRef.current = null;
    finishDpRoiInteraction();
    setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false);
    setDraggingDpProfileEndpoint(null);
    setIsDraggingDpProfileLine(false);
    setHoveredDpProfileEndpoint(null);
    setIsHoveringDpProfileLine(false);
    dpProfileDragStartRef.current = null;
    setIsHoveringResize(false); setIsHoveringResizeInner(false);
    setCursorInfo(prev => prev?.panel === "DP" ? null : prev);
  };
  // Keep global delivery on the latest committed geometry/state, without
  // detaching listeners during drag rerenders or processing canvas events twice.
  const dpPointerHandlersRef = React.useRef({ move: handleDpMouseMove, up: handleDpMouseUp });
  React.useLayoutEffect(() => {
    dpPointerHandlersRef.current = { move: handleDpMouseMove, up: handleDpMouseUp };
  });
  React.useEffect(() => {
    const onMove = (event: PointerEvent) => {
      if (!dpPointerOwner.owns(event) || event.target === dpOverlayRef.current) return;
      dpPointerHandlersRef.current.move(event);
      event.preventDefault();
    };
    const onUp = (event: PointerEvent) => {
      if (dpPointerOwner.owns(event)) dpPointerHandlersRef.current.up(event);
    };
    window.addEventListener("pointermove", onMove, { passive: false });
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      dpPointerOwner.reset();
    };
  }, [dpPointerOwner]);

  // Set a panel's zoom/pan in its live view ref (read by wheel and touch gestures) and in React state.
  const setPanelView = (
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

  const handleDpDoubleClick = () => setPanelView(dpViewRef, setDpZoom, setDpPanX, setDpPanY, 1, 0, 0);

  const handleViMouseDown = (event: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
    // Capture the pointer so a touch/mouse probe-drag keeps receiving move/up
    // events even when the finger leaves the small canvas. Needed for mobile
    // parity: touchscreens deliver these as pointer events (mirrors DP #751).
    if ("pointerId" in event) {
      try { event.currentTarget.setPointerCapture(event.pointerId); } catch {}
    }
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (event.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (event.clientY - rect.top) * (canvas.height / rect.height);
    // scan_indices and pos_row/pos_col center scan pixel i at i (the image draws it over [i, i + 1))
    const maskRow = canvasToMask(screenY, viZoom, viPanY);
    const maskCol = canvasToMask(screenX, viZoom, viPanX);

    // VI Profile mode - click to set points
    if (viProfileActive) {
      viClickStartRef.current = { x: screenX, y: screenY };
      if (viProfilePoints.length === 2) {
        const p0 = viProfilePoints[0];
        const p1 = viProfilePoints[1];
        const hitRadius = 10 / viZoom;
        const d0 = Math.sqrt((maskCol - p0.col) ** 2 + (maskRow - p0.row) ** 2);
        const d1 = Math.sqrt((maskCol - p1.col) ** 2 + (maskRow - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingViProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingVI(false);
          return;
        }
        if (pointToSegmentDistance(maskCol, maskRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
          setIsDraggingViProfileLine(true);
          viProfileDragStartRef.current = {
            row: maskRow,
            col: maskCol,
            p0: { row: p0.row, col: p0.col },
            p1: { row: p1.row, col: p1.col },
          };
          setIsDraggingVI(false);
          return;
        }
      }
      return;
    }

    // Scan ROI active: resize from its edge, move from inside, otherwise teleport.
    // A resize remembers the press so the edge follows the pointer from where it was grabbed.
    if (viRoiMode && viRoiMode !== "off") {
      const grab = grabViRoi(maskRow, maskCol);
      if (grab.action === "resize") {
        viResizeStartRef.current = {
          pointerRow: maskRow,
          pointerCol: maskCol,
          radius: viRoiRadius || 5,
          radiusInner: 0,
          width: viRoiWidth || 10,
          height: viRoiHeight || 10,
          rows: grab.rows,
          cols: grab.cols,
        };
        setIsDraggingViRoiResize(true);
        return;
      }
      setIsDraggingViRoi(true);
      if (grab.action === "move") {
        viRoiDragOffsetRef.current = { dRow: maskRow - localViRoiCenterRow, dCol: maskCol - localViRoiCenterCol };
      } else {
        viRoiDragOffsetRef.current = { dRow: 0, dCol: 0 };
        setLocalViRoiCenterRow(maskRow);
        setLocalViRoiCenterCol(maskCol);
        setViRoiCenterRow(Math.round(Math.max(0, Math.min(shapeRows - 1, maskRow))));
        setViRoiCenterCol(Math.round(Math.max(0, Math.min(shapeCols - 1, maskCol))));
      }
      return;
    }

    // Regular position selection (when ROI is off)
    setIsDraggingVI(true);
    // Snap to the scan pixel under the pointer so the crosshair marks the exact
    // pixel the CBED is sampled from (not the fractional cursor position).
    const scanRow = Math.round(Math.max(0, Math.min(shapeRows - 1, maskRow)));
    const scanCol = Math.round(Math.max(0, Math.min(shapeCols - 1, maskCol)));
    updateScanPosition(scanRow, scanCol);
  };

  const handleViMouseMove = (event: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (event.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (event.clientY - rect.top) * (canvas.height / rect.height);
    const imageRow = (screenY - viPanY) / viZoom;
    const imageCol = (screenX - viPanX) / viZoom;
    const maskRow = canvasToMask(screenY, viZoom, viPanY);
    const maskCol = canvasToMask(screenX, viZoom, viPanX);

    // Fast path: skip cursor readout during any active drag, which avoids setCursorInfo re-renders
    const anyViDrag = isDraggingVI || isDraggingViRoi || isDraggingViRoiResize
      || draggingViProfileEndpoint !== null || isDraggingViProfileLine;

    // Cursor readout: look up raw VI value at pixel position
    if (!anyViDrag) {
      const pxRow = Math.floor(imageRow);
      const pxCol = Math.floor(imageCol);
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
      const d0 = Math.sqrt((maskCol - p0.col) ** 2 + (maskRow - p0.row) ** 2);
      const d1 = Math.sqrt((maskCol - p1.col) ** 2 + (maskRow - p1.row) ** 2);
      if (draggingViProfileEndpoint !== null) {
        const clampedRow = Math.max(0, Math.min(shapeRows - 1, maskRow));
        const clampedCol = Math.max(0, Math.min(shapeCols - 1, maskCol));
        const next = [
          draggingViProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : viProfilePoints[0],
          draggingViProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : viProfilePoints[1],
        ];
        setViProfilePoints(next);
        return;
      }
      if (isDraggingViProfileLine && viProfileDragStartRef.current) {
        const next = dragProfileLine(viProfileDragStartRef.current, maskRow, maskCol, shapeRows, shapeCols);
        setViProfilePoints(next);
        return;
      }
      const nextHoveredEndpoint: 0 | 1 | null = d0 <= hitRadius ? 0 : d1 <= hitRadius ? 1 : null;
      const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(maskCol, maskRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius;
      setHoveredViProfileEndpoint(nextHoveredEndpoint);
      setIsHoveringViProfileLine(nextHoverLine);
      return;
    } else {
      if (hoveredViProfileEndpoint !== null) setHoveredViProfileEndpoint(null);
      if (isHoveringViProfileLine) setIsHoveringViProfileLine(false);
    }

    // Scan ROI resize: the edge follows the pointer from the press (resizeDetectorFromPointer), in whole scan pixels
    if (isDraggingViRoiResize && viResizeStartRef.current) {
      const geometry = resizeDetectorFromPointer({
        mode: viRoiMode as DetectorRoiMode,
        centerRow: localViRoiCenterRow,
        centerCol: localViRoiCenterCol,
        pointerRow: maskRow,
        pointerCol: maskCol,
        start: viResizeStartRef.current,
        preserveAspect: event.shiftKey,
      });
      if (geometry?.width !== undefined && geometry.height !== undefined) {
        setViRoiWidth(Math.max(2, Math.round(geometry.width)));
        setViRoiHeight(Math.max(2, Math.round(geometry.height)));
      } else if (geometry?.radius !== undefined) {
        setViRoiRadius(Math.max(1, Math.round(geometry.radius)));
      }
      return;
    }

    // Check hover state for resize handles (same as DP)
    if (!isDraggingViRoi) {
      setIsHoveringViRoiResize(grabViRoi(maskRow, maskCol).action === "resize");
      if (viRoiMode && viRoiMode !== "off") return;  // Don't update position when ROI active
    }

    // VI ROI center drag keeps the press offset, as the DP ROI drag does
    if (isDraggingViRoi) {
      const centerRow = maskRow - viRoiDragOffsetRef.current.dRow;
      const centerCol = maskCol - viRoiDragOffsetRef.current.dCol;
      setLocalViRoiCenterRow(centerRow);
      setLocalViRoiCenterCol(centerCol);
      // Compound trait update: a single observer fires Python-side, so the reduced DP is
      // never computed against split-trait state (old col + new row, or vice versa).
      const roiRow = Math.round(Math.max(0, Math.min(shapeRows - 1, centerRow)));
      const roiCol = Math.round(Math.max(0, Math.min(shapeCols - 1, centerCol)));
      model.set("vi_roi_center", [roiRow, roiCol]);
      model.save_changes();
      return;
    }

    // Handle regular position dragging (when ROI is off)
    if (!isDraggingVI) return;
    // Snap to the scan pixel under the pointer so the crosshair tracks discrete
    // sampled positions, matching the CBED actually shown.
    const scanRow = Math.round(Math.max(0, Math.min(shapeRows - 1, maskRow)));
    const scanCol = Math.round(Math.max(0, Math.min(shapeCols - 1, maskCol)));
    updateScanPosition(scanRow, scanCol);
  };

  const handleViMouseUp = (event: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
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
        const endX = (event.clientX - rect.left) * (canvas.width / rect.width);
        const endY = (event.clientY - rect.top) * (canvas.height / rect.height);
        const dx = endX - viClickStartRef.current.x;
        const dy = endY - viClickStartRef.current.y;
        const wasDrag = Math.sqrt(dx * dx + dy * dy) > 3;

        if (!wasDrag) {
          // Click to add a point on the scan pixel under the pointer (pixel i centered at i, as the sampler reads it)
          const point = {
            row: Math.round(Math.max(0, Math.min(shapeRows - 1, canvasToMask(endY, viZoom, viPanY)))),
            col: Math.round(Math.max(0, Math.min(shapeCols - 1, canvasToMask(endX, viZoom, viPanX)))),
          };
          if (viProfilePoints.length < 2) {
            setViProfilePoints([...viProfilePoints, point]);
          } else {
            setViProfilePoints([point]);
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
  const handleViDoubleClick = () => setPanelView(viViewRef, setViZoom, setViPanX, setViPanY, 1, 0, 0);
  const handleFftDoubleClick = () => {
    setPanelView(fftViewRef, setFftZoom, setFftPanX, setFftPanY, 1, 0, 0);
    setFftClickInfo(null);
  };

  const touchDistance = (first: React.Touch, second: React.Touch): number => {
    return Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY);
  };

  const touchMidpoint = (first: React.Touch, second: React.Touch): { x: number; y: number } => {
    return { x: (first.clientX + second.clientX) / 2, y: (first.clientY + second.clientY) / 2 };
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

  const handlePanelTouchStart = (kind: TouchPanelKind) => (event: React.TouchEvent<HTMLCanvasElement>) => {
    const refs = getTouchPanelRefs(kind);
    const canvas = refs.canvasRef.current;
    if (!canvas) return;

    if (event.touches.length === 1) {
      const now = window.performance.now();
      const previousTap = lastTapRef.current;
      lastTapRef.current = { kind, time: now };
      if (previousTap && previousTap.kind === kind && now - previousTap.time < 320) {
        event.preventDefault();
        refs.reset();
        touchTransformRef.current = null;
        return;
      }

      if (kind !== "fft" && refs.viewRef.current.zoom <= 1) {
        touchTransformRef.current = null;
        return;
      }

      const touch = event.touches[0];
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
      event.preventDefault();
      return;
    }

    if (event.touches.length >= 2) {
      const first = event.touches[0];
      const second = event.touches[1];
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
      event.preventDefault();
    }
  };

  const handlePanelTouchMove = (kind: TouchPanelKind) => (event: React.TouchEvent<HTMLCanvasElement>) => {
    const state = touchTransformRef.current;
    if (!state || state.kind !== kind) return;
    const refs = getTouchPanelRefs(kind);
    const canvas = refs.canvasRef.current;
    if (!canvas) return;

    if (state.mode === "pan" && event.touches.length === 1) {
      const touch = event.touches[0];
      const rect = canvas.getBoundingClientRect();
      const dx = (touch.clientX - state.startX) * (canvas.width / rect.width);
      const dy = (touch.clientY - state.startY) * (canvas.height / rect.height);
      setPanelView(
        refs.viewRef,
        refs.setZoom,
        refs.setPanX,
        refs.setPanY,
        state.startZoom,
        state.startPanX + dx,
        state.startPanY + dy,
      );
      event.preventDefault();
      return;
    }

    if (state.mode === "pinch" && event.touches.length >= 2) {
      const first = event.touches[0];
      const second = event.touches[1];
      const midpoint = touchMidpoint(first, second);
      const startCanvasPoint = canvasPointFromClient(canvas, state.startMidX, state.startMidY);
      const currentCanvasPoint = canvasPointFromClient(canvas, midpoint.x, midpoint.y);
      const ratio = state.startDistance > 0 ? touchDistance(first, second) / state.startDistance : 1;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, state.startZoom * ratio));
      const imageX = (startCanvasPoint.x - state.startPanX) / state.startZoom;
      const imageY = (startCanvasPoint.y - state.startPanY) / state.startZoom;
      setPanelView(
        refs.viewRef,
        refs.setZoom,
        refs.setPanX,
        refs.setPanY,
        newZoom,
        currentCanvasPoint.x - imageX * newZoom,
        currentCanvasPoint.y - imageY * newZoom,
      );
      event.preventDefault();
    }
  };

  const handlePanelTouchEnd = (event: React.TouchEvent<HTMLCanvasElement>) => {
    if (event.touches.length === 0) {
      touchTransformRef.current = null;
    }
  };

  // FFT drag-to-pan handlers
  const handleFftMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    fftClickStartRef.current = { x: event.clientX, y: event.clientY };
    setIsDraggingFFT(true);
    setFftDragStart({ x: event.clientX, y: event.clientY, panX: fftPanX, panY: fftPanY });
  };

  const handleFftMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDraggingFFT || !fftDragStart) return;
    const canvas = fftOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const dx = (event.clientX - fftDragStart.x) * scaleX;
    const dy = (event.clientY - fftDragStart.y) * scaleY;
    setFftPanX(fftDragStart.panX + dx);
    setFftPanY(fftDragStart.panY + dy);
  };

  const handleFftMouseUp = async (event: React.MouseEvent<HTMLCanvasElement>) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = event.clientX - fftClickStartRef.current.x;
      const dy = event.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        // Convert screen coords to FFT image coords
        const canvas = fftOverlayRef.current;
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const scaleX = canvas.width / rect.width;
          const scaleY = canvas.height / rect.height;
          const canvasX = (event.clientX - rect.left) * scaleX;
          const canvasY = (event.clientY - rect.top) * scaleY;
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
                snapped = await findFFTPeakBrowser(fftMagCacheRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
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
  const handleCanvasResizeStart = (event: React.MouseEvent) => {
    event.stopPropagation();
    event.preventDefault();
    setIsResizingCanvas(true);
    setResizeCanvasStart({ x: event.clientX, y: event.clientY, size: canvasSize });
  };

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    let rafId = 0;
    let latestSize = resizeCanvasStart ? resizeCanvasStart.size : canvasSize;
    const handleMouseMove = (event: MouseEvent) => {
      if (!resizeCanvasStart) return;
      const delta = Math.max(event.clientX - resizeCanvasStart.x, event.clientY - resizeCanvasStart.y);
      latestSize = Math.max(MIN_CANVAS_SIZE, resizeCanvasStart.size + delta);
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

  const handleCompareGridResizeStart = React.useCallback((event: React.PointerEvent<HTMLElement>, panelScale = 1) => {
    event.stopPropagation();
    event.preventDefault();
    try { event.currentTarget.setPointerCapture(event.pointerId); } catch {}
    compareGridResizeCleanupRef.current?.();
    const startX = event.clientX;
    const startY = event.clientY;
    const startWidth = compareGridWidth;
    const resizeScale = Math.max(1, Number.isFinite(panelScale) ? panelScale : 1);
    let rafId = 0;
    let latestWidth = startWidth;
    const handlePointerMove = (event: PointerEvent) => {
      const delta = Math.max(event.clientX - startX, event.clientY - startY);
      latestWidth = Math.max(MIN_COMPARE_GRID_WIDTH, startWidth + delta * resizeScale);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setCompareGridPreviewWidth(latestWidth);
        });
      }
      event.preventDefault();
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
  }, [compareGridWidth, setCompareGridWidthPx]);

  const selectCompareFrame = React.useCallback((idx: number) => {
    setFrameIdx(Math.max(0, Math.min(nFrames - 1, idx)));
  }, [nFrames, setFrameIdx]);
  const setCompareGpuRenderer = React.useCallback((renderNow: (() => number) | null) => {
    compareGpuRenderNowRef.current = renderNow;
  }, []);

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
  const ssbBfCountText = Number(ssbComputeBfPixels || 0) > 0
    ? `${Math.max(0, Math.round(Number(ssbComputeBfPixels || 0)))} BF px`
    : "BF count appears after first run";
  const hasSsbCalibrationDownload = String(ssbComputeCalibrationJson || "").trim().length > 0;
  const ssbProgressText = String(ssbComputeStatus || "").trim()
    || (ssbComputeBusy ? "Running SSB..." : "");
  const ssbStatusIsFailure = ssbProgressText.startsWith("SSB failed");
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
  const getActiveViCanvas = React.useCallback(async (): Promise<HTMLCanvasElement | null> => {
    return viGpuVisible && virtualGpuCaptureRef.current
      ? virtualGpuCaptureRef.current() : virtualCanvasRef.current;
  }, [viGpuVisible]);
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
  // Without WebGPU the saved export views stay readable: the chip sits on top
  // of them instead of a frosted cover.
  const renderPanelLoadingOverlay = React.useCallback((label: string, detail = "") => (
    <Box
      data-show4dstem-panel-loading="true"
      data-quantem-load-error={/\bfailed\b/i.test(label) ? "true" : undefined}
      sx={label === "Needs WebGPU"
        ? { ...panelLoadingOverlaySx, bgcolor: "transparent", backdropFilter: "none", alignItems: "flex-start", pt: 1 }
        : panelLoadingOverlaySx}
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
  const fftPanelLoading = offlineBackendLoading && showFft;
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
        {...{ webkitdirectory: "", directory: "" }}
      />
      {/* HEADER */}
      {showTitle && <Typography variant="h6" sx={{ ...themedTypography.title, mb: `${SPACING.SM}px` }}>
        {title || "4D-STEM Explorer"}
        <RenderPathBadge colors={themeColors} />
        {nFrames > 1 && <span style={{ fontWeight: "normal", fontSize: 13, marginLeft: 8, opacity: 0.7 }}>({frameLabels && frameLabels.length > frameIdx ? frameLabels[frameIdx] : `${frameDimLabel} ${frameIdx + 1}/${nFrames}`})</span>}
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
            ...themedTypography.label,
            color: offlineStatusIsError ? "#d32f2f" : themeColors.textMuted,
            width: "fit-content",
            maxWidth: "100%",
            lineHeight: 1.35,
            overflowWrap: "anywhere",
          }}
        >
          {offlineBackendError === OFFLINE_NEEDS_WEBGPU ? offlineStatusText : offlineStatusIsError ? `Show4DSTEM load failed: ${offlineStatusText}` : offlineStatusText}
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
          <Typography sx={{ ...themedTypography.label, color: themeColors.text }}>
            {localH5FolderName
              ? `No server needed - click Open data folder, then select "${localH5FolderName}".`
              : "No server needed - grant this page access to its exported HDF5 folder."}
          </Typography>
          {h5LocalSourceStatus && (
            <Typography sx={{ ...themedTypography.label, color: h5LocalSourceStatus.includes("No ") ? "#d32f2f" : themeColors.textMuted }}>
              {h5LocalSourceStatus}
            </Typography>
          )}
          <Typography sx={{ ...themedTypography.label, color: themeColors.textMuted, fontSize: 11 }}>
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
        direction="row"
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
            <Typography variant="caption" sx={{ ...themedTypography.label }}>
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
              <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Profile</Typography>
              <Switch checked={profileActive} onChange={(event) => {
                const on = event.target.checked;
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
                const canvas = dpCanvasRef.current;
                if (!canvas) return;
                try {
                  const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/png"));
                  if (!blob) return;
                  try { await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]); }
                  catch { downloadBlob(blob, "show4dstem_dp.png"); }
                } catch (error) {
                  setOfflineBackendError(`Could not copy the DP image: ${String(error)}`);
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
                onClick={(event) => setDpExportAnchor(event.currentTarget)}
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
                    Unhidden · rbin {reportScanBin} · DP kbin {reportDetBin} ({estimateReportHtmlSize(reportScanBin, "unhidden")})
                </MenuItem>
                {currentPageReportCount > 0 && (
                  <MenuItem onClick={() => handleHtmlExportSelect("report", "uint8", reportDetBin, detailedReportScanBin, "current_page")} sx={{ fontSize: 12 }}>
                    Current page · rbin {detailedReportScanBin} · DP kbin {reportDetBin} ({estimateReportHtmlSize(detailedReportScanBin, "current_page")})
                  </MenuItem>
                )}
                {starredReportCount > 0 && (
                  <MenuItem onClick={() => handleHtmlExportSelect("report", "uint8", reportDetBin, detailedReportScanBin, "starred")} sx={{ fontSize: 12 }}>
                    Starred · rbin {detailedReportScanBin} · DP kbin {reportDetBin} ({estimateReportHtmlSize(detailedReportScanBin, "starred")})
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
                onClick={(event) => setDpMoreAnchor(event.currentTarget)}
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
                    <Typography sx={{ ...themedTypography.label, color: themeColors.text }}>
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
                            Calibration sliders appear below the image after the phase is ready.
                          </Typography>
                        </Box>
                      }
                    />
                  </Stack>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.75, gap: 1 }}>
                    <Typography sx={themedTypography.label}>Trials</Typography>
                    <Select
                      value={Math.max(0, Math.round(Number(ssbComputeNTrials ?? 200)))}
                      onChange={(event) => setSsbComputeNTrials(Number(event.target.value))}
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
                    <Typography sx={themedTypography.label}>Refine</Typography>
                    <Switch
                      checked={Boolean(ssbComputeRefine)}
                      onChange={(event) => setSsbComputeRefine(event.target.checked)}
                      disabled={ssbComputeBusy}
                      size="small"
                      sx={switchStyles.small}
                    />
                  </Stack>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5, gap: 1 }}>
                    <Typography sx={themedTypography.label} title="Pin C10 at its slider value during the aberration search">
                      Lock C10 <Box component="span" sx={{ color: themeColors.textMuted }}>{Number(ssbComputeC10Nm ?? 0).toFixed(0)} nm</Box>
                    </Typography>
                    <Switch
                      checked={Boolean(ssbComputeLockC10)}
                      onChange={(event) => setSsbComputeLockC10(event.target.checked)}
                      disabled={ssbComputeBusy}
                      size="small"
                      sx={switchStyles.small}
                    />
                  </Stack>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.5, gap: 1 }}>
                    <Typography sx={themedTypography.label} title="Pin C12 and φ12 at their slider values during the aberration search">
                      Lock C12 <Box component="span" sx={{ color: themeColors.textMuted }}>{Number(ssbComputeC12Nm ?? 0).toFixed(0)} nm</Box>
                    </Typography>
                    <Switch
                      checked={Boolean(ssbComputeLockC12)}
                      onChange={(event) => setSsbComputeLockC12(event.target.checked)}
                      disabled={ssbComputeBusy}
                      size="small"
                      sx={switchStyles.small}
                    />
                  </Stack>
                  <Typography sx={{ ...themedTypography.label, color: themeColors.textMuted, whiteSpace: "normal", lineHeight: 1.35, mb: 0.75 }}>
                    {ssbBfCountText}
                  </Typography>
                  {ssbProgressText && (
                    <Typography
                      role="status"
                      sx={{
                        ...themedTypography.label,
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
                      {ssbProgressText}
                    </Typography>
                  )}
                  <Typography sx={{ ...themedTypography.label, color: themeColors.textMuted, whiteSpace: "normal", lineHeight: 1.35 }}>
                    Default is 200 trials with refinement on every detected BF pixel. Full 512 scans can take seconds to about a minute.
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
                    ...themedTypography.label,
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
                    ...themedTypography.label,
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
                    ...themedTypography.label,
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
                  title={ssbProgressText || "Running SSB..."}
                >
                  {ssbProgressText || "Running SSB..."}
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
              onPointerUp={handleDpMouseUp} onPointerCancel={handleDpMouseUp} onPointerLeave={handleDpMouseLeave}
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
              offlineBackendError === OFFLINE_NEEDS_WEBGPU ? "Needs WebGPU" : offlineBackendError ? "Show4DSTEM load failed" : "Loading DP",
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
                      {viSourceLabel(source)}
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
                onMouseDown={(event) => {
                  setIsResizingProfile(true);
                  profileResizeStart.current = { startY: event.clientY, startHeight: profileHeight };
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
                onClick={() => setMobileDpOptionsOpen(open => !open)}
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
                      <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Detector</Typography>
                      <Select value={roiMode || "point"} onChange={(event) => setRoiMode(event.target.value)} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
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
                            onChange={(_, value) => {
                              beginDpRoiInteraction();
                              if (roiMode === "annular") {
                                const [inner, outer] = value as number[];
                                setRoiRadiusInner(Math.min(inner, outer - 1));
                                setRoiRadius(Math.max(outer, inner + 1));
                              } else {
                                const next = Array.isArray(value) ? value[0] : value;
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
                          <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>
                            {roiMode === "annular" ? `${Math.round(roiRadiusInner)}-${Math.round(roiRadius)}px` : `${Math.round(roiRadius)}px`}
                          </Typography>
                        </>
                      )}
                    </Box>
                    {/* Row 2: Color + Scale + Colorbar */}
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                      <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Color</Typography>
                      <Select value={dpColormap} onChange={(event) => setDpColormap(String(event.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="inferno">Inferno</MenuItem>
                        <MenuItem value="viridis">Viridis</MenuItem>
                        <MenuItem value="plasma">Plasma</MenuItem>
                        <MenuItem value="magma">Magma</MenuItem>
                        <MenuItem value="hot">Hot</MenuItem>
                        <MenuItem value="RdBu_r">RdBu</MenuItem>
                        <MenuItem value="twilight_shifted">Twilight</MenuItem>
                        <MenuItem value="gray">Gray</MenuItem>
                      </Select>
                      <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Scale</Typography>
                      <Select value={dpScaleMode} onChange={(event) => setDpScaleMode(event.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="linear">Lin</MenuItem>
                        <MenuItem value="log">Log</MenuItem>

                      </Select>
                      <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Colorbar</Typography>
                      <Switch checked={showDpColorbar} onChange={(event) => setShowDpColorbar(event.target.checked)} size="small" sx={switchStyles.small} />
                    </Box>
                  </Box>
                  {/* Right: Histogram spanning both rows */}
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", justifyContent: "center", flex: "0 0 auto", maxWidth: "100%" }}>
                    <Histogram data={dpHistogramData} vminPct={dpVminPct} vmaxPct={dpVmaxPct} onRangeChange={(min, max) => { setDpVminPct(min); setDpVmaxPct(max); }} width={110} height={58} sliderInset={6} theme={themeInfo.theme} dataMin={dpGlobalMin} dataMax={dpGlobalMax} />
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
              <Typography sx={{ ...themedTypography.label, color: themeColors.textMuted, flexShrink: 0 }}>
                {compareMode ? "Multiple " : ""}{viSourceLabel(activeViSource)}{compareMode ? ` | ${shapeRows}×${shapeCols}` : ` | ${shapeRows}×${shapeCols} | ${detRows}×${detCols}`}
              </Typography>
              {controlsVisible && compareMode && activeComparePageCount > 1 && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.35, flexShrink: 0 }}>
                  <Typography sx={{ ...themedTypography.label, fontSize: 10, flexShrink: 0 }}>Group</Typography>
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
                      requestComparePage(activeComparePageIdx - 1);
                    }}
                    sx={{ color: activeComparePageIdx <= 0 ? themeColors.textMuted : themeColors.accent, p: 0.2 }}
                  >
                    <FastRewindIcon sx={{ fontSize: 15 }} />
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
                            sx={{ ...themedTypography.value, width: 16, textAlign: "center", color: themeColors.textMuted, lineHeight: "20px" }}
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
                      requestComparePage(activeComparePageIdx + 1);
                    }}
                    sx={{ color: activeComparePageIdx >= activeComparePageCount - 1 ? themeColors.textMuted : themeColors.accent, p: 0.2 }}
                  >
                    <FastForwardIcon sx={{ fontSize: 15 }} />
                  </IconButton>
                  </>}
                  <Typography sx={{ ...themedTypography.value, minWidth: compareAllGroups ? 58 : activeComparePageCount > 99 ? 52 : 34, textAlign: "left", flexShrink: 0 }}>{comparePageStatus}</Typography>
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
                <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Cols</Typography>
                <Select
                  value={compareCols || 0}
                  onChange={(event) => setCompareCols(Number(event.target.value))}
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
              <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>FFT</Typography>
              <Switch checked={showFft} onChange={(event) => setShowFft(event.target.checked)} size="small" sx={switchStyles.small} />
              {!compareMode && <>
                <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Profile</Typography>
                <Switch checked={viProfileActive} onChange={(event) => {
                  const on = event.target.checked;
                  setViProfileActive(on);
                  if (!on) {
                    setViProfilePoints([]);
                    setHoveredViProfileEndpoint(null);
                    setIsHoveringViProfileLine(false);
                  }
                }} size="small" sx={switchStyles.small} />
                <Button size="small" sx={compactButton} disabled={viZoom === 1 && viPanX === 0 && viPanY === 0} onClick={() => { setViZoom(1); setViPanX(0); setViPanY(0); }}>Reset</Button>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={async () => {
                  let canvas: HTMLCanvasElement | null = null;
                  try {
                    canvas = await getActiveViCanvas();
                    if (!canvas) return;
                    const captured = canvas;
                    const blob = await new Promise<Blob | null>(resolve => captured.toBlob(resolve, "image/png"));
                    if (!blob) return;
                    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                  } catch {
                    canvas?.toBlob((blob) => { if (blob) downloadBlob(blob, "show4dstem_vi.png"); }, "image/png");
                  }
                }}>Copy</Button>
              </>}
            </Stack>}
          </Stack>

          {/* VI Canvas */}
          {compareMode && compareDpMode === "all" && <AllDiffractionGrid
            bytes={compareDiffractionBytes} indices={compareDiffractionIndices || []}
            rows={detRows} cols={detCols}
            selectionLabel={viRoiMode === "off" ? "Point" : `${optionLabel(viRoiReduce || "mean")} over shared ${viRoiMode} ROI`}
            renderGrid={({engine, slots, ranges}) => <CompareVirtualGrid
              kind="diffraction" bytes={undefined} count={compareDiffractionIndices.length}
              indices={compareDiffractionIndices} gpuSlots={slots} gpuRanges={ranges}
              gpuEngine={engine} labels={frameLabels || []} activeIdx={frameIdx}
              shapeRows={detRows} shapeCols={detCols} cols={compareCols || 0}
              colormap={dpColormap} scaleMode={dpScaleMode} vminPct={dpVminPct}
              vmaxPct={dpVmaxPct} autoContrast={false} smooth={false}
              cursorRow={localKRow} cursorCol={localKCol} status=""
              themeColors={themeColors} panelChromeVisible={panelChromeVisible}
              showScaleBar={showScaleBar} pixelSize={kCalibrated ? kPixelSize : 1}
              pixelUnit={kCalibrated ? kPixelUnit : "px"}
              panelOrder={comparePanelOrder || []} hidden={compareHiddenPanels || []}
              starred={compareStarredPanels || []} reorderMode={compareReorderMode}
              draggingFrame={compareDraggingFrame} pendingMoveFrame={comparePendingMoveFrame}
              maxWidthPx={compareGridWidth} panelGapPx={0}
              onResizeStart={handleCompareGridResizeStart} onSelect={setFrameIdx}
              onToggleStar={toggleCompareStar} onHide={hideCompareFrame}
              onReorderFrame={moveCompareFrame} onDragFrameChange={setCompareDraggingFrame}
              onPendingMoveFrameChange={setComparePendingMoveFrame}
              onPositionChange={(row, col, commit) => {
                setLocalKRow(row); setLocalKCol(col);
                // All diffraction panels edit the same live virtual detector.
                // CompareVirtualGrid already coalesces pointer moves with rAF;
                // use the primary detector's preview and finalization paths.
                dpRoiInteractiveRef.current = true;
                model.set("roi_active", true);
                queueRoiCenter(row, col);
                if (commit) {
                  finishDpRoiInteraction();
                } else {
                  requestCompareViLive();
                }
              }}
            />}
          />}
          {compareMode ? (
            <CompareVirtualGrid
              scanRegion={{mode:viRoiMode, row:localViRoiCenterRow, col:localViRoiCenterCol,
                radius:viRoiRadius || 5, width:viRoiWidth || 10, height:viRoiHeight || 10}}
              bytes={displayedCompareVirtualImageBytes}
              count={comparePanelCount || 0}
              indices={comparePanelIndices || []}
              gpuSlots={compareGpuSlotsRef.current}
              gpuRanges={compareGpuRangesRef.current}
              gpuVersion={compareGpuVersion}
              gpuEngine={viGpuColormapRef.current}
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
              panelGapPx={0}
              onResizeStart={handleCompareGridResizeStart}
              onSelect={selectCompareFrame}
              onToggleStar={toggleCompareStar}
              onHide={hideCompareFrame}
              onReorderFrame={moveCompareFrame}
              onDragFrameChange={setCompareDraggingFrame}
              onPendingMoveFrameChange={setComparePendingMoveFrame}
              onPositionChange={(row, col, commit) => {
                if (viRoiMode === "off") { updateScanPosition(row, col, commit); return; }
                setLocalViRoiCenterRow(row); setLocalViRoiCenterCol(col);
                if (offline) {model.set("vi_roi_center", [row, col]);model.save_changes();}
                else regionRequests.request(row, col);
              }}
              onGpuRendererReady={setCompareGpuRenderer}
            />
          ) : (
            <Box sx={{ ...container.imageBox, width: "100%", maxWidth: viCanvasWidth, aspectRatio: `${shapeCols} / ${shapeRows}`, height: "auto", touchAction: "none", ...mobileImageBoxSx }}>
              <canvas
                data-quantem-scientific-output={viGpuVisible ? "show4dstem-virtual-image-cpu" : "show4dstem-virtual-image"}
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
                data-quantem-scientific-output={viGpuVisible ? "show4dstem-virtual-image" : "show4dstem-virtual-image-gpu"}
                ref={attachVirtualGpuCanvas}
                style={{ position: "absolute", width: "100%", height: "100%",
                  imageRendering: "pixelated", pointerEvents: "none", opacity: viGpuVisible ? 1 : 0 }}
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
                offlineBackendError === OFFLINE_NEEDS_WEBGPU ? "Needs WebGPU" : offlineBackendError ? "Show4DSTEM load failed" : "Loading virtual image",
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

          {/* VI Stats Bar: stats on left, Auto/Smooth toggles on right edge */}
          {showStats && !viPanelLoading && viStats && viStats.length === 4 && (
            <Box sx={statsBarSx}>
              <Typography sx={statsTextSx}>Mean <Box component="span" sx={statsValueSx}>{formatStat(viStats[0])}</Box></Typography>
              <Typography sx={statsTextSx}>Min <Box component="span" sx={statsValueSx}>{formatStat(viStats[1])}</Box></Typography>
              <Typography sx={statsTextSx}>Max <Box component="span" sx={statsValueSx}>{formatStat(viStats[2])}</Box></Typography>
              <Typography sx={statsTextSx}>Std <Box component="span" sx={statsValueSx}>{formatStat(viStats[3])}</Box></Typography>
              {controlsVisible && <Box sx={{ ml: "auto", display: "flex", alignItems: "center", gap: "2px", flexWrap: "nowrap", whiteSpace: "nowrap", flexShrink: 0 }}>
                <Typography sx={{ ...themedTypography.label, fontSize: 10, lineHeight: "20px" }}>Auto</Typography>
                <Switch checked={viAutoContrast} onChange={(event) => toggleViAutoContrast(event.target.checked)} size="small" sx={switchStyles.small} />
                <Typography sx={{ ...themedTypography.label, fontSize: 10, lineHeight: "20px" }} title="CSS bilinear interpolation. Same data, browser smooths visually.">Smooth</Typography>
                <Switch checked={viSmooth} onChange={(event) => setViSmooth(event.target.checked)} size="small" sx={switchStyles.small} />
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
                onMouseDown={(event) => {
                  setIsResizingViProfile(true);
                  viProfileResizeStart.current = { startY: event.clientY, startHeight: viProfileHeight };
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
                onClick={() => setMobileViOptionsOpen(open => !open)}
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
                      <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>ROI</Typography>
                      <Select inputProps={{"aria-label":"Scan region"}} value={viRoiMode || "off"} onChange={(event) => setViRoiMode(event.target.value)} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps}>
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
                                onChange={(_, value) => setViRoiRadius(value as number)}
                                min={1}
                                max={Math.min(shapeRows, shapeCols) / 2}
                                size="small"
                                sx={{ ...sliderStyles.small, width: 53, mx: 1 }}
                              />
                              <Typography sx={{ ...themedTypography.value, fontSize: 10, minWidth: 30 }}>
                                {Math.round(viRoiRadius || 5)}px
                              </Typography>
                            </>
                          )}
                          <Select inputProps={{"aria-label":"Scan region reduction"}} value={viRoiReduce || "mean"} onChange={(event) => setViRoiReduce(event.target.value)} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps}>
                            <MenuItem value="mean">Mean</MenuItem>
                            <MenuItem value="sum">Sum</MenuItem>
                            <MenuItem value="max">Max</MenuItem>
                          </Select>
                        </>
                      )}
                    </Box>
                    {/* Row 2: Color + Scale */}
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                      <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Color</Typography>
                      <Select value={viColormap} onChange={(event) => setViColormap(String(event.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="inferno">Inferno</MenuItem>
                        <MenuItem value="viridis">Viridis</MenuItem>
                        <MenuItem value="plasma">Plasma</MenuItem>
                        <MenuItem value="magma">Magma</MenuItem>
                        <MenuItem value="hot">Hot</MenuItem>
                        <MenuItem value="RdBu_r">RdBu</MenuItem>
                        <MenuItem value="twilight_shifted">Twilight</MenuItem>
                        <MenuItem value="gray">Gray</MenuItem>
                      </Select>
                      <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Scale</Typography>
                      <Select value={viScaleMode} onChange={(event) => setViScaleMode(event.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="linear">Lin</MenuItem>
                        <MenuItem value="log">Log</MenuItem>
                      </Select>
                    </Box>
                  </Box>
                  {/* Right: Histogram spanning both rows */}
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", justifyContent: "center", flex: "0 0 auto", maxWidth: "100%" }}>
                    <Histogram data={viHistogramData} bins={viHistogramBins} vminPct={viVminPct} vmaxPct={viVmaxPct} onRangeChange={(min, max) => { if (viAutoContrast) { viPreAutoPctRef.current = null; setViAutoContrast(false); } setViVminPct(min); setViVmaxPct(max); }} width={110} height={58} sliderInset={6} theme={themeInfo.theme} dataMin={viDataMin} dataMax={viDataMax} />
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
                    {ssbProgressText && (
                      <Typography
                        role="status"
                        sx={{
                          ...themedTypography.label,
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
                        title={ssbProgressText}
                      >
                        {ssbProgressText}
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
                    ] as const).map((control) => (
                      <Box key={control.key} sx={{ minWidth: 0 }}>
                        <Tooltip title={control.title} placement="top" arrow>
                          <Typography sx={{ ...themedTypography.label, color: themeColors.textMuted, cursor: "help", mb: -0.5 }}>
                            {control.label} <Box component="span" sx={{ color: themeColors.accent }}>{control.value.toFixed(control.precision)}</Box> {control.unit}
                          </Typography>
                        </Tooltip>
                        <Slider
                          value={control.value}
                          min={control.min}
                          max={control.max}
                          step={control.step}
                          disabled={ssbComputeBusy}
                          onChange={(_, value) => {
                            const next = Number(Array.isArray(value) ? value[0] : value);
                            control.setValue(next);
                            control.schedule(next);
                          }}
                          onChangeCommitted={(_, value) => control.commit(Number(Array.isArray(value) ? value[0] : value))}
                          size="small"
                          valueLabelDisplay="auto"
                          valueLabelFormat={(value) => `${Number(value).toFixed(control.precision)} ${control.unit}`}
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
        {showFft && (
          <Box sx={{ width: viPanelWidth, maxWidth: "100%" }}>
            {/* FFT Header */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={panelHeaderSx}>
              <Typography variant="caption" sx={{ ...themedTypography.label, color: roiFftActive && fftCropDims ? accentGreen : themeColors.textMuted }}>{roiFftActive && fftCropDims ? `ROI FFT (${fftCropDims.cropWidth}\u00D7${fftCropDims.cropHeight})` : "FFT"}</Typography>
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
                  onClick={() => setMobileFftOptionsOpen(open => !open)}
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
                        <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Scale</Typography>
                        <Select value={fftScaleMode} onChange={(event) => setFftScaleMode(event.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                          <MenuItem value="linear">Lin</MenuItem>
                          <MenuItem value="log">Log</MenuItem>

                        </Select>
                        <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Auto</Typography>
                        <Switch checked={fftAuto} onChange={(event) => toggleFftAuto(event.target.checked)} size="small" sx={switchStyles.small} />
                        {fftCropDims && (
                          <>
                            <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Win</Typography>
                            <Switch checked={fftWindow} onChange={(event) => setFftWindow(event.target.checked)} size="small" sx={switchStyles.small} />
                          </>
                        )}
                      </Box>
                      {/* Row 2: Color */}
                      <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                        <Typography sx={{ ...themedTypography.label, fontSize: 10 }}>Color</Typography>
                        <Select value={fftColormap} onChange={(event) => setFftColormap(String(event.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
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
                        <Histogram data={fftHistogramData} vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }} width={110} height={58} sliderInset={6} theme={themeInfo.theme} dataMin={fftDataMin} dataMax={fftDataMax} />
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

      {/* Frame controls (5D time/tilt series): matches Show3D playback */}
      {controlsVisible && nFrames > 1 && (<>
        <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
          <Typography sx={{ ...themedTypography.label, fontSize: 10, flexShrink: 0 }}>View</Typography>
          <Select
            value={displayViewMode}
            onChange={(event) => setViewMode(String(event.target.value))}
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
              <Typography sx={{ ...themedTypography.label, fontSize: 10, flexShrink: 0 }}>DP</Typography>
              <Select
                value={compareDpMode || "average"}
                onChange={(event) => setCompareDpMode(String(event.target.value))}
                size="small"
                inputProps={{ "aria-label": "Show4DSTEM multiple DP source" }}
                sx={{ ...themedSelect, minWidth: 82, fontSize: 10 }}
                MenuProps={themedMenuProps}
              >
                <MenuItem value="average">Average</MenuItem>
                <MenuItem value="selected">Selected</MenuItem>
                <MenuItem value="all">All (live)</MenuItem>
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
          {!compareMode && <>
          <Typography sx={{ ...themedTypography.label, fontSize: 10, flexShrink: 0 }}>{frameDimLabel}:</Typography>
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
          <Slider value={frameIdx} onChange={(_, value) => { setFramePlaying(false); setFrameIdx(value as number); }} min={0} max={Math.max(0, nFrames - 1)} size="small" aria-label={frameSliderAriaLabel} sx={{ flex: 1, minWidth: 60, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
          <Typography sx={{ ...themedTypography.value, minWidth: 50, textAlign: "right", flexShrink: 0 }}>{frameLabels && frameLabels.length > frameIdx ? frameLabels[frameIdx] : `${frameIdx + 1}/${nFrames}`}</Typography>
          </>}
        </Box>
        {!compareMode && <Box sx={{ ...controlRow, mt: `${SPACING.XS}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
          <Typography sx={{ ...themedTypography.label, fontSize: 10, color: themeColors.textMuted, flexShrink: 0 }}>fps</Typography>
          <Slider value={frameFps} min={1} max={30} step={1} onChange={(_, value) => setFrameFps(value as number)} size="small" sx={{ ...sliderStyles.small, width: 35, flexShrink: 0 }} />
          <Typography sx={{ ...themedTypography.label, fontSize: 10, color: themeColors.textMuted, minWidth: 14, flexShrink: 0 }}>{Math.round(frameFps)}</Typography>
          <Typography sx={{ ...themedTypography.label, fontSize: 10, color: themeColors.textMuted, flexShrink: 0 }}>Loop</Typography>
          <Switch size="small" checked={frameLoop} onChange={() => setFrameLoop(!frameLoop)} sx={{ ...switchStyles.small, flexShrink: 0 }} />
          <Typography sx={{ ...themedTypography.label, fontSize: 10, color: themeColors.textMuted, flexShrink: 0 }}>Bounce</Typography>
          <Switch size="small" checked={frameBoomerang} onChange={() => setFrameBoomerang(!frameBoomerang)} sx={{ ...switchStyles.small, flexShrink: 0 }} />
        </Box>}
      </>)}
    </Box>
  );
}

export const render = createRender(Show4DSTEM);
