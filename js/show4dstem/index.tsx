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
import { COLORMAPS, applyColormap } from "../colormaps";
import { WebGPUFFT, getWebGPUFFT, fft2d, fftshift, autoEnhanceFFT, nextPow2, applyHannWindow2D } from "../fft";
import { Show4DSTEMCompute, Show4DSTEMCpuCompute } from "../engine/compute";
import { readH5Volume } from "../engine/h5reader";
import { decodeBslz4ToStack, type Bslz4Spec } from "../engine/bslz4";
import { LazyShow4DSTEM } from "../engine/lazy";
import { drawScaleBarHiDPI, drawColorbar, roundToNiceValue } from "../figure";
import { findDataRange, sliderRange, computeStats, computeHistogramFromBytes, percentileClip } from "../stats";
import { downloadBlob, extractBytes, formatNumber, downloadDataView, preserveRestoredWidgetModelsOnSave } from "../format";
import { useHideStaticFallback } from "../staticFallback";
import { MetadataSection } from "../widgetInfo";

// Detector mask for the offline WebGPU virtual-image sum. Mirrors the Python
// mask geometry exactly (show4dstem.py _create_*_mask): cx pairs with column,
// cy with row, so the browser virtual image matches the kernel's pixel-for-pixel.
function buildDetectorMask(model: any, detRows: number, detCols: number): Uint32Array {
  const mask = new Uint32Array(detRows * detCols);
  const cx = model.get("roi_center_col");
  const cy = model.get("roi_center_row");
  const mode = model.get("roi_mode") || "circle";
  const radius = model.get("roi_radius") || 0;
  const inner = model.get("roi_radius_inner") || 0;
  const halfW = (model.get("roi_width") || 0) / 2;
  const halfH = (model.get("roi_height") || 0) / 2;
  for (let row = 0; row < detRows; row++) {
    for (let col = 0; col < detCols; col++) {
      const dx = col - cx, dy = row - cy, d2 = dx * dx + dy * dy;
      let inside = false;
      if (mode === "circle") inside = d2 <= radius * radius;
      else if (mode === "annular") inside = d2 > inner * inner && d2 <= radius * radius;
      else if (mode === "square") inside = Math.abs(dx) <= radius && Math.abs(dy) <= radius;
      else if (mode === "rect") inside = Math.abs(dx) <= halfW && Math.abs(dy) <= halfH;
      else if (mode === "point") inside = Math.round(cx) === col && Math.round(cy) === row;
      mask[row * detCols + col] = inside ? 1 : 0;
    }
  }
  return mask;
}

// Scan-ROI mask for the offline DP-from-region reduce (mirrors the vi_roi_mode
// geometry in show4dstem.py _compute_vi_roi_dp).
function buildScanMask(model: any, scanRows: number, scanCols: number): Uint32Array {
  const mask = new Uint32Array(scanRows * scanCols);
  const cx = model.get("vi_roi_center_col");
  const cy = model.get("vi_roi_center_row");
  const mode = model.get("vi_roi_mode") || "circle";
  const radius = model.get("vi_roi_radius") || 0;
  const halfW = (model.get("vi_roi_width") || 0) / 2;
  const halfH = (model.get("vi_roi_height") || 0) / 2;
  for (let row = 0; row < scanRows; row++) {
    for (let col = 0; col < scanCols; col++) {
      const dx = col - cx, dy = row - cy;
      let inside = false;
      if (mode === "circle") inside = dx * dx + dy * dy <= radius * radius;
      else if (mode === "square") inside = Math.abs(dx) <= radius && Math.abs(dy) <= radius;
      else if (mode === "rect") inside = Math.abs(dx) <= halfW && Math.abs(dy) <= halfH;
      mask[row * scanCols + col] = inside ? 1 : 0;
    }
  }
  return mask;
}

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

// ============================================================================
// UI Styles - component styling helpers
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
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
};

function makeHtmlExportFilename(title: string, nFrames: number, scanRows: number, scanCols: number, detRows: number, detCols: number, dtype: string, detBin: number): string {
  let slug = (title || "show4dstem")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  while (slug.includes("__")) slug = slug.replace(/__/g, "_");
  if (!slug) slug = "show4dstem";
  const binnedRows = Math.max(1, Math.floor(detRows / detBin));
  const binnedCols = Math.max(1, Math.floor(detCols / detBin));
  const shape = nFrames > 1
    ? `${nFrames}x${scanRows}x${scanCols}x${binnedRows}x${binnedCols}`
    : `${scanRows}x${scanCols}x${binnedRows}x${binnedCols}`;
  return `${slug}_${shape}_${dtype}_bin${detBin}.html`;
}

function formatSavedBytes(bytes: number): string {
  const mb = Math.max(0, bytes) / (1024 * 1024);
  if (mb >= 100) return `${Math.round(mb)} MB`;
  if (mb >= 10) return `${mb.toFixed(1)} MB`;
  return `${mb.toFixed(2)} MB`;
}

function formatEstimatedHtmlSize(payloadBytes: number): string {
  const htmlBytes = Math.max(0, payloadBytes) * 4 / 3 + HTML_EXPORT_OVERHEAD_BYTES;
  const mb = htmlBytes / (1024 * 1024);
  if (mb >= 1000) return `~${(mb / 1024).toFixed(1)} GB`;
  if (mb >= 100) return `~${Math.round(mb)} MB`;
  if (mb >= 10) return `~${mb.toFixed(1)} MB`;
  return `~${mb.toFixed(2)} MB`;
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
function findFFTPeak(mag: Float32Array, width: number, height: number, col: number, row: number, radius: number): { row: number; col: number } {
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
  isDragging: boolean
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
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);

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
  onResizeStart?: (event: React.PointerEvent<HTMLElement>) => void;
  onSelect: (idx: number) => void;
  onToggleStar: (idx: number) => void;
  onHide: (idx: number) => void;
  onReorderFrame: (dragFrame: number, targetFrame: number) => void;
  onDragFrameChange: (idx: number | null) => void;
  onPendingMoveFrameChange: (idx: number | null) => void;
}

function CompareVirtualGrid({
  bytes,
  count,
  indices,
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
}: CompareVirtualGridProps) {
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const tileRefs = React.useRef<(HTMLDivElement | null)[]>([]);
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
  const [previewIndices, setPreviewIndices] = React.useState<number[] | null>(null);
  const panelByFrame = React.useMemo(() => {
    const out = new Map<number, Float32Array>();
    (indices || []).forEach((frame, idx) => {
      const panel = panels[idx];
      if (panel) out.set(frame, panel);
    });
    return out;
  }, [indices, panels]);
  const displayIndices = React.useMemo(() => {
    const available = new Set(indices || []);
    const hiddenSet = new Set((hidden || []).filter((idx) => Number.isInteger(idx) && available.has(idx)));
    const ordered: number[] = [];
    const seen = new Set<number>();
    (panelOrder || []).forEach((idx) => {
      if (available.has(idx) && !hiddenSet.has(idx) && !seen.has(idx)) {
        ordered.push(idx);
        seen.add(idx);
      }
    });
    (indices || []).forEach((idx) => {
      if (!hiddenSet.has(idx) && !seen.has(idx)) ordered.push(idx);
    });
    return ordered;
  }, [hidden, indices, panelOrder]);
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
      .map((frame) => ({ frame, panel: panelByFrame.get(frame) }))
      .filter((entry): entry is { frame: number; panel: Float32Array } => entry.panel !== undefined);
  }, [panelByFrame, renderIndices]);

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
    renderEntries.forEach(({ panel }, idx) => {
      const canvas = canvasRefs.current[idx];
      if (!canvas) return;
      if (canvas.width !== shapeCols) canvas.width = shapeCols;
      if (canvas.height !== shapeRows) canvas.height = shapeRows;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
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
    });
  }, [autoContrast, colormap, renderEntries, scaleMode, shapeCols, shapeRows, smooth, vmaxPct, vminPct]);

  const displayCount = Math.max(1, renderEntries.length);
  const autoCols = displayCount >= 8 ? 4 : displayCount >= 5 ? 3 : displayCount >= 2 ? 2 : 1;
  const requestedMaxCols = cols > 0 ? Math.max(1, Math.floor(cols)) : autoCols;
  const gridCols = Math.max(1, Math.min(displayCount, requestedMaxCols));
  const mobileGridCols = cols > 0 ? gridCols : Math.max(1, Math.min(gridCols, 2));
  const gridGapPx = Math.max(0, Math.floor(Number.isFinite(panelGapPx) ? panelGapPx : 0));
  const markerLeft = `${((((Math.max(0, Math.min(shapeCols - 1, cursorCol)) + 0.5) * compareZoom) + comparePanX) / Math.max(1, shapeCols)) * 100}%`;
  const markerTop = `${((((Math.max(0, Math.min(shapeRows - 1, cursorRow)) + 0.5) * compareZoom) + comparePanY) / Math.max(1, shapeRows)) * 100}%`;
  const imageLeft = `${(comparePanX / Math.max(1, shapeCols)) * 100}%`;
  const imageTop = `${(comparePanY / Math.max(1, shapeRows)) * 100}%`;
  const imageWidth = `${compareZoom * 100}%`;
  const imageHeight = `${compareZoom * 100}%`;
  const availablePanelCount = Math.max(0, (indices || []).length);
  const statusText = renderEntries.length < availablePanelCount
    ? `${renderEntries.length}/${availablePanelCount} visible`
    : status;

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
    renderEntries.forEach((_, idx) => {
      const overlay = overlayRefs.current[idx];
      const tile = tileRefs.current[idx];
      if (!overlay || !tile) return;
      const cssWidth = Math.max(1, Math.round(tile.clientWidth));
      const cssHeight = Math.max(1, Math.round(tile.clientHeight));
      const width = Math.max(1, Math.round(cssWidth * dpr));
      const height = Math.max(1, Math.round(cssHeight * dpr));
      if (overlay.width !== width) overlay.width = width;
      if (overlay.height !== height) overlay.height = height;
      if (showScaleBar) {
        const unit = pixelSize > 0 ? pixelUnit || "px" : "px";
        const pxSize = pixelSize > 0 ? pixelSize : 1;
        drawScaleBarHiDPI(overlay, dpr, compareZoom, pxSize, unit, shapeCols);
      } else {
        const ctx = overlay.getContext("2d");
        ctx?.clearRect(0, 0, overlay.width, overlay.height);
      }
    });
  }, [compareZoom, overlayVersion, pixelSize, pixelUnit, renderEntries, shapeCols, showScaleBar]);

  if (renderEntries.length === 0) {
    return (
      <Box sx={{ border: `1px solid ${themeColors.border}`, bgcolor: themeColors.bgAlt, px: 1, py: 2 }}>
        <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
          {status || "Compare grid is waiting for multiple frames or datasets."}
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: "100%", maxWidth: maxWidthPx > 0 ? `${maxWidthPx}px` : "100%", position: "relative", "@media (max-width: 700px)": { maxWidth: "100%" } }}>
      {statusText && (
        <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mb: 0.5, "@media (max-width: 700px)": { display: "none" } }}>
          {statusText}
        </Typography>
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
        {renderEntries.map(({ frame }, localIdx) => {
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
              aria-label={`Show4DSTEM compare panel ${frame + 1}`}
              tabIndex={0}
              draggable={reorderMode}
              onDoubleClick={handleCompareDoubleClick}
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
                border: "none",
                boxSizing: "border-box",
                outline: "none",
                cursor: reorderMode ? "grab" : "pointer",
                overflow: "hidden",
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
                }}
              />
              <Box
                sx={{
                  position: "absolute",
                  left: markerLeft,
                  top: markerTop,
                  width: 11,
                  height: 11,
                  transform: "translate(-50%, -50%)",
                  pointerEvents: "none",
                  border: "1px solid rgba(255,255,255,0.95)",
                  borderRadius: "50%",
                  boxShadow: "0 0 0 1px rgba(0,0,0,0.65)",
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
                    aria-label={`${isStarred ? "Unstar" : "Star"} Show4DSTEM compare panel ${frame + 1}`}
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
                    aria-label={renderEntries.length <= 1 ? "Cannot hide the last visible panel" : `Hide Show4DSTEM compare panel ${frame + 1}`}
                    className="show4dstem-compare-hide-button"
                    data-frame={frame}
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
            </Box>
          );
        })}
      </Box>
      {panelChromeVisible && onResizeStart && (
        <Box
          onPointerDown={onResizeStart}
          aria-label="Resize Show4DSTEM compare grid"
          role="button"
          tabIndex={-1}
          sx={{
            position: "absolute",
            bottom: 0,
            right: 0,
            width: 16,
            height: 16,
            cursor: "nwse-resize",
            opacity: 0.6,
            background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
            zIndex: 5,
            "&:hover": { opacity: 1 },
            "@media (max-width: 700px)": { display: "none" },
          }}
        />
      )}
    </Box>
  );
}

// ============================================================================
// Main Component
// ============================================================================
function Show4DSTEM() {
  // Direct model access for batched updates
  const model = useModel();
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

  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [virtualImageBytes] = useModelState<DataView>("virtual_image_bytes");

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
  const [comparePanelGapPx] = useModelState<number>("compare_panel_gap_px");
  const [comparePanelOrder, setComparePanelOrder] = useModelState<number[]>("compare_panel_order");
  const [compareHiddenPanels, setCompareHiddenPanels] = useModelState<number[]>("compare_hidden_panels");
  const [compareStarredPanels, setCompareStarredPanels] = useModelState<number[]>("compare_starred_panels");

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
  const [isDraggingDP, setIsDraggingDP] = React.useState(false);
  // rAF coalescing for ROI drag: collapse rapid mousemove events into ≤1
  // Python comm message per animation frame. Without this, drag fires 60+
  // events/sec at >100ms Python compute each → queue piles up → laggy UX.
  const roiCenterPendingRef = React.useRef<[number, number] | null>(null);
  const roiCenterRafRef = React.useRef<number | null>(null);
  const flushRoiCenter = React.useCallback(() => {
    if (roiCenterPendingRef.current) {
      const [r, c] = roiCenterPendingRef.current;
      model.set("roi_center", [r, c]);
      model.save_changes();
      roiCenterPendingRef.current = null;
    }
    roiCenterRafRef.current = null;
  }, [model]);
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

  // ── Offline WebGPU compute backend ──────────────────────────────────────
  // Small datasets ship the full uint16 stack (the `_offline_stack` trait); we
  // run the virtual-image and DP-from-ROI reductions in WebGPU right here, with
  // NO Python kernel. We play Python's role: on any detector/ROI trait change we
  // recompute and set `virtual_image_bytes` / `vi_roi_dp_bytes` on the model, so
  // every existing render effect works unchanged. Detector counts are integers,
  // so the browser masked-sum (u32 accumulate) is bit-exact to the kernel.
  const [offline] = useModelState<boolean>("offline");
  React.useEffect(() => {
    if (!offline) return;
    let disposed = false;
    let detach: (() => void) | null = null;
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
      let compute: Show4DSTEMCompute | Show4DSTEMCpuCompute | null = null;
      let cpuStack: Uint8Array | null = null;  // full decompressed stack for the per-frame probe (single-chunk only)
      // Multi-VOLUME (5D): several datasets, decoded LAZILY (decode-on-scrub) with a
      // small LRU of resident volumes. Only the viewed dataset (plus a few recent)
      // lives in VRAM, so it runs on a laptop regardless of how many h5 files - and
      // first paint is one decode, not N. The frame slider picks the active dataset.
      let computes: (Show4DSTEMCompute | Show4DSTEMCpuCompute)[] = [];   // resident set for single / non-lazy paths
      let volMetas: any[] = [];                  // multi-volume descriptors (lazy)
      const volCache = new Map<number, Show4DSTEMCompute>();   // LRU: idx -> decoded volume
      const inlineVolCache = new Map<number, Show4DSTEMCompute | Show4DSTEMCpuCompute>(); // LRU for inline gzip 5D exports
      const compareResidentTarget = Math.max(3, Math.min(12, Math.max(1, Number(model.get("compare_max_panels") || 3))));
      const MAX_RESIDENT = compareResidentTarget; // recent / compare volumes kept hot for instant scrub and detector drags
      let volumeCount = 0;
      let getVol: ((idx: number) => Promise<Show4DSTEMCompute | Show4DSTEMCpuCompute | null>) | null = null;
      // H5 source: read the merged float32 .h5 file straight off disk via WebGPU
      // (jsfive parse + GPU bitshuffle+LZ4 decode). Nothing embedded - the data stays a
      // file; the HTML just points at it. This is the "click HTML, GPU decompresses the
      // merged H5, real Show4DSTEM renders it" path.
      // Lazy mode: a sidecar bundle (radial profile + CoM + frame index) at _lazy_url lets the
      // virtual image derive from a ~100 MB profile in VRAM and the CBED lazy-fetch one frame from
      // disk - NOTHING bulk-loads. LazyShow4DSTEM has the same interface as Show4DSTEMCompute, so
      // the render below (recomputeVI / recomputeFrame / recomputeCoM) works unchanged.
      const lazyUrl = model.get("_lazy_url") as string | undefined;
      const h5Url = model.get("_h5_url") as string | undefined;
      if (lazyUrl) {
        const lz = await LazyShow4DSTEM.create(lazyUrl);
        compute = lz as unknown as Show4DSTEMCompute;
        if (compute) computes.push(compute);
      } else if (h5Url) {
        if (/_master\.h5$/.test(h5Url)) {
          // Full no-bin merged = N external data files (each <2 GB, the browser ArrayBuffer cap).
          // The browser's fetch->arrayBuffer runs ~0.7 GB/s on ONE connection, so a sequential
          // read is fetch-bound (~47 s for 33 GB). Instead keep a WINDOW of W parallel fetches in
          // flight (8 connections saturate the disk) feeding an IN-ORDER GPU decode. Wall-clock
          // becomes max(parallel fetch, total decode) not their sum -> decode-bound ~15 s. Each
          // file's ArrayBuffer is freed right after its decode; CPU heap peak ~W files.
          const base = h5Url.replace(/_master\.h5$/, "");
          const W = 8;
          const fetchOne = async (n: number): Promise<ArrayBuffer | null> => {
            const resp = await fetch(`${base}_data_${String(n).padStart(6, "0")}.h5`);
            return resp.ok ? resp.arrayBuffer() : null;
          };
          const inflight = new Map<number, Promise<ArrayBuffer | null>>();
          let next = 1;
          for (; next <= W; next++) inflight.set(next, fetchOne(next));
          const gpuChunks: { buffer: GPUBuffer; startScan: number; nScan: number }[] = [];
          let startScan = 0, ds = 0;
          let dev: GPUDevice | null = null;
          const __t0 = performance.now(); let __decMs = 0, __parseMs = 0, __fetchBytes = 0, __waitMs = 0;
          try {
            for (let n = 1; !disposed; n++) {
              const p = inflight.get(n);
              if (!p) break;
              inflight.delete(n);
              const __wt = performance.now();
              const buf = await p;
              __waitMs += performance.now() - __wt;
              if (!buf) break;
              __fetchBytes += buf.byteLength;
              const __pt = performance.now();
              const vol = readH5Volume(buf, "merged");
              __parseMs += performance.now() - __pt;
              ds = vol.detSize;
              const __dt = performance.now();
              const dec = await decodeBslz4ToStack({ ...vol.chunks[0], startScan, nScan: vol.nFrames } as never, "float32", "float32");
              __decMs += performance.now() - __dt;
              if (!dec) break;
              dev = dec.device;
              gpuChunks.push({ buffer: dec.buffer, startScan, nScan: vol.nFrames });
              startScan += vol.nFrames;
              inflight.set(next, fetchOne(next));
              next++;
            }
          } catch (e) {
            gpuChunks.forEach((c) => c.buffer.destroy());   // no mid-stream VRAM leak on fetch/parse error
            throw e;
          }
          const __decGB = startScan * ds * 4 / 1e9;
          (window as unknown as { __loadprof: unknown }).__loadprof = { totalMs: Math.round(performance.now() - __t0),
            fetchedCompressedGB: +(__fetchBytes / 1e9).toFixed(1), decodedFloat32GB: +__decGB.toFixed(1),
            fetchWaitMs: Math.round(__waitMs), decompressMs: Math.round(__decMs), parseMs: Math.round(__parseMs),
            decompressGBps: +(__decGB / (__decMs / 1000)).toFixed(2) };
          if (dev) compute = Show4DSTEMCompute.fromGpuChunks(dev, gpuChunks, scanRows * scanCols, ds, 2);
        } else {
          const vol = readH5Volume(await (await fetch(h5Url)).arrayBuffer(), "merged");
          compute = await Show4DSTEMCompute.createFromBslz4Chunked([{ ...vol.chunks[0], startScan: 0, nScan: vol.nFrames }], scanRows * scanCols, vol.detSize, "float32", "float32");
        }
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
          const cc = await Show4DSTEMCompute.createFromBslz4Chunked(specs, scanRows * scanCols, detR * detC, "uint8", srcDtype);
          if (cc && v.badPx) cc.badPx = new Uint32Array(v.badPx);
          return cc;
        };
        if (Array.isArray(m.volumes)) {
          volMetas = m.volumes;
          volumeCount = volMetas.length;
          getVol = async (idx: number) => {
            if (volCache.has(idx)) return volCache.get(idx)!;
            const cc = await decodeVol(volMetas[idx]);
            if (cc) {
              volCache.set(idx, cc);
              while (volCache.size > MAX_RESIDENT) {           // evict the oldest non-active volume
                const old = [...volCache.keys()].find((k) => k !== idx);
                if (old === undefined) break;
                volCache.get(old)!.dispose(); volCache.delete(old);
              }
            }
            return cc;
          };
          compute = await getVol(Math.max(0, Math.min(volMetas.length - 1, model.get("frame_idx") | 0)));
        } else if (Array.isArray(m.chunks)) {
          const c = await decodeVol(m); if (c) computes.push(c); compute = c;
        } else {
          const raw = await fetchU8(offlineUrl!);
          compute = await Show4DSTEMCompute.createFromBslz4({ compressed: raw, blockMeta: new Uint32Array(m.blockMeta),
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
        compute = await Show4DSTEMCompute.createChunked(specs, scanRows * scanCols, detR * detC);
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
            const cc = await Show4DSTEMCompute.create(bytes, scanCount, detSize) ?? Show4DSTEMCpuCompute.create(bytes, scanCount, detSize);
            if (cc) {
              inlineVolCache.set(idx, cc);
              while (inlineVolCache.size > MAX_INLINE_RESIDENT) {
                const old = [...inlineVolCache.keys()].find((k) => k !== idx);
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
          compute = await Show4DSTEMCompute.create(stack, scanRows * scanCols, detR * detC) ?? Show4DSTEMCpuCompute.create(stack, scanRows * scanCols, detR * detC);
        }
      }
      if (!compute || disposed) { compute?.dispose(); return; }
      // Auto-filter hot/dead detector pixels (from the HDF5 pixel_mask) so the
      // offline result matches CUDA's apply_mask path - no manual masking needed.
      const badPxJson = model.get("_offline_bad_px") as string | undefined;
      if (badPxJson) compute.badPx = new Uint32Array(JSON.parse(badPxJson) as number[]);
      const recomputeVI = async () => {
        const vi = await compute!.maskedSum(buildDetectorMask(model, detR, detC));
        model.set("virtual_image_bytes", new DataView(vi.buffer));
      };
      let compareViGen = 0;
      const compareVisibleIndices = () => {
        const total = Math.max(0, Number(model.get("n_frames") || 0));
        if (total <= 1 || model.get("view_mode") !== "compare") return [] as number[];
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
        return ordered.filter((idx) => !hidden.has(idx)).slice(0, maxPanels);
      };
      const recomputeCompareVI = async () => {
        const indices = compareVisibleIndices();
        if (!indices.length) return;
        const gen = ++compareViGen;
        const mask = buildDetectorMask(model, detR, detC);
        let maskArea = 0;
        for (let i = 0; i < mask.length; i++) maskArea += mask[i] ? 1 : 0;
        maskArea = Math.max(1, maskArea);
        const panelPixels = scanRows * scanCols;
        const stack = new Float32Array(indices.length * panelPixels);
        for (let slot = 0; slot < indices.length; slot++) {
          const idx = indices[slot];
          const panelCompute = getVol ? await getVol(idx) : compute;
          if (gen !== compareViGen || !panelCompute) return;
          const vi = await panelCompute.maskedSum(mask);
          if (gen !== compareViGen) return;
          for (let p = 0; p < panelPixels; p++) {
            stack[slot * panelPixels + p] = vi[p] / maskArea;
          }
        }
        model.set("compare_virtual_image_bytes", new DataView(stack.buffer));
        model.set("compare_panel_count", indices.length);
        model.set("compare_panel_indices", indices);
      };
      (window as unknown as { __sh4d: unknown }).__sh4d = { model, recomputeVI, recomputeCompareVI,
        detMask: () => buildDetectorMask(model, detR, detC),
        deriveOnly: async () => { const vi = await compute!.maskedSum(buildDetectorMask(model, detR, detC)); return vi.length; },
        comLen: () => { const c = compute as unknown as { com?: Float32Array | null }; return c && c.com ? c.com.length : -1; },
        rd: () => ({ mode: model.get("roi_mode"), r: model.get("roi_radius"), ri: model.get("roi_radius_inner"),
          cr: model.get("roi_center_row"), cc: model.get("roi_center_col"), active: model.get("roi_active") }) };
      const recomputeDP = async () => {
        const mode = model.get("vi_roi_mode");
        if (!mode || mode === "off") { model.set("vi_roi_dp_bytes", new DataView(new ArrayBuffer(0))); return; }
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
        // bslz4 / chunked stacks have no CPU copy -> extract the frame on the GPU.
        const frame = cpuStack
          ? (() => { const f = new Float32Array(detSize); const base = scanIdx * detSize; for (let k = 0; k < detSize; k++) f[k] = sample(base + k); return f; })()
          : await compute!.frameAt(scanIdx);
        model.set("frame_bytes", new DataView(frame.buffer)); model.save_changes();
      };
      const onVI = () => { void recomputeVI(); void recomputeCompareVI(); };
      const onDP = () => { void recomputeDP(); };
      const onPos = () => { void recomputeFrame(); };
      // 5D multi-volume: the slider picks the active dataset; decode-on-scrub (LRU).
      let frameGen = 0;
      const onFrame = async () => {
        if (!getVol) return;
        const nVolumes = volumeCount || volMetas.length || 1;
        const v = Math.max(0, Math.min(nVolumes - 1, model.get("frame_idx") | 0));
        const gen = ++frameGen;                  // ignore a stale decode if the user keeps scrubbing
        const cc = await getVol(v);
        if (gen !== frameGen || !cc) return;      // a newer scroll superseded this one
        compute = cc;
        void recomputeVI(); void recomputeCompareVI(); void recomputeDP(); void recomputeFrame();
      };
      if (getVol) model.on("change:frame_idx", onFrame);
      void recomputeFrame();  // initial DP at mount (so the panel isn't blank)
      // BF/ABF/ADF/HAADF presets normally route through the Python kernel
      // (_preset_request -> apply_preset). With no kernel we translate them into
      // the same detector-ROI geometry here so the buttons work offline too.
      const onPreset = () => {
        const name = String(model.get("_preset_request") || "").toLowerCase();
        if (!name) return;
        const bf = model.get("bf_radius") || 1;
        model.set("roi_active", true);
        model.set("roi_center_row", model.get("center_row"));
        model.set("roi_center_col", model.get("center_col"));
        if (name === "bf") { model.set("roi_mode", "circle"); model.set("roi_radius", Math.max(1, bf)); }
        else if (name === "abf") { model.set("roi_mode", "annular"); model.set("roi_radius_inner", Math.max(0.5, bf * 0.5)); model.set("roi_radius", Math.max(1, bf)); }
        else if (name === "adf") { model.set("roi_mode", "annular"); model.set("roi_radius_inner", bf); model.set("roi_radius", bf * 2); }
        else if (name === "haadf") { model.set("roi_mode", "annular"); model.set("roi_radius_inner", bf * 2); model.set("roi_radius", bf * 4); }
        model.set("_preset_request", "");  // consume so the same preset can fire again
        void recomputeVI(); void recomputeCompareVI();
      };
      // Dragging the aperture sets the COMPOUND roi_center [row, col]; the kernel
      // normally splits it into roi_center_row/col. With no kernel we split it
      // ourselves so the mask sees the dragged center (else only presets/sliders,
      // which write the scalars directly, would move the detector). Same for the
      // real-space vi_roi_center drag.
      const onRoiCenter = () => {
        const rc = model.get("roi_center");
        if (Array.isArray(rc) && rc.length === 2) { model.set("roi_center_row", rc[0]); model.set("roi_center_col", rc[1]); }
        void recomputeVI(); void recomputeCompareVI();
      };
      const onViCenter = () => {
        const rc = model.get("vi_roi_center");
        if (Array.isArray(rc) && rc.length === 2) { model.set("vi_roi_center_row", rc[0]); model.set("vi_roi_center_col", rc[1]); }
        void recomputeDP();
      };
      const viTraits = ["roi_center_row", "roi_center_col", "roi_radius", "roi_radius_inner", "roi_mode", "roi_width", "roi_height"];
      const dpTraits = ["vi_roi_center_row", "vi_roi_center_col", "vi_roi_radius", "vi_roi_mode", "vi_roi_width", "vi_roi_height", "vi_roi_reduce"];
      viTraits.forEach((t) => model.on("change:" + t, onVI));
      dpTraits.forEach((t) => model.on("change:" + t, onDP));
      model.on("change:roi_center", onRoiCenter);
      model.on("change:vi_roi_center", onViCenter);
      model.on("change:_preset_request", onPreset);
      model.on("change:pos_row", onPos);
      model.on("change:pos_col", onPos);
      detach = () => {
        viTraits.forEach((t) => model.off("change:" + t, onVI));
        dpTraits.forEach((t) => model.off("change:" + t, onDP));
        model.off("change:roi_center", onRoiCenter);
        model.off("change:vi_roi_center", onViCenter);
        model.off("change:_preset_request", onPreset);
        model.off("change:pos_row", onPos);
        model.off("change:pos_col", onPos);
        model.off("change:frame_idx", onFrame);
        computes.forEach((c) => c.dispose());          // single / non-lazy resident set
        volCache.forEach((c) => c.dispose()); volCache.clear();  // every cached lazy volume
        inlineVolCache.forEach((c) => c.dispose()); inlineVolCache.clear();
      };
      await recomputeVI();  // initial virtual image, no interaction needed
      await recomputeCompareVI();
      // Safety re-run: at first mount the offline stack / roi-detector traits can
      // still be settling, so the very first maskedSum can return an empty (zero)
      // virtual image - leaving the panel blank until the user nudges the detector.
      // A deferred recompute guarantees the BF image appears with no interaction.
      requestAnimationFrame(() => { if (!disposed) { void recomputeVI(); void recomputeCompareVI(); } });
      setTimeout(() => { if (!disposed) { void recomputeVI(); void recomputeCompareVI(); } }, 200);
    })();
    return () => { disposed = true; detach?.(); };
  }, [offline]);
  // dp_stats are computed in JS from frameBytes (Python side no longer
  // syncs a dp_stats trait — saves 4 trait sync round-trips per click).
  const [viStats, setViStats] = React.useState<number[]>([0, 0, 0, 0]);
  const [viDataMin, setViDataMin] = React.useState<number>(0);
  const [viDataMax, setViDataMax] = React.useState<number>(1);
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const [showControls] = useModelState<boolean>("show_controls");
  const [controlsCollapsed, setControlsCollapsed] = useModelState<boolean>("controls_collapsed");
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

  const effectiveShowFft = showFft;
  const compareMode = viewMode === "compare" && nFrames > 1;
  const compareGridWidth = compareGridPreviewWidth ?? (compareGridWidthPx > 0 ? compareGridWidthPx : COMPARE_GRID_DEFAULT_WIDTH);
  React.useEffect(() => {
    if (!compareMode) {
      setCompareReorderMode(false);
      setCompareDraggingFrame(null);
      setComparePendingMoveFrame(null);
      setCompareGridPreviewWidth(null);
      compareGridResizeCleanupRef.current?.();
    }
  }, [compareMode]);
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
    setComparePendingMoveFrame(null);
    setCompareDraggingFrame(null);
  }, [setCompareHiddenPanels, setComparePanelOrder, setCompareStarredPanels]);

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
  const [, setGifExportRequested] = useModelState<boolean>("_gif_export_requested");
  const [gifData] = useModelState<DataView>("_gif_data");
  const [gifMetadataJson] = useModelState<string>("_gif_metadata_json");
  const [exporting, setExporting] = React.useState(false);
  const [dpExportAnchor, setDpExportAnchor] = React.useState<HTMLElement | null>(null);
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
  const estimateHtmlExportSize = React.useCallback((dtype: string, detBin: number) => {
    const binnedRows = Math.max(1, Math.floor(detRows / detBin));
    const binnedCols = Math.max(1, Math.floor(detCols / detBin));
    const bytesPerPixel = dtype === "uint16" ? 2 : 1;
    const payloadBytes = Math.max(0, nFrames) * Math.max(0, shapeRows) * Math.max(0, shapeCols) * binnedRows * binnedCols * bytesPerPixel;
    return formatEstimatedHtmlSize(payloadBytes);
  }, [detCols, detRows, nFrames, shapeCols, shapeRows]);

  const handleHtmlExportSelect = async (dtype: string, detBin: number) => {
    setDpExportAnchor(null);
    if (!["uint8", "uint16"].includes(dtype) || ![1, 2, 4, 8].includes(detBin)) return;
    const mode = `${dtype}-bin${detBin}`;
    const filename = makeHtmlExportFilename(title, nFrames, shapeRows, shapeCols, detRows, detCols, dtype, detBin);
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
    setExportRequest(JSON.stringify({ mode, id, filename, download: true }));
  };

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
    if (!isDraggingVI) { setLocalPosRow(posRow); setLocalPosCol(posCol); }
  }, [posRow, posCol, isDraggingVI]);

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
  const fftWorkRealRef = React.useRef<Float32Array | null>(null);
  const fftWorkImagRef = React.useRef<Float32Array | null>(null);
  const fftMagnitudeRef = React.useRef<Float32Array | null>(null);
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);

  // Parse virtual image bytes into Float32Array and apply scale for histogram
  React.useEffect(() => {
    if (!virtualImageBytes) return;
    // Parse as Float32Array
    const numFloats = virtualImageBytes.byteLength / 4;
    const rawData = new Float32Array(virtualImageBytes.buffer, virtualImageBytes.byteOffset, numFloats);

    // Store a copy for filtering/FFT (rawData is a view, we need a copy)
    let storedData = rawVirtualImageRef.current;
    if (!storedData || storedData.length !== numFloats) {
      storedData = new Float32Array(numFloats);
      rawVirtualImageRef.current = storedData;
    }
    storedData.set(rawData);

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
      setViDataMin(s.min);
      setViDataMax(s.max);
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
      setViHistogramData(scaledData);
    }
  }, [compareMode, virtualImageBytes, viScaleMode]);

  React.useEffect(() => {
    if (!compareMode) return;
    const expectedFloats = Math.max(0, (comparePanelCount || 0) * shapeRows * shapeCols);
    if (!compareVirtualImageBytes || expectedFloats === 0 || compareVirtualImageBytes.byteLength < expectedFloats * 4) {
      return;
    }
    const rawData = new Float32Array(
      compareVirtualImageBytes.buffer,
      compareVirtualImageBytes.byteOffset,
      expectedFloats,
    );
    const s = computeStats(rawData);
    setViStats([s.mean, s.min, s.max, s.std]);
    setViDataMin(s.min);
    setViDataMax(s.max);

    const scaledData = new Float32Array(expectedFloats);
    if (viScaleMode === "log") {
      for (let i = 0; i < expectedFloats; i++) {
        scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
      }
    } else {
      scaledData.set(rawData);
    }
    setViHistogramData(scaledData);
  }, [compareMode, comparePanelCount, compareVirtualImageBytes, shapeCols, shapeRows, viScaleMode]);

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
    const dataMin = r.min;
    const dataMax = r.max;

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
  }, [virtualImageBytes, shapeRows, shapeCols, viColormap, viVminPct, viVmaxPct, viScaleMode, traitViVmin, traitViVmax, viAutoContrast]);

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
  }, [viOffscreenVersion, viZoom, viPanX, viPanY, viSmooth]);

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
  const [fftVersion, setFftVersion] = React.useState(0);

  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !effectiveShowFft) { setFftCropDims(null); return; }
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
    if (gpuFFTRef.current && gpuReady) {
      const runGpuFFT = async () => {
        const real = sourceData.slice();
        const imag = new Float32Array(real.length);
        const { real: fReal, imag: fImag } = await gpuFFTRef.current!.fft2D(real, imag, fftW, fftH, false);
        if (cancelled) return;
        fftshift(fReal, fftW, fftH);
        fftshift(fImag, fftW, fftH);
        fftRealRef.current = fReal;
        fftImagRef.current = fImag;
        if (origCropW > 0) {
          setFftCropDims({ cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH });
        } else if (fftW !== shapeCols || fftH !== shapeRows) {
          setFftCropDims({ cropWidth: shapeCols, cropHeight: shapeRows, fftWidth: fftW, fftHeight: fftH });
        } else {
          setFftCropDims(null);
        }
        setFftVersion(v => v + 1);
      };
      runGpuFFT();
      return () => { cancelled = true; };
    } else {
      const len = sourceData.length;
      let real = fftWorkRealRef.current;
      if (!real || real.length !== len) { real = new Float32Array(len); fftWorkRealRef.current = real; }
      real.set(sourceData);
      let imag = fftWorkImagRef.current;
      if (!imag || imag.length !== len) { imag = new Float32Array(len); fftWorkImagRef.current = imag; } else { imag.fill(0); }
      fft2d(real, imag, fftW, fftH, false);
      fftshift(real, fftW, fftH);
      fftshift(imag, fftW, fftH);
      fftRealRef.current = real;
      fftImagRef.current = imag;
      if (origCropW > 0) {
        setFftCropDims({ cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH });
      } else if (fftW !== shapeCols || fftH !== shapeRows) {
        setFftCropDims({ cropWidth: shapeCols, cropHeight: shapeRows, fftWidth: fftW, fftHeight: fftH });
      } else {
        setFftCropDims(null);
      }
      setFftVersion(v => v + 1);
    }
  }, [virtualImageBytes, shapeRows, shapeCols, gpuReady, effectiveShowFft, roiFftActive, viRoiMode, viRoiCenterRow, viRoiCenterCol, localViRoiCenterRow, localViRoiCenterCol, viRoiRadius, viRoiWidth, viRoiHeight, fftWindow]);

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
    for (let i = 0; i < real.length; i++) {
      const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
      rawMag[i] = mag;
      if (fftScaleMode === "log") { magnitude[i] = Math.log1p(mag); }
      else { magnitude[i] = mag; }
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
  }, [virtualImageBytes]);

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
    // Draw ROI overlay (circle, square, rect, annular) or point crosshair
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
  }, [dpZoom, dpPanX, dpPanY, kPixelSize, kPixelUnit, kCalibrated, detRows, detCols, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, localKCol, localKRow, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner,
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
  }, [viZoom, viPanX, viPanY, pixelSize, pixelUnit, showScaleBar, shapeRows, shapeCols, localPosRow, localPosCol, isDraggingVI,
      viRoiMode, localViRoiCenterRow, localViRoiCenterCol, viRoiRadius, viRoiWidth, viRoiHeight,
      isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize, canvasSize, viProfileActive, viProfilePoints]);

  // ── DP Profile computation ──
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawDpDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      setProfileData(sampleLineProfile(rawDpDataRef.current, detCols, detRows, p0.row, p0.col, p1.row, p1.col, profileWidth));
      if (!profileActive) setProfileActive(true);
    } else {
      setProfileData(null);
    }
  }, [profilePoints, profileWidth, frameBytes]);

  // ── VI Profile computation ──
  React.useEffect(() => {
    if (viProfilePoints.length === 2 && rawViDataRef.current && shapeCols > 0 && shapeRows > 0) {
      const p0 = viProfilePoints[0], p1 = viProfilePoints[1];
      setViProfileData(sampleLineProfile(rawViDataRef.current, shapeCols, shapeRows, p0.row, p0.col, p1.row, p1.col, 1));
    } else {
      setViProfileData(null);
    }
  }, [viProfilePoints, virtualImageBytes, shapeCols, shapeRows]);

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
      return true;
    }

    return false;
  }, [
    activeRoiCenterCol, activeRoiCenterRow, isDraggingResize, isDraggingResizeInner,
    roiMode, roiRadius, roiRadiusInner, sendRoiRadius, setRoiHeight, setRoiRadiusInner, setRoiWidth
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
  }, [getDpImageCoordsFromClient, isDraggingResize, isDraggingResizeInner, resizeDpRoiFromImagePoint]);

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
    model.set("roi_center", [newRow, newCol]);
    model.save_changes();
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
        setProfileData(sampleLineProfile(rawDpDataRef.current, detCols, detRows, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth));
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
        setProfileData(sampleLineProfile(rawDpDataRef.current, detCols, detRows, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth));
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
    if (resizeDpRoiFromImagePoint(imgX, imgY, e.shiftKey)) {
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
  };

  const handleDpMouseUp = (e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
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
              setProfileData(sampleLineProfile(rawDpDataRef.current, detCols, detRows, p0.row, p0.col, pt.row, pt.col, profileWidth));
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
    setLocalPosRow(newX); setLocalPosCol(newY);
    model.set("pos_row", newX);
    model.set("pos_col", newY);
    model.save_changes();
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
    setLocalPosRow(newX); setLocalPosCol(newY);
    model.set("pos_row", newX);
    model.set("pos_col", newY);
    model.save_changes();
  };

  const handleViMouseUp = (e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
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

  const handleFftMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
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
              const snapped = findFFTPeak(fftMagCacheRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
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
                const binC = ((Math.round(imgCol) - halfW) % fftW + fftW) % fftW;
                const binR = ((Math.round(imgRow) - halfH) % fftH + fftH) % fftH;
                const freqC = binC <= paddedW / 2 ? binC / (paddedW * pixelSize) : (binC - paddedW) / (paddedW * pixelSize);
                const freqR = binR <= paddedH / 2 ? binR / (paddedH * pixelSize) : (binR - paddedH) / (paddedH * pixelSize);
                spatialFreq = Math.sqrt(freqC * freqC + freqR * freqR);
                dSpacing = spatialFreq > 0 ? 1 / spatialFreq : null;
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

  const handleCompareGridResizeStart = (e: React.PointerEvent<HTMLElement>) => {
    e.stopPropagation();
    e.preventDefault();
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch {}
    compareGridResizeCleanupRef.current?.();
    const startX = e.clientX;
    const startY = e.clientY;
    const startWidth = compareGridWidth;
    let rafId = 0;
    let latestWidth = startWidth;
    const handlePointerMove = (e: PointerEvent) => {
      const delta = Math.max(e.clientX - startX, e.clientY - startY);
      latestWidth = Math.max(MIN_COMPARE_GRID_WIDTH, startWidth + delta);
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

  const handleDpExportGif = () => {
    setDpExportAnchor(null);
    setExporting(true);
    setGifExportRequested(true);
  };

  // Download GIF when data arrives from Python
  React.useEffect(() => {
    if (!gifData || gifData.byteLength === 0) return;
    downloadDataView(gifData, "show4dstem_dp_animation.gif", "image/gif");
    const metaText = (gifMetadataJson || "").trim();
    if (metaText) {
      downloadBlob(new Blob([metaText], { type: "application/json" }), "show4dstem_dp_animation.json");
    }
    setExporting(false);
  }, [gifData, gifMetadataJson]);


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
  const temporalMode = viewMode === "temporal" && nFrames > 1;
  const squarePanelWidth = `min(${canvasSize}px, 100%)`;
  const viPanelWidth = compareMode ? `min(${compareGridWidth}px, 100%)` : `min(${viCanvasWidth}px, 100%)`;
  const mobileTightLayout = temporalMode || compareMode;
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
  const viOptionSummary = `${viRoiMode === "off" ? "ROI off" : `${optionLabel(viRoiMode)} ${Math.round(viRoiRadius || 5)}px`} | ${optionLabel(viColormap)} | ${viScaleMode === "log" ? "Log" : "Lin"}`;
  const fftOptionSummary = `${fftScaleMode === "log" ? "Log" : "Lin"} | ${optionLabel(fftColormap)}${fftAuto ? " | Auto" : ""}`;

  return (
    <Box
      ref={rootRef}
      className="show4dstem-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
      sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, outline: "none", borderRadius: "2px", width: "100%", maxWidth: "100%", boxSizing: "border-box", "@media (max-width: 700px)": { p: 0, overflowX: "hidden", ".jp-OutputArea-output &, .jp-OutputArea-child &": { width: "calc(100vw - 96px)", maxWidth: "calc(100vw - 96px)" } } }}
    >
      {/* HEADER */}
      {showTitle && <Typography variant="h6" sx={{ ...typo.title, mb: `${SPACING.SM}px` }}>
        {title || "4D-STEM Explorer"}
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
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Detector: ROI mask shape — defines which DP pixels are integrated for the virtual image.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>BF/ABF/ADF: Preset detector configurations (bright-field, annular bright-field, annular dark-field).</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Image: Virtual image — integrated intensity within detector ROI at each scan position.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Spatial frequency content of the virtual image. Auto masks DC + clips to 99.9th percentile.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Smooth: CSS bilinear blit on the VI canvas. No data change — browser smooths the upscale visually. Off = nearest-neighbor (sharp pixel boundaries).</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile contrast (1st–99th). Clips outliers automatically.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on DP to draw a line intensity profile.</Typography>
          {nFrames > 1 && <>
            <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Frame Playback ({frameDimLabel})</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop: Loop playback. Bounce: Ping-pong — alternates forward and reverse.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FPS: Adjust playback speed (1–30 frames per second).</Typography>
          </>}
          <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
          <KeyboardShortcuts items={keyboardShortcutItems} />
        </Box>} theme={themeInfo.theme} />}
      </Typography>}
      {showControls && (
        <Box sx={{ display: "flex", justifyContent: "flex-end", mb: `${SPACING.XS}px`, minHeight: 24 }}>
          <Button
            size="small"
            sx={compactButton}
            onClick={() => setControlsCollapsed(!controlsCollapsed)}
            aria-label={controlsCollapsed ? "Show controls" : "Hide controls"}
          >
            {controlsCollapsed ? "Controls" : "Hide"}
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
            {controlsVisible && <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
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
              {exportEnabled && <Button
                size="small"
                sx={{ ...compactButton, color: themeColors.accent }}
                onClick={(e) => setDpExportAnchor(e.currentTarget)}
                disabled={exporting || htmlExportBusy}
                title={localHtmlExportStatus || exportStatus || "Export images or standalone HTML"}
              >
                {exporting || htmlExportBusy ? "..." : "Export"}
              </Button>}
              {exportEnabled && <Menu anchorEl={dpExportAnchor} open={Boolean(dpExportAnchor)} onClose={() => setDpExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem onClick={() => handleHtmlExportSelect("uint8", 1)} sx={{ fontSize: 12 }}>Quantized uint8 ({estimateHtmlExportSize("uint8", 1)})</MenuItem>
                {detRows % 2 === 0 && detCols % 2 === 0 && <MenuItem onClick={() => handleHtmlExportSelect("uint8", 2)} sx={{ fontSize: 12 }}>Binned 2x uint8 ({estimateHtmlExportSize("uint8", 2)})</MenuItem>}
                {detRows % 4 === 0 && detCols % 4 === 0 && <MenuItem onClick={() => handleHtmlExportSelect("uint8", 4)} sx={{ fontSize: 12 }}>Binned 4x uint8 ({estimateHtmlExportSize("uint8", 4)})</MenuItem>}
                {detRows % 8 === 0 && detCols % 8 === 0 && <MenuItem onClick={() => handleHtmlExportSelect("uint8", 8)} sx={{ fontSize: 12 }}>Binned 8x uint8 ({estimateHtmlExportSize("uint8", 8)})</MenuItem>}
                <MenuItem onClick={() => handleHtmlExportSelect("uint16", 1)} sx={{ fontSize: 12 }}>Exact uint16 ({estimateHtmlExportSize("uint16", 1)})</MenuItem>
                {detRows % 2 === 0 && detCols % 2 === 0 && <MenuItem onClick={() => handleHtmlExportSelect("uint16", 2)} sx={{ fontSize: 12 }}>Binned 2x uint16 ({estimateHtmlExportSize("uint16", 2)})</MenuItem>}
                {detRows % 4 === 0 && detCols % 4 === 0 && <MenuItem onClick={() => handleHtmlExportSelect("uint16", 4)} sx={{ fontSize: 12 }}>Binned 4x uint16 ({estimateHtmlExportSize("uint16", 4)})</MenuItem>}
                {detRows % 8 === 0 && detCols % 8 === 0 && <MenuItem onClick={() => handleHtmlExportSelect("uint16", 8)} sx={{ fontSize: 12 }}>Binned 8x uint16 ({estimateHtmlExportSize("uint16", 8)})</MenuItem>}
                {pathLength > 0 && <MenuItem onClick={handleDpExportGif} sx={{ fontSize: 12 }}>GIF (path animation)</MenuItem>}
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
            </Stack>}
          </Stack>

          {/* DP Canvas */}
          <Box sx={{ ...container.imageBox, width: "100%", maxWidth: canvasSize, aspectRatio: "1 / 1", height: "auto", touchAction: "none", ...mobileImageBoxSx }}>
            <canvas ref={dpCanvasRef} width={detCols} height={detRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
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
          {showStats && dpStats && dpStats.length === 4 && (
            <Box sx={{ ...statsBarSx, ...hideBetweenPanelsOnMobileSx }}>
              <Typography sx={statsTextSx}>Mean <Box component="span" sx={statsValueSx}>{formatStat(dpStats[0])}</Box></Typography>
              <Typography sx={statsTextSx}>Min <Box component="span" sx={statsValueSx}>{formatStat(dpStats[1])}</Box></Typography>
              <Typography sx={statsTextSx}>Max <Box component="span" sx={statsValueSx}>{formatStat(dpStats[2])}</Box></Typography>
              <Typography sx={statsTextSx}>Std <Box component="span" sx={statsValueSx}>{formatStat(dpStats[3])}</Box></Typography>
              {controlsVisible && <>
                <Box sx={{ flex: 1, minWidth: 4, "@media (max-width: 700px)": { display: "none" } }} />
                <Typography component="span" onClick={() => { model.set("_preset_request", "bf"); model.save_changes(); }} sx={{ ...statsTextSx, color: roiColors.textColor, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>BF</Typography>
                <Typography component="span" onClick={() => { model.set("_preset_request", "abf"); model.save_changes(); }} sx={{ ...statsTextSx, color: "#4af", fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ABF</Typography>
                <Typography component="span" onClick={() => { model.set("_preset_request", "adf"); model.save_changes(); }} sx={{ ...statsTextSx, color: "#fa4", fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ADF</Typography>
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
                              if (roiMode === "annular") {
                                const [inner, outer] = v as number[];
                                setRoiRadiusInner(Math.min(inner, outer - 1));
                                setRoiRadius(Math.max(outer, inner + 1));
                              } else {
                                const next = Array.isArray(v) ? v[0] : v;
                                setRoiRadius(next);
                              }
                            }}
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
            <Typography sx={{ ...typo.label, color: themeColors.textMuted }}>
              {compareMode ? `Compare grid | ${shapeRows}×${shapeCols}` : `${shapeRows}×${shapeCols} | ${detRows}×${detCols}`}
            </Typography>
            {controlsVisible && <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
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
                  if (!virtualCanvasRef.current) return;
                  try {
                    const blob = await new Promise<Blob | null>(resolve => virtualCanvasRef.current!.toBlob(resolve, "image/png"));
                    if (!blob) return;
                    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                  } catch {
                    virtualCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4dstem_vi.png"); }, "image/png");
                  }
                }}>Copy</Button>
              </>}
            </Stack>}
          </Stack>

          {/* VI Canvas */}
          {compareMode ? (
            <CompareVirtualGrid
              bytes={compareVirtualImageBytes}
              count={comparePanelCount || 0}
              indices={comparePanelIndices || []}
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
                setFramePlaying(false);
                setFrameIdx(Math.max(0, Math.min(nFrames - 1, idx)));
              }}
              onToggleStar={toggleCompareStar}
              onHide={hideCompareFrame}
              onReorderFrame={moveCompareFrame}
              onDragFrameChange={setCompareDraggingFrame}
              onPendingMoveFrameChange={setComparePendingMoveFrame}
            />
          ) : (
            <Box sx={{ ...container.imageBox, width: "100%", maxWidth: viCanvasWidth, aspectRatio: `${shapeCols} / ${shapeRows}`, height: "auto", touchAction: "none", ...mobileImageBoxSx }}>
              <canvas ref={virtualCanvasRef} width={shapeCols} height={shapeRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
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
          {showStats && viStats && viStats.length === 4 && (
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
                    <Histogram data={viHistogramData} vminPct={viVminPct} vmaxPct={viVmaxPct} onRangeChange={(min, max) => { if (viAutoContrast) { viPreAutoPctRef.current = null; setViAutoContrast(false); } setViVminPct(min); setViVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={viDataMin} dataMax={viDataMax} />
                  </Box>
                </Box>
              </Box>
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
              <canvas ref={fftCanvasRef} width={shapeCols} height={shapeRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
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
              {panelChromeVisible && <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />}
            </Box>

            {/* FFT Stats Bar */}
            {showStats && fftStats && fftStats.length === 4 && (
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
            value={viewMode || "single"}
            onChange={(e) => setViewMode(String(e.target.value))}
            size="small"
            inputProps={{ "aria-label": "Show4DSTEM view mode" }}
            sx={{ ...themedSelect, minWidth: 82, fontSize: 10 }}
            MenuProps={themedMenuProps}
          >
            <MenuItem value="single">Single</MenuItem>
            <MenuItem value="temporal">Temporal</MenuItem>
            <MenuItem value="compare">Compare</MenuItem>
          </Select>
          {compareMode && (
            <>
              <Typography sx={{ ...typo.label, fontSize: 10, flexShrink: 0 }}>DP</Typography>
              <Select
                value={compareDpMode || "average"}
                onChange={(e) => setCompareDpMode(String(e.target.value))}
                size="small"
                inputProps={{ "aria-label": "Show4DSTEM compare DP source" }}
                sx={{ ...themedSelect, minWidth: 82, fontSize: 10 }}
                MenuProps={themedMenuProps}
              >
                <MenuItem value="average">Average</MenuItem>
                <MenuItem value="selected">Selected</MenuItem>
              </Select>
              <Typography sx={{ ...typo.label, fontSize: 10, flexShrink: 0 }}>Cols</Typography>
              <Select
                value={compareCols || 0}
                onChange={(e) => setCompareCols(Number(e.target.value))}
                size="small"
                inputProps={{ "aria-label": "Show4DSTEM compare columns" }}
                sx={{ ...themedSelect, minWidth: 54, fontSize: 10 }}
                MenuProps={themedMenuProps}
              >
                <MenuItem value={0}>Auto</MenuItem>
                <MenuItem value={2}>2</MenuItem>
                <MenuItem value={3}>3</MenuItem>
                <MenuItem value={4}>4</MenuItem>
                <MenuItem value={5}>5</MenuItem>
              </Select>
              <Tooltip title={compareReorderMode ? "Finish reordering" : "Reorder compare panels"}>
                <IconButton
                  size="small"
                  aria-label="Show4DSTEM compare reorder"
                  className="show4dstem-compare-reorder"
                  onClick={() => {
                    setCompareReorderMode((value) => !value);
                    setComparePendingMoveFrame(null);
                    setCompareDraggingFrame(null);
                  }}
                  sx={{ color: compareReorderMode ? themeColors.accent : themeColors.textMuted, p: 0.25 }}
                >
                  <DragIndicatorIcon sx={{ fontSize: 17 }} />
                </IconButton>
              </Tooltip>
              {compareHiddenCount > 0 && (
                <Button
                  size="small"
                  sx={compactButton}
                  className="show4dstem-compare-show-all"
                  onClick={() => setCompareHiddenPanels([])}
                >
                  Show all
                </Button>
              )}
              <Button
                size="small"
                sx={compactButton}
                className="show4dstem-compare-reset"
                disabled={
                  !(comparePanelOrder || []).length
                  && !(compareHiddenPanels || []).length
                  && !(compareStarredPanels || []).length
                }
                onClick={resetComparePanelState}
              >
                Reset
              </Button>
            </>
          )}
          <Typography sx={{ ...typo.label, fontSize: 10, flexShrink: 0 }}>{frameDimLabel}:</Typography>
          <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
            <IconButton size="small" onClick={() => { setFrameReverse(true); setFramePlaying(true); }} sx={{ color: frameReverse && framePlaying ? themeColors.accent : themeColors.textMuted, p: 0.25 }}>
              <FastRewindIcon sx={{ fontSize: 18 }} />
            </IconButton>
            <IconButton size="small" onClick={() => setFramePlaying(!framePlaying)} sx={{ color: themeColors.accent, p: 0.25 }}>
              {framePlaying ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
            </IconButton>
            <IconButton size="small" onClick={() => { setFrameReverse(false); setFramePlaying(true); }} sx={{ color: !frameReverse && framePlaying ? themeColors.accent : themeColors.textMuted, p: 0.25 }}>
              <FastForwardIcon sx={{ fontSize: 18 }} />
            </IconButton>
            <IconButton size="small" onClick={() => { setFramePlaying(false); setFrameIdx(0); }} sx={{ color: themeColors.textMuted, p: 0.25 }}>
              <StopIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Stack>
          <Slider value={frameIdx} onChange={(_, v) => { setFramePlaying(false); setFrameIdx(v as number); }} min={0} max={Math.max(0, nFrames - 1)} size="small" sx={{ flex: 1, minWidth: 60, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
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
