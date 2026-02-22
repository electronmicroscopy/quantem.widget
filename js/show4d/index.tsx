import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import StopIcon from "@mui/icons-material/Stop";
import "./styles.css";
import { useTheme } from "../theme";
import { COLORMAPS, applyColormap, renderToOffscreen } from "../colormaps";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, roundToNiceValue, exportFigure, canvasToPDF } from "../scalebar";
import { findDataRange, sliderRange, computeStats, applyLogScale } from "../stats";
import { formatNumber, downloadBlob, downloadDataView } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { getWebGPUFFT, WebGPUFFT, fft2d, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2 } from "../webgpu-fft";
import { ControlCustomizer } from "../control-customizer";
import { computeToolVisibility } from "../tool-parity";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const CANVAS_SIZE = 450;
const RESIZE_HIT_AREA_PX = 10;
const CIRCLE_HANDLE_ANGLE = 0.707;

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

function resolveScaleBarParams(pixelSize: number, unit: string): { pixelSize: number; unit: "Å" | "mrad" | "px" } {
  if (!(pixelSize > 0)) return { pixelSize: 1, unit: "px" };
  if (unit === "Å") return { pixelSize, unit: "Å" };
  if (unit === "nm") return { pixelSize: pixelSize * 10, unit: "Å" };
  if (unit === "mrad") return { pixelSize, unit: "mrad" };
  return { pixelSize, unit: "px" };
}

// ============================================================================
// UI Styles
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

const SPACING = {
  XS: 4,
  SM: 8,
  MD: 12,
  LG: 16,
};

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};

const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};

const switchStyles = {
  small: { '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

// ============================================================================
// Helpers
// ============================================================================
function formatStat(value: number): string {
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 0.001 || abs >= 10000) return value.toExponential(2);
  if (abs < 0.01) return value.toFixed(4);
  if (abs < 1) return value.toFixed(3);
  return value.toFixed(2);
}

// ============================================================================
// HiDPI Drawing Functions
// ============================================================================

/** Draw position crosshair on high-DPI canvas */
function drawPositionMarker(
  canvas: HTMLCanvasElement,
  dpr: number,
  posRow: number,
  posCol: number,
  zoom: number,
  panX: number,
  panY: number,
  imageWidth: number,
  imageHeight: number,
  isDragging: boolean,
  snapEnabled: boolean = false,
  snapRadius: number = 5
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const scaleY = cssHeight / imageHeight;

  const screenX = posCol * zoom * scaleX + panX * scaleX;
  const screenY = posRow * zoom * scaleY + panY * scaleY;

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

  // Snap radius circle
  if (snapEnabled && snapRadius > 0) {
    const radiusScreenX = snapRadius * zoom * scaleX;
    const radiusScreenY = snapRadius * zoom * scaleY;
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "rgba(0, 200, 255, 0.7)";
    ctx.lineWidth = 1.2;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, radiusScreenX, radiusScreenY, 0, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.restore();
}

/** Draw ROI overlay on high-DPI canvas for navigation panel */
function drawNavRoiOverlay(
  canvas: HTMLCanvasElement,
  dpr: number,
  roiMode: string,
  centerX: number,
  centerY: number,
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

  const screenX = centerY * zoom * scaleX + panX * scaleX;
  const screenY = centerX * zoom * scaleY + panY * scaleY;

  const lineWidth = 2.5;
  const crosshairSize = 10;
  const handleRadius = 6;

  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  const drawResizeHandle = (handleX: number, handleY: number) => {
    let handleFill: string;
    let handleStroke: string;
    if (isDraggingResize) {
      handleFill = "rgba(0, 200, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else if (isHoveringResize) {
      handleFill = "rgba(255, 100, 100, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else {
      handleFill = "rgba(0, 255, 0, 0.8)";
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
    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(screenX - crosshairSize, screenY);
    ctx.lineTo(screenX + crosshairSize, screenY);
    ctx.moveTo(screenX, screenY - crosshairSize);
    ctx.lineTo(screenX, screenY + crosshairSize);
    ctx.stroke();
  };

  const strokeColor = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
  const fillColor = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";

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
    const handleOffsetX = screenRadiusX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetY = screenRadiusY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetX, screenY + handleOffsetY);

  } else if (roiMode === "square" && radius > 0) {
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

// ============================================================================
// KeyboardShortcuts Component
// ============================================================================
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

// ============================================================================
// InfoTooltip Component
// ============================================================================
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
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);

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

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);

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

    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);

    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;
      const inRange = i >= vminBin && i <= vmaxBin;
      ctx.fillStyle = inRange ? colors.barActive : colors.barInactive;
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}` }}
      />
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
        valueLabelFormat={(pct) => {
          const val = dataMin + (pct / 100) * (dataMax - dataMin);
          return val >= 1000 ? val.toExponential(1) : val.toFixed(1);
        }}
        sx={{
          width,
          py: 0,
          "& .MuiSlider-thumb": { width: 8, height: 8 },
          "& .MuiSlider-rail": { height: 2 },
          "& .MuiSlider-track": { height: 2 },
          "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
        }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vminPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vmaxPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography></Box>
    </Box>
  );
}

// ============================================================================
// Line Profile Functions
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
// Snap-to-peak: find local intensity maximum
// ============================================================================
function findLocalMax(data: Float32Array, width: number, height: number, cc: number, cr: number, radius: number): { row: number; col: number } {
  let bestCol = cc, bestRow = cr, bestVal = -Infinity;
  const c0 = Math.max(0, cc - radius), r0 = Math.max(0, cr - radius);
  const c1 = Math.min(width - 1, cc + radius), r1 = Math.min(height - 1, cr + radius);
  for (let ir = r0; ir <= r1; ir++) {
    for (let ic = c0; ic <= c1; ic++) {
      const val = data[ir * width + ic];
      if (val > bestVal) { bestVal = val; bestCol = ic; bestRow = ir; }
    }
  }
  return { row: bestRow, col: bestCol };
}

// ============================================================================
// Crop single-mode ROI region from raw float32 data for ROI-scoped FFT
// ============================================================================
function cropSingleROI(
  data: Float32Array, imgW: number, imgH: number,
  mode: string, centerRow: number, centerCol: number,
  radius: number, roiW: number, roiH: number,
): { cropped: Float32Array; cropW: number; cropH: number } | null {
  if (mode === "off" || mode === "point") return null;
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
  if (mode === "circle" || mode === "annular") {
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

// ============================================================================
// Main Component
// ============================================================================
function Show4D() {
  const model = useModel();
  const { themeInfo, colors: themeColors } = useTheme();
  const DPR = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;

  // Themed typography — applies theme colors to module-level font sizes
  const typo = React.useMemo(() => ({
    label: { ...typography.label, color: themeColors.textMuted },
    value: { ...typography.value, color: themeColors.textMuted },
    title: { ...typography.title, color: themeColors.accent },
  }), [themeColors]);

  // ── Model State ──
  const [navRows] = useModelState<number>("nav_rows");
  const [navCols] = useModelState<number>("nav_cols");
  const [sigRows] = useModelState<number>("sig_rows");
  const [sigCols] = useModelState<number>("sig_cols");
  const [posRow, setPosRow] = useModelState<number>("pos_row");
  const [posCol, setPosCol] = useModelState<number>("pos_col");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [navImageBytes] = useModelState<DataView>("nav_image_bytes");
  const [navDataMin] = useModelState<number>("nav_data_min");
  const [navDataMax] = useModelState<number>("nav_data_max");
  const [sigDataMin] = useModelState<number>("sig_data_min");
  const [sigDataMax] = useModelState<number>("sig_data_max");
  const [roiMode, setRoiMode] = useModelState<string>("roi_mode");
  const [roiReduce, setRoiReduce] = useModelState<string>("roi_reduce");
  const [roiCenterRow] = useModelState<number>("roi_center_row");
  const [roiCenterCol] = useModelState<number>("roi_center_col");
  const [roiRadius, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");
  const [navStats] = useModelState<number[]>("nav_stats");
  const [sigStats] = useModelState<number[]>("sig_stats");
  const [navPixelSize] = useModelState<number>("nav_pixel_size");
  const [sigPixelSize] = useModelState<number>("sig_pixel_size");
  const [navPixelUnit] = useModelState<string>("nav_pixel_unit");
  const [sigPixelUnit] = useModelState<string>("sig_pixel_unit");
  const [title] = useModelState<string>("title");
  const [snapEnabled, setSnapEnabled] = useModelState<boolean>("snap_enabled");
  const [snapRadius, setSnapRadius] = useModelState<number>("snap_radius");
  const [profileLine, setProfileLine] = useModelState<{row: number; col: number}[]>("profile_line");
  const [profileWidth, setProfileWidth] = useModelState<number>("profile_width");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("Show4D", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );

  const hideDisplay = toolVisibility.isHidden("display");
  const hideHistogram = toolVisibility.isHidden("histogram");
  const hideStats = toolVisibility.isHidden("stats");
  const hideNavigation = toolVisibility.isHidden("navigation");
  const hidePlayback = toolVisibility.isHidden("playback");
  const hideView = toolVisibility.isHidden("view");
  const hideExport = toolVisibility.isHidden("export");
  const hideRoi = toolVisibility.isHidden("roi");
  const hideProfile = toolVisibility.isHidden("profile");
  const hideFft = toolVisibility.isHidden("fft");

  const lockDisplay = toolVisibility.isLocked("display");
  const lockHistogram = toolVisibility.isLocked("histogram");
  const lockStats = toolVisibility.isLocked("stats");
  const lockNavigation = toolVisibility.isLocked("navigation");
  const lockPlayback = toolVisibility.isLocked("playback");
  const lockView = toolVisibility.isLocked("view");
  const lockExport = toolVisibility.isLocked("export");
  const lockRoi = toolVisibility.isLocked("roi");
  const lockProfile = toolVisibility.isLocked("profile");
  const lockFft = toolVisibility.isLocked("fft");
  const effectiveShowFft = showFft && !hideFft;
  const accentGreen = themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a";

  // ROI FFT state
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);
  const roiFftActive = effectiveShowFft && roiMode !== "off" && roiMode !== "point";

  // Path animation
  const [pathPlaying, setPathPlaying] = useModelState<boolean>("path_playing");
  const [pathIndex, setPathIndex] = useModelState<number>("path_index");
  const [pathLength] = useModelState<number>("path_length");
  const [pathIntervalMs] = useModelState<number>("path_interval_ms");
  const [pathLoop] = useModelState<boolean>("path_loop");

  // Export
  const [, setGifExportRequested] = useModelState<boolean>("_gif_export_requested");
  const [gifData] = useModelState<DataView>("_gif_data");
  const [gifMetadataJson] = useModelState<string>("_gif_metadata_json");
  const [exporting, setExporting] = React.useState(false);
  const [navExportAnchor, setNavExportAnchor] = React.useState<HTMLElement | null>(null);
  const [sigExportAnchor, setSigExportAnchor] = React.useState<HTMLElement | null>(null);

  // ── Local State ──
  const [localPosRow, setLocalPosRow] = React.useState(posRow + 0.5);
  const [localPosCol, setLocalPosCol] = React.useState(posCol + 0.5);
  const [isDraggingNav, setIsDraggingNav] = React.useState(false);
  const [isDraggingRoi, setIsDraggingRoi] = React.useState(false);
  const [isDraggingRoiResize, setIsDraggingRoiResize] = React.useState(false);
  const [isHoveringRoiResize, setIsHoveringRoiResize] = React.useState(false);
  const resizeAspectRef = React.useRef<number | null>(null);
  const [localRoiCenterRow, setLocalRoiCenterRow] = React.useState(roiCenterRow);
  const [localRoiCenterCol, setLocalRoiCenterCol] = React.useState(roiCenterCol);

  // Signal panel drag-to-pan
  const [isDraggingSig, setIsDraggingSig] = React.useState(false);
  const [sigDragStart, setSigDragStart] = React.useState<{ x: number; y: number; panX: number; panY: number } | null>(null);

  // Independent colormaps and scales
  const [navColormap, setNavColormap] = React.useState("inferno");
  const [sigColormap, setSigColormap] = React.useState("inferno");
  const [navScaleMode, setNavScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const [sigScaleMode, setSigScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const navPowerExp = 0.5;
  const sigPowerExp = 0.5;
  const [navVminPct, setNavVminPct] = React.useState(0);
  const [navVmaxPct, setNavVmaxPct] = React.useState(100);
  const [sigVminPct, setSigVminPct] = React.useState(0);
  const [sigVmaxPct, setSigVmaxPct] = React.useState(100);

  // Zoom state
  const [navZoom, setNavZoom] = React.useState(1);
  const [navPanX, setNavPanX] = React.useState(0);
  const [navPanY, setNavPanY] = React.useState(0);
  const [sigZoom, setSigZoom] = React.useState(1);
  const [sigPanX, setSigPanX] = React.useState(0);
  const [sigPanY, setSigPanY] = React.useState(0);

  // Canvas resize state
  const [canvasSize, setCanvasSize] = React.useState(CANVAS_SIZE);
  const [isResizing, setIsResizing] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number } | null>(null);

  // Histogram data
  const [navHistogramData, setNavHistogramData] = React.useState<Float32Array | null>(null);
  const [sigHistogramData, setSigHistogramData] = React.useState<Float32Array | null>(null);
  const [navOffscreenVersion, setNavOffscreenVersion] = React.useState(0);
  const [sigOffscreenVersion, setSigOffscreenVersion] = React.useState(0);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profilePoints = profileLine || [];
  const rawSigDataRef = React.useRef<Float32Array | null>(null);
  const sigClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [draggingProfileEndpoint, setDraggingProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingProfileLine, setIsDraggingProfileLine] = React.useState(false);
  const [hoveredProfileEndpoint, setHoveredProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringProfileLine, setIsHoveringProfileLine] = React.useState(false);
  const sigProfileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);

  // FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftUiRef = React.useRef<HTMLCanvasElement>(null);
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const fftMagRef = React.useRef<Float32Array | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);
  const [fftOffscreenVersion, setFftOffscreenVersion] = React.useState(0);
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftLogScale, setFftLogScale] = React.useState(false);
  const [fftAuto, setFftAuto] = React.useState(true);
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftStats, setFftStats] = React.useState<{ mean: number; min: number; max: number; std: number }>({ mean: 0, min: 0, max: 0, std: 0 });
  const [isDraggingFft, setIsDraggingFft] = React.useState(false);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number; y: number; panX: number; panY: number } | null>(null);

  // FFT d-spacing click measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);

  // ROI toggle memory
  const lastRoiModeRef = React.useRef<string>("circle");

  // Cursor readout
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number; panel: string } | null>(null);

  // Aspect-ratio-aware canvas sizes
  const navCanvasWidth = navRows > navCols ? Math.round(canvasSize * (navCols / navRows)) : canvasSize;
  const navCanvasHeight = navCols > navRows ? Math.round(canvasSize * (navRows / navCols)) : canvasSize;
  const sigCanvasWidth = sigRows > sigCols ? Math.round(canvasSize * (sigCols / sigRows)) : canvasSize;
  const sigCanvasHeight = sigCols > sigRows ? Math.round(canvasSize * (sigRows / sigCols)) : canvasSize;

  // Canvas refs (three-canvas stack per panel)
  const navCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const navOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const navUiRef = React.useRef<HTMLCanvasElement>(null);
  const navOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const navImageDataRef = React.useRef<ImageData | null>(null);
  const rawNavImageRef = React.useRef<Float32Array | null>(null);

  const sigCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const sigOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const sigUiRef = React.useRef<HTMLCanvasElement>(null);
  const sigOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const sigImageDataRef = React.useRef<ImageData | null>(null);
  const rootRef = React.useRef<HTMLDivElement>(null);

  const isTypingTarget = React.useCallback((target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) return false;
    if (target.isContentEditable) return true;
    return target.closest("input, textarea, select, [role='textbox'], [contenteditable='true']") !== null;
  }, []);

  const handleRootMouseDownCapture = React.useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement | null;
    if (target?.closest("canvas")) rootRef.current?.focus();
  }, []);

  React.useEffect(() => {
    if (hideFft && showFft) {
      setShowFft(false);
    }
  }, [hideFft, showFft, setShowFft]);

  React.useEffect(() => {
    if (lockPlayback && pathPlaying) {
      setPathPlaying(false);
    }
  }, [lockPlayback, pathPlaying, setPathPlaying]);

  React.useEffect(() => {
    if (hideRoi && roiMode !== "off") {
      setRoiMode("off");
    }
  }, [hideRoi, roiMode, setRoiMode]);

  React.useEffect(() => {
    if (hideProfile && profileActive) {
      setProfileActive(false);
      setProfileLine([]);
      setProfileData(null);
      setHoveredProfileEndpoint(null);
      setIsHoveringProfileLine(false);
    }
  }, [hideProfile, profileActive, setProfileLine]);

  // ── Sync local state ──
  React.useEffect(() => {
    if (!isDraggingNav) { setLocalPosRow(posRow + 0.5); setLocalPosCol(posCol + 0.5); }
  }, [posRow, posCol, isDraggingNav]);

  React.useEffect(() => {
    if (!isDraggingRoi && !isDraggingRoiResize) {
      setLocalRoiCenterRow(roiCenterRow);
      setLocalRoiCenterCol(roiCenterCol);
    }
  }, [roiCenterRow, roiCenterCol, isDraggingRoi, isDraggingRoiResize]);

  // ── Prevent scroll on canvases ──
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const overlays = [navOverlayRef.current, sigOverlayRef.current, fftOverlayRef.current];
    overlays.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => overlays.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, [effectiveShowFft]);

  // ── GPU FFT init ──
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) { gpuFFTRef.current = fft; setGpuReady(true); }
    });
  }, []);

  // ── Path animation timer ──
  React.useEffect(() => {
    if (!pathPlaying || pathLength === 0) return;
    const timer = setInterval(() => {
      setPathIndex((prev: number) => {
        const next = prev + 1;
        if (next >= pathLength) {
          if (pathLoop) {
            return 0;
          } else {
            setPathPlaying(false);
            return prev;
          }
        }
        return next;
      });
    }, pathIntervalMs);
    return () => clearInterval(timer);
  }, [pathPlaying, pathLength, pathIntervalMs, pathLoop, setPathIndex, setPathPlaying]);

  // ── Parse nav image bytes ──
  React.useEffect(() => {
    if (!navImageBytes) return;
    const numFloats = navImageBytes.byteLength / 4;
    const rawData = new Float32Array(navImageBytes.buffer, navImageBytes.byteOffset, numFloats);
    let storedData = rawNavImageRef.current;
    if (!storedData || storedData.length !== numFloats) {
      storedData = new Float32Array(numFloats);
      rawNavImageRef.current = storedData;
    }
    storedData.set(rawData);

    const scaledData = new Float32Array(numFloats);
    if (navScaleMode === "log") {
      for (let i = 0; i < numFloats; i++) scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
    } else if (navScaleMode === "power") {
      for (let i = 0; i < numFloats; i++) scaledData[i] = Math.pow(Math.max(0, rawData[i]), navPowerExp);
    } else {
      scaledData.set(rawData);
    }
    setNavHistogramData(scaledData);
  }, [navImageBytes, navScaleMode, navPowerExp]);

  // ── Parse signal frame bytes ──
  React.useEffect(() => {
    if (!frameBytes) return;
    const rawData = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
    // Store raw data for profile sampling
    if (!rawSigDataRef.current || rawSigDataRef.current.length !== rawData.length) {
      rawSigDataRef.current = new Float32Array(rawData.length);
    }
    rawSigDataRef.current.set(rawData);
    const scaledData = new Float32Array(rawData.length);
    if (sigScaleMode === "log") {
      for (let i = 0; i < rawData.length; i++) scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
    } else if (sigScaleMode === "power") {
      for (let i = 0; i < rawData.length; i++) scaledData[i] = Math.pow(Math.max(0, rawData[i]), sigPowerExp);
    } else {
      scaledData.set(rawData);
    }
    setSigHistogramData(scaledData);
  }, [frameBytes, sigScaleMode, sigPowerExp]);

  // ── Render nav image to offscreen (colormap pipeline — no zoom/pan deps) ──
  React.useEffect(() => {
    if (!rawNavImageRef.current) return;

    const rawData = rawNavImageRef.current;
    let scaled: Float32Array;
    if (navScaleMode === "log") {
      scaled = new Float32Array(rawData.length);
      for (let i = 0; i < rawData.length; i++) scaled[i] = Math.log1p(Math.max(0, rawData[i]));
    } else if (navScaleMode === "power") {
      scaled = new Float32Array(rawData.length);
      for (let i = 0; i < rawData.length; i++) scaled[i] = Math.pow(Math.max(0, rawData[i]), navPowerExp);
    } else {
      scaled = rawData;
    }

    const { min: dataMin, max: dataMax } = findDataRange(scaled);
    const { vmin, vmax } = sliderRange(dataMin, dataMax, navVminPct, navVmaxPct);

    const width = navCols;
    const height = navRows;
    const lut = COLORMAPS[navColormap] || COLORMAPS.inferno;

    let offscreen = navOffscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      navOffscreenRef.current = offscreen;
    }
    if (offscreen.width !== width || offscreen.height !== height) {
      offscreen.width = width;
      offscreen.height = height;
      navImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    let imgData = navImageDataRef.current;
    if (!imgData) {
      imgData = offCtx.createImageData(width, height);
      navImageDataRef.current = imgData;
    }
    applyColormap(scaled, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);
    setNavOffscreenVersion(v => v + 1);
  }, [navImageBytes, navColormap, navVminPct, navVmaxPct, navScaleMode, navPowerExp, navRows, navCols]);

  // ── Nav zoom/pan redraw (lightweight — just drawImage with transform) ──
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  React.useLayoutEffect(() => {
    if (!navCanvasRef.current || !navOffscreenRef.current) return;
    const canvas = navCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(navPanX, navPanY);
    ctx.scale(navZoom, navZoom);
    ctx.drawImage(navOffscreenRef.current, 0, 0);
    ctx.restore();
  }, [navOffscreenVersion, navZoom, navPanX, navPanY]);

  // ── Render signal frame to offscreen (colormap pipeline — no zoom/pan deps) ──
  React.useEffect(() => {
    if (!frameBytes) return;

    const rawData = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
    let scaled: Float32Array;
    if (sigScaleMode === "log") {
      scaled = new Float32Array(rawData.length);
      for (let i = 0; i < rawData.length; i++) scaled[i] = Math.log1p(Math.max(0, rawData[i]));
    } else if (sigScaleMode === "power") {
      scaled = new Float32Array(rawData.length);
      for (let i = 0; i < rawData.length; i++) scaled[i] = Math.pow(Math.max(0, rawData[i]), sigPowerExp);
    } else {
      scaled = rawData;
    }

    const { min: dataMin, max: dataMax } = findDataRange(scaled);
    const { vmin, vmax } = sliderRange(dataMin, dataMax, sigVminPct, sigVmaxPct);

    const width = sigCols;
    const height = sigRows;
    const lut = COLORMAPS[sigColormap] || COLORMAPS.inferno;

    let offscreen = sigOffscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      sigOffscreenRef.current = offscreen;
    }
    if (offscreen.width !== width || offscreen.height !== height) {
      offscreen.width = width;
      offscreen.height = height;
      sigImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    let imgData = sigImageDataRef.current;
    if (!imgData) {
      imgData = offCtx.createImageData(width, height);
      sigImageDataRef.current = imgData;
    }
    applyColormap(scaled, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);
    setSigOffscreenVersion(v => v + 1);
  }, [frameBytes, sigColormap, sigVminPct, sigVmaxPct, sigScaleMode, sigPowerExp, sigRows, sigCols]);

  // ── Signal zoom/pan redraw (lightweight — just drawImage with transform) ──
  React.useLayoutEffect(() => {
    if (!sigCanvasRef.current || !sigOffscreenRef.current) return;
    const canvas = sigCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(sigPanX, sigPanY);
    ctx.scale(sigZoom, sigZoom);
    ctx.drawImage(sigOffscreenRef.current, 0, 0);
    ctx.restore();
  }, [sigOffscreenVersion, sigZoom, sigPanX, sigPanY]);

  // ── Compute FFT from signal frame (supports ROI-scoped FFT) ──
  React.useEffect(() => {
    if (!effectiveShowFft || !rawSigDataRef.current) { setFftCropDims(null); return; }
    let cancelled = false;
    const data = rawSigDataRef.current;
    let w = sigCols, h = sigRows;
    let inputData = data;
    let origCropW = 0, origCropH = 0;

    // ROI FFT: crop to ROI region and pre-pad to power-of-2
    if (roiFftActive) {
      const crop = cropSingleROI(data, sigCols, sigRows, roiMode, roiCenterRow, roiCenterCol, roiRadius, roiWidth, roiHeight);
      if (crop) {
        origCropW = crop.cropW;
        origCropH = crop.cropH;
        const padW = nextPow2(crop.cropW);
        const padH = nextPow2(crop.cropH);
        const padded = new Float32Array(padW * padH);
        for (let y = 0; y < crop.cropH; y++) {
          for (let x = 0; x < crop.cropW; x++) {
            padded[y * padW + x] = crop.cropped[y * crop.cropW + x];
          }
        }
        inputData = padded;
        w = padW;
        h = padH;
      }
    }

    const fftW = w, fftH = h;
    const computeFFT = async () => {
      let real: Float32Array, imag: Float32Array;
      if (gpuReady && gpuFFTRef.current) {
        const result = await gpuFFTRef.current.fft2D(inputData.slice(), new Float32Array(inputData.length), fftW, fftH, false);
        real = result.real;
        imag = result.imag;
      } else {
        real = inputData.slice();
        imag = new Float32Array(inputData.length);
        fft2d(real, imag, fftW, fftH, false);
      }
      if (cancelled) return;
      fftshift(real, fftW, fftH);
      fftshift(imag, fftW, fftH);
      fftMagRef.current = computeMagnitude(real, imag);
      setFftCropDims(origCropW > 0 ? { cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH } : null);
      setFftMagVersion(v => v + 1);
      setFftClickInfo(null);
    };
    computeFFT();
    return () => { cancelled = true; };
  }, [effectiveShowFft, roiFftActive, frameBytes, sigRows, sigCols, gpuReady, roiMode, roiCenterRow, roiCenterCol, roiRadius, roiWidth, roiHeight]);

  // ── Render FFT image ──
  React.useEffect(() => {
    const mag = fftMagRef.current;
    if (!effectiveShowFft || !mag) return;

    const w = fftCropDims?.fftWidth ?? sigCols, h = fftCropDims?.fftHeight ?? sigRows;
    let displayMin: number, displayMax: number;
    if (fftAuto) {
      ({ min: displayMin, max: displayMax } = autoEnhanceFFT(mag, w, h));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(mag));
    }

    const displayData = fftLogScale ? applyLogScale(mag) : mag;
    if (fftLogScale) {
      displayMin = Math.log1p(displayMin);
      displayMax = Math.log1p(displayMax);
    }

    setFftHistogramData(displayData);
    setFftDataRange({ min: displayMin, max: displayMax });
    setFftStats(computeStats(displayData));

    const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(displayData, w, h, lut, vmin, vmax);
    if (!offscreen) return;
    fftOffscreenRef.current = offscreen;
    setFftOffscreenVersion(v => v + 1);
  }, [effectiveShowFft, fftMagVersion, fftLogScale, fftAuto, fftVminPct, fftVmaxPct, fftColormap, sigRows, sigCols, fftCropDims]);

  // ── FFT zoom/pan redraw (lightweight — just drawImage with transform) ──
  React.useLayoutEffect(() => {
    if (!effectiveShowFft || !fftCanvasRef.current || !fftOffscreenRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(fftPanX, fftPanY);
    ctx.scale(fftZoom, fftZoom);
    ctx.drawImage(fftOffscreenRef.current, 0, 0);
    ctx.restore();
  }, [effectiveShowFft, fftOffscreenVersion, fftZoom, fftPanX, fftPanY]);

  // ── FFT UI overlay (scale bar + d-spacing crosshair) ──
  React.useEffect(() => {
    if (!fftUiRef.current || !effectiveShowFft) return;
    const canvas = fftUiRef.current;
    canvas.width = sigCanvasWidth * DPR;
    canvas.height = sigCanvasHeight * DPR;
    const fftW = fftCropDims?.fftWidth ?? sigCols;
    if (sigPixelSize > 0) {
      const recipPxSize = 1.0 / (sigPixelSize * fftW);
      drawFFTScaleBarHiDPI(canvas, DPR, fftZoom, recipPxSize, fftW);
    } else {
      drawScaleBarHiDPI(canvas, DPR, fftZoom, 1, "px", fftW);
    }

    // D-spacing crosshair marker
    if (fftClickInfo) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.save();
        ctx.scale(DPR, DPR);
        const screenX = (fftPanX + fftClickInfo.col * fftZoom) * sigCanvasWidth / sigCols;
        const screenY = (fftPanY + fftClickInfo.row * fftZoom) * sigCanvasHeight / sigRows;
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
        if (fftClickInfo.dSpacing != null) {
          const d = fftClickInfo.dSpacing;
          const label = d >= 10 ? `d = ${(d / 10).toFixed(2)} nm` : `d = ${d.toFixed(2)} \u00C5`;
          ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
          ctx.fillStyle = "white";
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(label, screenX + 10, screenY - 4);
        }
        ctx.restore();
      }
    }
  }, [effectiveShowFft, fftZoom, fftPanX, fftPanY, sigPixelSize, sigPixelUnit, sigCols, sigRows, sigCanvasWidth, sigCanvasHeight, fftCropDims, fftClickInfo]);

  // ── Nav HiDPI UI overlay ──
  React.useEffect(() => {
    if (!navUiRef.current) return;
    const navScale = resolveScaleBarParams(navPixelSize, navPixelUnit);
    drawScaleBarHiDPI(navUiRef.current, DPR, navZoom, navScale.pixelSize, navScale.unit, navCols);

    if (roiMode === "off") {
      drawPositionMarker(navUiRef.current, DPR, localPosRow, localPosCol, navZoom, navPanX, navPanY, navCols, navRows, isDraggingNav, snapEnabled, snapRadius);
    } else {
      drawNavRoiOverlay(
        navUiRef.current, DPR, roiMode,
        localRoiCenterRow, localRoiCenterCol, roiRadius, roiWidth, roiHeight,
        navZoom, navPanX, navPanY, navCols, navRows,
        isDraggingRoi, isDraggingRoiResize, isHoveringRoiResize
      );
    }
  }, [navZoom, navPanX, navPanY, navPixelSize, navPixelUnit, navRows, navCols,
      localPosRow, localPosCol, isDraggingNav, snapEnabled, snapRadius,
      roiMode, localRoiCenterRow, localRoiCenterCol, roiRadius, roiWidth, roiHeight,
      isDraggingRoi, isDraggingRoiResize, isHoveringRoiResize]);

  // ── Signal HiDPI UI overlay ──
  React.useEffect(() => {
    if (!sigUiRef.current) return;
    const canvas = sigUiRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const sigScale = resolveScaleBarParams(sigPixelSize, sigPixelUnit);
    drawScaleBarHiDPI(canvas, DPR, sigZoom, sigScale.pixelSize, sigScale.unit, sigCols);

    // Profile line overlay
    if (profileActive && profilePoints.length > 0) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const cssW = canvas.width / DPR;
      const cssH = canvas.height / DPR;
      const scaleX = cssW / sigCols;
      const scaleY = cssH / sigRows;
      const toScreenX = (col: number) => col * sigZoom * scaleX + sigPanX * scaleX;
      const toScreenY = (row: number) => row * sigZoom * scaleY + sigPanY * scaleY;

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

        // Draw line A→B
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
  }, [sigZoom, sigPanX, sigPanY, sigPixelSize, sigPixelUnit, sigRows, sigCols,
      profileActive, profilePoints, profileWidth, themeColors]);

  // ── Profile computation ──
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawSigDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      setProfileData(sampleLineProfile(rawSigDataRef.current, sigCols, sigRows, p0.row, p0.col, p1.row, p1.col, profileWidth));
      if (!profileActive) setProfileActive(true);
    } else {
      setProfileData(null);
    }
  }, [profilePoints, profileWidth, frameBytes]);

  // ── Profile sparkline rendering ──
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = sigCanvasWidth;
    const cssH = 76;
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
      ctx.fillText("Click two points on the signal to draw a profile", cssW / 2, cssH / 2);
      return;
    }

    const padTop = 6;
    const padBottom = 18;
    const plotH = cssH - padTop - padBottom;

    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < profileData.length; i++) {
      if (profileData[i] < gMin) gMin = profileData[i];
      if (profileData[i] > gMax) gMax = profileData[i];
    }
    const range = gMax - gMin || 1;

    // Draw profile curve
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < profileData.length; i++) {
      const x = (i / (profileData.length - 1)) * cssW;
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
      if (sigPixelSize > 0) {
        totalDist = distPx * sigPixelSize;
        xUnit = sigPixelUnit;
      } else {
        totalDist = distPx;
      }
    }

    // Draw x-axis ticks
    const tickY = padTop + plotH;
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    const idealTicks = Math.max(2, Math.floor(cssW / 70));
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
      const x = frac * cssW;
      ctx.beginPath(); ctx.moveTo(x, tickY); ctx.lineTo(x, tickY + 3); ctx.stroke();
      ctx.textAlign = frac < 0.05 ? "left" : frac > 0.95 ? "right" : "center";
      const valStr = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${valStr} ${xUnit}` : valStr, x, tickY + 4);
    }

    // Y-axis min/max labels
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), 2, 1);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), 2, padTop + plotH - 1);
  }, [profileData, profilePoints, sigPixelSize, sigPixelUnit, sigCanvasWidth, themeInfo.theme, themeColors.accent]);

  // ── Zoom handler factory ──
  const createZoomHandler = (
    setZoom: React.Dispatch<React.SetStateAction<number>>,
    setPanXFn: React.Dispatch<React.SetStateAction<number>>,
    setPanYFn: React.Dispatch<React.SetStateAction<number>>,
    zoom: number, panX: number, panY: number,
    canvasRef: React.RefObject<HTMLCanvasElement | null>,
    locked: boolean = false,
  ) => (e: React.WheelEvent<HTMLCanvasElement>) => {
    if (locked) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * zoomFactor));
    const zoomRatio = newZoom / zoom;
    setZoom(newZoom);
    setPanXFn(mouseX - (mouseX - panX) * zoomRatio);
    setPanYFn(mouseY - (mouseY - panY) * zoomRatio);
  };

  // ── Resize handle hit test ──
  const isNearRoiResizeHandle = (imgX: number, imgY: number): boolean => {
    if (roiMode === "off") return false;
    if (roiMode === "rect") {
      const halfH = (roiHeight || 10) / 2;
      const halfW = (roiWidth || 10) / 2;
      const handleX = localRoiCenterRow + halfH;
      const handleY = localRoiCenterCol + halfW;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      const cornerDist = Math.sqrt(halfW ** 2 + halfH ** 2);
      const hitArea = Math.min(RESIZE_HIT_AREA_PX / navZoom, cornerDist * 0.5);
      return dist < hitArea;
    }
    if (roiMode === "circle" || roiMode === "square") {
      const radius = roiRadius || 5;
      const offset = roiMode === "square" ? radius : radius * CIRCLE_HANDLE_ANGLE;
      const handleX = localRoiCenterRow + offset;
      const handleY = localRoiCenterCol + offset;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      const hitArea = Math.min(RESIZE_HIT_AREA_PX / navZoom, radius * 0.5);
      return dist < hitArea;
    }
    return false;
  };

  // ── Nav mouse handlers ──
  const handleNavMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = navOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - navPanY) / navZoom;
    const imgY = (screenX - navPanX) / navZoom;

    if (roiMode !== "off") {
      if (lockRoi) return;
      if (isNearRoiResizeHandle(imgX, imgY)) {
        e.preventDefault();
        resizeAspectRef.current = roiMode === "rect" && roiWidth > 0 && roiHeight > 0 ? roiWidth / roiHeight : null;
        setIsDraggingRoiResize(true);
        return;
      }
      setIsDraggingRoi(true);
      setLocalRoiCenterRow(imgX);
      setLocalRoiCenterCol(imgY);
      const newX = Math.round(Math.max(0, Math.min(navRows - 1, imgX)));
      const newY = Math.round(Math.max(0, Math.min(navCols - 1, imgY)));
      model.set("roi_center", [newX, newY]);
      model.save_changes();
      return;
    }

    if (lockNavigation) return;
    setIsDraggingNav(true);
    let newX = Math.round(Math.max(0, Math.min(navRows - 1, imgX)));
    let newY = Math.round(Math.max(0, Math.min(navCols - 1, imgY)));
    if (snapEnabled && rawNavImageRef.current) {
      const snapped = findLocalMax(rawNavImageRef.current, navCols, navRows, newY, newX, snapRadius);
      newX = snapped.row;
      newY = snapped.col;
    }
    setLocalPosRow(newX + 0.5);
    setLocalPosCol(newY + 0.5);
    model.set("pos_row", newX);
    model.set("pos_col", newY);
    model.save_changes();
  };

  const handleNavMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = navOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - navPanY) / navZoom;
    const imgY = (screenX - navPanX) / navZoom;

    // Fast path: during active drags, skip cursor readout and hover detection
    if (isDraggingRoiResize) {
      if (lockRoi) return;
      const dx = Math.abs(imgX - localRoiCenterRow);
      const dy = Math.abs(imgY - localRoiCenterCol);
      if (roiMode === "rect") {
        let newW = Math.max(2, Math.round(dy * 2));
        let newH = Math.max(2, Math.round(dx * 2));
        if (e.shiftKey && resizeAspectRef.current != null) {
          const aspect = resizeAspectRef.current;
          if (newW / newH > aspect) newH = Math.max(2, Math.round(newW / aspect));
          else newW = Math.max(2, Math.round(newH * aspect));
        }
        setRoiWidth(newW);
        setRoiHeight(newH);
      } else if (roiMode === "square") {
        setRoiRadius(Math.max(1, Math.round(Math.max(dx, dy))));
      } else {
        setRoiRadius(Math.max(1, Math.round(Math.sqrt(dx ** 2 + dy ** 2))));
      }
      return;
    }

    // ROI center dragging — skip cursor readout
    if (isDraggingRoi) {
      if (lockRoi) return;
      setLocalRoiCenterRow(imgX);
      setLocalRoiCenterCol(imgY);
      const newX = Math.round(Math.max(0, Math.min(navRows - 1, imgX)));
      const newY = Math.round(Math.max(0, Math.min(navCols - 1, imgY)));
      model.set("roi_center", [newX, newY]);
      model.save_changes();
      return;
    }

    // Position dragging — skip cursor readout
    if (isDraggingNav) {
      if (lockNavigation) return;
      let newX = Math.round(Math.max(0, Math.min(navRows - 1, imgX)));
      let newY = Math.round(Math.max(0, Math.min(navCols - 1, imgY)));
      if (snapEnabled && rawNavImageRef.current) {
        const snapped = findLocalMax(rawNavImageRef.current, navCols, navRows, newY, newX, snapRadius);
        newX = snapped.row;
        newY = snapped.col;
      }
      setLocalPosRow(newX + 0.5);
      setLocalPosCol(newY + 0.5);
      model.set("pos_row", newX);
      model.set("pos_col", newY);
      model.save_changes();
      return;
    }

    // Idle: cursor readout + hover detection
    const pxRow = Math.floor(imgX);
    const pxCol = Math.floor(imgY);
    if (pxRow >= 0 && pxRow < navRows && pxCol >= 0 && pxCol < navCols && rawNavImageRef.current) {
      setCursorInfo({ row: pxRow, col: pxCol, value: rawNavImageRef.current[pxRow * navCols + pxCol], panel: "nav" });
    } else {
      setCursorInfo(prev => prev?.panel === "nav" ? null : prev);
    }

    // Hover check for resize handles
    if (!lockRoi) {
      setIsHoveringRoiResize(isNearRoiResizeHandle(imgX, imgY));
    } else {
      setIsHoveringRoiResize(false);
    }
  };

  const handleNavMouseUp = () => {
    setIsDraggingNav(false);
    setIsDraggingRoi(false);
    setIsDraggingRoiResize(false);
  };
  const handleNavMouseLeave = () => {
    setIsDraggingNav(false);
    setIsDraggingRoi(false);
    setIsDraggingRoiResize(false);
    setIsHoveringRoiResize(false);
    setCursorInfo(prev => prev?.panel === "nav" ? null : prev);
  };
  const handleNavDoubleClick = () => {
    if (lockView) return;
    setNavZoom(1);
    setNavPanX(0);
    setNavPanY(0);
  };

  // ── FFT screen-to-image coordinate helper ──
  const fftScreenToImg = (e: React.MouseEvent): { row: number; col: number } | null => {
    const canvas = fftOverlayRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const canvasX = (e.clientX - rect.left) * scaleX;
    const canvasY = (e.clientY - rect.top) * scaleY;
    const imgCol = (canvasX - fftPanX) / fftZoom;
    const imgRow = (canvasY - fftPanY) / fftZoom;
    return { row: imgRow, col: imgCol };
  };

  // ── FFT mouse handlers ──
  const handleFftMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (lockView || lockFft) return;
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsDraggingFft(true);
    setFftDragStart({ x: e.clientX, y: e.clientY, panX: fftPanX, panY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (lockView || lockFft) return;
    if (!isDraggingFft || !fftDragStart) return;
    const canvas = fftOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    setFftPanX(fftDragStart.panX + (e.clientX - fftDragStart.x) * scaleX);
    setFftPanY(fftDragStart.panY + (e.clientY - fftDragStart.y) * scaleY);
  };

  const handleFftMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const pos = fftScreenToImg(e);
        if (pos) {
          const fftW = fftCropDims?.fftWidth ?? sigCols;
          const fftH = fftCropDims?.fftHeight ?? sigRows;
          let imgCol = pos.col;
          let imgRow = pos.row;
          // Snap to nearest Bragg spot (local max in FFT magnitude)
          if (fftMagRef.current) {
            const snapped = findFFTPeak(fftMagRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
            imgCol = snapped.col;
            imgRow = snapped.row;
          }
          const halfW = Math.floor(fftW / 2);
          const halfH = Math.floor(fftH / 2);
          const dcol = imgCol - halfW;
          const drow = imgRow - halfH;
          const distPx = Math.sqrt(dcol * dcol + drow * drow);
          if (distPx < 1) {
            setFftClickInfo(null);
          } else {
            let spatialFreq: number | null = null;
            let dSpacing: number | null = null;
            if (sigPixelSize > 0) {
              const paddedW = nextPow2(fftW);
              const paddedH = nextPow2(fftH);
              const binC = ((Math.round(imgCol) - halfW) % fftW + fftW) % fftW;
              const binR = ((Math.round(imgRow) - halfH) % fftH + fftH) % fftH;
              const freqC = binC <= paddedW / 2 ? binC / (paddedW * sigPixelSize) : (binC - paddedW) / (paddedW * sigPixelSize);
              const freqR = binR <= paddedH / 2 ? binR / (paddedH * sigPixelSize) : (binR - paddedH) / (paddedH * sigPixelSize);
              spatialFreq = Math.sqrt(freqC * freqC + freqR * freqR);
              dSpacing = spatialFreq > 0 ? 1 / spatialFreq : null;
            }
            setFftClickInfo({ row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
          }
        }
      }
      fftClickStartRef.current = null;
    }
    setIsDraggingFft(false);
    setFftDragStart(null);
  };
  const handleFftMouseLeave = () => { fftClickStartRef.current = null; setIsDraggingFft(false); setFftDragStart(null); };
  const handleFftDoubleClick = () => {
    if (lockView || lockFft) return;
    setFftZoom(1);
    setFftPanX(0);
    setFftPanY(0);
    setFftClickInfo(null);
  };

  // ── Signal mouse handlers (drag-to-pan + profile click) ──
  const handleSigMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (profileActive && lockProfile) return;
    if (!profileActive && lockView) return;
    sigClickStartRef.current = { x: e.clientX, y: e.clientY };
    const canvas = sigOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgCol = (screenX - sigPanX) / sigZoom;
    const imgRow = (screenY - sigPanY) / sigZoom;

    if (profileActive) {
      if (profilePoints.length === 2) {
        const p0 = profilePoints[0];
        const p1 = profilePoints[1];
        const hitRadius = 10 / sigZoom;
        const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
        const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingSig(false);
          setSigDragStart(null);
          return;
        }
        if (pointToSegmentDistance(imgCol, imgRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
          setIsDraggingProfileLine(true);
          sigProfileDragStartRef.current = {
            row: imgRow,
            col: imgCol,
            p0: { row: p0.row, col: p0.col },
            p1: { row: p1.row, col: p1.col },
          };
          setIsDraggingSig(false);
          setSigDragStart(null);
          return;
        }
      }
      setIsDraggingSig(true);
      setSigDragStart({ x: e.clientX, y: e.clientY, panX: sigPanX, panY: sigPanY });
      return;
    }

    setIsDraggingSig(true);
    setSigDragStart({ x: e.clientX, y: e.clientY, panX: sigPanX, panY: sigPanY });
  };

  const handleSigMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = sigOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();

    // Fast path: during pan drag, skip all cursor/hover/profile work — just update pan
    if (isDraggingSig && sigDragStart && !lockView) {
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - sigDragStart.x) * scaleX;
      const dy = (e.clientY - sigDragStart.y) * scaleY;
      setSigPanX(sigDragStart.panX + dx);
      setSigPanY(sigDragStart.panY + dy);
      return;
    }

    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgCol = (screenX - sigPanX) / sigZoom;
    const imgRow = (screenY - sigPanY) / sigZoom;
    const pxRow = Math.floor(imgRow);
    const pxCol = Math.floor(imgCol);

    // Cursor readout
    if (pxRow >= 0 && pxRow < sigRows && pxCol >= 0 && pxCol < sigCols && frameBytes) {
      const raw = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
      setCursorInfo({ row: pxRow, col: pxCol, value: raw[pxRow * sigCols + pxCol], panel: "sig" });
    } else {
      setCursorInfo(prev => prev?.panel === "sig" ? null : prev);
    }

    if (profileActive && !lockProfile && rawSigDataRef.current && profilePoints.length === 2) {
      const p0 = profilePoints[0];
      const p1 = profilePoints[1];
      const hitRadius = 10 / sigZoom;
      const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
      const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
      if (draggingProfileEndpoint !== null) {
        const clampedRow = Math.max(0, Math.min(sigRows - 1, imgRow));
        const clampedCol = Math.max(0, Math.min(sigCols - 1, imgCol));
        const next = [
          draggingProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
          draggingProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
        ];
        setProfileLine(next);
        setProfileData(sampleLineProfile(rawSigDataRef.current, sigCols, sigRows, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth));
        return;
      }
      if (isDraggingProfileLine && sigProfileDragStartRef.current) {
        const drag = sigProfileDragStartRef.current;
        let deltaRow = imgRow - drag.row;
        let deltaCol = imgCol - drag.col;
        const minRow = Math.min(drag.p0.row, drag.p1.row);
        const maxRow = Math.max(drag.p0.row, drag.p1.row);
        const minCol = Math.min(drag.p0.col, drag.p1.col);
        const maxCol = Math.max(drag.p0.col, drag.p1.col);
        deltaRow = Math.max(deltaRow, -minRow);
        deltaRow = Math.min(deltaRow, (sigRows - 1) - maxRow);
        deltaCol = Math.max(deltaCol, -minCol);
        deltaCol = Math.min(deltaCol, (sigCols - 1) - maxCol);
        const next = [
          { row: drag.p0.row + deltaRow, col: drag.p0.col + deltaCol },
          { row: drag.p1.row + deltaRow, col: drag.p1.col + deltaCol },
        ];
        setProfileLine(next);
        setProfileData(sampleLineProfile(rawSigDataRef.current, sigCols, sigRows, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth));
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

  };

  const handleSigMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (draggingProfileEndpoint !== null || isDraggingProfileLine) {
      setDraggingProfileEndpoint(null);
      setIsDraggingProfileLine(false);
      sigProfileDragStartRef.current = null;
      sigClickStartRef.current = null;
      setIsDraggingSig(false);
      setSigDragStart(null);
      setHoveredProfileEndpoint(null);
      setIsHoveringProfileLine(false);
      return;
    }

    // Profile click capture
    if (profileActive && !lockProfile && sigClickStartRef.current) {
      const dx = e.clientX - sigClickStartRef.current.x;
      const dy = e.clientY - sigClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = sigOverlayRef.current;
        if (canvas && rawSigDataRef.current) {
          const rect = canvas.getBoundingClientRect();
          const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const imgCol = (screenX - sigPanX) / sigZoom;
          const imgRow = (screenY - sigPanY) / sigZoom;
          if (imgCol >= 0 && imgCol < sigCols && imgRow >= 0 && imgRow < sigRows) {
            const pt = { row: imgRow, col: imgCol };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              setProfileLine([pt]);
              setProfileData(null);
            } else {
              const p0 = profilePoints[0];
              setProfileLine([p0, pt]);
              setProfileData(sampleLineProfile(rawSigDataRef.current, sigCols, sigRows, p0.row, p0.col, pt.row, pt.col, profileWidth));
            }
          }
        }
      }
    }
    sigClickStartRef.current = null;
    setIsDraggingSig(false);
    setSigDragStart(null);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
  };
  const handleSigMouseLeave = () => {
    setIsDraggingSig(false);
    setSigDragStart(null);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    sigProfileDragStartRef.current = null;
    setCursorInfo(prev => prev?.panel === "sig" ? null : prev);
  };
  const handleSigDoubleClick = () => {
    if (lockView) return;
    setSigZoom(1);
    setSigPanX(0);
    setSigPanY(0);
  };

  // ── Canvas resize handlers ──
  const handleResizeStart = (e: React.MouseEvent) => {
    if (lockView) return;
    e.stopPropagation();
    e.preventDefault();
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: canvasSize });
  };

  React.useEffect(() => {
    if (!isResizing) return;
    let rafId = 0;
    let latestSize = resizeStart ? resizeStart.size : canvasSize;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      latestSize = Math.max(CANVAS_SIZE, Math.min(800, resizeStart.size + delta));
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
      setIsResizing(false);
      setResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, resizeStart]);

  // ── Nav Export Handlers ──
  const handleNavExportFigure = (withColorbar: boolean) => {
    if (lockExport) return;
    setNavExportAnchor(null);
    if (!navCanvasRef.current) return;
    const navData = new Float32Array(navImageBytes.buffer, navImageBytes.byteOffset, navImageBytes.byteLength / 4);
    const lut = COLORMAPS[navColormap] || COLORMAPS.inferno;
    const { min: dMin, max: dMax } = findDataRange(navData);
    const offscreen = renderToOffscreen(navData, navCols, navRows, lut, dMin, dMax);
    if (!offscreen) return;
    const pixelSizeAngstrom = navPixelSize > 0 && navPixelUnit === "\u00C5" ? navPixelSize : navPixelSize > 0 && navPixelUnit === "nm" ? navPixelSize * 10 : 0;
    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title || "Navigation",
      lut,
      vmin: dMin,
      vmax: dMax,
      pixelSize: pixelSizeAngstrom > 0 ? pixelSizeAngstrom : undefined,
      showColorbar: withColorbar,
      showScaleBar: pixelSizeAngstrom > 0,
    });
    canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, "show4d_nav_figure.pdf"));
  };

  const handleNavExportPng = () => {
    if (lockExport) return;
    setNavExportAnchor(null);
    if (!navCanvasRef.current) return;
    navCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4d_nav.png"); }, "image/png");
  };

  // ── Signal Export Handlers ──
  const handleSigExportFigure = (withColorbar: boolean) => {
    if (lockExport) return;
    setSigExportAnchor(null);
    const frameData = rawSigDataRef.current;
    if (!frameData) return;
    let processed: Float32Array;
    if (sigScaleMode === "log") {
      processed = new Float32Array(frameData.length);
      for (let i = 0; i < frameData.length; i++) processed[i] = Math.log1p(Math.max(0, frameData[i]));
    } else if (sigScaleMode === "power") {
      processed = new Float32Array(frameData.length);
      for (let i = 0; i < frameData.length; i++) processed[i] = Math.pow(Math.max(0, frameData[i]), sigPowerExp);
    } else {
      processed = frameData;
    }
    const lut = COLORMAPS[sigColormap] || COLORMAPS.inferno;
    const { min: pMin, max: pMax } = findDataRange(processed);
    const { vmin, vmax } = sliderRange(pMin, pMax, sigVminPct, sigVmaxPct);
    const offscreen = renderToOffscreen(processed, sigCols, sigRows, lut, vmin, vmax);
    if (!offscreen) return;
    const pixelSizeAngstrom = sigPixelSize > 0 && sigPixelUnit === "\u00C5" ? sigPixelSize : sigPixelSize > 0 && sigPixelUnit === "nm" ? sigPixelSize * 10 : 0;
    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title ? `${title} \u2014 Signal` : "Signal",
      lut,
      vmin,
      vmax,
      pixelSize: pixelSizeAngstrom > 0 ? pixelSizeAngstrom : undefined,
      showColorbar: withColorbar,
      showScaleBar: pixelSizeAngstrom > 0,
    });
    canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, "show4d_signal_figure.pdf"));
  };

  const handleSigExportPng = () => {
    if (lockExport) return;
    setSigExportAnchor(null);
    if (!sigCanvasRef.current) return;
    sigCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4d_signal.png"); }, "image/png");
  };

  const handleSigExportGif = () => {
    if (lockExport) return;
    setSigExportAnchor(null);
    setExporting(true);
    setGifExportRequested(true);
  };

  // Download GIF when data arrives from Python
  React.useEffect(() => {
    if (!gifData || gifData.byteLength === 0) return;
    downloadDataView(gifData, "show4d_animation.gif", "image/gif");
    const metaText = (gifMetadataJson || "").trim();
    if (metaText) {
      downloadBlob(new Blob([metaText], { type: "application/json" }), "show4d_animation.json");
    }
    setExporting(false);
  }, [gifData, gifMetadataJson]);

  // ── Keyboard shortcuts (focus-scoped to widget root) ──
  const handleKeyDown = React.useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    if (isTypingTarget(e.target)) return;

    const step = e.shiftKey ? 10 : 1;
    let handled = false;

    switch (e.key) {
        case "ArrowUp":
          if (!lockNavigation) {
            setPosRow(Math.max(0, posRow - step));
            handled = true;
          }
          break;
        case "ArrowDown":
          if (!lockNavigation) {
            setPosRow(Math.min(navRows - 1, posRow + step));
            handled = true;
          }
          break;
        case "ArrowLeft":
          if (!lockNavigation) {
            setPosCol(Math.max(0, posCol - step));
            handled = true;
          }
          break;
        case "ArrowRight":
          if (!lockNavigation) {
            setPosCol(Math.min(navCols - 1, posCol + step));
            handled = true;
          }
          break;
        case "r":
        case "R":
          if (!lockView) {
            setNavZoom(1); setNavPanX(0); setNavPanY(0);
            setSigZoom(1); setSigPanX(0); setSigPanY(0);
            setFftZoom(1); setFftPanX(0); setFftPanY(0);
            handled = true;
          }
          break;
        case "t":
        case "T":
          if (!lockRoi) {
            if (roiMode === "off") {
              setRoiMode(lastRoiModeRef.current);
            } else {
              lastRoiModeRef.current = roiMode;
              setRoiMode("off");
            }
            handled = true;
          }
          break;
        case " ":
          if (!lockPlayback && pathLength > 0) {
            setPathPlaying(!pathPlaying);
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
    isTypingTarget,
    lockNavigation,
    lockPlayback,
    lockRoi,
    lockView,
    navCols,
    navRows,
    pathLength,
    pathPlaying,
    posCol,
    posRow,
    roiMode,
    setPathPlaying, setPosCol, setPosRow, setRoiMode,
  ]);

  // ── Theme-aware select style ──
  const themedSelect = {
    minWidth: 65,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    fontSize: 11,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } },
  };

  // ── Render ──
  return (
    <Box
      ref={rootRef}
      className="show4d-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
      sx={{ p: `${SPACING.LG}px`, bgcolor: themeColors.bg, color: themeColors.text, outline: "none" }}
    >
      {/* Title */}
      <Typography variant="h6" sx={{ ...typo.title, mb: `${SPACING.SM}px` }}>
        {title || "4D Explorer"}
        <InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
          <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>ROI: Region of Interest on navigation image — integrates signal over enclosed area.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Snap: Snap to local intensity maximum within search radius.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum of signal image.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points to draw a line intensity profile.</Typography>
          <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
          <KeyboardShortcuts items={[["↑ / ↓", "Move row"], ["← / →", "Move col"], ["Shift+Arrows", "Move ×10"], ["T", "Toggle ROI on/off"], ["Space", "Play / pause path"], ["R", "Reset zoom"], ["Esc", "Release keyboard focus"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
        </Box>} theme={themeInfo.theme} />
        <ControlCustomizer
          widgetName="Show4D"
          hiddenTools={hiddenTools}
          setHiddenTools={setHiddenTools}
          disabledTools={disabledTools}
          setDisabledTools={setDisabledTools}
          themeColors={themeColors}
        />
      </Typography>

      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* ── LEFT COLUMN: Navigation Panel ── */}
        <Box sx={{ width: navCanvasWidth }}>
          {/* Nav Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typo.label }}>
              Navigation ({Math.round(localPosRow)}, {Math.round(localPosCol)})
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`}>
              {!hideExport && (
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={async () => {
                  if (lockExport || !navCanvasRef.current) return;
                  try {
                    const blob = await new Promise<Blob | null>(resolve => navCanvasRef.current!.toBlob(resolve, "image/png"));
                    if (!blob) return;
                    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                  } catch {
                    navCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4d_nav.png"); }, "image/png");
                  }
                }}>COPY</Button>
              )}
              {!hideExport && (
                <Button
                  size="small"
                  sx={{ ...compactButton, color: themeColors.accent }}
                  onClick={(e) => { if (!lockExport) setNavExportAnchor(e.currentTarget); }}
                  disabled={lockExport || exporting}
                >
                  {exporting ? "..." : "Export"}
                </Button>
              )}
              {!hideExport && (
                <Menu anchorEl={navExportAnchor} open={Boolean(navExportAnchor)} onClose={() => setNavExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                  <MenuItem disabled={lockExport} onClick={() => handleNavExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                  <MenuItem disabled={lockExport} onClick={() => handleNavExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                  <MenuItem disabled={lockExport} onClick={handleNavExportPng} sx={{ fontSize: 12 }}>PNG</MenuItem>
                </Menu>
              )}
              {!hideView && (
                <Button size="small" sx={compactButton} disabled={lockView || (navZoom === 1 && navPanX === 0 && navPanY === 0)} onClick={() => { if (!lockView) { setNavZoom(1); setNavPanX(0); setNavPanY(0); } }}>Reset</Button>
              )}
            </Stack>
          </Stack>

          {/* Nav Canvas */}
          <Box sx={{ ...container.imageBox, width: navCanvasWidth, height: navCanvasHeight }}>
            <canvas ref={navCanvasRef} width={navCols} height={navRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={navOverlayRef} width={navCols} height={navRows}
              onMouseDown={handleNavMouseDown} onMouseMove={handleNavMouseMove}
              onMouseUp={handleNavMouseUp} onMouseLeave={handleNavMouseLeave}
              onWheel={createZoomHandler(setNavZoom, setNavPanX, setNavPanY, navZoom, navPanX, navPanY, navOverlayRef, lockView)}
              onDoubleClick={handleNavDoubleClick}
              style={{
                position: "absolute",
                width: "100%",
                height: "100%",
                cursor: lockView
                  ? "default"
                  : isHoveringRoiResize || isDraggingRoiResize
                    ? "nwse-resize"
                    : snapEnabled && !lockNavigation
                      ? "cell"
                      : "crosshair",
              }}
            />
            <canvas ref={navUiRef} width={navCanvasWidth * DPR} height={navCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
            {cursorInfo && cursorInfo.panel === "nav" && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            {!hideView && (
              <Box onMouseDown={handleResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.2 : 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: lockView ? 0.2 : 1 } }} />
            )}
          </Box>

          {/* Nav Stats Bar */}
          {showStats && !hideStats && navStats && navStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", opacity: lockStats ? 0.6 : 1 }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(navStats[3])}</Box></Typography>
            </Box>
          )}

          {/* Nav Controls */}
          {showControls && (!hideRoi || !hideNavigation || !hideDisplay || !hideHistogram) && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: ROI + Snap */}
                {(!hideRoi || !hideNavigation) && (
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockRoi && lockNavigation ? 0.6 : 1 }}>
                    {!hideRoi && (
                      <>
                        <Typography sx={{ ...typo.label, fontSize: 10 }}>ROI:</Typography>
                        <Select
                          value={roiMode || "off"}
                          onChange={(e) => {
                            if (lockRoi) return;
                            const v = e.target.value;
                            if (v !== "off") lastRoiModeRef.current = v;
                            setRoiMode(v);
                          }}
                          disabled={lockRoi}
                          size="small"
                          sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }}
                          MenuProps={themedMenuProps}
                        >
                          <MenuItem value="off">Off</MenuItem>
                          <MenuItem value="circle">Circle</MenuItem>
                          <MenuItem value="square">Square</MenuItem>
                          <MenuItem value="rect">Rect</MenuItem>
                        </Select>
                      </>
                    )}
                    {!hideRoi && roiMode !== "off" && (
                      <Select value={roiReduce || "mean"} onChange={(e) => { if (!lockRoi) setRoiReduce(String(e.target.value)); }} disabled={lockRoi} size="small" sx={{ ...themedSelect, minWidth: 55, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="mean">Mean</MenuItem>
                        <MenuItem value="max">Max</MenuItem>
                        <MenuItem value="min">Min</MenuItem>
                        <MenuItem value="sum">Sum</MenuItem>
                      </Select>
                    )}
                    {!hideRoi && roiMode !== "off" && (roiMode === "circle" || roiMode === "square") && (
                      <>
                        <Slider
                          value={roiRadius || 5}
                          onChange={(_, v) => { if (!lockRoi) setRoiRadius(v as number); }}
                          disabled={lockRoi}
                          min={1}
                          max={Math.min(navRows, navCols) / 2}
                          size="small"
                          sx={{ width: 80, mx: 1, "& .MuiSlider-thumb": { width: 14, height: 14 } }}
                        />
                        <Typography sx={{ ...typo.value, fontSize: 10, minWidth: 30 }}>
                          {Math.round(roiRadius || 5)}px
                        </Typography>
                      </>
                    )}
                    {!hideNavigation && roiMode === "off" && (
                      <>
                        <Typography sx={{ ...typo.label, fontSize: 10 }}>Snap:</Typography>
                        <Switch checked={snapEnabled} onChange={(_, v) => { if (!lockNavigation) setSnapEnabled(v); }} disabled={lockNavigation} size="small" sx={switchStyles.small} />
                        {snapEnabled && (
                          <>
                            <Slider value={snapRadius} min={1} max={20} step={1} disabled={lockNavigation} onChange={(_, v) => { if (!lockNavigation && typeof v === "number") setSnapRadius(v); }} size="small" sx={{ width: 60, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                            <Typography sx={{ ...typo.value, fontSize: 10 }}>{snapRadius}px</Typography>
                          </>
                        )}
                      </>
                    )}
                  </Box>
                )}
                {/* Row 2: Color + Scale */}
                {!hideDisplay && (
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.6 : 1 }}>
                    <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                    <Select disabled={lockDisplay} value={navColormap} onChange={(e) => { if (!lockDisplay) setNavColormap(String(e.target.value)); }} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                      <MenuItem value="inferno">Inferno</MenuItem>
                      <MenuItem value="viridis">Viridis</MenuItem>
                      <MenuItem value="plasma">Plasma</MenuItem>
                      <MenuItem value="magma">Magma</MenuItem>
                      <MenuItem value="hot">Hot</MenuItem>
                      <MenuItem value="gray">Gray</MenuItem>
                    </Select>
                    <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                    <Select disabled={lockDisplay} value={navScaleMode} onChange={(e) => { if (!lockDisplay) setNavScaleMode(e.target.value as "linear" | "log" | "power"); }} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                      <MenuItem value="power">Pow</MenuItem>
                    </Select>
                  </Box>
                )}
              </Box>
              {!hideHistogram && (
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: lockHistogram ? 0.6 : 1 }}>
                  <Histogram data={navHistogramData} vminPct={navVminPct} vmaxPct={navVmaxPct} onRangeChange={(min, max) => { if (!lockHistogram) { setNavVminPct(min); setNavVmaxPct(max); } }} width={110} height={58} theme={themeInfo.theme} dataMin={navDataMin} dataMax={navDataMax} />
                </Box>
              )}
            </Box>
          )}
        </Box>

        {/* ── RIGHT COLUMN: Signal Panel ── */}
        <Box sx={{ width: sigCanvasWidth }}>
          {/* Signal Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typo.label }}>
              Signal
              {!hideRoi && roiMode !== "off"
                ? <span style={{ color: themeColors.accent, marginLeft: SPACING.SM }}>(ROI {roiReduce || "mean"})</span>
                : <span style={{ color: themeColors.textMuted, marginLeft: SPACING.SM }}>at ({posRow}, {posCol})</span>
              }
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typo.label, color: themeColors.textMuted, fontSize: 10 }}>
                {navRows}×{navCols} | {sigRows}×{sigCols}
              </Typography>
              {!hideFft && (
                <>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>FFT:</Typography>
                  <Switch checked={effectiveShowFft} onChange={(e) => { if (!lockFft) setShowFft(e.target.checked); }} disabled={lockFft} size="small" sx={switchStyles.small} />
                </>
              )}
              {!hideExport && (
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={async () => {
                  if (lockExport || !sigCanvasRef.current) return;
                  try {
                    const blob = await new Promise<Blob | null>(resolve => sigCanvasRef.current!.toBlob(resolve, "image/png"));
                    if (!blob) return;
                    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                  } catch {
                    sigCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4d_signal.png"); }, "image/png");
                  }
                }}>COPY</Button>
              )}
              {!hideExport && (
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => { if (!lockExport) setSigExportAnchor(e.currentTarget); }} disabled={lockExport || exporting}>{exporting ? "Exporting..." : "Export"}</Button>
              )}
              {!hideExport && (
                <Menu anchorEl={sigExportAnchor} open={Boolean(sigExportAnchor)} onClose={() => setSigExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                  <MenuItem disabled={lockExport} onClick={() => handleSigExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                  <MenuItem disabled={lockExport} onClick={() => handleSigExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                  <MenuItem disabled={lockExport} onClick={handleSigExportPng} sx={{ fontSize: 12 }}>PNG (current frame)</MenuItem>
                  {pathLength > 0 && <MenuItem disabled={lockExport} onClick={handleSigExportGif} sx={{ fontSize: 12 }}>GIF (path animation)</MenuItem>}
                </Menu>
              )}
              {!hideView && (
                <Button size="small" sx={compactButton} disabled={lockView || (sigZoom === 1 && sigPanX === 0 && sigPanY === 0)} onClick={() => { if (!lockView) { setSigZoom(1); setSigPanX(0); setSigPanY(0); } }}>Reset</Button>
              )}
            </Stack>
          </Stack>

          {/* Signal Canvas */}
          <Box sx={{ ...container.imageBox, width: sigCanvasWidth, height: sigCanvasHeight }}>
            <canvas ref={sigCanvasRef} width={sigCols} height={sigRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={sigOverlayRef} width={sigCols} height={sigRows}
              onMouseDown={handleSigMouseDown} onMouseMove={handleSigMouseMove}
              onMouseUp={handleSigMouseUp} onMouseLeave={handleSigMouseLeave}
              onWheel={createZoomHandler(setSigZoom, setSigPanX, setSigPanY, sigZoom, sigPanX, sigPanY, sigOverlayRef, lockView)}
              onDoubleClick={handleSigDoubleClick}
              style={{
                position: "absolute",
                width: "100%",
                height: "100%",
                cursor: (profileActive && lockProfile) || (!profileActive && lockView)
                  ? "default"
                  : (draggingProfileEndpoint !== null || isDraggingProfileLine)
                    ? "grabbing"
                    : (profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine))
                      ? "grab"
                      : profileActive
                        ? "crosshair"
                        : isDraggingSig
                          ? "grabbing"
                          : "grab",
              }}
            />
            <canvas ref={sigUiRef} width={sigCanvasWidth * DPR} height={sigCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
            {cursorInfo && cursorInfo.panel === "sig" && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            {!hideView && (
              <Box onMouseDown={handleResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.2 : 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: lockView ? 0.2 : 1 } }} />
            )}
          </Box>

          {/* Signal Stats Bar */}
          {showStats && !hideStats && sigStats && sigStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", opacity: lockStats ? 0.6 : 1 }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(sigStats[3])}</Box></Typography>
            </Box>
          )}

          {/* Profile sparkline */}
          {profileActive && !hideProfile && (
            <Box sx={{ mt: `${SPACING.XS}px`, maxWidth: sigCanvasWidth, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                style={{ width: sigCanvasWidth, height: 76, display: "block", border: `1px solid ${themeColors.border}` }}
              />
            </Box>
          )}

          {/* Signal Controls */}
          {showControls && (!hideProfile || !hideDisplay || !hideHistogram) && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Profile */}
                {!hideProfile && (
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockProfile ? 0.6 : 1 }}>
                    <Typography sx={{ ...typo.label, fontSize: 10 }}>Profile:</Typography>
                    <Switch checked={profileActive} onChange={(e) => {
                      if (lockProfile) return;
                      const on = e.target.checked;
                      setProfileActive(on);
                      if (!on) {
                        setProfileLine([]);
                        setProfileData(null);
                        setHoveredProfileEndpoint(null);
                        setIsHoveringProfileLine(false);
                      }
                    }} disabled={lockProfile} size="small" sx={switchStyles.small} />
                    {profileActive && profileWidth > 1 && (
                      <>
                        <Typography sx={{ ...typo.value, fontSize: 10 }}>w={profileWidth}</Typography>
                        <Slider value={profileWidth} min={1} max={20} step={1} disabled={lockProfile} onChange={(_, v) => { if (!lockProfile && typeof v === "number") setProfileWidth(v); }} size="small" sx={{ width: 50, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                      </>
                    )}
                  </Box>
                )}
                {/* Row 2: Color + Scale */}
                {!hideDisplay && (
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.6 : 1 }}>
                    <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                    <Select disabled={lockDisplay} value={sigColormap} onChange={(e) => { if (!lockDisplay) setSigColormap(String(e.target.value)); }} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                      <MenuItem value="inferno">Inferno</MenuItem>
                      <MenuItem value="viridis">Viridis</MenuItem>
                      <MenuItem value="plasma">Plasma</MenuItem>
                      <MenuItem value="magma">Magma</MenuItem>
                      <MenuItem value="hot">Hot</MenuItem>
                      <MenuItem value="gray">Gray</MenuItem>
                    </Select>
                    <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                    <Select disabled={lockDisplay} value={sigScaleMode} onChange={(e) => { if (!lockDisplay) setSigScaleMode(e.target.value as "linear" | "log" | "power"); }} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                      <MenuItem value="power">Pow</MenuItem>
                    </Select>
                  </Box>
                )}
              </Box>
              {!hideHistogram && (
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: lockHistogram ? 0.6 : 1 }}>
                  <Histogram data={sigHistogramData} vminPct={sigVminPct} vmaxPct={sigVmaxPct} onRangeChange={(min, max) => { if (!lockHistogram) { setSigVminPct(min); setSigVmaxPct(max); } }} width={110} height={58} theme={themeInfo.theme} dataMin={sigDataMin} dataMax={sigDataMax} />
                </Box>
              )}
            </Box>
          )}
        </Box>

        {/* ── THIRD COLUMN: FFT Panel ── */}
        {effectiveShowFft && (
          <Box sx={{ width: sigCanvasWidth }}>
            {/* FFT Header */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              <Typography variant="caption" sx={{ ...typo.label, color: fftCropDims ? accentGreen : themeColors.textMuted }}>
                {fftCropDims ? `ROI FFT (${fftCropDims.cropWidth}\u00D7${fftCropDims.cropHeight})` : "FFT (Signal)"}
              </Typography>
              <Stack direction="row" spacing={`${SPACING.SM}px`}>
                {!hideExport && (
                  <Button size="small" sx={compactButton} disabled={lockExport || lockFft} onClick={() => {
                    if (lockExport || lockFft || !fftCanvasRef.current) return;
                    fftCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4d_fft.png"); }, "image/png");
                  }}>Export</Button>
                )}
                {!hideView && (
                  <Button size="small" sx={compactButton} disabled={lockView || lockFft || (fftZoom === 1 && fftPanX === 0 && fftPanY === 0)} onClick={() => { if (!lockView && !lockFft) { setFftZoom(1); setFftPanX(0); setFftPanY(0); } }}>Reset</Button>
                )}
              </Stack>
            </Stack>

            {/* FFT Canvas */}
            <Box sx={{ ...container.imageBox, width: sigCanvasWidth, height: sigCanvasHeight }}>
              <canvas ref={fftCanvasRef} width={sigCols} height={sigRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
              <canvas
                ref={fftOverlayRef} width={sigCols} height={sigRows}
                onMouseDown={handleFftMouseDown} onMouseMove={handleFftMouseMove}
                onMouseUp={handleFftMouseUp} onMouseLeave={handleFftMouseLeave}
                onWheel={createZoomHandler(setFftZoom, setFftPanX, setFftPanY, fftZoom, fftPanX, fftPanY, fftOverlayRef, lockView || lockFft)}
                onDoubleClick={handleFftDoubleClick}
                style={{ position: "absolute", width: "100%", height: "100%", cursor: lockView || lockFft ? "default" : (isDraggingFft ? "grabbing" : "grab") }}
              />
              <canvas ref={fftUiRef} width={sigCanvasWidth * DPR} height={sigCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
              {!hideView && (
                <Box onMouseDown={handleResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.2 : 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: lockView ? 0.2 : 1 } }} />
              )}
            </Box>

            {/* FFT Stats Bar */}
            {showStats && !hideStats && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", opacity: lockStats ? 0.6 : 1 }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats.mean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats.min)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats.max)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats.std)}</Box></Typography>
              </Box>
            )}

            {/* D-spacing readout */}
            {fftClickInfo && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                  Dist <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box>
                </Typography>
                {fftClickInfo.spatialFreq != null && (
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                    Freq <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.spatialFreq.toFixed(4)} {"\u00C5\u207B\u00B9"}</Box>
                  </Typography>
                )}
                {fftClickInfo.dSpacing != null && (
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontWeight: "bold" }}>
                    d = <Box component="span" sx={{ color: themeColors.accent }}>
                      {fftClickInfo.dSpacing >= 10
                        ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm`
                        : `${fftClickInfo.dSpacing.toFixed(2)} \u00C5`}
                    </Box>
                  </Typography>
                )}
              </Box>
            )}

            {/* FFT Controls */}
            {showControls && (!hideDisplay || !hideHistogram) && (
              <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
                {!hideDisplay && (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: (lockDisplay || lockFft) ? 0.6 : 1 }}>
                      <Typography sx={{ ...typo.label, fontSize: 10 }}>Auto:</Typography>
                      <Switch checked={fftAuto} onChange={(e) => { if (!lockDisplay && !lockFft) setFftAuto(e.target.checked); }} disabled={lockDisplay || lockFft} size="small" sx={switchStyles.small} />
                      <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                      <Select disabled={lockDisplay || lockFft} value={fftColormap} onChange={(e) => { if (!lockDisplay && !lockFft) setFftColormap(String(e.target.value)); }} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="inferno">Inferno</MenuItem>
                        <MenuItem value="viridis">Viridis</MenuItem>
                        <MenuItem value="plasma">Plasma</MenuItem>
                        <MenuItem value="magma">Magma</MenuItem>
                        <MenuItem value="hot">Hot</MenuItem>
                        <MenuItem value="gray">Gray</MenuItem>
                      </Select>
                      <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                      <Select disabled={lockDisplay || lockFft} value={fftLogScale ? "log" : "linear"} onChange={(e) => { if (!lockDisplay && !lockFft) setFftLogScale(e.target.value === "log"); }} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="linear">Lin</MenuItem>
                        <MenuItem value="log">Log</MenuItem>
                      </Select>
                    </Box>
                  </Box>
                )}
                {!hideHistogram && (
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: (lockHistogram || lockFft) ? 0.6 : 1 }}>
                    <Histogram data={fftHistogramData} vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { if (!lockHistogram && !lockFft) { setFftVminPct(min); setFftVmaxPct(max); } }} width={110} height={58} theme={themeInfo.theme} dataMin={fftDataRange.min} dataMax={fftDataRange.max} />
                  </Box>
                )}
              </Box>
            )}
          </Box>
        )}
      </Stack>

      {/* Playback bar (only when path is set) */}
      {showControls && !hidePlayback && pathLength > 0 && (
        <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
          <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
            <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) setPathPlaying(!pathPlaying); }} sx={{ color: themeColors.accent, p: 0.25 }}>
              {pathPlaying ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
            </IconButton>
            <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) { setPathPlaying(false); setPathIndex(0); } }} sx={{ color: themeColors.textMuted, p: 0.25 }}>
              <StopIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Stack>
          <Slider disabled={lockPlayback} value={pathIndex} onChange={(_, v) => { if (!lockPlayback) { setPathPlaying(false); setPathIndex(v as number); } }} min={0} max={Math.max(0, pathLength - 1)} size="small" sx={{ flex: 1, minWidth: 60, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
          <Typography sx={{ ...typo.value, minWidth: 50, textAlign: "right", flexShrink: 0 }}>{pathIndex + 1}/{pathLength}</Typography>
          <Typography sx={{ ...typo.label, fontSize: 10 }}>Loop:</Typography>
          <Switch checked={pathLoop} onChange={() => { if (!lockPlayback) { model.set("path_loop", !pathLoop); model.save_changes(); } }} disabled={lockPlayback} size="small" sx={switchStyles.small} />
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(Show4D);
