/**
 * ShowComplex - Interactive complex-valued image viewer.
 *
 * Features:
 * - 5 display modes: amplitude, phase, HSV, real, imaginary
 * - Phase colorwheel inset for phase/HSV modes
 * - Scroll to zoom, double-click to reset
 * - WebGPU-accelerated FFT
 * - Scale bar, histogram, statistics
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar, exportFigure, canvasToPDF } from "../scalebar";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, applyLogScale, percentileClip, sliderRange } from "../stats";
import { getWebGPUFFT, WebGPUFFT, fft2d, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2 } from "../webgpu-fft";
import { COLORMAPS, COLORMAP_NAMES, applyColormap, renderToOffscreen } from "../colormaps";
import { ControlCustomizer } from "../control-customizer";
import { computeToolVisibility } from "../tool-parity";
import "./showcomplex.css";

// ============================================================================
// Helper components (per-widget, not shared)
// ============================================================================

function InfoTooltip({ text, theme = "dark" }: { text: React.ReactNode; theme?: "light" | "dark" }) {
  const isDark = theme === "dark";
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  return (
    <Tooltip
      title={content}
      arrow placement="bottom"
      componentsProps={{
        tooltip: { sx: { bgcolor: isDark ? "#333" : "#fff", color: isDark ? "#ddd" : "#333", border: `1px solid ${isDark ? "#555" : "#ccc"}`, maxWidth: 280, p: 1 } },
        arrow: { sx: { color: isDark ? "#333" : "#fff", "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` } } },
      }}
    >
      <Typography component="span" sx={{ fontSize: 12, color: isDark ? "#888" : "#666", cursor: "help", ml: 0.5, "&:hover": { color: isDark ? "#aaa" : "#444" } }}>ⓘ</Typography>
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

// ============================================================================
// Style constants
// ============================================================================

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};
const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
};
const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};
const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

const DEFAULT_CANVAS_SIZE = 500;

// ROI crop helper (single-mode, same as Show4D)
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
        const dR = (y0 + dy) - centerRow, dC = (x0 + dx) - centerCol;
        cropped[dy * cropW + dx] = dR * dR + dC * dC <= rSq ? data[(y0 + dy) * imgW + (x0 + dx)] : 0;
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

type DisplayMode = "amplitude" | "phase" | "hsv" | "real" | "imag";

// ============================================================================
// HSV rendering
// ============================================================================

function renderHSV(
  real: Float32Array, imag: Float32Array,
  rgba: Uint8ClampedArray,
  ampMin: number, ampMax: number,
): void {
  const ampRange = ampMax > ampMin ? ampMax - ampMin : 1;
  for (let i = 0; i < real.length; i++) {
    const phase = Math.atan2(imag[i], real[i]);
    const amp = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
    const h = (phase + Math.PI) / (2 * Math.PI); // [0, 1]
    const v = Math.max(0, Math.min(1, (amp - ampMin) / ampRange));
    // HSV→RGB with S=1
    const hi = Math.floor(h * 6) % 6;
    const f = h * 6 - Math.floor(h * 6);
    const q = v * (1 - f);
    const t = v * f;
    let r: number, g: number, b: number;
    switch (hi) {
      case 0: r = v; g = t; b = 0; break;
      case 1: r = q; g = v; b = 0; break;
      case 2: r = 0; g = v; b = t; break;
      case 3: r = 0; g = q; b = v; break;
      case 4: r = t; g = 0; b = v; break;
      default: r = v; g = 0; b = q; break;
    }
    const j = i * 4;
    rgba[j] = r * 255;
    rgba[j + 1] = g * 255;
    rgba[j + 2] = b * 255;
    rgba[j + 3] = 255;
  }
}

// ============================================================================
// Phase colorwheel drawing
// ============================================================================

function drawPhaseColorwheel(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  radius: number,
): void {
  // Draw filled circle with hue sweep
  for (let angle = 0; angle < 360; angle += 1) {
    const rad = (angle * Math.PI) / 180;
    const rad2 = ((angle + 2) * Math.PI) / 180;
    // phase = angle mapped to [-pi, pi] → hue
    const h = angle / 360;
    const hi = Math.floor(h * 6) % 6;
    const f = h * 6 - Math.floor(h * 6);
    const q = 1 - f;
    let r: number, g: number, b: number;
    switch (hi) {
      case 0: r = 1; g = f; b = 0; break;
      case 1: r = q; g = 1; b = 0; break;
      case 2: r = 0; g = 1; b = f; break;
      case 3: r = 0; g = q; b = 1; break;
      case 4: r = f; g = 0; b = 1; break;
      default: r = 1; g = 0; b = q; break;
    }
    ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, rad, rad2);
    ctx.closePath();
    ctx.fill();
  }

  // White center gradient for "brightness = amplitude"
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
  grad.addColorStop(0, "rgba(255,255,255,0.8)");
  grad.addColorStop(0.5, "rgba(255,255,255,0.2)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();

  // Border
  ctx.strokeStyle = "rgba(255,255,255,0.6)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  // Labels
  ctx.fillStyle = "white";
  ctx.shadowColor = "rgba(0,0,0,0.7)";
  ctx.shadowBlur = 2;
  ctx.font = "10px -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("0", cx + radius + 10, cy);
  ctx.fillText("π", cx - radius - 8, cy);
  ctx.fillText("π/2", cx, cy - radius - 8);
  ctx.fillText("-π/2", cx, cy + radius + 8);
  ctx.shadowBlur = 0;
}

// ============================================================================
// Histogram component
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

function Histogram({ data, vminPct, vmaxPct, onRangeChange, width = 110, height = 40, theme = "dark", dataMin = 0, dataMax = 1 }: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);
  const isDark = theme === "dark";
  const colors = isDark ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" } : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };

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
      for (let j = 0; j < binRatio; j++) sum += bins[i * binRatio + j] || 0;
      reducedBins.push(sum / binRatio);
    }
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      ctx.fillStyle = (i >= vminBin && i <= vmaxBin) ? colors.barActive : colors.barInactive;
      ctx.fillRect(i * barWidth + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas ref={canvasRef} style={{ width, height, border: `1px solid ${colors.border}` }} />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => { const [lo, hi] = v as number[]; onRangeChange(Math.min(lo, hi - 1), Math.max(hi, lo + 1)); }}
        min={0} max={100} size="small" valueLabelDisplay="auto"
        valueLabelFormat={(pct) => { const val = dataMin + (pct / 100) * (dataMax - dataMin); return val >= 1000 ? val.toExponential(1) : val.toFixed(1); }}
        sx={{ width, py: 0, "& .MuiSlider-thumb": { width: 8, height: 8 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vminPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vmaxPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography></Box>
    </Box>
  );
}

// ============================================================================
// Main Component
// ============================================================================

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

function ShowComplex2D() {
  // Theme
  const { themeInfo, colors: tc } = useTheme();
  const themeColors = tc;

  const themedSelect = {
    fontSize: 10,
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

  // Model state
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [realBytes] = useModelState<DataView>("real_bytes");
  const [imagBytes] = useModelState<DataView>("imag_bytes");
  const [title] = useModelState<string>("title");
  const [displayMode, setDisplayMode] = useModelState<string>("display_mode");
  const [cmap, setCmap] = useModelState<string>("cmap");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [percentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh] = useModelState<number>("percentile_high");

  // Scale bar
  const [pixelSize] = useModelState<number>("pixel_size");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");

  // UI
  const [showStats] = useModelState<boolean>("show_stats");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [showControls] = useModelState<boolean>("show_controls");
  const [imageWidthPx] = useModelState<number>("image_width_px");

  // Stats
  const [statsMean] = useModelState<number>("stats_mean");
  const [statsMin] = useModelState<number>("stats_min");
  const [statsMax] = useModelState<number>("stats_max");
  const [statsStd] = useModelState<number>("stats_std");

  // Tool visibility
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("ShowComplex2D", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );

  const hideDisplay = toolVisibility.isHidden("display");
  const hideHistogram = toolVisibility.isHidden("histogram");
  const hideFft = toolVisibility.isHidden("fft");
  const hideStats = toolVisibility.isHidden("stats");
  const hideExport = toolVisibility.isHidden("export");
  const hideView = toolVisibility.isHidden("view");

  const lockDisplay = toolVisibility.isLocked("display");
  const lockHistogram = toolVisibility.isLocked("histogram");
  const lockFft = toolVisibility.isLocked("fft");
  const lockStats = toolVisibility.isLocked("stats");
  const lockExport = toolVisibility.isLocked("export");
  const lockView = toolVisibility.isLocked("view");

  const effectiveShowFft = showFft && !hideFft;

  // Canvas refs
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const uiCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // FFT refs
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftContainerRef = React.useRef<HTMLDivElement>(null);

  // Zoom/pan
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [isDragging, setIsDragging] = React.useState(false);
  const dragRef = React.useRef<{ startX: number; startY: number; startPanX: number; startPanY: number; wasDrag: boolean }>({
    startX: 0, startY: 0, startPanX: 0, startPanY: 0, wasDrag: false,
  });

  // Canvas sizing
  const [canvasW, setCanvasW] = React.useState(DEFAULT_CANVAS_SIZE);
  const [canvasH, setCanvasH] = React.useState(DEFAULT_CANVAS_SIZE);
  const [isResizing, setIsResizing] = React.useState(false);
  const resizeRef = React.useRef<{ startX: number; startY: number; startW: number; startH: number }>({ startX: 0, startY: 0, startW: 0, startH: 0 });

  // Histogram state
  const [histData, setHistData] = React.useState<Float32Array | null>(null);
  const [histRange, setHistRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [vminPct, setVminPct] = React.useState(0);
  const [vmaxPct, setVmaxPct] = React.useState(100);

  // FFT state
  const [gpuFFT, setGpuFFT] = React.useState<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  const [fftZoom, setFftZoom] = React.useState(3);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const fftMagRef = React.useRef<Float32Array | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);
  const [fftHistData, setFftHistData] = React.useState<Float32Array | null>(null);
  const [fftHistRange, setFftHistRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftLogScale, setFftLogScale] = React.useState(true);
  const [fftAuto, setFftAuto] = React.useState(true);
  const [fftColormap] = React.useState("inferno");

  // FFT d-spacing measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);

  // FFT drag state
  const [isFftDragging, setIsFftDragging] = React.useState(false);
  const [fftPanStart, setFftPanStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);

  // ROI state (single-mode, same pattern as Show4D)
  const [roiMode, setRoiMode] = useModelState<string>("roi_mode");
  const [roiCenterRow, setRoiCenterRow] = useModelState<number>("roi_center_row");
  const [roiCenterCol, setRoiCenterCol] = useModelState<number>("roi_center_col");
  const [, setRoiCenter] = useModelState<number[]>("roi_center");
  const [roiRadius, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");

  const hideRoi = toolVisibility.isHidden("roi");
  const lockRoi = toolVisibility.isLocked("roi");

  // ROI drag state
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [isDraggingROIResize, setIsDraggingROIResize] = React.useState<string | null>(null);
  const roiDragOffsetRef = React.useRef<{ dRow: number; dCol: number }>({ dRow: 0, dCol: 0 });
  const resizeAspectRef = React.useRef<number | null>(null);
  const lastRoiModeRef = React.useRef<string>("circle");

  // ROI FFT state
  const accentGreen = themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a";
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);
  const roiFftActive = effectiveShowFft && roiMode !== "off" && !hideRoi;

  // Hover cursor
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: string } | null>(null);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);

  // Data refs
  const realDataRef = React.useRef<Float32Array | null>(null);
  const imagDataRef = React.useRef<Float32Array | null>(null);
  const displayDataRef = React.useRef<Float32Array | null>(null);

  // Cached colormapped offscreen canvas (avoids recomputing colormap on zoom/pan)
  const offscreenCacheRef = React.useRef<HTMLCanvasElement | null>(null);

  // ============================================================================
  // Init GPU FFT
  // ============================================================================
  React.useEffect(() => {
    let cancelled = false;
    getWebGPUFFT().then((fft) => {
      if (!cancelled && fft) { setGpuFFT(fft); setGpuReady(true); }
    }).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // ============================================================================
  // Extract real/imag data
  // ============================================================================
  React.useEffect(() => {
    const r = extractFloat32(realBytes);
    const im = extractFloat32(imagBytes);
    if (!r || !im || r.length === 0) return;
    realDataRef.current = r;
    imagDataRef.current = im;
  }, [realBytes, imagBytes]);

  // ============================================================================
  // Compute display data + histogram based on display mode
  // ============================================================================
  React.useEffect(() => {
    const r = realDataRef.current;
    const im = imagDataRef.current;
    if (!r || !im) return;

    let data: Float32Array;
    const mode = displayMode as DisplayMode;

    if (mode === "amplitude" || mode === "hsv") {
      data = new Float32Array(r.length);
      for (let i = 0; i < r.length; i++) {
        data[i] = Math.sqrt(r[i] * r[i] + im[i] * im[i]);
      }
    } else if (mode === "phase") {
      data = new Float32Array(r.length);
      for (let i = 0; i < r.length; i++) {
        data[i] = Math.atan2(im[i], r[i]);
      }
    } else if (mode === "real") {
      data = r;
    } else {
      data = im;
    }

    if (logScale && mode !== "phase") {
      data = applyLogScale(data);
    }

    displayDataRef.current = data;
    setHistData(data);
    setHistRange(findDataRange(data));
  }, [realBytes, imagBytes, displayMode, logScale]);

  // ============================================================================
  // Canvas sizing
  // ============================================================================
  React.useEffect(() => {
    if (!width || !height) return;
    const targetW = imageWidthPx > 0 ? imageWidthPx : DEFAULT_CANVAS_SIZE;
    const scale = targetW / width;
    setCanvasW(Math.round(width * scale));
    setCanvasH(Math.round(height * scale));
  }, [width, height, imageWidthPx]);

  // ============================================================================
  // Build colormapped offscreen canvas (expensive: HSV render, colormap LUT, percentile clip)
  // Excludes zoom/pan so dragging only triggers the cheap redraw below.
  // ============================================================================
  React.useEffect(() => {
    if (!width || !height) return;

    const r = realDataRef.current;
    const im = imagDataRef.current;
    const dispData = displayDataRef.current;
    if (!r || !im) return;

    const mode = displayMode as DisplayMode;

    const offscreen = document.createElement("canvas");
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;
    const imgData = offCtx.createImageData(width, height);

    if (mode === "hsv") {
      const range = histRange;
      renderHSV(r, im, imgData.data, range.min, range.max);
    } else {
      if (!dispData) return;
      const lut = COLORMAPS[mode === "phase" ? "hsv" : cmap] || COLORMAPS.inferno;
      let vmin: number, vmax: number;
      if (autoContrast && mode !== "phase") {
        const pc = percentileClip(dispData, percentileLow, percentileHigh);
        vmin = pc.vmin;
        vmax = pc.vmax;
      } else if (mode === "phase") {
        vmin = -Math.PI;
        vmax = Math.PI;
      } else {
        ({ vmin, vmax } = sliderRange(histRange.min, histRange.max, vminPct, vmaxPct));
      }
      applyColormap(dispData, imgData.data, lut, vmin, vmax);
    }

    offCtx.putImageData(imgData, 0, 0);
    offscreenCacheRef.current = offscreen;
  }, [realBytes, imagBytes, displayMode, cmap, logScale, autoContrast, percentileLow, percentileHigh,
      vminPct, vmaxPct, width, height, histRange]);

  // ============================================================================
  // Redraw with zoom/pan (cheap: just drawImage from cached offscreen)
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  // ============================================================================
  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const offscreen = offscreenCacheRef.current;
    if (!canvas || !offscreen || !width || !height) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.imageSmoothingEnabled = zoom < 4;
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (zoom !== 1 || panX !== 0 || panY !== 0) {
      ctx.save();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
    }
  }, [realBytes, imagBytes, displayMode, cmap, logScale, autoContrast, percentileLow, percentileHigh,
      vminPct, vmaxPct, zoom, panX, panY, canvasW, canvasH, width, height, histRange]);

  // ============================================================================
  // UI overlay canvas (scale bar, colorwheel, colorbar)
  // ============================================================================
  React.useEffect(() => {
    const canvas = uiCanvasRef.current;
    if (!canvas || !width || !height) return;
    canvas.width = Math.round(canvasW * DPR);
    canvas.height = Math.round(canvasH * DPR);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(DPR, DPR);

    const mode = displayMode as DisplayMode;

    // Scale bar
    if (scaleBarVisible && pixelSize > 0) {
      drawScaleBarHiDPI(canvas, DPR, zoom, pixelSize, "Å", width);
    }

    // Zoom indicator (when no scale bar)
    if (!scaleBarVisible || pixelSize <= 0) {
      ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
      ctx.shadowBlur = 2;
      ctx.fillStyle = "white";
      ctx.font = "16px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${zoom.toFixed(1)}×`, 12, canvasH - 12);
      ctx.shadowBlur = 0;
    }

    // Phase colorwheel for phase/HSV modes
    if (mode === "phase" || mode === "hsv") {
      const cwRadius = 25;
      const cwX = 12 + cwRadius;
      const cwY = 12 + cwRadius;
      drawPhaseColorwheel(ctx, cwX, cwY, cwRadius);
    }

    // Colorbar (for non-HSV modes)
    if (mode !== "hsv" && displayDataRef.current) {
      const dispData = displayDataRef.current;
      const lut = COLORMAPS[mode === "phase" ? "hsv" : cmap] || COLORMAPS.inferno;
      let vmin: number, vmax: number;
      if (autoContrast && mode !== "phase") {
        const pc = percentileClip(dispData, percentileLow, percentileHigh);
        vmin = pc.vmin;
        vmax = pc.vmax;
      } else if (mode === "phase") {
        vmin = -Math.PI;
        vmax = Math.PI;
      } else {
        ({ vmin, vmax } = sliderRange(histRange.min, histRange.max, vminPct, vmaxPct));
      }
      drawColorbar(ctx, canvasW, canvasH, lut, vmin, vmax, logScale && mode !== "phase");
    }

    // ROI overlay
    if (roiMode && roiMode !== "off" && !hideRoi) {
      const roiColor = themeInfo.theme === "dark" ? "rgba(0, 255, 0, 0.7)" : "rgba(26, 122, 26, 0.8)";
      const scX = (canvasW / width) * zoom;
      const scY = (canvasH / height) * zoom;
      const offX = (canvasW - canvasW * zoom) / 2 + panX;
      const offY = (canvasH - canvasH * zoom) / 2 + panY;
      const cx = offX + roiCenterCol * scX;
      const cy = offY + roiCenterRow * scY;
      ctx.save();
      ctx.strokeStyle = roiColor;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      if (roiMode === "circle") {
        const r = roiRadius * scX;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
        // Resize handle
        ctx.setLineDash([]);
        ctx.fillStyle = roiColor;
        ctx.beginPath();
        ctx.arc(cx + r, cy, 4, 0, Math.PI * 2);
        ctx.fill();
      } else if (roiMode === "square") {
        const r = roiRadius * scX;
        ctx.strokeRect(cx - r, cy - r, r * 2, r * 2);
        ctx.setLineDash([]);
        ctx.fillStyle = roiColor;
        ctx.beginPath();
        ctx.arc(cx + r, cy + r, 4, 0, Math.PI * 2);
        ctx.fill();
      } else if (roiMode === "rect") {
        const hw = (roiWidth / 2) * scX;
        const hh = (roiHeight / 2) * scY;
        ctx.strokeRect(cx - hw, cy - hh, hw * 2, hh * 2);
        ctx.setLineDash([]);
        ctx.fillStyle = roiColor;
        ctx.beginPath();
        ctx.arc(cx + hw, cy + hh, 4, 0, Math.PI * 2);
        ctx.fill();
      }
      // Center crosshair
      ctx.setLineDash([]);
      ctx.strokeStyle = roiColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx - 5, cy); ctx.lineTo(cx + 5, cy);
      ctx.moveTo(cx, cy - 5); ctx.lineTo(cx, cy + 5);
      ctx.stroke();
      ctx.restore();
    }

    ctx.restore();
  }, [displayMode, cmap, zoom, canvasW, canvasH, width, height, pixelSize, scaleBarVisible,
      logScale, autoContrast, vminPct, vmaxPct, histRange, percentileLow, percentileHigh,
      roiMode, roiCenterRow, roiCenterCol, roiRadius, roiWidth, roiHeight, panX, panY, themeInfo.theme]);

  // ============================================================================
  // FFT computation (async) — crops to ROI when roiFftActive
  // ============================================================================
  React.useEffect(() => {
    if (!effectiveShowFft || !displayDataRef.current || !width || !height) return;
    let cancelled = false;
    const data = displayDataRef.current;

    let fftW: number, fftH: number;
    let inputData: Float32Array;
    let origCropW = 0, origCropH = 0;

    if (roiFftActive) {
      const crop = cropSingleROI(data, width, height, roiMode, roiCenterRow, roiCenterCol, roiRadius, roiWidth, roiHeight);
      if (crop) {
        origCropW = crop.cropW;
        origCropH = crop.cropH;
        fftW = nextPow2(crop.cropW);
        fftH = nextPow2(crop.cropH);
        const padded = new Float32Array(fftW * fftH);
        for (let y = 0; y < crop.cropH; y++)
          for (let x = 0; x < crop.cropW; x++)
            padded[y * fftW + x] = crop.cropped[y * crop.cropW + x];
        inputData = padded;
      } else {
        fftW = nextPow2(width);
        fftH = nextPow2(height);
        inputData = new Float32Array(fftW * fftH);
        for (let r = 0; r < height; r++)
          for (let c = 0; c < width; c++)
            inputData[r * fftW + c] = data[r * width + c];
      }
    } else {
      fftW = nextPow2(width);
      fftH = nextPow2(height);
      inputData = new Float32Array(fftW * fftH);
      for (let r = 0; r < height; r++)
        for (let c = 0; c < width; c++)
          inputData[r * fftW + c] = data[r * width + c];
    }

    const computeFFT = async () => {
      const padI = new Float32Array(fftW * fftH);
      let fReal: Float32Array, fImag: Float32Array;
      if (gpuFFT) {
        const result = await gpuFFT.fft2D(inputData, padI, fftW, fftH, false);
        if (cancelled) return;
        fReal = result.real;
        fImag = result.imag;
      } else {
        fft2d(inputData, padI, fftW, fftH, false);
        if (cancelled) return;
        fReal = inputData;
        fImag = padI;
      }
      fftshift(fReal, fftW, fftH);
      fftshift(fImag, fftW, fftH);
      const mag = computeMagnitude(fReal, fImag);
      if (cancelled) return;
      fftMagRef.current = mag;
      setFftCropDims(origCropW > 0 ? { cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH } : null);
      setFftMagVersion((v) => v + 1);
    };
    computeFFT();
    return () => { cancelled = true; };
  }, [effectiveShowFft, realBytes, imagBytes, displayMode, logScale, width, height, gpuReady,
      roiFftActive, roiMode, roiCenterRow, roiCenterCol, roiRadius, roiWidth, roiHeight]);

  // FFT rendering (cheap, sync) — uses fftCropDims for ROI FFT
  React.useEffect(() => {
    const mag = fftMagRef.current;
    if (!effectiveShowFft || !mag) return;
    const pw = fftCropDims?.fftWidth ?? nextPow2(width);
    const ph = fftCropDims?.fftHeight ?? nextPow2(height);

    let processedMag: Float32Array;
    let vmin: number, vmax: number;

    if (fftAuto) {
      const enhanced = autoEnhanceFFT(mag, pw, ph);
      processedMag = fftLogScale ? applyLogScale(mag) : mag;
      const enh2 = fftLogScale ? autoEnhanceFFT(processedMag, pw, ph) : enhanced;
      vmin = enh2.min;
      vmax = enh2.max;
    } else {
      processedMag = fftLogScale ? applyLogScale(mag) : mag;
      const range = findDataRange(processedMag);
      ({ vmin, vmax } = sliderRange(range.min, range.max, fftVminPct, fftVmaxPct));
    }

    setFftHistData(processedMag);
    setFftHistRange(findDataRange(processedMag));

    const fftCanvas = fftCanvasRef.current;
    if (!fftCanvas) return;
    const ctx = fftCanvas.getContext("2d");
    if (!ctx) return;

    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(processedMag, pw, ph, lut, vmin, vmax);
    if (!offscreen) return;

    ctx.imageSmoothingEnabled = fftZoom < 4;
    ctx.clearRect(0, 0, canvasW, canvasH);
    const scaleX = canvasW / pw;
    const scaleY = canvasH / ph;
    ctx.save();
    ctx.translate(canvasW / 2 + fftPanX, canvasH / 2 + fftPanY);
    ctx.scale(fftZoom * scaleX, fftZoom * scaleY);
    ctx.translate(-pw / 2, -ph / 2);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [effectiveShowFft, fftMagVersion, fftLogScale, fftAuto, fftVminPct, fftVmaxPct, fftColormap,
      fftZoom, fftPanX, fftPanY, width, height, canvasW, canvasH, fftCropDims]);

  // ============================================================================
  // Coordinate helper: screen → image
  // ============================================================================
  const screenToImage = React.useCallback((e: React.MouseEvent): { row: number; col: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas || !width || !height) return null;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvasW / 2, cy = canvasH / 2;
    const imgX = (mx - cx - panX) / zoom + cx;
    const imgY = (my - cy - panY) / zoom + cy;
    const s = canvasW / width;
    const col = imgX / s, row = imgY / s;
    if (col >= 0 && col < width && row >= 0 && row < height) return { row, col };
    return null;
  }, [canvasW, canvasH, width, height, zoom, panX, panY]);

  // ============================================================================
  // ROI hit detection helper
  // ============================================================================
  const roiHitTest = React.useCallback((imgRow: number, imgCol: number): "body" | "resize" | null => {
    if (!roiMode || roiMode === "off" || lockRoi || hideRoi) return null;
    if (roiMode === "circle") {
      const dR = imgRow - roiCenterRow, dC = imgCol - roiCenterCol;
      const dist = Math.sqrt(dR * dR + dC * dC);
      if (Math.abs(dist - roiRadius) < Math.max(3, roiRadius * 0.15)) return "resize";
      if (dist < roiRadius) return "body";
    } else if (roiMode === "square") {
      const dR = Math.abs(imgRow - roiCenterRow), dC = Math.abs(imgCol - roiCenterCol);
      const tol = Math.max(2, roiRadius * 0.12);
      if (dR <= roiRadius + tol && dC <= roiRadius + tol) {
        if (Math.abs(dR - roiRadius) < tol || Math.abs(dC - roiRadius) < tol) return "resize";
        return "body";
      }
    } else if (roiMode === "rect") {
      const hw = roiWidth / 2, hh = roiHeight / 2;
      const dR = Math.abs(imgRow - roiCenterRow), dC = Math.abs(imgCol - roiCenterCol);
      const tol = Math.max(2, Math.min(hw, hh) * 0.12);
      if (dR <= hh + tol && dC <= hw + tol) {
        if (Math.abs(dR - hh) < tol || Math.abs(dC - hw) < tol) return "resize";
        return "body";
      }
    }
    return null;
  }, [roiMode, roiCenterRow, roiCenterCol, roiRadius, roiWidth, roiHeight, lockRoi, hideRoi]);

  // ============================================================================
  // Mouse handlers (with ROI drag/resize support)
  // ============================================================================
  const handleMouseDown = React.useCallback((e: React.MouseEvent) => {
    if (lockView && lockRoi) return;
    // Check for ROI hit first
    if (roiMode && roiMode !== "off" && !lockRoi) {
      const pos = screenToImage(e);
      if (pos) {
        const hit = roiHitTest(pos.row, pos.col);
        if (hit === "body") {
          setIsDraggingROI(true);
          roiDragOffsetRef.current = { dRow: pos.row - roiCenterRow, dCol: pos.col - roiCenterCol };
          return;
        }
        if (hit === "resize") {
          resizeAspectRef.current = roiMode === "rect" && roiWidth > 0 && roiHeight > 0 ? roiWidth / roiHeight : null;
          setIsDraggingROIResize(roiMode);
          return;
        }
      }
    }
    if (!lockView) {
      dragRef.current = { startX: e.clientX, startY: e.clientY, startPanX: panX, startPanY: panY, wasDrag: false };
      setIsDragging(true);
    }
  }, [panX, panY, lockView, lockRoi, roiMode, roiCenterRow, roiCenterCol, screenToImage, roiHitTest]);

  const handleMouseMove = React.useCallback((e: React.MouseEvent) => {
    // ROI drag
    if (isDraggingROI) {
      const pos = screenToImage(e);
      if (pos) {
        const newRow = pos.row - roiDragOffsetRef.current.dRow;
        const newCol = pos.col - roiDragOffsetRef.current.dCol;
        setRoiCenter([newRow, newCol]);
      }
      return;
    }
    // ROI resize
    if (isDraggingROIResize) {
      const pos = screenToImage(e);
      if (pos) {
        if (isDraggingROIResize === "circle" || isDraggingROIResize === "square") {
          const dr = pos.row - roiCenterRow, dc = pos.col - roiCenterCol;
          setRoiRadius(Math.max(2, Math.sqrt(dr * dr + dc * dc)));
        } else if (isDraggingROIResize === "rect") {
          let newW = Math.max(4, Math.abs(pos.col - roiCenterCol) * 2);
          let newH = Math.max(4, Math.abs(pos.row - roiCenterRow) * 2);
          if (e.shiftKey && resizeAspectRef.current != null) {
            const aspect = resizeAspectRef.current;
            if (newW / newH > aspect) newH = Math.max(4, Math.round(newW / aspect));
            else newW = Math.max(4, Math.round(newH * aspect));
          }
          setRoiWidth(newW);
          setRoiHeight(newH);
        }
      }
      return;
    }

    // Fast-path: skip cursor readout during pan drag for 60fps
    if (isDragging) {
      const dx = e.clientX - dragRef.current.startX;
      const dy = e.clientY - dragRef.current.startY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) dragRef.current.wasDrag = true;
      setPanX(dragRef.current.startPanX + dx);
      setPanY(dragRef.current.startPanY + dy);
      return;
    }

    // Cursor info (only when not dragging)
    const canvas = canvasRef.current;
    if (canvas && width && height) {
      const rect = canvas.getBoundingClientRect();
      const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
      const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const imageCanvasX = (mouseCanvasX - cx - panX) / zoom + cx;
      const imageCanvasY = (mouseCanvasY - cy - panY) / zoom + cy;
      const displayScale = canvasW / width;
      const col = Math.floor(imageCanvasX / displayScale);
      const row = Math.floor(imageCanvasY / displayScale);
      if (col >= 0 && col < width && row >= 0 && row < height) {
        const r = realDataRef.current;
        const im = imagDataRef.current;
        if (r && im) {
          const idx = row * width + col;
          const re = r[idx];
          const ims = im[idx];
          const amp = Math.sqrt(re * re + ims * ims);
          const phase = Math.atan2(ims, re);
          setCursorInfo({ row, col, value: `amp=${formatNumber(amp)} phase=${phase.toFixed(2)}` });
        }
      } else {
        setCursorInfo(null);
      }
    }
  }, [isDragging, isDraggingROI, isDraggingROIResize, canvasW, canvasH, width, height, zoom, panX, panY, screenToImage, roiCenterRow, roiCenterCol, setRoiCenter, setRoiRadius, setRoiWidth, setRoiHeight]);

  const handleMouseUp = React.useCallback(() => {
    setIsDragging(false);
    setIsDraggingROI(false);
    setIsDraggingROIResize(null);
  }, []);

  const handleMouseLeave = React.useCallback(() => {
    setIsDragging(false);
    setIsDraggingROI(false);
    setIsDraggingROIResize(null);
    setCursorInfo(null);
  }, []);

  const handleWheel = React.useCallback((e: WheelEvent) => {
    e.preventDefault();
    if (lockView) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();

    // Mouse position in canvas pixel coordinates
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);

    // Canvas center
    const cx = canvasW / 2;
    const cy = canvasH / 2;

    // Mouse position in image-canvas space (undo zoom+pan transform)
    const mouseImageX = (mouseCanvasX - cx - panX) / zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - panY) / zoom + cy;

    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * factor));

    // New pan to keep mouse on the same image point
    const newPanX = mouseCanvasX - (mouseImageX - cx) * newZoom - cx;
    const newPanY = mouseCanvasY - (mouseImageY - cy) * newZoom - cy;

    setZoom(newZoom);
    setPanX(newPanX);
    setPanY(newPanY);
  }, [zoom, panX, panY, canvasW, canvasH, lockView]);

  const handleDoubleClick = React.useCallback(() => {
    if (lockView) return;
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, [lockView]);

  // Prevent scroll on canvas
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  // ============================================================================
  // Keyboard shortcuts
  // ============================================================================
  const handleKeyDown = React.useCallback((e: React.KeyboardEvent) => {
    if (e.key === "r" || e.key === "R") {
      if (!lockView) { setZoom(1); setPanX(0); setPanY(0); }
    }
    if (e.key === "f" || e.key === "F") {
      if (!lockFft) setShowFft(!showFft);
    }
  }, [lockView, lockFft, showFft]);

  // ============================================================================
  // Resize handle
  // ============================================================================
  const handleResizeMouseDown = React.useCallback((e: React.MouseEvent) => {
    if (lockView) return;
    e.stopPropagation();
    e.preventDefault();
    resizeRef.current = { startX: e.clientX, startY: e.clientY, startW: canvasW, startH: canvasH };
    setIsResizing(true);
  }, [canvasW, canvasH, lockView]);

  React.useEffect(() => {
    if (!isResizing) return;
    let rafId = 0;
    let latestW = resizeRef.current.startW;
    const aspect = height / width;
    const onMove = (e: MouseEvent) => {
      const dx = e.clientX - resizeRef.current.startX;
      const dy = e.clientY - resizeRef.current.startY;
      const delta = Math.max(dx, dy);
      latestW = Math.max(200, resizeRef.current.startW + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setCanvasW(latestW);
          setCanvasH(Math.round(latestW * aspect));
        });
      }
    };
    const onUp = () => {
      if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
      setCanvasW(latestW);
      setCanvasH(Math.round(latestW * aspect));
      setIsResizing(false);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { if (rafId) cancelAnimationFrame(rafId); window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [isResizing, width, height]);

  // ============================================================================
  // FFT overlay (d-spacing crosshair)
  // ============================================================================
  React.useEffect(() => {
    const overlay = fftOverlayRef.current;
    if (!overlay || !effectiveShowFft) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    overlay.width = Math.round(canvasW * DPR);
    overlay.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const fftW = fftCropDims?.fftWidth ?? nextPow2(width);
    const fftH = fftCropDims?.fftHeight ?? nextPow2(height);

    // D-spacing crosshair marker
    if (fftClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      // Map FFT image coords to screen coords using the center-based transform
      const scaleX = canvasW / fftW;
      const scaleY = canvasH / fftH;
      const screenX = canvasW / 2 + fftPanX + (fftClickInfo.col - fftW / 2) * fftZoom * scaleX;
      const screenY = canvasH / 2 + fftPanY + (fftClickInfo.row - fftH / 2) * fftZoom * scaleY;
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
        const label = d >= 10 ? `d = ${(d / 10).toFixed(2)} nm` : `d = ${d.toFixed(2)} Å`;
        ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        ctx.fillStyle = "white";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(label, screenX + 10, screenY - 4);
      }
      ctx.restore();
    }

    // Reciprocal-space scale bar
    if (pixelSize > 0) {
      const fftPixelSize = 1 / (fftW * pixelSize);
      drawFFTScaleBarHiDPI(overlay, DPR, fftZoom, fftPixelSize, fftW);
    }
  }, [effectiveShowFft, fftZoom, fftPanX, fftPanY, canvasW, canvasH, pixelSize, width, height,
      fftClickInfo, fftCropDims]);

  // ============================================================================
  // FFT mouse handlers (zoom/pan + d-spacing click)
  // ============================================================================

  // Convert FFT canvas mouse position to FFT image pixel coordinates
  // Accounts for center-based transform: translate(canvasW/2 + fftPanX, ...) scale(fftZoom * scaleX, ...)
  const fftScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const fftW = fftCropDims?.fftWidth ?? nextPow2(width);
    const fftH = fftCropDims?.fftHeight ?? nextPow2(height);
    const scaleX = canvasW / fftW;
    const scaleY = canvasH / fftH;
    const imgCol = (mouseX - canvasW / 2 - fftPanX) / (fftZoom * scaleX) + fftW / 2;
    const imgRow = (mouseY - canvasH / 2 - fftPanY) / (fftZoom * scaleY) + fftH / 2;
    if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
      return { col: imgCol, row: imgRow };
    }
    return null;
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    if (lockView) return;
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsFftDragging(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: fftPanX, pY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent) => {
    if (lockView) return;
    if (isFftDragging && fftPanStart) {
      const canvas = fftCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - fftPanStart.x) * scaleX;
      const dy = (e.clientY - fftPanStart.y) * scaleY;
      setFftPanX(fftPanStart.pX + dx);
      setFftPanY(fftPanStart.pY + dy);
    }
  };

  const handleFftMouseUp = (e: React.MouseEvent) => {
    if (lockView) {
      fftClickStartRef.current = null;
      setIsFftDragging(false);
      setFftPanStart(null);
      return;
    }
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const pos = fftScreenToImg(e);
        if (pos) {
          const fftW = fftCropDims?.fftWidth ?? nextPow2(width);
          const fftH = fftCropDims?.fftHeight ?? nextPow2(height);
          let imgCol = pos.col;
          let imgRow = pos.row;
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
      fftClickStartRef.current = null;
    }
    setIsFftDragging(false);
    setFftPanStart(null);
  };

  const handleFftWheel = React.useCallback((e: WheelEvent) => {
    e.preventDefault();
    if (lockView) return;
    const canvas = fftCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftZoom * zoomFactor));
    const zoomRatio = newZoom / fftZoom;
    // Zoom centered on mouse: adjust pan so the image point under the cursor stays fixed
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const newPanX = (mouseX - cx) - ((mouseX - cx) - fftPanX) * zoomRatio;
    const newPanY = (mouseY - cy) - ((mouseY - cy) - fftPanY) * zoomRatio;
    setFftZoom(newZoom);
    setFftPanX(newPanX);
    setFftPanY(newPanY);
  }, [fftZoom, fftPanX, fftPanY, canvasW, canvasH, lockView]);

  const handleFftReset = () => {
    if (lockView) return;
    setFftZoom(3);
    setFftPanX(0);
    setFftPanY(0);
    setFftClickInfo(null);
  };

  // Prevent scroll on FFT canvas
  React.useEffect(() => {
    const el = fftContainerRef.current;
    if (!el || !effectiveShowFft) return;
    el.addEventListener("wheel", handleFftWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleFftWheel);
  }, [handleFftWheel, effectiveShowFft]);

  // Clear d-spacing marker when display mode or ROI changes
  React.useEffect(() => {
    setFftClickInfo(null);
  }, [displayMode, roiMode, roiCenterRow, roiCenterCol, roiRadius, roiWidth, roiHeight]);

  // ============================================================================
  // Export figure
  // ============================================================================
  const handleExportFigure = React.useCallback((withColorbar: boolean) => {
    setExportAnchor(null);
    if (lockExport) return;
    const r = realDataRef.current;
    const im = imagDataRef.current;
    const dispData = displayDataRef.current;
    if (!r || !im || !width || !height) return;

    const mode = displayMode as DisplayMode;

    // Create offscreen canvas at native resolution
    const offscreen = document.createElement("canvas");
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;
    const imgData = offCtx.createImageData(width, height);

    let vmin: number, vmax: number;
    let lut: Uint8Array | undefined;

    if (mode === "hsv") {
      // Custom HSV rendering
      const range = histRange;
      renderHSV(r, im, imgData.data, range.min, range.max);
      vmin = range.min;
      vmax = range.max;
    } else {
      if (!dispData) return;
      lut = COLORMAPS[mode === "phase" ? "hsv" : cmap] || COLORMAPS.inferno;
      if (autoContrast && mode !== "phase") {
        const pc = percentileClip(dispData, percentileLow, percentileHigh);
        vmin = pc.vmin;
        vmax = pc.vmax;
      } else if (mode === "phase") {
        vmin = -Math.PI;
        vmax = Math.PI;
      } else {
        ({ vmin, vmax } = sliderRange(histRange.min, histRange.max, vminPct, vmaxPct));
      }
      applyColormap(dispData, imgData.data, lut, vmin, vmax);
    }

    offCtx.putImageData(imgData, 0, 0);

    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title || undefined,
      lut: mode !== "hsv" ? lut : undefined,
      vmin,
      vmax,
      logScale: logScale && mode !== "phase",
      pixelSize: pixelSize > 0 ? pixelSize : undefined,
      showColorbar: withColorbar && mode !== "hsv",
      showScaleBar: pixelSize > 0,
    });

    canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, `showcomplex_${mode}.pdf`));
  }, [displayMode, cmap, logScale, autoContrast, percentileLow, percentileHigh,
      vminPct, vmaxPct, histRange, width, height, title, pixelSize, lockExport]);

  const handleExport = React.useCallback(() => {
    setExportAnchor(null);
    if (lockExport) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, `showcomplex2d_${(displayMode as string)}.png`);
    }, "image/png");
  }, [displayMode, lockExport]);

  const handleCopy = React.useCallback(async () => {
    if (lockExport) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/png"));
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch {
      canvas.toBlob((b) => { if (b) downloadBlob(b, `showcomplex2d_${(displayMode as string)}.png`); }, "image/png");
    }
  }, [displayMode, lockExport]);

  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;

  // ============================================================================
  // Render
  // ============================================================================
  const borderColor = themeColors.border;
  const mode = displayMode as DisplayMode;
  const isColormapEnabled = mode === "amplitude" || mode === "real" || mode === "imag";

  return (
    <Box className="showcomplex-root" tabIndex={0} onKeyDown={handleKeyDown}
      sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, overflow: "visible" }}>
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        <Box>
        {/* Title */}
        <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block" }}>
          {title || "ShowComplex2D"}
          <InfoTooltip theme={themeInfo.theme} text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
            <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum alongside image.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Mode: Display mode — amplitude, phase, HSV (phase→hue), real, or imaginary part.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile-based contrast (clips outliers). Disabled for phase/HSV modes.</Typography>
            <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
            <KeyboardShortcuts items={[
              ["R", "Reset zoom & pan"],
              ["F", "Toggle FFT"],
              ["Scroll", "Zoom in/out"],
              ["Drag", "Pan image"],
              ["Dbl-click", "Reset view"],
            ]} />
          </Box>} />
          <ControlCustomizer
            widgetName="ShowComplex2D"
            hiddenTools={hiddenTools}
            setHiddenTools={setHiddenTools}
            disabledTools={disabledTools}
            setDisabledTools={setDisabledTools}
            themeColors={themeColors}
          />
        </Typography>

        {/* Controls row */}
        <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
          {!hideFft && (
            <>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={showFft} onChange={(e) => { if (!lockFft) setShowFft(e.target.checked); }} disabled={lockFft} size="small" sx={switchStyles.small} />
            </>
          )}
          <Box sx={{ flex: 1 }} />
          {!hideView && (
            <Button size="small" sx={compactButton} disabled={lockView || !needsReset} onClick={handleDoubleClick}>Reset</Button>
          )}
          {!hideExport && (
            <>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }}>Export</Button>
              <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem disabled={lockExport} onClick={() => handleExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                <MenuItem disabled={lockExport} onClick={() => handleExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                <MenuItem disabled={lockExport} onClick={handleExport} sx={{ fontSize: 12 }}>PNG</MenuItem>
              </Menu>
              <Button size="small" sx={compactButton} disabled={lockExport} onClick={handleCopy}>Copy</Button>
            </>
          )}
        </Stack>

        {/* Canvas */}
        <Box ref={containerRef}
          sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${borderColor}`, cursor: isDragging ? "grabbing" : "grab" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onDoubleClick={handleDoubleClick}
        >
          <canvas ref={canvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
          <canvas ref={uiCanvasRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
          {cursorInfo && (
            <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
              <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                ({cursorInfo.row}, {cursorInfo.col}) {cursorInfo.value}
              </Typography>
            </Box>
          )}
          {!hideView && (
            <Box onMouseDown={handleResizeMouseDown} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.3 : 0.6, pointerEvents: lockView ? "none" : "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: lockView ? 0.3 : 1 } }} />
          )}
        </Box>

        {/* Stats bar */}
        {!hideStats && showStats && (
          <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", boxSizing: "border-box", overflow: "hidden", whiteSpace: "nowrap", opacity: lockStats ? 0.5 : 1 }}>
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMean)}</Box></Typography>
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMin)}</Box></Typography>
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMax)}</Box></Typography>
            <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsStd)}</Box></Typography>
          </Box>
        )}

        {/* Controls: two rows left + histogram right (matches Show2D layout) */}
        {showControls && (
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, boxSizing: "border-box" }}>
            <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Color + Mode */}
                {!hideDisplay && (
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                    <Select disabled={lockDisplay} value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45 }} MenuProps={themedMenuProps}>
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                    <Select disabled={lockDisplay || !isColormapEnabled} size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60, opacity: isColormapEnabled ? 1 : 0.4 }}>
                      {COLORMAP_NAMES.filter(n => n !== "hsv").map((name) => (
                        <MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>
                      ))}
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Mode:</Typography>
                    <Select disabled={lockDisplay} size="small" value={displayMode} onChange={(e) => setDisplayMode(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 90 }}>
                      <MenuItem value="amplitude">Amplitude</MenuItem>
                      <MenuItem value="phase">Phase</MenuItem>
                      <MenuItem value="hsv">HSV</MenuItem>
                      <MenuItem value="real">Real</MenuItem>
                      <MenuItem value="imag">Imaginary</MenuItem>
                    </Select>
                  </Box>
                )}
                {/* Row 2: Auto + zoom indicator */}
                {!hideDisplay && (
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                    <Switch checked={autoContrast} onChange={() => { if (!lockDisplay) setAutoContrast(!autoContrast); }} disabled={lockDisplay || mode === "phase" || mode === "hsv"} size="small" sx={switchStyles.small} />
                    {zoom !== 1 && (
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold" }}>{zoom.toFixed(1)}x</Typography>
                    )}
                  </Box>
                )}
                {/* Row 3: ROI */}
                {!hideRoi && (
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockRoi ? 0.5 : 1, pointerEvents: lockRoi ? "none" : "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>ROI:</Typography>
                    <Select disabled={lockRoi} size="small" value={roiMode} onChange={(e) => { if (!lockRoi) { if (e.target.value !== "off") lastRoiModeRef.current = e.target.value; setRoiMode(e.target.value); } }} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60 }}>
                      <MenuItem value="off">Off</MenuItem>
                      <MenuItem value="circle">Circle</MenuItem>
                      <MenuItem value="square">Square</MenuItem>
                      <MenuItem value="rect">Rect</MenuItem>
                    </Select>
                    {roiMode === "rect" ? (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10 }}>W:</Typography>
                        <Slider disabled={lockRoi} value={roiWidth} min={4} max={width} step={1} onChange={(_, v) => setRoiWidth(v as number)} size="small" sx={{ width: 60, py: 0, "& .MuiSlider-thumb": { width: 8, height: 8 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 } }} />
                        <Typography sx={{ ...typography.label, fontSize: 10 }}>H:</Typography>
                        <Slider disabled={lockRoi} value={roiHeight} min={4} max={height} step={1} onChange={(_, v) => setRoiHeight(v as number)} size="small" sx={{ width: 60, py: 0, "& .MuiSlider-thumb": { width: 8, height: 8 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 } }} />
                      </>
                    ) : roiMode !== "off" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10 }}>R:</Typography>
                        <Slider disabled={lockRoi} value={roiRadius} min={2} max={Math.min(width, height) / 2} step={1} onChange={(_, v) => setRoiRadius(v as number)} size="small" sx={{ width: 80, py: 0, "& .MuiSlider-thumb": { width: 8, height: 8 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 } }} />
                      </>
                    )}
                  </Box>
                )}
              </Box>
              {!hideHistogram && (
                <Box sx={{ opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
                  <Histogram
                    data={histData}
                    vminPct={vminPct} vmaxPct={vmaxPct}
                    onRangeChange={(lo, hi) => { if (!lockHistogram) { setVminPct(lo); setVmaxPct(hi); } }}
                    width={110} height={58}
                    theme={themeInfo.theme}
                    dataMin={histRange.min} dataMax={histRange.max}
                  />
                </Box>
              )}
            </Box>
          </Box>
        )}
        </Box>

        {/* FFT Panel — side panel (same layout as Show2D) */}
        {effectiveShowFft && (
          <Box sx={{ width: canvasW }}>
            {/* FFT label — shows "ROI FFT (WxH)" when ROI active */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16, display: "flex", alignItems: "center" }}>
              {fftCropDims ? (
                <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: accentGreen, fontWeight: "bold" }}>
                  ROI FFT ({fftCropDims.cropWidth}×{fftCropDims.cropHeight})
                </Typography>
              ) : (
                <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: themeColors.textMuted }}>
                  FFT
                </Typography>
              )}
            </Box>
            {/* Controls row — matches main panel controls row height */}
            <Stack direction="row" justifyContent="flex-end" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              {!hideView && (
                <Button size="small" sx={compactButton} disabled={lockView || (fftZoom === 3 && fftPanX === 0 && fftPanY === 0)} onClick={handleFftReset}>Reset</Button>
              )}
            </Stack>
            <Box
              ref={fftContainerRef}
              sx={{ position: "relative", border: `1px solid ${borderColor}`, bgcolor: "#000", cursor: lockView ? "default" : (isFftDragging ? "grabbing" : "grab"), width: canvasW, height: canvasH }}
              onMouseDown={lockView ? undefined : handleFftMouseDown}
              onMouseMove={lockView ? undefined : handleFftMouseMove}
              onMouseUp={lockView ? undefined : handleFftMouseUp}
              onMouseLeave={() => { fftClickStartRef.current = null; setIsFftDragging(false); setFftPanStart(null); }}
              onDoubleClick={lockView ? undefined : handleFftReset}
            >
              <canvas ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
              <canvas ref={fftOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
            </Box>
            {/* FFT d-spacing readout */}
            {fftClickInfo && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                  {fftClickInfo.dSpacing != null ? (
                    <>d = <Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} Å`}</Box>{" | |g| = "}<Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.spatialFreq!.toFixed(4)} Å⁻¹</Box></>
                  ) : (
                    <>dist = <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                  )}
                </Typography>
              </Box>
            )}
            {/* FFT controls + histogram */}
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>
              <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
                <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                  <Box sx={{ ...controlRow, border: `1px solid ${borderColor}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                    <Select disabled={lockDisplay} value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45 }} MenuProps={themedMenuProps}>
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                    <Switch checked={fftAuto} onChange={(e) => { if (!lockDisplay) setFftAuto(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                  </Box>
                </Box>
                {!hideHistogram && (
                  <Box sx={{ opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
                    <Histogram
                      data={fftHistData}
                      vminPct={fftVminPct} vmaxPct={fftVmaxPct}
                      onRangeChange={(lo, hi) => { if (!lockHistogram) { setFftVminPct(lo); setFftVmaxPct(hi); setFftAuto(false); } }}
                      width={110} height={40}
                      theme={themeInfo.theme}
                      dataMin={fftHistRange.min} dataMax={fftHistRange.max}
                    />
                  </Box>
                )}
              </Box>
            </Box>
          </Box>
        )}
      </Stack>
    </Box>
  );
}

export const render = createRender(ShowComplex2D);
