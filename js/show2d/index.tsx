/**
 * Show2D - Static 2D image viewer with gallery support.
 * 
 * Features:
 * - Single image or gallery mode with configurable columns
 * - Scroll to zoom, double-click to reset
 * - WebGPU-accelerated FFT with default 3x zoom
 * - Equal-sized FFT and histogram panels
 * - Click to select image in gallery mode
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar, roundToNiceValue, exportFigure, canvasToPDF } from "../scalebar";
import JSZip from "jszip";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, applyLogScale, applyLogScaleInPlace, percentileClip, sliderRange, computeStats } from "../stats";
import { ControlCustomizer } from "../control-customizer";
import { computeToolVisibility } from "../tool-parity";

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

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};
import { getWebGPUFFT, WebGPUFFT, fft2d, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2 } from "../webgpu-fft";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen, renderToOffscreenReuse } from "../colormaps";
import "./show2d.css";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

const DPR = window.devicePixelRatio || 1;

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
        onChange={(_, v) => { const [newMin, newMax] = v as number[]; onRangeChange(Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1)); }}
        min={0} max={100} size="small" valueLabelDisplay="auto"
        valueLabelFormat={(pct) => { const val = dataMin + (pct / 100) * (dataMax - dataMin); return val >= 1000 ? val.toExponential(1) : val.toFixed(1); }}
        sx={{ width, py: 0, "& .MuiSlider-thumb": { width: 8, height: 8 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vminPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vmaxPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography></Box>
    </Box>
  );
}

// ============================================================================
// Line profile sampling (bilinear interpolation along line)
// ============================================================================
function sampleLineProfile(data: Float32Array, w: number, h: number, row0: number, col0: number, row1: number, col1: number): Float32Array {
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
// FFT peak finder (snap to Bragg spot with sub-pixel centroid refinement)
// ============================================================================
function findFFTPeak(mag: Float32Array, width: number, height: number, col: number, row: number, radius: number): { row: number; col: number } {
  // Find brightest pixel in search window
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
  // Sub-pixel refinement via weighted centroid in 3×3 window
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

// ============================================================================
// Types
// ============================================================================
type ZoomState = { zoom: number; panX: number; panY: number };

// ============================================================================
// Constants
// ============================================================================
const SINGLE_IMAGE_TARGET = 500;
const GALLERY_IMAGE_TARGET = 300;
const DEFAULT_FFT_ZOOM = 3;
const DEFAULT_ZOOM_STATE: ZoomState = { zoom: 1, panX: 0, panY: 0 };
const PROFILE_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];
type ROIItem = { row: number; col: number; shape: string; radius: number; radius_inner: number; width: number; height: number; color: string; line_width: number; highlight: boolean };
const ROI_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];
const RESIZE_HIT_AREA_PX = 10;

function drawROI(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  shape: "circle" | "square" | "rectangle" | "annular",
  radius: number, w: number, h: number,
  activeColor: string, inactiveColor: string,
  active: boolean = false, innerRadius: number = 0
): void {
  const strokeColor = active ? activeColor : inactiveColor;
  ctx.strokeStyle = strokeColor;
  if (shape === "circle") {
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.stroke();
  } else if (shape === "square") {
    ctx.strokeRect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (shape === "rectangle") {
    ctx.strokeRect(x - w / 2, y - h / 2, w, h);
  } else if (shape === "annular") {
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = active ? "#0ff" : inactiveColor;
    ctx.beginPath(); ctx.arc(x, y, innerRadius, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = (active ? activeColor : inactiveColor) + "15";
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.arc(x, y, innerRadius, 0, Math.PI * 2, true); ctx.fill();
    ctx.strokeStyle = strokeColor;
  }
  if (active) {
    ctx.beginPath();
    ctx.moveTo(x - 5, y); ctx.lineTo(x + 5, y);
    ctx.moveTo(x, y - 5); ctx.lineTo(x, y + 5);
    ctx.stroke();
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
  let x0: number, y0: number, x1: number, y1: number;

  if (shape === "rectangle") {
    const hw = roi.width / 2;
    const hh = roi.height / 2;
    x0 = Math.max(0, Math.floor(roi.col - hw));
    y0 = Math.max(0, Math.floor(roi.row - hh));
    x1 = Math.min(imgW, Math.ceil(roi.col + hw));
    y1 = Math.min(imgH, Math.ceil(roi.row + hh));
  } else {
    const r = roi.radius;
    x0 = Math.max(0, Math.floor(roi.col - r));
    y0 = Math.max(0, Math.floor(roi.row - r));
    x1 = Math.min(imgW, Math.ceil(roi.col + r));
    y1 = Math.min(imgH, Math.ceil(roi.row + r));
  }

  const cropW = x1 - x0;
  const cropH = y1 - y0;
  if (cropW < 2 || cropH < 2) return null;

  const cropped = new Float32Array(cropW * cropH);

  if (shape === "circle" || shape === "annular") {
    const r = roi.radius;
    const rSq = r * r;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const imgX = x0 + dx;
        const imgY = y0 + dy;
        const distSq = (imgX - roi.col) * (imgX - roi.col) + (imgY - roi.row) * (imgY - roi.row);
        cropped[dy * cropW + dx] = distSq <= rSq ? data[imgY * imgW + imgX] : 0;
      }
    }
  } else {
    for (let dy = 0; dy < cropH; dy++) {
      const srcOffset = (y0 + dy) * imgW + x0;
      cropped.set(data.subarray(srcOffset, srcOffset + cropW), dy * cropW);
    }
  }

  return { cropped, cropW, cropH };
}

// ============================================================================
// Main Component
// ============================================================================
// Show4DSTEM-style UI constants
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
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
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
};
const sliderStyles = {
  small: { py: 0, "& .MuiSlider-thumb": { width: 10, height: 10 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 } },
};

function Show2D() {
  // Theme
  const { themeInfo, colors: tc } = useTheme();
  const themeColors = {
    ...tc,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
  };

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
  const [nImages] = useModelState<number>("n_images");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [labels] = useModelState<string[]>("labels");
  const [title] = useModelState<string>("title");
  const [widgetVersion] = useModelState<string>("widget_version");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [ncols] = useModelState<number>("ncols");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");

  // Customization
  const [canvasSizeTrait] = useModelState<number>("canvas_size");

  // Scale bar
  const [pixelSize] = useModelState<number>("pixel_size");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");

  // UI visibility
  const [showControls] = useModelState<boolean>("show_controls");
  const [showStats] = useModelState<boolean>("show_stats");
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");
  const [statsMean] = useModelState<number[]>("stats_mean");
  const [statsMin] = useModelState<number[]>("stats_min");
  const [statsMax] = useModelState<number[]>("stats_max");
  const [statsStd] = useModelState<number[]>("stats_std");

  // Analysis Panels (FFT + Histogram)
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");

  // Selection
  const [selectedIdx, setSelectedIdx] = useModelState<number>("selected_idx");

  // ROI
  const [roiActive, setRoiActive] = useModelState<boolean>("roi_active");
  const [roiList, setRoiList] = useModelState<ROIItem[]>("roi_list");
  const [roiSelectedIdx, setRoiSelectedIdx] = useModelState<number>("roi_selected_idx");
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false);
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const resizeAspectRef = React.useRef<number | null>(null);
  const [newRoiShape, setNewRoiShape] = React.useState<"circle" | "square" | "rectangle" | "annular">("square");
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);
  const selectedRoi = roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0) ? roiList[roiSelectedIdx] : null;

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("Show2D", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideDisplay = toolVisibility.isHidden("display");
  const hideHistogram = toolVisibility.isHidden("histogram");
  const hideStats = toolVisibility.isHidden("stats");
  const hideView = toolVisibility.isHidden("view");
  const hideExport = toolVisibility.isHidden("export");
  const hideRoi = toolVisibility.isHidden("roi");
  const hideProfile = toolVisibility.isHidden("profile");

  const lockDisplay = toolVisibility.isLocked("display");
  const lockHistogram = toolVisibility.isLocked("histogram");
  const lockStats = toolVisibility.isLocked("stats");
  const lockNavigation = toolVisibility.isLocked("navigation");
  const lockView = toolVisibility.isLocked("view");
  const lockExport = toolVisibility.isLocked("export");
  const lockRoi = toolVisibility.isLocked("roi");
  const lockProfile = toolVisibility.isLocked("profile");
  const effectiveShowFft = showFft && !hideDisplay;

  const updateSelectedRoi = (updates: Partial<ROIItem>) => {
    if (lockRoi) return;
    if (roiSelectedIdx < 0 || !roiList) return;
    const newList = [...roiList];
    newList[roiSelectedIdx] = { ...newList[roiSelectedIdx], ...updates };
    setRoiList(newList);
  };

  React.useEffect(() => {
    if (hideRoi && roiActive) {
      setRoiActive(false);
      setRoiSelectedIdx(-1);
    }
  }, [hideRoi, roiActive, setRoiActive, setRoiSelectedIdx]);

  // Canvas refs
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const imageContainerRefs = React.useRef<(HTMLDivElement | null)[]>([]);
  const fftContainerRefs = React.useRef<(HTMLDivElement | null)[]>([]);
  const singleFftContainerRef = React.useRef<HTMLDivElement>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const [canvasReady, setCanvasReady] = React.useState(0);  // Trigger re-render when refs attached

  // Zoom/Pan state - per-image when not linked, shared when linked
  const [zoomStates, setZoomStates] = React.useState<Map<number, ZoomState>>(new Map());
  const [linkedZoomState, setLinkedZoomState] = React.useState<ZoomState>(DEFAULT_ZOOM_STATE);
  const [linkedZoom, setLinkedZoom] = React.useState(false);  // Link zoom across gallery images
  const [isDraggingPan, setIsDraggingPan] = React.useState(false);
  const [panStart, setPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  // Helper to get zoom state for an image
  const getZoomState = React.useCallback((idx: number): ZoomState => {
    if (linkedZoom) return linkedZoomState;
    return zoomStates.get(idx) || DEFAULT_ZOOM_STATE;
  }, [linkedZoom, linkedZoomState, zoomStates]);

  // Helper to set zoom state for an image
  const setZoomState = React.useCallback((idx: number, state: ZoomState) => {
    if (linkedZoom) {
      setLinkedZoomState(state);
    } else {
      setZoomStates(prev => new Map(prev).set(idx, state));
    }
  }, [linkedZoom]);

  // FFT zoom/pan state (single mode)
  const [fftZoom, setFftZoom] = React.useState(DEFAULT_FFT_ZOOM);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const [isDraggingFftPan, setIsDraggingFftPan] = React.useState(false);
  const [fftPanStart, setFftPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  // Histogram state — per-image contrast ranges (gallery) or single (one image)
  const [linkedContrast, setLinkedContrast] = React.useState(true); // link contrast across gallery images
  const [linkedContrastState, setLinkedContrastState] = React.useState<{ vminPct: number; vmaxPct: number }>({ vminPct: 0, vmaxPct: 100 });
  const [contrastStates, setContrastStates] = React.useState<Map<number, { vminPct: number; vmaxPct: number }>>(new Map());
  const getContrastState = React.useCallback((idx: number) => {
    if (linkedContrast) return linkedContrastState;
    return contrastStates.get(idx) || { vminPct: 0, vmaxPct: 100 };
  }, [linkedContrast, linkedContrastState, contrastStates]);
  const setContrastState = React.useCallback((idx: number, state: { vminPct: number; vmaxPct: number }) => {
    if (linkedContrast) {
      setLinkedContrastState(state);
    } else {
      setContrastStates(prev => new Map(prev).set(idx, state));
    }
  }, [linkedContrast]);
  // Convenience accessors for active image
  const activeContrastIdx = nImages > 1 ? selectedIdx : 0;
  const imageVminPct = getContrastState(activeContrastIdx).vminPct;
  const imageVmaxPct = getContrastState(activeContrastIdx).vmaxPct;

  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });

  // FFT display state (single mode)
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftScaleMode, setFftScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const [fftAuto, setFftAuto] = React.useState(true);
  const [fftStats, setFftStats] = React.useState<number[] | null>(null);
  const [fftShowColorbar, setFftShowColorbar] = React.useState(false);

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);

  // Colorbar state (single image mode only)
  const [showColorbar, setShowColorbar] = React.useState(false);

  // Inset magnifier state
  const [showLens, setShowLens] = React.useState(false);
  const [lensPos, setLensPos] = React.useState<{ row: number; col: number } | null>(null);
  const [lensMag, setLensMag] = React.useState(4);       // magnification 2×–8×
  const [lensDisplaySize, setLensDisplaySize] = React.useState(128); // CSS px 64–256
  const [lensAnchor, setLensAnchor] = React.useState<{ x: number; y: number } | null>(null); // custom position (CSS px from top-left of canvas)
  const [isDraggingLens, setIsDraggingLens] = React.useState(false);
  const [isResizingLens, setIsResizingLens] = React.useState(false);
  const [isHoveringLensEdge, setIsHoveringLensEdge] = React.useState(false);
  const lensDragStartRef = React.useRef<{ mx: number; my: number; ax: number; ay: number } | null>(null);
  const lensResizeStartRef = React.useRef<{ my: number; startSize: number } | null>(null);
  const lensCanvasRef = React.useRef<HTMLCanvasElement | null>(null);

  // FFT d-spacing measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileLine, setProfileLine] = useModelState<{ row: number; col: number }[]>("profile_line");
  const [profileDataAll, setProfileDataAll] = React.useState<(Float32Array | null)[]>([]);
  React.useEffect(() => {
    if (hideProfile && profileActive) {
      setProfileActive(false);
    }
  }, [hideProfile, profileActive]);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);

  // Sync profile points from model state
  const profilePoints = profileLine || [];
  const setProfilePoints = (pts: { row: number; col: number }[]) => setProfileLine(pts);

  // Distance measurement state (JS-only, not persisted)
  const [measureActive, setMeasureActive] = React.useState(false);
  const [measurePoints, setMeasurePoints] = React.useState<{row: number; col: number}[]>([]);

  // FFT zoom/pan state (gallery mode — per-image or linked)
  const [galleryFftStates, setGalleryFftStates] = React.useState<Map<number, ZoomState>>(new Map());
  const [linkedFftZoomState, setLinkedFftZoomState] = React.useState<ZoomState>({ zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 });
  const [fftPanningIdx, setFftPanningIdx] = React.useState<number | null>(null);
  const getGalleryFftState = React.useCallback((idx: number) => {
    if (linkedZoom) return linkedFftZoomState;
    return galleryFftStates.get(idx) || { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 };
  }, [linkedZoom, linkedFftZoomState, galleryFftStates]);
  const setGalleryFftState = React.useCallback((idx: number, state: ZoomState) => {
    if (linkedZoom) {
      setLinkedFftZoomState(state);
    } else {
      setGalleryFftStates(prev => new Map(prev).set(idx, state));
    }
  }, [linkedZoom]);

  // Resizable state (gallery starts smaller)
  const [canvasSize, setCanvasSize] = React.useState(nImages > 1 ? GALLERY_IMAGE_TARGET : SINGLE_IMAGE_TARGET);

  // Sync initial sizes from traits
  React.useEffect(() => {
    if (canvasSizeTrait > 0) setCanvasSize(canvasSizeTrait);
  }, [canvasSizeTrait]);

  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number, y: number, size: number } | null>(null);

  // Profile height resize
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const [profileResizeStart, setProfileResizeStart] = React.useState<{ y: number; height: number } | null>(null);

  // WebGPU FFT
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  const rawDataRef = React.useRef<Float32Array[] | null>(null);

  // Cached offscreen canvases for main image rendering (avoids per-zoom/pan recompute)
  const mainOffscreensRef = React.useRef<HTMLCanvasElement[]>([]);
  const mainImgDatasRef = React.useRef<ImageData[]>([]);
  const logBufferRef = React.useRef<Float32Array | null>(null);
  const colorbarVminRef = React.useRef(0);
  const colorbarVmaxRef = React.useRef(1);
  const [offscreenVersion, setOffscreenVersion] = React.useState(0);

  // Inline FFT refs for gallery mode
  const fftCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const fftOffscreensRef = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const fftMagCacheGalleryRef = React.useRef<(Float32Array | null)[]>([]);
  const galleryFftDimsRef = React.useRef<{ w: number; h: number } | null>(null);
  const [galleryFftMagVersion, setGalleryFftMagVersion] = React.useState(0);

  // Cached FFT magnitude for single image mode (avoids recomputing on zoom/pan)
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);

  // Cached FFT offscreen canvas for single mode (avoids reprocessing on zoom/pan)
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const [fftOffscreenVersion, setFftOffscreenVersion] = React.useState(0);

  // ROI FFT state: when ROI + FFT are both active, compute FFT of cropped ROI region
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);

  // Layout calculations
  const isGallery = nImages > 1;
  const displayScale = canvasSize / Math.max(width, height);
  const canvasW = Math.round(width * displayScale);
  const canvasH = Math.round(height * displayScale);
  const floatsPerImage = width * height;
  const galleryGridWidth = isGallery ? ncols * canvasW + (ncols - 1) * 8 : canvasW;
  const profileCanvasWidth = galleryGridWidth;

  // ROI FFT active: both ROI and FFT on, with a selected ROI
  const roiFftActive = effectiveShowFft && roiActive && roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0);

  // Stable key for ROI geometry — only changes when the selected ROI's geometry changes,
  // not when other ROIs move or roiList gets a new reference from unrelated edits.
  // Shared by both ROI FFT and preview panel to avoid redundant recomputes.
  const selectedRoiKey = React.useMemo(() => {
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return "";
    const r = roiList[roiSelectedIdx];
    return `${r.row},${r.col},${r.radius},${r.radius_inner},${r.width},${r.height},${r.shape}`;
  }, [roiList, roiSelectedIdx]);
  const roiFftKey = roiFftActive ? selectedRoiKey : "";

  // Extract raw float32 bytes and parse into Float32Arrays
  const allFloats = React.useMemo(() => extractFloat32(frameBytes), [frameBytes]);

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
      }
    });
  }, []);

  const [dataReady, setDataReady] = React.useState(false);

  // Keep inline FFT ref arrays in sync with nImages
  React.useEffect(() => {
    fftCanvasRefs.current = fftCanvasRefs.current.slice(0, nImages);
    fftOffscreensRef.current = fftOffscreensRef.current.slice(0, nImages);
  }, [nImages]);

  // Parse frame data and store raw floats for FFT
  React.useEffect(() => {
    if (!allFloats || allFloats.length === 0) return;
    const dataArrays: Float32Array[] = [];
    for (let i = 0; i < nImages; i++) {
      const start = i * floatsPerImage;
      const imageData = allFloats.subarray(start, start + floatsPerImage);
      dataArrays.push(new Float32Array(imageData));
    }
    rawDataRef.current = dataArrays;
    setDataReady(true);

  }, [allFloats, nImages, floatsPerImage]);

  // Initialize reusable offscreen canvases (one per image, resized when dimensions change)
  React.useEffect(() => {
    if (width <= 0 || height <= 0 || nImages <= 0) return;
    const canvases: HTMLCanvasElement[] = [];
    const imgDatas: ImageData[] = [];
    for (let i = 0; i < nImages; i++) {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      canvases.push(canvas);
      imgDatas.push(canvas.getContext("2d")!.createImageData(width, height));
    }
    mainOffscreensRef.current = canvases;
    mainImgDatasRef.current = imgDatas;
    logBufferRef.current = new Float32Array(width * height);
  }, [width, height, nImages]);

  // Compute histogram data for the displayed image (reflects log scale)
  // In gallery mode, uses the selected image; in single mode, uses the only image
  React.useEffect(() => {
    if (!rawDataRef.current) return;
    const idx = nImages > 1 ? selectedIdx : 0;
    const raw = rawDataRef.current[idx];
    if (!raw) return;
    const d = logScale && logBufferRef.current
      ? applyLogScaleInPlace(raw, logBufferRef.current)
      : raw;
    setImageHistogramData(d);
    setImageDataRange(findDataRange(d));
  }, [allFloats, nImages, floatsPerImage, logScale, selectedIdx]);

  // Prevent page scroll when scrolling on canvases (must use native listener with passive: false)
  // In gallery mode, only block scroll on the selected image (or all if linkedZoom)
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const elements: (HTMLElement | null)[] = isGallery
      ? (linkedZoom
          ? [
              ...imageContainerRefs.current,
              ...(effectiveShowFft ? fftContainerRefs.current : []),
            ]
          : [
              imageContainerRefs.current[selectedIdx],
              ...(effectiveShowFft ? [fftContainerRefs.current[selectedIdx]] : []),
            ])
      : [
          imageContainerRefs.current[0],
          ...(effectiveShowFft ? [singleFftContainerRef.current] : []),
        ];
    elements.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => elements.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, [canvasReady, effectiveShowFft, isGallery, selectedIdx, linkedZoom]);

  // -------------------------------------------------------------------------
  // Data effect: normalize + colormap → reusable offscreen canvases
  // (does NOT depend on zoom/pan — avoids recomputing 16M pixels on every pan/zoom)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!dataReady || !rawDataRef.current || rawDataRef.current.length === 0) return;
    if (mainOffscreensRef.current.length === 0 || mainImgDatasRef.current.length === 0) return;

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    for (let i = 0; i < nImages; i++) {
      const offscreen = mainOffscreensRef.current[i];
      const imgData = mainImgDatasRef.current[i];
      if (!offscreen || !imgData) continue;

      const rawData = rawDataRef.current[i];
      if (!rawData) continue;

      // Apply log scale if enabled (reuse pre-allocated buffer to avoid per-image allocation)
      const processed = logScale && logBufferRef.current
        ? applyLogScaleInPlace(rawData, logBufferRef.current)
        : rawData;

      // Compute min/max for normalization (per-image contrast when unlinked)
      let vmin: number, vmax: number;
      const cs = linkedContrast ? linkedContrastState : (contrastStates.get(i) || { vminPct: 0, vmaxPct: 100 });
      const imgRange = findDataRange(processed);
      if (imgRange.min !== imgRange.max && (cs.vminPct > 0 || cs.vmaxPct < 100)) {
        ({ vmin, vmax } = sliderRange(imgRange.min, imgRange.max, cs.vminPct, cs.vmaxPct));
      } else if (autoContrast) {
        ({ vmin, vmax } = percentileClip(processed, 2, 98));
      } else {
        vmin = imgRange.min;
        vmax = imgRange.max;
      }

      // Cache vmin/vmax for colorbar and lens (avoids recomputing on zoom/pan/mousemove)
      if (i === 0) {
        colorbarVminRef.current = vmin;
        colorbarVmaxRef.current = vmax;
      }

      // Render to cached offscreen canvas (no per-frame allocation)
      renderToOffscreenReuse(processed, lut, vmin, vmax, offscreen, imgData);
    }
    setOffscreenVersion(v => v + 1);
  }, [dataReady, nImages, width, height, cmap, logScale, autoContrast, linkedContrast, linkedContrastState, contrastStates]);

  // -------------------------------------------------------------------------
  // Draw effect: zoom/pan changes — cheap, just drawImage from cached offscreens
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  // -------------------------------------------------------------------------
  React.useLayoutEffect(() => {
    if (mainOffscreensRef.current.length === 0) return;

    for (let i = 0; i < nImages; i++) {
      const canvas = canvasRefs.current[i];
      const offscreen = mainOffscreensRef.current[i];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const zs = linkedZoom ? linkedZoomState : (zoomStates.get(i) || DEFAULT_ZOOM_STATE);
      const { zoom, panX, panY } = zs;

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
    }
  }, [offscreenVersion, nImages, width, height, displayScale, canvasW, canvasH, canvasReady, linkedZoom, linkedZoomState, zoomStates]);

  // -------------------------------------------------------------------------
  // Render Overlays (scale bar, colorbar, zoom indicator)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    for (let i = 0; i < nImages; i++) {
      const overlay = overlayRefs.current[i];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;

      if (scaleBarVisible) {
        const zs = linkedZoom ? linkedZoomState : (zoomStates.get(i) || DEFAULT_ZOOM_STATE);
        const unit = pixelSize > 0 ? "Å" as const : "px" as const;
        const pxSize = pixelSize > 0 ? pixelSize : 1;
        drawScaleBarHiDPI(overlay, DPR, zs.zoom, pxSize, unit, width);
      } else {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }

      // Colorbar (single image mode only) — uses cached vmin/vmax from data effect
      if (!hideDisplay && showColorbar && !isGallery) {
        const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
        const cssW = overlay.width / DPR;
        const cssH = overlay.height / DPR;
        const vmin = colorbarVminRef.current;
        const vmax = colorbarVmaxRef.current;

        ctx.save();
        ctx.scale(DPR, DPR);
        drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, logScale);
        ctx.restore();
      }

      // ROI overlay — draw all ROIs
      if (!hideRoi && roiActive && roiList && roiList.length > 0) {
        const zs = linkedZoom ? linkedZoomState : (zoomStates.get(i) || DEFAULT_ZOOM_STATE);
        const { zoom, panX, panY } = zs;
        const cx = canvasW / 2;
        const cy = canvasH / 2;

        // Highlight mask: dim everything outside highlighted ROIs
        const highlightedRois = roiList.filter(r => r.highlight);
        if (highlightedRois.length > 0) {
          ctx.save();
          ctx.scale(DPR, DPR);
          ctx.fillStyle = "rgba(0,0,0,0.6)";
          ctx.fillRect(0, 0, canvasW, canvasH);
          ctx.globalCompositeOperation = "destination-out";
          for (const roi of highlightedRois) {
            const sx = (roi.col * displayScale - cx) * zoom + cx + panX;
            const sy = (roi.row * displayScale - cy) * zoom + cy + panY;
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
              // Re-darken inner ring
              ctx.globalCompositeOperation = "source-over";
              ctx.fillStyle = "rgba(0,0,0,0.6)";
              const sir = roi.radius_inner * displayScale * zoom;
              ctx.beginPath(); ctx.arc(sx, sy, sir, 0, Math.PI * 2); ctx.fill();
              ctx.globalCompositeOperation = "destination-out";
            }
          }
          ctx.restore();
        }

        ctx.save();
        ctx.scale(DPR, DPR);
        for (let ri = 0; ri < roiList.length; ri++) {
          const roi = roiList[ri];
          const isSelected = ri === roiSelectedIdx;
          const screenX = (roi.col * displayScale - cx) * zoom + cx + panX;
          const screenY = (roi.row * displayScale - cy) * zoom + cy + panY;
          const screenRadius = roi.radius * displayScale * zoom;
          const screenW = roi.width * displayScale * zoom;
          const screenH = roi.height * displayScale * zoom;
          const screenRadiusInner = roi.radius_inner * displayScale * zoom;
          const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
          ctx.lineWidth = roi.line_width || 2;
          drawROI(ctx, screenX, screenY, shape, screenRadius, screenW, screenH, roi.color || ROI_COLORS[ri % ROI_COLORS.length], roi.color || ROI_COLORS[ri % ROI_COLORS.length], isSelected && isDraggingROI, screenRadiusInner);
          if (isSelected) {
            ctx.setLineDash([4, 3]);
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1;
            if (shape === "circle" || shape === "annular") {
              ctx.beginPath(); ctx.arc(screenX, screenY, screenRadius + 3, 0, Math.PI * 2); ctx.stroke();
            } else if (shape === "square") {
              ctx.strokeRect(screenX - screenRadius - 3, screenY - screenRadius - 3, (screenRadius + 3) * 2, (screenRadius + 3) * 2);
            } else if (shape === "rectangle") {
              ctx.strokeRect(screenX - screenW / 2 - 3, screenY - screenH / 2 - 3, screenW + 6, screenH + 6);
            }
            ctx.setLineDash([]);
          }
        }
        ctx.restore();
      }

      // Line profile overlay
      if (!hideProfile && profileActive && profilePoints.length > 0) {
        const zs = linkedZoom ? linkedZoomState : (zoomStates.get(i) || DEFAULT_ZOOM_STATE);
        const { zoom, panX, panY } = zs;
        ctx.save();
        ctx.scale(DPR, DPR);

        // Transform image coords to screen coords
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        const toScreenX = (ix: number) => (ix * displayScale - cx) * zoom + cx + panX;
        const toScreenY = (iy: number) => (iy * displayScale - cy) * zoom + cy + panY;

        // Draw point A
        const ax = toScreenX(profilePoints[0].col);
        const ay = toScreenY(profilePoints[0].row);
        ctx.fillStyle = themeColors.accent;
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();

        // Draw line and point B if complete
        if (profilePoints.length === 2) {
          const bx = toScreenX(profilePoints[1].col);
          const by = toScreenY(profilePoints[1].row);

          ctx.strokeStyle = themeColors.accent;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = themeColors.accent;
          ctx.beginPath();
          ctx.arc(bx, by, 4, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.restore();
      }

      // Distance measurement overlay
      if (measureActive && measurePoints.length >= 1) {
        const zs = linkedZoom ? linkedZoomState : (zoomStates.get(i) || DEFAULT_ZOOM_STATE);
        const { zoom, panX, panY } = zs;
        ctx.save();
        ctx.scale(DPR, DPR);
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        const toSX = (ix: number) => (ix * displayScale - cx) * zoom + cx + panX;
        const toSY = (iy: number) => (iy * displayScale - cy) * zoom + cy + panY;

        ctx.shadowColor = "rgba(0,0,0,0.6)";
        ctx.shadowBlur = 3;

        // Endpoint A
        const ax = toSX(measurePoints[0].col);
        const ay = toSY(measurePoints[0].row);
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();

        if (measurePoints.length === 2) {
          const bx = toSX(measurePoints[1].col);
          const by = toSY(measurePoints[1].row);

          // Solid white line (distinct from profile's dashed accent line)
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();

          // Endpoint B
          ctx.beginPath();
          ctx.arc(bx, by, 4, 0, Math.PI * 2);
          ctx.fill();

          // Distance label
          const dc = measurePoints[1].col - measurePoints[0].col;
          const dr = measurePoints[1].row - measurePoints[0].row;
          const distPx = Math.sqrt(dc * dc + dr * dr);
          let label: string;
          if (pixelSize > 0) {
            const distA = distPx * pixelSize;
            label = distA >= 10 ? `${(distA / 10).toFixed(2)} nm` : `${distA.toFixed(2)} Å`;
          } else {
            label = `${distPx.toFixed(1)} px`;
          }

          const mx = (ax + bx) / 2;
          const my = (ay + by) / 2;
          ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillStyle = "#fff";
          ctx.fillText(label, mx, my - 8);
        }

        ctx.shadowBlur = 0;
        ctx.restore();
      }
    }
  }, [nImages, pixelSize, scaleBarVisible, selectedIdx, isGallery, canvasW, canvasH, width, displayScale, linkedZoom, linkedZoomState, zoomStates, dataReady, showColorbar, cmap, offscreenVersion, logScale, profileActive, profilePoints, roiActive, roiList, roiSelectedIdx, isDraggingROI, themeColors, hideDisplay, hideRoi, hideProfile, measureActive, measurePoints]);

  // -------------------------------------------------------------------------
  // Inset magnifier (lens) — renders magnified region at cursor in bottom-left
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const lensCanvas = lensCanvasRef.current;
    if (lensCanvas) {
      const lctx = lensCanvas.getContext("2d");
      if (lctx) lctx.clearRect(0, 0, lensCanvas.width, lensCanvas.height);
    }
    if (!showLens || lockDisplay || isGallery || !lensPos || !rawDataRef.current?.[0]) return;
    if (!lensCanvas) return;
    const ctx = lensCanvas.getContext("2d");
    if (!ctx) return;

    const raw = rawDataRef.current[0];
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    // Use cached vmin/vmax from data effect (avoids full-image applyLogScale + findDataRange)
    const vmin = colorbarVminRef.current;
    const vmax = colorbarVmaxRef.current;

    // Extract region around cursor — regionSize = displaySize / magnification
    const regionSize = Math.max(4, Math.round(lensDisplaySize / lensMag));
    const lensSize = lensDisplaySize;
    const margin = 12;
    const half = Math.floor(regionSize / 2);
    const r0 = lensPos.row - half;
    const c0 = lensPos.col - half;

    // Create small offscreen canvas for the region
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
          // Apply log scale inline per-pixel (only for the small region, not full image)
          const rawVal = raw[sr * width + sc];
          const val = logScale ? Math.log1p(rawVal) : rawVal;
          const t = Math.max(0, Math.min(1, (val - vmin) / range));
          const li = Math.round(t * 255);
          imgData.data[idx] = lut[li * 3]; imgData.data[idx + 1] = lut[li * 3 + 1]; imgData.data[idx + 2] = lut[li * 3 + 2]; imgData.data[idx + 3] = 255;
        }
      }
    }
    rctx.putImageData(imgData, 0, 0);

    // Draw lens inset on overlay — use custom anchor or default bottom-left
    ctx.save();
    ctx.scale(DPR, DPR);
    const lx = lensAnchor ? lensAnchor.x : margin;
    const ly = lensAnchor ? lensAnchor.y : canvasH - lensSize - margin - 20;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(regionCanvas, lx, ly, lensSize, lensSize);
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(lx, ly, lensSize, lensSize);
    // Crosshair at center
    const cx = lx + lensSize / 2;
    const cy = ly + lensSize / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();
    // Magnification label
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "10px monospace";
    ctx.fillText(`${lensMag}×`, lx + 4, ly + lensSize - 4);
    ctx.restore();
  }, [showLens, lockDisplay, lensPos, isGallery, cmap, logScale, offscreenVersion, width, height, canvasH, themeColors, lensMag, lensDisplaySize, lensAnchor]);

  // -------------------------------------------------------------------------
  // Auto-compute profile when profile_line is set (e.g. from Python)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (hideProfile) return;
    if (profilePoints.length === 2 && rawDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      const allProfiles: (Float32Array | null)[] = [];
      for (let i = 0; i < rawDataRef.current.length; i++) {
        const raw = rawDataRef.current[i];
        allProfiles.push(raw ? sampleLineProfile(raw, width, height, p0.row, p0.col, p1.row, p1.col) : null);
      }
      setProfileDataAll(allProfiles);
      if (!profileActive) setProfileActive(true);
    }
  }, [profilePoints, dataReady, hideProfile, profileActive]);

  // -------------------------------------------------------------------------
  // Render sparkline for line profile
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = profileCanvasWidth;
    const cssH = profileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, cssW, cssH);

    const hasData = profileDataAll.some(d => d && d.length >= 2);
    if (!hasData) {
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

    // Find global min/max across all profiles
    let gMin = Infinity, gMax = -Infinity;
    for (const d of profileDataAll) {
      if (!d) continue;
      for (let i = 0; i < d.length; i++) {
        if (d[i] < gMin) gMin = d[i];
        if (d[i] > gMax) gMax = d[i];
      }
    }
    const range = gMax - gMin || 1;

    // Draw each profile
    const colors = profileDataAll.length === 1 ? [themeColors.accent] : PROFILE_COLORS;
    for (let pIdx = 0; pIdx < profileDataAll.length; pIdx++) {
      const d = profileDataAll[pIdx];
      if (!d || d.length < 2) continue;
      ctx.strokeStyle = colors[pIdx % colors.length];
      ctx.lineWidth = pIdx === selectedIdx || profileDataAll.length === 1 ? 1.5 : 1;
      ctx.globalAlpha = pIdx === selectedIdx || profileDataAll.length === 1 ? 1 : 0.5;
      ctx.beginPath();
      for (let i = 0; i < d.length; i++) {
        const x = padLeft + (i / (d.length - 1)) * plotW;
        const y = padTop + plotH - ((d[i] - gMin) / range) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Compute total distance for x-axis
    const firstProfile = profileDataAll.find(d => d);
    let totalDist = (firstProfile?.length ?? 2) - 1;
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

    // Draw y-axis min/max labels
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

    // Legend (gallery mode with multiple images)
    if (profileDataAll.length > 1) {
      ctx.textAlign = "right";
      ctx.textBaseline = "top";
      ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      let legendX = cssW - 4;
      for (let pIdx = profileDataAll.length - 1; pIdx >= 0; pIdx--) {
        if (!profileDataAll[pIdx]) continue;
        const label = labels?.[pIdx] || `#${pIdx + 1}`;
        const color = colors[pIdx % colors.length];
        const textW = ctx.measureText(label).width;
        ctx.globalAlpha = pIdx === selectedIdx ? 1 : 0.5;
        ctx.fillStyle = color;
        ctx.fillRect(legendX - textW - 10, 2, 6, 6);
        ctx.fillStyle = isDark ? "#aaa" : "#555";
        ctx.fillText(label, legendX, 1);
        legendX -= textW + 16;
      }
      ctx.globalAlpha = 1;
    }

    // Save base rendering + layout for hover overlay
    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [profileDataAll, themeInfo.theme, themeColors.accent, profilePoints, pixelSize, selectedIdx, labels, profileCanvasWidth, profileHeight]);

  // Profile hover handler — draws crosshair + value readout
  const handleProfileMouseMove = React.useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
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

    // Restore base image
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

    // Dot on each profile line + collect values
    const colors = profileDataAll.length === 1 ? [themeColors.accent] : PROFILE_COLORS;
    const activeIdx = isGallery ? selectedIdx : 0;
    let displayVal: number | null = null;
    for (let pIdx = 0; pIdx < profileDataAll.length; pIdx++) {
      const d = profileDataAll[pIdx];
      if (!d || d.length < 2) continue;
      const dataIdx = Math.min(d.length - 1, Math.max(0, Math.round(frac * (d.length - 1))));
      const val = d[dataIdx];
      const y = padTop + plotH - ((val - gMin) / range) * plotH;
      ctx.fillStyle = colors[pIdx % colors.length];
      ctx.globalAlpha = pIdx === activeIdx || profileDataAll.length === 1 ? 1 : 0.5;
      ctx.beginPath();
      ctx.arc(cssX, y, 3, 0, Math.PI * 2);
      ctx.fill();
      if (pIdx === activeIdx || profileDataAll.length === 1) displayVal = val;
    }
    ctx.globalAlpha = 1;

    // Value readout label
    if (displayVal !== null) {
      const dist = frac * totalDist;
      const label = `${formatNumber(displayVal)}  @  ${dist.toFixed(1)} ${xUnit}`;
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
  }, [profileDataAll, themeInfo.theme, themeColors.accent, isGallery, selectedIdx]);

  const handleProfileMouseLeave = React.useCallback(() => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  }, []);

  // -------------------------------------------------------------------------
  // Compute FFT magnitude (cached — only recomputes when data changes)
  // Supports ROI-scoped FFT: when ROI is active with a selected ROI, compute
  // FFT of the cropped region instead of the full image.
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!effectiveShowFft || isGallery || !rawDataRef.current) return;
    if (!rawDataRef.current[selectedIdx]) return;
    let cancelled = false;

    const doCompute = async () => {
      const data = rawDataRef.current![selectedIdx];
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

      const real = inputData.slice();
      const imag = new Float32Array(inputData.length);

      let fReal: Float32Array;
      let fImag: Float32Array;
      if (gpuFFTRef.current && gpuReady) {
        const result = await gpuFFTRef.current.fft2D(real, imag, fftW, fftH, false);
        fReal = result.real;
        fImag = result.imag;
      } else {
        fft2d(real, imag, fftW, fftH, false);
        fReal = real;
        fImag = imag;
      }
      if (cancelled) return;
      fftshift(fReal, fftW, fftH);
      fftshift(fImag, fftW, fftH);

      fftMagCacheRef.current = computeMagnitude(fReal, fImag);
      setFftCropDims(origCropW > 0 ? { cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH } : null);
      setFftMagVersion(v => v + 1);
    };

    doCompute();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveShowFft, isGallery, selectedIdx, width, height, gpuReady, dataReady, roiFftKey]);

  // Clear FFT measurement when image, FFT state, or ROI changes
  React.useEffect(() => { setFftClickInfo(null); }, [selectedIdx, effectiveShowFft, roiFftActive, roiSelectedIdx]);

  // -------------------------------------------------------------------------
  // FFT data effect: normalize + colormap → cached offscreen canvas
  // (does NOT depend on fftZoom/fftPanX/fftPanY — avoids reprocessing on zoom/pan)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!effectiveShowFft || isGallery || !fftMagCacheRef.current) return;

    const fftMag = fftMagCacheRef.current;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;

    // Use crop dimensions when ROI FFT is active
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;

    // Apply scale mode
    const magnitude = new Float32Array(fftMag.length);
    for (let i = 0; i < fftMag.length; i++) {
      if (fftScaleMode === "log") {
        magnitude[i] = Math.log1p(fftMag[i]);
      } else if (fftScaleMode === "power") {
        magnitude[i] = Math.pow(fftMag[i], 0.5);
      } else {
        magnitude[i] = fftMag[i];
      }
    }

    let displayMin: number, displayMax: number;
    if (fftAuto) {
      ({ min: displayMin, max: displayMax } = autoEnhanceFFT(magnitude, fftW, fftH));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(magnitude));
    }

    const { mean, std } = computeStats(magnitude);
    setFftStats([mean, displayMin, displayMax, std]);

    // Store histogram data
    setFftHistogramData(magnitude.slice());
    setFftDataRange({ min: displayMin, max: displayMax });

    // Apply histogram slider clipping and render to cached offscreen
    const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);
    const offscreen = renderToOffscreen(magnitude, fftW, fftH, lut, vmin, vmax);
    if (!offscreen) return;
    fftOffscreenRef.current = offscreen;
    setFftOffscreenVersion(v => v + 1);
  }, [effectiveShowFft, isGallery, fftMagVersion, fftVminPct, fftVmaxPct, fftColormap, fftScaleMode, fftAuto, width, height, fftCropDims]);

  // -------------------------------------------------------------------------
  // FFT draw effect: cheap drawImage from cached offscreen (zoom/pan changes)
  // -------------------------------------------------------------------------
  React.useLayoutEffect(() => {
    if (!effectiveShowFft || isGallery || !fftCanvasRef.current || !fftOffscreenRef.current) return;

    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const offscreen = fftOffscreenRef.current;
    const fftW = offscreen.width;
    const fftH = offscreen.height;

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();

    const centerOffsetX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
    const centerOffsetY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;

    ctx.translate(centerOffsetX, centerOffsetY);
    ctx.scale(fftZoom, fftZoom);
    // Stretch cropped FFT to fill the full canvas (no layout change during drag)
    ctx.drawImage(offscreen, 0, 0, fftW, fftH, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [effectiveShowFft, isGallery, fftOffscreenVersion, canvasW, canvasH, fftZoom, fftPanX, fftPanY]);

  // -------------------------------------------------------------------------
  // Render FFT overlay (scale bar + colorbar + d-spacing marker)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const overlay = fftOverlayRef.current;
    if (!overlay || !effectiveShowFft || isGallery) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Use crop dimensions for reciprocal-space calculations
    const fftW = fftCropDims?.fftWidth ?? width;

    // Reciprocal-space scale bar
    if (pixelSize > 0) {
      const fftPixelSize = 1 / (fftW * pixelSize);
      drawFFTScaleBarHiDPI(overlay, DPR, fftZoom, fftPixelSize, fftW);
    }

    // FFT colorbar
    if (fftShowColorbar && fftDataRange.min !== fftDataRange.max) {
      const { vmin, vmax } = sliderRange(fftDataRange.min, fftDataRange.max, fftVminPct, fftVmaxPct);
      const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      const cssW = overlay.width / DPR;
      const cssH = overlay.height / DPR;
      drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, fftScaleMode === "log");
      ctx.restore();
    }

    // D-spacing crosshair marker — use crop dims for coordinate mapping
    const fftH = fftCropDims?.fftHeight ?? height;
    if (fftClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const centerOffsetX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
      const centerOffsetY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;
      const screenX = centerOffsetX + fftZoom * (fftClickInfo.col / fftW * canvasW);
      const screenY = centerOffsetY + fftZoom * (fftClickInfo.row / fftH * canvasH);
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
  }, [effectiveShowFft, isGallery, fftClickInfo, canvasW, canvasH, fftZoom, fftPanX, fftPanY, width, height, pixelSize, fftDataRange, fftVminPct, fftVmaxPct, fftColormap, fftScaleMode, fftShowColorbar, fftCropDims]);

  // -------------------------------------------------------------------------
  // Compute FFT magnitudes for gallery mode (cache raw magnitudes)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!effectiveShowFft || !isGallery || !rawDataRef.current) return;
    if (rawDataRef.current.length === 0) return;
    let cancelled = false;

    const computeAllFFTs = async () => {
      fftMagCacheGalleryRef.current = new Array(nImages).fill(null);

      // When ROI is active, crop each image before computing FFT
      const useRoiCrop = roiFftActive && roiList && roiSelectedIdx >= 0 && roiSelectedIdx < roiList.length;
      const roi = useRoiCrop ? roiList[roiSelectedIdx] : null;

      let fftW = width;
      let fftH = height;

      for (let idx = 0; idx < nImages; idx++) {
        if (cancelled) return;

        const data = rawDataRef.current![idx];
        if (!data) continue;

        let inputData = data;
        fftW = width;
        fftH = height;

        if (roi) {
          const crop = cropROIRegion(data, width, height, roi);
          if (crop) {
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

        const real = inputData.slice();
        const imag = new Float32Array(inputData.length);
        let fReal: Float32Array;
        let fImag: Float32Array;

        if (gpuFFTRef.current && gpuReady) {
          const result = await gpuFFTRef.current.fft2D(real, imag, fftW, fftH, false);
          fReal = result.real;
          fImag = result.imag;
        } else {
          fft2d(real, imag, fftW, fftH, false);
          fReal = real;
          fImag = imag;
        }

        if (cancelled) return;

        fftshift(fReal, fftW, fftH);
        fftshift(fImag, fftW, fftH);

        fftMagCacheGalleryRef.current[idx] = computeMagnitude(fReal, fImag);
      }
      if (!cancelled) {
        galleryFftDimsRef.current = { w: fftW, h: fftH };
        setGalleryFftMagVersion(v => v + 1);
      }
    };

    computeAllFFTs();

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveShowFft, isGallery, nImages, width, height, gpuReady, dataReady, roiFftKey]);

  // Gallery FFT data effect: normalize + colormap → cached offscreen canvases
  // (does NOT depend on gallery zoom/pan states)
  const [galleryFftOffscreenVersion, setGalleryFftOffscreenVersion] = React.useState(0);
  React.useEffect(() => {
    if (!effectiveShowFft || !isGallery) return;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const fftW = galleryFftDimsRef.current?.w ?? width;
    const fftH = galleryFftDimsRef.current?.h ?? height;

    for (let idx = 0; idx < nImages; idx++) {
      const magnitude = fftMagCacheGalleryRef.current[idx];
      if (!magnitude) continue;

      // Apply scale transform (same logic as single mode)
      let displayData: Float32Array;
      let displayMin: number, displayMax: number;
      if (fftScaleMode === "log") {
        displayData = applyLogScale(magnitude);
      } else if (fftScaleMode === "power") {
        displayData = new Float32Array(magnitude.length);
        for (let j = 0; j < magnitude.length; j++) displayData[j] = Math.sqrt(magnitude[j]);
      } else {
        displayData = magnitude;
      }
      if (fftAuto) {
        ({ min: displayMin, max: displayMax } = autoEnhanceFFT(magnitude, fftW, fftH));
        if (fftScaleMode === "log") { displayMin = Math.log1p(displayMin); displayMax = Math.log1p(displayMax); }
        else if (fftScaleMode === "power") { displayMin = Math.sqrt(displayMin); displayMax = Math.sqrt(displayMax); }
      } else {
        ({ min: displayMin, max: displayMax } = findDataRange(displayData));
      }
      const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);

      const offscreen = renderToOffscreen(displayData, fftW, fftH, lut, vmin, vmax);
      if (!offscreen) continue;
      fftOffscreensRef.current[idx] = offscreen;
    }

    // Update FFT histogram from selected image
    const selMag = fftMagCacheGalleryRef.current[selectedIdx];
    if (selMag) {
      let histData: Float32Array;
      if (fftScaleMode === "log") histData = applyLogScale(selMag);
      else if (fftScaleMode === "power") { histData = new Float32Array(selMag.length); for (let j = 0; j < selMag.length; j++) histData[j] = Math.sqrt(selMag[j]); }
      else histData = selMag;
      setFftHistogramData(histData);
      setFftDataRange(findDataRange(histData));
    }
    setGalleryFftOffscreenVersion(v => v + 1);
  }, [effectiveShowFft, isGallery, nImages, width, height, galleryFftMagVersion, fftColormap, fftScaleMode, fftAuto, fftVminPct, fftVmaxPct, selectedIdx]);

  // Gallery FFT draw effect: cheap drawImage from cached offscreens (zoom/pan changes)
  React.useLayoutEffect(() => {
    if (!effectiveShowFft || !isGallery) return;
    const fftW = galleryFftDimsRef.current?.w ?? width;
    const fftH = galleryFftDimsRef.current?.h ?? height;

    for (let idx = 0; idx < nImages; idx++) {
      const offscreen = fftOffscreensRef.current[idx];
      const canvas = fftCanvasRefs.current[idx];
      if (!offscreen || !canvas) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      const { zoom, panX, panY } = getGalleryFftState(idx);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvasW, canvasH);
      ctx.save();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, fftW, fftH, 0, 0, canvasW, canvasH);
      ctx.restore();
    }
  }, [effectiveShowFft, isGallery, nImages, canvasW, canvasH, width, height, galleryFftOffscreenVersion, galleryFftStates, linkedZoom, linkedFftZoomState]);

  // -------------------------------------------------------------------------
  // Mouse Handlers for Zoom/Pan
  // -------------------------------------------------------------------------
  const handleWheel = (e: React.WheelEvent, idx: number) => {
    if (lockView) return;
    // In gallery mode, only allow zoom on the selected image (unless linked)
    if (isGallery && idx !== selectedIdx && !linkedZoom) return;
    e.preventDefault(); // Prevent page scroll when zooming

    const canvas = canvasRefs.current[idx];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    
    // Get current zoom state
    const zs = getZoomState(idx);
    
    // Mouse position relative to canvas (in canvas pixel coordinates)
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    // Canvas center
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    
    // Mouse position relative to the current view (accounting for pan and zoom)
    // The transformation is: translate(cx + panX, cy + panY) -> scale(zoom) -> translate(-cx, -cy)
    // So a point on screen at (screenX, screenY) maps to image space as:
    // imageX = (screenX - cx - panX) / zoom + cx
    const mouseImageX = (mouseCanvasX - cx - zs.panX) / zs.zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - zs.panY) / zs.zoom + cy;

    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * zoomFactor));
    
    // Calculate new pan to keep the mouse position fixed on the same image point
    // After zoom: screenX = (imageX - cx) * newZoom + cx + newPanX
    // We want screenX to stay at mouseCanvasX, so:
    // newPanX = mouseCanvasX - (imageX - cx) * newZoom - cx
    const newPanX = mouseCanvasX - (mouseImageX - cx) * newZoom - cx;
    const newPanY = mouseCanvasY - (mouseImageY - cy) * newZoom - cy;

    setZoomState(idx, { zoom: newZoom, panX: newPanX, panY: newPanY });
  };

  const handleDoubleClick = (idx: number) => {
    if (lockView) return;
    setZoomState(idx, DEFAULT_ZOOM_STATE);
  };

  // Reset view (zoom/pan only — preserves profile, FFT state, etc.)
  const handleResetAll = () => {
    if (lockView) return;
    setZoomStates(new Map());
    setLinkedZoomState(DEFAULT_ZOOM_STATE);
    setGalleryFftStates(new Map());
    setLinkedFftZoomState({ zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 });
    setFftZoom(DEFAULT_FFT_ZOOM);
    setFftPanX(0);
    setFftPanY(0);
  };

  // FFT zoom/pan handlers
  const handleFftWheel = (e: React.WheelEvent) => {
    if (lockView) return;
    e.preventDefault(); // Prevent page scroll when zooming FFT
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setFftZoom(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftZoom * zoomFactor)));
  };

  const handleFftDoubleClick = () => {
    if (lockView) return;
    setFftZoom(DEFAULT_FFT_ZOOM);
    setFftPanX(0);
    setFftPanY(0);
    setFftClickInfo(null);
  };

  // Convert FFT canvas mouse position to FFT image pixel coordinates
  const fftScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const cOffX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
    const cOffY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    const imgCol = ((mouseX - cOffX) / fftZoom) / canvasW * fftW;
    const imgRow = ((mouseY - cOffY) / fftZoom) / canvasH * fftH;
    if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
      return { col: imgCol, row: imgRow };
    }
    return null;
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    if (lockView) return;
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsDraggingFftPan(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: fftPanX, pY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingFftPan || !fftPanStart) return;
    const dx = e.clientX - fftPanStart.x;
    const dy = e.clientY - fftPanStart.y;
    setFftPanX(fftPanStart.pX + dx);
    setFftPanY(fftPanStart.pY + dy);
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
          let imgCol = pos.col;
          let imgRow = pos.row;
          // Snap to nearest Bragg spot (local max in FFT magnitude)
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
    setIsDraggingFftPan(false);
    setFftPanStart(null);
  };

  const handleFftMouseLeave = () => {
    fftClickStartRef.current = null;
    setIsDraggingFftPan(false);
    setFftPanStart(null);
  };

  // Gallery FFT zoom/pan handlers (only selected image's FFT responds)
  const handleGalleryFftWheel = (e: React.WheelEvent, idx: number) => {
    if (lockView) return;
    if (isGallery && idx !== selectedIdx && !linkedZoom) return;
    e.preventDefault(); // Prevent page scroll when zooming FFT
    const zs = getGalleryFftState(idx);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setGalleryFftState(idx, { ...zs, zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * zoomFactor)) });
  };

  const handleGalleryFftMouseDown = (e: React.MouseEvent, idx: number) => {
    if (isGallery && idx !== selectedIdx) {
      if (lockNavigation) return;
      setSelectedIdx(idx);
      return; // Select first, don't start panning
    }
    if (lockView) return;
    const zs = getGalleryFftState(idx);
    setFftPanningIdx(idx);
    setIsDraggingFftPan(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
  };

  const handleGalleryFftMouseMove = (e: React.MouseEvent, idx: number) => {
    if (!isDraggingFftPan || !fftPanStart || fftPanningIdx !== idx) return;
    const dx = e.clientX - fftPanStart.x;
    const dy = e.clientY - fftPanStart.y;
    const zs = getGalleryFftState(idx);
    setGalleryFftState(idx, { ...zs, panX: fftPanStart.pX + dx, panY: fftPanStart.pY + dy });
  };

  const handleGalleryFftMouseUp = () => {
    setIsDraggingFftPan(false);
    setFftPanStart(null);
    setFftPanningIdx(null);
  };

  // Track which image is being panned
  const [panningIdx, setPanningIdx] = React.useState<number | null>(null);
  const clickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [draggingProfileEndpoint, setDraggingProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingProfileLine, setIsDraggingProfileLine] = React.useState(false);
  const [hoveredProfileEndpoint, setHoveredProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringProfileLine, setIsHoveringProfileLine] = React.useState(false);
  const profileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);

  const screenToImg = (e: React.MouseEvent, idx: number): { imgCol: number; imgRow: number } => {
    const canvas = canvasRefs.current[idx];
    if (!canvas) return { imgCol: 0, imgRow: 0 };
    const rect = canvas.getBoundingClientRect();
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zs = linkedZoom ? linkedZoomState : (zoomStates.get(idx) || DEFAULT_ZOOM_STATE);
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    return {
      imgCol: ((mouseCanvasX - cx - zs.panX) / zs.zoom + cx) / displayScale,
      imgRow: ((mouseCanvasY - cy - zs.panY) / zs.zoom + cy) / displayScale,
    };
  };

  const updateAllProfileData = (p0: { row: number; col: number }, p1: { row: number; col: number }) => {
    if (!rawDataRef.current) return;
    const allProfiles: (Float32Array | null)[] = [];
    for (let j = 0; j < rawDataRef.current.length; j++) {
      const raw = rawDataRef.current[j];
      allProfiles.push(raw ? sampleLineProfile(raw, width, height, p0.row, p0.col, p1.row, p1.col) : null);
    }
    setProfileDataAll(allProfiles);
  };

  const updateROI = (e: React.MouseEvent, idx: number) => {
    const { imgCol, imgRow } = screenToImg(e, idx);
    updateSelectedRoi({ col: Math.max(0, Math.min(width - 1, Math.floor(imgCol))), row: Math.max(0, Math.min(height - 1, Math.floor(imgRow))) });
  };

  const hitTestROI = (imgCol: number, imgRow: number): number => {
    if (!roiActive || !roiList) return -1;
    for (let ri = roiList.length - 1; ri >= 0; ri--) {
      const roi = roiList[ri];
      const shape = roi.shape || "circle";
      if (shape === "circle" || shape === "annular") {
        if (Math.sqrt((imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2) <= roi.radius) return ri;
      } else if (shape === "square") {
        if (Math.abs(imgCol - roi.col) <= roi.radius && Math.abs(imgRow - roi.row) <= roi.radius) return ri;
      } else if (shape === "rectangle") {
        if (Math.abs(imgCol - roi.col) <= roi.width / 2 && Math.abs(imgRow - roi.row) <= roi.height / 2) return ri;
      }
    }
    return -1;
  };

  const getHitArea = () => {
    const zoom = (linkedZoom ? linkedZoomState : (zoomStates.get(selectedIdx) || DEFAULT_ZOOM_STATE)).zoom;
    return RESIZE_HIT_AREA_PX / (displayScale * zoom);
  };

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
    if (!roiActive || !selectedRoi) return false;
    return isNearEdge(imgCol, imgRow, selectedRoi);
  };

  const isNearAnyEdge = (imgCol: number, imgRow: number): boolean => {
    if (!roiActive || !roiList) return false;
    return roiList.some(roi => isNearEdge(imgCol, imgRow, roi));
  };

  const isNearResizeHandleInner = (imgCol: number, imgRow: number): boolean => {
    if (!roiActive || !selectedRoi || selectedRoi.shape !== "annular") return false;
    const hitArea = getHitArea();
    const dist = Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
    return Math.abs(dist - selectedRoi.radius_inner) < hitArea;
  };

  const handleMouseDown = (e: React.MouseEvent, idx: number) => {
    const zs = getZoomState(idx);
    if (isGallery && idx !== selectedIdx) {
      if (lockNavigation) return;
      setSelectedIdx(idx);
      return;
    }
    // Check if click is on the lens inset — edge = resize, interior = drag
    if (!lockDisplay && showLens && !isGallery && idx === 0) {
      const canvas = canvasRefs.current[0];
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        if (cssX >= lx && cssX <= lx + lensDisplaySize && cssY >= ly && cssY <= ly + lensDisplaySize) {
          const edgeHit = 8;
          const nearEdge = cssX - lx < edgeHit || lx + lensDisplaySize - cssX < edgeHit || cssY - ly < edgeHit || ly + lensDisplaySize - cssY < edgeHit;
          if (nearEdge) {
            setIsResizingLens(true);
            lensResizeStartRef.current = { my: e.clientY, startSize: lensDisplaySize };
          } else {
            setIsDraggingLens(true);
            lensDragStartRef.current = { mx: e.clientX, my: e.clientY, ax: lx, ay: ly };
          }
          e.preventDefault();
          return;
        }
      }
    }
    clickStartRef.current = { x: e.clientX, y: e.clientY };
    if (profileActive && !lockProfile) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      if (profilePoints.length === 2) {
        const p0 = profilePoints[0];
        const p1 = profilePoints[1];
        const hitRadius = 10 / (displayScale * zs.zoom);
        const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
        const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingPan(false);
          setPanStart(null);
          setPanningIdx(null);
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
          setPanningIdx(null);
          return;
        }
      }
      if (!lockView) {
        setIsDraggingPan(true);
        setPanningIdx(idx);
        setPanStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
      }
      return;
    }
    if (roiActive) {
      if (lockRoi) {
        if (!lockView) {
          setIsDraggingPan(true);
          setPanningIdx(idx);
          setPanStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
        }
        return;
      }
      const { imgCol, imgRow } = screenToImg(e, idx);
      // Check resize handles on selected ROI first
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
      // Check edge of any ROI — auto-select and start resize
      if (roiList) {
        for (let ri = 0; ri < roiList.length; ri++) {
          if (isNearEdge(imgCol, imgRow, roiList[ri])) {
            e.preventDefault();
            const roi = roiList[ri];
            resizeAspectRef.current = roi && (roi.shape === "rectangle") && roi.width > 0 && roi.height > 0 ? roi.width / roi.height : null;
            setRoiSelectedIdx(ri);
            setIsDraggingResize(true);
            return;
          }
        }
      }
      // Hit-test existing ROIs (click inside to select + drag)
      const hitIdx = hitTestROI(imgCol, imgRow);
      if (hitIdx >= 0) {
        setRoiSelectedIdx(hitIdx);
        setIsDraggingROI(true);
        return;
      }
      // Click on empty space — deselect and allow panning
      setRoiSelectedIdx(-1);
    }
    // Start panning (works in both ROI-active and normal modes)
    {
      if (lockView) return;
      setIsDraggingPan(true);
      setPanningIdx(idx);
      setPanStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
    }
  };

  const handleMouseMove = (e: React.MouseEvent, idx: number) => {
    // Fast path: during pan drag, skip all cursor/hover/lens work — just update pan
    if (isDraggingPan && panStart && panningIdx !== null && !lockView) {
      const canvas = canvasRefs.current[idx];
      if (!canvas || idx !== panningIdx) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - panStart.x) * scaleX;
      const dy = (e.clientY - panStart.y) * scaleY;
      const zs = getZoomState(idx);
      setZoomState(idx, { ...zs, panX: panStart.pX + dx, panY: panStart.pY + dy });
      return;
    }

    // Cursor readout: convert screen position to image pixel coordinates
    const canvas = canvasRefs.current[idx];
    if (canvas && rawDataRef.current) {
      const rect = canvas.getBoundingClientRect();
      const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
      const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
      const zs = linkedZoom ? linkedZoomState : (zoomStates.get(idx) || DEFAULT_ZOOM_STATE);
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const imageCanvasX = (mouseCanvasX - cx - zs.panX) / zs.zoom + cx;
      const imageCanvasY = (mouseCanvasY - cy - zs.panY) / zs.zoom + cy;
      const imgX = Math.floor(imageCanvasX / displayScale);
      const imgY = Math.floor(imageCanvasY / displayScale);
      if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
        const rawData = rawDataRef.current[idx];
        if (rawData) setCursorInfo({ row: imgY, col: imgX, value: rawData[imgY * width + imgX] });
        if (!lockDisplay && showLens && !isGallery) setLensPos({ row: imgY, col: imgX });
      } else {
        setCursorInfo(null);
        // Don't clear lensPos — lens stays at last position when toggle is on
      }
    }

    // Lens drag
    if (!lockDisplay && isDraggingLens && lensDragStartRef.current) {
      const dx = e.clientX - lensDragStartRef.current.mx;
      const dy = e.clientY - lensDragStartRef.current.my;
      setLensAnchor({ x: lensDragStartRef.current.ax + dx, y: lensDragStartRef.current.ay + dy });
      return;
    }
    // Lens resize drag
    if (!lockDisplay && isResizingLens && lensResizeStartRef.current) {
      const dy = e.clientY - lensResizeStartRef.current.my;
      setLensDisplaySize(Math.max(64, Math.min(256, lensResizeStartRef.current.startSize + dy)));
      return;
    }

    if (profileActive && !lockProfile && profilePoints.length === 2) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      const p0 = profilePoints[0];
      const p1 = profilePoints[1];
      const activeZoom = linkedZoom ? linkedZoomState.zoom : (zoomStates.get(idx) || DEFAULT_ZOOM_STATE).zoom;
      const hitRadius = 10 / (displayScale * activeZoom);
      const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
      const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
      if (draggingProfileEndpoint !== null) {
        const clampedRow = Math.max(0, Math.min(height - 1, imgRow));
        const clampedCol = Math.max(0, Math.min(width - 1, imgCol));
        const next = [
          draggingProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
          draggingProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
        ];
        setProfilePoints(next);
        updateAllProfileData(next[0], next[1]);
        return;
      }
      if (isDraggingProfileLine && profileDragStartRef.current) {
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
        setProfilePoints(next);
        updateAllProfileData(next[0], next[1]);
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

    // ROI resize drag (inner annular ring)
    if (!lockRoi && isDraggingResizeInner && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e, idx);
      const newR = Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
      updateSelectedRoi({ radius_inner: Math.max(1, Math.min(selectedRoi.radius - 1, Math.round(newR))) });
      return;
    }
    // ROI resize drag (outer)
    if (!lockRoi && isDraggingResize && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e, idx);
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
        const newR = shape === "square" ? Math.max(Math.abs(ic - selectedRoi.col), Math.abs(ir - selectedRoi.row)) : Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
        const minR = shape === "annular" ? selectedRoi.radius_inner + 1 : 1;
        updateSelectedRoi({ radius: Math.max(minR, Math.round(newR)) });
      }
      return;
    }
    // ROI drag (move center)
    if (!lockRoi && isDraggingROI) {
      updateROI(e, idx);
      return;
    }
    // Lens edge hover detection
    if (!lockDisplay && showLens && !isGallery && canvas) {
      const rect = canvas.getBoundingClientRect();
      const cssX = e.clientX - rect.left;
      const cssY = e.clientY - rect.top;
      const margin = 12;
      const lx = lensAnchor ? lensAnchor.x : margin;
      const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
      const inside = cssX >= lx && cssX <= lx + lensDisplaySize && cssY >= ly && cssY <= ly + lensDisplaySize;
      const edgeHit = 8;
      const nearEdge = inside && (cssX - lx < edgeHit || lx + lensDisplaySize - cssX < edgeHit || cssY - ly < edgeHit || ly + lensDisplaySize - cssY < edgeHit);
      setIsHoveringLensEdge(nearEdge);
    } else {
      setIsHoveringLensEdge(false);
    }
    // Hover detection for resize handles (show cursor on any ROI edge)
    if (roiActive && !lockRoi && !isDraggingPan) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e, idx);
      setIsHoveringResizeInner(isNearResizeHandleInner(ic, ir));
      setIsHoveringResize(isNearAnyEdge(ic, ir));
    }

    // Panning
    if (lockView) return;
    if (!isDraggingPan || !panStart || panningIdx === null) return;
    if (idx !== panningIdx) return;
    if (!canvas) return;
    const rect2 = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect2.width;
    const scaleY = canvas.height / rect2.height;
    const dx = (e.clientX - panStart.x) * scaleX;
    const dy = (e.clientY - panStart.y) * scaleY;

    const zs = getZoomState(idx);
    setZoomState(idx, { ...zs, panX: panStart.pX + dx, panY: panStart.pY + dy });
  };

  const handleMouseUp = (e: React.MouseEvent, idx: number) => {
    if (isDraggingLens) {
      setIsDraggingLens(false);
      lensDragStartRef.current = null;
      return;
    }
    if (isResizingLens) {
      setIsResizingLens(false);
      lensResizeStartRef.current = null;
      return;
    }
    if (draggingProfileEndpoint !== null || isDraggingProfileLine) {
      setDraggingProfileEndpoint(null);
      setIsDraggingProfileLine(false);
      profileDragStartRef.current = null;
      clickStartRef.current = null;
      setIsDraggingROI(false);
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      setIsDraggingPan(false);
      setPanStart(null);
      setPanningIdx(null);
      setHoveredProfileEndpoint(null);
      setIsHoveringProfileLine(false);
      return;
    }
    // Detect click (vs drag) for profile mode
    if (profileActive && !lockProfile && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        // It's a click — compute image coordinates
        const canvas = canvasRefs.current[idx];
        if (canvas && rawDataRef.current) {
          const rect = canvas.getBoundingClientRect();
          const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const zs = linkedZoom ? linkedZoomState : (zoomStates.get(idx) || DEFAULT_ZOOM_STATE);
          const cx = canvasW / 2;
          const cy = canvasH / 2;
          const imgX = ((mouseCanvasX - cx - zs.panX) / zs.zoom + cx) / displayScale;
          const imgY = ((mouseCanvasY - cy - zs.panY) / zs.zoom + cy) / displayScale;
          if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
            const pt = { row: imgY, col: imgX };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              // Start new line
              setProfilePoints([pt]);
              setProfileDataAll([]);
            } else {
              // Complete the line
              const p0 = profilePoints[0];
              setProfilePoints([p0, pt]);
              updateAllProfileData(p0, pt);
            }
          }
        }
      }
    }
    // Detect click for measurement mode (only when profile is not active)
    if (measureActive && !profileActive && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = canvasRefs.current[idx];
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const zs = linkedZoom ? linkedZoomState : (zoomStates.get(idx) || DEFAULT_ZOOM_STATE);
          const cx = canvasW / 2;
          const cy = canvasH / 2;
          const imgX = ((mouseCanvasX - cx - zs.panX) / zs.zoom + cx) / displayScale;
          const imgY = ((mouseCanvasY - cy - zs.panY) / zs.zoom + cy) / displayScale;
          if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
            const pt = { row: imgY, col: imgX };
            if (measurePoints.length < 2) {
              setMeasurePoints([...measurePoints, pt]);
            } else {
              setMeasurePoints([pt]);
            }
          }
        }
      }
    }
    clickStartRef.current = null;
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    profileDragStartRef.current = null;
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingPan(false);
    setPanStart(null);
    setPanningIdx(null);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
  };

  const handleMouseLeave = (idx: number) => {
    setCursorInfo(null);
    // Don't clear lensPos — lens stays at last position when toggle is on
    setIsDraggingLens(false);
    setIsResizingLens(false);
    lensDragStartRef.current = null;
    lensResizeStartRef.current = null;
    setIsHoveringLensEdge(false);
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    profileDragStartRef.current = null;
    setIsHoveringResize(false);
    setIsHoveringResizeInner(false);
    if (panningIdx === idx) {
      setIsDraggingPan(false);
      setPanStart(null);
      setPanningIdx(null);
    }
  };

  // -------------------------------------------------------------------------
  // Copy to clipboard handler
  const handleCopy = React.useCallback(async () => {
    if (lockExport) return;
    const canvas = canvasRefs.current[isGallery ? selectedIdx : 0];
    if (!canvas) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/png"));
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch {
      // Fallback: download if clipboard API unavailable
      canvas.toBlob((b) => { if (b) downloadBlob(b, `show2d_${labels?.[selectedIdx] || "image"}.png`); }, "image/png");
    }
  }, [isGallery, selectedIdx, labels, lockExport]);

  // Export publication-quality figure with scale bar, colorbar, annotations
  const handleExportFigure = React.useCallback((withScaleBar: boolean, withColorbar: boolean) => {
    if (lockExport) return;
    setExportAnchor(null);
    const idx = isGallery ? selectedIdx : 0;
    const rawData = rawDataRef.current?.[idx];
    if (!rawData) return;

    const processed = logScale ? applyLogScale(rawData) : rawData;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    let vmin: number, vmax: number;
    if (!isGallery && imageDataRange.min !== imageDataRange.max) {
      ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
    } else if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else {
      const r = findDataRange(processed);
      vmin = r.min;
      vmax = r.max;
    }

    const offscreen = renderToOffscreen(processed, width, height, lut, vmin, vmax);
    if (!offscreen) return;

    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title || undefined,
      lut,
      vmin,
      vmax,
      logScale,
      pixelSize: pixelSize > 0 ? pixelSize : undefined,
      showColorbar: withColorbar,
      showScaleBar: withScaleBar && pixelSize > 0,
      drawAnnotations: (ctx) => {
        // ROI highlight mask
        if (roiActive && roiList) {
          const hlRois = roiList.filter(r => r.highlight);
          if (hlRois.length > 0) {
            ctx.save();
            ctx.fillStyle = "rgba(0,0,0,0.6)";
            ctx.fillRect(0, 0, width, height);
            ctx.globalCompositeOperation = "destination-out";
            for (const roi of hlRois) {
              ctx.fillStyle = "rgba(0,0,0,1)";
              const shape = roi.shape || "circle";
              if (shape === "circle") { ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius, 0, Math.PI * 2); ctx.fill(); }
              else if (shape === "square") { ctx.fillRect(roi.col - roi.radius, roi.row - roi.radius, roi.radius * 2, roi.radius * 2); }
              else if (shape === "rectangle") { ctx.fillRect(roi.col - roi.width / 2, roi.row - roi.height / 2, roi.width, roi.height); }
              else if (shape === "annular") {
                ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius, 0, Math.PI * 2); ctx.fill();
                ctx.globalCompositeOperation = "source-over";
                ctx.fillStyle = "rgba(0,0,0,0.6)";
                ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius_inner, 0, Math.PI * 2); ctx.fill();
                ctx.globalCompositeOperation = "destination-out";
              }
            }
            ctx.restore();
          }
          // ROI outlines
          for (const roi of roiList) {
            const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
            ctx.lineWidth = roi.line_width || 2;
            drawROI(ctx, roi.col, roi.row, shape, roi.radius, roi.width, roi.height, roi.color, roi.color, false, roi.radius_inner);
          }
        }
        // Profile line
        if (profileActive && profilePoints.length === 2) {
          ctx.strokeStyle = "#4fc3f7";
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(profilePoints[0].col, profilePoints[0].row);
          ctx.lineTo(profilePoints[1].col, profilePoints[1].row);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = "#4fc3f7";
          ctx.beginPath();
          ctx.arc(profilePoints[0].col, profilePoints[0].row, 3, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(profilePoints[1].col, profilePoints[1].row, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      },
    });

    canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, `show2d_figure_${labels?.[selectedIdx] || "image"}.pdf`));
  }, [isGallery, selectedIdx, labels, width, height, cmap, logScale, autoContrast, imageDataRange, imageVminPct, imageVmaxPct, pixelSize, title, roiActive, roiList, profileActive, profilePoints, lockExport]);

  // Export all variants (PNG + PDF) as zip
  const handleExportAll = React.useCallback(async () => {
    if (lockExport) return;
    setExportAnchor(null);
    const idx = isGallery ? selectedIdx : 0;
    const rawData = rawDataRef.current?.[idx];
    if (!rawData) return;

    const processed = logScale ? applyLogScale(rawData) : rawData;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    let vmin: number, vmax: number;
    if (!isGallery && imageDataRange.min !== imageDataRange.max) {
      ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
    } else if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else {
      const r = findDataRange(processed);
      vmin = r.min;
      vmax = r.max;
    }

    const offscreen = renderToOffscreen(processed, width, height, lut, vmin, vmax);
    if (!offscreen) return;

    const drawAnnotations = (ctx: CanvasRenderingContext2D) => {
      if (roiActive && roiList) {
        const hlRois = roiList.filter(r => r.highlight);
        if (hlRois.length > 0) {
          ctx.save();
          ctx.fillStyle = "rgba(0,0,0,0.6)";
          ctx.fillRect(0, 0, width, height);
          ctx.globalCompositeOperation = "destination-out";
          for (const roi of hlRois) {
            ctx.fillStyle = "rgba(0,0,0,1)";
            const shape = roi.shape || "circle";
            if (shape === "circle") { ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius, 0, Math.PI * 2); ctx.fill(); }
            else if (shape === "square") { ctx.fillRect(roi.col - roi.radius, roi.row - roi.radius, roi.radius * 2, roi.radius * 2); }
            else if (shape === "rectangle") { ctx.fillRect(roi.col - roi.width / 2, roi.row - roi.height / 2, roi.width, roi.height); }
            else if (shape === "annular") {
              ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius, 0, Math.PI * 2); ctx.fill();
              ctx.globalCompositeOperation = "source-over";
              ctx.fillStyle = "rgba(0,0,0,0.6)";
              ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius_inner, 0, Math.PI * 2); ctx.fill();
              ctx.globalCompositeOperation = "destination-out";
            }
          }
          ctx.restore();
          for (const roi of roiList) {
            const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
            ctx.lineWidth = roi.line_width || 2;
            drawROI(ctx, roi.col, roi.row, shape, roi.radius, roi.width, roi.height, roi.color, roi.color, false, roi.radius_inner);
          }
        }
      }
      if (profileActive && profilePoints.length === 2) {
        ctx.strokeStyle = "#4fc3f7";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(profilePoints[0].col, profilePoints[0].row);
        ctx.lineTo(profilePoints[1].col, profilePoints[1].row);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#4fc3f7";
        ctx.beginPath(); ctx.arc(profilePoints[0].col, profilePoints[0].row, 3, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(profilePoints[1].col, profilePoints[1].row, 3, 0, Math.PI * 2); ctx.fill();
      }
    };

    const hasScale = pixelSize > 0;
    const baseOpts = {
      imageCanvas: offscreen,
      title: title || undefined,
      lut,
      vmin,
      vmax,
      logScale,
      pixelSize: hasScale ? pixelSize : undefined,
      drawAnnotations,
    };

    const variants: { name: string; showScaleBar: boolean; showColorbar: boolean }[] = [
      { name: "figure", showScaleBar: false, showColorbar: false },
      { name: "figure_scalebar", showScaleBar: true, showColorbar: false },
      { name: "figure_scalebar_colorbar", showScaleBar: true, showColorbar: true },
    ];

    const zip = new JSZip();
    const prefix = `show2d_${labels?.[selectedIdx] || "image"}`;
    const metadata = {
      metadata_version: "1.0",
      widget_name: "Show2D",
      widget_version: widgetVersion || "unknown",
      exported_at: new Date().toISOString(),
      format: "zip",
      export_kind: "figure_variants",
      selected_idx: idx,
      image_shape: { rows: height, cols: width },
      display: {
        cmap,
        log_scale: logScale,
        auto_contrast: autoContrast,
        vmin_pct: imageVminPct,
        vmax_pct: imageVmaxPct,
      },
      variants,
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));

    for (const v of variants) {
      const figCanvas = exportFigure({ ...baseOpts, showScaleBar: v.showScaleBar && hasScale, showColorbar: v.showColorbar });
      const pngBlob = await new Promise<Blob>((resolve) => figCanvas.toBlob((b) => resolve(b!), "image/png"));
      zip.file(`${prefix}_${v.name}.png`, pngBlob);
      const pdfBlob = await canvasToPDF(figCanvas);
      zip.file(`${prefix}_${v.name}.pdf`, pdfBlob);
    }

    const blob = await zip.generateAsync({ type: "blob" });
    downloadBlob(blob, `${prefix}_all.zip`);
  }, [isGallery, selectedIdx, labels, width, height, cmap, logScale, autoContrast, imageDataRange, imageVminPct, imageVmaxPct, pixelSize, title, roiActive, roiList, profileActive, profilePoints, widgetVersion, lockExport]);

  // Resize Handlers
  // -------------------------------------------------------------------------
  const handleCanvasResizeStart = (e: React.MouseEvent) => {
    if (lockView) return;
    e.stopPropagation();
    e.preventDefault();
    setIsResizingCanvas(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: canvasSize });
  };

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    let rafId = 0;
    let latestSize = resizeStart ? resizeStart.size : canvasSize;

    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      latestSize = Math.max(200, Math.min(600, resizeStart.size + delta));
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
      setIsResizingCanvas(false);
      setResizeStart(null);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingCanvas, resizeStart]);

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

  // -------------------------------------------------------------------------
  // Keyboard shortcuts
  // -------------------------------------------------------------------------
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Number keys 1-9 select gallery images (avoids arrow key conflicts with Jupyter)
    if (!lockNavigation && isGallery && e.key >= "1" && e.key <= "9") {
      const idx = parseInt(e.key) - 1;
      if (idx < nImages) { e.preventDefault(); setSelectedIdx(idx); }
      return;
    }
    switch (e.key) {
      case "ArrowLeft":
        if (!lockNavigation && isGallery) { e.preventDefault(); setSelectedIdx(Math.max(0, selectedIdx - 1)); }
        break;
      case "ArrowRight":
        if (!lockNavigation && isGallery) { e.preventDefault(); setSelectedIdx(Math.min(nImages - 1, selectedIdx + 1)); }
        break;
      case "r":
      case "R":
        if (!lockView) handleResetAll();
        break;
      case "m":
      case "M":
        if (measureActive) {
          setMeasureActive(false);
          setMeasurePoints([]);
        } else {
          setMeasureActive(true);
          setMeasurePoints([]);
        }
        break;
      case "Escape":
        if (measureActive) {
          setMeasureActive(false);
          setMeasurePoints([]);
        }
        break;
      case "Delete":
      case "Backspace":
        if (!lockRoi && roiActive && roiSelectedIdx >= 0 && roiList && roiSelectedIdx < roiList.length) {
          e.preventDefault();
          const newList = roiList.filter((_, i) => i !== roiSelectedIdx);
          setRoiList(newList);
          setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1);
        }
        break;
    }
  };

  // -------------------------------------------------------------------------
  // Render (Show3D-style layout)
  // -------------------------------------------------------------------------
  const needsReset = getZoomState(isGallery ? selectedIdx : 0).zoom !== 1 || getZoomState(isGallery ? selectedIdx : 0).panX !== 0 || getZoomState(isGallery ? selectedIdx : 0).panY !== 0;
  const statsIdx = isGallery ? selectedIdx : 0;

  // Calibrated cursor position
  const calibratedUnit = pixelSize > 0 ? (Math.max(height, width) * pixelSize >= 10 ? "nm" : "Å") : "";
  const calibratedFactor = calibratedUnit === "nm" ? pixelSize / 10 : pixelSize;

  return (
    <Box className="show2d-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, overflow: "visible" }}>
      <Stack direction="row" spacing={`${SPACING.LG}px`} alignItems="flex-start">
        {/* Main panel */}
        <Box sx={{ width: galleryGridWidth }}>
          {/* Title row */}
          <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block", height: 16, lineHeight: "16px", overflow: "hidden" }}>
            {title || (isGallery ? "Gallery" : "Image")}
            <InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum (Fourier transform) alongside image.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on image to draw a line intensity profile.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>ROI: Region of Interest — click to place, drag to move.</Typography>
              {!isGallery && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Lens: Magnifier inset that follows the cursor.</Typography>}
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile-based contrast (2nd–98th percentile). FFT Auto masks DC + clips to 99.9th.</Typography>
              {isGallery && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Link Zoom / Contrast: Sync zoom or histogram range across all gallery images.</Typography>}
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
              <KeyboardShortcuts items={isGallery ? [["← / →", "Prev / Next image"], ["1 – 9", "Select image"], ["Del / ⌫", "Delete selected ROI"], ["M", "Measure distance"], ["Esc", "Exit measure"], ["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]] : [["Del / ⌫", "Delete selected ROI"], ["M", "Measure distance"], ["Esc", "Exit measure"], ["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
            </Box>} theme={themeInfo.theme} />
            <ControlCustomizer
              widgetName="Show2D"
              hiddenTools={hiddenTools}
              setHiddenTools={setHiddenTools}
              disabledTools={disabledTools}
              setDisabledTools={setDisabledTools}
              themeColors={themeColors}
            />
          </Typography>
          {/* Controls row: Profile, ROI, Lens, FFT, Export, Reset, Copy */}
          <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            {!hideProfile && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Profile:</Typography>
                <Switch
                  checked={profileActive}
                  disabled={lockProfile}
                  onChange={(e) => {
                    if (lockProfile) return;
                    const on = e.target.checked;
                    setProfileActive(on);
                    if (on) {
                      if (!lockRoi) setRoiActive(false);
                    } else {
                      setProfilePoints([]);
                      setProfileDataAll([]);
                      setHoveredProfileEndpoint(null);
                      setIsHoveringProfileLine(false);
                    }
                  }}
                  size="small"
                  sx={switchStyles.small}
                />
              </>
            )}
            {!hideRoi && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>ROI:</Typography>
                <Switch
                  checked={roiActive}
                  disabled={lockRoi}
                  onChange={(e) => {
                    if (lockRoi) return;
                    const on = e.target.checked;
                    setRoiActive(on);
                    if (on) {
                      if (!lockProfile) setProfileActive(false);
                      setProfilePoints([]);
                      setProfileDataAll([]);
                      setHoveredProfileEndpoint(null);
                      setIsHoveringProfileLine(false);
                    } else {
                      setRoiSelectedIdx(-1);
                    }
                  }}
                  size="small"
                  sx={switchStyles.small}
                />
              </>
            )}
            {!hideDisplay && (
              <>
                {!isGallery && (
                  <>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Lens:</Typography>
                    <Switch
                      checked={showLens}
                      onChange={() => {
                        if (lockDisplay) return;
                        if (!showLens) {
                          setShowLens(true);
                          setLensPos({ row: Math.floor(height / 2), col: Math.floor(width / 2) });
                        } else {
                          setShowLens(false);
                          setLensPos(null);
                        }
                      }}
                      disabled={lockDisplay}
                      size="small"
                      sx={switchStyles.small}
                    />
                  </>
                )}
                <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
                <Switch
                  checked={showFft}
                  onChange={(e) => { if (!lockDisplay) setShowFft(e.target.checked); }}
                  disabled={lockDisplay}
                  size="small"
                  sx={switchStyles.small}
                />
              </>
            )}
            <Box sx={{ flex: 1 }} />
            {!hideView && (
              <Button size="small" sx={compactButton} disabled={lockView || !needsReset} onClick={handleResetAll}>Reset</Button>
            )}
            {!hideExport && (
              <>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }}>Export</Button>
                <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                  <MenuItem disabled={lockExport} onClick={() => handleExportFigure(true, true)} sx={{ fontSize: 12 }}>Figure + scalebar + colorbar</MenuItem>
                  <MenuItem disabled={lockExport} onClick={() => handleExportFigure(true, false)} sx={{ fontSize: 12 }}>Figure + scalebar</MenuItem>
                  <MenuItem disabled={lockExport} onClick={() => handleExportFigure(false, false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                  <MenuItem disabled={lockExport} onClick={handleExportAll} sx={{ fontSize: 12 }}>All (PNG + PDF)</MenuItem>
                </Menu>
                <Button size="small" sx={compactButton} disabled={lockExport} onClick={handleCopy}>Copy</Button>
              </>
            )}
          </Stack>

          {isGallery ? (
            /* Gallery mode */
            <Box sx={{ display: "grid", gridTemplateColumns: `repeat(${ncols}, auto)`, gap: 1 }}>
              {Array.from({ length: nImages }).map((_, i) => (
                <Box key={i} sx={{ cursor: i === selectedIdx ? ((isDraggingResize || isDraggingResizeInner || isHoveringResize || isHoveringResizeInner) ? "nwse-resize" : isDraggingROI ? "move" : (draggingProfileEndpoint !== null || isDraggingProfileLine) ? "grabbing" : (profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine)) ? "grab" : (profileActive || roiActive || measureActive) ? "crosshair" : "grab") : (lockNavigation ? "default" : "pointer") }}>
                  <Box
                    ref={(el: HTMLDivElement | null) => { imageContainerRefs.current[i] = el; }}
                    sx={{ position: "relative", bgcolor: "#000", border: `2px solid ${i === selectedIdx ? themeColors.accent : themeColors.border}`, borderRadius: 0 }}
                    onMouseDown={(e) => handleMouseDown(e, i)}
                    onMouseMove={(e) => handleMouseMove(e, i)}
                    onMouseUp={(e) => handleMouseUp(e, i)}
                    onMouseLeave={() => handleMouseLeave(i)}
                    onWheel={(i === selectedIdx || linkedZoom) ? (e) => handleWheel(e, i) : undefined}
                    onDoubleClick={() => handleDoubleClick(i)}
                  >
                    <canvas
                      ref={(el) => { if (el && canvasRefs.current[i] !== el) { canvasRefs.current[i] = el; setCanvasReady(c => c + 1); } }}
                      width={canvasW} height={canvasH}
                      style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }}
                    />
                    <canvas
                      ref={(el) => { overlayRefs.current[i] = el; }}
                      width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)}
                      style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
                    />
                    {!hideView && (
                      <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.3 : 0.6, pointerEvents: lockView ? "none" : "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: lockView ? 0.3 : 1 } }} />
                    )}
                  </Box>
                  <Typography sx={{ fontSize: 10, color: themeColors.textMuted, textAlign: "center", mt: 0.25 }}>
                    {labels?.[i] || `Image ${i + 1}`}
                  </Typography>
                  {effectiveShowFft && (
                    <Box
                      ref={(el: HTMLDivElement | null) => { fftContainerRefs.current[i] = el; }}
                      sx={{ mt: 0.5, border: `2px solid ${i === selectedIdx ? themeColors.accent : themeColors.border}`, borderRadius: 0, bgcolor: "#000", cursor: lockView ? "default" : "grab" }}
                      onWheel={(i === selectedIdx || linkedZoom) ? (e) => handleGalleryFftWheel(e, i) : undefined}
                      onDoubleClick={() => setGalleryFftState(i, { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 })}
                      onMouseDown={(e) => handleGalleryFftMouseDown(e, i)}
                      onMouseMove={(e) => handleGalleryFftMouseMove(e, i)}
                      onMouseUp={handleGalleryFftMouseUp}
                      onMouseLeave={handleGalleryFftMouseUp}
                    >
                      <canvas
                        ref={(el) => { fftCanvasRefs.current[i] = el; }}
                        width={canvasW} height={canvasH}
                        style={{ width: canvasW, height: canvasH, imageRendering: "pixelated", display: "block" }}
                      />
                    </Box>
                  )}
                </Box>
              ))}
            </Box>
          ) : (
            /* Single image mode */
            <Box
              ref={(el: HTMLDivElement | null) => { imageContainerRefs.current[0] = el; }}
              sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, width: canvasW, height: canvasH, cursor: isHoveringLensEdge ? "nwse-resize" : isDraggingROI ? "move" : (isDraggingResize || isDraggingResizeInner || isHoveringResize || isHoveringResizeInner) ? "nwse-resize" : (draggingProfileEndpoint !== null || isDraggingProfileLine) ? "grabbing" : (profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine)) ? "grab" : (profileActive || roiActive || measureActive) ? "crosshair" : "grab" }}
              onMouseDown={(e) => handleMouseDown(e, 0)}
              onMouseMove={(e) => handleMouseMove(e, 0)}
              onMouseUp={(e) => handleMouseUp(e, 0)}
              onMouseLeave={() => handleMouseLeave(0)}
              onWheel={(e) => handleWheel(e, 0)}
              onDoubleClick={() => handleDoubleClick(0)}
            >
              <canvas
                ref={(el) => { if (el && canvasRefs.current[0] !== el) { canvasRefs.current[0] = el; setCanvasReady(c => c + 1); } }}
                width={canvasW} height={canvasH}
                style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }}
              />
              <canvas
                ref={(el) => { overlayRefs.current[0] = el; }}
                width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)}
                style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
              />
              <canvas
                ref={lensCanvasRef}
                width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)}
                style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
              />
              {cursorInfo && (
                <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                  <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                    ({cursorInfo.row}, {cursorInfo.col}){pixelSize > 0 ? ` = (${(cursorInfo.row * calibratedFactor).toFixed(1)}, ${(cursorInfo.col * calibratedFactor).toFixed(1)} ${calibratedUnit})` : ""} {formatNumber(cursorInfo.value)}
                  </Typography>
                </Box>
              )}
              {!hideView && (
                <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.3 : 0.6, pointerEvents: lockView ? "none" : "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: lockView ? 0.3 : 1 } }} />
              )}
            </Box>
          )}

          {/* Stats bar - right below canvas (Show3D style) */}
          {!hideStats && showStats && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", boxSizing: "border-box", overflow: "hidden", whiteSpace: "nowrap", opacity: lockStats ? 0.7 : 1 }}>
              {isGallery && (
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>{labels?.[statsIdx] || `#${statsIdx + 1}`}</Typography>
              )}
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMean?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMin?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMax?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsStd?.[statsIdx] ?? 0)}</Box></Typography>
              {measureActive && (
                <>
                  <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                  <Typography sx={{ fontSize: 11, color: "#fff", fontWeight: "bold" }}>Measuring</Typography>
                </>
              )}
            </Box>
          )}

          {/* Gallery FFT Controls - below gallery grid */}
          {effectiveShowFft && isGallery && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, boxSizing: "border-box" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT Scale:</Typography>
                  <Select disabled={lockDisplay} value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                    <MenuItem value="power">Pow</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                  <Switch checked={fftAuto} onChange={(e) => { if (!lockDisplay) setFftAuto(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                  <Select disabled={lockDisplay} value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                </Box>
              </Box>
              {!hideHistogram && (
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
                  {fftHistogramData && (
                    <Histogram data={fftHistogramData} vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { if (!lockHistogram) { setFftVminPct(min); setFftVmaxPct(max); } }} width={110} height={58} theme={themeInfo.theme === "dark" ? "dark" : "light"} dataMin={fftDataRange.min} dataMax={fftDataRange.max} />
                  )}
                </Box>
              )}
            </Box>
          )}

          {/* Line profile sparkline — always reserve space when profile is active */}
          {!hideProfile && profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, maxWidth: profileCanvasWidth, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                onMouseMove={handleProfileMouseMove}
                onMouseLeave={handleProfileMouseLeave}
                style={{ width: profileCanvasWidth, height: profileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
              />
              <div
                onMouseDown={(e) => {
                  if (lockProfile) return;
                  e.preventDefault();
                  setIsResizingProfile(true);
                  setProfileResizeStart({ y: e.clientY, height: profileHeight });
                }}
                style={{ width: profileCanvasWidth, height: 4, cursor: lockProfile ? "default" : "ns-resize", borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, background: `linear-gradient(to bottom, ${themeColors.border}, transparent)`, opacity: lockProfile ? 0.5 : 1, pointerEvents: lockProfile ? "none" : "auto" }}
              />
            </Box>
          )}

          {/* Controls: two rows left + histogram right, ROI below */}
          {showControls && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, boxSizing: "border-box" }}>
              {/* Top: control rows + histogram side by side */}
              <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
                <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                  {/* Row 1: Scale + Color */}
                  {!hideDisplay && (
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                      <Select disabled={lockDisplay} value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45 }} MenuProps={themedMenuProps}>
                        <MenuItem value="linear">Lin</MenuItem>
                        <MenuItem value="log">Log</MenuItem>
                      </Select>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                      <Select disabled={lockDisplay} size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60 }}>
                        {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                      </Select>
                      {!isGallery && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Colorbar:</Typography>
                          <Switch checked={showColorbar} onChange={() => { if (!lockDisplay) setShowColorbar(!showColorbar); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                        </>
                      )}
                    </Box>
                  )}
                  {/* Row 2: Auto + Lens settings + Link Zoom (gallery) + zoom indicator */}
                  {!hideDisplay && (
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                      <Switch checked={autoContrast} onChange={() => { if (!lockDisplay) setAutoContrast(!autoContrast); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                      {!isGallery && showLens && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Lens {lensMag}×</Typography>
                          <Slider disabled={lockDisplay} value={lensMag} min={2} max={8} step={1} onChange={(_, v) => setLensMag(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} />
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>{lensDisplaySize}px</Typography>
                          <Slider disabled={lockDisplay} value={lensDisplaySize} min={64} max={256} step={16} onChange={(_, v) => setLensDisplaySize(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} />
                        </>
                      )}
                      {isGallery && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Link Zoom</Typography>
                          <Switch checked={linkedZoom} onChange={() => { if (!lockDisplay) setLinkedZoom(!linkedZoom); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Link Contrast</Typography>
                          <Switch checked={linkedContrast} onChange={() => { if (!lockDisplay) setLinkedContrast(!linkedContrast); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                        </>
                      )}
                      {getZoomState(isGallery ? selectedIdx : 0).zoom !== 1 && (
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold" }}>{getZoomState(isGallery ? selectedIdx : 0).zoom.toFixed(1)}x</Typography>
                      )}
                    </Box>
                  )}
                </Box>
                {/* Right: Histogram aligned to the two rows */}
                {!hideHistogram && imageHistogramData && (
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
                    <Histogram data={imageHistogramData} vminPct={imageVminPct} vmaxPct={imageVmaxPct} onRangeChange={(min, max) => { if (!lockHistogram) setContrastState(activeContrastIdx, { vminPct: min, vmaxPct: max }); }} width={110} height={58} theme={themeInfo.theme === "dark" ? "dark" : "light"} dataMin={imageDataRange.min} dataMax={imageDataRange.max} />
                  </Box>
                )}
              </Box>
              {/* ROI Section (own box, below control rows) */}
              {!hideRoi && roiActive && (
                <Box sx={{ border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, px: 1, py: 0.5, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, opacity: lockRoi ? 0.5 : 1, pointerEvents: lockRoi ? "none" : "auto" }}>
                  {/* ROI: shape + ADD + CLEAR */}
                  <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>ROI:</Typography>
                    <Select
                      size="small"
                      value={newRoiShape}
                      onChange={(e) => setNewRoiShape(e.target.value as "circle" | "square" | "rectangle" | "annular")}
                      MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                    >
                      {(["square", "rectangle", "circle", "annular"] as const).map((s) => (<MenuItem key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</MenuItem>))}
                    </Select>
                    <Button size="small" sx={compactButton} onClick={() => {
                      const defR = Math.max(10, Math.round(Math.min(width, height) * 0.05));
                      const newRoi: ROIItem = { row: Math.floor(height / 2), col: Math.floor(width / 2), shape: newRoiShape, radius: defR, radius_inner: Math.max(5, Math.round(defR * 0.5)), width: defR * 2, height: defR * 2, color: ROI_COLORS[(roiList?.length ?? 0) % ROI_COLORS.length], line_width: 2, highlight: false };
                      const newList = [...(roiList || []), newRoi];
                      setRoiList(newList);
                      setRoiSelectedIdx(newList.length - 1);
                    }}>ADD</Button>
                    <Box sx={{ flex: 1 }} />
                    <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 24, color: "#ef5350" }} disabled={!roiList?.length} onClick={() => { setRoiList([]); setRoiSelectedIdx(-1); }}>CLEAR</Button>
                  </Box>
                  {/* Selected ROI details */}
                  {selectedRoi && (
                    <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: `${SPACING.SM}px`, borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: selectedRoi.color }}>#{roiSelectedIdx + 1}/{roiList?.length ?? 0}</Typography>
                      <Select
                        size="small"
                        value={selectedRoi.shape || "circle"}
                        onChange={(e) => updateSelectedRoi({ shape: e.target.value })}
                        MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                      >
                        {(["square", "rectangle", "circle", "annular"] as const).map((s) => (<MenuItem key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</MenuItem>))}
                      </Select>
                      {selectedRoi.shape === "rectangle" && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>W</Typography>
                          <Slider value={selectedRoi.width} min={5} max={width} onChange={(_, v) => updateSelectedRoi({ width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>H</Typography>
                          <Slider value={selectedRoi.height} min={5} max={height} onChange={(_, v) => updateSelectedRoi({ height: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                        </>
                      )}
                      {selectedRoi.shape === "annular" && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Inner</Typography>
                          <Slider value={selectedRoi.radius_inner} min={1} max={selectedRoi.radius - 1} onChange={(_, v) => updateSelectedRoi({ radius_inner: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Outer</Typography>
                          <Slider value={selectedRoi.radius} min={selectedRoi.radius_inner + 1} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                        </>
                      )}
                      {selectedRoi.shape !== "rectangle" && selectedRoi.shape !== "annular" && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Size</Typography>
                          <Slider value={selectedRoi.radius} min={5} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 50 }} />
                        </>
                      )}
                      <Box sx={{ display: "flex", gap: "2px" }}>
                        {ROI_COLORS.map(c => (
                          <Box key={c} onClick={() => updateSelectedRoi({ color: c })} sx={{ width: 12, height: 12, bgcolor: c, cursor: "pointer", border: c === selectedRoi.color ? `2px solid ${themeColors.text}` : "1px solid transparent", "&:hover": { opacity: 0.8 } }} />
                        ))}
                      </Box>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Border</Typography>
                      <Slider value={selectedRoi.line_width} min={1} max={6} step={1} onChange={(_, v) => updateSelectedRoi({ line_width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 30 }} />
                      <Box
                        onClick={() => updateSelectedRoi({ highlight: !selectedRoi.highlight })}
                        sx={{ cursor: "pointer", fontSize: 10, color: selectedRoi.highlight ? themeColors.accentGreen : themeColors.textMuted, "&:hover": { opacity: 0.8 } }}
                        title="Focus (dim outside)"
                      >{selectedRoi.highlight ? "\u25C9 Focus" : "\u25CB Focus"}</Box>
                      <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 20, color: "#ef5350" }} onClick={() => {
                        const newList = roiList!.filter((_, j) => j !== roiSelectedIdx);
                        setRoiList(newList);
                        setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1);
                      }}>&times;</Button>
                    </Box>
                  )}
                  {/* ROI list */}
                  {roiList && roiList.length > 0 && (
                    <Box sx={{ display: "flex", flexDirection: "column", borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                      {roiList.map((roi, i) => {
                        const c = roi.color || ROI_COLORS[i % ROI_COLORS.length];
                        const isSelected = i === roiSelectedIdx;
                        const shapeLabel = roi.shape === "rectangle" ? `${roi.width}×${roi.height}` : roi.shape === "annular" ? `r${roi.radius_inner}-${roi.radius}` : `r${roi.radius}`;
                        return (
                          <Box key={i} onClick={() => setRoiSelectedIdx(i)} sx={{ display: "flex", alignItems: "center", gap: "3px", lineHeight: 1.6, cursor: "pointer", "&:hover .roi-delete": { opacity: 1 } }}>
                            <Box sx={{ width: 8, height: 8, borderRadius: roi.shape === "square" || roi.shape === "rectangle" ? 0 : "50%", bgcolor: c, border: isSelected ? "2px solid #fff" : "1px solid transparent", flexShrink: 0 }} />
                            <Typography component="span" sx={{ fontSize: 10, fontFamily: "monospace", color: isSelected ? themeColors.text : themeColors.textMuted, fontWeight: isSelected ? "bold" : "normal" }}>
                              <Box component="span" sx={{ color: c }}>{i + 1}</Box>{" "}
                              {roi.shape} ({roi.row}, {roi.col}) {shapeLabel}
                            </Typography>
                            <Box
                              onClick={(e) => { e.stopPropagation(); const newList = roiList.map((r, j) => ({ ...r, highlight: j === i ? !r.highlight : false })); setRoiList(newList); }}
                              sx={{ cursor: "pointer", fontSize: 10, color: roi.highlight ? themeColors.accentGreen : themeColors.textMuted, lineHeight: 1, opacity: roi.highlight ? 1 : 0.5, "&:hover": { opacity: 1 } }}
                              title="Focus (dim outside)"
                            >{roi.highlight ? "\u25C9" : "\u25CB"}</Box>
                            <Box
                              className="roi-delete"
                              onClick={(e) => { e.stopPropagation(); const newList = roiList.filter((_, j) => j !== i); setRoiList(newList); setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1); }}
                              sx={{ opacity: 0, cursor: "pointer", fontSize: 10, color: themeColors.textMuted, ml: 0.5, lineHeight: 1, "&:hover": { color: "#f44336" } }}
                            >&times;</Box>
                          </Box>
                        );
                      })}
                    </Box>
                  )}
                </Box>
              )}
            </Box>
          )}
        </Box>

        {/* FFT Panel - canvas + stats (single mode only) */}
        {effectiveShowFft && !isGallery && (
          <Box sx={{ width: canvasW }}>
            {/* Spacer — matches main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {/* Controls row — matches main panel controls row height */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              {fftCropDims ? (
                <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: themeColors.accentGreen }}>
                  ROI FFT ({fftCropDims.cropWidth}&times;{fftCropDims.cropHeight})
                </Typography>
              ) : <Box />}
              {!hideView && (
                <Button size="small" sx={compactButton} disabled={lockView || (fftZoom === DEFAULT_FFT_ZOOM && fftPanX === 0 && fftPanY === 0)} onClick={handleFftDoubleClick}>Reset</Button>
              )}
            </Stack>
            <Box
              ref={singleFftContainerRef}
              sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, cursor: lockView ? "default" : "crosshair", width: canvasW, height: canvasH }}
              onWheel={lockView ? undefined : handleFftWheel}
              onDoubleClick={lockView ? undefined : handleFftDoubleClick}
              onMouseDown={lockView ? undefined : handleFftMouseDown}
              onMouseMove={lockView ? undefined : handleFftMouseMove}
              onMouseUp={lockView ? undefined : handleFftMouseUp}
              onMouseLeave={handleFftMouseLeave}
            >
              <canvas ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
              <canvas ref={fftOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
              {!hideView && (
                <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.3 : 0.6, pointerEvents: lockView ? "none" : "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: lockView ? 0.3 : 1 } }} />
              )}
            </Box>
            {/* FFT Stats Bar */}
            {!hideStats && fftStats && fftStats.length === 4 && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2 }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[0])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[1])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[2])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[3])}</Box></Typography>
                {fftClickInfo && (
                  <>
                    <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                      {fftClickInfo.dSpacing != null ? (
                        <>d = <Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} Å`}</Box>{" | |g| = "}<Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.spatialFreq!.toFixed(4)} Å⁻¹</Box></>
                      ) : (
                        <>dist = <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                      )}
                    </Typography>
                  </>
                )}
              </Box>
            )}
            {/* FFT Controls - two rows + histogram (matching main panel layout) */}
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: canvasW, boxSizing: "border-box" }}>
              <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
                <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                  {/* Row 1: Scale + Color + Colorbar */}
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                    <Select disabled={lockDisplay} value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                      <MenuItem value="power">Pow</MenuItem>
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                    <Select disabled={lockDisplay} value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                      {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Colorbar:</Typography>
                    <Switch checked={fftShowColorbar} onChange={(e) => { if (!lockDisplay) setFftShowColorbar(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                  </Box>
                  {/* Row 2: Auto + zoom indicator */}
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                    <Switch checked={fftAuto} onChange={(e) => { if (!lockDisplay) setFftAuto(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                    {fftZoom !== DEFAULT_FFT_ZOOM && (
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold" }}>{fftZoom.toFixed(1)}x</Typography>
                    )}
                  </Box>
                </Box>
                {/* Right: FFT Histogram */}
                {!hideHistogram && (
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
                    {fftHistogramData && (
                      <Histogram data={fftHistogramData} vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { if (!lockHistogram) { setFftVminPct(min); setFftVmaxPct(max); } }} width={110} height={58} theme={themeInfo.theme === "dark" ? "dark" : "light"} dataMin={fftDataRange.min} dataMax={fftDataRange.max} />
                    )}
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

export const render = createRender(Show2D);
