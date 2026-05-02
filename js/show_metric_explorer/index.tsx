/**
 * MetricExplorer - Interactive metric explorer for SSB aberration parameter sweeps.
 *
 * Layout:
 *   Row 1: metric charts (resizable, proportional)
 *   Row 2: selected image (zoom/pan, FFT toggle) → reference images (resizable) → info
 *
 * Features:
 * - Hover/click chart point → crosshair on all charts + image updates
 * - Best-point star (per group) + diamond (global best) markers
 * - Scroll-to-zoom + drag-to-pan on selected image
 * - FFT toggle with Hann window + auto-enhance
 * - Histogram + contrast slider on selected image
 * - ← → keyboard navigation
 * - Drag bottom-right corner to resize charts and images
 * - Colormap selector, light/dark theme
 */

import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
// import Button removed (unused)
import Stack from "@mui/material/Stack";
import IconButton from "@mui/material/IconButton";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import StopIcon from "@mui/icons-material/Stop";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import FastForwardIcon from "@mui/icons-material/FastForward";
import "./show_metric_explorer.css";
import { useTheme } from "../theme";
import { roundToNiceValue } from "../scalebar";
import { extractFloat32, formatNumber } from "../format";
import { findDataRange, sliderRange, applyLogScale } from "../stats";
import { COLORMAPS, COLORMAP_NAMES, applyColormap } from "../colormaps";
import { fft2d, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2, applyHannWindow2D } from "../webgpu-fft";
import { computeHistogramFromBytes } from "../histogram";

// ============================================================================
// UI Styles (matching Show4DSTEM)
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
};



const SPACING = {
  XS: 4,
  SM: 8,
  MD: 12,
  LG: 16,
};

const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
};

const DPR = window.devicePixelRatio || 1;

// ============================================================================
// Default sizes
// ============================================================================
const DEFAULT_CHART_W = 340;
const DEFAULT_CHART_H = 200;

const MARGIN = { top: 24, right: 12, bottom: 34, left: 52 };
const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const AXIS_TICK_PX = 4;
const MIN_CHART_W = 180;
const MIN_CHART_H = 120;
const MIN_IMG_SIZE = 120;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

const GROUP_COLORS = [
  "#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350",
  "#ffd54f", "#90a4ae", "#a1887f", "#4dd0e1", "#aed581",
];

// ============================================================================
// Types
// ============================================================================
interface PointData {
  metrics: Record<string, number>;
  params: Record<string, number | string>;
  label: string;
  groupKey: string;
  index: number;
}

interface GroupData {
  key: string;
  color: string;
  points: PointData[];
  bestIndices: Record<string, number>;
}

// ============================================================================
// Helpers
// ============================================================================
function computeTicks(min: number, max: number, maxTicks: number = 6): number[] {
  const range = max - min;
  if (range <= 0 || !isFinite(range)) return [min];
  const step = roundToNiceValue(range / maxTicks);
  if (step <= 0 || !isFinite(step)) return [min, max];
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 0.001; v += step) {
    if (v >= min - step * 0.001) ticks.push(v);
  }
  if (ticks.length === 0) ticks.push(min, max);
  return ticks;
}

function snap(v: number): number {
  return Math.round(v) + 0.5;
}

function drawStar(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  const inner = r * 0.4;
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const angle = (i * Math.PI) / 5 - Math.PI / 2;
    const radius = i % 2 === 0 ? r : inner;
    ctx.lineTo(cx + radius * Math.cos(angle), cy + radius * Math.sin(angle));
  }
  ctx.closePath();
}

function drawDiamond(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx + r, cy);
  ctx.lineTo(cx, cy + r);
  ctx.lineTo(cx - r, cy);
  ctx.closePath();
}

// ============================================================================
// Resize handle
// ============================================================================
function ResizeHandle({ onStart, accentColor }: {
  onStart: (e: React.MouseEvent) => void;
  accentColor: string;
}) {
  return (
    <Box
      onMouseDown={onStart}
      sx={{
        position: "absolute", bottom: 0, right: 0, width: 14, height: 14,
        cursor: "nwse-resize", opacity: 0.5,
        background: `linear-gradient(135deg, transparent 50%, ${accentColor} 50%)`,
        "&:hover": { opacity: 1 },
      }}
    />
  );
}

// ============================================================================
// Mini Histogram (matching Show2D pattern)
// ============================================================================
function MiniHistogram({ data, vminPct, vmaxPct, onRangeChange, width = 120, height = 36, theme = "dark", dataMin = 0, dataMax = 1 }: {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);
  const isDark = theme === "dark";
  const barColors = isDark
    ? { bg: "#1a1a1a", active: "#888", inactive: "#444", border: "#333" }
    : { bg: "#f0f0f0", active: "#666", inactive: "#bbb", border: "#ccc" };

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = DPR;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = barColors.bg;
    ctx.fillRect(0, 0, width, height);
    const nBins = 64;
    const ratio = Math.floor(bins.length / nBins);
    const reduced: number[] = [];
    for (let i = 0; i < nBins; i++) {
      let sum = 0;
      for (let j = 0; j < ratio; j++) sum += bins[i * ratio + j] || 0;
      reduced.push(sum / ratio);
    }
    const maxVal = Math.max(...reduced, 0.001);
    const barW = width / nBins;
    const lo = Math.floor((vminPct / 100) * nBins);
    const hi = Math.floor((vmaxPct / 100) * nBins);
    for (let i = 0; i < nBins; i++) {
      const bh = (reduced[i] / maxVal) * (height - 2);
      ctx.fillStyle = (i >= lo && i <= hi) ? barColors.active : barColors.inactive;
      ctx.fillRect(i * barW + 0.5, height - bh, Math.max(1, barW - 1), bh);
    }
  }, [bins, vminPct, vmaxPct, width, height, barColors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas ref={canvasRef} style={{ width, height, border: `1px solid ${barColors.border}` }} />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => { const [a, b] = v as number[]; onRangeChange(Math.min(a, b - 1), Math.max(b, a + 1)); }}
        min={0} max={100} size="small" valueLabelDisplay="auto"
        valueLabelFormat={(pct) => { const val = dataMin + (pct / 100) * (dataMax - dataMin); return val >= 1000 ? val.toExponential(1) : val.toFixed(1); }}
        sx={{ width, py: 0, "& .MuiSlider-thumb": { width: 8, height: 8 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>
          {(() => { const v = dataMin + (vminPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}
        </Typography>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>
          {(() => { const v = dataMin + (vmaxPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}
        </Typography>
      </Box>
    </Box>
  );
}

// ============================================================================
// Main Component
// ============================================================================
function MetricExplorer() {
  const { themeInfo, colors } = useTheme();
  const isDark = themeInfo.theme === "dark";
  const model = useModel();

  // Model state
  const [pointsData] = useModelState<DataView>("points_bytes");
  const [nPoints] = useModelState<number>("n_points");
  const [imgHeight] = useModelState<number>("height");
  const [imgWidth] = useModelState<number>("width");
  const [metricsJson] = useModelState<string>("metrics_json");
  const [paramsJson] = useModelState<string>("params_json");
  const [pointLabels] = useModelState<string[]>("labels");
  const [xKey] = useModelState<string>("x_key");
  const [xLabel] = useModelState<string>("x_label");
  const [groupKey] = useModelState<string>("group_key");
  const [metricNames] = useModelState<string[]>("metric_names");
  const [metricLabelsArr] = useModelState<string[]>("metric_labels");
  const [metricDirections] = useModelState<string[]>("metric_directions");
  const [selectedIndex, setSelectedIndex] = useModelState<number>("selected_index");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelUnit] = useModelState<string>("pixel_unit");

  // Local state
  const [hoveredIndex, setHoveredIndex] = React.useState<number | null>(null);
  // Charts and image resize independently
  const [chartScale, setChartScale] = React.useState(1.0);
  const [imgSize, setImgSize] = React.useState(450);
  const chartW = Math.max(MIN_CHART_W, Math.round(DEFAULT_CHART_W * chartScale));
  const chartH = Math.max(MIN_CHART_H, Math.round(DEFAULT_CHART_H * chartScale));
  const [showFft, setShowFft] = React.useState(false);
  const [logScale, setLogScale] = React.useState(false);
  const [hiddenGroups, setHiddenGroups] = React.useState<Set<string>>(new Set());
  const [, setHiddenMetrics] = React.useState<Set<string>>(new Set());
  const [playing, setPlaying] = React.useState(false);
  const [playReverse, setPlayReverse] = React.useState(false);
  const [playFps, setPlayFps] = React.useState(3);
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; val: number } | null>(null);
  const [copyFeedback, setCopyFeedback] = React.useState(false);
  // FFT contrast: one pair per slot (0 = selected, 1..N = refs)
  const [fftContrast, setFftContrast] = React.useState<{ vmin: number; vmax: number }[]>([]);
  const [fftLinked] = React.useState(true);
  // D-spacing measurements (multiple spots)
  interface DSpacingMark {
    fftIdx: number; col: number; row: number;
    freq: number; dSpacing: number; unit: string;
    friedelCol: number; friedelRow: number;
  }
  const [dSpacingMarks, setDSpacingMarks] = React.useState<DSpacingMark[]>([]);
  const [zoom, setZoom] = React.useState(1.0);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [vminPct, setVminPct] = React.useState(0);
  const [vmaxPct, setVmaxPct] = React.useState(100);

  // Refs
  const chartCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const selectedImgRef = React.useRef<HTMLCanvasElement>(null);
  const rootRef = React.useRef<HTMLDivElement>(null);
  const imgContainerRef = React.useRef<HTMLDivElement>(null);
  const fftSelectedRef = React.useRef<HTMLCanvasElement>(null);
  const chartResizeRef = React.useRef<{ active: boolean; startX: number; startY: number; startScale: number; startW: number } | null>(null);
  const imgResizeRef = React.useRef<{ active: boolean; startX: number; startY: number; startSize: number } | null>(null);
  const panDragRef = React.useRef<{ active: boolean; startX: number; startY: number; startPanX: number; startPanY: number } | null>(null);

  // ========================================================================
  // Window-level drag handling
  // ========================================================================
  React.useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (chartResizeRef.current?.active) {
        const rd = chartResizeRef.current;
        const newW = rd.startW + (e.clientX - rd.startX);
        const newScale = Math.max(0.5, Math.min(3.0, rd.startScale * (newW / rd.startW)));
        setChartScale(newScale);
      }
      if (imgResizeRef.current?.active) {
        const rd = imgResizeRef.current;
        const delta = Math.max(e.clientX - rd.startX, e.clientY - rd.startY);
        setImgSize(Math.max(MIN_IMG_SIZE, rd.startSize + delta));
      }
      if (panDragRef.current?.active) {
        const rd = panDragRef.current;
        setPanX(rd.startPanX + (e.clientX - rd.startX));
        setPanY(rd.startPanY + (e.clientY - rd.startY));
      }
    };
    const handleUp = () => {
      if (chartResizeRef.current?.active) chartResizeRef.current = null;
      if (imgResizeRef.current?.active) imgResizeRef.current = null;
      if (panDragRef.current?.active) panDragRef.current = null;
    };
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, []);

  // Playback interval
  React.useEffect(() => {
    if (!playing || nPoints <= 1) return;
    const interval = setInterval(() => {
      setSelectedIndex((prev: number) => {
        const next = playReverse
          ? (prev <= 0 ? nPoints - 1 : prev - 1)
          : (prev >= nPoints - 1 ? 0 : prev + 1);
        model.set("selected_index", next);
        model.save_changes();
        return next;
      });
    }, 1000 / playFps);
    return () => clearInterval(interval);
  }, [playing, playReverse, playFps, nPoints, model, setSelectedIndex]);

  // Prevent page scroll on image canvas
  React.useEffect(() => {
    const el = imgContainerRef.current;
    if (!el) return;
    const prevent = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", prevent, { passive: false });
    return () => el.removeEventListener("wheel", prevent);
  }, []);

  // ========================================================================
  // Parse metrics and params
  // ========================================================================
  const metrics: Record<string, number>[] = React.useMemo(() => {
    try { return JSON.parse(metricsJson); } catch { return []; }
  }, [metricsJson]);

  const params: Record<string, number | string>[] = React.useMemo(() => {
    try { return JSON.parse(paramsJson); } catch { return []; }
  }, [paramsJson]);

  // ========================================================================
  // Extract point data, group, compute best points + global best
  // ========================================================================
  const { groups, allPoints, sortedIndices, globalBest } = React.useMemo(() => {
    const pts: PointData[] = [];
    for (let i = 0; i < nPoints; i++) {
      pts.push({
        metrics: metrics[i] || {},
        params: params[i] || {},
        label: (pointLabels && pointLabels[i]) || `Point ${i}`,
        groupKey: groupKey && params[i] && params[i][groupKey] !== undefined ? String(params[i][groupKey]) : "",
        index: i,
      });
    }

    const groupMap = new Map<string, PointData[]>();
    for (const pt of pts) {
      const existing = groupMap.get(pt.groupKey);
      if (existing) existing.push(pt);
      else groupMap.set(pt.groupKey, [pt]);
    }

    const grps: GroupData[] = [];
    let ci = 0;
    for (const [key, gpts] of groupMap) {
      if (xKey) {
        gpts.sort((a, b) => {
          const av = typeof a.params[xKey] === "number" ? (a.params[xKey] as number) : 0;
          const bv = typeof b.params[xKey] === "number" ? (b.params[xKey] as number) : 0;
          return av - bv;
        });
      }

      const bestIndices: Record<string, number> = {};
      if (metricNames && metricDirections) {
        for (let mi = 0; mi < metricNames.length; mi++) {
          const mName = metricNames[mi];
          const dir = metricDirections[mi] || "min";
          let bestVal = dir === "min" ? Infinity : -Infinity;
          let bestIdx = gpts[0]?.index ?? 0;
          for (const pt of gpts) {
            const v = pt.metrics[mName];
            if (!isFinite(v)) continue;
            if ((dir === "min" && v < bestVal) || (dir === "max" && v > bestVal)) {
              bestVal = v; bestIdx = pt.index;
            }
          }
          bestIndices[mName] = bestIdx;
        }
      }
      grps.push({ key: key || "default", color: GROUP_COLORS[ci % GROUP_COLORS.length], points: gpts, bestIndices });
      ci++;
    }

    // Global best per metric (across all groups)
    const gb: Record<string, number> = {};
    if (metricNames && metricDirections) {
      for (let mi = 0; mi < metricNames.length; mi++) {
        const mName = metricNames[mi];
        const dir = metricDirections[mi] || "min";
        let bestVal = dir === "min" ? Infinity : -Infinity;
        let bestIdx = 0;
        for (const pt of pts) {
          const v = pt.metrics[mName];
          if (!isFinite(v)) continue;
          if ((dir === "min" && v < bestVal) || (dir === "max" && v > bestVal)) {
            bestVal = v; bestIdx = pt.index;
          }
        }
        gb[mName] = bestIdx;
      }
    }

    const sorted: number[] = [];
    for (const g of grps) for (const pt of g.points) sorted.push(pt.index);
    return { groups: grps, allPoints: pts, sortedIndices: sorted, globalBest: gb };
  }, [nPoints, metrics, params, pointLabels, xKey, groupKey, metricNames, metricDirections]);

  // ========================================================================
  // Image data extraction
  // ========================================================================
  const allImages = React.useMemo(() => {
    if (!pointsData || pointsData.byteLength < 4) return null;
    return extractFloat32(pointsData);
  }, [pointsData]);



  const getImage = React.useCallback((idx: number): Float32Array | null => {
    if (!allImages || idx < 0 || idx >= nPoints) return null;
    const offset = idx * imgHeight * imgWidth;
    return allImages.subarray(offset, offset + imgHeight * imgWidth);
  }, [allImages, nPoints, imgHeight, imgWidth]);


  // ========================================================================
  // Render selected image with zoom/pan/contrast/FFT
  // ========================================================================
  const displayIndex = hoveredIndex !== null ? hoveredIndex : selectedIndex;

  // Display data — always the real image (FFT is in row 3)
  const displayData = React.useMemo((): { data: Float32Array; w: number; h: number } | null => {
    const img = getImage(displayIndex);
    if (!img) return null;
    return { data: img, w: imgWidth, h: imgHeight };
  }, [displayIndex, allImages, imgWidth, imgHeight, getImage]);

  // Global data range across ALL images (stable histogram)
  const dataRange = React.useMemo(() => {
    if (!allImages) return { min: 0, max: 1 };
    return findDataRange(allImages);
  }, [allImages]);

  // Reset contrast when switching images
  React.useEffect(() => {
    setVminPct(0);
    setVmaxPct(100);
  }, [displayIndex]);

  // Reset zoom/pan when switching images
  React.useEffect(() => {
    setZoom(1.0);
    setPanX(0);
    setPanY(0);
  }, [displayIndex]);

  // Render selected image
  React.useEffect(() => {
    const canvas = selectedImgRef.current;
    if (!canvas || !displayData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { data, w, h } = displayData;

    canvas.width = imgSize * DPR;
    canvas.height = imgSize * DPR;
    ctx.scale(DPR, DPR);
    ctx.fillStyle = isDark ? "#111" : "#eee";
    ctx.fillRect(0, 0, imgSize, imgSize);

    const lut = COLORMAPS[cmap] || COLORMAPS["inferno"];
    const renderData = logScale ? applyLogScale(data) : data;
    const renderRange = logScale ? findDataRange(renderData) : dataRange;
    const { vmin, vmax } = sliderRange(renderRange.min, renderRange.max, vminPct, vmaxPct);

    const offscreen = document.createElement("canvas");
    offscreen.width = w; offscreen.height = h;
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;
    const imgData = offCtx.createImageData(w, h);
    applyColormap(renderData, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);

    // Draw with zoom/pan
    const baseScale = Math.min(imgSize / w, imgSize / h);
    const cx = imgSize / 2;
    const cy = imgSize / 2;
    ctx.save();
    ctx.translate(cx + panX, cy + panY);
    ctx.scale(zoom * baseScale, zoom * baseScale);
    ctx.translate(-w / 2, -h / 2);
    ctx.imageSmoothingEnabled = zoom < 2;
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();

    // Zoom indicator
    if (zoom !== 1.0) {
      ctx.fillStyle = isDark ? "rgba(0,0,0,0.6)" : "rgba(255,255,255,0.7)";
      ctx.fillRect(4, imgSize - 18, 42, 14);
      ctx.fillStyle = isDark ? "#fff" : "#000";
      ctx.font = `10px ${FONT}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${zoom.toFixed(1)}×`, 6, imgSize - 5);
    }

    // Cursor readout overlay
    if (cursorInfo) {
      const text = `(${cursorInfo.row}, ${cursorInfo.col}) = ${formatNumber(cursorInfo.val, 3)}`;
      ctx.font = `10px ${FONT}`;
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = isDark ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.8)";
      ctx.fillRect(imgSize - tw - 10, imgSize - 18, tw + 6, 14);
      ctx.fillStyle = isDark ? "#ddd" : "#222";
      ctx.textAlign = "right";
      ctx.textBaseline = "bottom";
      ctx.fillText(text, imgSize - 6, imgSize - 5);
    }
  }, [displayData, imgSize, cmap, isDark, zoom, panX, panY, vminPct, vmaxPct, dataRange, cursorInfo, logScale]);

  // Render reference images

  // ========================================================================
  // Compute and render FFT row (Row 3)
  // ========================================================================
  const computeFFT = React.useCallback((data: Float32Array, w: number, h: number): { mag: Float32Array; pw: number; ph: number } => {
    const real = new Float32Array(data);
    const imag = new Float32Array(data.length);
    applyHannWindow2D(real, w, h);
    fft2d(real, imag, w, h);
    const pw = nextPow2(w), ph = nextPow2(h);
    const mag = computeMagnitude(real, imag);
    fftshift(mag, pw, ph);
    autoEnhanceFFT(mag, pw, ph);
    for (let i = 0; i < mag.length; i++) mag[i] = Math.log1p(mag[i]);
    return { mag, pw, ph };
  }, []);

  // Compute FFT magnitude for selected image only (refs are static, not useful)
  const fftMags = React.useMemo((): { data: Float32Array; w: number; h: number }[] => {
    if (!showFft) return [];
    const selImg = getImage(displayIndex);
    if (!selImg) return [{ data: new Float32Array(0), w: 0, h: 0 }];
    const { mag, pw, ph } = computeFFT(selImg, imgWidth, imgHeight);
    return [{ data: mag, w: pw, h: ph }];
  }, [showFft, displayIndex, allImages, imgWidth, imgHeight, computeFFT, getImage]);

  // Reset FFT contrast when FFT data changes
  React.useEffect(() => {
    if (fftMags.length > 0) {
      setFftContrast(fftMags.map(() => ({ vmin: 0, vmax: 100 })));
    }
  }, [fftMags.length, displayIndex, showFft]); // eslint-disable-line react-hooks/exhaustive-deps

  // Update FFT contrast — linked mode updates all slots together
  const updateFftContrast = React.useCallback((fftIdx: number, vmin: number, vmax: number) => {
    setFftContrast((prev) => {
      if (fftLinked) {
        return prev.map(() => ({ vmin, vmax }));
      }
      const next = [...prev];
      next[fftIdx] = { vmin, vmax };
      return next;
    });
  }, [fftLinked]);

  // D-spacing click on FFT canvas — snap to brightest pixel in 5px radius
  // Click = replace all, Shift+click = add spot
  const handleFftClick = React.useCallback((e: React.MouseEvent, fftIdx: number, srcW: number, srcH: number) => {
    const canvas = e.currentTarget as HTMLElement;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const pw = nextPow2(srcW), ph = nextPow2(srcH);
    const scale = Math.min(imgSize / pw, imgSize / ph);
    const dw = pw * scale, dh = ph * scale;
    const ox = (imgSize - dw) / 2, oy = (imgSize - dh) / 2;
    let fftCol = (mx - ox) / scale;
    let fftRow = (my - oy) / scale;

    if (fftCol < 0 || fftCol >= pw || fftRow < 0 || fftRow >= ph) {
      if (!e.shiftKey) setDSpacingMarks([]);
      return;
    }

    // Snap to brightest pixel in 5px search radius
    const fftData = fftMags[fftIdx];
    if (fftData && fftData.data.length > 0) {
      const searchR = 5;
      const c0 = Math.max(0, Math.floor(fftCol) - searchR);
      const r0 = Math.max(0, Math.floor(fftRow) - searchR);
      const c1 = Math.min(pw - 1, Math.floor(fftCol) + searchR);
      const r1 = Math.min(ph - 1, Math.floor(fftRow) + searchR);
      let bestVal = -Infinity, bestC = Math.round(fftCol), bestR = Math.round(fftRow);
      for (let ir = r0; ir <= r1; ir++) {
        for (let ic = c0; ic <= c1; ic++) {
          const v = fftData.data[ir * pw + ic];
          if (v > bestVal) { bestVal = v; bestC = ic; bestR = ir; }
        }
      }
      fftCol = bestC;
      fftRow = bestR;
    }

    const dcol = fftCol - pw / 2;
    const drow = fftRow - ph / 2;
    const freqPx = Math.sqrt(dcol * dcol + drow * drow);

    if (freqPx < 1) {
      if (!e.shiftKey) setDSpacingMarks([]);
      return;
    }

    const freq = freqPx / Math.max(pw, ph);
    const hasCalibration = pixelSize > 0;
    const dSpacing = hasCalibration ? pixelSize / freq : 1.0 / freq;
    const unit = hasCalibration ? pixelUnit : "px";
    const friedelCol = Math.round(pw - fftCol);
    const friedelRow = Math.round(ph - fftRow);

    const mark: DSpacingMark = {
      fftIdx,
      col: Math.round(fftCol), row: Math.round(fftRow),
      freq, dSpacing, unit,
      friedelCol, friedelRow,
    };

    if (e.shiftKey) {
      setDSpacingMarks((prev) => [...prev, mark]);
    } else {
      setDSpacingMarks([mark]);
    }
  }, [imgSize, pixelSize, pixelUnit, fftMags]);

  const renderFFTToCanvas = React.useCallback((
    canvas: HTMLCanvasElement | null, fftIdx: number,
  ) => {
    if (!canvas || fftIdx >= fftMags.length) return;
    const { data: mag, w: pw, h: ph } = fftMags[fftIdx];
    if (!mag || mag.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = imgSize * DPR; canvas.height = imgSize * DPR;
    ctx.scale(DPR, DPR);
    ctx.fillStyle = isDark ? "#111" : "#eee";
    ctx.fillRect(0, 0, imgSize, imgSize);
    const lut = COLORMAPS[cmap] || COLORMAPS["inferno"];
    const range = findDataRange(mag);
    const contrast = fftContrast[fftIdx] || { vmin: 0, vmax: 100 };
    const { vmin, vmax } = sliderRange(range.min, range.max, contrast.vmin, contrast.vmax);
    const offscreen = document.createElement("canvas");
    offscreen.width = pw; offscreen.height = ph;
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;
    const imgData = offCtx.createImageData(pw, ph);
    applyColormap(mag, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);
    const scale = Math.min(imgSize / pw, imgSize / ph);
    const dw = pw * scale, dh = ph * scale;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, (imgSize - dw) / 2, (imgSize - dh) / 2, dw, dh);
    // FFT label
    ctx.fillStyle = "rgba(0,180,80,0.85)";
    ctx.font = `bold 11px ${FONT}`;
    ctx.textAlign = "right"; ctx.textBaseline = "top";
    ctx.fillText("FFT", imgSize - 6, 4);

    // D-spacing markers + Friedel pairs
    const marksForThis = dSpacingMarks.filter((m) => m.fftIdx === fftIdx);
    if (marksForThis.length > 0) {
      const ox = (imgSize - dw) / 2, oy = (imgSize - dh) / 2;
      const centerX = ox + (pw / 2) * scale;
      const centerY = oy + (ph / 2) * scale;
      const spotColors = ["#ff3333", "#ffaa00", "#00ff88", "#ff66cc", "#66aaff"];

      for (let mi = 0; mi < marksForThis.length; mi++) {
        const mark = marksForThis[mi];
        const sColor = spotColors[mi % spotColors.length];
        const sx = ox + mark.col * scale;
        const sy = oy + mark.row * scale;
        const fx = ox + mark.friedelCol * scale;
        const fy = oy + mark.friedelRow * scale;

        // Lines from center
        ctx.strokeStyle = sColor + "80"; // 50% alpha
        ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(centerX, centerY); ctx.lineTo(sx, sy); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(centerX, centerY); ctx.lineTo(fx, fy); ctx.stroke();
        ctx.setLineDash([]);

        // Spot circle
        ctx.strokeStyle = sColor; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.stroke();

        // Friedel pair circle (same color, dashed)
        ctx.setLineDash([2, 2]);
        ctx.beginPath(); ctx.arc(fx, fy, 6, 0, Math.PI * 2); ctx.stroke();
        ctx.setLineDash([]);

        // D-spacing label
        const dText = `${mark.dSpacing.toFixed(2)} ${mark.unit}`;
        ctx.font = `bold 10px ${FONT}`;
        const tw = ctx.measureText(dText).width;
        const tx = Math.min(sx + 10, imgSize - tw - 6);
        const ty = Math.max(sy - 18, 4 + mi * 16);
        ctx.fillStyle = "rgba(0,0,0,0.85)";
        ctx.beginPath(); ctx.roundRect(tx - 3, ty - 1, tw + 6, 14, 3); ctx.fill();
        ctx.fillStyle = sColor;
        ctx.textAlign = "left"; ctx.textBaseline = "top";
        ctx.fillText(dText, tx, ty);
      }
    }
  }, [imgSize, cmap, isDark, fftMags, fftContrast, dSpacingMarks]);

  // Render FFT canvas (selected image only)
  React.useEffect(() => {
    if (!showFft || fftMags.length === 0) return;
    renderFFTToCanvas(fftSelectedRef.current, 0);
  }, [showFft, fftMags, renderFFTToCanvas]);

  // ========================================================================
  // Image cursor readout
  // ========================================================================
  const handleImgMouseMove = React.useCallback((e: React.MouseEvent) => {
    const canvas = selectedImgRef.current;
    if (!canvas || !displayData) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const { data, w, h } = displayData;
    // Convert screen coords to image coords (accounting for zoom/pan + aspect fit)
    const baseScale = Math.min(imgSize / w, imgSize / h);
    const cx = imgSize / 2, cy = imgSize / 2;
    const imgX = (mx - cx - panX) / (zoom * baseScale) + w / 2;
    const imgY = (my - cy - panY) / (zoom * baseScale) + h / 2;
    const col = Math.floor(imgX);
    const row = Math.floor(imgY);
    if (row >= 0 && row < h && col >= 0 && col < w) {
      setCursorInfo({ row, col, val: data[row * w + col] });
    } else {
      setCursorInfo(null);
    }
  }, [displayData, imgSize, zoom, panX, panY]);

  const handleImgMouseLeaveImg = React.useCallback(() => {
    setCursorInfo(null);
  }, []);

  // ========================================================================
  // Image zoom/pan handlers
  // ========================================================================
  const handleImgWheel = React.useCallback((e: React.WheelEvent) => {
    const canvas = selectedImgRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const cx = imgSize / 2;
    const cy = imgSize / 2;

    // Mouse position in image space
    const imgX = (mx - cx - panX) / zoom + cx;
    const imgY = (my - cy - panY) / zoom + cy;

    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * factor));

    // Keep mouse position fixed
    const newPanX = mx - (imgX - cx) * newZoom - cx;
    const newPanY = my - (imgY - cy) * newZoom - cy;

    setZoom(newZoom);
    setPanX(newPanX);
    setPanY(newPanY);
  }, [zoom, panX, panY, imgSize]);

  const handleImgMouseDown = React.useCallback((e: React.MouseEvent) => {
    panDragRef.current = { active: true, startX: e.clientX, startY: e.clientY, startPanX: panX, startPanY: panY };
  }, [panX, panY]);

  const handleImgDoubleClick = React.useCallback(() => {
    setZoom(1.0); setPanX(0); setPanY(0);
  }, []);

  // ========================================================================
  // Draw metric chart
  // ========================================================================
  const drawChart = React.useCallback((
    canvas: HTMLCanvasElement | null,
    metricIdx: number,
  ) => {
    if (!canvas || groups.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const mName = metricNames[metricIdx];
    const mLabel = (metricLabelsArr && metricLabelsArr[metricIdx]) || mName;
    const mDir = (metricDirections && metricDirections[metricIdx]) || "min";

    canvas.width = chartW * DPR;
    canvas.height = chartH * DPR;
    ctx.scale(DPR, DPR);

    ctx.fillStyle = colors.bgAlt;
    ctx.fillRect(0, 0, chartW, chartH);

    const plotW = chartW - MARGIN.left - MARGIN.right;
    const plotH = chartH - MARGIN.top - MARGIN.bottom;
    if (plotW <= 0 || plotH <= 0) return;

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const g of groups) for (const pt of g.points) {
      const xv = xKey && typeof pt.params[xKey] === "number" ? (pt.params[xKey] as number) : pt.index;
      const yv = pt.metrics[mName];
      if (isFinite(xv)) { if (xv < xMin) xMin = xv; if (xv > xMax) xMax = xv; }
      if (isFinite(yv)) { if (yv < yMin) yMin = yv; if (yv > yMax) yMax = yv; }
    }
    if (!isFinite(xMin)) { xMin = 0; xMax = 1; }
    if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
    if (xMin === xMax) { xMin -= 0.5; xMax += 0.5; }
    if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }
    const yPad = (yMax - yMin) * 0.08;
    yMin -= yPad; yMax += yPad;

    const toX = (v: number) => MARGIN.left + ((v - xMin) / (xMax - xMin)) * plotW;
    const toY = (v: number) => MARGIN.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // Grid
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
    ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
    const xTicks = computeTicks(xMin, xMax, Math.max(3, Math.floor(plotW / 55)));
    for (const tv of xTicks) { const cx = snap(toX(tv)); if (cx >= MARGIN.left && cx <= MARGIN.left + plotW) { ctx.beginPath(); ctx.moveTo(cx, MARGIN.top); ctx.lineTo(cx, MARGIN.top + plotH); ctx.stroke(); } }
    const yTicks = computeTicks(yMin, yMax, Math.max(2, Math.floor(plotH / 35)));
    for (const tv of yTicks) { const cy = snap(toY(tv)); if (cy >= MARGIN.top && cy <= MARGIN.top + plotH) { ctx.beginPath(); ctx.moveTo(MARGIN.left, cy); ctx.lineTo(MARGIN.left + plotW, cy); ctx.stroke(); } }
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle = isDark ? "#666" : "#999"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(snap(MARGIN.left), MARGIN.top); ctx.lineTo(snap(MARGIN.left), snap(MARGIN.top + plotH)); ctx.lineTo(MARGIN.left + plotW, snap(MARGIN.top + plotH)); ctx.stroke();

    // Tick labels
    ctx.fillStyle = isDark ? "#bbb" : "#444"; ctx.font = `11px ${FONT}`;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const tv of xTicks) { const cx = toX(tv); if (cx >= MARGIN.left && cx <= MARGIN.left + plotW) { ctx.beginPath(); ctx.moveTo(cx, MARGIN.top + plotH); ctx.lineTo(cx, MARGIN.top + plotH + AXIS_TICK_PX); ctx.stroke(); ctx.fillText(formatNumber(tv), cx, MARGIN.top + plotH + AXIS_TICK_PX + 2); } }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const tv of yTicks) { const cy = toY(tv); if (cy >= MARGIN.top && cy <= MARGIN.top + plotH) { ctx.beginPath(); ctx.moveTo(MARGIN.left - AXIS_TICK_PX, cy); ctx.lineTo(MARGIN.left, cy); ctx.stroke(); ctx.fillText(formatNumber(tv), MARGIN.left - AXIS_TICK_PX - 2, cy); } }

    // Title
    ctx.fillStyle = isDark ? "#ddd" : "#222"; ctx.font = `bold 12px ${FONT}`;
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(`${mLabel} ${mDir === "min" ? "\u2193" : "\u2191"}`, MARGIN.left, 4);

    // Clip to plot area
    ctx.save(); ctx.beginPath(); ctx.rect(MARGIN.left, MARGIN.top, plotW, plotH); ctx.clip();

    // Lines + dots (skip hidden groups, dim during playback)
    const hiIdx = hoveredIndex !== null ? hoveredIndex : selectedIndex;
    for (const g of groups) {
      if (hiddenGroups.has(g.key)) continue;
      // Dim lines during playback
      ctx.globalAlpha = playing ? 0.25 : 1.0;
      ctx.strokeStyle = g.color; ctx.lineWidth = 2;
      ctx.beginPath(); let started = false;
      for (const pt of g.points) {
        const xv = xKey && typeof pt.params[xKey] === "number" ? (pt.params[xKey] as number) : pt.index;
        const yv = pt.metrics[mName];
        if (!isFinite(yv)) continue;
        const cx = toX(xv), cy = toY(yv);
        if (!started) { ctx.moveTo(cx, cy); started = true; } else ctx.lineTo(cx, cy);
      }
      ctx.stroke();

      for (const pt of g.points) {
        const xv = xKey && typeof pt.params[xKey] === "number" ? (pt.params[xKey] as number) : pt.index;
        const yv = pt.metrics[mName];
        if (!isFinite(yv)) continue;
        const cx = toX(xv), cy = toY(yv);

        // During playback: only the selected point is fully visible
        const isActive = pt.index === hiIdx;
        ctx.globalAlpha = playing && !isActive ? 0.15 : 1.0;

        // Global best → diamond
        if (pt.index === globalBest[mName]) {
          ctx.fillStyle = "#ffd700";
          ctx.strokeStyle = isDark ? "#000" : "#fff";
          ctx.lineWidth = 1.5;
          drawDiamond(ctx, cx, cy, isActive ? 9 : 7);
          ctx.fill(); ctx.stroke();
        }
        // Per-group best → star
        else if (pt.index === g.bestIndices[mName]) {
          ctx.fillStyle = g.color;
          ctx.strokeStyle = isDark ? "#000" : "#fff";
          ctx.lineWidth = 1.2;
          drawStar(ctx, cx, cy, isActive ? 8 : 6);
          ctx.fill(); ctx.stroke();
        }
        // Regular → dot
        else {
          ctx.fillStyle = g.color;
          ctx.beginPath(); ctx.arc(cx, cy, isActive ? 5 : 2.5, 0, Math.PI * 2); ctx.fill();
        }
      }
      ctx.globalAlpha = 1.0;
    }
    ctx.restore();

    // Active highlight (crosshair + value badge)
    if (hiIdx >= 0 && hiIdx < allPoints.length) {
      const pt = allPoints[hiIdx];
      const xv = xKey && typeof pt.params[xKey] === "number" ? (pt.params[xKey] as number) : pt.index;
      const yv = pt.metrics[mName];
      if (isFinite(yv)) {
        const cx = toX(xv), cy = toY(yv);
        ctx.strokeStyle = isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.15)";
        ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(cx, MARGIN.top); ctx.lineTo(cx, MARGIN.top + plotH);
        ctx.moveTo(MARGIN.left, cy); ctx.lineTo(MARGIN.left + plotW, cy); ctx.stroke(); ctx.setLineDash([]);

        ctx.fillStyle = "#ff3333"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

        const valText = formatNumber(yv, 3);
        ctx.font = `bold 11px ${FONT}`;
        const tw = ctx.measureText(valText).width;
        const tx = Math.min(cx + 10, chartW - tw - 8);
        const ty = Math.max(cy - 22, MARGIN.top + 2);
        ctx.fillStyle = isDark ? "rgba(30,30,30,0.92)" : "rgba(255,255,255,0.95)";
        ctx.strokeStyle = isDark ? "#555" : "#ccc"; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.roundRect(tx - 4, ty - 2, tw + 8, 18, 3); ctx.fill(); ctx.stroke();
        ctx.fillStyle = isDark ? "#fff" : "#000";
        ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(valText, tx, ty);
      }
    }

    // Persistent selected ring
    if (hoveredIndex !== null && selectedIndex !== hoveredIndex && selectedIndex >= 0 && selectedIndex < allPoints.length) {
      const pt = allPoints[selectedIndex];
      const xv = xKey && typeof pt.params[xKey] === "number" ? (pt.params[xKey] as number) : pt.index;
      const yv = pt.metrics[mName];
      if (isFinite(yv)) {
        const cx = toX(xv), cy = toY(yv);
        ctx.strokeStyle = "#ff3333"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.stroke();
      }
    }

    // X axis label
    if (xLabel) {
      ctx.fillStyle = isDark ? "#999" : "#555"; ctx.font = `11px ${FONT}`;
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(xLabel, MARGIN.left + plotW / 2, chartH - 11);
    }

    // Legend (click to toggle group visibility)
    if (groups.length > 1) {
      ctx.font = `10px ${FONT}`; ctx.textAlign = "right"; ctx.textBaseline = "top";
      let ly = MARGIN.top + 2;
      for (const g of groups) {
        const isHidden = hiddenGroups.has(g.key);
        // Color swatch
        ctx.globalAlpha = isHidden ? 0.3 : 1.0;
        ctx.fillStyle = g.color;
        ctx.fillRect(chartW - MARGIN.right - 8, ly + 2, 6, 6);
        // Label — strikethrough if hidden
        ctx.fillStyle = isDark ? "#aaa" : "#555";
        ctx.fillText(g.key, chartW - MARGIN.right - 12, ly);
        if (isHidden) {
          // Draw strikethrough line
          const tw = ctx.measureText(g.key).width;
          ctx.strokeStyle = isDark ? "#aaa" : "#555";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(chartW - MARGIN.right - 12 - tw, ly + 5);
          ctx.lineTo(chartW - MARGIN.right - 12, ly + 5);
          ctx.stroke();
        }
        ctx.globalAlpha = 1.0;
        ly += 13;
      }
    }
  }, [groups, allPoints, metricNames, metricLabelsArr, metricDirections, xKey, xLabel, isDark, hoveredIndex, selectedIndex, chartW, chartH, globalBest, hiddenGroups, colors, playing]);

  React.useEffect(() => {
    for (let i = 0; i < metricNames.length; i++) drawChart(chartCanvasRefs.current[i], i);
  }, [metricNames, drawChart]);

  // ========================================================================
  // Chart mouse
  // ========================================================================
  const handleChartMouse = React.useCallback((e: React.MouseEvent<HTMLCanvasElement>, metricIdx: number, isClick: boolean) => {
    const canvas = chartCanvasRefs.current[metricIdx];
    if (!canvas || allPoints.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const plotW = chartW - MARGIN.left - MARGIN.right;
    const plotH = chartH - MARGIN.top - MARGIN.bottom;
    const mName = metricNames[metricIdx];
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const g of groups) for (const pt of g.points) {
      const xv = xKey && typeof pt.params[xKey] === "number" ? (pt.params[xKey] as number) : pt.index;
      const yv = pt.metrics[mName];
      if (isFinite(xv)) { if (xv < xMin) xMin = xv; if (xv > xMax) xMax = xv; }
      if (isFinite(yv)) { if (yv < yMin) yMin = yv; if (yv > yMax) yMax = yv; }
    }
    if (!isFinite(xMin)) { xMin = 0; xMax = 1; } if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
    if (xMin === xMax) xMax += 1; if (yMin === yMax) yMax += 1;
    const yPad = (yMax - yMin) * 0.08; yMin -= yPad; yMax += yPad;
    const toX = (v: number) => MARGIN.left + ((v - xMin) / (xMax - xMin)) * plotW;
    const toY = (v: number) => MARGIN.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // Check if click is on a legend entry (top-right area)
    if (isClick && groups.length > 1) {
      let ly = MARGIN.top + 2;
      for (const g of groups) {
        // Legend hit area: right side, ~80px wide, 13px tall per entry
        if (mx >= chartW - MARGIN.right - 80 && mx <= chartW - MARGIN.right &&
            my >= ly && my <= ly + 13) {
          // Toggle this group
          setHiddenGroups((prev) => {
            const next = new Set(prev);
            if (next.has(g.key)) next.delete(g.key);
            else next.add(g.key);
            return next;
          });
          return; // Don't also select a point
        }
        ly += 13;
      }
    }

    // Find nearest data point
    let bestDist = Infinity, bestIdx = -1;
    for (const pt of allPoints) {
      if (hiddenGroups.has(pt.groupKey)) continue; // skip hidden groups
      const xv = xKey && typeof pt.params[xKey] === "number" ? (pt.params[xKey] as number) : pt.index;
      const yv = pt.metrics[mName];
      if (!isFinite(yv)) continue;
      const dist = Math.sqrt((mx - toX(xv)) ** 2 + (my - toY(yv)) ** 2);
      if (dist < bestDist) { bestDist = dist; bestIdx = pt.index; }
    }
    if (bestIdx >= 0 && bestDist < 30) {
      if (isClick) { setSelectedIndex(bestIdx); model.set("selected_index", bestIdx); model.save_changes(); rootRef.current?.focus(); }
      else setHoveredIndex(bestIdx);
    } else if (!isClick) setHoveredIndex(null);
  }, [allPoints, groups, metricNames, xKey, model, setSelectedIndex, chartW, chartH, hiddenGroups]);

  // ========================================================================
  // Keyboard navigation
  // ========================================================================
  const handleKeyDown = React.useCallback((e: React.KeyboardEvent) => {
    if (sortedIndices.length === 0) return;
    const target = e.target as HTMLElement;
    if (target.tagName === "INPUT" || target.tagName === "SELECT" || target.tagName === "TEXTAREA") return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      const pos = sortedIndices.indexOf(selectedIndex);
      const newPos = e.key === "ArrowLeft"
        ? (pos <= 0 ? sortedIndices.length - 1 : pos - 1)
        : (pos >= sortedIndices.length - 1 ? 0 : pos + 1);
      const newIdx = sortedIndices[newPos];
      setSelectedIndex(newIdx); model.set("selected_index", newIdx); model.save_changes();
    } else if (e.key === "r" || e.key === "R") {
      setSelectedIndex(0); model.set("selected_index", 0); model.save_changes();
    } else if (e.key === "f" || e.key === "F") {
      setShowFft((v) => !v);
    }
  }, [sortedIndices, selectedIndex, model, setSelectedIndex]);

  // ========================================================================
  // JSX
  // ========================================================================
  // nMetrics removed (unused)
  const displayPt = displayIndex >= 0 && displayIndex < allPoints.length ? allPoints[displayIndex] : null;

  const themedSelect = React.useMemo(() => ({
    minWidth: 90, fontSize: 11,
    bgcolor: colors.controlBg, color: colors.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent },
    "& .MuiSelect-icon": { color: colors.textMuted },
  }), [colors]);

  const themedMenuProps = React.useMemo(() => ({
    PaperProps: { sx: {
      bgcolor: colors.controlBg, color: colors.text, border: `1px solid ${colors.border}`,
      "& .MuiMenuItem-root": { fontSize: 11 },
      "& .MuiMenuItem-root:hover": { bgcolor: isDark ? "#333" : "#e0e0e0" },
      "& .MuiMenuItem-root.Mui-selected": { bgcolor: isDark ? "#2a2a2a" : "#d0d0d0" },
    } },
  }), [colors, isDark]);

  return (
    <Box ref={rootRef} className="show-metric-explorer-root" tabIndex={0} onKeyDown={handleKeyDown} sx={container.root}>
      {/* Header — Show2D style: title + toggles + actions in one line */}
      <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.MD}px`, mb: `${SPACING.SM}px` }}>
        <Typography sx={{ fontSize: 12, color: colors.accent, ...typography.title }}>
          {displayPt ? displayPt.label : "MetricExplorer"}
        </Typography>
        <Typography sx={typography.labelSmall}>FFT:</Typography>
        <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} sx={switchStyles.small} size="small" />
        <Typography
          sx={{ fontSize: 10, color: colors.accent, cursor: "pointer", "&:hover": { opacity: 0.7 } }}
          onClick={() => {
            setVminPct(0); setVmaxPct(100);
            setZoom(1.0); setPanX(0); setPanY(0);
            setHiddenGroups(new Set()); setHiddenMetrics(new Set());
            setShowFft(false); setLogScale(false);
            setFftContrast([]); setDSpacingMarks([]);
            setSelectedIndex(0); model.set("selected_index", 0); model.save_changes();
          }}
        >RESET</Typography>
        <Typography
          sx={{ fontSize: 10, color: colors.accent, cursor: "pointer", "&:hover": { opacity: 0.7 } }}
          onClick={() => {
            if (!displayPt) return;
            navigator.clipboard.writeText(JSON.stringify(displayPt.params, null, 2)).then(() => {
              setCopyFeedback(true);
              setTimeout(() => setCopyFeedback(false), 1500);
            });
          }}
        >{copyFeedback ? "COPIED" : "COPY"}</Typography>
        <Box sx={{ flex: 1 }} />
        <Typography sx={{ fontSize: 9, color: colors.textMuted, opacity: 0.5 }}>
          ← → navigate | F fft | scroll zoom
        </Typography>
      </Box>

      {/* Row 1: Metric charts */}
      <Box sx={{ display: "flex", gap: `${SPACING.SM}px`, flexWrap: "wrap" }}>
        {metricNames.map((_, mi) => (
          <Box key={mi} sx={{ position: "relative" }}>
            <canvas
              ref={(el) => { chartCanvasRefs.current[mi] = el; }}
              style={{ width: chartW, height: chartH, display: "block", cursor: "crosshair" }}
              onMouseMove={(e) => handleChartMouse(e, mi, false)}
              onClick={(e) => handleChartMouse(e, mi, true)}
              onMouseLeave={() => setHoveredIndex(null)}
            />
            <ResizeHandle accentColor={colors.accent} onStart={(e) => {
              chartResizeRef.current = { active: true, startX: e.clientX, startY: e.clientY, startScale: chartScale, startW: chartW };
              e.stopPropagation();
            }} />
          </Box>
        ))}
      </Box>

      {/* Playback row — Show3D style: transport + slider + label */}
      {nPoints > 1 && (
        <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${colors.border}`, bgcolor: colors.controlBg }}>
          <Stack direction="row" spacing={0} sx={{ flexShrink: 0, mr: 0.5 }}>
            <IconButton size="small" onClick={() => { setPlayReverse(true); setPlaying(true); }} sx={{ color: playReverse && playing ? colors.accent : colors.textMuted, p: 0.25 }}>
              <FastRewindIcon sx={{ fontSize: 18 }} />
            </IconButton>
            <IconButton size="small" onClick={() => setPlaying(!playing)} sx={{ color: colors.accent, p: 0.25 }}>
              {playing ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
            </IconButton>
            <IconButton size="small" onClick={() => { setPlayReverse(false); setPlaying(true); }} sx={{ color: !playReverse && playing ? colors.accent : colors.textMuted, p: 0.25 }}>
              <FastForwardIcon sx={{ fontSize: 18 }} />
            </IconButton>
            <IconButton size="small" onClick={() => { setPlaying(false); setSelectedIndex(0); model.set("selected_index", 0); model.save_changes(); }} sx={{ color: colors.textMuted, p: 0.25 }}>
              <StopIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Stack>
          <Slider
            value={selectedIndex}
            onChange={(_, v) => {
              if (playing) setPlaying(false);
              const idx = v as number;
              setSelectedIndex(idx);
              model.set("selected_index", idx);
              model.save_changes();
            }}
            min={0}
            max={nPoints - 1}
            step={1}
            size="small"
            valueLabelDisplay="auto"
            valueLabelFormat={(idx) => {
              const pt = idx >= 0 && idx < allPoints.length ? allPoints[idx] : null;
              return pt ? pt.label : String(idx);
            }}
            sx={{
              flex: 1, minWidth: 40,
              "& .MuiSlider-thumb": { width: 12, height: 12 },
              "& .MuiSlider-rail": { height: 3 },
              "& .MuiSlider-track": { height: 3 },
              "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 6px" },
            }}
          />
          <Typography sx={{ ...typography.value, color: colors.textMuted, minWidth: "6ch", textAlign: "right", flexShrink: 0 }}>
            {selectedIndex + 1}/{nPoints}
          </Typography>
          <Typography sx={typography.labelSmall}>fps</Typography>
          <Slider value={playFps} min={1} max={15} step={1} onChange={(_, v) => setPlayFps(v as number)} size="small" sx={{ width: 40, flexShrink: 0, "& .MuiSlider-thumb": { width: 10, height: 10 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 } }} />
          <Typography sx={{ ...typography.value, color: colors.textMuted, minWidth: 14, flexShrink: 0 }}>{playFps}</Typography>
        </Box>
      )}


      {/* Row 2: Images */}
      <Box sx={{ display: "flex", gap: `${SPACING.SM}px`, alignItems: "flex-start" }}>
        {/* Selected image — accent border */}
        <Box sx={{ flexShrink: 0 }}>
          <Box
            ref={imgContainerRef}
            sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${hoveredIndex !== null ? "#ff3333" : colors.accent}` }}
            onWheel={handleImgWheel}
            onMouseDown={handleImgMouseDown}
            onMouseMove={handleImgMouseMove}
            onMouseLeave={handleImgMouseLeaveImg}
            onDoubleClick={handleImgDoubleClick}
          >
            <canvas ref={selectedImgRef} style={{ width: imgSize, height: imgSize, display: "block", cursor: "grab" }} />
            <ResizeHandle accentColor={colors.accent} onStart={(e) => {
              imgResizeRef.current = { active: true, startX: e.clientX, startY: e.clientY, startSize: imgSize };
              e.stopPropagation();
            }} />
          </Box>
        </Box>

      </Box>

      {/* Controls — Show2D style: Scale + Color + Histogram */}
      <Box sx={{ display: "flex", gap: `${SPACING.SM}px`, mt: `${SPACING.XS}px` }}>
        <Box sx={{ ...controlRow, border: `1px solid ${colors.border}`, bgcolor: colors.controlBg }}>
          <Typography sx={typography.labelSmall}>Log:</Typography>
          <Switch checked={logScale} onChange={(e) => setLogScale(e.target.checked)} sx={switchStyles.small} size="small" />
          <Typography sx={typography.labelSmall}>Color:</Typography>
          <Select value={cmap} onChange={(e) => setCmap(e.target.value as string)} size="small" sx={themedSelect} MenuProps={themedMenuProps}>
            {COLORMAP_NAMES.map((name) => <MenuItem key={name} value={name}>{name}</MenuItem>)}
          </Select>
        </Box>
        <MiniHistogram
          data={allImages}
          vminPct={vminPct}
          vmaxPct={vmaxPct}
          onRangeChange={(a, b) => { setVminPct(a); setVmaxPct(b); }}
          width={200}
          height={36}
          theme={themeInfo.theme}
          dataMin={dataRange.min}
          dataMax={dataRange.max}
        />
      </Box>

      {/* FFT row (when toggled) */}
      {showFft && fftMags.length > 0 && (
        <>
          {/* FFT d-spacing readout */}
          <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, mt: `${SPACING.SM}px`, mb: `${SPACING.XS}px` }}>
            {dSpacingMarks.length > 0 && (
              <Box sx={{ display: "flex", gap: `${SPACING.SM}px`, flexWrap: "wrap", alignItems: "center" }}>
                {dSpacingMarks.map((m, i) => (
                  <Typography key={i} sx={{ fontSize: 11, fontFamily: "monospace", color: ["#ff3333", "#ffaa00", "#00ff88", "#ff66cc", "#66aaff"][i % 5] }}>
                    d{dSpacingMarks.length > 1 ? `${i + 1}` : ""} = {m.dSpacing.toFixed(2)} {m.unit}
                  </Typography>
                ))}
                <Typography
                  sx={{ fontSize: 9, color: colors.textMuted, cursor: "pointer", "&:hover": { color: colors.text } }}
                  onClick={() => setDSpacingMarks([])}
                >clear</Typography>
              </Box>
            )}
          </Box>
          <Box sx={{ display: "flex", gap: `${SPACING.SM}px`, alignItems: "flex-start" }}>
            {/* FFT of selected */}
            <Box sx={{ flexShrink: 0 }}>
              <Box
                sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${isDark ? "rgba(0,180,80,0.4)" : "rgba(0,140,60,0.3)"}`, cursor: "crosshair" }}
                onClick={(e) => handleFftClick(e, 0, imgWidth, imgHeight)}
              >
                <canvas ref={fftSelectedRef} style={{ width: imgSize, height: imgSize, display: "block" }} />
                <Box sx={{ position: "absolute", top: 4, right: 6 }}>
                  <Typography sx={{ fontSize: 10, color: "rgba(0,220,80,0.9)", fontWeight: "bold", textShadow: "0 1px 2px rgba(0,0,0,0.8)" }}>FFT</Typography>
                </Box>
              </Box>
              {fftMags[0] && fftMags[0].data.length > 0 && (
                <MiniHistogram
                  data={fftMags[0].data}
                  vminPct={fftContrast[0]?.vmin ?? 0}
                  vmaxPct={fftContrast[0]?.vmax ?? 100}
                  onRangeChange={(a, b) => updateFftContrast(0, a, b)}
                  width={imgSize}
                  height={28}
                  theme={themeInfo.theme}
                  dataMin={findDataRange(fftMags[0].data).min}
                  dataMax={findDataRange(fftMags[0].data).max}
                />
              )}
            </Box>

          </Box>
        </>
      )}
    </Box>
  );
}

export const render = createRender(MetricExplorer);
