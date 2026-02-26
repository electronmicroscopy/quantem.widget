import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Select, { type SelectChangeEvent } from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import "./mark2d.css";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar, roundToNiceValue, exportFigure, canvasToPDF } from "../scalebar";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen, renderToOffscreenReuse } from "../colormaps";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, computeStats, percentileClip, sliderRange, applyLogScale } from "../stats";
import { fft2d, fftshift, computeMagnitude, autoEnhanceFFT, getWebGPUFFT, nextPow2, applyHannWindow2D, type WebGPUFFT } from "../webgpu-fft";
import JSZip from "jszip";

type MarkerShape = "circle" | "triangle" | "square" | "diamond" | "star";
type Point = { row: number; col: number; shape: MarkerShape; color: string };
type ZoomState = { zoom: number; panX: number; panY: number };

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DRAG_THRESHOLD = 3;
const DEFAULT_ZOOM: ZoomState = { zoom: 1, panX: 0, panY: 0 };
const CANVAS_TARGET_SIZE = 600;
const GALLERY_TARGET_SIZE = 300;
const DPR = window.devicePixelRatio || 1;

const SPACING = {
  XS: 4,
  SM: 8,
  MD: 12,
  LG: 16,
};

const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" as const },
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
  minWidth: 0,
  px: 1,
  py: 0.25,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

const switchStyles = {
  small: { '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
};

const containerStyles = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
};

const MARKER_COLORS = [
  "#f44336", // red
  "#4caf50", // green
  "#2196f3", // blue
  "#ff9800", // orange
  "#9c27b0", // purple
  "#00bcd4", // cyan
  "#ffeb3b", // yellow
  "#e91e63", // pink
  "#8bc34a", // lime
  "#ff5722", // deep orange
];

const MARKER_SHAPES: MarkerShape[] = ["circle", "triangle", "square", "diamond", "star"];

const ROI_SHAPES = ["square", "rectangle", "circle"] as const;
type RoiShape = typeof ROI_SHAPES[number];

type ROI = {
  id: number;
  shape: RoiShape;
  row: number;
  col: number;
  radius: number;
  width: number;
  height: number;
  color: string;
  opacity: number;
};

const ROI_COLORS = ["#0f0", "#ff0", "#0af", "#f0f", "#f80", "#f44"];

type RoiStats = { mean: number; std: number; min: number; max: number; count: number };

function computeRoiStats(roi: ROI, data: Float32Array, width: number, height: number): RoiStats | null {
  if (!data || data.length === 0) return null;
  let sum = 0, sumSq = 0, min = Infinity, max = -Infinity, count = 0;
  let x0: number, y0: number, x1: number, y1: number;
  if (roi.shape === "rectangle") {
    x0 = Math.max(0, Math.floor(roi.col - roi.width / 2));
    y0 = Math.max(0, Math.floor(roi.row - roi.height / 2));
    x1 = Math.min(width - 1, Math.ceil(roi.col + roi.width / 2));
    y1 = Math.min(height - 1, Math.ceil(roi.row + roi.height / 2));
  } else {
    x0 = Math.max(0, Math.floor(roi.col - roi.radius));
    y0 = Math.max(0, Math.floor(roi.row - roi.radius));
    x1 = Math.min(width - 1, Math.ceil(roi.col + roi.radius));
    y1 = Math.min(height - 1, Math.ceil(roi.row + roi.radius));
  }
  const r2 = roi.radius * roi.radius;
  for (let py = y0; py <= y1; py++) {
    for (let px = x0; px <= x1; px++) {
      if (roi.shape === "circle") {
        const dx = px - roi.col, dy = py - roi.row;
        if (dx * dx + dy * dy > r2) continue;
      }
      const val = data[py * width + px];
      sum += val; sumSq += val * val;
      if (val < min) min = val;
      if (val > max) max = val;
      count++;
    }
  }
  if (count === 0) return null;
  const mean = sum / count;
  const std = Math.sqrt(Math.max(0, sumSq / count - mean * mean));
  return { mean, std, min, max, count };
}

// ============================================================================
// Crop ROI region from raw float32 data for ROI-scoped FFT
// ============================================================================
function cropROIRegion(
  data: Float32Array, imgW: number, imgH: number,
  roi: ROI,
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

  if (shape === "circle") {
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

function formatDistance(p1: Point, p2: Point, pixelSize: number): string {
  const dx = p2.col - p1.col, dy = p2.row - p1.row;
  const distPx = Math.sqrt(dx * dx + dy * dy);
  if (pixelSize > 0) {
    const distAng = distPx * pixelSize;
    return distAng >= 10 ? `${(distAng / 10).toFixed(2)} nm` : `${distAng.toFixed(2)} \u00C5`;
  }
  return `${distPx.toFixed(1)} px`;
}

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

function brightenColor(hex: string, amount: number): string {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) return hex;
  const r = Math.min(255, parseInt(m[1], 16) + amount);
  const g = Math.min(255, parseInt(m[2], 16) + amount);
  const b = Math.min(255, parseInt(m[3], 16) + amount);
  return `rgb(${r},${g},${b})`;
}

function drawROI(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  shape: RoiShape,
  radius: number, w: number, h: number,
  color: string, opacity: number,
  isActive: boolean,
  isHovered: boolean,
): void {
  ctx.save();
  const highlighted = isActive || isHovered;
  const halfW = shape === "rectangle" ? w / 2 : radius;
  const halfH = shape === "rectangle" ? h / 2 : radius;
  ctx.globalAlpha = highlighted ? Math.min(1, opacity + 0.2) : opacity;
  ctx.strokeStyle = highlighted ? brightenColor(color, 80) : color;
  ctx.lineWidth = isActive ? 3 : isHovered ? 2.5 : 2;
  if (highlighted) {
    ctx.shadowColor = brightenColor(color, 120);
    ctx.shadowBlur = isActive ? 8 : 5;
  }
  if (shape === "circle") {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
  } else if (shape === "square") {
    ctx.strokeRect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (shape === "rectangle") {
    ctx.strokeRect(x - w / 2, y - h / 2, w, h);
  }
  if (isActive) {
    ctx.shadowBlur = 0;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(x - halfW, y);
    ctx.lineTo(x + halfW, y);
    ctx.moveTo(x, y - halfH);
    ctx.lineTo(x, y + halfH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = highlighted ? brightenColor(color, 80) : color;
    ctx.fill();
  }
  ctx.restore();
}

function drawMarker(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, shape: MarkerShape, fillColor: string, strokeColor: string, opacity: number, strokeWidth: number) {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.beginPath();
  switch (shape) {
    case "circle":
      ctx.arc(x, y, r, 0, Math.PI * 2);
      break;
    case "triangle":
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r * 0.87, y + r * 0.5);
      ctx.lineTo(x - r * 0.87, y + r * 0.5);
      ctx.closePath();
      break;
    case "square":
      ctx.rect(x - r * 0.75, y - r * 0.75, r * 1.5, r * 1.5);
      break;
    case "diamond":
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r * 0.7, y);
      ctx.lineTo(x, y + r);
      ctx.lineTo(x - r * 0.7, y);
      ctx.closePath();
      break;
    case "star": {
      const spikes = 5;
      const outerR = r;
      const innerR = r * 0.4;
      for (let s = 0; s < spikes * 2; s++) {
        const rad = (s * Math.PI) / spikes - Math.PI / 2;
        const sr = s % 2 === 0 ? outerR : innerR;
        if (s === 0) ctx.moveTo(x + sr * Math.cos(rad), y + sr * Math.sin(rad));
        else ctx.lineTo(x + sr * Math.cos(rad), y + sr * Math.sin(rad));
      }
      ctx.closePath();
      break;
    }
  }
  ctx.fillStyle = fillColor;
  ctx.fill();
  if (strokeWidth > 0) {
    ctx.clip();
    ctx.lineWidth = strokeWidth * 2;
    ctx.strokeStyle = strokeColor;
    ctx.stroke();
  }
  ctx.restore();
}

function ShapeIcon({ shape, color, size }: { shape: MarkerShape; color: string; size: number }) {
  const h = size / 2;
  const r = h * 0.8;
  let path: React.ReactNode;
  switch (shape) {
    case "circle": path = <circle cx={h} cy={h} r={r} />; break;
    case "triangle": path = <polygon points={`${h},${h - r} ${h + r * 0.87},${h + r * 0.5} ${h - r * 0.87},${h + r * 0.5}`} />; break;
    case "square": path = <rect x={h - r * 0.75} y={h - r * 0.75} width={r * 1.5} height={r * 1.5} />; break;
    case "diamond": path = <polygon points={`${h},${h - r} ${h + r * 0.7},${h} ${h},${h + r} ${h - r * 0.7},${h}`} />; break;
    case "star": {
      const pts: string[] = [];
      for (let i = 0; i < 10; i++) {
        const a = (i * Math.PI) / 5 - Math.PI / 2;
        const sr = i % 2 === 0 ? r : r * 0.4;
        pts.push(`${h + sr * Math.cos(a)},${h + sr * Math.sin(a)}`);
      }
      path = <polygon points={pts.join(" ")} />;
      break;
    }
  }
  return (
    <svg width={size} height={size} style={{ display: "block", flexShrink: 0 }}>
      <g fill={color}>{path}</g>
    </svg>
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
          sx: { bgcolor: isDark ? "#333" : "#fff", color: isDark ? "#ddd" : "#333", border: `1px solid ${isDark ? "#555" : "#ccc"}`, maxWidth: 280, p: 1 },
        },
        arrow: {
          sx: { color: isDark ? "#333" : "#fff", "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` } },
        },
      }}
    >
      <Typography
        component="span"
        sx={{ fontSize: 12, color: isDark ? "#888" : "#666", cursor: "help", ml: 0.5, "&:hover": { color: isDark ? "#aaa" : "#444" } }}
      >
        ⓘ
      </Typography>
    </Tooltip>
  );
}

function HistogramWidget({
  data,
  vminPct,
  vmaxPct,
  onRangeChange,
  width = 110,
  height = 40,
  theme = "dark",
  dataMin = 0,
  dataMax = 1,
}: {
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
  const bins = React.useMemo(() => data ? computeHistogramFromBytes(data) : new Array(256).fill(0), [data]);
  const colors = theme === "dark" ? {
    bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333",
  } : {
    bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc",
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
    const reduced: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) sum += bins[i * binRatio + j] || 0;
      reduced.push(sum / binRatio);
    }
    const maxVal = Math.max(...reduced, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barH = (reduced[i] / maxVal) * (height - 2);
      ctx.fillStyle = i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
      ctx.fillRect(i * barWidth + 0.5, height - barH, Math.max(1, barWidth - 1), barH);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas ref={canvasRef} style={{ width, height, border: `1px solid ${colors.border}` }} />
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

const render = createRender(() => {
  const { themeInfo, colors: tc } = useTheme();
  const accentGreen = themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a";

  const themedSelect = {
    minWidth: 50, fontSize: 11, bgcolor: tc.controlBg, color: tc.text,
    "& .MuiSelect-select": { py: 0.5, px: 1 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: tc.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: tc.accent },
  };
  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: tc.controlBg, color: tc.text, border: `1px solid ${tc.border}` } },
  };

  // Model state
  const [nImages] = useModelState<number>("n_images");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [imgMin] = useModelState<number[]>("img_min");
  const [imgMax] = useModelState<number[]>("img_max");
  const [selectedIdx, setSelectedIdx] = useModelState<number>("selected_idx");
  const [ncols] = useModelState<number>("ncols");
  const [labels] = useModelState<string[]>("labels");
  const [scale] = useModelState<number>("scale");
  const [selectedPoints, setSelectedPoints] = useModelState<Point[] | Point[][]>("selected_points");
  const [dotSize, setDotSize] = useModelState<number>("dot_size");
  const [maxPoints, setMaxPoints] = useModelState<number>("max_points");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [title] = useModelState<string>("title");
  const [widgetVersion] = useModelState<string>("widget_version");
  const [showStats] = useModelState<boolean>("show_stats");
  const [canvasSizeTrait] = useModelState<number>("canvas_size");
  const [showControls] = useModelState<boolean>("show_controls");
  const [disabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools] = useModelState<string[]>("hidden_tools");
  const [percentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh] = useModelState<number>("percentile_high");

  const disabledToolSet = React.useMemo(
    () =>
      new Set(
        (disabledTools || [])
          .map((name) => String(name).trim().toLowerCase())
          .filter(Boolean),
      ),
    [disabledTools],
  );
  const hiddenToolSet = React.useMemo(
    () =>
      new Set(
        (hiddenTools || [])
          .map((name) => String(name).trim().toLowerCase())
          .filter(Boolean),
      ),
    [hiddenTools],
  );
  const hideAll = hiddenToolSet.has("all");
  const hidePoints = hideAll || hiddenToolSet.has("points");
  const hideRoi = hideAll || hiddenToolSet.has("roi");
  const hideProfile = hideAll || hiddenToolSet.has("profile");
  const hideDisplay = hideAll || hiddenToolSet.has("display");
  const hideMarkers = hideAll || hiddenToolSet.has("marker_style");
  const hideSnap = hideAll || hiddenToolSet.has("snap");
  const hideNavigation = hideAll || hiddenToolSet.has("navigation");
  const hideView = hideAll || hiddenToolSet.has("view");
  const hideExport = hideAll || hiddenToolSet.has("export");

  const lockAll = disabledToolSet.has("all") || hideAll;
  const lockPoints = lockAll || hidePoints || disabledToolSet.has("points");
  const lockRoi = lockAll || hideRoi || disabledToolSet.has("roi");
  const lockProfile = lockAll || hideProfile || disabledToolSet.has("profile");
  const lockDisplay = lockAll || hideDisplay || disabledToolSet.has("display");
  const lockMarkers =
    lockAll ||
    hideMarkers ||
    disabledToolSet.has("marker_style");
  const lockSnap = lockAll || hideSnap || disabledToolSet.has("snap");
  const lockNavigation = lockAll || hideNavigation || disabledToolSet.has("navigation");
  const lockView = lockAll || hideView || disabledToolSet.has("view");
  const lockExport = lockAll || hideExport || disabledToolSet.has("export");
  const lockMarkerSettings = lockMarkers || lockPoints;
  const showMarkerStyleControls = !hidePoints && !hideMarkers;
  const showSnapControls = !hideSnap;
  const showProfileControls = !hideProfile;
  const showPrimaryControlRow =
    showMarkerStyleControls || showSnapControls || showProfileControls;

  const isGallery = nImages > 1;

  // Marker styling (Python-configurable)
  const [borderWidth, setBorderWidth] = useModelState<number>("marker_border");
  const [markerOpacity, setMarkerOpacity] = useModelState<number>("marker_opacity");
  const [labelSize, setLabelSize] = useModelState<number>("label_size");
  const [labelColor, setLabelColor] = useModelState<string>("label_color");

  // Current marker style (synced to Python for state portability)
  const [currentShape, setCurrentShape] = useModelState<string>("marker_shape");
  const [currentColor, setCurrentColor] = useModelState<string>("marker_color");

  // ROI state (synced to Python via roi_list trait)
  const [rois, setRois] = useModelState<ROI[]>("roi_list");
  const [activeRoiIdx, setActiveRoiIdx] = React.useState(-1);
  const [hoveredRoiIdx, setHoveredRoiIdx] = React.useState(-1);
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [newRoiShape, setNewRoiShape] = React.useState<RoiShape>("square");
  const safeRois = rois || [];
  const activeRoi = activeRoiIdx >= 0 && activeRoiIdx < safeRois.length ? safeRois[activeRoiIdx] : null;

  // ROI FFT state
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);

  const pushRoiHistory = React.useCallback(() => {
    roiHistoryRef.current = [...roiHistoryRef.current.slice(-49), safeRois.map(r => ({ ...r }))];
    roiRedoRef.current = [];
  }, [safeRois]);

  const updateActiveRoi = React.useCallback((updates: Partial<ROI>) => {
    pushRoiHistory();
    setRois(safeRois.map((r, i) => i === activeRoiIdx ? { ...r, ...updates } : r));
  }, [activeRoiIdx, safeRois, setRois, pushRoiHistory]);

  const undoRoi = React.useCallback(() => {
    if (roiHistoryRef.current.length === 0) return false;
    const prev = roiHistoryRef.current.pop()!;
    roiRedoRef.current.push(safeRois.map(r => ({ ...r })));
    setRois(prev);
    return true;
  }, [safeRois, setRois]);

  const redoRoi = React.useCallback(() => {
    if (roiRedoRef.current.length === 0) return false;
    const next = roiRedoRef.current.pop()!;
    roiHistoryRef.current.push(safeRois.map(r => ({ ...r })));
    setRois(next);
    return true;
  }, [safeRois, setRois]);

  // Snap-to-peak (synced to Python for state portability)
  const [snapEnabled, setSnapEnabled] = useModelState<boolean>("snap_enabled");
  const [snapRadius, setSnapRadius] = useModelState<number>("snap_radius");
  const snapActive = snapEnabled && !hideSnap;

  // Colormap, contrast, FFT (synced to Python)
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const effectiveShowFft = showFft && !hideDisplay;
  const roiFftActive = effectiveShowFft && !isGallery && activeRoiIdx >= 0 && activeRoiIdx < safeRois.length;

  // Histogram slider state (local)
  const [vminPct, setVminPct] = React.useState(0);
  const [vmaxPct, setVmaxPct] = React.useState(100);

  const [showColorbar, setShowColorbar] = React.useState(false);

  // Point dragging state
  const draggingPointRef = React.useRef<{ idx: number; imageIdx: number } | null>(null);
  const [hoveredPointIdx, setHoveredPointIdx] = React.useState(-1);

  // ROI undo/redo history
  const roiHistoryRef = React.useRef<ROI[][]>([]);
  const roiRedoRef = React.useRef<ROI[][]>([]);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileLine, setProfileLine] = useModelState<{ row: number; col: number }[]>("profile_line");
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profilePoints = profileLine || [];
  const [hoveredProfileEndpoint, setHoveredProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringProfileLine, setIsHoveringProfileLine] = React.useState(false);
  const profileDragRef = React.useRef<{
    imageIdx: number;
    mode: "endpoint" | "line";
    endpointIdx?: 0 | 1;
    startRow: number;
    startCol: number;
    p0: { row: number; col: number };
    p1: { row: number; col: number };
  } | null>(null);

  // Colorbar vmin/vmax cache (updated by data effect, read by UI effect)
  const colorbarVminRef = React.useRef(0);
  const colorbarVmaxRef = React.useRef(1);

  // FFT refs
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const fftContainerRef = React.useRef<HTMLDivElement | null>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement | null>(null);

  // FFT d-spacing measurement
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);

  // Collapsible advanced options
  const [showAdvanced, setShowAdvanced] = React.useState(false);

  // Refs
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const uiRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const offscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const mainImgDatasRef = React.useRef<ImageData[]>([]);
  const [offscreenVersion, setOffscreenVersion] = React.useState(0);
  const canvasContainerRefs = React.useRef<(HTMLDivElement | null)[]>([]);

  const [hover, setHover] = React.useState<{
    row: number;
    col: number;
    raw?: number;
    norm?: number;
  } | null>(null);

  // Per-image zoom state
  const [zoomStates, setZoomStates] = React.useState<Map<number, ZoomState>>(new Map());
  const getZoom = React.useCallback((idx: number): ZoomState => zoomStates.get(idx) || DEFAULT_ZOOM, [zoomStates]);
  const setZoom = React.useCallback((idx: number, zs: ZoomState) => {
    setZoomStates(prev => new Map(prev).set(idx, zs));
  }, []);

  const dragRef = React.useRef<{
    startX: number;
    startY: number;
    startPanX: number;
    startPanY: number;
    dragging: boolean;
    wasDrag: boolean;
    imageIdx: number;
  } | null>(null);

  // Resize state
  const [mainCanvasSize, setMainCanvasSize] = React.useState(CANVAS_TARGET_SIZE);
  const [galleryCanvasSize, setGalleryCanvasSize] = React.useState(GALLERY_TARGET_SIZE);
  const [isResizing, setIsResizing] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number } | null>(null);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);
  const initialCanvasSizeRef = React.useRef<number>(CANVAS_TARGET_SIZE);

  // Sync initial size when image loads — never shrink below target
  React.useEffect(() => {
    if (width > 0 && height > 0) {
      const sz = canvasSizeTrait > 0
        ? canvasSizeTrait
        : Math.max(CANVAS_TARGET_SIZE, Math.round(Math.max(width, height) * scale));
      if (!isGallery) setMainCanvasSize(sz);
      initialCanvasSizeRef.current = canvasSizeTrait > 0 ? canvasSizeTrait : CANVAS_TARGET_SIZE;
    }
  }, [width, height, scale, isGallery, canvasSizeTrait]);

  // Compute display dimensions
  const targetSize = isGallery ? galleryCanvasSize : mainCanvasSize;
  const displayScale = width > 0 && height > 0 ? targetSize / Math.max(width, height) : 1;
  const canvasW = width > 0 ? Math.round(width * displayScale) : targetSize;
  const canvasH = height > 0 ? Math.round(height * displayScale) : targetSize;
  const contentW = isGallery
    ? ncols * canvasW + (ncols - 1) * 8
    : effectiveShowFft ? canvasW * 2 + SPACING.LG : canvasW;

  // Parse frame_bytes into per-image Float32Arrays
  const floatsPerImage = width * height;
  const perImageData = React.useMemo(() => {
    if (!frameBytes || !width || !height) return [];
    const allFloats = extractFloat32(frameBytes);
    if (!allFloats) return [];
    const result: Float32Array[] = [];
    for (let i = 0; i < nImages; i++) {
      const start = i * floatsPerImage;
      result.push(allFloats.subarray(start, start + floatsPerImage));
    }
    return result;
  }, [frameBytes, nImages, floatsPerImage, width, height]);

  // ROI pixel statistics
  const roiStats = React.useMemo(() => {
    if (hideRoi || !activeRoi || perImageData.length === 0) return null;
    const imgIdx = isGallery ? selectedIdx : 0;
    const data = perImageData[imgIdx];
    if (!data) return null;
    return computeRoiStats(activeRoi, data, width, height);
  }, [activeRoi, perImageData, isGallery, selectedIdx, width, height, hideRoi]);

  // Initialize reusable offscreen canvases (one per image, resized when dimensions change)
  React.useEffect(() => {
    if (width <= 0 || height <= 0 || nImages <= 0) return;
    const canvases: (HTMLCanvasElement | null)[] = [];
    const imgDatas: ImageData[] = [];
    for (let i = 0; i < nImages; i++) {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      canvases.push(canvas);
      imgDatas.push(canvas.getContext("2d")!.createImageData(width, height));
    }
    offscreenRefs.current = canvases;
    mainImgDatasRef.current = imgDatas;
  }, [width, height, nImages]);

  // -------------------------------------------------------------------------
  // Data effect: normalize + colormap → reusable offscreen canvases
  // (does NOT depend on zoom/pan — avoids recomputing colormap on every pan/zoom)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (perImageData.length === 0 || !width || !height) return;
    if (offscreenRefs.current.length === 0 || mainImgDatasRef.current.length === 0) return;
    const lut = COLORMAPS[cmap || "gray"] || COLORMAPS.gray;
    for (let i = 0; i < nImages; i++) {
      const offscreen = offscreenRefs.current[i];
      const imgData = mainImgDatasRef.current[i];
      if (!offscreen || !imgData) continue;
      const f32 = perImageData[i];
      if (!f32) continue;
      const data = logScale ? applyLogScale(f32) : f32;
      let vmin: number, vmax: number;
      if (autoContrast) {
        ({ vmin, vmax } = percentileClip(data, percentileLow, percentileHigh));
      } else {
        const { min: dMin, max: dMax } = findDataRange(data);
        ({ vmin, vmax } = sliderRange(dMin, dMax, vminPct, vmaxPct));
      }
      renderToOffscreenReuse(data, lut, vmin, vmax, offscreen, imgData);
      // Cache vmin/vmax for the currently-displayed image (used by colorbar in UI effect)
      const displayIdx = isGallery ? selectedIdx : 0;
      if (i === displayIdx) {
        colorbarVminRef.current = vmin;
        colorbarVmaxRef.current = vmax;
      }
    }
    setOffscreenVersion(v => v + 1);
  }, [perImageData, nImages, width, height, cmap, autoContrast, logScale, vminPct, vmaxPct, percentileLow, percentileHigh, isGallery, selectedIdx]);

  // Histogram data for current image
  const histogramData = React.useMemo(() => {
    const idx = isGallery ? selectedIdx : 0;
    const f32 = perImageData[idx];
    if (!f32) return null;
    return logScale ? applyLogScale(f32) : f32;
  }, [perImageData, isGallery, selectedIdx, logScale]);

  const dataRange = React.useMemo(() => {
    if (!histogramData) return { min: 0, max: 1 };
    return findDataRange(histogramData);
  }, [histogramData]);

  // Image statistics (Mean/Min/Max/Std) for the active image
  const imageStats = React.useMemo(() => {
    const idx = isGallery ? selectedIdx : 0;
    const f32 = perImageData[idx];
    if (!f32 || f32.length === 0) return null;
    return computeStats(f32);
  }, [perImageData, isGallery, selectedIdx]);

  // Per-image points helpers
  const getPointsForImage = React.useCallback((idx: number): Point[] => {
    if (!isGallery) return (selectedPoints as Point[]) || [];
    const nested = (selectedPoints as Point[][]) || [];
    return nested[idx] || [];
  }, [isGallery, selectedPoints]);

  const setPointsForImage = React.useCallback((idx: number, points: Point[]) => {
    if (!isGallery) {
      setSelectedPoints(points);
    } else {
      setSelectedPoints((prev) => {
        const nested = [...((prev as Point[][]) || [])];
        while (nested.length < nImages) nested.push([]);
        nested[idx] = points;
        return nested;
      });
    }
  }, [isGallery, nImages, setSelectedPoints]);

  // Dot size
  const size = Number.isFinite(dotSize) && dotSize > 0 ? dotSize : 12;

  // -------------------------------------------------------------------------
  // Draw effect: zoom/pan changes — cheap, just drawImage from cached offscreens
  // (does NOT recompute colormap pipeline — only composites cached images + markers)
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  // -------------------------------------------------------------------------
  React.useLayoutEffect(() => {
    if (!width || !height || offscreenRefs.current.length === 0) return;
    for (let i = 0; i < nImages; i++) {
      const canvas = canvasRefs.current[i];
      const offscreen = offscreenRefs.current[i];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      canvas.width = canvasW;
      canvas.height = canvasH;

      const { zoom, panX, panY } = getZoom(i);
      const cx = canvasW / 2;
      const cy = canvasH / 2;

      ctx.clearRect(0, 0, canvasW, canvasH);
      ctx.save();
      ctx.imageSmoothingEnabled = false;

      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);

      ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);

      // Draw points for this image
      if (!hidePoints) {
        const pts = getPointsForImage(i);
        const dotRadius = (size / 2) * displayScale;
        for (let j = 0; j < pts.length; j++) {
          const p = pts[j];
          const px = (p.col / width) * canvasW;
          const py = (p.row / height) * canvasH;
          const color = p.color || MARKER_COLORS[j % MARKER_COLORS.length];
          const shape = p.shape || MARKER_SHAPES[j % MARKER_SHAPES.length];

          drawMarker(ctx, px, py, dotRadius, shape, color, tc.bg, markerOpacity, borderWidth);
        }
      }

      ctx.restore();

    }
  }, [offscreenVersion, width, height, canvasW, canvasH, displayScale, zoomStates, selectedPoints, size, tc.bg, nImages, getZoom, getPointsForImage, markerOpacity, borderWidth, hidePoints]);

  // Render ROI overlays + edge tick marks
  React.useEffect(() => {
    if (!width || !height) return;
    const idx = isGallery ? selectedIdx : 0;
    for (let i = 0; i < nImages; i++) {
      const overlay = overlayRefs.current[i];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      overlay.width = canvasW;
      overlay.height = canvasH;
      ctx.clearRect(0, 0, canvasW, canvasH);
      if (!hideRoi && safeRois.length > 0 && i === idx) {
        const { zoom, panX, panY } = getZoom(i);
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        for (let ri = 0; ri < safeRois.length; ri++) {
          const roi = safeRois[ri];
          const screenX = ((roi.col / width) * canvasW - cx) * zoom + cx + panX;
          const screenY = ((roi.row / height) * canvasH - cy) * zoom + cy + panY;
          const screenRadius = roi.radius * displayScale * zoom;
          const screenW = roi.width * displayScale * zoom;
          const screenH = roi.height * displayScale * zoom;
          const isActive = ri === activeRoiIdx;
          const isHovered = ri === hoveredRoiIdx && !isActive;
          drawROI(ctx, screenX, screenY, roi.shape, screenRadius, screenW, screenH, roi.color, roi.opacity, isActive, isHovered);
        }
      }
      // Line profile overlay
      if (!hideProfile && profileActive && profilePoints.length > 0 && i === idx) {
        const { zoom, panX, panY } = getZoom(i);
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        const toScreenX = (ic: number) => ((ic / width) * canvasW - cx) * zoom + cx + panX;
        const toScreenY = (ir: number) => ((ir / height) * canvasH - cy) * zoom + cy + panY;

        // Draw point A
        const ax = toScreenX(profilePoints[0].col);
        const ay = toScreenY(profilePoints[0].row);
        ctx.fillStyle = tc.accent;
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();

        // Draw line and point B if complete
        if (profilePoints.length === 2) {
          const bx = toScreenX(profilePoints[1].col);
          const by = toScreenY(profilePoints[1].row);
          ctx.strokeStyle = tc.accent;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = tc.accent;
          ctx.beginPath();
          ctx.arc(bx, by, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      // Edge tick marks — short indicators on the 4 canvas edges tracking cursor position
      if (hover && i === idx) {
        const { zoom, panX, panY } = getZoom(i);
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        const imgX = (hover.col / width) * canvasW;
        const imgY = (hover.row / height) * canvasH;
        const screenX = (imgX - cx) * zoom + cx + panX;
        const screenY = (imgY - cy) * zoom + cy + panY;
        const tickLen = 10;
        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
        ctx.lineWidth = 1.5;
        ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
        ctx.shadowBlur = 2;
        ctx.beginPath();
        // Top edge
        ctx.moveTo(screenX, 0);
        ctx.lineTo(screenX, tickLen);
        // Bottom edge
        ctx.moveTo(screenX, canvasH);
        ctx.lineTo(screenX, canvasH - tickLen);
        // Left edge
        ctx.moveTo(0, screenY);
        ctx.lineTo(tickLen, screenY);
        // Right edge
        ctx.moveTo(canvasW, screenY);
        ctx.lineTo(canvasW - tickLen, screenY);
        ctx.stroke();
        // Snap radius indicator
        if (snapActive && snapRadius > 0) {
          const radiusPx = snapRadius * displayScale * zoom;
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = "rgba(0, 200, 255, 0.7)";
          ctx.lineWidth = 1.2;
          ctx.shadowBlur = 0;
          ctx.beginPath();
          ctx.arc(screenX, screenY, radiusPx, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.restore();
      }
    }
  }, [safeRois, activeRoiIdx, hoveredRoiIdx, isDraggingROI, canvasW, canvasH, displayScale, width, height, nImages, isGallery, selectedIdx, getZoom, hover, profileActive, profilePoints, tc.accent, snapActive, snapRadius, hideRoi, hideProfile]);

  // Auto-compute profile when profile_line is set (e.g. from Python)
  React.useEffect(() => {
    if (profilePoints.length === 2 && perImageData.length > 0) {
      const imgIdx = isGallery ? selectedIdx : 0;
      const raw = perImageData[imgIdx];
      if (raw) {
        const p0 = profilePoints[0], p1 = profilePoints[1];
        setProfileData(sampleLineProfile(raw, width, height, p0.row, p0.col, p1.row, p1.col));
        if (!hideProfile && !profileActive) setProfileActive(true);
      }
    }
  }, [profilePoints, perImageData, isGallery, selectedIdx, width, height, hideProfile, profileActive]);

  // Hidden profile controls should never trap clicks in profile mode.
  React.useEffect(() => {
    if (!hideProfile || !profileActive) return;
    setProfileActive(false);
    setProfileLine([]);
    setProfileData(null);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
  }, [hideProfile, profileActive, setProfileLine]);

  // Render sparkline for line profile
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasW;
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
      ctx.fillText("Click two points on the image to draw a profile", cssW / 2, cssH / 2);
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

    ctx.strokeStyle = tc.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < profileData.length; i++) {
      const x = (i / (profileData.length - 1)) * cssW;
      const y = padTop + plotH - ((profileData[i] - gMin) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Compute total distance for x-axis
    let totalDist = profileData.length - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const dx = profilePoints[1].col - profilePoints[0].col;
      const dy = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      const ps = pixelSize || 0;
      if (ps > 0) {
        const distA = distPx * ps;
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

    // Draw y-axis min/max labels
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), 2, 1);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), 2, padTop + plotH - 1);
  }, [profileData, canvasW, themeInfo.theme, tc.accent, profilePoints, pixelSize]);

  // Scale bar + colorbar + marker labels (HiDPI UI overlay)
  React.useEffect(() => {
    if (!width || !height) return;
    const pxSize = pixelSize || 0;
    const unit = pxSize > 0 ? "Å" as const : "px" as const;
    const pxSizeVal = pxSize > 0 ? pxSize : 1;
    const dotRadius = (size / 2) * displayScale;
    for (let i = 0; i < nImages; i++) {
      const uiCanvas = uiRefs.current[i];
      if (!uiCanvas) continue;
      uiCanvas.width = Math.round(canvasW * DPR);
      uiCanvas.height = Math.round(canvasH * DPR);
      const { zoom, panX, panY } = getZoom(i);
      drawScaleBarHiDPI(uiCanvas, DPR, zoom, pxSizeVal, unit, width);

      // Colorbar overlay (reads cached vmin/vmax from data effect — no recomputation)
      if (showColorbar) {
        const lut = COLORMAPS[cmap || "gray"] || COLORMAPS.gray;
        const vmin = colorbarVminRef.current;
        const vmax = colorbarVmaxRef.current;
        const cssW = uiCanvas.width / DPR;
        const cssH = uiCanvas.height / DPR;
        const ctx = uiCanvas.getContext("2d");
        if (ctx) {
          ctx.save();
          ctx.scale(DPR, DPR);
          drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, logScale);
          ctx.restore();
        }
      }

      // Draw marker labels on HiDPI UI canvas (pixel-independent)
      const pts = getPointsForImage(i);
      if (hidePoints) continue;
      if (pts.length === 0) continue;
      const ctx = uiCanvas.getContext("2d");
      if (!ctx) continue;
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const fontSize = (labelSize > 0 ? labelSize : Math.max(10, size * 0.9)) * DPR;
      ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
      ctx.fillStyle = labelColor || tc.text;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.shadowColor = "rgba(0,0,0,0.7)";
      ctx.shadowBlur = 3 * DPR;
      for (let j = 0; j < pts.length; j++) {
        const p = pts[j];
        const imgX = (p.col / width) * canvasW;
        const imgY = (p.row / height) * canvasH;
        const screenX = ((imgX - cx) * zoom + cx + panX) * DPR;
        const screenY = ((imgY - cy) * zoom + cy + panY) * DPR;
        const screenDotR = dotRadius * zoom * DPR;
        ctx.fillText(`${j + 1}`, screenX, screenY - screenDotR - 2 * DPR);
      }
    }
  }, [pixelSize, canvasW, canvasH, width, height, nImages, zoomStates, getZoom, selectedPoints, getPointsForImage, size, displayScale, labelSize, labelColor, tc.text, showColorbar, cmap, logScale, offscreenVersion, hidePoints]);

  // Map screen coordinates to image pixel coordinates
  const clientToImage = React.useCallback(
    (clientX: number, clientY: number, idx: number): { row: number; col: number } | null => {
      const canvas = canvasRefs.current[idx];
      if (!canvas || !width || !height) return null;
      const rect = canvas.getBoundingClientRect();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const { zoom, panX, panY } = getZoom(idx);

      const canvasX = ((clientX - rect.left) / rect.width) * canvasW;
      const canvasY = ((clientY - rect.top) / rect.height) * canvasH;

      const imgDisplayX = (canvasX - cx - panX) / zoom + cx;
      const imgDisplayY = (canvasY - cy - panY) / zoom + cy;

      const col = Math.floor((imgDisplayX / canvasW) * width);
      const row = Math.floor((imgDisplayY / canvasH) * height);
      if (col < 0 || row < 0 || col >= width || row >= height) return null;
      return { row, col };
    },
    [width, height, canvasW, canvasH, getZoom],
  );

  // Init GPU FFT
  React.useEffect(() => {
    getWebGPUFFT().then(fft => { if (fft) gpuFFTRef.current = fft; });
  }, []);

  // Compute FFT when toggled (supports ROI-scoped FFT)
  React.useEffect(() => {
    if (!effectiveShowFft || !width || !height) { fftOffscreenRef.current = null; setFftCropDims(null); return; }
    const idx = isGallery ? selectedIdx : 0;
    const f32 = perImageData[idx];
    if (!f32) return;
    const lut = COLORMAPS[cmap || "gray"] || COLORMAPS.gray;
    const compute = async () => {
      let inputData = f32;
      let fftW = width, fftH = height;
      let origCropW = 0, origCropH = 0;

      // ROI FFT: crop to selected ROI region and pre-pad to power-of-2
      if (roiFftActive && safeRois.length > 0 && activeRoiIdx >= 0 && activeRoiIdx < safeRois.length) {
        const roi = safeRois[activeRoiIdx];
        const crop = cropROIRegion(f32, width, height, roi);
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
      if (gpuFFTRef.current) {
        const gpuReal = inputData.slice();
        const result = await gpuFFTRef.current.fft2D(gpuReal, new Float32Array(inputData.length), fftW, fftH, false);
        real = result.real; imag = result.imag;
      } else {
        real = inputData.slice(); imag = new Float32Array(inputData.length);
        fft2d(real, imag, fftW, fftH, false);
      }
      fftshift(real, fftW, fftH);
      fftshift(imag, fftW, fftH);
      const mag = computeMagnitude(real, imag);
      fftMagCacheRef.current = mag.slice();
      autoEnhanceFFT(mag, fftW, fftH);
      const logMag = applyLogScale(mag);
      const { min: logMin, max: logMax } = findDataRange(logMag);
      fftOffscreenRef.current = renderToOffscreen(logMag, fftW, fftH, lut, logMin, logMax);
      // Track FFT dimensions when they differ from image dimensions (ROI crop or non-pow2 padding)
      if (origCropW > 0) {
        setFftCropDims({ cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH });
      } else if (fftW !== width || fftH !== height) {
        setFftCropDims({ cropWidth: width, cropHeight: height, fftWidth: fftW, fftHeight: fftH });
      } else {
        setFftCropDims(null);
      }
      // Trigger redraw
      if (fftCanvasRef.current && fftOffscreenRef.current) {
        const ctx = fftCanvasRef.current.getContext("2d");
        if (ctx) {
          fftCanvasRef.current.width = canvasW;
          fftCanvasRef.current.height = canvasH;
          ctx.imageSmoothingEnabled = fftW < canvasW || fftH < canvasH;
          ctx.clearRect(0, 0, canvasW, canvasH);
          ctx.drawImage(fftOffscreenRef.current, 0, 0, canvasW, canvasH);
        }
      }
    };
    compute();
  }, [effectiveShowFft, roiFftActive, perImageData, isGallery, selectedIdx, width, height, cmap, canvasW, canvasH, safeRois, activeRoiIdx, fftWindow]);

  // Clear FFT measurement when image, FFT state, or ROI changes
  React.useEffect(() => { setFftClickInfo(null); }, [selectedIdx, effectiveShowFft, roiFftActive, activeRoiIdx]);

  // Render FFT overlay (reciprocal-space scale bar + colorbar + d-spacing marker)
  React.useEffect(() => {
    const overlay = fftOverlayRef.current;
    if (!overlay || !effectiveShowFft) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    overlay.width = Math.round(canvasW * DPR);
    overlay.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Reciprocal-space scale bar (use crop dims when ROI FFT active)
    if (pixelSize && pixelSize > 0) {
      const fftW = fftCropDims?.fftWidth ?? width;
      const fftPixelSize = 1 / (fftW * pixelSize);
      drawFFTScaleBarHiDPI(overlay, DPR, 1, fftPixelSize, fftW);
    }

    // D-spacing crosshair marker
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    if (fftClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const screenX = fftClickInfo.col / fftW * canvasW;
      const screenY = fftClickInfo.row / fftH * canvasH;
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

  }, [effectiveShowFft, canvasW, canvasH, pixelSize, width, height, fftCropDims, fftClickInfo]);

  // Convert FFT canvas mouse position to FFT image pixel coordinates
  const fftScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    const imgCol = (mouseX / canvasW) * fftW;
    const imgRow = (mouseY / canvasH) * fftH;
    if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
      return { col: imgCol, row: imgRow };
    }
    return null;
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleFftMouseUp = (e: React.MouseEvent) => {
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const pos = fftScreenToImg(e);
        if (pos) {
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
  };

  const handleFftMouseLeave = () => {
    fftClickStartRef.current = null;
  };

  // Prevent page scroll on canvas containers
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const containers = [...canvasContainerRefs.current.filter(Boolean)];
    if (fftContainerRef.current) containers.push(fftContainerRef.current);
    containers.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => {
      containers.forEach(el => el?.removeEventListener("wheel", preventDefault));
    };
  }, [nImages, effectiveShowFft]);

  // Scroll to zoom
  const handleWheel = React.useCallback(
    (e: React.WheelEvent, idx: number) => {
      if (lockView) return;
      e.preventDefault();
      if (isGallery && idx !== selectedIdx) return;
      const canvas = canvasRefs.current[idx];
      if (!canvas || !width || !height) return;
      const rect = canvas.getBoundingClientRect();

      const mouseX = ((e.clientX - rect.left) / rect.width) * canvasW;
      const mouseY = ((e.clientY - rect.top) / rect.height) * canvasH;

      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      const prev = getZoom(idx);
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, prev.zoom * factor));
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const wx = (mouseX - cx - prev.panX) / prev.zoom + cx;
      const wy = (mouseY - cy - prev.panY) / prev.zoom + cy;
      const newPanX = mouseX - cx - (wx - cx) * newZoom;
      const newPanY = mouseY - cy - (wy - cy) * newZoom;
      setZoom(idx, { zoom: newZoom, panX: newPanX, panY: newPanY });
    },
    [width, height, canvasW, canvasH, isGallery, selectedIdx, getZoom, setZoom, lockView],
  );

  // Track when we just switched focus (don't place a point on the same click)
  const justSwitchedRef = React.useRef(false);

  // Mouse down
  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent, idx: number) => {
      if (e.button !== 0) return;
      justSwitchedRef.current = false;
      if (isGallery && idx !== selectedIdx) {
        if (lockNavigation) return;
        setSelectedIdx(idx);
        justSwitchedRef.current = true;
        return;
      }
      if (!hideProfile && profileActive && !lockProfile) {
        const coords = clientToImage(e.clientX, e.clientY, idx);
        if (coords && profilePoints.length === 2) {
          const { zoom } = getZoom(idx);
          const hitDist = 10 / (displayScale * zoom);
          const p0 = profilePoints[0];
          const p1 = profilePoints[1];
          const d0 = Math.sqrt((coords.col - p0.col) ** 2 + (coords.row - p0.row) ** 2);
          const d1 = Math.sqrt((coords.col - p1.col) ** 2 + (coords.row - p1.row) ** 2);
          if (d0 <= hitDist || d1 <= hitDist) {
            profileDragRef.current = {
              imageIdx: idx,
              mode: "endpoint",
              endpointIdx: d0 <= d1 ? 0 : 1,
              startRow: coords.row,
              startCol: coords.col,
              p0: { row: p0.row, col: p0.col },
              p1: { row: p1.row, col: p1.col },
            };
            return;
          }
          if (pointToSegmentDistance(coords.col, coords.row, p0.col, p0.row, p1.col, p1.row) <= hitDist) {
            profileDragRef.current = {
              imageIdx: idx,
              mode: "line",
              startRow: coords.row,
              startCol: coords.col,
              p0: { row: p0.row, col: p0.col },
              p1: { row: p1.row, col: p1.col },
            };
            return;
          }
        }
      }
      // Check if click is near any ROI center to drag it
      if (!lockRoi && safeRois.length > 0) {
        const coords = clientToImage(e.clientX, e.clientY, idx);
        if (coords) {
          const { zoom } = getZoom(idx);
          const hitDist = 20 / (displayScale * zoom);
          let hitIdx = -1;
          let bestDist = Infinity;
          for (let ri = 0; ri < safeRois.length; ri++) {
            const dx = coords.col - safeRois[ri].col;
            const dy = coords.row - safeRois[ri].row;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < hitDist && dist < bestDist) {
              bestDist = dist;
              hitIdx = ri;
            }
          }
          if (hitIdx >= 0) {
            pushRoiHistory();
            setActiveRoiIdx(hitIdx);
            setIsDraggingROI(true);
            return;
          }
        }
      }
      // Check if click is near any existing point to drag it
      if (!lockPoints) {
        const coords = clientToImage(e.clientX, e.clientY, idx);
        if (coords) {
          const pts = getPointsForImage(idx);
          const { zoom } = getZoom(idx);
          const hitDist = 15 / (displayScale * zoom);
          let bestPtIdx = -1;
          let bestDist = Infinity;
          for (let pi = 0; pi < pts.length; pi++) {
            const dx = coords.col - pts[pi].col;
            const dy = coords.row - pts[pi].row;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < hitDist && dist < bestDist) {
              bestDist = dist;
              bestPtIdx = pi;
            }
          }
          if (bestPtIdx >= 0) {
            draggingPointRef.current = { idx: bestPtIdx, imageIdx: idx };
            return;
          }
        }
      }
      if (lockView) {
        dragRef.current = null;
        return;
      }
      const zs = getZoom(idx);
      dragRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        startPanX: zs.panX,
        startPanY: zs.panY,
        dragging: false,
        wasDrag: false,
        imageIdx: idx,
      };
    },
    [
      isGallery,
      selectedIdx,
      setSelectedIdx,
      getZoom,
      safeRois,
      clientToImage,
      displayScale,
      getPointsForImage,
      pushRoiHistory,
      lockNavigation,
      hideProfile,
      profileActive,
      profilePoints,
      lockProfile,
      lockRoi,
      lockPoints,
      lockView,
    ],
  );

  // Mouse move
  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent, idx: number) => {
      if (!hideProfile && profileActive && !lockProfile && profileDragRef.current?.imageIdx === idx && profilePoints.length === 2) {
        const coords = clientToImage(e.clientX, e.clientY, idx);
        if (coords) {
          const drag = profileDragRef.current;
          if (drag.mode === "endpoint" && drag.endpointIdx !== undefined) {
            const clampedRow = Math.max(0, Math.min(height - 1, coords.row));
            const clampedCol = Math.max(0, Math.min(width - 1, coords.col));
            const next = [
              drag.endpointIdx === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
              drag.endpointIdx === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
            ];
            setProfileLine(next);
            const imgIdx = isGallery ? selectedIdx : idx;
            const raw = perImageData[imgIdx];
            if (raw) {
              setProfileData(sampleLineProfile(raw, width, height, next[0].row, next[0].col, next[1].row, next[1].col));
            }
          } else if (drag.mode === "line") {
            let deltaRow = coords.row - drag.startRow;
            let deltaCol = coords.col - drag.startCol;
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
            const imgIdx = isGallery ? selectedIdx : idx;
            const raw = perImageData[imgIdx];
            if (raw) {
              setProfileData(sampleLineProfile(raw, width, height, next[0].row, next[0].col, next[1].row, next[1].col));
            }
          }
        }
        return;
      }
      if (!hideProfile && profileActive && profilePoints.length === 2) {
        const coords = clientToImage(e.clientX, e.clientY, idx);
        if (coords) {
          const { zoom } = getZoom(idx);
          const hitDist = 10 / (displayScale * zoom);
          const p0 = profilePoints[0];
          const p1 = profilePoints[1];
          const d0 = Math.sqrt((coords.col - p0.col) ** 2 + (coords.row - p0.row) ** 2);
          const d1 = Math.sqrt((coords.col - p1.col) ** 2 + (coords.row - p1.row) ** 2);
          const nextHoveredEndpoint: 0 | 1 | null = d0 <= hitDist ? 0 : d1 <= hitDist ? 1 : null;
          const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(coords.col, coords.row, p0.col, p0.row, p1.col, p1.row) <= hitDist;
          setHoveredProfileEndpoint(nextHoveredEndpoint);
          setIsHoveringProfileLine(nextHoverLine);
        }
      } else {
        if (hoveredProfileEndpoint !== null) setHoveredProfileEndpoint(null);
        if (isHoveringProfileLine) setIsHoveringProfileLine(false);
      }

      // Point dragging
      if (!lockPoints && draggingPointRef.current && draggingPointRef.current.imageIdx === idx) {
        const coords = clientToImage(e.clientX, e.clientY, idx);
        if (coords) {
          const pts = getPointsForImage(idx);
          const pi = draggingPointRef.current.idx;
          if (pi < pts.length) {
            const updated = pts.map((p, j) => j === pi ? { ...p, row: coords.row, col: coords.col } : p);
            setPointsForImage(idx, updated);
          }
        }
        return;
      }
      if (!lockRoi && isDraggingROI && activeRoiIdx >= 0) {
        const coords = clientToImage(e.clientX, e.clientY, idx);
        if (coords) {
          setRois(safeRois.map((r, i) => i === activeRoiIdx ? { ...r, row: coords.row, col: coords.col } : r));
        }
        return;
      }
      const drag = dragRef.current;
      if (drag && drag.imageIdx === idx) {
        const dx = e.clientX - drag.startX;
        const dy = e.clientY - drag.startY;
        if (!drag.dragging && Math.abs(dx) + Math.abs(dy) > DRAG_THRESHOLD) {
          drag.dragging = true;
        }
        if (drag.dragging) {
          if (lockView) return;
          drag.wasDrag = true;
          const canvas = canvasRefs.current[idx];
          if (!canvas) return;
          const rect = canvas.getBoundingClientRect();
          const scaleX = canvasW / rect.width;
          const scaleY = canvasH / rect.height;
          setZoom(idx, {
            zoom: getZoom(idx).zoom,
            panX: drag.startPanX + dx * scaleX,
            panY: drag.startPanY + dy * scaleY,
          });
          return;
        }
      }

      // Hover readout (only for selected image in gallery)
      if (isGallery && idx !== selectedIdx) return;
      const p = clientToImage(e.clientX, e.clientY, idx);
      if (!p) { setHover(null); return; }
      let raw: number | undefined;
      let norm: number | undefined;
      const f32 = perImageData[idx];
      if (f32) {
        raw = f32[p.row * width + p.col];
        const min = imgMin?.[idx] ?? 0;
        const max = imgMax?.[idx] ?? 1;
        const denom = max > min ? max - min : 1;
        norm = (raw - min) / denom;
      }
      setHover({ row: p.row, col: p.col, raw, norm });

      // ROI hover detection
      if (safeRois.length > 0) {
        const { zoom } = getZoom(idx);
        const hitDist = 20 / (displayScale * zoom);
        let hitIdx = -1;
        let bestDist = Infinity;
        for (let ri = 0; ri < safeRois.length; ri++) {
          const dx = p.col - safeRois[ri].col;
          const dy = p.row - safeRois[ri].row;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < hitDist && dist < bestDist) {
            bestDist = dist;
            hitIdx = ri;
          }
        }
        setHoveredRoiIdx(hitIdx);
      } else {
        setHoveredRoiIdx(-1);
      }

      // Point hover detection
      const pts = getPointsForImage(idx);
      if (pts.length > 0) {
        const { zoom } = getZoom(idx);
        const hitDist = 15 / (displayScale * zoom);
        let bestPtIdx = -1;
        let bestDist = Infinity;
        for (let pi = 0; pi < pts.length; pi++) {
          const dx = p.col - pts[pi].col;
          const dy = p.row - pts[pi].row;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < hitDist && dist < bestDist) {
            bestDist = dist;
            bestPtIdx = pi;
          }
        }
        setHoveredPointIdx(bestPtIdx);
      } else {
        setHoveredPointIdx(-1);
      }
    },
    [
      clientToImage,
      width,
      canvasW,
      canvasH,
      perImageData,
      imgMin,
      imgMax,
      isGallery,
      selectedIdx,
      getZoom,
      setZoom,
      isDraggingROI,
      activeRoiIdx,
      safeRois,
      setRois,
      hideProfile,
      profileActive,
      lockProfile,
      profilePoints,
      setProfileLine,
      setProfileData,
      hoveredProfileEndpoint,
      isHoveringProfileLine,
      displayScale,
      getPointsForImage,
      setPointsForImage,
      lockPoints,
      lockRoi,
      lockView,
      isGallery,
      selectedIdx,
    ],
  );

  // Mouse up — place point
  const handleMouseUp = React.useCallback(
    (e: React.MouseEvent, idx: number) => {
      if (profileDragRef.current) {
        profileDragRef.current = null;
        setHoveredProfileEndpoint(null);
        setIsHoveringProfileLine(false);
        return;
      }
      if (draggingPointRef.current) {
        draggingPointRef.current = null;
        return;
      }
      if (isDraggingROI) {
        setIsDraggingROI(false);
        return;
      }
      const drag = dragRef.current;
      dragRef.current = null;
      if (drag?.wasDrag) return;
      if (justSwitchedRef.current) { justSwitchedRef.current = false; return; }
      if (isGallery && idx !== selectedIdx) return;

      const coords = clientToImage(e.clientX, e.clientY, idx);
      if (!coords) return;

      // Profile mode: place profile endpoints instead of markers
      if (!hideProfile && profileActive) {
        if (lockProfile) return;
        const pt = { row: coords.row, col: coords.col };
        if (profilePoints.length === 0 || profilePoints.length === 2) {
          setProfileLine([pt]);
          setProfileData(null);
        } else {
          const p0 = profilePoints[0];
          setProfileLine([p0, pt]);
          const imgIdx = isGallery ? selectedIdx : 0;
          const raw = perImageData[imgIdx];
          if (raw) {
            setProfileData(sampleLineProfile(raw, width, height, p0.row, p0.col, pt.row, pt.col));
          }
        }
        return;
      }

      if (lockPoints) return;

      // Snap to local intensity peak if enabled
      let snappedCoords = coords;
      if (snapActive && perImageData[idx]) {
        const snapped = findLocalMax(perImageData[idx], width, height, coords.col, coords.row, snapRadius);
        snappedCoords = { row: snapped.row, col: snapped.col };
      }
      redoStackRef.current.set(idx, []); // clear redo on new point
      const p: Point = { row: snappedCoords.row, col: snappedCoords.col, shape: (currentShape || "circle") as MarkerShape, color: currentColor || MARKER_COLORS[0] };
      const currentPts = getPointsForImage(idx);
      const limit = Number.isFinite(maxPoints) && maxPoints > 0 ? maxPoints : 3;
      const next = [...currentPts, p];
      setPointsForImage(idx, next.length <= limit ? next : next.slice(next.length - limit));
    },
    [
      clientToImage,
      maxPoints,
      isGallery,
      selectedIdx,
      getPointsForImage,
      setPointsForImage,
      isDraggingROI,
      currentShape,
      currentColor,
      snapActive,
      snapRadius,
      perImageData,
      width,
      height,
      profileActive,
      hideProfile,
      profilePoints,
      setProfileLine,
      setProfileData,
      lockProfile,
      lockPoints,
      setHoveredProfileEndpoint,
      setIsHoveringProfileLine,
    ],
  );

  // Double-click — reset zoom
  const handleDoubleClick = React.useCallback((idx: number) => {
    if (lockView) return;
    if (isGallery && idx !== selectedIdx) return;
    setZoom(idx, DEFAULT_ZOOM);
  }, [isGallery, selectedIdx, setZoom, lockView]);

  const handleExport = React.useCallback(async () => {
    if (lockExport) return;
    setExportAnchor(null);
    const idx = isGallery ? selectedIdx : 0;
    const label = isGallery && labels?.[idx] ? labels[idx] : "mark2d";
    const prefix = `${label}_${width}x${height}`;

    const canvasToBlob = (c: HTMLCanvasElement): Promise<Blob> =>
      new Promise((resolve) => c.toBlob((b) => resolve(b!), "image/png"));

    const mainCanvas = canvasRefs.current[idx];
    const overlay = overlayRefs.current[idx];
    const ui = uiRefs.current[idx];

    if (!effectiveShowFft) {
      // Single annotated PNG (no FFT)
      if (mainCanvas) {
        const comp = document.createElement("canvas");
        comp.width = canvasW;
        comp.height = canvasH;
        const ctx = comp.getContext("2d");
        if (ctx) {
          ctx.drawImage(mainCanvas, 0, 0);
          if (overlay) ctx.drawImage(overlay, 0, 0);
          if (ui) ctx.drawImage(ui, 0, 0, canvasW, canvasH);
          downloadBlob(await canvasToBlob(comp), `${prefix}.png`);
        }
      }
    } else {
      // ZIP with raw + annotated + FFT
      const zip = new JSZip();
      const metadata = {
        metadata_version: "1.0",
        widget_name: "Mark2D",
        widget_version: widgetVersion || "unknown",
        exported_at: new Date().toISOString(),
        format: "zip",
        export_kind: "raw_annotated_fft_bundle",
        selected_idx: idx,
        image_shape: { rows: height, cols: width },
        display: { cmap, log_scale: logScale, auto_contrast: autoContrast },
      };
      zip.file("metadata.json", JSON.stringify(metadata, null, 2));

      const offscreen = offscreenRefs.current[idx];
      if (offscreen) {
        const raw = document.createElement("canvas");
        raw.width = width;
        raw.height = height;
        const rawCtx = raw.getContext("2d");
        if (rawCtx) {
          rawCtx.drawImage(offscreen, 0, 0);
          zip.file(`${prefix}_raw.png`, await canvasToBlob(raw));
        }
      }

      if (mainCanvas) {
        const comp = document.createElement("canvas");
        comp.width = canvasW;
        comp.height = canvasH;
        const ctx = comp.getContext("2d");
        if (ctx) {
          ctx.drawImage(mainCanvas, 0, 0);
          if (overlay) ctx.drawImage(overlay, 0, 0);
          if (ui) ctx.drawImage(ui, 0, 0, canvasW, canvasH);
          zip.file(`${prefix}_annotated.png`, await canvasToBlob(comp));
        }
      }

      const fftCanvas = fftCanvasRef.current;
      if (fftCanvas) {
        zip.file(`${prefix}_fft.png`, await canvasToBlob(fftCanvas));
      }

      const blob = await zip.generateAsync({ type: "blob" });
      downloadBlob(blob, `${prefix}.zip`);
    }
  }, [isGallery, selectedIdx, labels, width, height, canvasW, canvasH, effectiveShowFft, lockExport, widgetVersion, cmap, logScale, autoContrast]);

  const handleExportFigure = React.useCallback((withColorbar: boolean) => {
    if (lockExport) return;
    setExportAnchor(null);
    const idx = isGallery ? selectedIdx : 0;
    const rawData = perImageData[idx];
    if (!rawData || !width || !height) return;

    const processed = logScale ? applyLogScale(rawData) : rawData;
    const lut = COLORMAPS[cmap || "gray"] || COLORMAPS.gray;

    let vmin: number, vmax: number;
    if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, percentileLow, percentileHigh));
    } else {
      const { min: dMin, max: dMax } = findDataRange(processed);
      ({ vmin, vmax } = sliderRange(dMin, dMax, vminPct, vmaxPct));
    }

    const offscreen = renderToOffscreen(processed, width, height, lut, vmin, vmax);
    if (!offscreen) return;

    const pts = getPointsForImage(idx);
    const dotRadius = size / 2;
    const label = isGallery && labels?.[idx] ? labels[idx] : "mark2d";

    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title || undefined,
      lut,
      vmin,
      vmax,
      logScale,
      pixelSize: pixelSize > 0 ? pixelSize : undefined,
      showColorbar: withColorbar,
      showScaleBar: pixelSize > 0,
      drawAnnotations: (ctx) => {
        if (hidePoints) return;
        // Draw markers at native image resolution
        for (let j = 0; j < pts.length; j++) {
          const p = pts[j];
          const color = p.color || MARKER_COLORS[j % MARKER_COLORS.length];
          const shape = p.shape || MARKER_SHAPES[j % MARKER_SHAPES.length];
          drawMarker(ctx, p.col, p.row, dotRadius, shape, color, "white", markerOpacity, borderWidth);
        }
      },
    });

    canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, `${label}_figure.pdf`));
  }, [isGallery, selectedIdx, perImageData, width, height, cmap, logScale, autoContrast,
      percentileLow, percentileHigh, vminPct, vmaxPct, getPointsForImage, size,
      title, pixelSize, markerOpacity, borderWidth, labels, lockExport, hidePoints]);

  const activeIdx = isGallery ? selectedIdx : 0;

  // Redo stack: per-image undone points
  const redoStackRef = React.useRef<Map<number, Point[]>>(new Map());

  const resetPoints = React.useCallback(() => {
    if (lockPoints) return;
    if (!isGallery) {
      setSelectedPoints([]);
    } else {
      setSelectedPoints([...Array(nImages)].map(() => []));
    }
    setHover(null);
    setDotSize(12);
    setMaxPoints(10);
    redoStackRef.current = new Map();
  }, [isGallery, nImages, setSelectedPoints, setDotSize, setMaxPoints, lockPoints]);

  const undoPoint = React.useCallback(() => {
    if (lockPoints) return;
    const pts = getPointsForImage(activeIdx);
    if (pts.length === 0) return;
    const removed = pts[pts.length - 1];
    const stack = redoStackRef.current.get(activeIdx) || [];
    redoStackRef.current.set(activeIdx, [...stack, removed]);
    setPointsForImage(activeIdx, pts.slice(0, -1));
  }, [activeIdx, getPointsForImage, setPointsForImage, lockPoints]);

  const redoPoint = React.useCallback(() => {
    if (lockPoints) return;
    const stack = redoStackRef.current.get(activeIdx) || [];
    if (stack.length === 0) return;
    const point = stack[stack.length - 1];
    redoStackRef.current.set(activeIdx, stack.slice(0, -1));
    const pts = getPointsForImage(activeIdx);
    const limit = Number.isFinite(maxPoints) && maxPoints > 0 ? maxPoints : 3;
    if (pts.length < limit) {
      setPointsForImage(activeIdx, [...pts, point]);
    }
  }, [activeIdx, getPointsForImage, setPointsForImage, maxPoints, lockPoints]);

  const canRedo = (redoStackRef.current.get(activeIdx) || []).length > 0;

  // Keyboard shortcuts
  const handleKeyDown = React.useCallback((e: React.KeyboardEvent) => {
    const isMeta = e.metaKey || e.ctrlKey;
    switch (e.key) {
      case "Delete":
      case "Backspace":
        if (activeRoi && !lockRoi) {
          e.preventDefault();
          pushRoiHistory();
          const next = safeRois.filter((_, i) => i !== activeRoiIdx);
          setActiveRoiIdx(next.length === 0 ? -1 : Math.min(activeRoiIdx, next.length - 1));
          setRois(next);
        } else if (!lockPoints) {
          e.preventDefault();
          undoPoint();
        }
        break;
      case "z":
      case "Z":
        if (isMeta && e.shiftKey) {
          e.preventDefault();
          if (!lockRoi && redoRoi()) break;
          if (!lockPoints) redoPoint();
        } else if (isMeta) {
          e.preventDefault();
          if (!lockPoints) undoPoint();
          if (!lockRoi && getPointsForImage(isGallery ? selectedIdx : 0).length === 0) {
            undoRoi();
          }
        }
        break;
      case "1": case "2": case "3": case "4": case "5": case "6": {
        if (lockRoi) break;
        const roiIdx = parseInt(e.key) - 1;
        if (roiIdx < safeRois.length) { e.preventDefault(); setActiveRoiIdx(roiIdx); }
        break;
      }
      case "ArrowLeft":
        if (activeRoi && !lockRoi) {
          e.preventDefault();
          const step = e.shiftKey ? 10 : 1;
          updateActiveRoi({ col: Math.max(0, activeRoi.col - step) });
        } else if (isGallery && !lockNavigation) {
          e.preventDefault();
          setSelectedIdx(Math.max(0, selectedIdx - 1));
        }
        break;
      case "ArrowRight":
        if (activeRoi && !lockRoi) {
          e.preventDefault();
          const step = e.shiftKey ? 10 : 1;
          updateActiveRoi({ col: Math.min(width - 1, activeRoi.col + step) });
        } else if (isGallery && !lockNavigation) {
          e.preventDefault();
          setSelectedIdx(Math.min(nImages - 1, selectedIdx + 1));
        }
        break;
      case "ArrowUp":
        if (activeRoi && !lockRoi) {
          e.preventDefault();
          const step = e.shiftKey ? 10 : 1;
          updateActiveRoi({ row: Math.max(0, activeRoi.row - step) });
        }
        break;
      case "ArrowDown":
        if (activeRoi && !lockRoi) {
          e.preventDefault();
          const step = e.shiftKey ? 10 : 1;
          updateActiveRoi({ row: Math.min(height - 1, activeRoi.row + step) });
        }
        break;
      case "Escape":
        e.preventDefault();
        setActiveRoiIdx(-1);
        break;
      case "r":
      case "R":
        if (lockView) break;
        handleDoubleClick(activeIdx);
        break;
    }
  }, [
    undoPoint,
    redoPoint,
    undoRoi,
    redoRoi,
    safeRois,
    activeRoi,
    activeRoiIdx,
    updateActiveRoi,
    pushRoiHistory,
    setActiveRoiIdx,
    setRois,
    width,
    height,
    isGallery,
    selectedIdx,
    nImages,
    setSelectedIdx,
    getPointsForImage,
    handleDoubleClick,
    activeIdx,
    lockRoi,
    lockPoints,
    lockNavigation,
    lockView,
  ]);

  // Resize handlers
  const handleResizeStart = (e: React.MouseEvent) => {
    if (lockView) return;
    e.stopPropagation();
    e.preventDefault();
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: isGallery ? galleryCanvasSize : mainCanvasSize });
  };

  React.useEffect(() => {
    if (!isResizing) return;
    let rafId = 0;
    let latestSize = resizeStart ? resizeStart.size : (isGallery ? galleryCanvasSize : mainCanvasSize);
    const setSize = isGallery ? setGalleryCanvasSize : setMainCanvasSize;

    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      const minSize = isGallery ? 100 : initialCanvasSizeRef.current;
      latestSize = Math.max(minSize, resizeStart.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setSize(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      cancelAnimationFrame(rafId);
      setSize(latestSize);
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

  const activeZoom = getZoom(activeIdx);
  const needsReset = activeZoom.zoom !== 1 || activeZoom.panX !== 0 || activeZoom.panY !== 0;
  const maxPtsVal = Number.isFinite(maxPoints) && maxPoints > 0 ? maxPoints : 3;
  const activePts = getPointsForImage(activeIdx);
  const hasAnyPoints = isGallery
    ? (selectedPoints as Point[][])?.some((pts) => pts?.length > 0)
    : (selectedPoints as Point[])?.length > 0;

  // Pre-compute stats for all ROIs (avoids recomputing in render on every mousemove)
  const allRoiStats = React.useMemo(() => {
    if (hideRoi || safeRois.length < 2 || perImageData.length === 0) return [];
    const imgIdx = isGallery ? selectedIdx : 0;
    const data = perImageData[imgIdx];
    if (!data) return [];
    return safeRois.map(roi => computeRoiStats(roi, data, width, height));
  }, [safeRois, perImageData, isGallery, selectedIdx, width, height, hideRoi]);

  // Render a single canvas box (shared between single and gallery mode)
  const renderCanvasBox = (idx: number, showResizeHandle: boolean) => (
    <Box
      ref={(el: HTMLDivElement | null) => { canvasContainerRefs.current[idx] = el; }}
      sx={{
        ...containerStyles.imageBox,
        width: canvasW,
        height: canvasH,
        cursor: isGallery && idx !== selectedIdx
          ? (lockNavigation ? "default" : "pointer")
          : isDraggingROI || draggingPointRef.current || profileDragRef.current ? "grabbing"
          : hoveredPointIdx >= 0 ? "move"
          : profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine) ? "grab"
          : hoveredRoiIdx >= 0 ? "grab"
          : snapActive && !lockSnap ? "cell"
          : lockPoints && lockRoi ? "default" : "crosshair",
        border: isGallery && idx === selectedIdx
          ? `3px solid ${tc.accent}`
          : containerStyles.imageBox.border,
        borderRadius: 0,
      }}
      onMouseDown={(e) => handleMouseDown(e, idx)}
      onMouseMove={(e) => handleMouseMove(e, idx)}
      onMouseUp={(e) => handleMouseUp(e, idx)}
      onMouseLeave={() => { dragRef.current = null; draggingPointRef.current = null; profileDragRef.current = null; setHover(null); setIsDraggingROI(false); setHoveredRoiIdx(-1); setHoveredPointIdx(-1); setHoveredProfileEndpoint(null); setIsHoveringProfileLine(false); }}
      onWheel={(e) => handleWheel(e, idx)}
      onDoubleClick={() => handleDoubleClick(idx)}
    >
      <canvas
        ref={(el) => { canvasRefs.current[idx] = el; }}
        width={canvasW}
        height={canvasH}
        style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }}
      />
      <canvas
        ref={(el) => { overlayRefs.current[idx] = el; }}
        width={canvasW}
        height={canvasH}
        style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
      />
      <canvas
        ref={(el) => { uiRefs.current[idx] = el; }}
        width={Math.round(canvasW * DPR)}
        height={Math.round(canvasH * DPR)}
        style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
      />
      {showResizeHandle && (
        <Box
          onMouseDown={handleResizeStart}
          sx={{
            position: "absolute",
            bottom: 0,
            right: 0,
            width: 16,
            height: 16,
            cursor: "nwse-resize",
            opacity: 0.6,
            background: `linear-gradient(135deg, transparent 50%, ${tc.accent} 50%)`,
            "&:hover": { opacity: 1 },
          }}
        />
      )}
    </Box>
  );

  return (
    <Box className="mark2d-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ ...containerStyles.root, bgcolor: tc.bg, color: tc.text, outline: "none" }}>
      {/* Header row */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28, maxWidth: contentW }}>
        <Typography variant="caption" sx={{ ...typography.label, color: tc.text }}>
          {title || "Mark2D"}
          <InfoTooltip theme={themeInfo.theme} text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
            <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Scale: Linear or logarithmic intensity mapping. Log emphasizes low-intensity features.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile-based contrast — clips outliers for better visibility.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Snap: Snap to local intensity maximum. Positions jump to brightest pixel within search radius R.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Line profile mode — click two points to sample intensity along a line.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>ROI: Region of Interest — click ADD to place, drag to reposition. Shows pixel statistics.</Typography>
            <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
            <KeyboardShortcuts items={[["Click", "Place point"], ["Drag point", "Reposition"], ["\u2318Z / Ctrl+Z", "Undo"], ["\u2318\u21E7Z", "Redo"], ["\u232B / Del", "Delete ROI or undo"], ["1 \u2013 6", "Select ROI"], ["\u2190 / \u2192", "Nudge ROI / Nav gallery"], ["\u21E7 + \u2190\u2192", "Nudge ROI 10 px"], ["R", "Reset zoom"], ["Esc", "Deselect ROI"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
          </Box>} />
          {isGallery && labels?.[activeIdx] && (
            <Box component="span" sx={{ color: tc.textMuted, ml: 1 }}>
              {labels[activeIdx]}
            </Box>
          )}
          {!hidePoints && activePts.length > 0 && (
            <Box component="span" sx={{ color: tc.textMuted, ml: 1 }}>
              {activePts.length}/{maxPtsVal} pts
            </Box>
          )}
        </Typography>
        <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
          {!hideDisplay && (
            <>
              <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>FFT:</Typography>
              <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} disabled={lockDisplay} size="small" sx={switchStyles.small} />
            </>
          )}
          {!hideExport && (
            <>
              <Button size="small" disabled={lockExport} sx={{ ...compactButton, color: tc.accent }} onClick={async () => {
                if (lockExport) return;
                const idx = isGallery ? selectedIdx : 0;
                const canvas = canvasRefs.current[idx];
                if (!canvas) return;
                try {
                  const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/png"));
                  if (!blob) return;
                  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                } catch {
                  canvas.toBlob((b) => { if (b) downloadBlob(b, `mark2d_${labels?.[idx] || "image"}.png`); }, "image/png");
                }
              }}>COPY</Button>
              <Button size="small" disabled={lockExport} sx={{ ...compactButton, color: tc.accent }} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }}>Export</Button>
              <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem disabled={lockExport} onClick={() => handleExportFigure(true)} sx={{ fontSize: 12 }}>PDF + colorbar</MenuItem>
                <MenuItem disabled={lockExport} onClick={() => handleExportFigure(false)} sx={{ fontSize: 12 }}>PDF</MenuItem>
                <MenuItem disabled={lockExport} onClick={handleExport} sx={{ fontSize: 12 }}>PNG</MenuItem>
              </Menu>
            </>
          )}
          {!hidePoints && (
            <>
              <Button size="small" sx={compactButton} onClick={undoPoint} disabled={lockPoints || !activePts.length}>UNDO</Button>
              <Button size="small" sx={compactButton} onClick={redoPoint} disabled={lockPoints || !canRedo}>REDO</Button>
              <Button size="small" sx={compactButton} onClick={resetPoints} disabled={lockPoints || !hasAnyPoints}>RESET ALL</Button>
            </>
          )}
          {!hideView && (
            <Button size="small" sx={compactButton} disabled={lockView || !needsReset} onClick={() => handleDoubleClick(activeIdx)}>RESET VIEW</Button>
          )}
        </Stack>
      </Stack>

      {/* Canvas area + FFT side-by-side */}
      {isGallery ? (
        <Box sx={{ display: "inline-grid", gridTemplateColumns: `repeat(${ncols}, ${canvasW}px)`, gap: 1 }}>
          {Array.from({ length: nImages }).map((_, i) => (
            <Box key={i}>
              {renderCanvasBox(i, i === selectedIdx)}
              <Typography sx={{ fontSize: 10, color: i === selectedIdx ? tc.accent : tc.textMuted, textAlign: "center", mt: 0.25 }}>
                {labels?.[i] || `Image ${i + 1}`}
              </Typography>
            </Box>
          ))}
        </Box>
      ) : (
        <Stack direction="row" spacing={`${SPACING.LG}px`}>
          {renderCanvasBox(0, true)}
          {effectiveShowFft && (
            <Box ref={fftContainerRef} sx={{ ...containerStyles.imageBox, width: canvasW, height: canvasH }}>
              <canvas
                ref={fftCanvasRef}
                width={canvasW}
                height={canvasH}
                style={{ width: canvasW, height: canvasH, imageRendering: "pixelated", cursor: "crosshair" }}
                onMouseDown={handleFftMouseDown}
                onMouseUp={handleFftMouseUp}
                onMouseLeave={handleFftMouseLeave}
              />
              <canvas ref={fftOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
              <Typography sx={{ position: "absolute", top: 4, left: 8, fontSize: 10, color: roiFftActive && fftCropDims ? accentGreen : "#fff", textShadow: "0 0 3px #000" }}>
                {roiFftActive && fftCropDims ? `ROI FFT (${fftCropDims.cropWidth}\u00D7${fftCropDims.cropHeight})` : "FFT"}
              </Typography>
            </Box>
          )}
        </Stack>
      )}

      {/* Stats + Readout bar */}
      <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: tc.bgAlt, display: "flex", gap: 2, alignItems: "center", minHeight: 20, boxSizing: "border-box" }}>
        {showStats && imageStats ? (
          <>
            <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
              Mean <Box component="span" sx={{ color: tc.accent }}>{formatNumber(imageStats.mean)}</Box>
            </Typography>
            <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
              Min <Box component="span" sx={{ color: tc.accent }}>{formatNumber(imageStats.min)}</Box>
            </Typography>
            <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
              Max <Box component="span" sx={{ color: tc.accent }}>{formatNumber(imageStats.max)}</Box>
            </Typography>
            <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
              Std <Box component="span" sx={{ color: tc.accent }}>{formatNumber(imageStats.std)}</Box>
            </Typography>
          </>
        ) : (
          <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
            {width}×{height} px
          </Typography>
        )}
        {hover && (
          <>
            <Box sx={{ borderLeft: `1px solid ${tc.border}`, height: 14 }} />
            <Typography sx={{ fontSize: 11, color: tc.textMuted, fontFamily: "monospace" }}>
              ({hover.row}, {hover.col}) <Box component="span" sx={{ color: tc.accent }}>{hover.raw !== undefined ? formatNumber(hover.raw) : ""}</Box>
            </Typography>
          </>
        )}
        {fftClickInfo && (
          <>
            <Box sx={{ borderLeft: `1px solid ${tc.border}`, height: 14 }} />
            <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
              {fftClickInfo.dSpacing != null ? (
                <>d = <Box component="span" sx={{ color: tc.accent, fontWeight: "bold" }}>{fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} \u00C5`}</Box>{" | |g| = "}<Box component="span" sx={{ color: tc.accent }}>{fftClickInfo.spatialFreq!.toFixed(4)} \u00C5\u207B\u00B9</Box></>
              ) : (
                <>dist = <Box component="span" sx={{ color: tc.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
              )}
            </Typography>
          </>
        )}
        <Typography sx={{ fontSize: 11, color: tc.textMuted, ml: "auto" }}>
          {width}×{height}
          {activeZoom.zoom !== 1 && (
            <Box component="span" sx={{ color: tc.accent, fontWeight: "bold", ml: 1 }}>
              {activeZoom.zoom.toFixed(1)}x
            </Box>
          )}
        </Typography>
      </Box>

      {/* Image controls + ROI basics + Histogram (Show3D layout) */}
      {showControls && (<>
      {!hideDisplay && (
        <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, boxSizing: "border-box" }}>
          {/* Left: image controls */}
          <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Scale:</Typography>
            <Select disabled={lockDisplay} value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" variant="outlined" MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 50 }}>
              <MenuItem value="linear" sx={{ fontSize: 11 }}>Lin</MenuItem>
              <MenuItem value="log" sx={{ fontSize: 11 }}>Log</MenuItem>
            </Select>
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Auto:</Typography>
            <Switch checked={autoContrast} onChange={(e) => setAutoContrast(e.target.checked)} disabled={lockDisplay} size="small" sx={switchStyles.small} />
            {roiFftActive && fftCropDims && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Win:</Typography>
                <Switch checked={fftWindow} onChange={(e) => { if (!lockDisplay) setFftWindow(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
              </>
            )}
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Color:</Typography>
            <Select disabled={lockDisplay} size="small" value={cmap || "gray"} onChange={(e) => setCmap(e.target.value)} variant="outlined" MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 65 }}>
              {COLORMAP_NAMES.map((name) => (
                <MenuItem key={name} value={name} sx={{ fontSize: 11 }}>
                  {name.charAt(0).toUpperCase() + name.slice(1)}
                </MenuItem>
              ))}
            </Select>
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Colorbar:</Typography>
            <Switch checked={showColorbar} onChange={(e) => setShowColorbar(e.target.checked)} disabled={lockDisplay} size="small" sx={switchStyles.small} />
          </Box>
          {/* Right: Histogram */}
          <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
            {histogramData && (
              <HistogramWidget
                data={histogramData}
                vminPct={vminPct}
                vmaxPct={vmaxPct}
                onRangeChange={(min, max) => {
                  if (lockDisplay) return;
                  // Moving the histogram range implies manual contrast mode.
                  if (autoContrast) setAutoContrast(false);
                  setVminPct(min);
                  setVmaxPct(max);
                }}
                width={110}
                height={40}
                theme={themeInfo.theme}
                dataMin={dataRange.min}
                dataMax={dataRange.max}
              />
            )}
          </Box>
        </Box>
      )}

      {/* Shape + Color picker row */}
      {showMarkerStyleControls && (
      <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, mt: 0.5, width: "fit-content", boxSizing: "border-box" }}>
        <Box sx={{ display: "flex", gap: "3px", flexShrink: 0 }}>
          {MARKER_SHAPES.map(s => {
            const sz = 16;
            const half = sz / 2;
            const r = half * 0.7;
            const selected = s === currentShape;
            let path: React.ReactNode;
            switch (s) {
              case "circle": path = <circle cx={half} cy={half} r={r} />; break;
              case "triangle": path = <polygon points={`${half},${half - r} ${half + r * 0.87},${half + r * 0.5} ${half - r * 0.87},${half + r * 0.5}`} />; break;
              case "square": path = <rect x={half - r * 0.75} y={half - r * 0.75} width={r * 1.5} height={r * 1.5} />; break;
              case "diamond": path = <polygon points={`${half},${half - r} ${half + r * 0.7},${half} ${half},${half + r} ${half - r * 0.7},${half}`} />; break;
              case "star": {
                const pts: string[] = [];
                for (let i = 0; i < 10; i++) {
                  const angle = (i * Math.PI) / 5 - Math.PI / 2;
                  const sr = i % 2 === 0 ? r : r * 0.4;
                  pts.push(`${half + sr * Math.cos(angle)},${half + sr * Math.sin(angle)}`);
                }
                path = <polygon points={pts.join(" ")} />;
                break;
              }
            }
            return (
              <Box
                key={s}
                onClick={() => { if (!lockMarkerSettings) setCurrentShape(s); }}
                sx={{
                  width: sz, height: sz, cursor: lockMarkerSettings ? "default" : "pointer", borderRadius: "2px",
                  border: selected ? `2px solid ${tc.text}` : "2px solid transparent",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  opacity: lockMarkerSettings ? 0.5 : 1,
                  "&:hover": { opacity: lockMarkerSettings ? 0.5 : 0.8 },
                }}
              >
                <svg width={sz} height={sz} style={{ display: "block" }}>
                  <g fill={currentColor} stroke={tc.bg} strokeWidth={1}>{path}</g>
                </svg>
              </Box>
            );
          })}
        </Box>
        <Box sx={{ display: "flex", gap: "3px" }}>
          {MARKER_COLORS.map(c => (
            <Box
              key={c}
              onClick={() => { if (!lockMarkerSettings) setCurrentColor(c); }}
              sx={{
                width: 16, height: 16, bgcolor: c, borderRadius: "2px", cursor: lockMarkerSettings ? "default" : "pointer",
                border: c === currentColor ? `2px solid ${tc.text}` : "2px solid transparent",
                opacity: lockMarkerSettings ? 0.5 : 1,
                "&:hover": { opacity: lockMarkerSettings ? 0.5 : 0.8 },
              }}
            />
          ))}
        </Box>
      </Box>
      )}

      {/* Controls row: Marker size + Max + Advanced toggle */}
      {showPrimaryControlRow && (
      <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, mt: 0.5, width: "fit-content", boxSizing: "border-box" }}>
        {showMarkerStyleControls && (
          <>
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Marker:</Typography>
            <Slider
              value={size}
              min={4}
              max={40}
              step={1}
              disabled={lockMarkerSettings}
              onChange={(_, v) => { if (typeof v === "number") setDotSize(v); }}
              size="small"
              sx={{ ...sliderStyles.small, width: 60 }}
            />
            <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 20 }}>{size}px</Typography>
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Max:</Typography>
            <Select
              disabled={lockMarkerSettings}
              value={maxPtsVal}
              onChange={(e: SelectChangeEvent<number>) => {
                const v = Number(e.target.value);
                setMaxPoints(v);
                if (!isGallery) {
                  setSelectedPoints((prev) => {
                    const flat = (prev as Point[]) || [];
                    return flat.length <= v ? flat : flat.slice(flat.length - v);
                  });
                } else {
                  setSelectedPoints((prev) => {
                    const nested = ((prev as Point[][]) || []).map(pts =>
                      pts.length <= v ? pts : pts.slice(pts.length - v)
                    );
                    return nested;
                  });
                }
              }}
              size="small"
              variant="outlined"
              MenuProps={themedMenuProps}
              sx={themedSelect}
            >
              {Array.from({ length: 20 }, (_, i) => i + 1).map(n => (
                <MenuItem key={n} value={n} sx={{ fontSize: 11 }}>{n}</MenuItem>
              ))}
            </Select>
          </>
        )}
        {showSnapControls && (
          <>
            <Typography
              onClick={() => { if (!lockSnap) setSnapEnabled(!snapEnabled); }}
              sx={{ ...typography.labelSmall, color: snapEnabled ? accentGreen : tc.textMuted, cursor: lockSnap ? "default" : "pointer", userSelect: "none", fontWeight: snapEnabled ? "bold" : "normal", opacity: lockSnap ? 0.5 : 1, "&:hover": { textDecoration: lockSnap ? "none" : "underline" } }}
            >
              {snapEnabled ? "\u25C9" : "\u25CB"} Snap
            </Typography>
            {snapEnabled && (
              <>
                <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>R:</Typography>
                <Slider disabled={lockSnap} value={snapRadius} min={1} max={20} step={1} onChange={(_, v) => { if (typeof v === "number") setSnapRadius(v); }} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                <Typography sx={{ ...typography.value, color: tc.textMuted }}>{snapRadius}px</Typography>
              </>
            )}
          </>
        )}
        {showProfileControls && (
          <Typography
            onClick={() => {
              if (lockProfile) return;
              profileDragRef.current = null;
              setHoveredProfileEndpoint(null);
              setIsHoveringProfileLine(false);
              setProfileActive(!profileActive);
              if (profileActive) { setProfileLine([]); setProfileData(null); }
            }}
            sx={{ ...typography.labelSmall, color: profileActive ? tc.accent : tc.textMuted, cursor: lockProfile ? "default" : "pointer", userSelect: "none", fontWeight: profileActive ? "bold" : "normal", opacity: lockProfile ? 0.5 : 1, "&:hover": { textDecoration: lockProfile ? "none" : "underline" } }}
          >
            {profileActive ? "\u25C9" : "\u25CB"} Profile
          </Typography>
        )}
        {showMarkerStyleControls && (
          <Typography
            onClick={() => setShowAdvanced(!showAdvanced)}
            sx={{ ...typography.labelSmall, color: tc.accent, cursor: "pointer", userSelect: "none", "&:hover": { textDecoration: "underline" } }}
          >
            {showAdvanced ? "\u25BE Advanced" : "\u25B8 Advanced"}
          </Typography>
        )}
      </Box>
      )}

      {/* ROI basics */}
      {!hideRoi && (
      <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, mt: 0.5, width: "fit-content", boxSizing: "border-box" }}>
        <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>ROI:</Typography>
        <Select
          disabled={lockRoi}
          size="small"
          value={activeRoi ? activeRoi.shape : newRoiShape}
          onChange={(e) => {
            const val = e.target.value as RoiShape;
            setNewRoiShape(val);
            if (activeRoi) updateActiveRoi({ shape: val });
          }}
          variant="outlined"
          MenuProps={themedMenuProps}
          sx={{ ...themedSelect, minWidth: 75 }}
        >
          {ROI_SHAPES.map((m) => (
            <MenuItem key={m} value={m} sx={{ fontSize: 11 }}>{m.charAt(0).toUpperCase() + m.slice(1)}</MenuItem>
          ))}
        </Select>
        <Button
          size="small"
          variant="outlined"
          disabled={lockRoi}
          onClick={() => {
            if (lockRoi) return;
            pushRoiHistory();
            const id = Math.max(0, ...safeRois.map(r => r.id)) + 1;
            const color = ROI_COLORS[safeRois.length % ROI_COLORS.length];
            const roi: ROI = {
              id, shape: newRoiShape,
              row: Math.floor(height / 2), col: Math.floor(width / 2),
              radius: 30, width: 60, height: 40,
              color, opacity: 0.8,
            };
            setActiveRoiIdx(safeRois.length);
            setRois([...safeRois, roi]);
          }}
          sx={{ fontSize: 10, minWidth: 0, px: 1, py: 0.25, color: tc.accent, borderColor: tc.border }}
        >
          ADD
        </Button>
      </Box>
      )}

      {/* Advanced options row (collapsible) */}
      {showAdvanced && showMarkerStyleControls && (
        <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, width: "fit-content", boxSizing: "border-box" }}>
          <Typography sx={{ ...typography.label, color: tc.textMuted }}>Border</Typography>
          <Slider
            value={borderWidth}
            min={0}
            max={6}
            step={1}
            disabled={lockMarkerSettings}
            onChange={(_, v) => { if (typeof v === "number") setBorderWidth(v); }}
            size="small"
            sx={{ ...sliderStyles.small, width: 50 }}
          />
          <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 16 }}>{borderWidth}</Typography>
          <Typography sx={{ ...typography.label, color: tc.textMuted }}>Opacity</Typography>
          <Slider
            value={markerOpacity}
            min={0.1}
            max={1.0}
            step={0.1}
            disabled={lockMarkerSettings}
            onChange={(_, v) => { if (typeof v === "number") setMarkerOpacity(v); }}
            size="small"
            sx={{ ...sliderStyles.small, width: 50 }}
          />
          <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 20 }}>{Math.round(markerOpacity * 100)}%</Typography>
          <Typography sx={{ ...typography.label, color: tc.textMuted }}>Label</Typography>
          <Slider
            value={labelSize}
            min={0}
            max={36}
            step={1}
            disabled={lockMarkerSettings}
            onChange={(_, v) => { if (typeof v === "number") setLabelSize(v); }}
            size="small"
            sx={{ ...sliderStyles.small, width: 50 }}
          />
          <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 28 }}>{labelSize === 0 ? "Auto" : `${labelSize}px`}</Typography>
          <Typography sx={{ ...typography.label, color: tc.textMuted }}>Color</Typography>
          <Select disabled={lockMarkerSettings} value={labelColor} onChange={(e) => setLabelColor(e.target.value)} size="small" variant="outlined" displayEmpty MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60 }}>
            <MenuItem value="" sx={{ fontSize: 11 }}>Auto</MenuItem>
            <MenuItem value="white" sx={{ fontSize: 11 }}>White</MenuItem>
            <MenuItem value="black" sx={{ fontSize: 11 }}>Black</MenuItem>
            <MenuItem value="#ff0" sx={{ fontSize: 11 }}>Yellow</MenuItem>
            <MenuItem value="#0f0" sx={{ fontSize: 11 }}>Green</MenuItem>
            <MenuItem value="#f00" sx={{ fontSize: 11 }}>Red</MenuItem>
            <MenuItem value="#0af" sx={{ fontSize: 11 }}>Cyan</MenuItem>
          </Select>
        </Box>
      )}

      {/* Active ROI details (only when a ROI is selected) */}
      {!hideRoi && activeRoi && (
        <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, mt: 0.5, width: "fit-content", boxSizing: "border-box" }}>
          <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 24 }}>
            ROI #{activeRoiIdx + 1}/{safeRois.length}
          </Typography>
          {safeRois.length > 1 && (
            <>
              <Typography
                onClick={() => { if (!lockRoi) setActiveRoiIdx((activeRoiIdx - 1 + safeRois.length) % safeRois.length); }}
                sx={{ ...typography.labelSmall, color: tc.accent, cursor: lockRoi ? "default" : "pointer", userSelect: "none", opacity: lockRoi ? 0.5 : 1 }}
              >&larr;</Typography>
              <Typography
                onClick={() => { if (!lockRoi) setActiveRoiIdx((activeRoiIdx + 1) % safeRois.length); }}
                sx={{ ...typography.labelSmall, color: tc.accent, cursor: lockRoi ? "default" : "pointer", userSelect: "none", opacity: lockRoi ? 0.5 : 1 }}
              >&rarr;</Typography>
            </>
          )}
          {activeRoi.shape === "rectangle" ? (
            <>
              <Typography sx={{ ...typography.label, color: tc.textMuted }}>W</Typography>
              <Slider disabled={lockRoi} value={activeRoi.width} min={5} max={Math.max(width, 10)} onChange={(_, v) => { if (typeof v === "number") updateActiveRoi({ width: v }); }} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
              <Typography sx={{ ...typography.label, color: tc.textMuted }}>H</Typography>
              <Slider disabled={lockRoi} value={activeRoi.height} min={5} max={Math.max(height, 10)} onChange={(_, v) => { if (typeof v === "number") updateActiveRoi({ height: v }); }} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
            </>
          ) : (
            <>
              <Typography sx={{ ...typography.label, color: tc.textMuted }}>Size</Typography>
              <Slider disabled={lockRoi} value={activeRoi.radius} min={5} max={Math.max(width, height, 10)} onChange={(_, v) => { if (typeof v === "number") updateActiveRoi({ radius: v }); }} size="small" sx={{ ...sliderStyles.small, width: 50 }} />
            </>
          )}
          <Box sx={{ display: "flex", gap: "2px" }}>
            {ROI_COLORS.map(c => (
              <Box
                key={c}
                onClick={() => { if (!lockRoi) updateActiveRoi({ color: c }); }}
                sx={{
                  width: 12, height: 12, bgcolor: c, cursor: lockRoi ? "default" : "pointer",
                  border: c === activeRoi.color ? `2px solid ${tc.text}` : "1px solid transparent",
                  opacity: lockRoi ? 0.5 : 1,
                  "&:hover": { opacity: lockRoi ? 0.5 : 0.8 },
                }}
              />
            ))}
          </Box>
          <Typography sx={{ ...typography.label, color: tc.textMuted }}>Op</Typography>
          <Slider
            value={activeRoi.opacity}
            min={0.1}
            max={1.0}
            step={0.1}
            disabled={lockRoi}
            onChange={(_, v) => { if (typeof v === "number") updateActiveRoi({ opacity: v }); }}
            size="small"
            sx={{ ...sliderStyles.small, width: 40 }}
          />
          <Button
            size="small"
            variant="outlined"
            disabled={lockRoi}
            onClick={() => {
              if (lockRoi) return;
              pushRoiHistory();
              const next = safeRois.filter((_, i) => i !== activeRoiIdx);
              setActiveRoiIdx(next.length === 0 ? -1 : Math.min(activeRoiIdx, next.length - 1));
              setRois(next);
            }}
            sx={{ fontSize: 10, minWidth: 0, px: 0.5, py: 0.25, color: tc.textMuted, borderColor: tc.border }}
          >
            &times;
          </Button>
        </Box>
      )}

      {/* ROI pixel statistics */}
      {!hideRoi && roiStats && (
        <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, width: "fit-content", boxSizing: "border-box" }}>
          <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>ROI Stats:</Typography>
          <Typography sx={{ ...typography.value, color: tc.textMuted }}>
            Mean <Box component="span" sx={{ color: tc.accent }}>{roiStats.mean.toFixed(4)}</Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: tc.textMuted }}>
            Std <Box component="span" sx={{ color: tc.accent }}>{roiStats.std.toFixed(4)}</Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: tc.textMuted }}>
            Min <Box component="span" sx={{ color: tc.accent }}>{roiStats.min.toFixed(4)}</Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: tc.textMuted }}>
            Max <Box component="span" sx={{ color: tc.accent }}>{roiStats.max.toFixed(4)}</Box>
          </Typography>
          <Typography sx={{ ...typography.value, color: tc.textMuted }}>
            N <Box component="span" sx={{ color: tc.accent }}>{roiStats.count}</Box>
          </Typography>
        </Box>
      )}

      {/* Line profile sparkline */}
      {!hideProfile && profileActive && (
        <Box sx={{ mt: 0.5, boxSizing: "border-box" }}>
          <canvas
            ref={profileCanvasRef}
            style={{ width: canvasW, height: 76, display: "block", border: `1px solid ${tc.border}` }}
          />
        </Box>
      )}
      </>)}

      {/* Selected points list */}
      {!hidePoints && (isGallery ? (
        hasAnyPoints && (
          <Box sx={{ mt: 0.5 }}>
            {Array.from({ length: nImages }).map((_, imgIdx) => {
              const pts = getPointsForImage(imgIdx);
              if (pts.length === 0) return null;
              return (
                <Box key={imgIdx} sx={{ mb: 0.5 }}>
                  <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: imgIdx === selectedIdx ? tc.accent : tc.textMuted, fontWeight: "bold", lineHeight: 1.6 }}>
                    {labels?.[imgIdx] || `Image ${imgIdx + 1}`}
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "repeat(5, auto)", gap: `0 ${SPACING.MD}px`, width: "fit-content", pl: 1 }}>
                    {pts.map((p, i) => {
                      const c = p.color || MARKER_COLORS[i % MARKER_COLORS.length];
                      return (
                        <Box key={`pt-${imgIdx}-${i}`} sx={{ display: "flex", alignItems: "center", gap: "2px", lineHeight: 1.6, "&:hover .pt-delete": { opacity: 1 } }}>
                          <ShapeIcon shape={p.shape || "circle"} color={c} size={10} />
                          <Typography component="span" sx={{ fontSize: 10, fontFamily: "monospace", color: tc.textMuted }}>
                            <Box component="span" sx={{ color: c }}>{i + 1}</Box> ({p.row}, {p.col})
                            {i > 0 && (
                              <Box component="span" sx={{ color: tc.textMuted, ml: 0.5, fontSize: 9 }}>
                                {"\u2194"} {formatDistance(pts[i - 1], p, pixelSize || 0)}
                              </Box>
                            )}
                          </Typography>
                          <Box
                            className="pt-delete"
                            onClick={() => {
                              if (lockPoints) return;
                              const updated = pts.filter((_, j) => j !== i);
                              setPointsForImage(imgIdx, updated);
                            }}
                            sx={{ opacity: lockPoints ? 0.25 : 0, cursor: lockPoints ? "default" : "pointer", fontSize: 10, color: tc.textMuted, ml: 0.5, lineHeight: 1, "&:hover": { color: lockPoints ? tc.textMuted : "#f44336" } }}
                          >&times;</Box>
                        </Box>
                      );
                    })}
                  </Box>
                </Box>
              );
            })}
          </Box>
        )
      ) : (
        activePts.length > 0 && (
          <Box sx={{ mt: 0.5, display: "grid", gridTemplateColumns: "repeat(5, auto)", gap: `0 ${SPACING.MD}px`, width: "fit-content" }}>
            {activePts.map((p, i) => {
              const c = p.color || MARKER_COLORS[i % MARKER_COLORS.length];
              return (
                <Box key={`pt-${p.row}-${p.col}-${i}`} sx={{ display: "flex", alignItems: "center", gap: "2px", lineHeight: 1.6, "&:hover .pt-delete": { opacity: 1 } }}>
                  <ShapeIcon shape={p.shape || "circle"} color={c} size={10} />
                  <Typography component="span" sx={{ fontSize: 10, fontFamily: "monospace", color: tc.textMuted }}>
                    <Box component="span" sx={{ color: c }}>{i + 1}</Box> ({p.row}, {p.col})
                    {i > 0 && (
                      <Box component="span" sx={{ color: tc.textMuted, ml: 0.5, fontSize: 9 }}>
                        {"\u2194"} {formatDistance(activePts[i - 1], p, pixelSize || 0)}
                      </Box>
                    )}
                  </Typography>
                  <Box
                    className="pt-delete"
                    onClick={() => {
                      if (lockPoints) return;
                      const idx = isGallery ? selectedIdx : 0;
                      const updated = activePts.filter((_, j) => j !== i);
                      setPointsForImage(idx, updated);
                    }}
                    sx={{ opacity: lockPoints ? 0.25 : 0, cursor: lockPoints ? "default" : "pointer", fontSize: 10, color: tc.textMuted, ml: 0.5, lineHeight: 1, "&:hover": { color: lockPoints ? tc.textMuted : "#f44336" } }}
                  >&times;</Box>
                </Box>
              );
            })}
          </Box>
        )
      ))}

      {/* Pairwise distances */}
      {!hidePoints && activePts.length >= 2 && (
        <Box sx={{ mt: 0.5, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, px: 1, py: 0.5, width: "fit-content" }}>
          <Typography sx={{ ...typography.labelSmall, color: tc.textMuted, mb: 0.25 }}>Pairwise Distances</Typography>
          <Box sx={{ display: "grid", gridTemplateColumns: "repeat(3, auto)", gap: `0 ${SPACING.MD}px`, width: "fit-content" }}>
            {activePts.flatMap((p1, i) =>
              activePts.slice(i + 1).map((p2, j) => (
                <Typography key={`d-${i}-${i + 1 + j}`} component="span" sx={{ ...typography.value, color: tc.textMuted, lineHeight: 1.6 }}>
                  <Box component="span" sx={{ color: p1.color }}>{i + 1}</Box>
                  {"\u2194"}
                  <Box component="span" sx={{ color: p2.color }}>{i + 1 + j + 1}</Box>
                  {" "}{formatDistance(p1, p2, pixelSize || 0)}
                </Typography>
              ))
            )}
          </Box>
        </Box>
      )}

      {/* Multi-ROI stats table */}
      {!hideRoi && safeRois.length >= 2 && perImageData.length > 0 && (
        <Box sx={{ mt: 0.5, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, px: 1, py: 0.5, width: "fit-content" }}>
          <Typography sx={{ ...typography.labelSmall, color: tc.textMuted, mb: 0.25 }}>All ROI Stats</Typography>
          <Box component="table" sx={{ borderCollapse: "collapse", ...typography.value, color: tc.textMuted }}>
            <thead>
              <tr>
                {["#", "Mode", "Mean", "Std", "Min", "Max", "N"].map(h => (
                  <Box component="th" key={h} sx={{ px: 0.5, py: 0.25, textAlign: "right", fontWeight: "bold", borderBottom: `1px solid ${tc.border}` }}>{h}</Box>
                ))}
              </tr>
            </thead>
            <tbody>
              {safeRois.map((roi, ri) => {
                const stats = allRoiStats[ri] ?? null;
                return (
                  <Box
                    component="tr"
                    key={roi.id}
                    onClick={() => setActiveRoiIdx(ri)}
                    sx={{ cursor: "pointer", bgcolor: ri === activeRoiIdx ? `${tc.accent}22` : "transparent", "&:hover": { bgcolor: `${tc.accent}11` } }}
                  >
                    <Box component="td" sx={{ px: 0.5, py: 0.25, color: roi.color, fontWeight: "bold" }}>{ri + 1}</Box>
                    <Box component="td" sx={{ px: 0.5, py: 0.25 }}>{roi.shape}</Box>
                    <Box component="td" sx={{ px: 0.5, py: 0.25, textAlign: "right", color: tc.accent }}>{stats ? stats.mean.toFixed(4) : "—"}</Box>
                    <Box component="td" sx={{ px: 0.5, py: 0.25, textAlign: "right", color: tc.accent }}>{stats ? stats.std.toFixed(4) : "—"}</Box>
                    <Box component="td" sx={{ px: 0.5, py: 0.25, textAlign: "right", color: tc.accent }}>{stats ? stats.min.toFixed(4) : "—"}</Box>
                    <Box component="td" sx={{ px: 0.5, py: 0.25, textAlign: "right", color: tc.accent }}>{stats ? stats.max.toFixed(4) : "—"}</Box>
                    <Box component="td" sx={{ px: 0.5, py: 0.25, textAlign: "right", color: tc.accent }}>{stats ? stats.count : "—"}</Box>
                  </Box>
                );
              })}
            </tbody>
          </Box>
        </Box>
      )}
    </Box>
  );
});

export default { render };
