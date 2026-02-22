/**
 * Align2D - Interactive image alignment widget.
 * Three-column layout: Image A | Image B | Merged overlay.
 * Circular pad + drag to shift image B relative to A.
 * Auto-alignment via cross-correlation with live NCC display.
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import "./align2d.css";
import { useTheme } from "../theme";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen } from "../colormaps";
import { drawScaleBarHiDPI, exportFigure, canvasToPDF } from "../scalebar";
import { extractFloat32, downloadBlob } from "../format";
import { findDataRange, applyLogScale, sliderRange } from "../stats";
import { computeHistogramFromBytes } from "../histogram";
import { fft2d, fftshift, computeMagnitude, autoEnhanceFFT } from "../webgpu-fft";
import { computeToolVisibility } from "../tool-parity";

// ============================================================================
// UI Styles (matching Show3D / Show4DSTEM)
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

const SPACING = {
  XS: 4,
  SM: 8,
  MD: 12,
  LG: 16,
};

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
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

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
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

// ============================================================================
// Helpers
// ============================================================================

const DPR = window.devicePixelRatio || 1;

// ============================================================================
// InfoTooltip & KeyboardShortcuts (matching Show3D pattern)
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
// NCC Computation (live in JS)
// ============================================================================

function computeNCC(
  dataA: Float32Array,
  dataB: Float32Array,
  w: number,
  h: number,
  dx: number,
  dy: number,
  rotationDeg: number = 0,
): number {
  // Fast path: translation only (no rotation)
  if (Math.abs(rotationDeg) < 1e-6) {
    const idx = Math.floor(dx);
    const idy = Math.floor(dy);
    const fx = dx - idx;
    const fy = dy - idy;
    const w0 = (1 - fx) * (1 - fy);
    const w1 = fx * (1 - fy);
    const w2 = (1 - fx) * fy;
    const w3 = fx * fy;
    const yStart = Math.max(0, idy);
    const yEnd = Math.min(h, h + idy - 1);
    const xStart = Math.max(0, idx);
    const xEnd = Math.min(w, w + idx - 1);
    if (yEnd <= yStart || xEnd <= xStart) return 0;
    const overlapArea = (yEnd - yStart) * (xEnd - xStart);
    const step = overlapArea > 500000 ? Math.max(1, Math.floor(Math.sqrt(overlapArea / 500000))) : 1;
    let sumA = 0, sumB = 0, n = 0;
    for (let y = yStart; y < yEnd; y += step) {
      for (let x = xStart; x < xEnd; x += step) {
        sumA += dataA[y * w + x];
        const bi = (y - idy) * w + (x - idx);
        sumB += dataB[bi] * w0 + dataB[bi + 1] * w1 + dataB[bi + w] * w2 + dataB[bi + w + 1] * w3;
        n++;
      }
    }
    if (n === 0) return 0;
    const meanA = sumA / n;
    const meanB = sumB / n;
    let sumAB = 0, sumA2 = 0, sumB2 = 0;
    for (let y = yStart; y < yEnd; y += step) {
      for (let x = xStart; x < xEnd; x += step) {
        const a = dataA[y * w + x] - meanA;
        const bi = (y - idy) * w + (x - idx);
        const b = dataB[bi] * w0 + dataB[bi + 1] * w1 + dataB[bi + w] * w2 + dataB[bi + w + 1] * w3 - meanB;
        sumAB += a * b;
        sumA2 += a * a;
        sumB2 += b * b;
      }
    }
    const denom = Math.sqrt(sumA2 * sumB2);
    return denom > 0 ? sumAB / denom : 0;
  }

  // Rotation path: inverse affine transform with bilinear interpolation
  const rad = -rotationDeg * Math.PI / 180; // inverse rotation
  const cosR = Math.cos(rad);
  const sinR = Math.sin(rad);
  const cxB = w / 2 + dx; // center of B after shift
  const cyB = h / 2 + dy;
  const margin = 2;
  const area = (h - 2 * margin) * (w - 2 * margin);
  const step = area > 500000 ? Math.max(1, Math.floor(Math.sqrt(area / 500000))) : 1;
  // Two-pass without temporary arrays: first compute means, then NCC directly
  let sumA = 0, sumB = 0, n = 0;
  for (let y = margin; y < h - margin; y += step) {
    for (let x = margin; x < w - margin; x += step) {
      const rx = x - cxB;
      const ry = y - cyB;
      const bx = cosR * rx - sinR * ry + w / 2;
      const by = sinR * rx + cosR * ry + h / 2;
      if (bx < 0 || bx >= w - 1 || by < 0 || by >= h - 1) continue;
      const ix = Math.floor(bx);
      const iy = Math.floor(by);
      const fx = bx - ix;
      const fy = by - iy;
      const bi = iy * w + ix;
      sumA += dataA[y * w + x];
      sumB += dataB[bi] * (1 - fx) * (1 - fy) +
              dataB[bi + 1] * fx * (1 - fy) +
              dataB[bi + w] * (1 - fx) * fy +
              dataB[bi + w + 1] * fx * fy;
      n++;
    }
  }
  if (n < 10) return 0;
  const meanA = sumA / n;
  const meanB = sumB / n;
  // Second pass: recompute bilinear interp (cheap ~10 flops/px) to avoid GC from dynamic arrays
  let sumAB = 0, sumA2 = 0, sumB2 = 0;
  for (let y = margin; y < h - margin; y += step) {
    for (let x = margin; x < w - margin; x += step) {
      const rx = x - cxB;
      const ry = y - cyB;
      const bx = cosR * rx - sinR * ry + w / 2;
      const by = sinR * rx + cosR * ry + h / 2;
      if (bx < 0 || bx >= w - 1 || by < 0 || by >= h - 1) continue;
      const ix = Math.floor(bx);
      const iy = Math.floor(by);
      const fx = bx - ix;
      const fy = by - iy;
      const bi = iy * w + ix;
      const a = dataA[y * w + x] - meanA;
      const b = dataB[bi] * (1 - fx) * (1 - fy) +
                dataB[bi + 1] * fx * (1 - fy) +
                dataB[bi + w] * (1 - fx) * fy +
                dataB[bi + w + 1] * fx * fy - meanB;
      sumAB += a * b;
      sumA2 += a * a;
      sumB2 += b * b;
    }
  }
  const denom = Math.sqrt(sumA2 * sumB2);
  return denom > 0 ? sumAB / denom : 0;
}

// ============================================================================
// Difference Image Computation
// ============================================================================

function computeDifferenceImage(
  dataA: Float32Array,
  dataB: Float32Array,
  w: number,
  h: number,
  dx: number,
  dy: number,
  rotationDeg: number,
): Float32Array {
  const result = new Float32Array(w * h);

  if (Math.abs(rotationDeg) < 1e-6) {
    // Translation only — fast path
    const idx = Math.floor(dx);
    const idy = Math.floor(dy);
    const fx = dx - idx;
    const fy = dy - idy;
    const w0 = (1 - fx) * (1 - fy);
    const w1 = fx * (1 - fy);
    const w2 = (1 - fx) * fy;
    const w3 = fx * fy;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const by = y - idy;
        const bx = x - idx;
        if (bx < 0 || bx >= w - 1 || by < 0 || by >= h - 1) continue;
        const bi = by * w + bx;
        const bVal = dataB[bi] * w0 + dataB[bi + 1] * w1 + dataB[bi + w] * w2 + dataB[bi + w + 1] * w3;
        result[y * w + x] = Math.abs(dataA[y * w + x] - bVal);
      }
    }
  } else {
    // Rotation path — inverse affine with bilinear interpolation
    const rad = -rotationDeg * Math.PI / 180;
    const cosR = Math.cos(rad);
    const sinR = Math.sin(rad);
    const cxB = w / 2 + dx;
    const cyB = h / 2 + dy;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const rx = x - cxB;
        const ry = y - cyB;
        const bx = cosR * rx - sinR * ry + w / 2;
        const by = sinR * rx + cosR * ry + h / 2;
        if (bx < 0 || bx >= w - 1 || by < 0 || by >= h - 1) continue;
        const ix = Math.floor(bx);
        const iy = Math.floor(by);
        const ffx = bx - ix;
        const ffy = by - iy;
        const bi = iy * w + ix;
        const bVal = dataB[bi] * (1 - ffx) * (1 - ffy) + dataB[bi + 1] * ffx * (1 - ffy) +
                     dataB[bi + w] * (1 - ffx) * ffy + dataB[bi + w + 1] * ffx * ffy;
        result[y * w + x] = Math.abs(dataA[y * w + x] - bVal);
      }
    }
  }

  return result;
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

function Histogram({ data, vminPct, vmaxPct, onRangeChange, width = 110, height = 40, theme = "dark", dataMin = 0, dataMax = 1 }: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);
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
      const x = i * barWidth;
      const inRange = i >= vminBin && i <= vmaxBin;
      ctx.fillStyle = inRange ? colors.barActive : colors.barInactive;
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
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
        min={0} max={100} size="small" valueLabelDisplay="auto"
        valueLabelFormat={(pct) => {
          const val = dataMin + (pct / 100) * (dataMax - dataMin);
          return val >= 1000 ? val.toExponential(1) : val.toFixed(1);
        }}
        sx={{
          width, py: 0,
          "& .MuiSlider-thumb": { width: 8, height: 8 },
          "& .MuiSlider-rail": { height: 2 },
          "& .MuiSlider-track": { height: 2 },
          "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
        }}
      />
    </Box>
  );
}

// ============================================================================
// Align Pad — 2D circular control for dx/dy offset
// ============================================================================

interface AlignPadProps {
  dx: number;
  dy: number;
  maxDx: number;
  maxDy: number;
  onMove: (dx: number, dy: number) => void;
  size?: number;
  theme: "light" | "dark";
  accentColor: string;
}

function AlignPad({ dx, dy, maxDx, maxDy, onMove, size = 80, theme, accentColor }: AlignPadProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const isDragging = React.useRef(false);

  const radius = size / 2 - 4;
  const fracX = maxDx > 0 ? dx / maxDx : 0;
  const fracY = maxDy > 0 ? dy / maxDy : 0;

  const clampToCircle = (fx: number, fy: number): [number, number] => {
    const dist = Math.sqrt(fx * fx + fy * fy);
    if (dist <= 1) return [fx, fy];
    return [fx / dist, fy / dist];
  };

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    const cx = size / 2;
    const cy = size / 2;
    const bgColor = theme === "dark" ? "#1a1a1a" : "#f0f0f0";
    const lineColor = theme === "dark" ? "#333" : "#ccc";
    const ringColor = theme === "dark" ? "#252525" : "#e0e0e0";

    // Outer circle
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = bgColor;
    ctx.fill();
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Concentric rings
    for (const frac of [1 / 3, 2 / 3]) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius * frac, 0, Math.PI * 2);
      ctx.strokeStyle = ringColor;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Crosshairs
    ctx.beginPath();
    ctx.moveTo(cx - radius, cy);
    ctx.lineTo(cx + radius, cy);
    ctx.moveTo(cx, cy - radius);
    ctx.lineTo(cx, cy + radius);
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Clamped dot position
    const [cfx, cfy] = clampToCircle(fracX, fracY);
    const dotX = cfx * radius + cx;
    const dotY = cfy * radius + cy;

    // Line from center to dot
    if (Math.abs(cfx) > 0.01 || Math.abs(cfy) > 0.01) {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(dotX, dotY);
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.3;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Dot
    ctx.beginPath();
    ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
    ctx.fillStyle = accentColor;
    ctx.fill();
    ctx.strokeStyle = theme === "dark" ? "rgba(255,255,255,0.8)" : "rgba(0,0,0,0.6)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }, [dx, dy, maxDx, maxDy, size, theme, accentColor, fracX, fracY, radius]);

  const getPosition = React.useCallback(
    (e: MouseEvent | React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const scale = size / rect.width;
      const px = (e.clientX - rect.left) * scale;
      const py = (e.clientY - rect.top) * scale;
      const fx = (px - size / 2) / radius;
      const fy = (py - size / 2) / radius;
      const [cfx, cfy] = clampToCircle(fx, fy);
      return { dx: cfx * maxDx, dy: cfy * maxDy };
    },
    [size, radius, maxDx, maxDy],
  );

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    isDragging.current = true;
    const pos = getPosition(e);
    if (pos) onMove(pos.dx, pos.dy);
  };

  React.useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const pos = getPosition(e);
      if (pos) onMove(pos.dx, pos.dy);
    };
    const handleUp = () => { isDragging.current = false; };
    document.addEventListener("mousemove", handleMove);
    document.addEventListener("mouseup", handleUp);
    return () => {
      document.removeEventListener("mousemove", handleMove);
      document.removeEventListener("mouseup", handleUp);
    };
  }, [getPosition, onMove]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size, cursor: "pointer", display: "block" }}
      onMouseDown={handleMouseDown}
      onDoubleClick={() => onMove(0, 0)}
    />
  );
}

// ============================================================================
// Constants
// ============================================================================
const CANVAS_TARGET_SIZE = 300;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

// ============================================================================
// Main Component
// ============================================================================
function Align2D() {
  const { themeInfo, colors: baseColors } = useTheme();
  const themeColors = {
    ...baseColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#b08800",
  };

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

  // Model state (synced with Python) — images are unpadded
  const [imgH] = useModelState<number>("height");
  const [imgW] = useModelState<number>("width");
  const [imageABytes] = useModelState<DataView>("image_a_bytes");
  const [imageBBytes] = useModelState<DataView>("image_b_bytes");
  const [padding, setPadding] = useModelState<number>("padding");
  const [medianA] = useModelState<number>("median_a");
  const [medianB] = useModelState<number>("median_b");
  const [dx, setDx] = useModelState<number>("dx");
  const [dy, setDy] = useModelState<number>("dy");
  const [rotation, setRotation] = useModelState<number>("rotation");
  const [xcorrZero] = useModelState<number>("xcorr_zero");
  const [nccAligned] = useModelState<number>("ncc_aligned");
  const [autoDx] = useModelState<number>("auto_dx");
  const [autoDy] = useModelState<number>("auto_dy");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [opacity, setOpacity] = useModelState<number>("opacity");
  const [labelA] = useModelState<string>("label_a");
  const [labelB] = useModelState<string>("label_b");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [canvasSize] = useModelState<number>("canvas_size");
  const [maxShift] = useModelState<number>("max_shift");
  const [histSource, setHistSource] = useModelState<string>("hist_source");
  const [disabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools] = useModelState<string[]>("hidden_tools");

  // Tool visibility — normalized sets and per-group booleans
  const toolVisibility = React.useMemo(
    () => computeToolVisibility("Align2D", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideAlignment = toolVisibility.isHidden("alignment");
  const hideOverlay = toolVisibility.isHidden("overlay");
  const hideDisplay = toolVisibility.isHidden("display");
  const hideHistogram = toolVisibility.isHidden("histogram");
  const hideStats = toolVisibility.isHidden("stats");
  const hideExport = toolVisibility.isHidden("export");
  const hideView = toolVisibility.isHidden("view");

  const lockAlignment = toolVisibility.isLocked("alignment");
  const lockOverlay = toolVisibility.isLocked("overlay");
  const lockDisplay = toolVisibility.isLocked("display");
  const lockHistogram = toolVisibility.isLocked("histogram");
  const lockExport = toolVisibility.isLocked("export");
  const lockView = toolVisibility.isLocked("view");

  // Compute padding in pixels from fractional padding
  const padY = Math.round(imgH * padding);
  const padX = Math.round(imgW * padding);
  const paddedH = imgH + 2 * padY;
  const paddedW = imgW + 2 * padX;

  // Effective shift bounds (max_shift overrides padding-based limit)
  const effectiveMaxDx = maxShift > 0 ? maxShift : padX;
  const effectiveMaxDy = maxShift > 0 ? maxShift : padY;

  // Local state
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [mainCanvasSize, setMainCanvasSize] = React.useState(canvasSize || CANVAS_TARGET_SIZE);
  const [isResizingMain, setIsResizingMain] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number } | null>(null);
  const [showPanels, setShowPanels] = React.useState(true);
  const [showFft, setShowFft] = React.useState(false);
  const [fineMode, setFineMode] = React.useState(false);

  // Blend mode
  const [blendMode, setBlendMode] = React.useState<"blend" | "difference" | "flicker">("blend");
  const [flickerShowA, setFlickerShowA] = React.useState(true);

  // Flicker timer (~3 Hz toggle)
  React.useEffect(() => {
    if (blendMode !== "flicker") return;
    const interval = setInterval(() => setFlickerShowA(p => !p), 333);
    return () => clearInterval(interval);
  }, [blendMode]);

  // Rotation playback
  const [rotPlaying, setRotPlaying] = React.useState(false);
  const [rotFps, setRotFps] = React.useState(30);
  const [rotRange, setRotRange] = React.useState(5); // ± degrees from center
  const rotCenterRef = React.useRef(0); // rotation angle when play started
  const rotDirRef = React.useRef(1); // bounce direction: 1 = forward, -1 = backward

  // Rotation playback params ref (avoids stale closures in rAF loop)
  const rotPlayRef = React.useRef({ rotFps, rotRange, fineMode });
  React.useEffect(() => {
    rotPlayRef.current = { rotFps, rotRange, fineMode };
  }, [rotFps, rotRange, fineMode]);

  // Rotation playback animation loop
  React.useEffect(() => {
    if (!rotPlaying) return;
    rotCenterRef.current = rotation;
    rotDirRef.current = 1;
    let lastFrameTime = 0;
    let animId: number;

    const tick = (now: number) => {
      const c = rotPlayRef.current;
      const intervalMs = 1000 / c.rotFps;
      if (lastFrameTime === 0) {
        lastFrameTime = now;
        animId = requestAnimationFrame(tick);
        return;
      }
      const elapsed = now - lastFrameTime;
      if (elapsed < intervalMs) {
        animId = requestAnimationFrame(tick);
        return;
      }
      lastFrameTime = now - (elapsed % intervalMs);

      const step = c.fineMode ? 0.1 : 0.5;
      const center = rotCenterRef.current;
      const lo = Math.max(-180, center - c.rotRange);
      const hi = Math.min(180, center + c.rotRange);

      setRotation((prev) => {
        let next = prev + step * rotDirRef.current;
        if (next >= hi) { next = hi; rotDirRef.current = -1; }
        else if (next <= lo) { next = lo; rotDirRef.current = 1; }
        return Math.round(next * 10) / 10;
      });

      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rotPlaying]);

  // FFT data (log magnitude)
  const fftARef = React.useRef<Float32Array | null>(null);
  const fftBRef = React.useRef<Float32Array | null>(null);
  const fftRangeRef = React.useRef<{ min: number; max: number }>({ min: 0, max: 1 });

  // FFT zoom: save/restore merged view zoom when toggling FFT
  const preFftZoomRef = React.useRef<{ zoom: number; panX: number; panY: number } | null>(null);
  React.useEffect(() => {
    if (showFft) {
      preFftZoomRef.current = { zoom, panX, panY };
    } else if (preFftZoomRef.current) {
      setZoom(preFftZoomRef.current.zoom);
      setPanX(preFftZoomRef.current.panX);
      setPanY(preFftZoomRef.current.panY);
      preFftZoomRef.current = null;
    }
  }, [showFft]); // only run on FFT toggle

  // Histogram state
  const [vminPct, setVminPct] = React.useState(0);
  const [vmaxPct, setVmaxPct] = React.useState(100);
  const [histogramData, setHistogramData] = React.useState<Float32Array | null>(null);
  const [dataRange, setDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });

  // Live NCC
  const [nccCurrent, setNccCurrent] = React.useState(0);

  // Export menu
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);

  // Drag state for merged view
  const dragRef = React.useRef<{
    startX: number; startY: number;
    startDx: number; startDy: number;
    mode: "align" | "pan";
  } | null>(null);

  // Canvas refs
  const canvasARef = React.useRef<HTMLCanvasElement>(null);
  const canvasBCorrectedRef = React.useRef<HTMLCanvasElement>(null);
  const mergedCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const uiCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const mergedContainerRef = React.useRef<HTMLDivElement>(null);
  const panelAContainerRef = React.useRef<HTMLDivElement>(null);
  const panelBContainerRef = React.useRef<HTMLDivElement>(null);

  // Raw float data refs
  const rawARef = React.useRef<Float32Array | null>(null);
  const rawBRef = React.useRef<Float32Array | null>(null);

  // Cached colormapped offscreen canvases (avoids recomputing colormap on zoom/pan)
  const offscreenARef = React.useRef<HTMLCanvasElement | null>(null);
  const offscreenBRef = React.useRef<HTMLCanvasElement | null>(null);
  // Cached difference mode offscreen (pre-built, not rebuilt in render loop)
  const offscreenDiffRef = React.useRef<HTMLCanvasElement | null>(null);
  const [offscreenDiffVersion, setOffscreenDiffVersion] = React.useState(0);

  // Global contrast range (computed from both images)
  const globalRangeRef = React.useRef<{ min: number; max: number }>({ min: 0, max: 1 });

  // Offscreen version counter — incremented when colormapped offscreens are rebuilt.
  // Layout effects depend on this instead of cmap/vminPct/vmaxPct/getEffectiveRange/etc.
  const [offscreenVersion, setOffscreenVersion] = React.useState(0);
  // Cached background fill colors (computed during offscreen build, read by layout effects)
  const bgFillARef = React.useRef<string>("#000");
  const bgFillBRef = React.useRef<string>("#000");

  // Parse image data
  React.useEffect(() => {
    const floatsA = extractFloat32(imageABytes);
    const floatsB = extractFloat32(imageBBytes);
    if (!floatsA || !floatsB) return;
    rawARef.current = floatsA;
    rawBRef.current = floatsB;

    // Compute global range from both images
    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < floatsA.length; i++) {
      if (floatsA[i] < gMin) gMin = floatsA[i];
      if (floatsA[i] > gMax) gMax = floatsA[i];
    }
    for (let i = 0; i < floatsB.length; i++) {
      if (floatsB[i] < gMin) gMin = floatsB[i];
      if (floatsB[i] > gMax) gMax = floatsB[i];
    }
    globalRangeRef.current = { min: gMin, max: gMax };
    setDataRange({ min: gMin, max: gMax });
  }, [imageABytes, imageBBytes]);

  // Compute FFT magnitudes with autoEnhanceFFT (DC mask + 99.9% percentile)
  // Does NOT depend on histSource — histogram selection is handled in the effect below.
  React.useEffect(() => {
    if (!showFft) { fftARef.current = null; fftBRef.current = null; return; }
    const a = rawARef.current;
    const b = rawBRef.current;
    if (!a || !b) return;
    const computeMag = (data: Float32Array): Float32Array => {
      const real = data.slice();
      const imag = new Float32Array(data.length);
      fft2d(real, imag, imgW, imgH, false);
      fftshift(real, imgW, imgH);
      fftshift(imag, imgW, imgH);
      const mag = computeMagnitude(real, imag);
      autoEnhanceFFT(mag, imgW, imgH); // DC mask + 99.9% percentile clip
      return applyLogScale(mag);
    };
    const magA = computeMag(a);
    const magB = computeMag(b);
    fftARef.current = magA;
    fftBRef.current = magB;
    // Compute FFT range from enhanced data
    let fMin = Infinity, fMax = -Infinity;
    for (let i = 0; i < magA.length; i++) {
      if (magA[i] < fMin) fMin = magA[i];
      if (magA[i] > fMax) fMax = magA[i];
    }
    for (let i = 0; i < magB.length; i++) {
      if (magB[i] < fMin) fMin = magB[i];
      if (magB[i] > fMax) fMax = magB[i];
    }
    fftRangeRef.current = { min: fMin, max: fMax };
  }, [showFft, imageABytes, imageBBytes, imgW, imgH]);

  // Unified histogram source effect — handles both FFT and raw data modes
  React.useEffect(() => {
    if (showFft) {
      const data = histSource === "a" ? fftARef.current : fftBRef.current;
      if (data) { setHistogramData(data); setDataRange(fftRangeRef.current); }
    } else {
      const data = histSource === "a" ? rawARef.current : rawBRef.current;
      if (data) { setHistogramData(data); setDataRange(globalRangeRef.current); }
    }
  }, [imageABytes, imageBBytes, histSource, showFft]);

  // Live NCC computation (always debounced 100ms — avoids blocking during drag/nudge/rotation)
  const nccTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  React.useEffect(() => {
    if (nccTimerRef.current) clearTimeout(nccTimerRef.current);
    nccTimerRef.current = setTimeout(() => {
      const a = rawARef.current;
      const b = rawBRef.current;
      if (!a || !b) return;
      setNccCurrent(computeNCC(a, b, imgW, imgH, dx, dy, rotation));
    }, 100);
    return () => { if (nccTimerRef.current) { clearTimeout(nccTimerRef.current); nccTimerRef.current = null; } };
  }, [dx, dy, rotation, imageABytes, imageBBytes, imgW, imgH]);

  // Memoized difference image (only recomputed when alignment or data changes)
  const differenceImage = React.useMemo(() => {
    const a = rawARef.current;
    const b = rawBRef.current;
    if (!a || !b || blendMode !== "difference") return null;
    return computeDifferenceImage(a, b, imgW, imgH, dx, dy, rotation);
    // imageABytes/imageBBytes as proxies for rawARef/rawBRef changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [blendMode, dx, dy, rotation, imgW, imgH, imageABytes, imageBBytes]);

  // Auto-range for difference image
  const diffRange = React.useMemo(() => {
    if (!differenceImage) return { min: 0, max: 1 };
    let dMax = 0;
    for (let i = 0; i < differenceImage.length; i++) {
      if (differenceImage[i] > dMax) dMax = differenceImage[i];
    }
    return { min: 0, max: dMax > 0 ? dMax : 1 };
  }, [differenceImage]);

  // Pre-build difference offscreen canvas (avoids rebuilding inside merged render loop)
  React.useEffect(() => {
    if (!differenceImage || blendMode !== "difference") {
      offscreenDiffRef.current = null;
      setOffscreenDiffVersion(v => v + 1);
      return;
    }
    const lut = COLORMAPS[cmap] || COLORMAPS.gray;
    offscreenDiffRef.current = renderToOffscreen(differenceImage, imgW, imgH, lut, diffRange.min, diffRange.max);
    setOffscreenDiffVersion(v => v + 1);
  }, [differenceImage, diffRange, cmap, imgW, imgH, blendMode]);

  // Canvas sizing — each panel uses padded dimensions
  const displayScale = mainCanvasSize / Math.max(paddedW, paddedH);
  const canvasW = Math.round(paddedW * displayScale);
  const canvasH = Math.round(paddedH * displayScale);
  // Offset where unpadded image starts within padded canvas
  const imgOffsetX = Math.round(padX * displayScale);
  const imgOffsetY = Math.round(padY * displayScale);
  const imgDisplayW = Math.round(imgW * displayScale);
  const imgDisplayH = Math.round(imgH * displayScale);

  // Prevent page scroll on all canvas containers
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const els = [mergedContainerRef.current, panelAContainerRef.current, panelBContainerRef.current];
    els.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => { els.forEach(el => el?.removeEventListener("wheel", preventDefault)); };
  }, [showPanels]);

  // Resize handler (on merged view)
  const handleMainResizeStart = (e: React.MouseEvent) => {
    if (lockView) return;
    e.stopPropagation();
    e.preventDefault();
    setIsResizingMain(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: mainCanvasSize });
  };

  React.useEffect(() => {
    if (!isResizingMain) return;
    let rafId = 0;
    let latestSize = resizeStart ? resizeStart.size : mainCanvasSize;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      latestSize = Math.max(150, Math.min(600, resizeStart.size + delta));
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setMainCanvasSize(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
      setMainCanvasSize(latestSize);
      setIsResizingMain(false);
      setResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingMain, resizeStart]);

  // Build offscreen colormapped canvas (unpadded image)
  const buildOffscreen = React.useCallback((
    data: Float32Array | null,
    w: number,
    h: number,
    lut: Uint8Array,
    vmin: number,
    vmax: number,
  ): HTMLCanvasElement | null => {
    if (!data || data.length === 0) return null;
    return renderToOffscreen(data, w, h, lut, vmin, vmax);
  }, []);

  // Get colormapped RGB for a median value
  const getMedianRGB = React.useCallback((median: number, lut: Uint8Array, vmin: number, vmax: number): string => {
    const range = vmax > vmin ? vmax - vmin : 1;
    const clipped = Math.max(vmin, Math.min(vmax, median));
    const v = Math.floor(((clipped - vmin) / range) * 255);
    const idx = v * 3;
    return `rgb(${lut[idx]},${lut[idx + 1]},${lut[idx + 2]})`;
  }, []);

  // Compute effective vmin/vmax from global range (or FFT range)
  const getEffectiveRange = React.useCallback((forFft = false) => {
    const { min: gMin, max: gMax } = forFft ? fftRangeRef.current : globalRangeRef.current;
    return sliderRange(gMin, gMax, vminPct, vmaxPct);
  }, [vminPct, vmaxPct]);

  // Build colormapped offscreen canvases (expensive: colormap LUT application)
  // Excludes zoom/pan so dragging only triggers the cheap redraws below.
  // Also caches background fill colors so layout effects don't depend on cmap/vmin/vmax.
  React.useEffect(() => {
    if (!rawARef.current || !rawBRef.current || !imgW || !imgH) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.gray;
    const dataA = showFft && fftARef.current ? fftARef.current : rawARef.current;
    const dataB = showFft && fftBRef.current ? fftBRef.current : rawBRef.current;
    const { vmin, vmax } = getEffectiveRange(showFft);
    offscreenARef.current = buildOffscreen(dataA, imgW, imgH, lut, vmin, vmax);
    offscreenBRef.current = buildOffscreen(dataB, imgW, imgH, lut, vmin, vmax);
    // Cache background fill colors for layout effects
    bgFillARef.current = showFft ? "#000" : getMedianRGB(medianA, lut, vmin, vmax);
    bgFillBRef.current = showFft ? "#000" : getMedianRGB(medianB, lut, vmin, vmax);
    setOffscreenVersion(v => v + 1);
  }, [imageABytes, imageBBytes, imgW, imgH, cmap, vminPct, vmaxPct, showFft, medianA, medianB, buildOffscreen, getEffectiveRange, getMedianRGB]);

  // Render Image A canvas (with padding + zoom/pan synced from merged view)
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  React.useLayoutEffect(() => {
    if (!canvasARef.current || !rawARef.current) return;
    const offA = offscreenARef.current;
    if (!offA) return;
    const canvas = canvasARef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.fillStyle = bgFillARef.current;
    ctx.fillRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offA, 0, 0, imgW, imgH, imgOffsetX, imgOffsetY, imgDisplayW, imgDisplayH);
    ctx.restore();
  }, [offscreenVersion, imgW, imgH, canvasW, canvasH, imgOffsetX, imgOffsetY, imgDisplayW, imgDisplayH, showFft, showPanels, zoom, panX, panY]);

  // Render Image B corrected canvas (shifted + zoom/pan synced from merged view)
  React.useLayoutEffect(() => {
    if (!canvasBCorrectedRef.current || !rawBRef.current) return;
    const offB = offscreenBRef.current;
    if (!offB) return;
    const canvas = canvasBCorrectedRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.fillStyle = bgFillBRef.current;
    ctx.fillRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    if (showFft) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offB, 0, 0, imgW, imgH, imgOffsetX, imgOffsetY, imgDisplayW, imgDisplayH);
    } else {
      ctx.imageSmoothingEnabled = true;
      const shiftX = dx * displayScale;
      const shiftY = dy * displayScale;
      if (Math.abs(rotation) > 1e-6) {
        const cx = imgOffsetX + shiftX + imgDisplayW / 2;
        const cy = imgOffsetY + shiftY + imgDisplayH / 2;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(rotation * Math.PI / 180);
        ctx.drawImage(offB, 0, 0, imgW, imgH, -imgDisplayW / 2, -imgDisplayH / 2, imgDisplayW, imgDisplayH);
        ctx.restore();
      } else {
        ctx.drawImage(offB, 0, 0, imgW, imgH, imgOffsetX + shiftX, imgOffsetY + shiftY, imgDisplayW, imgDisplayH);
      }
    }
    ctx.restore();
  }, [offscreenVersion, imgW, imgH, canvasW, canvasH, imgOffsetX, imgOffsetY, imgDisplayW, imgDisplayH, dx, dy, rotation, displayScale, showFft, showPanels, zoom, panX, panY]);

  // Render merged view (with padding) — supports blend, difference, flicker modes
  React.useLayoutEffect(() => {
    if (!mergedCanvasRef.current || !rawARef.current || !rawBRef.current) return;
    const offA = offscreenARef.current;
    const offB = offscreenBRef.current;
    if (!offA || !offB) return;
    const canvas = mergedCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = bgFillARef.current;
    ctx.fillRect(0, 0, canvasW, canvasH);

    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);

    // FFT always uses blend mode
    const activeMode = showFft ? "blend" : blendMode;

    if (activeMode === "difference") {
      // Difference: use pre-built offscreen from offscreenDiffRef
      const offDiff = offscreenDiffRef.current;
      if (offDiff) {
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(offDiff, 0, 0, imgW, imgH, imgOffsetX, imgOffsetY, imgDisplayW, imgDisplayH);
      }
    } else if (activeMode === "flicker") {
      // Flicker: alternate between A and B
      if (flickerShowA) {
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(offA, 0, 0, imgW, imgH, imgOffsetX, imgOffsetY, imgDisplayW, imgDisplayH);
      } else {
        ctx.imageSmoothingEnabled = true;
        const shiftX = dx * displayScale;
        const shiftY = dy * displayScale;
        if (Math.abs(rotation) > 1e-6) {
          const cx = imgOffsetX + shiftX + imgDisplayW / 2;
          const cy = imgOffsetY + shiftY + imgDisplayH / 2;
          ctx.save();
          ctx.translate(cx, cy);
          ctx.rotate(rotation * Math.PI / 180);
          ctx.drawImage(offB, 0, 0, imgW, imgH, -imgDisplayW / 2, -imgDisplayH / 2, imgDisplayW, imgDisplayH);
          ctx.restore();
        } else {
          ctx.drawImage(offB, 0, 0, imgW, imgH, imgOffsetX + shiftX, imgOffsetY + shiftY, imgDisplayW, imgDisplayH);
        }
      }
    } else {
      // Default: alpha blend
      ctx.globalAlpha = 1 - opacity;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offA, 0, 0, imgW, imgH, imgOffsetX, imgOffsetY, imgDisplayW, imgDisplayH);

      ctx.globalAlpha = opacity;
      if (showFft) {
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(offB, 0, 0, imgW, imgH, imgOffsetX, imgOffsetY, imgDisplayW, imgDisplayH);
      } else {
        ctx.imageSmoothingEnabled = true;
        const shiftX = dx * displayScale;
        const shiftY = dy * displayScale;
        if (Math.abs(rotation) > 1e-6) {
          const cx = imgOffsetX + shiftX + imgDisplayW / 2;
          const cy = imgOffsetY + shiftY + imgDisplayH / 2;
          ctx.save();
          ctx.translate(cx, cy);
          ctx.rotate(rotation * Math.PI / 180);
          ctx.drawImage(offB, 0, 0, imgW, imgH, -imgDisplayW / 2, -imgDisplayH / 2, imgDisplayW, imgDisplayH);
          ctx.restore();
        } else {
          ctx.drawImage(offB, 0, 0, imgW, imgH, imgOffsetX + shiftX, imgOffsetY + shiftY, imgDisplayW, imgDisplayH);
        }
      }
      ctx.globalAlpha = 1;
    }

    ctx.restore();
  }, [offscreenVersion, offscreenDiffVersion, imgW, imgH, canvasW, canvasH, imgOffsetX, imgOffsetY, imgDisplayW, imgDisplayH, opacity, dx, dy, rotation, zoom, panX, panY, displayScale, showFft, blendMode, flickerShowA]);

  // Scale bar on merged view
  React.useEffect(() => {
    if (!uiCanvasRef.current) return;
    if (pixelSize > 0) {
      drawScaleBarHiDPI(uiCanvasRef.current, DPR, zoom, pixelSize, "Å", paddedW);
    } else {
      const ctx = uiCanvasRef.current.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, uiCanvasRef.current.width, uiCanvasRef.current.height);
    }
  }, [pixelSize, paddedW, canvasW, canvasH, zoom]);

  // Side panel wheel handler (center-based zoom, shared state with merged view)
  const handlePanelWheel = (e: React.WheelEvent) => {
    if (lockView) return;
    const el = e.currentTarget as HTMLElement;
    const rect = el.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) / rect.width * canvasW;
    const mouseY = (e.clientY - rect.top) / rect.height * canvasH;
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * zoomFactor));
    const zoomRatio = newZoom / zoom;
    setZoom(newZoom);
    setPanX(mouseX - (mouseX - panX) * zoomRatio);
    setPanY(mouseY - (mouseY - panY) * zoomRatio);
  };

  // Merged view mouse handlers — drag to align, Alt+drag to pan
  const handleWheel = (e: React.WheelEvent) => {
    if (e.shiftKey) {
      if (lockAlignment) return;
      // Shift+scroll = rotate image B
      const step = fineMode ? 0.1 : 0.5;
      const delta = e.deltaY > 0 ? -step : step;
      setRotation(Math.max(-180, Math.min(180, rotation + delta)));
      return;
    }
    if (lockView) return;
    const canvas = mergedCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * zoomFactor));
    const zoomRatio = newZoom / zoom;
    setZoom(newZoom);
    setPanX(mouseX - (mouseX - panX) * zoomRatio);
    setPanY(mouseY - (mouseY - panY) * zoomRatio);
  };

  const handleDoubleClick = () => { if (lockView) return; setZoom(1); setPanX(0); setPanY(0); };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    const isAltOrMiddle = e.altKey || e.button === 1;
    if (isAltOrMiddle && lockView) return;
    if (!isAltOrMiddle && lockAlignment) return;
    dragRef.current = {
      startX: e.clientX, startY: e.clientY,
      startDx: isAltOrMiddle ? panX : dx,
      startDy: isAltOrMiddle ? panY : dy,
      mode: isAltOrMiddle ? "pan" : "align",
    };
  };

  // Drag uses document-level listeners for smooth tracking outside canvas
  const dragParamsRef = React.useRef({ displayScale, zoom, effectiveMaxDx, effectiveMaxDy });
  React.useEffect(() => {
    dragParamsRef.current = { displayScale, zoom, effectiveMaxDx, effectiveMaxDy };
  }, [displayScale, zoom, effectiveMaxDx, effectiveMaxDy]);

  React.useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      const canvas = mergedCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const pxDx = (e.clientX - dragRef.current.startX) * scaleX;
      const pxDy = (e.clientY - dragRef.current.startY) * scaleX;
      const p = dragParamsRef.current;
      if (dragRef.current.mode === "align") {
        const newDx = dragRef.current.startDx + pxDx / (p.displayScale * p.zoom);
        const newDy = dragRef.current.startDy + pxDy / (p.displayScale * p.zoom);
        setDx(Math.max(-p.effectiveMaxDx, Math.min(p.effectiveMaxDx, newDx)));
        setDy(Math.max(-p.effectiveMaxDy, Math.min(p.effectiveMaxDy, newDy)));
      } else {
        setPanX(dragRef.current.startDx + pxDx);
        setPanY(dragRef.current.startDy + pxDy);
      }
    };
    const handleMouseUp = () => { dragRef.current = null; };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [setDx, setDy, setPanX, setPanY]);

  // Clamped setters
  const clampDx = React.useCallback((v: number) => setDx(Math.max(-effectiveMaxDx, Math.min(effectiveMaxDx, v))), [effectiveMaxDx, setDx]);
  const clampDy = React.useCallback((v: number) => setDy(Math.max(-effectiveMaxDy, Math.min(effectiveMaxDy, v))), [effectiveMaxDy, setDy]);

  // Keyboard: arrow keys / WASD nudge dx/dy
  const handleKeyDown = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 0.1 : 1;
    switch (e.key) {
      case "ArrowLeft": case "a": case "A": if (lockAlignment) return; e.preventDefault(); clampDx(dx - step); break;
      case "ArrowRight": case "d": case "D": if (lockAlignment) return; e.preventDefault(); clampDx(dx + step); break;
      case "ArrowUp": case "w": case "W": if (lockAlignment) return; e.preventDefault(); clampDy(dy - step); break;
      case "ArrowDown": case "s": case "S": if (lockAlignment) return; e.preventDefault(); clampDy(dy + step); break;
      case "r": case "R": if (lockView) return; handleDoubleClick(); break;
      case " ": if (lockAlignment) return; e.preventDefault(); setRotPlaying((p) => !p); break;
    }
  };

  const handleExportFigure = (withColorbar: boolean) => {
    setExportAnchor(null);
    if (lockExport) return;
    const dataA = rawARef.current;
    if (!dataA) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.gray;
    const { min: gMin, max: gMax } = findDataRange(dataA);
    const { vmin, vmax } = sliderRange(gMin, gMax, vminPct, vmaxPct);
    const offscreen = renderToOffscreen(dataA, imgW, imgH, lut, vmin, vmax);
    if (!offscreen) return;
    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title || undefined,
      lut,
      vmin,
      vmax,
      pixelSize: pixelSize > 0 ? pixelSize : undefined,
      showColorbar: withColorbar,
      showScaleBar: pixelSize > 0,
    });
    canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, "align2d_figure.pdf"));
  };

  const handleExport = () => {
    setExportAnchor(null);
    if (lockExport) return;
    if (!mergedCanvasRef.current) return;
    mergedCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "align2d_merged.png"); }, "image/png");
  };

  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;
  const hasOffset = dx !== 0 || dy !== 0 || rotation !== 0;
  const hasAutoAlign = autoDx !== 0 || autoDy !== 0;
  const isAtAuto = dx === autoDx && dy === autoDy && rotation === 0;

  return (
    <Box className="align2d-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ ...container.root, bgcolor: themeColors.bg, color: themeColors.text }}>
      {/* Header */}
      <Typography variant="caption" sx={{ ...typography.label, mb: `${SPACING.XS}px`, display: "block" }}>{title || "Alignment"}<InfoTooltip theme={themeInfo.theme} text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
        <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
        <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Blend: Alpha-blend A and B (opacity slider controls mix).</Typography>
        <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Diff: |A - B| — bright where images differ, dark where they match.</Typography>
        <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Flicker: Rapidly blink between A and B (~3 Hz).</Typography>
        <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: FFT-based auto-alignment. Zero: Reset offset to (0,0).</Typography>
        <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Fine: Restrict pad range for sub-pixel precision.</Typography>
        <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Panels: Show side-by-side A/B comparison above merged view.</Typography>
        <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
        <KeyboardShortcuts items={[
          ["Drag", "Align image B"],
          ["Alt + drag", "Pan view"],
          ["Scroll", "Zoom"],
          ["Shift + scroll", "Rotate image B"],
          ["\u2190 \u2192 / A D", "Nudge dx (Shift: 0.1px)"],
          ["\u2191 \u2193 / W S", "Nudge dy (Shift: 0.1px)"],
          ["Space", "Play / pause rotation"],
          ["R", "Reset zoom / pan"],
          ["Dbl-click pad", "Reset offset"],
        ]} />
      </Box>} /></Typography>

      {/* Panel labels + toggles (always visible — toggles at right edge of Image B) */}
      <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="flex-end" sx={{ mb: `${SPACING.XS}px` }}>
        {showPanels && (
          <Box sx={{ width: canvasW }}>
            <Typography sx={{ ...typography.labelSmall, color: themeColors.accentGreen }}>{labelA} (reference)</Typography>
          </Box>
        )}
        <Stack direction="row" justifyContent={showPanels ? "space-between" : "flex-end"} alignItems="center" sx={{ width: canvasW }}>
          {showPanels && <Typography sx={{ ...typography.labelSmall, color: themeColors.accentYellow }}>{labelB} (aligned)</Typography>}
          {!hideView && (
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typography.label, fontSize: 10 }}>Panels:</Typography>
              <Switch checked={showPanels} onChange={() => { if (!lockView) setShowPanels(!showPanels); }} disabled={lockView} size="small" sx={switchStyles.small} />
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={showFft} onChange={() => { if (!lockView) setShowFft(!showFft); }} disabled={lockView} size="small" sx={switchStyles.small} />
            </Stack>
          )}
        </Stack>
      </Stack>

      {/* Panel canvases (conditionally shown) */}
      {showPanels && <Stack direction="row" spacing={`${SPACING.SM}px`} sx={{ mb: `${SPACING.SM}px` }}>
        <Box ref={panelAContainerRef} onWheel={handlePanelWheel} onDoubleClick={handleDoubleClick} sx={{ ...container.imageBox, width: canvasW, height: canvasH, border: `1px solid ${themeColors.border}` }}>
          <canvas ref={canvasARef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }} />
          {!hideView && (
            <Box onMouseDown={handleMainResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.3 : 0.6, pointerEvents: lockView ? "none" : "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: lockView ? 0.3 : 1 } }} />
          )}
        </Box>
        <Box ref={panelBContainerRef} onWheel={handlePanelWheel} onDoubleClick={handleDoubleClick} sx={{ ...container.imageBox, width: canvasW, height: canvasH, border: `1px solid ${themeColors.border}` }}>
          <canvas ref={canvasBCorrectedRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH }} />
          {!hideView && (
            <Box onMouseDown={handleMainResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.3 : 0.6, pointerEvents: lockView ? "none" : "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: lockView ? 0.3 : 1 } }} />
          )}
        </Box>
      </Stack>}

      {/* Row 2: Merged view + AlignPad */}
      <Box>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 0.25, width: canvasW }}>
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>Merged</Typography>
            {!hideOverlay && (
              <Select size="small" value={blendMode} onChange={(e) => { if (!lockOverlay) setBlendMode(e.target.value as "blend" | "difference" | "flicker"); }} disabled={lockOverlay} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }}>
                <MenuItem value="blend">Blend</MenuItem>
                <MenuItem value="difference">Diff</MenuItem>
                <MenuItem value="flicker">Flicker</MenuItem>
              </Select>
            )}
          </Stack>
          <Stack direction="row" spacing={`${SPACING.XS}px`} alignItems="center">
            {!hideAlignment && (
              <Button size="small" sx={{ ...compactButton, color: themeColors.accentGreen }} disabled={lockAlignment || !hasAutoAlign || isAtAuto} onClick={() => { if (!lockAlignment) { setDx(autoDx); setDy(autoDy); setRotation(0); } }}>AUTO</Button>
            )}
            {!hideAlignment && (
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockAlignment || !hasOffset} onClick={() => { if (!lockAlignment) { setDx(0); setDy(0); setRotation(0); } }}>ZERO</Button>
            )}
            {!hideExport && (
              <Button size="small" sx={compactButton} disabled={lockExport} onClick={async () => {
                if (lockExport) return;
                if (!mergedCanvasRef.current) return;
                try {
                  const blob = await new Promise<Blob | null>(resolve => mergedCanvasRef.current!.toBlob(resolve, "image/png"));
                  if (!blob) return;
                  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                } catch {
                  mergedCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "align2d_merged.png"); }, "image/png");
                }
              }}>COPY</Button>
            )}
            {!hideExport && (
              <>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }}>Export</Button>
                <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                  <MenuItem disabled={lockExport} onClick={() => handleExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                  <MenuItem disabled={lockExport} onClick={() => handleExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                  <MenuItem disabled={lockExport} onClick={handleExport} sx={{ fontSize: 12 }}>PNG</MenuItem>
                </Menu>
              </>
            )}
            {!hideView && (
              <Button size="small" sx={compactButton} disabled={lockView || !needsReset} onClick={handleDoubleClick}>RESET VIEW</Button>
            )}
          </Stack>
        </Stack>
        <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="flex-start">
          <Box
            ref={mergedContainerRef}
            sx={{ ...container.imageBox, width: canvasW, height: canvasH, cursor: "move" }}
            onMouseDown={handleMouseDown}
            onWheel={handleWheel}
            onDoubleClick={handleDoubleClick}
          >
            <canvas ref={mergedCanvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH }} />
            <canvas ref={uiCanvasRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
            {!hideView && (
              <Box onMouseDown={handleMainResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.3 : 0.6, pointerEvents: lockView ? "none" : "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: lockView ? 0.3 : 1 } }} />
            )}
          </Box>
          <Stack direction="row" spacing={`${SPACING.MD}px`} sx={{ pt: 0.5 }}>
            {!hideAlignment && (
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0.25, width: 90, flexShrink: 0, opacity: lockAlignment ? 0.5 : 1, pointerEvents: lockAlignment ? "none" : "auto" }}>
                <AlignPad
                  dx={dx} dy={dy}
                  maxDx={fineMode ? Math.min(5, effectiveMaxDx) : effectiveMaxDx}
                  maxDy={fineMode ? Math.min(5, effectiveMaxDy) : effectiveMaxDy}
                  onMove={(newDx, newDy) => { if (!lockAlignment) { clampDx(newDx); clampDy(newDy); } }}
                  size={80}
                  theme={themeInfo.theme}
                  accentColor={themeColors.accent}
                />
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: themeColors.textMuted, whiteSpace: "nowrap" }}>
                  <Box component="span" sx={{ color: themeColors.accent }}>{dx >= 0 ? "+" : ""}{dx.toFixed(1)}</Box>
                  {", "}
                  <Box component="span" sx={{ color: themeColors.accent }}>{dy >= 0 ? "+" : ""}{dy.toFixed(1)}</Box>
                  {" px"}
                </Typography>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: themeColors.accent }}>{rotation.toFixed(1)}&deg;</Typography>
                <Stack direction="row" alignItems="center" spacing={0.5}>
                  <Typography sx={{ fontSize: 10, color: themeColors.textMuted }}>Fine:</Typography>
                  <Switch checked={fineMode} onChange={() => { if (!lockAlignment) setFineMode(!fineMode); }} disabled={lockAlignment} size="small" sx={switchStyles.small} />
                </Stack>
              </Box>
            )}
            {!hideHistogram && (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25, opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
                <Stack direction="row" spacing={0.5} alignItems="center">
                  <Typography sx={{ fontSize: 10, color: themeColors.textMuted }}>Histogram:</Typography>
                  <Select size="small" value={histSource} onChange={(e) => { if (!lockHistogram) setHistSource(e.target.value as "a" | "b"); }} disabled={lockHistogram} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 32, fontSize: 10 }}>
                    <MenuItem value="a">A</MenuItem>
                    <MenuItem value="b">B</MenuItem>
                  </Select>
                </Stack>
                <Histogram
                  data={histogramData}
                  vminPct={vminPct}
                  vmaxPct={vmaxPct}
                  onRangeChange={(min, max) => { if (!lockHistogram) { setVminPct(min); setVmaxPct(max); } }}
                  width={110}
                  height={58}
                  theme={themeInfo.theme}
                  dataMin={dataRange.min}
                  dataMax={dataRange.max}
                />
              </Box>
            )}
          </Stack>
        </Stack>
      </Box>

      {/* Row 3: Controls below merged */}
      <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, mt: `${SPACING.SM}px` }}>
        {/* Sliders */}
        <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
          {!hideOverlay && (
            <>
              <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Opacity:</Typography>
              <Slider value={opacity} min={0} max={1} step={0.05} onChange={(_, v) => { if (!lockOverlay) setOpacity(v as number); }} disabled={lockOverlay} size="small" sx={{ ...sliderStyles.small, width: 60 }} />
              <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 20 }}>{Math.round(opacity * 100)}%</Typography>
              <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted, ml: 0.5 }}>Pad:</Typography>
              <Slider value={padding} min={0} max={0.5} step={0.05} onChange={(_, v) => { if (!lockOverlay) setPadding(v as number); }} disabled={lockOverlay} size="small" sx={{ ...sliderStyles.small, width: 50 }} />
              <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 20 }}>{Math.round(padding * 100)}%</Typography>
            </>
          )}
          {!hideDisplay && (
            <>
              <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted, ml: 0.5 }}>Color:</Typography>
              <Select size="small" value={cmap} onChange={(e) => { if (!lockDisplay) setCmap(e.target.value); }} disabled={lockDisplay} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }}>
                {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
              </Select>
            </>
          )}
          {!hideStats && (
            <Typography sx={{ fontSize: 10, color: themeColors.textMuted, ml: 0.5 }}>
              NCC: <Box component="span" sx={{ color: themeColors.textMuted }}>{xcorrZero.toFixed(3)}</Box>
              {" → "}
              <Box component="span" sx={{ color: (nccAligned > 0 ? nccAligned : nccCurrent) > xcorrZero ? themeColors.accentGreen : themeColors.accent, fontWeight: "bold" }}>{(nccAligned > 0 ? nccAligned : nccCurrent).toFixed(3)}</Box>
            </Typography>
          )}
          {!hideView && zoom !== 1 && (
            <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold", ml: 0.5 }}>{zoom.toFixed(1)}x</Typography>
          )}
        </Box>

        {/* Rotation controls */}
        {!hideAlignment && (
          <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockAlignment ? 0.5 : 1, pointerEvents: lockAlignment ? "none" : "auto" }}>
            <IconButton size="small" onClick={() => { if (!lockAlignment) setRotPlaying(!rotPlaying); }} disabled={lockAlignment} sx={{ color: rotPlaying ? themeColors.accent : themeColors.textMuted, p: 0.25 }}>
              {rotPlaying ? <PauseIcon sx={{ fontSize: 16 }} /> : <PlayArrowIcon sx={{ fontSize: 16 }} />}
            </IconButton>
            <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Rot:</Typography>
            <Slider value={rotation} min={-180} max={180} step={fineMode ? 0.1 : 0.5} onChange={(_, v) => { if (!lockAlignment) { if (rotPlaying) setRotPlaying(false); setRotation(v as number); } }} disabled={lockAlignment} size="small" sx={{ ...sliderStyles.small, width: 80 }} />
            <Typography sx={{ ...typography.value, color: themeColors.accent, minWidth: 40 }}>{rotation.toFixed(1)}&deg;</Typography>
            <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>±</Typography>
            <Slider value={rotRange} min={1} max={90} step={1} onChange={(_, v) => { if (!lockAlignment) setRotRange(v as number); }} disabled={lockAlignment} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
            <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 18 }}>{rotRange}&deg;</Typography>
            <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>fps:</Typography>
            <Slider value={rotFps} min={0} max={120} step={1} onChange={(_, v) => { if (!lockAlignment) setRotFps(v as number); }} disabled={lockAlignment} size="small" sx={{ ...sliderStyles.small, width: 35 }} />
            <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 16 }}>{rotFps}</Typography>
            <Button size="small" sx={compactButton} disabled={lockAlignment || rotation === 0} onClick={() => { if (!lockAlignment) { setRotPlaying(false); setRotation(0); } }}>RESET</Button>
          </Box>
        )}

      </Box>
    </Box>
  );
}

export const render = createRender(Align2D);
