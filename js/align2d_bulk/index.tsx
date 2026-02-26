/**
 * Align2DBulk - Bulk alignment verification widget.
 * Side-by-side: Before (raw + moving avg) | After (aligned overlay).
 * Controls below images following Show3D pattern.
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import "./align2d_bulk.css";
import { useTheme } from "../theme";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen } from "../colormaps";
import { extractFloat32, downloadBlob } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, sliderRange, applyLogScale } from "../stats";
import { fft2d, fftshift, computeMagnitude, autoEnhanceFFT } from "../webgpu-fft";
import { computeToolVisibility } from "../tool-parity";
import { ControlCustomizer } from "../control-customizer";

// ============================================================================
// Styles (matching Show3D / Show4DSTEM)
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };

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

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: "6px",
  px: 1,
  py: 0.5,
  width: "fit-content",
};

const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
};

const DPR = window.devicePixelRatio || 1;
const PANEL_SIZE = 350;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

// ============================================================================
// Histogram component (matching Show3D)
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
  data, vminPct, vmaxPct, onRangeChange,
  width = 110, height = 40, theme = "dark", dataMin = 0, dataMax = 1,
}: HistogramProps) {
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
      ctx.fillStyle = (i >= vminBin && i <= vmaxBin) ? colors.barActive : colors.barInactive;
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
// NCC bar chart
// ============================================================================
interface NCCChartProps {
  offsets: { idx: number; dx: number; dy: number; ncc: number }[];
  currentIdx: number;
  referenceIdx: number;
  onSelect: (idx: number) => void;
  width: number;
  height: number;
  theme: "light" | "dark";
}

function NCCChart({ offsets, currentIdx, referenceIdx, onSelect, width, height, theme }: NCCChartProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const isDark = theme === "dark";

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || offsets.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, width, height);
    const margin = { top: 12, bottom: 4, left: 2, right: 2 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    // Title
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("NCC", margin.left, 9);
    const n = offsets.length;
    const barW = Math.max(2, plotW / n - 1);
    const gap = 1;
    for (let i = 0; i < n; i++) {
      const ncc = Math.max(0, offsets[i].ncc);
      const barH = ncc * (plotH - 2);
      const x = margin.left + i * (barW + gap);
      const y = margin.top + plotH - barH;
      if (i === currentIdx) {
        ctx.fillStyle = isDark ? "#4fc3f7" : "#0288d1";
      } else if (i === referenceIdx) {
        ctx.fillStyle = isDark ? "#66bb6a" : "#2e7d32";
      } else {
        ctx.fillStyle = isDark ? "#666" : "#aaa";
      }
      ctx.fillRect(x, y, barW, barH);
    }
  }, [offsets, currentIdx, referenceIdx, width, height, isDark]);

  const handleClick = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || offsets.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const n = offsets.length;
    const plotW = width - 4;
    const barW = Math.max(2, plotW / n - 1);
    const gap = 1;
    const idx = Math.floor((x - 2) / (barW + gap));
    if (idx >= 0 && idx < n) onSelect(idx);
  };

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, cursor: "pointer", border: `1px solid ${isDark ? "#333" : "#ccc"}` }}
      onClick={handleClick}
    />
  );
}

// ============================================================================
// Drift XY trajectory plot
// ============================================================================
interface DriftXYProps {
  offsets: { idx: number; dx: number; dy: number; ncc: number }[];
  currentIdx: number;
  referenceIdx: number;
  size: number;
  theme: "light" | "dark";
  onSelect: (idx: number) => void;
}

function DriftXY({ offsets, currentIdx, referenceIdx, size, theme, onSelect }: DriftXYProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const isDark = theme === "dark";

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || offsets.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, size, size);
    const margin = { top: 14, bottom: 14, left: 22, right: 6 };
    const plotW = size - margin.left - margin.right;
    const plotH = size - margin.top - margin.bottom;
    // Find range
    let dxMin = 0, dxMax = 0, dyMin = 0, dyMax = 0;
    for (const o of offsets) {
      if (o.dx < dxMin) dxMin = o.dx;
      if (o.dx > dxMax) dxMax = o.dx;
      if (o.dy < dyMin) dyMin = o.dy;
      if (o.dy > dyMax) dyMax = o.dy;
    }
    const pad = 0.15;
    const dxRange = Math.max(dxMax - dxMin, 1);
    const dyRange = Math.max(dyMax - dyMin, 1);
    const maxRange = Math.max(dxRange, dyRange);
    const dxCenter = (dxMin + dxMax) / 2;
    const dyCenter = (dyMin + dyMax) / 2;
    const halfRange = maxRange * (1 + pad) / 2;
    const toX = (dx: number) => margin.left + ((dx - dxCenter + halfRange) / (2 * halfRange)) * plotW;
    const toY = (dy: number) => margin.top + ((dy - dyCenter + halfRange) / (2 * halfRange)) * plotH;
    // Title
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("Drift XY (px)", margin.left, 10);
    // Grid / crosshair at origin
    ctx.strokeStyle = isDark ? "#333" : "#ddd";
    ctx.lineWidth = 0.5;
    ctx.setLineDash([2, 2]);
    const ox = toX(0), oy = toY(0);
    ctx.beginPath();
    ctx.moveTo(margin.left, oy);
    ctx.lineTo(size - margin.right, oy);
    ctx.moveTo(ox, margin.top);
    ctx.lineTo(ox, size - margin.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    // Axis labels
    ctx.fillStyle = isDark ? "#666" : "#999";
    ctx.font = "8px monospace";
    ctx.textAlign = "center";
    ctx.fillText("dx", size / 2, size - 2);
    ctx.save();
    ctx.translate(6, size / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("dy", 0, 0);
    ctx.restore();
    // Axis tick values
    ctx.font = "7px monospace";
    ctx.fillStyle = isDark ? "#555" : "#aaa";
    ctx.textAlign = "left";
    ctx.fillText(`${(dxCenter - halfRange).toFixed(0)}`, margin.left, size - margin.bottom + 10);
    ctx.textAlign = "right";
    ctx.fillText(`${(dxCenter + halfRange).toFixed(0)}`, size - margin.right, size - margin.bottom + 10);
    // Trajectory line
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < offsets.length; i++) {
      const x = toX(offsets[i].dx), y = toY(offsets[i].dy);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // Points
    for (let i = 0; i < offsets.length; i++) {
      const x = toX(offsets[i].dx), y = toY(offsets[i].dy);
      if (i === currentIdx) {
        ctx.fillStyle = isDark ? "#4fc3f7" : "#0288d1";
        ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
      } else if (i === referenceIdx) {
        ctx.fillStyle = isDark ? "#66bb6a" : "#2e7d32";
        ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
      } else {
        ctx.fillStyle = isDark ? "#888" : "#999";
        ctx.beginPath(); ctx.arc(x, y, 1.5, 0, Math.PI * 2); ctx.fill();
      }
    }
  }, [offsets, currentIdx, referenceIdx, size, isDark]);

  const handleClick = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || offsets.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    // Find nearest point
    const margin = { top: 14, bottom: 14, left: 22, right: 6 };
    const plotW = size - margin.left - margin.right;
    const plotH = size - margin.top - margin.bottom;
    let dxMin = 0, dxMax = 0, dyMin = 0, dyMax = 0;
    for (const o of offsets) {
      if (o.dx < dxMin) dxMin = o.dx; if (o.dx > dxMax) dxMax = o.dx;
      if (o.dy < dyMin) dyMin = o.dy; if (o.dy > dyMax) dyMax = o.dy;
    }
    const maxRange = Math.max(Math.max(dxMax - dxMin, 1), Math.max(dyMax - dyMin, 1));
    const dxCenter = (dxMin + dxMax) / 2, dyCenter = (dyMin + dyMax) / 2;
    const halfRange = maxRange * 1.15 / 2;
    let bestIdx = -1, bestDist = 15;
    for (let i = 0; i < offsets.length; i++) {
      const x = margin.left + ((offsets[i].dx - dxCenter + halfRange) / (2 * halfRange)) * plotW;
      const y = margin.top + ((offsets[i].dy - dyCenter + halfRange) / (2 * halfRange)) * plotH;
      const dist = Math.hypot(mx - x, my - y);
      if (dist < bestDist) { bestDist = dist; bestIdx = i; }
    }
    if (bestIdx >= 0) onSelect(bestIdx);
  };

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size, cursor: "pointer", border: `1px solid ${isDark ? "#333" : "#ccc"}` }}
      onClick={handleClick}
    />
  );
}

// ============================================================================
// Drift over time plot (dx, dy vs frame)
// ============================================================================
interface DriftTimeProps {
  offsets: { idx: number; dx: number; dy: number; ncc: number }[];
  currentIdx: number;
  referenceIdx: number;
  width: number;
  height: number;
  theme: "light" | "dark";
  onSelect: (idx: number) => void;
}

function DriftTime({ offsets, currentIdx, referenceIdx, width, height, theme, onSelect }: DriftTimeProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const isDark = theme === "dark";

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || offsets.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, width, height);
    const margin = { top: 14, bottom: 14, left: 28, right: 6 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    // Find y range (shared for dx and dy)
    let yMin = 0, yMax = 0;
    for (const o of offsets) {
      if (o.dx < yMin) yMin = o.dx; if (o.dx > yMax) yMax = o.dx;
      if (o.dy < yMin) yMin = o.dy; if (o.dy > yMax) yMax = o.dy;
    }
    const yPad = Math.max((yMax - yMin) * 0.1, 0.5);
    yMin -= yPad; yMax += yPad;
    const yRange = yMax - yMin || 1;
    const n = offsets.length;
    const toX = (i: number) => margin.left + (i / Math.max(1, n - 1)) * plotW;
    const toY = (v: number) => margin.top + plotH - ((v - yMin) / yRange) * plotH;
    // Title
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("Drift (px)", margin.left, 10);
    // Zero line
    ctx.strokeStyle = isDark ? "#333" : "#ddd";
    ctx.lineWidth = 0.5;
    ctx.setLineDash([2, 2]);
    const zy = toY(0);
    ctx.beginPath(); ctx.moveTo(margin.left, zy); ctx.lineTo(width - margin.right, zy); ctx.stroke();
    ctx.setLineDash([]);
    // Y axis labels
    ctx.fillStyle = isDark ? "#555" : "#aaa";
    ctx.font = "7px monospace";
    ctx.textAlign = "right";
    ctx.fillText(`${yMax.toFixed(0)}`, margin.left - 2, margin.top + 5);
    ctx.fillText(`${yMin.toFixed(0)}`, margin.left - 2, height - margin.bottom);
    // dx line (blue)
    ctx.strokeStyle = isDark ? "#4fc3f7" : "#0288d1";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = toX(i), y = toY(offsets[i].dx);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // dy line (orange)
    ctx.strokeStyle = isDark ? "#ffb74d" : "#e65100";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = toX(i), y = toY(offsets[i].dy);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // Current frame indicator
    const cx = toX(currentIdx);
    ctx.strokeStyle = isDark ? "#4fc3f7" : "#0288d1";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(cx, margin.top); ctx.lineTo(cx, height - margin.bottom); ctx.stroke();
    ctx.setLineDash([]);
    // Current frame dots
    const cdx = toY(offsets[currentIdx].dx), cdy = toY(offsets[currentIdx].dy);
    ctx.fillStyle = isDark ? "#4fc3f7" : "#0288d1";
    ctx.beginPath(); ctx.arc(cx, cdx, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = isDark ? "#ffb74d" : "#e65100";
    ctx.beginPath(); ctx.arc(cx, cdy, 3, 0, Math.PI * 2); ctx.fill();
    // Legend
    ctx.font = "8px monospace";
    ctx.fillStyle = isDark ? "#4fc3f7" : "#0288d1";
    ctx.textAlign = "right";
    ctx.fillText("dx", width - margin.right - 16, 10);
    ctx.fillStyle = isDark ? "#ffb74d" : "#e65100";
    ctx.fillText("dy", width - margin.right, 10);
  }, [offsets, currentIdx, referenceIdx, width, height, isDark]);

  const handleClick = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || offsets.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const margin = { left: 28, right: 6 };
    const plotW = width - margin.left - margin.right;
    const n = offsets.length;
    const idx = Math.round(((mx - margin.left) / plotW) * (n - 1));
    if (idx >= 0 && idx < n) onSelect(idx);
  };

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, cursor: "pointer", border: `1px solid ${isDark ? "#333" : "#ccc"}` }}
      onClick={handleClick}
    />
  );
}

// ============================================================================
// Image panel — renders a single float32 image to canvas with colormap
// ============================================================================
function useImagePanel(
  bytes: DataView | null,
  imgW: number,
  imgH: number,
  cmap: string,
  vminPct: number,
  vmaxPct: number,
  zoom: number,
  panX: number,
  panY: number,
  canvasW: number,
  canvasH: number,
  overlay?: { refBytes: DataView | null; opacity: number; blendMode: string; flickerShowRef: boolean },
  cropBox?: { y0: number; y1: number; x0: number; x1: number } | null,
) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const rawDataRef = React.useRef<Float32Array | null>(null);
  const rawRefDataRef = React.useRef<Float32Array | null>(null);
  const offscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const offscreenRefRef = React.useRef<HTMLCanvasElement | null>(null);
  const globalRangeRef = React.useRef<{ min: number; max: number }>({ min: 0, max: 1 });
  const [version, setVersion] = React.useState(0);

  React.useEffect(() => {
    const parsed = extractFloat32(bytes);
    if (!parsed) return;
    rawDataRef.current = parsed;
    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < parsed.length; i++) {
      if (parsed[i] < gMin) gMin = parsed[i];
      if (parsed[i] > gMax) gMax = parsed[i];
    }
    if (overlay?.refBytes) {
      const refParsed = extractFloat32(overlay.refBytes);
      if (refParsed) {
        rawRefDataRef.current = refParsed;
        for (let i = 0; i < refParsed.length; i++) {
          if (refParsed[i] < gMin) gMin = refParsed[i];
          if (refParsed[i] > gMax) gMax = refParsed[i];
        }
      }
    }
    globalRangeRef.current = { min: gMin, max: gMax };
  }, [bytes, overlay?.refBytes]);

  React.useEffect(() => {
    if (!rawDataRef.current || !imgW || !imgH) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.gray;
    const { min: gMin, max: gMax } = globalRangeRef.current;
    const { vmin, vmax } = sliderRange(gMin, gMax, vminPct, vmaxPct);
    offscreenRef.current = renderToOffscreen(rawDataRef.current, imgW, imgH, lut, vmin, vmax);
    if (overlay && rawRefDataRef.current) {
      offscreenRefRef.current = renderToOffscreen(rawRefDataRef.current, imgW, imgH, lut, vmin, vmax);
    }
    setVersion(v => v + 1);
  }, [bytes, overlay?.refBytes, imgW, imgH, cmap, vminPct, vmaxPct]);

  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const off = offscreenRef.current;
    if (!off) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasW * DPR;
    canvas.height = canvasH * DPR;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    ctx.imageSmoothingEnabled = false;
    const dw = canvasW / zoom;
    const dh = canvasH / zoom;

    if (overlay && offscreenRefRef.current) {
      const offRef = offscreenRefRef.current;
      if (overlay.blendMode === "flicker") {
        const src = overlay.flickerShowRef ? offRef : off;
        ctx.drawImage(src, 0, 0, imgW, imgH, 0, 0, dw, dh);
      } else {
        ctx.globalAlpha = 1 - overlay.opacity;
        ctx.drawImage(offRef, 0, 0, imgW, imgH, 0, 0, dw, dh);
        ctx.globalAlpha = overlay.opacity;
        ctx.drawImage(off, 0, 0, imgW, imgH, 0, 0, dw, dh);
        ctx.globalAlpha = 1;
      }
    } else {
      ctx.drawImage(off, 0, 0, imgW, imgH, 0, 0, dw, dh);
    }

    if (cropBox) {
      const sx = dw / imgW;
      const sy = dh / imgH;
      ctx.strokeStyle = "rgba(255,255,0,0.6)";
      ctx.lineWidth = 1.5 / zoom;
      ctx.setLineDash([4 / zoom, 4 / zoom]);
      ctx.strokeRect(
        cropBox.x0 * sx, cropBox.y0 * sy,
        (cropBox.x1 - cropBox.x0) * sx, (cropBox.y1 - cropBox.y0) * sy,
      );
      ctx.setLineDash([]);
    }

    ctx.restore();
  }, [version, canvasW, canvasH, imgW, imgH, zoom, panX, panY,
      overlay?.opacity, overlay?.blendMode, overlay?.flickerShowRef, cropBox]);

  return canvasRef;
}

// ============================================================================
// FFT panel — computes and renders FFT of a float32 image
// ============================================================================
function useFFTPanel(
  bytes: DataView | null,
  imgW: number,
  imgH: number,
  zoom: number,
  panX: number,
  panY: number,
  canvasW: number,
  canvasH: number,
  theme: "light" | "dark",
) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const offscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const [version, setVersion] = React.useState(0);

  React.useEffect(() => {
    const parsed = extractFloat32(bytes);
    if (!parsed || !imgW || !imgH) return;
    const complex = fft2d(parsed, imgW, imgH);
    const shifted = fftshift(complex, imgW, imgH);
    const mag = computeMagnitude(shifted);
    const enhanced = autoEnhanceFFT(mag, imgW, imgH);
    const log = applyLogScale(enhanced);
    const lut = COLORMAPS.gray;
    const range = findDataRange(log);
    offscreenRef.current = renderToOffscreen(log, imgW, imgH, lut, range.min, range.max);
    setVersion(v => v + 1);
  }, [bytes, imgW, imgH]);

  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const off = offscreenRef.current;
    if (!off) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasW * DPR;
    canvas.height = canvasH * DPR;
    ctx.scale(DPR, DPR);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    ctx.imageSmoothingEnabled = false;
    const dw = canvasW / zoom;
    const dh = canvasH / zoom;
    ctx.drawImage(off, 0, 0, imgW, imgH, 0, 0, dw, dh);
    ctx.restore();
  }, [version, canvasW, canvasH, imgW, imgH, zoom, panX, panY]);

  return canvasRef;
}

// ============================================================================
// Main component
// ============================================================================
function Align2DBulk() {
  const { themeInfo, colors: baseColors } = useTheme();
  const isDark = themeInfo.theme === "dark";

  const themeColors = {
    ...baseColors,
    accentBlue: isDark ? "#4fc3f7" : "#0288d1",
    accentGreen: isDark ? "#66bb6a" : "#2e7d32",
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

  // Model state
  const [nImages] = useModelState<number>("n_images");
  const [currentIdx, setCurrentIdx] = useModelState<number>("current_idx");
  const [imgH] = useModelState<number>("height");
  const [imgW] = useModelState<number>("width");
  const [rawH] = useModelState<number>("raw_height");
  const [rawW] = useModelState<number>("raw_width");
  const [refBytes] = useModelState<DataView>("ref_bytes");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [rawFrameBytes] = useModelState<DataView>("raw_frame_bytes");
  const [offsetsJson] = useModelState<string>("offsets_json");
  const [cropBoxJson] = useModelState<string>("crop_box_json");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [opacity, setOpacity] = useModelState<number>("opacity");
  const [avgWindow, setAvgWindow] = useModelState<number>("avg_window");
  const [referenceIdx] = useModelState<number>("reference_idx");
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");

  // Tool visibility
  const vis = React.useMemo(
    () => computeToolVisibility("Align2DBulk", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideDisplay = vis.isHidden("display");
  const lockDisplay = vis.isLocked("display");
  const hideHistogram = vis.isHidden("histogram");
  const lockHistogram = vis.isLocked("histogram");
  const hideNavigation = vis.isHidden("navigation");
  const lockNavigation = vis.isLocked("navigation");
  const hideStats = vis.isHidden("stats");
  const hideExport = vis.isHidden("export");
  const lockExport = vis.isLocked("export");
  const hideView = vis.isHidden("view");
  const lockView = vis.isLocked("view");

  // Parse offsets + crop box
  const offsets = React.useMemo(() => {
    try { return JSON.parse(offsetsJson) as { idx: number; dx: number; dy: number; ncc: number }[]; }
    catch { return []; }
  }, [offsetsJson]);

  const cropBox = React.useMemo(() => {
    if (!cropBoxJson) return null;
    try { return JSON.parse(cropBoxJson) as { y0: number; y1: number; x0: number; x1: number }; }
    catch { return null; }
  }, [cropBoxJson]);

  const currentOffset = offsets[currentIdx] || { dx: 0, dy: 0, ncc: 0 };

  // Frame slider step matches avg window
  const frameStep = avgWindow > 1 ? avgWindow : 1;

  // Avg window options
  const avgOptions = React.useMemo(() => {
    const opts = [1, 5, 10, 20];
    if (nImages > 20 && !opts.includes(nImages)) opts.push(nImages);
    return opts.filter(v => v <= nImages);
  }, [nImages]);

  // Shared view state
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [blendMode, setBlendMode] = React.useState<"blend" | "flicker">("blend");
  const [flickerShowRef, setFlickerShowRef] = React.useState(true);
  const [vminPct, setVminPct] = React.useState(0);
  const [vmaxPct, setVmaxPct] = React.useState(100);
  const [showFFT, setShowFFT] = React.useState(false);

  // Histogram data from aligned frame
  const histData = React.useMemo(() => extractFloat32(frameBytes), [frameBytes]);
  const dataRange = React.useMemo(() => {
    if (!histData) return { min: 0, max: 1 };
    return findDataRange(histData);
  }, [histData]);

  // Resizable canvas
  const [canvasSize, setCanvasSize] = React.useState(PANEL_SIZE);
  const resizeRef = React.useRef<{ startX: number; startY: number; size: number } | null>(null);
  const [isResizing, setIsResizing] = React.useState(false);

  // Flicker timer
  React.useEffect(() => {
    if (blendMode !== "flicker") return;
    const interval = setInterval(() => setFlickerShowRef(p => !p), 333);
    return () => clearInterval(interval);
  }, [blendMode]);

  // Canvas sizing — both panels same width AND height
  const panelW = canvasSize;
  const maxAR = Math.max(rawH / Math.max(rawW, 1), imgH / Math.max(imgW, 1));
  const panelH = Math.round(panelW * maxAR);
  const totalW = panelW * 2 + SPACING.SM;

  // "Before" panel (raw + moving average + crop boundary)
  const beforeRef = useImagePanel(
    rawFrameBytes, rawW, rawH, cmap, vminPct, vmaxPct,
    zoom, panX, panY, panelW, panelH,
    undefined, cropBox,
  );

  // "After" panel (aligned overlay)
  const afterRef = useImagePanel(
    frameBytes, imgW, imgH, cmap, vminPct, vmaxPct,
    zoom, panX, panY, panelW, panelH,
    { refBytes, opacity, blendMode, flickerShowRef },
  );

  // FFT panel (computed from aligned frame)
  const fftRef = useFFTPanel(
    showFFT ? frameBytes : null, imgW, imgH,
    zoom, panX, panY, panelW, panelH, themeInfo.theme,
  );

  // Shared zoom/pan handlers
  const handleWheel = (e: React.WheelEvent) => {
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * zoomFactor));
    setZoom(newZoom);
  };
  const handleDoubleClick = () => { setZoom(1); setPanX(0); setPanY(0); };

  // Pan drag
  const dragRef = React.useRef<{ startX: number; startY: number; pX: number; pY: number } | null>(null);
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    dragRef.current = { startX: e.clientX, startY: e.clientY, pX: panX, pY: panY };
  };
  React.useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      setPanX(dragRef.current.pX + (e.clientX - dragRef.current.startX));
      setPanY(dragRef.current.pY + (e.clientY - dragRef.current.startY));
    };
    const handleUp = () => { dragRef.current = null; };
    document.addEventListener("mousemove", handleMove);
    document.addEventListener("mouseup", handleUp);
    return () => { document.removeEventListener("mousemove", handleMove); document.removeEventListener("mouseup", handleUp); };
  }, []);

  // Prevent wheel scroll propagation
  const containerRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const prevent = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", prevent, { passive: false });
    return () => el.removeEventListener("wheel", prevent);
  }, []);

  // Resize handle
  const handleResizeStart = React.useCallback((e: React.MouseEvent) => {
    if (lockView) return;
    e.stopPropagation();
    e.preventDefault();
    setIsResizing(true);
    resizeRef.current = { startX: e.clientX, startY: e.clientY, size: canvasSize };
  }, [canvasSize, lockView]);

  React.useEffect(() => {
    if (!isResizing) return;
    let rafId = 0;
    let latestSize = resizeRef.current ? resizeRef.current.size : canvasSize;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeRef.current) return;
      const delta = Math.max(e.clientX - resizeRef.current.startX, e.clientY - resizeRef.current.startY);
      latestSize = Math.max(200, Math.min(1200, resizeRef.current.size + delta));
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
      resizeRef.current = null;
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing]);

  // Keyboard — step by avg_window
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowLeft" || e.key === "a" || e.key === "A") {
      e.preventDefault();
      setCurrentIdx(Math.max(0, currentIdx - frameStep));
    } else if (e.key === "ArrowRight" || e.key === "d" || e.key === "D") {
      e.preventDefault();
      setCurrentIdx(Math.min(nImages - 1, currentIdx + frameStep));
    } else if (e.key === "r" || e.key === "R") {
      handleDoubleClick();
    }
  };

  // Export
  const handleExportPNG = () => {
    const canvas = afterRef.current;
    if (!canvas || lockExport) return;
    canvas.toBlob((b) => { if (b) downloadBlob(b, "align2d_bulk.png"); }, "image/png");
  };
  const handleCopy = async () => {
    const canvas = afterRef.current;
    if (!canvas || lockExport) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/png"));
      if (blob) await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch {
      handleExportPNG();
    }
  };

  const canvasStyle = { cursor: "move", display: "block" } as const;
  const panelLabel = { ...typography.labelSmall, color: themeColors.textMuted, mb: 0.25 };
  const chartH = 90;
  const driftXYSize = chartH;
  const nccW = Math.round((totalW - driftXYSize - SPACING.XS * 2) * 0.4);
  const driftTimeW = totalW - nccW - driftXYSize - SPACING.XS * 2;

  return (
    <Box
      className="align2d-bulk-root"
      sx={{ p: 1.5, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" }}
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      {/* Title + settings */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 0.5, width: totalW }}>
        {title && (
          <Typography sx={{ ...typography.label, fontWeight: "bold" }}>{title}</Typography>
        )}
        <ControlCustomizer
          widgetName="Align2DBulk"
          hiddenTools={hiddenTools}
          setHiddenTools={setHiddenTools}
          disabledTools={disabledTools}
          setDisabledTools={setDisabledTools}
          themeColors={themeColors}
        />
      </Stack>

      {/* ═══ Side-by-side panels ═══ */}
      <Box
        ref={containerRef}
        sx={{ display: "flex", gap: `${SPACING.SM}px`, position: "relative" }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
      >
        {/* Before panel */}
        <Box>
          <Typography sx={panelLabel}>
            Before{avgWindow > 1 ? ` (avg ${avgWindow})` : ""}
          </Typography>
          <Box sx={{ bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" }}>
            <canvas ref={beforeRef} style={{ ...canvasStyle, width: panelW, height: panelH }} />
          </Box>
        </Box>

        {/* After panel */}
        <Box>
          <Typography sx={panelLabel}>
            {showFFT ? "FFT (aligned)" : "After (aligned)"}
          </Typography>
          <Box sx={{ bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" }}>
            <canvas
              ref={showFFT ? fftRef : afterRef}
              style={{ ...canvasStyle, width: panelW, height: panelH }}
            />
            {/* Resize handle */}
            {!hideView && (
              <Box
                onMouseDown={handleResizeStart}
                sx={{
                  position: "absolute",
                  bottom: 0,
                  right: 0,
                  width: 16,
                  height: 16,
                  cursor: lockView ? "default" : "nwse-resize",
                  opacity: lockView ? 0.3 : 0.6,
                  pointerEvents: lockView ? "none" : "auto",
                  background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
                  "&:hover": { opacity: lockView ? 0.3 : 1 },
                }}
              />
            )}
          </Box>
        </Box>
      </Box>

      {/* ═══ Controls below ═══ */}
      <Box sx={{ width: totalW, mt: 1 }}>
        {/* Frame slider */}
        {!hideNavigation && (
          <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, width: totalW, boxSizing: "border-box", opacity: lockNavigation ? 0.5 : 1, pointerEvents: lockNavigation ? "none" : "auto" }}>
            <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted, flexShrink: 0 }}>Frame</Typography>
            <Slider
              value={currentIdx}
              onChange={(_, v) => { if (!lockNavigation) setCurrentIdx(v as number); }}
              min={0} max={Math.max(0, nImages - 1)} step={frameStep}
              size="small" valueLabelDisplay="auto" disabled={lockNavigation}
              sx={{ ...sliderStyles.small, flex: 1 }}
            />
            <Typography sx={{ ...typography.value, color: themeColors.textMuted, flexShrink: 0 }}>
              {currentIdx}/{nImages - 1}
            </Typography>
          </Box>
        )}

        {/* Stats + charts row */}
        {!hideStats && (
          <Box sx={{ mt: `${SPACING.XS}px` }}>
            <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, width: totalW, boxSizing: "border-box" }}>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                dx={currentOffset.dx.toFixed(2)} dy={currentOffset.dy.toFixed(2)}
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.accentBlue }}>
                NCC={currentOffset.ncc.toFixed(4)}
              </Typography>
              {currentIdx === referenceIdx && (
                <Typography sx={{ ...typography.value, color: themeColors.accentGreen }}>(ref)</Typography>
              )}
            </Box>
            {/* Three charts in a row: NCC | Drift XY | Drift over time */}
            <Box sx={{ display: "flex", gap: `${SPACING.XS}px`, mt: `${SPACING.XS}px` }}>
              <NCCChart
                offsets={offsets}
                currentIdx={currentIdx}
                referenceIdx={referenceIdx}
                onSelect={(idx) => { if (!lockNavigation) setCurrentIdx(idx); }}
                width={nccW}
                height={chartH}
                theme={themeInfo.theme}
              />
              <DriftXY
                offsets={offsets}
                currentIdx={currentIdx}
                referenceIdx={referenceIdx}
                size={driftXYSize}
                theme={themeInfo.theme}
                onSelect={(idx) => { if (!lockNavigation) setCurrentIdx(idx); }}
              />
              <DriftTime
                offsets={offsets}
                currentIdx={currentIdx}
                referenceIdx={referenceIdx}
                width={driftTimeW}
                height={chartH}
                theme={themeInfo.theme}
                onSelect={(idx) => { if (!lockNavigation) setCurrentIdx(idx); }}
              />
            </Box>
          </Box>
        )}

        {/* Display controls + Histogram */}
        {(!hideDisplay || !hideHistogram) && (
          <Box sx={{ display: "flex", alignItems: "flex-start", gap: `${SPACING.SM}px`, mt: `${SPACING.SM}px` }}>
            {/* Left: control rows */}
            {!hideDisplay && (
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Avg:</Typography>
                  <Select
                    value={avgWindow}
                    onChange={(e) => { if (!lockDisplay) setAvgWindow(Number(e.target.value)); }}
                    size="small" disabled={lockDisplay}
                    sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }}
                    MenuProps={themedMenuProps}
                  >
                    {avgOptions.map((v) => (
                      <MenuItem key={v} value={v}>
                        {v === 1 ? "1" : v === nImages ? `All (${v})` : String(v)}
                      </MenuItem>
                    ))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color:</Typography>
                  <Select
                    value={cmap}
                    onChange={(e) => { if (!lockDisplay) setCmap(e.target.value); }}
                    size="small" disabled={lockDisplay}
                    sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }}
                    MenuProps={themedMenuProps}
                  >
                    {COLORMAP_NAMES.map((name) => (
                      <MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>
                    ))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>FFT:</Typography>
                  <Switch checked={showFFT} onChange={(e) => setShowFFT(e.target.checked)} size="small" sx={switchStyles.small} />
                  {zoom !== 1 && (
                    <Typography sx={{ ...typography.value, color: themeColors.accent }}>{zoom.toFixed(1)}×</Typography>
                  )}
                </Box>
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Mode:</Typography>
                  <Select
                    value={blendMode}
                    onChange={(e) => { if (!lockDisplay) setBlendMode(e.target.value as "blend" | "flicker"); }}
                    size="small" disabled={lockDisplay}
                    sx={{ ...themedSelect, minWidth: 55, fontSize: 10 }}
                    MenuProps={themedMenuProps}
                  >
                    <MenuItem value="blend">Blend</MenuItem>
                    <MenuItem value="flicker">Flicker</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Opacity</Typography>
                  <Slider
                    value={opacity}
                    onChange={(_, v) => { if (!lockDisplay) setOpacity(v as number); }}
                    min={0} max={1} step={0.01} size="small" disabled={lockDisplay}
                    sx={{ ...sliderStyles.small, width: 60 }}
                  />
                </Box>
              </Box>
            )}
            {/* Right: Histogram */}
            {!hideHistogram && (
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
                <Histogram
                  data={histData}
                  vminPct={vminPct}
                  vmaxPct={vmaxPct}
                  onRangeChange={(min, max) => { if (!lockHistogram) { setVminPct(min); setVmaxPct(max); } }}
                  width={110}
                  height={58}
                  theme={themeInfo.theme === "dark" ? "dark" : "light"}
                  dataMin={dataRange.min}
                  dataMax={dataRange.max}
                />
              </Box>
            )}
          </Box>
        )}

        {/* Export */}
        {!hideExport && (
          <Stack direction="row" spacing={0.5} sx={{ mt: `${SPACING.XS}px` }}>
            <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={handleCopy}>COPY</Button>
            <Button size="small" sx={compactButton} disabled={lockExport} onClick={handleExportPNG}>PNG</Button>
          </Stack>
        )}
      </Box>
    </Box>
  );
}

export const render = createRender(Align2DBulk);
