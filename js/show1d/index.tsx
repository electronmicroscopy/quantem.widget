/**
 * Show1D - Interactive 1D data viewer for spectra, profiles, and time series.
 * Self-contained widget with all utilities inlined.
 *
 * Features:
 * - Scroll to zoom, drag to pan, double-click to reset
 * - Multi-trace overlay with distinct colors and legend
 * - Cursor crosshair with snapped value readout
 * - Calibrated X/Y axes with nice tick values
 * - Log scale, grid lines, stats bar
 * - Publication-quality figure export
 * - Automatic theme detection (light/dark mode)
 * - Peak markers with local max search and selectable peaks
 * - Grid density slider
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Tooltip from "@mui/material/Tooltip";
import "./styles.css";
import { useTheme } from "../theme";
import { roundToNiceValue, canvasToPDF } from "../scalebar";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { findDataRange, percentileClip } from "../stats";
import { ControlCustomizer } from "../control-customizer";
import { computeToolVisibility } from "../tool-parity";

// ============================================================================
// UI Styles (per-widget, matching Show3D/Show4DSTEM)
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

const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
};

// ============================================================================
// Types
// ============================================================================
interface PeakMarker {
  x: number;
  y: number;
  trace_idx: number;
  label: string;
  type?: "peak";
}

// ============================================================================
// Constants
// ============================================================================
const DPR = window.devicePixelRatio || 1;
const UNFOCUSED_ALPHA = 0.2;

/** Snap a coordinate to the nearest pixel boundary for crisp 1px lines. */
function snap(v: number): number {
  return Math.round(v) + 0.5;
}
const DEFAULT_CANVAS_W = 600;
const DEFAULT_CANVAS_H = 400;
const MARGIN_TOP = 12;
const MARGIN_RIGHT = 16;
const MARGIN_BOTTOM_BASE = 28; // tick marks + tick labels
const MARGIN_BOTTOM_LABEL = 18; // extra when axis label present
const MARGIN_LEFT_MIN = 48;
const MARGIN_LEFT_LABEL = 26; // extra when Y-axis label present
const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const AXIS_TICK_PX = 4;
const MAX_TICKS_Y = 8;
const TICK_LABEL_WIDTH_PX = 55;
const Y_PAD_FRAC = 0.05;
// Peak drag-to-search types
interface PeakSearchRegion {
  x0: number;       // data X start
  x1: number;       // data X end
  peakX: number;    // found peak data X
  peakY: number;    // found peak data Y
  traceIdx: number; // which trace
}

// ============================================================================
// InfoTooltip (per-widget, matching Show3D)
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
        &#9432;
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
// Tick computation
// ============================================================================
function computeTicks(min: number, max: number, maxTicks: number = MAX_TICKS_Y): number[] {
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

function computeLogTicks(min: number, max: number, maxTicks: number = MAX_TICKS_Y): number[] {
  const logMin = Math.floor(Math.log10(Math.max(min, 1e-30)));
  const logMax = Math.ceil(Math.log10(Math.max(max, 1e-30)));
  const totalDecades = logMax - logMin;
  const step = totalDecades <= maxTicks ? 1 : Math.ceil(totalDecades / maxTicks);
  const ticks: number[] = [];
  for (let e = logMin; e <= logMax; e += step) {
    const val = Math.pow(10, e);
    if (val >= min && val <= max) ticks.push(val);
  }
  if (ticks.length === 0) ticks.push(min, max);
  return ticks;
}

// ============================================================================
// Main Component
// ============================================================================
function Show1D() {
  // Theme
  const { themeInfo, colors } = useTheme();
  const isDark = themeInfo.theme === "dark";

  // Model state (synced from Python)
  const [yData] = useModelState<DataView>("y_bytes");
  const [xData] = useModelState<DataView>("x_bytes");
  const [nTraces] = useModelState<number>("n_traces");
  const [nPoints] = useModelState<number>("n_points");
  const [traceLabels] = useModelState<string[]>("labels");
  const [traceColors] = useModelState<string[]>("colors");
  const [title] = useModelState<string>("title");
  const [xLabel] = useModelState<string>("x_label");
  const [yLabel] = useModelState<string>("y_label");
  const [xUnit] = useModelState<string>("x_unit");
  const [yUnit] = useModelState<string>("y_unit");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [percentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh] = useModelState<number>("percentile_high");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showLegend, setShowLegend] = useModelState<boolean>("show_legend");
  const [showGrid, setShowGrid] = useModelState<boolean>("show_grid");
  const [showControls] = useModelState<boolean>("show_controls");
  const [lineWidth] = useModelState<number>("line_width");
  const [statsMean] = useModelState<number[]>("stats_mean");
  const [statsMin] = useModelState<number[]>("stats_min");
  const [statsMax] = useModelState<number[]>("stats_max");
  const [statsStd] = useModelState<number[]>("stats_std");
  const [focusedTrace, setFocusedTrace] = useModelState<number>("focused_trace");
  const [peakMarkers, setPeakMarkers] = useModelState<PeakMarker[]>("peak_markers");
  const [peakActive, setPeakActive] = useModelState<boolean>("peak_active");
  const [peakSearchRadius, setPeakSearchRadius] = useModelState<number>("peak_search_radius");
  const [selectedPeaks, setSelectedPeaks] = useModelState<number[]>("selected_peaks");
  const [gridDensity, setGridDensity] = useModelState<number>("grid_density");
  const [modelXRange, setModelXRange] = useModelState<number[]>("x_range");
  const [modelYRange, setModelYRange] = useModelState<number[]>("y_range");
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");

  // Range stats (Feature 1+4)
  const [rangeStats] = useModelState<Array<{mean: number; min: number; max: number; std: number; integral: number; n_points: number}>>("range_stats");

  // Peak FWHM (Feature 5)
  const [peakFwhm] = useModelState<Array<{peak_idx: number; fwhm: number | null; center?: number; amplitude?: number; sigma?: number; offset?: number; fit_quality?: number; error?: string}>>("peak_fwhm");

  // Tool visibility
  const toolVisibility = React.useMemo(
    () => computeToolVisibility("Show1D", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideDisplay = toolVisibility.isHidden("display");
  const hidePeaks = toolVisibility.isHidden("peaks");
  const hideStats = toolVisibility.isHidden("stats");
  const hideExport = toolVisibility.isHidden("export");
  const lockDisplay = toolVisibility.isLocked("display");
  const lockPeaks = toolVisibility.isLocked("peaks");
  const lockExport = toolVisibility.isLocked("export");

  // Canvas refs
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const uiCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Canvas size
  const [canvasW, setCanvasW] = React.useState(DEFAULT_CANVAS_W);
  const [canvasH, setCanvasH] = React.useState(DEFAULT_CANVAS_H);

  // View state (data coordinate ranges)
  const [xMin, setXMin] = React.useState(0);
  const [xMax, setXMax] = React.useState(1);
  const [yMin, setYMin] = React.useState(0);
  const [yMax, setYMax] = React.useState(1);
  const [xLocked, setXLocked] = React.useState(false);
  const xLockedRef = React.useRef(false);
  xLockedRef.current = xLocked;
  const [yLocked, setYLocked] = React.useState(false);
  const yLockedRef = React.useRef(false);
  yLockedRef.current = yLocked;
  // Axis drag selection state
  const axisDragRef = React.useRef<{
    axis: "x" | "y";
    startPx: number;
    startVal: number;
  } | null>(null);
  const [axisDragCurrent, setAxisDragCurrent] = React.useState<{ axis: "x" | "y"; startVal: number; currentVal: number } | null>(null);

  // Data ranges (for reset)
  const xDataRangeRef = React.useRef({ min: 0, max: 1 });
  const yDataRangeRef = React.useRef({ min: 0, max: 1 });

  // Extracted data
  const tracesRef = React.useRef<Float32Array[]>([]);
  const xValuesRef = React.useRef<Float32Array | null>(null);

  // Cursor state
  const [cursorInfo, setCursorInfo] = React.useState<{
    canvasX: number;
    canvasY: number;
    dataX: number;
    dataY: number;
    traceIdx: number;
    label: string;
    color: string;
  } | null>(null);
  const [hoverPeakIdx, setHoverPeakIdx] = React.useState<number | null>(null);

  // Peak drag-to-search state
  const peakDragRef = React.useRef<{
    active: boolean;
    startPx: number;        // canvas X pixel where drag started
    startDataX: number;     // data X where drag started
    traceIdx: number;       // which trace to search
  } | null>(null);
  const [peakSearchRegion, setPeakSearchRegion] = React.useState<PeakSearchRegion | null>(null);

  // Legend geometry for hit detection
  const legendGeoRef = React.useRef<{ lx: number; ly: number; w: number; h: number; entryH: number; pad: number; n: number } | null>(null);
  const hoverLegendRef = React.useRef(false);

  // Drag state
  const dragRef = React.useRef<{
    active: boolean;
    wasDrag: boolean;
    startX: number;
    startY: number;
    startXMin: number;
    startXMax: number;
    startYMin: number;
    startYMax: number;
  } | null>(null);

  // Resize handle state
  const resizeDragRef = React.useRef<{
    active: boolean;
    startX: number;
    startY: number;
    startW: number;
    startH: number;
  } | null>(null);

  // Range handle drag state
  const rangeDragRef = React.useRef<{
    handle: "left" | "right";
    startPx: number;
  } | null>(null);
  const [hoverRangeHandle, setHoverRangeHandle] = React.useState<"left" | "right" | null>(null);

  // Range input local state (committed on blur/enter)
  const [rangeInputMin, setRangeInputMin] = React.useState<string>("");
  const [rangeInputMax, setRangeInputMax] = React.useState<string>("");
  const [rangeInputActive, setRangeInputActive] = React.useState(false);

  // Y-range handle drag state (Feature 2)
  const rangeDragYRef = React.useRef<{
    handle: "top" | "bottom";
    startPx: number;
  } | null>(null);
  const [hoverRangeHandleY, setHoverRangeHandleY] = React.useState<"top" | "bottom" | null>(null);
  const [rangeInputYMin, setRangeInputYMin] = React.useState<string>("");
  const [rangeInputYMax, setRangeInputYMax] = React.useState<string>("");
  const [rangeInputYActive, setRangeInputYActive] = React.useState(false);

  // CSV copy feedback (Feature 6)
  const [csvCopied, setCsvCopied] = React.useState(false);

  // Export
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);

  // Dynamic margins based on tick label widths and axis labels
  const margin = React.useMemo(() => {
    const hasYLabel = !!(yLabel || yUnit);
    const hasXLabel = !!(xLabel || xUnit);

    // Measure max Y tick label width using offscreen canvas
    const yTicks = logScale
      ? computeLogTicks(Math.max(yMin, 1e-30), yMax)
      : computeTicks(yMin, yMax);
    let maxTickW = 30; // minimum fallback
    try {
      const offCtx = document.createElement("canvas").getContext("2d");
      if (offCtx) {
        offCtx.font = `11px ${FONT}`;
        for (const v of yTicks) {
          const w = offCtx.measureText(formatNumber(v)).width;
          if (w > maxTickW) maxTickW = w;
        }
      }
    } catch { /* fallback */ }

    return {
      top: MARGIN_TOP,
      right: MARGIN_RIGHT,
      bottom: MARGIN_BOTTOM_BASE + (hasXLabel ? MARGIN_BOTTOM_LABEL : 0),
      left: Math.max(MARGIN_LEFT_MIN, maxTickW + AXIS_TICK_PX + 6 + (hasYLabel ? MARGIN_LEFT_LABEL : 0)),
    };
  }, [yMin, yMax, logScale, yLabel, yUnit, xLabel, xUnit]);

  // Plot area dimensions
  const plotW = canvasW - margin.left - margin.right;
  const plotH = canvasH - margin.top - margin.bottom;

  // ========================================================================
  // Data extraction
  // ========================================================================
  React.useEffect(() => {
    if (!yData || yData.byteLength < 4 || nTraces < 1 || nPoints < 1) {
      tracesRef.current = [];
      return;
    }
    const allY = extractFloat32(yData);
    if (!allY) {
      tracesRef.current = [];
      return;
    }
    const traces: Float32Array[] = [];
    for (let t = 0; t < nTraces; t++) {
      const offset = t * nPoints;
      traces.push(allY.slice(offset, offset + nPoints));
    }
    tracesRef.current = traces;

    // X values
    if (xData && xData.byteLength >= 4) {
      xValuesRef.current = extractFloat32(xData);
    } else {
      xValuesRef.current = null;
    }

    // Compute data ranges
    let gXMin = 0;
    let gXMax = nPoints - 1;
    if (xValuesRef.current && xValuesRef.current.length > 0) {
      const xRange = findDataRange(xValuesRef.current);
      gXMin = xRange.min;
      gXMax = xRange.max;
    }
    if (gXMin === gXMax) gXMax = gXMin + 1;

    let gYMin = Infinity;
    let gYMax = -Infinity;
    if (autoContrast && traces.length > 0) {
      // Concatenate all visible trace data for percentile clipping
      const totalLen = traces.reduce((s, t) => s + t.length, 0);
      const allData = new Float32Array(totalLen);
      let offset = 0;
      for (const trace of traces) {
        allData.set(trace, offset);
        offset += trace.length;
      }
      const clipped = percentileClip(allData, percentileLow, percentileHigh);
      gYMin = clipped.vmin;
      gYMax = clipped.vmax;
    } else {
      for (const trace of traces) {
        const r = findDataRange(trace);
        if (r.min < gYMin) gYMin = r.min;
        if (r.max > gYMax) gYMax = r.max;
      }
    }
    if (!isFinite(gYMin)) gYMin = 0;
    if (!isFinite(gYMax)) gYMax = 1;
    if (gYMin === gYMax) { gYMin -= 0.5; gYMax += 0.5; }

    // Pad Y range
    const yPad = (gYMax - gYMin) * Y_PAD_FRAC;
    gYMin -= yPad;
    gYMax += yPad;

    xDataRangeRef.current = { min: gXMin, max: gXMax };
    yDataRangeRef.current = { min: gYMin, max: gYMax };

    setXMin(gXMin);
    setXMax(gXMax);
    setYMin(gYMin);
    setYMax(gYMax);
  }, [yData, xData, nTraces, nPoints, autoContrast, percentileLow, percentileHigh]);

  // ========================================================================
  // Coordinate transforms
  // ========================================================================
  const dataToCanvasX = React.useCallback(
    (dx: number) => margin.left + ((dx - xMin) / (xMax - xMin)) * plotW,
    [xMin, xMax, plotW, margin.left],
  );
  const dataToCanvasY = React.useCallback(
    (dy: number) => {
      if (logScale) {
        const lMin = Math.log10(Math.max(yMin, 1e-30));
        const lMax = Math.log10(Math.max(yMax, 1e-30));
        const lVal = Math.log10(Math.max(dy, 1e-30));
        return margin.top + plotH - ((lVal - lMin) / (lMax - lMin || 1)) * plotH;
      }
      return margin.top + plotH - ((dy - yMin) / (yMax - yMin || 1)) * plotH;
    },
    [yMin, yMax, plotH, logScale, margin.top],
  );
  const canvasToDataX = React.useCallback(
    (cx: number) => xMin + ((cx - margin.left) / plotW) * (xMax - xMin),
    [xMin, xMax, plotW, margin.left],
  );
  const canvasToDataY = React.useCallback(
    (cy: number) => {
      const frac = (margin.top + plotH - cy) / plotH;
      if (logScale) {
        const lMin = Math.log10(Math.max(yMin, 1e-30));
        const lMax = Math.log10(Math.max(yMax, 1e-30));
        return Math.pow(10, lMin + frac * (lMax - lMin));
      }
      return yMin + frac * (yMax - yMin);
    },
    [yMin, yMax, plotH, logScale, margin.top],
  );

  // ========================================================================
  // Reset view
  // ========================================================================
  const resetView = React.useCallback(() => {
    setXLocked(false);
    setYLocked(false);
    setModelXRange([]);
    setModelYRange([]);
    setXMin(xDataRangeRef.current.min);
    setXMax(xDataRangeRef.current.max);
    setYMin(yDataRangeRef.current.min);
    setYMax(yDataRangeRef.current.max);
  }, [setModelXRange, setModelYRange]);

  // ========================================================================
  // Main canvas render
  // ========================================================================
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = canvasW * DPR;
    canvas.height = canvasH * DPR;
    ctx.scale(DPR, DPR);

    // Background
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f8f8f8";
    ctx.fillRect(0, 0, canvasW, canvasH);

    if (plotW <= 0 || plotH <= 0) return;

    const traces = tracesRef.current;
    const xVals = xValuesRef.current;

    // Grid
    if (showGrid) {
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);

      // X grid — use gridDensity for tick count
      const xGridCount = Math.max(3, gridDensity);
      const xTicks = computeTicks(xMin, xMax, xGridCount);
      for (const tv of xTicks) {
        const cx = snap(dataToCanvasX(tv));
        if (cx >= margin.left && cx <= margin.left + plotW) {
          ctx.beginPath();
          ctx.moveTo(cx, margin.top);
          ctx.lineTo(cx, margin.top + plotH);
          ctx.stroke();
        }
      }

      // Y grid — use gridDensity for tick count
      const yGridCount = Math.max(3, gridDensity);
      const yTicks = logScale ? computeLogTicks(Math.max(yMin, 1e-30), yMax, yGridCount) : computeTicks(yMin, yMax, yGridCount);
      for (const tv of yTicks) {
        const cy = snap(dataToCanvasY(tv));
        if (cy >= margin.top && cy <= margin.top + plotH) {
          ctx.beginPath();
          ctx.moveTo(margin.left, cy);
          ctx.lineTo(margin.left + plotW, cy);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    }

    // Axes (pixel-snapped for crisp lines)
    ctx.strokeStyle = isDark ? "#666" : "#999";
    ctx.lineWidth = 1;
    const axisLeft = snap(margin.left);
    const axisBottom = snap(margin.top + plotH);
    ctx.beginPath();
    ctx.moveTo(axisLeft, margin.top);
    ctx.lineTo(axisLeft, axisBottom);
    ctx.lineTo(margin.left + plotW, axisBottom);
    ctx.stroke();

    // X ticks + labels
    const xTicks = computeTicks(xMin, xMax, Math.max(3, Math.floor(plotW / TICK_LABEL_WIDTH_PX)));
    ctx.fillStyle = isDark ? "#aaa" : "#555";
    ctx.font = `11px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tv of xTicks) {
      const cx = snap(dataToCanvasX(tv));
      if (cx >= margin.left && cx <= margin.left + plotW) {
        ctx.beginPath();
        ctx.moveTo(cx, margin.top + plotH);
        ctx.lineTo(cx, margin.top + plotH + AXIS_TICK_PX);
        ctx.stroke();
        ctx.fillText(formatNumber(tv), cx, margin.top + plotH + AXIS_TICK_PX + 2);
      }
    }

    // Y ticks + labels
    const yTicks = logScale ? computeLogTicks(Math.max(yMin, 1e-30), yMax) : computeTicks(yMin, yMax);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tv of yTicks) {
      const cy = snap(dataToCanvasY(tv));
      if (cy >= margin.top && cy <= margin.top + plotH) {
        ctx.beginPath();
        ctx.moveTo(margin.left - AXIS_TICK_PX, cy);
        ctx.lineTo(margin.left, cy);
        ctx.stroke();
        ctx.fillText(formatNumber(tv), margin.left - AXIS_TICK_PX - 2, cy);
      }
    }

    // X axis label
    if (xLabel || xUnit) {
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.font = `12px ${FONT}`;
      ctx.fillStyle = isDark ? "#999" : "#666";
      let lbl = xLabel || "";
      if (xUnit) lbl += lbl ? ` (${xUnit})` : xUnit;
      ctx.fillText(lbl, margin.left + plotW / 2, margin.top + plotH + AXIS_TICK_PX + 18);
    }

    // Y axis label (rotated)
    if (yLabel || yUnit) {
      ctx.save();
      ctx.translate(12, margin.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.font = `12px ${FONT}`;
      ctx.fillStyle = isDark ? "#999" : "#666";
      let lbl = yLabel || "";
      if (yUnit) lbl += lbl ? ` (${yUnit})` : yUnit;
      ctx.fillText(lbl, 0, 0);
      ctx.restore();
    }

    // Clip to plot area for traces
    ctx.save();
    ctx.beginPath();
    ctx.rect(margin.left, margin.top, plotW, plotH);
    ctx.clip();

    // Draw traces (unfocused first, then focused on top)
    const hasFocus = focusedTrace >= 0 && focusedTrace < traces.length;
    const drawTrace = (t: number, alpha: number, lw: number) => {
      const trace = traces[t];
      const color = (traceColors && traceColors[t]) || "#4fc3f7";
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < trace.length; i++) {
        const xv = xVals ? xVals[i] : i;
        const yv = trace[i];
        if (!isFinite(yv) || (logScale && yv <= 0)) continue;
        const cx = dataToCanvasX(xv);
        const cy = dataToCanvasY(yv);
        if (!started) { ctx.moveTo(cx, cy); started = true; }
        else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    };

    const baseLW = lineWidth || 1.5;
    if (hasFocus) {
      for (let t = 0; t < traces.length; t++) {
        if (t !== focusedTrace) drawTrace(t, UNFOCUSED_ALPHA, baseLW);
      }
      drawTrace(focusedTrace, 1.0, baseLW * 1.5);
    } else {
      for (let t = 0; t < traces.length; t++) {
        drawTrace(t, 1.0, baseLW);
      }
    }

    // Peak markers
    if (peakMarkers && peakMarkers.length > 0) {
      const selSet = new Set(selectedPeaks || []);
      for (let pi = 0; pi < peakMarkers.length; pi++) {
        const pk = peakMarkers[pi];
        const traceData = traces[pk.trace_idx];
        if (!traceData) continue;
        const color = (traceColors && traceColors[pk.trace_idx]) || "#4fc3f7";
        const isSelected = selSet.has(pi);

        const cx = dataToCanvasX(pk.x);
        const cy = dataToCanvasY(pk.y);

        // Vertical drop line (dashed)
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = isSelected ? 0.8 : 0.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx, margin.top + plotH);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;

        // Marker shape: triangle-up
        const s = isSelected ? 7 : 5;
        ctx.fillStyle = isSelected ? color : (isDark ? "#1a1a1a" : "#f8f8f8");
        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy - s);
        ctx.lineTo(cx + s, cy + s);
        ctx.lineTo(cx - s, cy + s);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Value label
        ctx.font = `9px ${FONT}`;
        ctx.fillStyle = isDark ? "#ddd" : "#333";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(pk.label || formatNumber(pk.x), cx, cy - s - 3);
      }
    }

    ctx.restore();
  }, [canvasW, canvasH, xMin, xMax, yMin, yMax, yData, xData, nTraces, nPoints, traceColors, lineWidth, logScale, showGrid, gridDensity, xLabel, yLabel, xUnit, yUnit, isDark, dataToCanvasX, dataToCanvasY, plotW, plotH, focusedTrace, margin, peakMarkers, selectedPeaks]);

  // ========================================================================
  // UI overlay (crosshair, legend)
  // ========================================================================
  React.useEffect(() => {
    const uiCanvas = uiCanvasRef.current;
    if (!uiCanvas) return;
    const ctx = uiCanvas.getContext("2d");
    if (!ctx) return;

    uiCanvas.width = canvasW * DPR;
    uiCanvas.height = canvasH * DPR;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (plotW <= 0 || plotH <= 0) return;

    // Crosshair
    if (cursorInfo) {
      const { canvasX, canvasY } = cursorInfo;
      if (canvasX >= margin.left && canvasX <= margin.left + plotW &&
          canvasY >= margin.top && canvasY <= margin.top + plotH) {
        ctx.strokeStyle = isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.25)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);

        const snapCX = snap(canvasX);
        ctx.beginPath();
        ctx.moveTo(snapCX, margin.top);
        ctx.lineTo(snapCX, margin.top + plotH);
        ctx.stroke();

        const snapCY = snap(canvasY);
        ctx.beginPath();
        ctx.moveTo(margin.left, snapCY);
        ctx.lineTo(margin.left + plotW, snapCY);
        ctx.stroke();

        ctx.setLineDash([]);

        // Snap dot
        ctx.fillStyle = cursorInfo.color;
        ctx.beginPath();
        ctx.arc(canvasX, dataToCanvasY(cursorInfo.dataY), 4, 0, Math.PI * 2);
        ctx.fill();

        // Readout box
        const xStr = formatNumber(cursorInfo.dataX);
        const yStr = formatNumber(cursorInfo.dataY);
        let readout = `${xStr}, ${yStr}`;
        if (cursorInfo.label) readout = `${cursorInfo.label}: ${readout}`;

        ctx.font = `10px monospace`;
        const textW = ctx.measureText(readout).width;
        const boxPad = 4;
        const boxW = textW + boxPad * 2;
        const boxH = 16;
        let boxX = canvasX + 10;
        let boxY = canvasY - boxH - 6;
        if (boxX + boxW > margin.left + plotW) boxX = canvasX - boxW - 10;
        if (boxY < margin.top) boxY = canvasY + 10;

        ctx.fillStyle = isDark ? "rgba(30,30,30,0.9)" : "rgba(255,255,255,0.9)";
        ctx.fillRect(boxX, boxY, boxW, boxH);
        ctx.strokeStyle = isDark ? "#555" : "#ccc";
        ctx.lineWidth = 1;
        ctx.strokeRect(boxX, boxY, boxW, boxH);
        ctx.fillStyle = isDark ? "#eee" : "#333";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(readout, boxX + boxPad, boxY + boxH / 2);
      }
    }

    // Legend (with focus state)
    const hasFocusLegend = focusedTrace >= 0 && focusedTrace < nTraces;
    if (showLegend && nTraces >= 1 && traceLabels && traceLabels.length > 0) {
      const entryH = 14;
      const lineLen = 16;
      const gap = 4;
      const legendPad = 6;
      let maxLabelW = 0;
      for (let t = 0; t < nTraces; t++) {
        const lbl = traceLabels[t] || `Data ${t + 1}`;
        ctx.font = hasFocusLegend && t === focusedTrace ? `bold 11px ${FONT}` : `11px ${FONT}`;
        const w = ctx.measureText(lbl).width;
        if (w > maxLabelW) maxLabelW = w;
      }
      const legendW = legendPad * 2 + lineLen + gap + maxLabelW;
      const legendH = legendPad * 2 + nTraces * entryH;
      const lx = margin.left + plotW - legendW - 8;
      const ly = margin.top + 8;

      // Store geometry for click/hover hit detection
      legendGeoRef.current = { lx, ly, w: legendW, h: legendH, entryH, pad: legendPad, n: nTraces };

      ctx.fillStyle = isDark ? "rgba(30,30,30,0.85)" : "rgba(255,255,255,0.85)";
      ctx.fillRect(lx, ly, legendW, legendH);
      ctx.strokeStyle = isDark ? "#555" : "#ccc";
      ctx.lineWidth = 1;
      ctx.strokeRect(lx, ly, legendW, legendH);

      for (let t = 0; t < nTraces; t++) {
        const ey = ly + legendPad + t * entryH + entryH / 2;
        const color = (traceColors && traceColors[t]) || "#4fc3f7";
        const isFocused = hasFocusLegend && t === focusedTrace;
        const dimmed = hasFocusLegend && !isFocused;

        ctx.globalAlpha = dimmed ? UNFOCUSED_ALPHA : 1.0;
        ctx.strokeStyle = color;
        ctx.lineWidth = isFocused ? 3 : 2;
        ctx.beginPath();
        ctx.moveTo(lx + legendPad, ey);
        ctx.lineTo(lx + legendPad + lineLen, ey);
        ctx.stroke();

        ctx.font = isFocused ? `bold 11px ${FONT}` : `11px ${FONT}`;
        ctx.fillStyle = isDark ? "#ddd" : "#333";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(traceLabels[t] || `Data ${t + 1}`, lx + legendPad + lineLen + gap, ey);
        ctx.globalAlpha = 1.0;
      }
    } else {
      legendGeoRef.current = null;
    }

    // Axis drag selection overlay
    if (axisDragCurrent) {
      const { axis, startVal, currentVal } = axisDragCurrent;
      ctx.fillStyle = isDark ? "rgba(0,170,255,0.15)" : "rgba(0,120,255,0.12)";
      ctx.strokeStyle = isDark ? "rgba(0,170,255,0.6)" : "rgba(0,120,255,0.5)";
      ctx.lineWidth = 1;
      if (axis === "x") {
        const x1 = Math.max(margin.left, Math.min(margin.left + plotW, dataToCanvasX(startVal)));
        const x2 = Math.max(margin.left, Math.min(margin.left + plotW, dataToCanvasX(currentVal)));
        const left = Math.min(x1, x2);
        const w = Math.abs(x2 - x1);
        ctx.fillRect(left, margin.top, w, plotH);
        ctx.strokeRect(left, margin.top, w, plotH);
      } else {
        const y1 = Math.max(margin.top, Math.min(margin.top + plotH, dataToCanvasY(startVal)));
        const y2 = Math.max(margin.top, Math.min(margin.top + plotH, dataToCanvasY(currentVal)));
        const top = Math.min(y1, y2);
        const h = Math.abs(y2 - y1);
        ctx.fillRect(margin.left, top, plotW, h);
        ctx.strokeRect(margin.left, top, plotW, h);
      }
    }

    // Peak search region overlay (drag band or hover band)
    // Determine which band to show: drag takes priority over hover
    let bandInfo: PeakSearchRegion | null = peakSearchRegion; // drag band
    if (!bandInfo && peakActive && !lockPeaks && cursorInfo) {
      // Hover band: compute ±search radius from cursor position
      const traces = tracesRef.current;
      const xVals = xValuesRef.current;
      const traceIdx = focusedTrace >= 0 && focusedTrace < traces.length
        ? focusedTrace : cursorInfo.traceIdx;
      const trace = traces[traceIdx];
      if (trace) {
        let nearestIdx = 0;
        if (xVals) {
          let bestDist = Infinity;
          for (let i = 0; i < xVals.length; i++) {
            const dist = Math.abs(xVals[i] - cursorInfo.dataX);
            if (dist < bestDist) { bestDist = dist; nearestIdx = i; }
          }
        } else {
          nearestIdx = Math.round(cursorInfo.dataX);
          nearestIdx = Math.max(0, Math.min(trace.length - 1, nearestIdx));
        }
        const radius = peakSearchRadius ?? 20;
        const lo = Math.max(0, nearestIdx - radius);
        const hi = Math.min(trace.length - 1, nearestIdx + radius);
        const x0 = xVals ? xVals[lo] : lo;
        const x1 = xVals ? xVals[hi] : hi;

        // Find peak within hover band
        let bestVal = -Infinity;
        let bestIdx = lo;
        for (let i = lo; i <= hi; i++) {
          if (isFinite(trace[i]) && trace[i] > bestVal) {
            bestVal = trace[i]; bestIdx = i;
          }
        }
        if (bestVal > -Infinity) {
          const peakX = xVals ? xVals[bestIdx] : bestIdx;
          bandInfo = { x0, x1, peakX, peakY: bestVal, traceIdx };
        }
      }
    }

    if (bandInfo) {
      const sr = bandInfo;
      const sx0 = Math.max(margin.left, dataToCanvasX(sr.x0));
      const sx1 = Math.min(margin.left + plotW, dataToCanvasX(sr.x1));
      const bandLeft = Math.min(sx0, sx1);
      const bandW = Math.abs(sx1 - sx0);

      // Blue translucent band (full plot height)
      ctx.fillStyle = isDark ? "rgba(66,165,245,0.15)" : "rgba(33,150,243,0.12)";
      ctx.fillRect(bandLeft, margin.top, bandW, plotH);
      ctx.strokeStyle = isDark ? "rgba(66,165,245,0.5)" : "rgba(33,150,243,0.4)";
      ctx.lineWidth = 1;
      ctx.strokeRect(bandLeft, margin.top, bandW, plotH);

      // Preview peak marker
      const peakCX = dataToCanvasX(sr.peakX);
      const peakCY = dataToCanvasY(sr.peakY);
      const traceColor = (traceColors && traceColors[sr.traceIdx]) || "#4fc3f7";

      // Dashed vertical line from peak to bottom
      ctx.strokeStyle = traceColor;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.6;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(peakCX, peakCY);
      ctx.lineTo(peakCX, margin.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1.0;

      // Preview triangle marker (dashed outline)
      const s = 6;
      ctx.fillStyle = isDark ? "rgba(66,165,245,0.3)" : "rgba(33,150,243,0.25)";
      ctx.strokeStyle = traceColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(peakCX, peakCY - s);
      ctx.lineTo(peakCX + s, peakCY + s);
      ctx.lineTo(peakCX - s, peakCY + s);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);

      // Value label above preview peak
      ctx.font = `9px ${FONT}`;
      ctx.fillStyle = isDark ? "#ddd" : "#333";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(formatNumber(sr.peakX), peakCX, peakCY - s - 3);
    }

    // Range handles (when X locked)
    if (xLocked) {
      const handleColor = isDark ? "rgba(0,170,255,0.7)" : "rgba(0,120,255,0.6)";
      const handleHoverColor = isDark ? "rgba(0,170,255,1.0)" : "rgba(0,120,255,0.9)";

      for (const side of ["left", "right"] as const) {
        const val = side === "left" ? xMin : xMax;
        const cx = dataToCanvasX(val);
        if (cx < margin.left - 2 || cx > margin.left + plotW + 2) continue;

        const isHovered = hoverRangeHandle === side || rangeDragRef.current?.handle === side;
        const color = isHovered ? handleHoverColor : handleColor;

        // Dashed vertical line (full plot height)
        ctx.strokeStyle = color;
        ctx.lineWidth = isHovered ? 2 : 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(snap(cx), margin.top);
        ctx.lineTo(snap(cx), margin.top + plotH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Triangle handle at bottom (▽)
        const triY = margin.top + plotH + 1;
        const triSize = 6;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(cx - triSize, triY);
        ctx.lineTo(cx + triSize, triY);
        ctx.lineTo(cx, triY + triSize + 2);
        ctx.closePath();
        ctx.fill();
      }
    }

    // Locked axis indicators
    if (xLocked) {
      ctx.fillStyle = isDark ? "rgba(0,170,255,0.5)" : "rgba(0,120,255,0.4)";
      ctx.font = `bold 9px ${FONT}`;
      ctx.textAlign = "right";
      ctx.textBaseline = "top";
      ctx.fillText("X LOCKED", margin.left + plotW, margin.top + plotH + 4);
    }
    if (yLocked) {
      ctx.fillStyle = isDark ? "rgba(0,170,255,0.5)" : "rgba(0,120,255,0.4)";
      ctx.font = `bold 9px ${FONT}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("Y LOCKED", margin.left + 4, margin.top + 2);
    }

    // Y-range handles (when Y locked) — Feature 2
    if (yLocked) {
      const handleColor = isDark ? "rgba(0,170,255,0.7)" : "rgba(0,120,255,0.6)";
      const handleHoverColor = isDark ? "rgba(0,170,255,1.0)" : "rgba(0,120,255,0.9)";

      for (const side of ["top", "bottom"] as const) {
        const val = side === "top" ? yMax : yMin;
        const cy = dataToCanvasY(val);
        if (cy < margin.top - 2 || cy > margin.top + plotH + 2) continue;

        const isHovered = hoverRangeHandleY === side || rangeDragYRef.current?.handle === side;
        const color = isHovered ? handleHoverColor : handleColor;

        // Dashed horizontal line
        ctx.strokeStyle = color;
        ctx.lineWidth = isHovered ? 2 : 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(margin.left, snap(cy));
        ctx.lineTo(margin.left + plotW, snap(cy));
        ctx.stroke();
        ctx.setLineDash([]);

        // Triangle handle on left edge (◁)
        const triX = margin.left - 1;
        const triSize = 6;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(triX, cy - triSize);
        ctx.lineTo(triX, cy + triSize);
        ctx.lineTo(triX - triSize - 2, cy);
        ctx.closePath();
        ctx.fill();
      }
    }

    // FWHM arrows — Feature 5
    if (peakFwhm && peakFwhm.length > 0) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(margin.left, margin.top, plotW, plotH);
      ctx.clip();

      for (const f of peakFwhm) {
        if (f.fwhm == null || f.center == null || f.amplitude == null || f.offset == null) continue;
        const halfMax = f.offset + f.amplitude / 2;
        const cy = dataToCanvasY(halfMax);
        const leftX = dataToCanvasX(f.center - f.fwhm / 2);
        const rightX = dataToCanvasX(f.center + f.fwhm / 2);

        // Find trace color
        const pkIdx = f.peak_idx;
        const pk = peakMarkers && pkIdx >= 0 && pkIdx < peakMarkers.length ? peakMarkers[pkIdx] : null;
        const trColor = pk && traceColors && traceColors[pk.trace_idx] ? traceColors[pk.trace_idx] : "#4fc3f7";

        // Horizontal double-arrow
        ctx.strokeStyle = trColor;
        ctx.fillStyle = trColor;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.8;

        ctx.beginPath();
        ctx.moveTo(leftX, cy);
        ctx.lineTo(rightX, cy);
        ctx.stroke();

        // Arrowheads
        const arrowSize = 4;
        ctx.beginPath();
        ctx.moveTo(leftX, cy);
        ctx.lineTo(leftX + arrowSize, cy - arrowSize);
        ctx.lineTo(leftX + arrowSize, cy + arrowSize);
        ctx.closePath();
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(rightX, cy);
        ctx.lineTo(rightX - arrowSize, cy - arrowSize);
        ctx.lineTo(rightX - arrowSize, cy + arrowSize);
        ctx.closePath();
        ctx.fill();

        // Label
        ctx.font = `9px ${FONT}`;
        ctx.fillStyle = isDark ? "#ddd" : "#333";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        const midX = (leftX + rightX) / 2;
        ctx.fillText(`FWHM: ${formatNumber(f.fwhm)}`, midX, cy - 3);
        ctx.globalAlpha = 1.0;
      }
      ctx.restore();
    }
  }, [canvasW, canvasH, cursorInfo, showLegend, nTraces, traceLabels, traceColors, isDark, dataToCanvasX, dataToCanvasY, plotW, plotH, focusedTrace, margin, axisDragCurrent, xLocked, yLocked, peakSearchRegion, peakActive, lockPeaks, peakSearchRadius, hoverRangeHandle, xMin, xMax, hoverRangeHandleY, yMin, yMax, logScale, lineWidth, peakFwhm, peakMarkers]);

  // ========================================================================
  // Helper: find nearest data point to cursor
  // ========================================================================
  const findNearestPoint = React.useCallback(
    (mx: number, my: number) => {
      const traces = tracesRef.current;
      const xVals = xValuesRef.current;
      if (traces.length === 0 || nPoints < 1) return null;

      const cursorDataX = canvasToDataX(mx);

      let nearestIdx = 0;
      if (xVals) {
        let bestDist = Infinity;
        for (let i = 0; i < xVals.length; i++) {
          const dist = Math.abs(xVals[i] - cursorDataX);
          if (dist < bestDist) { bestDist = dist; nearestIdx = i; }
        }
      } else {
        nearestIdx = Math.round(cursorDataX);
        nearestIdx = Math.max(0, Math.min(nPoints - 1, nearestIdx));
      }

      const actualX = xVals ? xVals[nearestIdx] : nearestIdx;
      const snapCX = dataToCanvasX(actualX);
      let bestTraceIdx = 0;
      let bestDist = Infinity;
      for (let t = 0; t < traces.length; t++) {
        const val = traces[t][nearestIdx];
        if (!isFinite(val)) continue;
        const traceCY = dataToCanvasY(val);
        const dist = Math.abs(traceCY - my);
        if (dist < bestDist) {
          bestDist = dist;
          bestTraceIdx = t;
        }
      }

      const snapVal = traces[bestTraceIdx][nearestIdx];
      return {
        canvasX: snapCX,
        canvasY: dataToCanvasY(snapVal),
        dataX: actualX,
        dataY: snapVal,
        traceIdx: bestTraceIdx,
        nearestIdx,
        label: (traceLabels && traceLabels[bestTraceIdx]) || "",
        color: (traceColors && traceColors[bestTraceIdx]) || "#4fc3f7",
      };
    },
    [nPoints, canvasToDataX, dataToCanvasX, dataToCanvasY, traceLabels, traceColors],
  );

  // ========================================================================
  // Helper: hit-test legend entries — returns trace index or null
  // ========================================================================
  const hitTestLegend = React.useCallback(
    (mx: number, my: number): number | null => {
      const geo = legendGeoRef.current;
      if (!geo) return null;
      if (mx < geo.lx || mx > geo.lx + geo.w || my < geo.ly || my > geo.ly + geo.h) return null;
      const idx = Math.floor((my - geo.ly - geo.pad) / geo.entryH);
      if (idx < 0 || idx >= geo.n) return null;
      return idx;
    },
    [],
  );

  // ========================================================================
  // Helper: find if click is near an existing peak marker
  // ========================================================================
  const findNearestPeakMarker = React.useCallback(
    (mx: number, my: number): number | null => {
      if (!peakMarkers || peakMarkers.length === 0) return null;
      const threshold = 12; // pixels
      let bestIdx: number | null = null;
      let bestDist = Infinity;
      for (let i = 0; i < peakMarkers.length; i++) {
        const pk = peakMarkers[i];
        const cx = dataToCanvasX(pk.x);
        const cy = dataToCanvasY(pk.y);
        const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);
        if (dist < threshold && dist < bestDist) {
          bestDist = dist;
          bestIdx = i;
        }
      }
      return bestIdx;
    },
    [peakMarkers, dataToCanvasX, dataToCanvasY],
  );

  // ========================================================================
  // Helper: hit-test range handles — returns "left" | "right" | null
  // ========================================================================
  const hitTestRangeHandle = React.useCallback(
    (mx: number, my: number): "left" | "right" | null => {
      if (!xLocked) return null;
      if (my < margin.top || my > margin.top + plotH + 12) return null;
      const leftX = dataToCanvasX(xMin);
      const rightX = dataToCanvasX(xMax);
      const leftDist = Math.abs(mx - leftX);
      const rightDist = Math.abs(mx - rightX);
      const threshold = 8;
      if (leftDist < threshold && leftDist <= rightDist) return "left";
      if (rightDist < threshold) return "right";
      return null;
    },
    [xLocked, xMin, xMax, dataToCanvasX, margin.top, plotH],
  );

  const hitTestRangeHandleY = React.useCallback(
    (mx: number, my: number): "top" | "bottom" | null => {
      if (!yLocked) return null;
      if (mx < margin.left - 12 || mx > margin.left + plotW) return null;
      const topY = dataToCanvasY(yMax);
      const bottomY = dataToCanvasY(yMin);
      const topDist = Math.abs(my - topY);
      const bottomDist = Math.abs(my - bottomY);
      const threshold = 8;
      if (topDist < threshold && topDist <= bottomDist) return "top";
      if (bottomDist < threshold) return "bottom";
      return null;
    },
    [yLocked, yMin, yMax, dataToCanvasY, margin.left, plotW],
  );

  // ========================================================================
  // Mouse handlers
  // ========================================================================
  const handleWheel = React.useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      if (mouseX < margin.left || mouseX > margin.left + plotW ||
          mouseY < margin.top || mouseY > margin.top + plotH) return;

      const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
      const dx = canvasToDataX(mouseX);
      const dy = canvasToDataY(mouseY);

      if (!xLockedRef.current) {
        setXMin((prev) => dx - (dx - prev) * factor);
        setXMax((prev) => dx + (prev - dx) * factor);
      }
      if (!yLockedRef.current) {
        setYMin((prev) => dy - (dy - prev) * factor);
        setYMax((prev) => dy + (prev - dy) * factor);
      }
    },
    [plotW, plotH, canvasToDataX, canvasToDataY, margin.left, margin.top],
  );

  // Wheel prevention (passive: false)
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  // Sync range inputs when xMin/xMax change (unless user is typing)
  React.useEffect(() => {
    if (!rangeInputActive) {
      setRangeInputMin(xMin.toPrecision(6));
      setRangeInputMax(xMax.toPrecision(6));
    }
  }, [xMin, xMax, rangeInputActive]);

  // Sync Y range inputs
  React.useEffect(() => {
    if (!rangeInputYActive) {
      setRangeInputYMin(yMin.toPrecision(6));
      setRangeInputYMax(yMax.toPrecision(6));
    }
  }, [yMin, yMax, rangeInputYActive]);

  // Sync Python x_range/y_range → JS view state
  React.useEffect(() => {
    if (modelXRange && modelXRange.length === 2) {
      setXMin(modelXRange[0]);
      setXMax(modelXRange[1]);
      setXLocked(true);
    } else if (modelXRange && modelXRange.length === 0 && xLocked) {
      setXLocked(false);
    }
  }, [modelXRange]); // eslint-disable-line react-hooks/exhaustive-deps

  React.useEffect(() => {
    if (modelYRange && modelYRange.length === 2) {
      setYMin(modelYRange[0]);
      setYMax(modelYRange[1]);
      setYLocked(true);
    } else if (modelYRange && modelYRange.length === 0 && yLocked) {
      setYLocked(false);
    }
  }, [modelYRange]); // eslint-disable-line react-hooks/exhaustive-deps

  // Window-level resize and range drag handling
  React.useEffect(() => {
    const handleWindowMove = (e: MouseEvent) => {
      // Resize drag
      if (resizeDragRef.current?.active) {
        const rd = resizeDragRef.current;
        const newW = Math.max(200, rd.startW + (e.clientX - rd.startX));
        const newH = Math.max(100, rd.startH + (e.clientY - rd.startY));
        setCanvasW(newW);
        setCanvasH(newH);
      }
      // Range handle drag (window-level for when mouse leaves canvas)
      if (rangeDragRef.current) {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const clampedMx = Math.max(margin.left, Math.min(margin.left + plotW, mx));
        const newVal = canvasToDataX(clampedMx);
        const dataRange = xDataRangeRef.current;
        const clamped = Math.max(dataRange.min, Math.min(dataRange.max, newVal));
        if (rangeDragRef.current.handle === "left") {
          setXMin((prevMin) => {
            return Math.min(clamped, xMax - (xMax - prevMin) * 0.001);
          });
          setModelXRange([clamped, xMax]);
        } else {
          setXMax((prevMax) => {
            return Math.max(clamped, xMin + (prevMax - xMin) * 0.001);
          });
          setModelXRange([xMin, clamped]);
        }
      }
      // Y-range handle drag (window-level)
      if (rangeDragYRef.current) {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const my = e.clientY - rect.top;
        const clampedMy = Math.max(margin.top, Math.min(margin.top + plotH, my));
        const newVal = canvasToDataY(clampedMy);
        const dataRange = yDataRangeRef.current;
        const clamped = Math.max(dataRange.min, Math.min(dataRange.max, newVal));
        if (rangeDragYRef.current.handle === "top") {
          setYMax((prevMax) => Math.max(clamped, yMin + (prevMax - yMin) * 0.001));
          setModelYRange([yMin, clamped]);
        } else {
          setYMin((prevMin) => Math.min(clamped, yMax - (yMax - prevMin) * 0.001));
          setModelYRange([clamped, yMax]);
        }
      }
    };
    const handleWindowUp = () => {
      if (resizeDragRef.current?.active) resizeDragRef.current = null;
      if (rangeDragRef.current) rangeDragRef.current = null;
      if (rangeDragYRef.current) rangeDragYRef.current = null;
    };
    window.addEventListener("mousemove", handleWindowMove);
    window.addEventListener("mouseup", handleWindowUp);
    return () => {
      window.removeEventListener("mousemove", handleWindowMove);
      window.removeEventListener("mouseup", handleWindowUp);
    };
  }, [margin.left, margin.top, plotW, plotH, canvasToDataX, canvasToDataY, xMin, xMax, yMin, yMax, setModelXRange, setModelYRange]);

  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      // X-axis area drag (below plot, within X extent)
      if (my > margin.top + plotH && my < canvasH &&
          mx >= margin.left && mx <= margin.left + plotW) {
        axisDragRef.current = { axis: "x", startPx: mx, startVal: canvasToDataX(mx) };
        setAxisDragCurrent({ axis: "x", startVal: canvasToDataX(mx), currentVal: canvasToDataX(mx) });
        return;
      }

      // Y-axis area drag (left of plot, within Y extent)
      if (mx < margin.left && mx >= 0 &&
          my >= margin.top && my <= margin.top + plotH) {
        axisDragRef.current = { axis: "y", startPx: my, startVal: canvasToDataY(my) };
        setAxisDragCurrent({ axis: "y", startVal: canvasToDataY(my), currentVal: canvasToDataY(my) });
        return;
      }

      // Range handle drag (intercept before plot area check)
      if (xLocked) {
        const handle = hitTestRangeHandle(mx, my);
        if (handle) {
          rangeDragRef.current = { handle, startPx: mx };
          return;
        }
      }
      if (yLocked) {
        const handleY = hitTestRangeHandleY(mx, my);
        if (handleY) {
          rangeDragYRef.current = { handle: handleY, startPx: my };
          return;
        }
      }

      if (mx < margin.left || mx > margin.left + plotW ||
          my < margin.top || my > margin.top + plotH) return;

      // Peak drag start — intercept before pan
      if (peakActive && !lockPeaks) {
        // Check if clicking an existing peak marker (let that fall through to mouseUp click handler)
        const nearPeak = findNearestPeakMarker(mx, my);
        if (nearPeak === null) {
          // Determine which trace to search
          const point = findNearestPoint(mx, my);
          const traceIdx = focusedTrace >= 0 && focusedTrace < tracesRef.current.length
            ? focusedTrace
            : (point?.traceIdx ?? 0);
          peakDragRef.current = {
            active: true,
            startPx: mx,
            startDataX: canvasToDataX(mx),
            traceIdx,
          };
          // Don't start pan drag — return early
          // Still set dragRef for click detection (wasDrag threshold)
          dragRef.current = {
            active: true,
            wasDrag: false,
            startX: e.clientX,
            startY: e.clientY,
            startXMin: xMin,
            startXMax: xMax,
            startYMin: yMin,
            startYMax: yMax,
          };
          return;
        }
      }

      // Pan start
      dragRef.current = {
        active: true,
        wasDrag: false,
        startX: e.clientX,
        startY: e.clientY,
        startXMin: xMin,
        startXMax: xMax,
        startYMin: yMin,
        startYMax: yMax,
      };
    },
    [plotW, plotH, canvasH, xMin, xMax, yMin, yMax, margin.left, margin.top, canvasToDataX, canvasToDataY, peakActive, lockPeaks, focusedTrace, findNearestPeakMarker, findNearestPoint, xLocked, yLocked, hitTestRangeHandle, hitTestRangeHandleY],
  );

  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      // Range handle drag (X)
      if (rangeDragRef.current) {
        const rd = rangeDragRef.current;
        const clampedMx = Math.max(margin.left, Math.min(margin.left + plotW, mx));
        const newVal = canvasToDataX(clampedMx);
        const dataRange = xDataRangeRef.current;
        const clamped = Math.max(dataRange.min, Math.min(dataRange.max, newVal));
        if (rd.handle === "left") {
          const newMin = Math.min(clamped, xMax - (xMax - xMin) * 0.001);
          setXMin(newMin);
          setModelXRange([newMin, xMax]);
        } else {
          const newMax = Math.max(clamped, xMin + (xMax - xMin) * 0.001);
          setXMax(newMax);
          setModelXRange([xMin, newMax]);
        }
        return;
      }

      // Range handle drag (Y)
      if (rangeDragYRef.current) {
        const rd = rangeDragYRef.current;
        const clampedMy = Math.max(margin.top, Math.min(margin.top + plotH, my));
        const newVal = canvasToDataY(clampedMy);
        const dataRange = yDataRangeRef.current;
        const clamped = Math.max(dataRange.min, Math.min(dataRange.max, newVal));
        if (rd.handle === "top") {
          const newMax = Math.max(clamped, yMin + (yMax - yMin) * 0.001);
          setYMax(newMax);
          setModelYRange([yMin, newMax]);
        } else {
          const newMin = Math.min(clamped, yMax - (yMax - yMin) * 0.001);
          setYMin(newMin);
          setModelYRange([newMin, yMax]);
        }
        return;
      }

      // Range handle hover detection
      const noActive = !dragRef.current?.active && !peakDragRef.current?.active && !axisDragRef.current;
      if (xLocked && noActive) {
        const handle = hitTestRangeHandle(mx, my);
        setHoverRangeHandle(handle);
      } else {
        if (hoverRangeHandle) setHoverRangeHandle(null);
      }
      if (yLocked && noActive) {
        const handleY = hitTestRangeHandleY(mx, my);
        setHoverRangeHandleY(handleY);
      } else {
        if (hoverRangeHandleY) setHoverRangeHandleY(null);
      }

      // Axis range drag
      if (axisDragRef.current) {
        const ad = axisDragRef.current;
        if (ad.axis === "x") {
          setAxisDragCurrent({ axis: "x", startVal: ad.startVal, currentVal: canvasToDataX(Math.max(margin.left, Math.min(margin.left + plotW, mx))) });
        } else {
          setAxisDragCurrent({ axis: "y", startVal: ad.startVal, currentVal: canvasToDataY(Math.max(margin.top, Math.min(margin.top + plotH, my))) });
        }
        return;
      }

      // Peak drag-to-search
      if (peakDragRef.current?.active) {
        const pd = peakDragRef.current;
        // Mark as drag if moved enough
        if (dragRef.current && (Math.abs(mx - pd.startPx) > 3)) {
          dragRef.current.wasDrag = true;
        }
        const currentDataX = canvasToDataX(Math.max(margin.left, Math.min(margin.left + plotW, mx)));
        const x0 = Math.min(pd.startDataX, currentDataX);
        const x1 = Math.max(pd.startDataX, currentDataX);

        // Find peak within search region
        const traces = tracesRef.current;
        const xVals = xValuesRef.current;
        const trace = traces[pd.traceIdx];
        if (trace) {
          let bestVal = -Infinity;
          let bestIdx = -1;
          for (let i = 0; i < trace.length; i++) {
            const xv = xVals ? xVals[i] : i;
            if (xv >= x0 && xv <= x1 && isFinite(trace[i]) && trace[i] > bestVal) {
              bestVal = trace[i];
              bestIdx = i;
            }
          }
          if (bestIdx >= 0) {
            const peakX = xVals ? xVals[bestIdx] : bestIdx;
            setPeakSearchRegion({ x0, x1, peakX, peakY: bestVal, traceIdx: pd.traceIdx });
          } else {
            setPeakSearchRegion(null);
          }
        }
        setCursorInfo(null);
        return;
      }

      // Pan drag
      if (dragRef.current?.active) {
        const d = dragRef.current;
        const dxPx = e.clientX - d.startX;
        const dyPx = e.clientY - d.startY;
        if (Math.abs(dxPx) > 3 || Math.abs(dyPx) > 3) d.wasDrag = true;
        const xRange = d.startXMax - d.startXMin;
        const yRange = d.startYMax - d.startYMin;
        const dxData = -(dxPx / plotW) * xRange;
        const dyData = (dyPx / plotH) * yRange;
        if (!xLockedRef.current) {
          setXMin(d.startXMin + dxData);
          setXMax(d.startXMax + dxData);
        }
        if (!yLockedRef.current) {
          setYMin(d.startYMin + dyData);
          setYMax(d.startYMax + dyData);
        }
        setCursorInfo(null);
        return;
      }

      // Cursor tracking
      if (mx < margin.left || mx > margin.left + plotW ||
          my < margin.top || my > margin.top + plotH) {
        setCursorInfo(null);
        return;
      }

      // Check if hovering over legend or peak marker
      const legendHit = hitTestLegend(mx, my);
      hoverLegendRef.current = legendHit !== null;
      setHoverPeakIdx(legendHit !== null ? null : findNearestPeakMarker(mx, my));

      // Don't show crosshair/snap when hovering over legend
      if (legendHit !== null) {
        setCursorInfo(null);
        return;
      }

      const point = findNearestPoint(mx, my);
      if (point) {
        setCursorInfo({
          canvasX: point.canvasX,
          canvasY: point.canvasY,
          dataX: point.dataX,
          dataY: point.dataY,
          traceIdx: point.traceIdx,
          label: point.label,
          color: point.color,
        });
      }
    },
    [plotW, plotH, findNearestPoint, findNearestPeakMarker, hitTestLegend, margin.left, margin.top, canvasToDataX, canvasToDataY, xLocked, yLocked, xMin, xMax, yMin, yMax, hitTestRangeHandle, hitTestRangeHandleY, hoverRangeHandle, hoverRangeHandleY, setModelXRange, setModelYRange],
  );

  const handleMouseUp = React.useCallback((e: React.MouseEvent) => {
    // Finalize range handle drag (X or Y)
    if (rangeDragRef.current) {
      rangeDragRef.current = null;
      return;
    }
    if (rangeDragYRef.current) {
      rangeDragYRef.current = null;
      return;
    }

    // Finalize axis range drag
    if (axisDragRef.current && axisDragCurrent) {
      const { startVal, currentVal, axis } = axisDragCurrent;
      const lo = Math.min(startVal, currentVal);
      const hi = Math.max(startVal, currentVal);
      // Only lock if dragged a meaningful distance (>1% of current range)
      const range = axis === "x" ? xMax - xMin : yMax - yMin;
      if (hi - lo > range * 0.01) {
        if (axis === "x") {
          setXMin(lo); setXMax(hi); setXLocked(true); setModelXRange([lo, hi]);
        } else {
          setYMin(lo); setYMax(hi); setYLocked(true); setModelYRange([lo, hi]);
        }
      }
      axisDragRef.current = null;
      setAxisDragCurrent(null);
      return;
    }

    // Peak drag finalization
    if (peakDragRef.current?.active) {
      const wasDrag = dragRef.current?.wasDrag ?? false;
      if (wasDrag && peakSearchRegion) {
        // Place peak from drag search
        const marker: PeakMarker = {
          x: peakSearchRegion.peakX,
          y: peakSearchRegion.peakY,
          trace_idx: peakSearchRegion.traceIdx,
          label: formatNumber(peakSearchRegion.peakX),
          type: "peak",
        };
        setPeakMarkers([...(peakMarkers || []), marker]);
      }
      peakDragRef.current = null;
      setPeakSearchRegion(null);
      dragRef.current = null;
      return;
    }

    // Click handlers (not drag)
    if (dragRef.current?.active && !dragRef.current.wasDrag) {
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Check if clicking on a legend entry
        const legendTraceIdx = hitTestLegend(mx, my);
        if (legendTraceIdx !== null) {
          setFocusedTrace(focusedTrace === legendTraceIdx ? -1 : legendTraceIdx);
          dragRef.current = null;
          return;
        }

        // Check if clicking near an existing peak marker (for selection)
        const nearPeakIdx = findNearestPeakMarker(mx, my);
        if (nearPeakIdx !== null) {
          // Toggle peak selection
          const current = selectedPeaks || [];
          if (e.shiftKey) {
            // Shift+click: multi-select toggle
            if (current.includes(nearPeakIdx)) {
              setSelectedPeaks(current.filter(i => i !== nearPeakIdx));
            } else {
              setSelectedPeaks([...current, nearPeakIdx]);
            }
          } else {
            // Click: toggle single selection
            if (current.length === 1 && current[0] === nearPeakIdx) {
              setSelectedPeaks([]);
            } else {
              setSelectedPeaks([nearPeakIdx]);
            }
          }
          dragRef.current = null;
          return;
        }

        if (peakActive && !lockPeaks && cursorInfo) {
          // Click-to-place: find peak within ±search radius
          const traces = tracesRef.current;
          const xVals = xValuesRef.current;
          const traceIdx = focusedTrace >= 0 && focusedTrace < traces.length
            ? focusedTrace : cursorInfo.traceIdx;
          const trace = traces[traceIdx];
          if (trace) {
            let nearestIdx = 0;
            if (xVals) {
              let bestDist = Infinity;
              for (let i = 0; i < xVals.length; i++) {
                const dist = Math.abs(xVals[i] - cursorInfo.dataX);
                if (dist < bestDist) { bestDist = dist; nearestIdx = i; }
              }
            } else {
              nearestIdx = Math.round(cursorInfo.dataX);
              nearestIdx = Math.max(0, Math.min(trace.length - 1, nearestIdx));
            }

            const radius = peakSearchRadius ?? 20;
            const lo = Math.max(0, nearestIdx - radius);
            const hi = Math.min(trace.length, nearestIdx + radius + 1);
            let bestIdx = lo;
            let bestVal = trace[lo];
            for (let i = lo + 1; i < hi; i++) {
              if (trace[i] > bestVal) {
                bestVal = trace[i]; bestIdx = i;
              }
            }

            const peakX = xVals ? xVals[bestIdx] : bestIdx;
            const marker: PeakMarker = {
              x: peakX,
              y: bestVal,
              trace_idx: traceIdx,
              label: formatNumber(peakX),
              type: "peak",
            };
            setPeakMarkers([...(peakMarkers || []), marker]);
          }
        } else if (!peakActive) {
          // Deselect all peaks when clicking empty area
          if (selectedPeaks && selectedPeaks.length > 0) {
            setSelectedPeaks([]);
          }
          // Toggle focus
          if (cursorInfo) {
            const newFocus = cursorInfo.traceIdx;
            setFocusedTrace(focusedTrace === newFocus ? -1 : newFocus);
          }
        }
      }
    }
    dragRef.current = null;
  }, [cursorInfo, focusedTrace, setFocusedTrace, peakActive, peakMarkers, setPeakMarkers, selectedPeaks, setSelectedPeaks, findNearestPeakMarker, hitTestLegend, lockPeaks, axisDragCurrent, xMin, xMax, yMin, yMax, setModelXRange, setModelYRange, peakSearchRegion, peakSearchRadius]);

  const handleMouseLeave = React.useCallback(() => {
    dragRef.current = null;
    peakDragRef.current = null;
    rangeDragRef.current = null;
    rangeDragYRef.current = null;
    setPeakSearchRegion(null);
    axisDragRef.current = null;
    setAxisDragCurrent(null);
    setCursorInfo(null);
    setHoverPeakIdx(null);
    setHoverRangeHandle(null);
    setHoverRangeHandleY(null);
    hoverLegendRef.current = false;
  }, []);

  const handleDoubleClick = React.useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) { resetView(); return; }
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Double-click on X axis area → unlock X
    if (my > margin.top + plotH && mx >= margin.left && mx <= margin.left + plotW && xLocked) {
      setXLocked(false);
      setModelXRange([]);
      setXMin(xDataRangeRef.current.min);
      setXMax(xDataRangeRef.current.max);
      return;
    }
    // Double-click on Y axis area → unlock Y
    if (mx < margin.left && my >= margin.top && my <= margin.top + plotH && yLocked) {
      setYLocked(false);
      setModelYRange([]);
      setYMin(yDataRangeRef.current.min);
      setYMax(yDataRangeRef.current.max);
      return;
    }
    resetView();
  }, [resetView, plotW, plotH, margin.left, margin.top, xLocked, yLocked, setModelXRange, setModelYRange]);

  // ========================================================================
  // Keyboard
  // ========================================================================
  const handleKeyDown = React.useCallback(
    (e: React.KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;

      switch (e.key) {
        case "r":
        case "R":
          if (!lockDisplay) {
            e.preventDefault();
            resetView();
          }
          break;
        case "Escape":
          e.preventDefault();
          setFocusedTrace(-1);
          if (selectedPeaks && selectedPeaks.length > 0) {
            setSelectedPeaks([]);
          }
          break;
        case "p":
        case "P":
          if (!lockPeaks) {
            e.preventDefault();
            setPeakActive(!peakActive);
          }
          break;
        case "c":
        case "C":
          if (!lockPeaks) {
            e.preventDefault();
            setPeakMarkers([]);
            setSelectedPeaks([]);
          }
          break;
        case "Delete":
        case "Backspace":
          if (!lockPeaks && peakMarkers && peakMarkers.length > 0) {
            e.preventDefault();
            if (selectedPeaks && selectedPeaks.length > 0) {
              // Delete selected peaks
              const selSet = new Set(selectedPeaks);
              const newMarkers = peakMarkers.filter((_: PeakMarker, i: number) => !selSet.has(i));
              setPeakMarkers(newMarkers);
              setSelectedPeaks([]);
            } else {
              // No selection — remove last peak
              const newMarkers = [...peakMarkers];
              newMarkers.pop();
              setPeakMarkers(newMarkers);
            }
          }
          break;
      }
    },
    [resetView, setFocusedTrace, peakMarkers, setPeakMarkers, peakActive, setPeakActive, selectedPeaks, setSelectedPeaks, lockDisplay, lockPeaks],
  );

  // ========================================================================
  // Export
  // ========================================================================
  const handleExportPNG = React.useCallback(() => {
    setExportAnchor(null);
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, `${title || "show1d"}.png`);
    });
  }, [title]);

  const handleExportFigure = React.useCallback((format: "pdf" | "png" = "pdf") => {
    setExportAnchor(null);
    const traces = tracesRef.current;
    const xVals = xValuesRef.current;
    if (traces.length === 0) return;

    const scale = 4;
    const figW = canvasW * scale;
    const figH = canvasH * scale;
    const offscreen = document.createElement("canvas");
    offscreen.width = figW;
    offscreen.height = figH;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return;

    ctx.scale(scale, scale);

    // White background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvasW, canvasH);

    // Grid
    if (showGrid) {
      ctx.strokeStyle = "rgba(0,0,0,0.08)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      const xGridCount = Math.max(3, gridDensity);
      const xTicks = computeTicks(xMin, xMax, xGridCount);
      for (const tv of xTicks) {
        const cx = dataToCanvasX(tv);
        if (cx >= margin.left && cx <= margin.left + plotW) {
          ctx.beginPath();
          ctx.moveTo(cx, margin.top);
          ctx.lineTo(cx, margin.top + plotH);
          ctx.stroke();
        }
      }
      const yGridCount = Math.max(3, gridDensity);
      const yTicks = logScale ? computeLogTicks(Math.max(yMin, 1e-30), yMax, yGridCount) : computeTicks(yMin, yMax, yGridCount);
      for (const tv of yTicks) {
        const cy = dataToCanvasY(tv);
        if (cy >= margin.top && cy <= margin.top + plotH) {
          ctx.beginPath();
          ctx.moveTo(margin.left, cy);
          ctx.lineTo(margin.left + plotW, cy);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    }

    // Axes
    ctx.strokeStyle = "#999";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + plotH);
    ctx.lineTo(margin.left + plotW, margin.top + plotH);
    ctx.stroke();

    // X ticks + labels
    const figXTicks = computeTicks(xMin, xMax, Math.max(3, Math.floor(plotW / TICK_LABEL_WIDTH_PX)));
    ctx.fillStyle = "#555";
    ctx.font = `10px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tv of figXTicks) {
      const cx = dataToCanvasX(tv);
      if (cx >= margin.left && cx <= margin.left + plotW) {
        ctx.beginPath();
        ctx.moveTo(cx, margin.top + plotH);
        ctx.lineTo(cx, margin.top + plotH + AXIS_TICK_PX);
        ctx.stroke();
        ctx.fillText(formatNumber(tv), cx, margin.top + plotH + AXIS_TICK_PX + 2);
      }
    }

    // Y ticks + labels
    const figYTicks = logScale ? computeLogTicks(Math.max(yMin, 1e-30), yMax) : computeTicks(yMin, yMax);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tv of figYTicks) {
      const cy = dataToCanvasY(tv);
      if (cy >= margin.top && cy <= margin.top + plotH) {
        ctx.beginPath();
        ctx.moveTo(margin.left - AXIS_TICK_PX, cy);
        ctx.lineTo(margin.left, cy);
        ctx.stroke();
        ctx.fillText(formatNumber(tv), margin.left - AXIS_TICK_PX - 2, cy);
      }
    }

    // X axis label
    if (xLabel || xUnit) {
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.font = `11px ${FONT}`;
      ctx.fillStyle = "#666";
      let lbl = xLabel || "";
      if (xUnit) lbl += lbl ? ` (${xUnit})` : xUnit;
      ctx.fillText(lbl, margin.left + plotW / 2, margin.top + plotH + AXIS_TICK_PX + 18);
    }

    // Y axis label
    if (yLabel || yUnit) {
      ctx.save();
      ctx.translate(12, margin.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.font = `11px ${FONT}`;
      ctx.fillStyle = "#666";
      let lbl = yLabel || "";
      if (yUnit) lbl += lbl ? ` (${yUnit})` : yUnit;
      ctx.fillText(lbl, 0, 0);
      ctx.restore();
    }

    // Title
    if (title) {
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.font = `bold 13px ${FONT}`;
      ctx.fillStyle = "#333";
      ctx.fillText(title, canvasW / 2, 2);
    }

    // Traces
    ctx.save();
    ctx.beginPath();
    ctx.rect(margin.left, margin.top, plotW, plotH);
    ctx.clip();

    for (let t = 0; t < traces.length; t++) {
      const trace = traces[t];
      const color = (traceColors && traceColors[t]) || "#4fc3f7";
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth || 1.5;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < trace.length; i++) {
        const xv = xVals ? xVals[i] : i;
        const yv = trace[i];
        if (!isFinite(yv) || (logScale && yv <= 0)) continue;
        const cx = dataToCanvasX(xv);
        const cy = dataToCanvasY(yv);
        if (!started) { ctx.moveTo(cx, cy); started = true; }
        else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
    }

    // Peak markers on export
    if (peakMarkers && peakMarkers.length > 0) {
      for (const pk of peakMarkers) {
        const color = (traceColors && traceColors[pk.trace_idx]) || "#4fc3f7";
        const cx = dataToCanvasX(pk.x);
        const cy = dataToCanvasY(pk.y);

        const s = 5;
        ctx.fillStyle = color;
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy - s);
        ctx.lineTo(cx + s, cy + s);
        ctx.lineTo(cx - s, cy + s);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        ctx.font = `9px ${FONT}`;
        ctx.fillStyle = "#333";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(pk.label || formatNumber(pk.x), cx, cy - s - 3);
      }
    }

    ctx.restore();

    // Legend on export figure
    if (showLegend && traces.length >= 1 && traceLabels && traceLabels.length > 0) {
      ctx.font = `10px ${FONT}`;
      const entryH = 14;
      const lineLen = 16;
      const gap = 4;
      const legendPad = 6;
      let maxLabelW = 0;
      for (let t = 0; t < traces.length; t++) {
        const lbl = traceLabels[t] || `Data ${t + 1}`;
        const w = ctx.measureText(lbl).width;
        if (w > maxLabelW) maxLabelW = w;
      }
      const legendW = legendPad * 2 + lineLen + gap + maxLabelW;
      const legendH = legendPad * 2 + traces.length * entryH;
      const lx = margin.left + plotW - legendW - 8;
      const ly = margin.top + 8;

      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.fillRect(lx, ly, legendW, legendH);
      ctx.strokeStyle = "#ccc";
      ctx.lineWidth = 1;
      ctx.strokeRect(lx, ly, legendW, legendH);

      for (let t = 0; t < traces.length; t++) {
        const ey = ly + legendPad + t * entryH + entryH / 2;
        const color = (traceColors && traceColors[t]) || "#4fc3f7";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(lx + legendPad, ey);
        ctx.lineTo(lx + legendPad + lineLen, ey);
        ctx.stroke();
        ctx.fillStyle = "#333";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(traceLabels[t] || `Data ${t + 1}`, lx + legendPad + lineLen + gap, ey);
      }
    }

    if (format === "pdf") {
      canvasToPDF(offscreen).then((blob) => downloadBlob(blob, `${title || "show1d"}_figure.pdf`));
    } else {
      offscreen.toBlob((blob) => {
        if (blob) downloadBlob(blob, `${title || "show1d"}_figure.png`);
      });
    }
  }, [canvasW, canvasH, xMin, xMax, yMin, yMax, traceColors, traceLabels, lineWidth, logScale, showGrid, gridDensity, showLegend, xLabel, yLabel, xUnit, yUnit, title, dataToCanvasX, dataToCanvasY, plotW, plotH, peakMarkers]);

  // ========================================================================
  // CSV Export (Feature 6)
  // ========================================================================
  const buildCSV = React.useCallback((rangeOnly: boolean): string => {
    const traces = tracesRef.current;
    const xVals = xValuesRef.current;
    if (traces.length === 0) return "";

    const labels = traceLabels && traceLabels.length > 0 ? traceLabels : traces.map((_, i) => `Data ${i + 1}`);
    const header = ["x", ...labels].join(",");
    const rows: string[] = [header];

    for (let i = 0; i < (xVals ? xVals.length : traces[0].length); i++) {
      const xv = xVals ? xVals[i] : i;
      if (rangeOnly && xLocked) {
        if (xv < xMin || xv > xMax) continue;
      }
      const vals = [String(xv)];
      for (const trace of traces) {
        vals.push(String(trace[i] ?? ""));
      }
      rows.push(vals.join(","));
    }
    return rows.join("\n");
  }, [traceLabels, xLocked, xMin, xMax]);

  const handleCopyCSV = React.useCallback(() => {
    const csv = buildCSV(true);
    if (!csv) return;
    navigator.clipboard.writeText(csv).then(() => {
      setCsvCopied(true);
      setTimeout(() => setCsvCopied(false), 1000);
    });
  }, [buildCSV]);

  const handleExportCSVRange = React.useCallback(() => {
    setExportAnchor(null);
    const csv = buildCSV(true);
    if (!csv) return;
    downloadBlob(new Blob([csv], { type: "text/csv" }), `${title || "show1d"}_range.csv`);
  }, [buildCSV, title]);

  const handleExportCSVAll = React.useCallback(() => {
    setExportAnchor(null);
    const csv = buildCSV(false);
    if (!csv) return;
    downloadBlob(new Blob([csv], { type: "text/csv" }), `${title || "show1d"}.csv`);
  }, [buildCSV, title]);

  // Cursor style
  const getCursor = () => {
    if (rangeDragRef.current) return "ew-resize";
    if (rangeDragYRef.current) return "ns-resize";
    if (hoverRangeHandle) return "ew-resize";
    if (hoverRangeHandleY) return "ns-resize";
    if (axisDragRef.current) return axisDragRef.current.axis === "x" ? "ew-resize" : "ns-resize";
    if (peakDragRef.current?.active) return "col-resize";
    if (dragRef.current?.active) return "grabbing";
    if (hoverPeakIdx !== null) return "pointer";
    if (hoverLegendRef.current) return "pointer";
    if (peakActive && !lockPeaks) return "col-resize";
    return "crosshair";
  };

  // ========================================================================
  // JSX
  // ========================================================================

  return (
    <Box
      className="show1d-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      sx={{ ...container.root, bgcolor: colors.bg, color: colors.text }}
    >
      {/* Header row (Show3D pattern) */}
      <Typography
        variant="caption"
        sx={{
          ...typography.label,
          color: colors.accent,
          mb: `${SPACING.XS}px`,
          display: "block",
        }}
      >
        {title || "Plot"}
        <InfoTooltip
          theme={themeInfo.theme}
          text={
            <KeyboardShortcuts
              items={[
                ["Scroll", "Zoom in/out"],
                ["Drag", "Pan (or search peak when Peak on)"],
                ["Drag axis", "Lock X or Y range"],
                ["Dbl-click axis", "Unlock range"],
                ["Click", "Focus trace / select peak"],
                ["Shift+Click", "Multi-select peaks"],
                ["P", "Toggle peak mode"],
                ["C", "Clear all peaks"],
                ["Del", "Remove last peak"],
                ["Esc", "Deselect all"],
                ["R", "Reset view"],
                ["Dbl-click", "Reset view"],
              ]}
            />
          }
        />
        <ControlCustomizer
          widgetName="Show1D"
          hiddenTools={hiddenTools}
          setHiddenTools={setHiddenTools}
          disabledTools={disabledTools}
          setDisabledTools={setDisabledTools}
          themeColors={colors}
        />
      </Typography>

      {/* Controls row */}
      {showControls && !toolVisibility.hideAll && (
        <Box sx={{ display: "flex", alignItems: "center", gap: "4px", mb: "2px", height: 28 }}>
          {!hideDisplay && (<>
            <Typography sx={{ ...typography.labelSmall, color: colors.text }}>
              Log:
            </Typography>
            <Switch
              size="small"
              checked={logScale}
              onChange={(_, v) => setLogScale(v)}
              sx={switchStyles.small}
              disabled={lockDisplay}
            />

            <Typography sx={{ ...typography.labelSmall, color: colors.text, ml: "2px" }}>
              Auto:
            </Typography>
            <Switch
              size="small"
              checked={autoContrast}
              onChange={(_, v) => setAutoContrast(v)}
              sx={switchStyles.small}
              disabled={lockDisplay}
            />

            <Typography sx={{ ...typography.labelSmall, color: colors.text, ml: "2px" }}>
              Grid:
            </Typography>
            <Switch
              size="small"
              checked={showGrid}
              onChange={(_, v) => setShowGrid(v)}
              sx={switchStyles.small}
              disabled={lockDisplay}
            />

            {showGrid && (
              <Slider
                size="small"
                min={5}
                max={50}
                step={1}
                value={gridDensity}
                onChange={(_, v) => setGridDensity(v as number)}
                sx={{ width: 60, ml: "2px", ...sliderStyles.small }}
                disabled={lockDisplay}
              />
            )}

            <Typography sx={{ ...typography.labelSmall, color: colors.text, ml: "2px" }}>
              Legend:
            </Typography>
            <Switch
              size="small"
              checked={showLegend}
              onChange={(_, v) => setShowLegend(v)}
              sx={switchStyles.small}
              disabled={lockDisplay}
            />
          </>)}

          {!hidePeaks && (
            <>
              <Typography sx={{ ...typography.labelSmall, color: colors.text, ml: "2px" }}>
                Peak:
              </Typography>
              <Switch
                size="small"
                checked={peakActive}
                onChange={(_, v) => setPeakActive(v)}
                sx={switchStyles.small}
                disabled={lockPeaks}
              />
              {peakActive && (
                <>
                  <Typography sx={{ ...typography.labelSmall, color: colors.textMuted || colors.text, ml: "2px" }}>
                    ±{peakSearchRadius}
                  </Typography>
                  <Slider
                    size="small"
                    min={1}
                    max={100}
                    step={1}
                    value={peakSearchRadius}
                    onChange={(_, v) => setPeakSearchRadius(v as number)}
                    sx={{ width: 50, ml: "2px", ...sliderStyles.small }}
                    disabled={lockPeaks}
                  />
                </>
              )}
            </>
          )}

          <Box sx={{ flex: 1 }} />

          {!hideDisplay && (
            <Button size="small" sx={compactButton} onClick={resetView} disabled={lockDisplay}>Reset</Button>
          )}
          {!hideExport && (<>
            <Button size="small" sx={{ ...compactButton, color: colors.accent }} onClick={(e) => setExportAnchor(e.currentTarget)} disabled={lockExport}>Export</Button>
            <Menu
              anchorEl={exportAnchor}
              open={!!exportAnchor}
              onClose={() => setExportAnchor(null)}
              {...upwardMenuProps}
            >
              <MenuItem onClick={() => handleExportFigure("pdf")} sx={{ fontSize: 12 }}>
                Figure (PDF)
              </MenuItem>
              <MenuItem onClick={() => handleExportFigure("png")} sx={{ fontSize: 12 }}>
                Figure (PNG)
              </MenuItem>
              <MenuItem onClick={handleExportPNG} sx={{ fontSize: 12 }}>
                PNG
              </MenuItem>
              <MenuItem onClick={handleExportCSVRange} sx={{ fontSize: 12 }}>
                CSV (range)
              </MenuItem>
              <MenuItem onClick={handleExportCSVAll} sx={{ fontSize: 12 }}>
                CSV (all)
              </MenuItem>
            </Menu>
          </>)}
        </Box>
      )}

      {/* Canvas container */}
      <Box
        ref={containerRef}
        sx={{
          position: "relative",
          width: canvasW,
          height: canvasH,
          border: `1px solid ${colors.border}`,
          cursor: getCursor(),
          bgcolor: isDark ? "#1a1a1a" : "#f8f8f8",
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onDoubleClick={handleDoubleClick}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: canvasW,
            height: canvasH,
            position: "absolute",
            top: 0,
            left: 0,
          }}
        />
        <canvas
          ref={uiCanvasRef}
          style={{
            width: canvasW,
            height: canvasH,
            position: "absolute",
            top: 0,
            left: 0,
            pointerEvents: "none",
          }}
        />
        {/* Resize handle (Show2D pattern) */}
        <Box
          onMouseDown={(e: React.MouseEvent) => {
            resizeDragRef.current = {
              active: true,
              startX: e.clientX,
              startY: e.clientY,
              startW: canvasW,
              startH: canvasH,
            };
            e.stopPropagation();
          }}
          sx={{
            position: "absolute",
            bottom: 0,
            right: 0,
            width: 16,
            height: 16,
            cursor: "nwse-resize",
            opacity: 0.6,
            background: `linear-gradient(135deg, transparent 50%, ${colors.accent} 50%)`,
            "&:hover": { opacity: 1 },
          }}
        />
      </Box>

      {/* Range input row (when X locked) */}
      {xLocked && (
        <Box sx={{ display: "flex", alignItems: "center", gap: "6px", mt: "4px", mb: "2px", height: 24 }}>
          <Typography sx={{ ...typography.labelSmall, color: colors.text }}>
            X range:
          </Typography>
          <input
            type="number"
            value={rangeInputMin}
            onChange={(e) => { setRangeInputActive(true); setRangeInputMin(e.target.value); }}
            onFocus={() => setRangeInputActive(true)}
            onBlur={() => {
              setRangeInputActive(false);
              const val = parseFloat(rangeInputMin);
              if (!isNaN(val)) {
                const dataRange = xDataRangeRef.current;
                let newMin = Math.max(dataRange.min, Math.min(dataRange.max, val));
                if (newMin >= xMax) { const tmp = newMin; newMin = xMax; setXMax(tmp); setModelXRange([xMax, tmp]); setXMin(xMax); return; }
                setXMin(newMin);
                setModelXRange([newMin, xMax]);
              }
            }}
            onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
            style={{
              width: 80,
              fontSize: 10,
              fontFamily: "monospace",
              padding: "2px 4px",
              border: `1px solid ${colors.border}`,
              background: isDark ? "#2a2a2a" : "#fff",
              color: isDark ? "#ddd" : "#333",
              outline: "none",
            }}
          />
          <Typography sx={{ ...typography.labelSmall, color: colors.textMuted || colors.text }}>—</Typography>
          <input
            type="number"
            value={rangeInputMax}
            onChange={(e) => { setRangeInputActive(true); setRangeInputMax(e.target.value); }}
            onFocus={() => setRangeInputActive(true)}
            onBlur={() => {
              setRangeInputActive(false);
              const val = parseFloat(rangeInputMax);
              if (!isNaN(val)) {
                const dataRange = xDataRangeRef.current;
                let newMax = Math.max(dataRange.min, Math.min(dataRange.max, val));
                if (newMax <= xMin) { const tmp = newMax; newMax = xMin; setXMin(tmp); setModelXRange([tmp, xMin]); setXMax(xMin); return; }
                setXMax(newMax);
                setModelXRange([xMin, newMax]);
              }
            }}
            onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
            style={{
              width: 80,
              fontSize: 10,
              fontFamily: "monospace",
              padding: "2px 4px",
              border: `1px solid ${colors.border}`,
              background: isDark ? "#2a2a2a" : "#fff",
              color: isDark ? "#ddd" : "#333",
              outline: "none",
            }}
          />
          <Button
            size="small"
            sx={compactButton}
            onClick={() => {
              setXLocked(false);
              setModelXRange([]);
              setXMin(xDataRangeRef.current.min);
              setXMax(xDataRangeRef.current.max);
            }}
          >
            RESET
          </Button>
          {!hideExport && (
            <Button size="small" sx={{ ...compactButton, ml: "4px" }} onClick={handleCopyCSV} disabled={lockExport}>
              {csvCopied ? "COPIED" : "COPY CSV"}
            </Button>
          )}
        </Box>
      )}

      {/* Y Range input row (when Y locked) — Feature 2 */}
      {yLocked && (
        <Box sx={{ display: "flex", alignItems: "center", gap: "6px", mt: "4px", mb: "2px", height: 24 }}>
          <Typography sx={{ ...typography.labelSmall, color: colors.text }}>
            Y range:
          </Typography>
          <input
            type="number"
            value={rangeInputYMin}
            onChange={(e) => { setRangeInputYActive(true); setRangeInputYMin(e.target.value); }}
            onFocus={() => setRangeInputYActive(true)}
            onBlur={() => {
              setRangeInputYActive(false);
              const val = parseFloat(rangeInputYMin);
              if (!isNaN(val)) {
                const dataRange = yDataRangeRef.current;
                let newMin = Math.max(dataRange.min, Math.min(dataRange.max, val));
                if (newMin >= yMax) { const tmp = newMin; newMin = yMax; setYMax(tmp); setModelYRange([yMax, tmp]); setYMin(yMax); return; }
                setYMin(newMin);
                setModelYRange([newMin, yMax]);
              }
            }}
            onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
            style={{
              width: 80,
              fontSize: 10,
              fontFamily: "monospace",
              padding: "2px 4px",
              border: `1px solid ${colors.border}`,
              background: isDark ? "#2a2a2a" : "#fff",
              color: isDark ? "#ddd" : "#333",
              outline: "none",
            }}
          />
          <Typography sx={{ ...typography.labelSmall, color: colors.textMuted || colors.text }}>—</Typography>
          <input
            type="number"
            value={rangeInputYMax}
            onChange={(e) => { setRangeInputYActive(true); setRangeInputYMax(e.target.value); }}
            onFocus={() => setRangeInputYActive(true)}
            onBlur={() => {
              setRangeInputYActive(false);
              const val = parseFloat(rangeInputYMax);
              if (!isNaN(val)) {
                const dataRange = yDataRangeRef.current;
                let newMax = Math.max(dataRange.min, Math.min(dataRange.max, val));
                if (newMax <= yMin) { const tmp = newMax; newMax = yMin; setYMin(tmp); setModelYRange([tmp, yMin]); setYMax(yMin); return; }
                setYMax(newMax);
                setModelYRange([yMin, newMax]);
              }
            }}
            onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
            style={{
              width: 80,
              fontSize: 10,
              fontFamily: "monospace",
              padding: "2px 4px",
              border: `1px solid ${colors.border}`,
              background: isDark ? "#2a2a2a" : "#fff",
              color: isDark ? "#ddd" : "#333",
              outline: "none",
            }}
          />
          <Button
            size="small"
            sx={compactButton}
            onClick={() => {
              setYLocked(false);
              setModelYRange([]);
              setYMin(yDataRangeRef.current.min);
              setYMax(yDataRangeRef.current.max);
            }}
          >
            RESET
          </Button>
        </Box>
      )}

      {/* Stats bar (Show4DSTEM pattern, one row per trace) */}
      {showStats && !hideStats && statsMean && statsMean.length > 0 && (() => {
        const showIndices = focusedTrace >= 0 && focusedTrace < nTraces
          ? [focusedTrace]
          : Array.from({ length: nTraces }, (_, i) => i);
        return (
          <Box
            sx={{
              border: `1px solid ${colors.border}`,
              borderTop: "none",
              bgcolor: isDark ? colors.bg : "#fafafa",
              maxWidth: canvasW,
            }}
          >
            {showIndices.map((t) => {
              const label = (traceLabels && traceLabels[t]) || `Data ${t + 1}`;
              const traceColor = (traceColors && traceColors[t]) || "#4fc3f7";
              return (
                <Box key={t} sx={{ display: "flex", gap: 2, alignItems: "center", px: 1, py: 0.25 }}>
                  <Box sx={{ width: 8, height: 8, bgcolor: traceColor, flexShrink: 0 }} />
                  {nTraces > 1 && (
                    <Typography sx={{ fontSize: 11, color: colors.text, fontWeight: "bold", minWidth: 40 }}>{label}</Typography>
                  )}
                  <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>Mean <Box component="span" sx={{ color: traceColor }}>{formatNumber(statsMean[t] ?? 0)}</Box></Typography>
                  <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>Min <Box component="span" sx={{ color: traceColor }}>{formatNumber(statsMin[t] ?? 0)}</Box></Typography>
                  <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>Max <Box component="span" sx={{ color: traceColor }}>{formatNumber(statsMax[t] ?? 0)}</Box></Typography>
                  <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>Std <Box component="span" sx={{ color: traceColor }}>{formatNumber(statsStd[t] ?? 0)}</Box></Typography>
                </Box>
              );
            })}

            {/* Range stats (Feature 1+4) */}
            {xLocked && rangeStats && rangeStats.length > 0 && (
              <>
                <Box sx={{ borderTop: `1px dashed ${colors.border}`, mx: 1, my: 0.25 }} />
                {showIndices.map((t) => {
                  const rs = rangeStats[t];
                  if (!rs) return null;
                  const traceColor = (traceColors && traceColors[t]) || "#4fc3f7";
                  return (
                    <Box key={`rs-${t}`} sx={{ display: "flex", gap: 2, alignItems: "center", px: 1, py: 0.25 }}>
                      <Typography sx={{ fontSize: 10, color: isDark ? "#777" : "#aaa", minWidth: 60 }}>
                        Range ({rs.n_points} pts)
                      </Typography>
                      <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>Mean <Box component="span" sx={{ color: traceColor }}>{formatNumber(rs.mean)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>Min <Box component="span" sx={{ color: traceColor }}>{formatNumber(rs.min)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>Max <Box component="span" sx={{ color: traceColor }}>{formatNumber(rs.max)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>Std <Box component="span" sx={{ color: traceColor }}>{formatNumber(rs.std)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>∫ <Box component="span" sx={{ color: traceColor }}>{formatNumber(rs.integral)}</Box></Typography>
                    </Box>
                  );
                })}
              </>
            )}

            {/* FWHM stats (Feature 5) */}
            {peakFwhm && peakFwhm.length > 0 && (
              <>
                <Box sx={{ borderTop: `1px dashed ${colors.border}`, mx: 1, my: 0.25 }} />
                {peakFwhm.map((f, i) => {
                  const pk = peakMarkers && f.peak_idx >= 0 && f.peak_idx < peakMarkers.length ? peakMarkers[f.peak_idx] : null;
                  const traceColor = pk && traceColors && traceColors[pk.trace_idx] ? traceColors[pk.trace_idx] : "#4fc3f7";
                  return (
                    <Box key={`fwhm-${i}`} sx={{ display: "flex", gap: 2, alignItems: "center", px: 1, py: 0.25 }}>
                      <Typography sx={{ fontSize: 10, color: isDark ? "#777" : "#aaa", minWidth: 60 }}>
                        Peak {f.peak_idx}
                      </Typography>
                      {f.fwhm != null ? (
                        <>
                          <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>FWHM <Box component="span" sx={{ color: traceColor }}>{formatNumber(f.fwhm)}</Box></Typography>
                          <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>Center <Box component="span" sx={{ color: traceColor }}>{formatNumber(f.center ?? 0)}</Box></Typography>
                          <Typography sx={{ fontSize: 11, color: isDark ? "#888" : "#999" }}>R² <Box component="span" sx={{ color: traceColor }}>{(f.fit_quality ?? 0).toFixed(4)}</Box></Typography>
                        </>
                      ) : (
                        <Typography sx={{ fontSize: 11, color: isDark ? "#666" : "#bbb" }}>
                          {f.error || "Fit failed"}
                        </Typography>
                      )}
                    </Box>
                  );
                })}
              </>
            )}
          </Box>
        );
      })()}

    </Box>
  );
}

export const render = createRender(Show1D);
