/**
 * ShowDiffraction — interactive d-spacing analysis for 2D/3D/4D diffraction patterns.
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
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawColorbar, exportFigure, canvasToPDF } from "../scalebar";
import { formatNumber, downloadBlob } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, sliderRange, applyLogScaleInPlace } from "../stats";
import { COLORMAPS, COLORMAP_NAMES, applyColormap, renderToOffscreen } from "../colormaps";
import { computeToolVisibility } from "../tool-parity";
import "./showdiffraction.css";

// ============================================================================
// Style constants
// ============================================================================

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const CANVAS_MIN = 384;
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const typography = {
  label: { fontSize: 11 },
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
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
};

// ============================================================================
// InfoTooltip + KeyboardShortcuts (same pattern as Show4DSTEM)
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
// Helper: Histogram (inline, same pattern as other widgets)
// ============================================================================

interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
}

function Histogram({ data, vminPct, vmaxPct, onRangeChange, width = 110, height = 50, theme = "dark" }: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => data ? computeHistogramFromBytes(data) : null, [data]);
  const draggingRef = React.useRef<"left" | "right" | null>(null);
  const isDark = theme === "dark";

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !bins) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = isDark ? "#1a1a2e" : "#f0f0f0";
    ctx.fillRect(0, 0, width, height);
    const maxBin = Math.max(...Array.from(bins));
    if (maxBin > 0) {
      ctx.fillStyle = isDark ? "#555" : "#999";
      for (let i = 0; i < bins.length; i++) {
        const x = (i / bins.length) * width;
        const bw = width / bins.length;
        const bh = (bins[i] / maxBin) * height;
        ctx.fillRect(x, height - bh, bw, bh);
      }
    }
    const lx = (vminPct / 100) * width;
    const rx = (vmaxPct / 100) * width;
    ctx.fillStyle = isDark ? "rgba(0,0,0,0.5)" : "rgba(0,0,0,0.2)";
    ctx.fillRect(0, 0, lx, height);
    ctx.fillRect(rx, 0, width - rx, height);
    ctx.fillStyle = isDark ? "#4fc3f7" : "#1976d2";
    ctx.fillRect(lx - 1, 0, 3, height);
    ctx.fillRect(rx - 1, 0, 3, height);
  }, [bins, vminPct, vmaxPct, width, height, isDark]);

  const handleMouse = (e: React.MouseEvent, isDown: boolean) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / width) * 100));
    if (isDown) {
      const dl = Math.abs(pct - vminPct);
      const dr = Math.abs(pct - vmaxPct);
      draggingRef.current = dl < dr ? "left" : "right";
    }
    if (draggingRef.current === "left") onRangeChange(Math.min(pct, vmaxPct - 1), vmaxPct);
    else if (draggingRef.current === "right") onRangeChange(vminPct, Math.max(pct, vminPct + 1));
  };

  return (
    <canvas
      ref={canvasRef} width={width} height={height}
      style={{ cursor: "ew-resize", display: "block" }}
      onMouseDown={(e) => handleMouse(e, true)}
      onMouseMove={(e) => { if (draggingRef.current) handleMouse(e, false); }}
      onMouseUp={() => { draggingRef.current = null; }}
      onMouseLeave={() => { draggingRef.current = null; }}
    />
  );
}

// ============================================================================
// Helper: format stat value
// ============================================================================
function formatStat(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) return v.toExponential(2);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

// ============================================================================
// Spot type
// ============================================================================

interface SpotDict {
  id: number;
  row: number;
  col: number;
  raw_row?: number;
  raw_col?: number;
  row_err?: number;
  col_err?: number;
  d_spacing: number | null;
  d_spacing_err?: number | null;
  g_magnitude: number | null;
  g_magnitude_err?: number | null;
  r_pixels: number;
  r_pixels_err?: number;
  angle_deg?: number | null;
  angle_deg_err?: number | null;
  fit_quality?: number | null;
  intensity: number;
  hkl?: string;
  note?: string;
}

interface RingDict {
  id: number;
  radius_px: number;
  g_magnitude: number | null;
  d_spacing: number | null;
  intensity: number;
  hkl?: string;
}

// Shared spot color palette — table rows and canvas overlay stay in sync by index.
const PICK_COLORS = [
  "#ff4d4f", "#40a9ff", "#73d13d", "#ffa940",
  "#9254de", "#13c2c2", "#f759ab", "#bae637",
];
const spotColorAt = (index: number) => PICK_COLORS[((index % PICK_COLORS.length) + PICK_COLORS.length) % PICK_COLORS.length];

// ============================================================================
// Main component
// ============================================================================

function ShowDiffraction() {
  const { themeInfo, colors: themeColors } = useTheme();
  const rootRef = React.useRef<HTMLDivElement>(null);

  const themedSelect = {
    "& .MuiSelect-select": { py: 0.25, px: 1, fontSize: 10, color: themeColors.text },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
    bgcolor: themeColors.controlBg,
    minWidth: 80,
  };
  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } },
  };

  // Model state
  const [title] = useModelState<string>("title");
  const [posRow, setPosRow] = useModelState<number>("pos_row");
  const [posCol, setPosCol] = useModelState<number>("pos_col");
  const [shapeRows] = useModelState<number>("shape_rows");
  const [shapeCols] = useModelState<number>("shape_cols");
  const [detRows] = useModelState<number>("det_rows");
  const [detCols] = useModelState<number>("det_cols");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [virtualImageBytes] = useModelState<DataView>("virtual_image_bytes");
  const [centerRow] = useModelState<number>("center_row");
  const [centerCol] = useModelState<number>("center_col");
  const [bfRadius] = useModelState<number>("bf_radius");
  const [kPixelSize] = useModelState<number>("k_pixel_size");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [spots, setSpots] = useModelState<SpotDict[]>("spots");
  const [snapEnabled, setSnapEnabled] = useModelState<boolean>("snap_enabled");
  const [snapRadius] = useModelState<number>("snap_radius");
  const [spotRefine, setSpotRefine] = useModelState<boolean>("spot_refine");
  const [referenceSpotId, setReferenceSpotId] = useModelState<number>("reference_spot_id");
  const [, setSpotAddRequest] = useModelState<number[]>("_spot_add_request");
  const [, setSpotUndoRequest] = useModelState<boolean>("_spot_undo_request");
  const [, setSpotClearRequest] = useModelState<boolean>("_spot_clear_request");
  const [, setDetectRequest] = useModelState<number>("_detect_spots_request");
  const [, setDetectRingsRequest] = useModelState<number>("_detect_rings_request");
  const [, setSpotRemoveRequest] = useModelState<number>("_spot_remove_request");
  const [, setRingRemoveRequest] = useModelState<number>("_ring_remove_request");
  const [dpColormap, setDpColormap] = useModelState<string>("dp_colormap");
  const [dpScaleMode, setDpScaleMode] = useModelState<string>("dp_scale_mode");
  const [dpInvert, setDpInvert] = useModelState<boolean>("dp_invert");
  const [dpVminPct, setDpVminPct] = useModelState<number>("dp_vmin_pct");
  const [dpVmaxPct, setDpVmaxPct] = useModelState<number>("dp_vmax_pct");
  const [viColormap, setViColormap] = useModelState<string>("vi_colormap");
  const [viVminPct, setViVminPct] = useModelState<number>("vi_vmin_pct");
  const [viVmaxPct, setViVmaxPct] = useModelState<number>("vi_vmax_pct");
  const [dpStats] = useModelState<number[]>("dp_stats");
  const [viStats] = useModelState<number[]>("vi_stats");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [disabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools] = useModelState<string[]>("hidden_tools");

  // ShowDiffraction core (2D input, center, radial I(q), rings, calib)
  const [is2d] = useModelState<boolean>("is_2d");
  const [centerMode, setCenterMode] = useModelState<string>("center_mode");
  const [showRadial, setShowRadial] = useModelState<boolean>("show_radial");
  const [radialGBytes] = useModelState<DataView>("radial_g_bytes");
  const [radialIBytes] = useModelState<DataView>("radial_i_bytes");
  const [radialCalibrated] = useModelState<boolean>("radial_calibrated");
  const [rings] = useModelState<RingDict[]>("rings");
  const [calibrationSource] = useModelState<string>("calibration_source");
  const [calibrationRefD] = useModelState<number>("calibration_ref_d");
  const [calibrationRefRadius] = useModelState<number>("calibration_ref_radius");
  const [, setRingAddRequest] = useModelState<number[]>("_ring_add_request");
  const [, setRingUndoRequest] = useModelState<boolean>("_ring_undo_request");
  const [, setRingClearRequest] = useModelState<boolean>("_ring_clear_request");
  const [, setCenterFromPointsRequest] = useModelState<number[]>("_center_from_points_request");
  const [, setCalibrateFromRingRequest] = useModelState<number[]>("_calibrate_from_ring_request");
  const [, setCalibrateFromSpotRequest] = useModelState<number[]>("_calibrate_from_spot_request");

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("ShowDiffraction", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideStats = toolVisibility.isHidden("stats");
  const hideHistogram = toolVisibility.isHidden("histogram");
  const hideDisplay = toolVisibility.isHidden("display");
  const hideExport = toolVisibility.isHidden("export");
  const hideSpots = toolVisibility.isHidden("spots");
  const hideNavigation = toolVisibility.isHidden("navigation");
  const hideView = toolVisibility.isHidden("view");
  const lockSpots = toolVisibility.isLocked("spots");
  const lockExport = toolVisibility.isLocked("export");
  const lockDisplay = toolVisibility.isLocked("display");
  const lockHistogram = toolVisibility.isLocked("histogram");
  const lockNavigation = toolVisibility.isLocked("navigation");
  const lockView = toolVisibility.isLocked("view");
  const lockStats = toolVisibility.isLocked("stats");

  // Measurement table helpers
  const updateSpotField = React.useCallback((id: number, field: "hkl" | "note", value: string) => {
    setSpots((spots || []).map((s) => (s.id === id ? { ...s, [field]: value } : s)));
  }, [spots, setSpots]);

  const exportMeasurements = React.useCallback((format: "csv" | "json") => {
    const cols = [
      "id", "kind", "row", "col", "r_pixels", "r_pixels_err",
      "g_inv_angstrom", "g_inv_angstrom_err", "d_angstrom", "d_angstrom_err",
      "angle_deg", "angle_deg_err", "intensity", "fit_quality", "hkl", "note",
    ];
    const rows: (string | number | null)[][] = [];
    for (const s of spots || []) {
      rows.push([s.id, "spot", s.row, s.col, s.r_pixels, s.r_pixels_err ?? null,
        s.g_magnitude, s.g_magnitude_err ?? null, s.d_spacing, s.d_spacing_err ?? null,
        s.angle_deg ?? null, s.angle_deg_err ?? null, s.intensity, s.fit_quality ?? null,
        s.hkl ?? "", s.note ?? ""]);
    }
    for (const r of rings || []) {
      rows.push([r.id, "ring", null, null, r.radius_px, null, r.g_magnitude, null,
        r.d_spacing, null, null, null, r.intensity, null, "", ""]);
    }
    if (format === "json") {
      const records = rows.map((r) => Object.fromEntries(cols.map((c, i) => [c, r[i]])));
      const blob = new Blob([JSON.stringify({ measurements: records }, null, 2)], { type: "application/json" });
      downloadBlob(blob, "measurements.json");
    } else {
      const esc = (v: string | number | null) => {
        const s = v == null ? "" : String(v);
        return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
      };
      const csv = [cols.join(","), ...rows.map((r) => r.map(esc).join(","))].join("\n");
      downloadBlob(new Blob([csv], { type: "text/csv" }), "measurements.csv");
    }
  }, [spots, rings]);

  // Local UI state
  const [canvasSize, setCanvasSize] = React.useState(CANVAS_MIN);
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeCanvasStart, setResizeCanvasStart] = React.useState<{ x: number; y: number; size: number } | null>(null);
  const [dpZoom, setDpZoom] = React.useState(1);
  const [dpPanX, setDpPanX] = React.useState(0);
  const [dpPanY, setDpPanY] = React.useState(0);
  const [viZoom, setViZoom] = React.useState(1);
  const [viPanX, setViPanX] = React.useState(0);
  const [viPanY, setViPanY] = React.useState(0);
  const [dpHistData, setDpHistData] = React.useState<Float32Array | null>(null);
  const [viHistData, setViHistData] = React.useState<Float32Array | null>(null);
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);
  const [dpExportAnchor, setDpExportAnchor] = React.useState<HTMLElement | null>(null);
  // 2 (midpoint) or 3 (ring) center-pick points
  const [centerPickPoints, setCenterPickPoints] = React.useState<{ row: number; col: number }[]>([]);
  const [dKnown, setDKnown] = React.useState("");

  // center the diffraction center at 2.5x zoom
  const zoomToCenter = React.useCallback(() => {
    const Z = 2.5;
    setDpZoom(Z);
    setDpPanX(canvasSize * Z * (0.5 - centerCol / Math.max(detCols, 1)));
    setDpPanY(canvasSize * Z * (0.5 - centerRow / Math.max(detRows, 1)));
  }, [canvasSize, centerRow, centerCol, detRows, detCols]);

  // Canvas refs
  const dpCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const dpUiRef = React.useRef<HTMLCanvasElement>(null);
  const dpScaleRef = React.useRef<HTMLCanvasElement>(null);
  const dpOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const viCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const viUiRef = React.useRef<HTMLCanvasElement>(null);
  const viScaleRef = React.useRef<HTMLCanvasElement>(null);
  const viOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const rawDpDataRef = React.useRef<Float32Array | null>(null);
  const [dpVersion, setDpVersion] = React.useState(0);
  const [viVersion, setViVersion] = React.useState(0);
  const dpVminRef = React.useRef(0);
  const dpVmaxRef = React.useRef(1);
  const radialCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const radialPlotRef = React.useRef<{ x: Float32Array; padL: number; plotW: number } | null>(null);

  // Canvas resize handle
  const handleCanvasResizeStart = (e: React.MouseEvent) => {
    if (lockView) return;
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
      latestSize = Math.max(CANVAS_MIN, resizeCanvasStart.size + delta);
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
      setResizeCanvasStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingCanvas, resizeCanvasStart]);

  // Colormap LUT, reversed when Invert is on (applyColormap can't take a reversed range).
  const dpLut = React.useMemo(() => {
    const base = COLORMAPS[dpColormap] || COLORMAPS.inferno;
    if (!dpInvert) return base;
    const n = base.length / 3;
    const inv = new Uint8Array(base.length);
    for (let k = 0; k < n; k++) {
      const s = (n - 1 - k) * 3, d = k * 3;
      inv[d] = base[s]; inv[d + 1] = base[s + 1]; inv[d + 2] = base[s + 2];
    }
    return inv;
  }, [dpColormap, dpInvert]);

  // DP rendering (expensive: colormap)
  React.useEffect(() => {
    if (!frameBytes || !frameBytes.byteLength) return;
    const raw = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
    rawDpDataRef.current = raw;
    let scaled: Float32Array;
    if (dpScaleMode === "log") {
      scaled = new Float32Array(raw.length);
      applyLogScaleInPlace(raw, scaled);
    } else if (dpScaleMode === "sqrt") {
      scaled = new Float32Array(raw.length);
      let mn = Infinity;
      for (let i = 0; i < raw.length; i++) if (raw[i] < mn) mn = raw[i];
      for (let i = 0; i < raw.length; i++) scaled[i] = Math.sqrt(Math.max(raw[i] - mn, 0));
    } else {
      scaled = raw;
    }
    const { min: dataMin, max: dataMax } = findDataRange(scaled);
    const { vmin, vmax } = sliderRange(dataMin, dataMax, dpVminPct, dpVmaxPct);
    dpVminRef.current = vmin;
    dpVmaxRef.current = vmax;
    let offscreen = dpOffscreenRef.current;
    if (!offscreen) { offscreen = document.createElement("canvas"); dpOffscreenRef.current = offscreen; }
    offscreen.width = detCols;
    offscreen.height = detRows;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(detCols, detRows);
    applyColormap(scaled, imgData.data, dpLut, vmin, vmax);
    ctx.putImageData(imgData, 0, 0);
    setDpHistData(scaled);
    setDpVersion(v => v + 1);
  }, [frameBytes, dpLut, dpScaleMode, dpVminPct, dpVmaxPct, detRows, detCols]);

  // DP draw (cheap: zoom/pan)
  React.useLayoutEffect(() => {
    const canvas = dpCanvasRef.current;
    const offscreen = dpOffscreenRef.current;
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasSize, canvasSize);
    const offX = (canvasSize - canvasSize * dpZoom) / 2 + dpPanX;
    const offY = (canvasSize - canvasSize * dpZoom) / 2 + dpPanY;
    ctx.drawImage(offscreen, offX, offY, canvasSize * dpZoom, canvasSize * dpZoom);
  }, [dpVersion, dpZoom, dpPanX, dpPanY, canvasSize, detRows, detCols]);

  // DP UI overlay (spots, center, scale bar)
  React.useLayoutEffect(() => {
    const canvas = dpUiRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const cssW = canvasSize;
    canvas.width = cssW * DPR;
    canvas.height = cssW * DPR;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, cssW, cssW);

    const scX = (cssW / detCols) * dpZoom;
    const scY = (cssW / detRows) * dpZoom;
    const offX = (cssW - cssW * dpZoom) / 2 + dpPanX;
    const offY = (cssW - cssW * dpZoom) / 2 + dpPanY;

    // Center crosshair
    const cx = offX + centerCol * scX;
    const cy = offY + centerRow * scY;
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx - 10, cy); ctx.lineTo(cx + 10, cy);
    ctx.moveTo(cx, cy - 10); ctx.lineTo(cx, cy + 10);
    ctx.stroke();
    // BF disk circle
    const br = bfRadius * scX;
    ctx.beginPath();
    ctx.arc(cx, cy, br, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);

    // Spot markers (per-spot color; the reference spot gets a filled core)
    if (spots && spots.length > 0) {
      const refId = referenceSpotId >= 0 ? referenceSpotId : spots[0].id;
      spots.forEach((spot, i) => {
        const sx = offX + spot.col * scX;
        const sy = offY + spot.row * scY;
        const color = spotColorAt(i);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
        ctx.stroke();
        if (spot.id === refId) {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(sx, sy, 2.5, 0, 2 * Math.PI);
          ctx.fill();
        }
        ctx.fillStyle = color;
        ctx.font = "bold 10px -apple-system, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(spot.hkl ? `${spot.id}:${spot.hkl}` : `${spot.id}`, sx + 8, sy - 2);
      });
    }

    // Picked rings (concentric circles about the center)
    if (rings && rings.length > 0) {
      ctx.strokeStyle = themeInfo.theme === "dark" ? "#ffb74d" : "#e65100";
      ctx.lineWidth = 1.2;
      for (const ring of rings) {
        ctx.beginPath();
        ctx.arc(cx, cy, ring.radius_px * scX, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }

    // In-progress center-pick points (midpoint / ring modes)
    if (centerPickPoints.length > 0) {
      ctx.strokeStyle = themeInfo.theme === "dark" ? "#ff4fd8" : "#c2186b";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 1.5;
      centerPickPoints.forEach((p, i) => {
        const px2 = offX + p.col * scX;
        const py2 = offY + p.row * scY;
        ctx.beginPath();
        ctx.moveTo(px2 - 6, py2); ctx.lineTo(px2 + 6, py2);
        ctx.moveTo(px2, py2 - 6); ctx.lineTo(px2, py2 + 6);
        ctx.stroke();
        ctx.font = "bold 10px -apple-system, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(`${i + 1}`, px2 + 7, py2 - 2);
      });
    }

    // Colorbar
    drawColorbar(ctx, cssW, cssW, dpLut, dpVminRef.current, dpVmaxRef.current, dpScaleMode === "log");

    // Zoom indicator
    if (dpZoom !== 1) {
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${dpZoom.toFixed(1)}×`, 8, cssW - 8);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [dpVersion, dpZoom, dpPanX, dpPanY, canvasSize, detRows, detCols, centerRow, centerCol, bfRadius, spots, rings, referenceSpotId, centerPickPoints, dpLut, dpScaleMode, kCalibrated, kPixelSize, themeInfo.theme]);

  // K-space scale bar on its own canvas (drawScaleBarHiDPI clears the whole canvas).
  React.useLayoutEffect(() => {
    const canvas = dpScaleRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasSize * DPR;   // resets + clears
    canvas.height = canvasSize * DPR;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (kCalibrated && kPixelSize > 0) {
      drawScaleBarHiDPI(canvas, DPR, dpZoom, kPixelSize, "mrad", detCols);
    }
  }, [canvasSize, dpZoom, kCalibrated, kPixelSize, detCols]);

  // VI rendering (expensive: colormap)
  React.useEffect(() => {
    if (!virtualImageBytes || !virtualImageBytes.byteLength) return;
    const raw = new Float32Array(virtualImageBytes.buffer, virtualImageBytes.byteOffset, virtualImageBytes.byteLength / 4);
    const { min: dataMin, max: dataMax } = findDataRange(raw);
    const { vmin, vmax } = sliderRange(dataMin, dataMax, viVminPct, viVmaxPct);
    const lut = COLORMAPS[viColormap] || COLORMAPS.inferno;
    let offscreen = viOffscreenRef.current;
    if (!offscreen) { offscreen = document.createElement("canvas"); viOffscreenRef.current = offscreen; }
    offscreen.width = shapeCols;
    offscreen.height = shapeRows;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return;
    const imgData = ctx.createImageData(shapeCols, shapeRows);
    applyColormap(raw, imgData.data, lut, vmin, vmax);
    ctx.putImageData(imgData, 0, 0);
    setViHistData(raw);
    setViVersion(v => v + 1);
  }, [virtualImageBytes, viColormap, viVminPct, viVmaxPct, shapeRows, shapeCols]);

  // VI draw (cheap: zoom/pan)
  React.useLayoutEffect(() => {
    const canvas = viCanvasRef.current;
    const offscreen = viOffscreenRef.current;
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasSize, canvasSize);
    const offX = (canvasSize - canvasSize * viZoom) / 2 + viPanX;
    const offY = (canvasSize - canvasSize * viZoom) / 2 + viPanY;
    ctx.drawImage(offscreen, offX, offY, canvasSize * viZoom, canvasSize * viZoom);
  }, [viVersion, viZoom, viPanX, viPanY, canvasSize]);

  // VI UI overlay (position crosshair, scale bar)
  React.useLayoutEffect(() => {
    const canvas = viUiRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const cssW = canvasSize;
    canvas.width = cssW * DPR;
    canvas.height = cssW * DPR;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, cssW, cssW);

    const scX = (cssW / shapeCols) * viZoom;
    const scY = (cssW / shapeRows) * viZoom;
    const offX = (cssW - cssW * viZoom) / 2 + viPanX;
    const offY = (cssW - cssW * viZoom) / 2 + viPanY;

    // Position crosshair
    const px = offX + (posCol + 0.5) * scX;
    const py = offY + (posRow + 0.5) * scY;
    ctx.strokeStyle = "rgba(255,100,100,0.8)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px - 8, py); ctx.lineTo(px + 8, py);
    ctx.moveTo(px, py - 8); ctx.lineTo(px, py + 8);
    ctx.stroke();

    // Zoom indicator
    if (viZoom !== 1) {
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${viZoom.toFixed(1)}×`, 8, cssW - 8);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [viVersion, viZoom, viPanX, viPanY, canvasSize, shapeRows, shapeCols, posRow, posCol, pixelSize]);

  // VI scale bar on its own canvas (drawScaleBarHiDPI clears the canvas)
  React.useLayoutEffect(() => {
    const canvas = viScaleRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvasSize * DPR;
    canvas.height = canvasSize * DPR;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (pixelSize > 0) {
      drawScaleBarHiDPI(canvas, DPR, viZoom, pixelSize, "Å", shapeCols);
    }
  }, [canvasSize, viZoom, pixelSize, shapeCols]);

  // Radial I(q) profile plot
  // Span both panels normally; match the single DP panel when the VI is hidden (2D).
  const radialW = is2d ? canvasSize : canvasSize * 2 + SPACING.LG;
  const radialH = 150;
  React.useLayoutEffect(() => {
    if (!showRadial) return;
    const canvas = radialCanvasRef.current;
    if (!canvas || !radialGBytes || !radialGBytes.byteLength) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const x = new Float32Array(radialGBytes.buffer, radialGBytes.byteOffset, radialGBytes.byteLength / 4);
    const iRaw = new Float32Array(radialIBytes.buffer, radialIBytes.byteOffset, radialIBytes.byteLength / 4);
    const n = Math.min(x.length, iRaw.length);
    if (n < 2) return;

    const W = radialW, H = radialH;
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = themeColors.bg;
    ctx.fillRect(0, 0, W, H);

    const padL = 46, padR = 14, padT = 10, padB = 26;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    // Optional log-intensity (shares the DP scale-mode toggle).
    const yv = new Float32Array(n);
    for (let i = 0; i < n; i++) yv[i] = dpScaleMode === "log" ? Math.log1p(Math.max(iRaw[i], 0)) : dpScaleMode === "sqrt" ? Math.sqrt(Math.max(iRaw[i], 0)) : iRaw[i];
    const xMin = x[0], xMax = x[n - 1];
    let yMax = 0;
    for (let i = 0; i < n; i++) if (yv[i] > yMax) yMax = yv[i];
    if (yMax <= 0) yMax = 1;
    const xSpan = xMax - xMin || 1;
    const sx = (xi: number) => padL + ((xi - xMin) / xSpan) * plotW;
    const sy = (yi: number) => padT + plotH - (yi / yMax) * plotH;

    // Axes
    ctx.strokeStyle = themeColors.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    // Picked-ring vertical markers
    if (rings && rings.length > 0) {
      ctx.strokeStyle = themeInfo.theme === "dark" ? "#ffb74d" : "#e65100";
      ctx.lineWidth = 1;
      for (const ring of rings) {
        const xr = radialCalibrated && kPixelSize > 0 ? ring.radius_px * kPixelSize : ring.radius_px;
        if (xr < xMin || xr > xMax) continue;
        const px = sx(xr);
        ctx.beginPath();
        ctx.moveTo(px, padT); ctx.lineTo(px, padT + plotH);
        ctx.stroke();
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = "9px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`${ring.id}`, px, padT + 8);
      }
    }

    // Curve
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const px = sx(x[i]), py = sy(yv[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Axis labels + ticks
    ctx.fillStyle = themeColors.textMuted;
    ctx.font = "10px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xLabel = radialCalibrated ? "g (1/Å)" : "radius (px)";
    for (let t = 0; t <= 4; t++) {
      const xi = xMin + (t / 4) * xSpan;
      ctx.fillText(xi.toFixed(radialCalibrated ? 2 : 0), sx(xi), padT + plotH + 4);
    }
    ctx.fillText(xLabel, padL + plotW / 2, padT + plotH + 14);
    ctx.save();
    ctx.translate(12, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillText(dpScaleMode === "log" ? "log I(g)" : dpScaleMode === "sqrt" ? "√I(g)" : "I(g)", 0, 0);
    ctx.restore();

    radialPlotRef.current = { x, padL, plotW };
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [showRadial, radialGBytes, radialIBytes, rings, radialCalibrated, kPixelSize, dpScaleMode, radialW, themeColors, themeInfo.theme]);

  const handleRadialClick = (e: React.MouseEvent) => {
    const canvas = radialCanvasRef.current;
    const meta = radialPlotRef.current;
    if (!canvas || !meta || meta.x.length < 2) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const frac = (mx - meta.padL) / meta.plotW;
    if (frac < 0 || frac > 1) return;
    const xMin = meta.x[0], xMax = meta.x[meta.x.length - 1];
    const clickedX = xMin + frac * (xMax - xMin);
    const radiusPx = radialCalibrated && kPixelSize > 0 ? clickedX / kPixelSize : clickedX;
    if (radiusPx > 0) setRingAddRequest([radiusPx]);
  };

  // DP mouse handlers
  const dpIsDragging = React.useRef(false);
  const dpDragStart = React.useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const dpToImage = (e: React.MouseEvent) => {
    const canvas = dpCanvasRef.current;
    if (!canvas) return { row: 0, col: 0 };
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const offX = (canvasSize - canvasSize * dpZoom) / 2 + dpPanX;
    const offY = (canvasSize - canvasSize * dpZoom) / 2 + dpPanY;
    const col = (mx - offX) / (canvasSize * dpZoom) * detCols;
    const row = (my - offY) / (canvasSize * dpZoom) * detRows;
    return { row, col };
  };

  const handleDpMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || e.button === 2 || e.shiftKey) {
      dpIsDragging.current = true;
      dpDragStart.current = { x: e.clientX, y: e.clientY, panX: dpPanX, panY: dpPanY };
      return;
    }
    const { row, col } = dpToImage(e);
    if (!(row >= 0 && row < detRows && col >= 0 && col < detCols)) return;
    // Center-finding modes collect points instead of adding spots.
    if (centerMode === "midpoint" || centerMode === "ring") {
      const need = centerMode === "midpoint" ? 2 : 3;
      const pts = [...centerPickPoints, { row, col }];
      if (pts.length >= need) {
        const flat: number[] = [];
        pts.slice(0, need).forEach((p) => flat.push(p.row, p.col));
        setCenterFromPointsRequest(flat);
        setCenterPickPoints([]);
        setCenterMode("auto");
      } else {
        setCenterPickPoints(pts);
      }
      return;
    }
    // Left click: add spot
    if (!lockSpots) setSpotAddRequest([row, col]);
  };

  const handleDpMouseMove = (e: React.MouseEvent) => {
    if (dpIsDragging.current) {
      setDpPanX(dpDragStart.current.panX + (e.clientX - dpDragStart.current.x));
      setDpPanY(dpDragStart.current.panY + (e.clientY - dpDragStart.current.y));
      return;
    }
    // Cursor readout
    if (!frameBytes || !frameBytes.byteLength) return;
    const { row, col } = dpToImage(e);
    const ri = Math.round(row), ci = Math.round(col);
    if (ri >= 0 && ri < detRows && ci >= 0 && ci < detCols) {
      const raw = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
      setCursorInfo({ row: ri, col: ci, value: raw[ri * detCols + ci] });
    } else {
      setCursorInfo(null);
    }
  };

  const handleDpMouseUp = () => { dpIsDragging.current = false; };
  const handleDpMouseLeave = () => { dpIsDragging.current = false; setCursorInfo(null); };

  const handleDpWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setDpZoom(z => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * delta)));
  };

  const resetDpView = () => { setDpZoom(1); setDpPanX(0); setDpPanY(0); };

  // VI mouse handlers
  const viIsDragging = React.useRef(false);
  const viDragStart = React.useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const viToImage = (e: React.MouseEvent) => {
    const canvas = viCanvasRef.current;
    if (!canvas) return { row: 0, col: 0 };
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const offX = (canvasSize - canvasSize * viZoom) / 2 + viPanX;
    const offY = (canvasSize - canvasSize * viZoom) / 2 + viPanY;
    const col = (mx - offX) / (canvasSize * viZoom) * shapeCols;
    const row = (my - offY) / (canvasSize * viZoom) * shapeRows;
    return { row, col };
  };

  const handleViMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || e.button === 2 || e.shiftKey) {
      viIsDragging.current = true;
      viDragStart.current = { x: e.clientX, y: e.clientY, panX: viPanX, panY: viPanY };
      return;
    }
    const { row, col } = viToImage(e);
    const r = Math.round(row), c = Math.round(col);
    if (r >= 0 && r < shapeRows && c >= 0 && c < shapeCols) {
      setPosRow(r);
      setPosCol(c);
    }
  };

  const handleViMouseMove = (e: React.MouseEvent) => {
    if (viIsDragging.current) {
      setViPanX(viDragStart.current.panX + (e.clientX - viDragStart.current.x));
      setViPanY(viDragStart.current.panY + (e.clientY - viDragStart.current.y));
    }
  };

  const handleViMouseUp = () => { viIsDragging.current = false; };
  const handleViWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setViZoom(z => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * delta)));
  };
  const resetViView = () => { setViZoom(1); setViPanX(0); setViPanY(0); };

  // Wheel scroll prevention
  const dpContainerRef = React.useRef<HTMLDivElement>(null);
  const viContainerRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const prevent = (e: WheelEvent) => e.preventDefault();
    const dp = dpContainerRef.current;
    const vi = viContainerRef.current;
    if (dp) dp.addEventListener("wheel", prevent, { passive: false });
    if (vi) vi.addEventListener("wheel", prevent, { passive: false });
    return () => {
      if (dp) dp.removeEventListener("wheel", prevent);
      if (vi) vi.removeEventListener("wheel", prevent);
    };
  }, []);

  // Export handlers
  const handleCopyDP = () => {
    const offscreen = dpOffscreenRef.current;
    if (!offscreen) return;
    offscreen.toBlob((blob) => {
      if (blob) {
        try { navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]); }
        catch { downloadBlob(blob, "diffraction.png"); }
      }
    });
  };

  const handleExportFigure = (withColorbar: boolean) => {
    if (lockExport) return;
    setDpExportAnchor(null);
    const frameData = rawDpDataRef.current;
    if (!frameData) return;
    let processed: Float32Array;
    if (dpScaleMode === "log") {
      processed = new Float32Array(frameData.length);
      applyLogScaleInPlace(frameData, processed);
    } else if (dpScaleMode === "sqrt") {
      processed = new Float32Array(frameData.length);
      let mn = Infinity;
      for (let i = 0; i < frameData.length; i++) if (frameData[i] < mn) mn = frameData[i];
      for (let i = 0; i < frameData.length; i++) processed[i] = Math.sqrt(Math.max(frameData[i] - mn, 0));
    } else {
      processed = frameData;
    }
    const { min: dMin, max: dMax } = findDataRange(processed);
    const { vmin, vmax } = sliderRange(dMin, dMax, dpVminPct, dpVmaxPct);
    const offscreen = renderToOffscreen(processed, detCols, detRows, dpLut, vmin, vmax);
    if (!offscreen) return;
    const kPxVal = kPixelSize > 0 && kCalibrated ? kPixelSize : 0;
    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: `DP at (${posRow}, ${posCol})`,
      lut: dpLut,
      vmin,
      vmax,
      logScale: dpScaleMode === "log",
      pixelSize: kPxVal > 0 ? kPxVal : undefined,
      showColorbar: withColorbar,
      showScaleBar: kPxVal > 0,
    });
    canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, "showdiffraction_dp_figure.pdf")).catch(console.error);
  };

  const handleExportPng = () => {
    if (lockExport) return;
    setDpExportAnchor(null);
    if (!dpCanvasRef.current) return;
    dpCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "showdiffraction_dp.png"); }, "image/png");
  };

  // Keyboard
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
        if (!lockNavigation) {
          setPosRow(Math.max(0, posRow - step));
          handled = true;
        }
        break;
      case "ArrowDown":
        if (!lockNavigation) {
          setPosRow(Math.min(shapeRows - 1, posRow + step));
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
          setPosCol(Math.min(shapeCols - 1, posCol + step));
          handled = true;
        }
        break;
      case "r":
      case "R":
        if (!lockView) {
          resetDpView();
          resetViView();
          handled = true;
        }
        break;
      case "z":
      case "Z":
        if (!lockSpots) {
          setSpotUndoRequest(true);
          handled = true;
        }
        break;
    }

    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, [isTypingTarget, lockNavigation, lockSpots, lockView, posRow, posCol, shapeRows, shapeCols]);

  // JSX
  const canvasBox = {
    position: "relative" as const,
    border: `1px solid ${themeColors.border}`,
    overflow: "hidden",
    width: canvasSize,
    height: canvasSize,
    bgcolor: "#000",
  };

  return (
    <Box
      ref={rootRef}
      className="showdiffraction-root"
      sx={{ p: `${SPACING.LG}px`, bgcolor: themeColors.bg, color: themeColors.text, outline: "none" }}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
    >
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.SM}px` }}>
        <Stack direction="row" alignItems="center" spacing={`${SPACING.XS}px`}>
          <Typography sx={{ fontSize: 13, fontWeight: 600 }}>{title || "Diffraction"}</Typography>
          <InfoTooltip theme={themeInfo.theme} text={
            <KeyboardShortcuts items={[
              ["Click", "Add spot on DP"],
              ["← → ↑ ↓", "Navigate scan position"],
              ["Shift+Arrow", "Move 10 steps"],
              ["Scroll", "Zoom in/out"],
              ["Shift+Drag", "Pan"],
              ["R", "Reset zoom/pan"],
              ["Z", "Undo last spot"],
              ["Double-click", "Reset view"],
            ]} />
          } />
        </Stack>
        <Stack direction="row" spacing={`${SPACING.XS}px`}>
          {!hideExport && (
            <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={handleCopyDP}>
              COPY
            </Button>
          )}
          {!hideExport && (
            <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={(e) => setDpExportAnchor(e.currentTarget)}>
              EXPORT
            </Button>
          )}
          {!hideExport && (
            <Menu anchorEl={dpExportAnchor} open={Boolean(dpExportAnchor)} onClose={() => setDpExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
              <MenuItem disabled={lockExport} onClick={() => handleExportFigure(true)} sx={{ fontSize: 12 }}>PDF + colorbar</MenuItem>
              <MenuItem disabled={lockExport} onClick={() => handleExportFigure(false)} sx={{ fontSize: 12 }}>PDF</MenuItem>
              <MenuItem disabled={lockExport} onClick={handleExportPng} sx={{ fontSize: 12 }}>PNG</MenuItem>
            </Menu>
          )}
        </Stack>
      </Stack>

      {/* Main panels */}
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* DP Panel */}
        <Box>
          <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mb: `${SPACING.XS}px` }}>
            DP at ({posRow}, {posCol})
            {cursorInfo && <span style={{ marginLeft: 8, color: themeColors.accent }}>
              ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
            </span>}
          </Typography>
          <Box ref={dpContainerRef} sx={canvasBox}>
            <canvas ref={dpCanvasRef} style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, imageRendering: "pixelated" }} />
            <canvas ref={dpUiRef} style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, pointerEvents: "none" }} />
            <canvas ref={dpScaleRef} style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, pointerEvents: "none" }} />
            <canvas
              style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, cursor: "crosshair", opacity: 0 }}
              width={canvasSize} height={canvasSize}
              onMouseDown={handleDpMouseDown}
              onMouseMove={handleDpMouseMove}
              onMouseUp={handleDpMouseUp}
              onMouseLeave={handleDpMouseLeave}
              onWheel={handleDpWheel}
              onDoubleClick={resetDpView}
            />
            {/* Resize handle */}
            {!hideView && (
              <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.2 : 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: lockView ? 0.2 : 1 } }} />
            )}
          </Box>
          {/* DP Stats */}
          {!hideStats && showStats && dpStats && dpStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.25, display: "flex", gap: 2, opacity: lockStats ? 0.6 : 1 }}>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[0])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[1])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[2])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[3])}</Box>
              </Typography>
            </Box>
          )}
        </Box>

        {/* VI Panel */}
        {!is2d && (
        <Box>
          <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mb: `${SPACING.XS}px` }}>
            Virtual Image (BF)
          </Typography>
          <Box ref={viContainerRef} sx={canvasBox}>
            <canvas ref={viCanvasRef} style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, imageRendering: "pixelated" }} />
            <canvas ref={viUiRef} style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, pointerEvents: "none" }} />
            <canvas ref={viScaleRef} style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, pointerEvents: "none" }} />
            <canvas
              style={{ position: "absolute", top: 0, left: 0, width: canvasSize, height: canvasSize, cursor: "crosshair", opacity: 0 }}
              width={canvasSize} height={canvasSize}
              onMouseDown={handleViMouseDown}
              onMouseMove={handleViMouseMove}
              onMouseUp={handleViMouseUp}
              onMouseLeave={() => { viIsDragging.current = false; }}
              onWheel={handleViWheel}
              onDoubleClick={resetViView}
            />
            {/* Resize handle */}
            {!hideView && (
              <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.2 : 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: lockView ? 0.2 : 1 } }} />
            )}
          </Box>
          {/* VI Stats */}
          {!hideStats && showStats && viStats && viStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.25, display: "flex", gap: 2, opacity: lockStats ? 0.6 : 1 }}>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[0])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[1])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[2])}</Box>
              </Typography>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[3])}</Box>
              </Typography>
            </Box>
          )}
        </Box>
        )}
      </Stack>

      {/* Radial I(g) profile */}
      {showRadial && (
        <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: radialW }}>
          <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mb: `${SPACING.XS}px` }}>
            Radial I(g)
          </Typography>
          <canvas
            ref={radialCanvasRef}
            style={{ width: radialW, height: radialH, cursor: "crosshair", border: `1px solid ${themeColors.border}`, display: "block" }}
            onClick={handleRadialClick}
          />
        </Box>
      )}

      {/* Spots Table */}
      {!hideSpots && (
        <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: canvasSize * 2 + SPACING.LG }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px` }}>
            <Typography sx={{ ...typography.label, color: themeColors.text }}>
              Spots ({spots ? spots.length : 0})
            </Typography>
            <Stack direction="row" spacing={`${SPACING.XS}px`}>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={lockSpots}
                onClick={() => setDetectRequest(20)}
                title="Auto-detect spots"
              >
                SPOTS
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={lockSpots}
                onClick={() => setDetectRingsRequest(8)}
                title="Auto-detect rings"
              >
                RINGS
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={lockExport || !spots || spots.length === 0}
                onClick={() => exportMeasurements("csv")}
              >
                CSV
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={lockExport || !spots || spots.length === 0}
                onClick={() => exportMeasurements("json")}
              >
                JSON
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={lockSpots || !spots || spots.length === 0}
                onClick={() => setSpotUndoRequest(true)}
              >
                UNDO
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={lockSpots || !spots || spots.length === 0}
                onClick={() => setSpotClearRequest(true)}
              >
                CLEAR
              </Button>
            </Stack>
          </Stack>
          {spots && spots.length > 0 && (
            <Box sx={{ maxHeight: 220, overflow: "auto", border: `1px solid ${themeColors.border}` }}>
              <table style={{ width: "100%", fontSize: 10, fontFamily: "monospace", borderCollapse: "collapse", color: themeColors.text }}>
                <thead>
                  <tr style={{ borderBottom: `1px solid ${themeColors.border}`, textAlign: "left" }}>
                    <th style={{ padding: "2px 4px" }} title="reference spot">ref</th>
                    <th style={{ padding: "2px 4px" }}>#</th>
                    <th style={{ padding: "2px 6px" }}>d (Å)</th>
                    <th style={{ padding: "2px 6px" }} title="|g| = 1/d in 1/Å · 1/nm">|g| (1/Å·1/nm)</th>
                    <th style={{ padding: "2px 6px" }} title="angle vs reference spot">∠ (°)</th>
                    <th style={{ padding: "2px 6px" }} title="Gaussian fit R²">fit</th>
                    <th style={{ padding: "2px 6px" }}>I</th>
                    <th style={{ padding: "2px 6px" }}>hkl</th>
                    <th style={{ padding: "2px 6px" }}>note</th>
                    <th style={{ padding: "2px 4px" }}></th>
                  </tr>
                </thead>
                <tbody>
                  {spots.map((spot: SpotDict, i: number) => {
                    const refId = referenceSpotId >= 0 ? referenceSpotId : spots[0].id;
                    const color = spotColorAt(i);
                    const dStr = spot.d_spacing != null
                      ? (spot.d_spacing_err ? `${spot.d_spacing.toFixed(3)}±${spot.d_spacing_err.toFixed(3)}` : spot.d_spacing.toFixed(3))
                      : "—";
                    const gStr = spot.g_magnitude != null
                      ? `${spot.g_magnitude.toFixed(4)}·${(spot.g_magnitude * 10).toFixed(3)}`
                      : `${spot.r_pixels.toFixed(1)} px`;
                    const aStr = spot.angle_deg != null
                      ? (spot.angle_deg_err ? `${spot.angle_deg.toFixed(1)}±${spot.angle_deg_err.toFixed(1)}` : spot.angle_deg.toFixed(1))
                      : "—";
                    return (
                      <tr key={spot.id} style={{ borderBottom: `1px solid ${themeColors.border}22` }}>
                        <td style={{ padding: "2px 4px", textAlign: "center" }}>
                          <input
                            type="radio" name="ref-spot" checked={spot.id === refId}
                            disabled={lockSpots}
                            onChange={() => setReferenceSpotId(spot.id)}
                            style={{ accentColor: color, cursor: "pointer" }}
                          />
                        </td>
                        <td style={{ padding: "2px 4px", color, fontWeight: "bold" }}>{spot.id}</td>
                        <td style={{ padding: "2px 6px" }}>{dStr}</td>
                        <td style={{ padding: "2px 6px" }}>{gStr}</td>
                        <td style={{ padding: "2px 6px" }}>{aStr}</td>
                        <td style={{ padding: "2px 6px" }}>{spot.fit_quality != null ? spot.fit_quality.toFixed(2) : "—"}</td>
                        <td style={{ padding: "2px 6px" }}>{formatNumber(spot.intensity)}</td>
                        <td style={{ padding: "1px 4px", whiteSpace: "nowrap" }}>
                          <input
                            type="text" value={spot.hkl ?? ""} placeholder="(hkl)"
                            disabled={lockSpots}
                            onChange={(e) => updateSpotField(spot.id, "hkl", e.target.value)}
                            style={{ width: 52, fontSize: 10, fontFamily: "monospace", background: "transparent", color: themeColors.text, border: `1px solid ${themeColors.border}55` }}
                          />
                        </td>
                        <td style={{ padding: "1px 4px" }}>
                          <input
                            type="text" value={spot.note ?? ""} placeholder="…"
                            disabled={lockSpots}
                            onChange={(e) => updateSpotField(spot.id, "note", e.target.value)}
                            style={{ width: 64, fontSize: 10, fontFamily: "monospace", background: "transparent", color: themeColors.text, border: `1px solid ${themeColors.border}55` }}
                          />
                        </td>
                        <td style={{ padding: "1px 4px", textAlign: "center" }}>
                          <span
                            onClick={() => { if (!lockSpots) setSpotRemoveRequest(spot.id); }}
                            title="Delete this spot"
                            style={{ cursor: "pointer", color: themeColors.textMuted, fontWeight: "bold", padding: "0 3px" }}
                          >×</span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Box>
          )}
        </Box>
      )}

      {/* Rings Table */}
      {!hideSpots && rings && rings.length > 0 && (
        <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: radialW }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px` }}>
            <Typography sx={{ ...typography.label, color: themeColors.text }}>
              Rings ({rings.length})
            </Typography>
            <Stack direction="row" spacing={`${SPACING.XS}px`}>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockSpots} onClick={() => setRingUndoRequest(true)}>UNDO</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockSpots} onClick={() => setRingClearRequest(true)}>CLEAR</Button>
            </Stack>
          </Stack>
          <Box sx={{ maxHeight: 160, overflow: "auto", border: `1px solid ${themeColors.border}` }}>
            <table style={{ width: "100%", fontSize: 10, fontFamily: "monospace", borderCollapse: "collapse", color: themeColors.text }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${themeColors.border}`, textAlign: "left" }}>
                  <th style={{ padding: "2px 6px" }}>#</th>
                  <th style={{ padding: "2px 6px" }}>radius (px)</th>
                  <th style={{ padding: "2px 6px" }}>d (Å)</th>
                  <th style={{ padding: "2px 6px" }}>|g| (1/Å)</th>
                  <th style={{ padding: "2px 6px" }}>I</th>
                  <th style={{ padding: "2px 4px" }}></th>
                </tr>
              </thead>
              <tbody>
                {rings.map((ring: RingDict) => (
                  <tr key={ring.id} style={{ borderBottom: `1px solid ${themeColors.border}22` }}>
                    <td style={{ padding: "2px 6px", color: themeColors.accent }}>{ring.id}</td>
                    <td style={{ padding: "2px 6px" }}>{ring.radius_px.toFixed(1)}</td>
                    <td style={{ padding: "2px 6px" }}>{ring.d_spacing != null ? ring.d_spacing.toFixed(3) : "—"}</td>
                    <td style={{ padding: "2px 6px" }}>{ring.g_magnitude != null ? ring.g_magnitude.toFixed(4) : "—"}</td>
                    <td style={{ padding: "2px 6px" }}>{formatNumber(ring.intensity)}</td>
                    <td style={{ padding: "1px 4px", textAlign: "center" }}>
                      <span
                        onClick={() => { if (!lockSpots) setRingRemoveRequest(ring.id); }}
                        title="Delete this ring"
                        style={{ cursor: "pointer", color: themeColors.textMuted, fontWeight: "bold", padding: "0 3px" }}
                      >×</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Box>
        </Box>
      )}

      {/* Controls */}
      {showControls && (
        <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: canvasSize * 2 + SPACING.LG }}>
          <Stack direction="row" spacing={`${SPACING.LG}px`} sx={{ flexWrap: "wrap" }}>
            {/* Center mode */}
            <Box sx={controlRow}>
              <Typography sx={typography.label}>Center:</Typography>
              <Select
                size="small" value={centerMode}
                onChange={(e) => setCenterMode(String(e.target.value))}
                sx={{ ...themedSelect, minWidth: 120 }}
                MenuProps={themedMenuProps}
              >
                <MenuItem value="auto" sx={{ fontSize: 10 }}>Auto</MenuItem>
                <MenuItem value="midpoint" sx={{ fontSize: 10 }}>Midpoint (2pt)</MenuItem>
                <MenuItem value="ring" sx={{ fontSize: 10 }}>Ring (3pt)</MenuItem>
              </Select>
              {(centerMode === "midpoint" || centerMode === "ring") && (
                <Typography sx={{ ...typography.value, color: themeColors.accent }}>
                  pick {centerPickPoints.length}/{centerMode === "midpoint" ? 2 : 3}
                </Typography>
              )}
            </Box>

            {/* Radial I(g) toggle */}
            <Box sx={controlRow}>
              <Typography sx={typography.label}>I(g):</Typography>
              <Switch size="small" checked={showRadial} onChange={(_, v) => setShowRadial(v)} sx={switchStyles.small} />
            </Box>

            {/* Spot refine */}
            {!hideSpots && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>Refine:</Typography>
                <Switch
                  size="small" checked={spotRefine}
                  onChange={(_, v) => { if (!lockSpots) setSpotRefine(v); }}
                  sx={switchStyles.small}
                  disabled={lockSpots}
                />
                <Typography sx={{ ...typography.label, ml: 1 }}>Snap:</Typography>
                <Switch
                  size="small" checked={snapEnabled}
                  onChange={(_, v) => { if (!lockSpots) setSnapEnabled(v); }}
                  sx={switchStyles.small}
                  disabled={lockSpots || spotRefine}
                />
                <Typography sx={typography.label}>r:</Typography>
                <Typography sx={typography.value}>{snapRadius}</Typography>
              </Box>
            )}

            {/* DP Colormap */}
            {!hideDisplay && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>DP:</Typography>
                <Select
                  size="small" value={dpColormap}
                  onChange={(e) => { if (!lockDisplay) setDpColormap(e.target.value); }}
                  sx={themedSelect}
                  MenuProps={themedMenuProps}
                  disabled={lockDisplay}
                >
                  {COLORMAP_NAMES.map(n => <MenuItem key={n} value={n} sx={{ fontSize: 10 }}>{n}</MenuItem>)}
                </Select>
              </Box>
            )}

            {/* Scale mode + invert */}
            {!hideDisplay && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>Scale:</Typography>
                <Select
                  size="small" value={dpScaleMode}
                  onChange={(e) => { if (!lockDisplay) setDpScaleMode(e.target.value); }}
                  sx={{ ...themedSelect, minWidth: 60 }}
                  MenuProps={themedMenuProps}
                  disabled={lockDisplay}
                >
                  <MenuItem value="linear" sx={{ fontSize: 10 }}>Linear</MenuItem>
                  <MenuItem value="log" sx={{ fontSize: 10 }}>Log</MenuItem>
                  <MenuItem value="sqrt" sx={{ fontSize: 10 }}>Sqrt</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, ml: 1 }}>Invert:</Typography>
                <Switch
                  size="small" checked={dpInvert}
                  onChange={(_, v) => { if (!lockDisplay) setDpInvert(v); }}
                  sx={switchStyles.small}
                  disabled={lockDisplay}
                />
              </Box>
            )}

            {/* DP Histogram */}
            {!hideHistogram && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>DP:</Typography>
                <Histogram
                  data={dpHistData}
                  vminPct={dpVminPct}
                  vmaxPct={dpVmaxPct}
                  onRangeChange={(min, max) => { if (!lockHistogram) { setDpVminPct(min); setDpVmaxPct(max); } }}
                  theme={themeInfo.theme}
                />
              </Box>
            )}

            {/* VI Colormap */}
            {!hideDisplay && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>VI:</Typography>
                <Select
                  size="small" value={viColormap}
                  onChange={(e) => { if (!lockDisplay) setViColormap(String(e.target.value)); }}
                  sx={{ ...themedSelect, minWidth: 65 }}
                  MenuProps={themedMenuProps}
                  disabled={lockDisplay}
                >
                  {COLORMAP_NAMES.map(n => <MenuItem key={n} value={n} sx={{ fontSize: 10 }}>{n}</MenuItem>)}
                </Select>
              </Box>
            )}

            {/* VI Histogram */}
            {!hideHistogram && (
              <Box sx={controlRow}>
                <Typography sx={typography.label}>VI:</Typography>
                <Histogram
                  data={viHistData}
                  vminPct={viVminPct}
                  vmaxPct={viVmaxPct}
                  onRangeChange={(min, max) => { if (!lockHistogram) { setViVminPct(min); setViVmaxPct(max); } }}
                  theme={themeInfo.theme}
                />
              </Box>
            )}
          </Stack>

          {/* Navigation info */}
          {!hideNavigation && !is2d && (
            <Box sx={{ ...controlRow, mt: `${SPACING.XS}px` }}>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Scan: ({posRow}, {posCol}) / ({shapeRows}×{shapeCols})
              </Typography>
            </Box>
          )}

          {/* Calibration */}
          <Box sx={{ ...controlRow, mt: `${SPACING.XS}px` }}>
            <Typography sx={typography.label}>Calibrate d (Å):</Typography>
            <input
              type="number" value={dKnown}
              onChange={(e) => setDKnown(e.target.value)}
              placeholder="2.355"
              style={{ width: 64, fontSize: 10, padding: "2px 4px", background: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` }}
            />
            <Button
              size="small" sx={{ ...compactButton, color: themeColors.accent }}
              disabled={!spots || spots.length === 0 || !(parseFloat(dKnown) > 0)}
              onClick={() => { const d = parseFloat(dKnown); const s = spots[spots.length - 1]; if (d > 0 && s) setCalibrateFromSpotRequest([s.row, s.col, d]); }}
            >FROM SPOT</Button>
            <Button
              size="small" sx={{ ...compactButton, color: themeColors.accent }}
              disabled={!rings || rings.length === 0 || !(parseFloat(dKnown) > 0)}
              onClick={() => { const d = parseFloat(dKnown); const r = rings[rings.length - 1]; if (d > 0 && r) setCalibrateFromRingRequest([r.radius_px, d]); }}
            >FROM RING</Button>
            {!hideView && (
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={lockView}
                onClick={zoomToCenter}
                title="Zoom to the diffraction center"
              >CENTER VIEW</Button>
            )}
          </Box>

          {/* Center info + calibration provenance */}
          <Box sx={{ ...controlRow, mt: `${SPACING.XS}px` }}>
            <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
              Center: ({centerRow.toFixed(1)}, {centerCol.toFixed(1)})  BF r={bfRadius.toFixed(1)}
              {kCalibrated && <span style={{ marginLeft: 8 }}>k={kPixelSize.toFixed(4)} 1/Å/px</span>}
            </Typography>
          </Box>
          {kCalibrated && (
            <Box sx={controlRow}>
              <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                Calib: {calibrationSource}
                {calibrationRefD > 0 && ` (d=${calibrationRefD.toFixed(3)} Å @ r=${calibrationRefRadius.toFixed(1)} px)`}
              </Typography>
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(ShowDiffraction);
