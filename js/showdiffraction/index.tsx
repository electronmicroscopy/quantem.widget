/**
 * ShowDiffraction — interactive d-spacing analysis for a single 2D/3D diffraction pattern.
 *
 * Lean single-panel viewer: one diffraction-pattern (DP) canvas with colormap,
 * scale mode, contrast, center/BF disk, spots, rings, calibration and a frame
 * slider to scrub a 3D stack.
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
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
import { drawScaleBarHiDPI, drawColorbar } from "../figure";
import { extractBytes, extractFloat32, formatNumber, downloadBlob, preserveRestoredWidgetModelsOnSave } from "../format";
import { computeHistogramFromBytes, findDataRange, sliderRange, applyLogScaleInPlace } from "../stats";
import { COLORMAPS, COLORMAP_NAMES, applyColormap } from "../colormaps";

// ============================================================================
// Style tokens
// ============================================================================

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const CANVAS_MIN = 384;
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
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
const sliderStyles = {
  small: { py: 0, "& .MuiSlider-thumb": { width: 10, height: 10 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 } },
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

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
};

// ============================================================================
// Info tooltip + keyboard shortcuts
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
// Contrast histogram with draggable min/max handles
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
// Format a stat value for the readout
// ============================================================================
function formatStat(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) return v.toExponential(2);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

// ============================================================================
// Spot and ring types
// ============================================================================

interface SpotDict {
  id: number;
  row: number;
  col: number;
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
}

// Spot colors, shared by table rows and canvas overlay.
const PICK_COLORS = [
  "#ff4d4f", "#40a9ff", "#73d13d", "#ffa940",
  "#9254de", "#13c2c2", "#f759ab", "#bae637",
];
const spotColorAt = (index: number) => PICK_COLORS[((index % PICK_COLORS.length) + PICK_COLORS.length) % PICK_COLORS.length];

// ============================================================================
// Main component
// ============================================================================

function ShowDiffraction() {
  // Force a light background for offline/export HTML renders.
  const [offline] = useModelState<boolean>("offline");
  const { themeInfo, colors: themeColors } = useTheme(offline);
  const rootRef = React.useRef<HTMLDivElement>(null);

  const model = useModel();
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);

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
  // Bordered control group.
  const controlBox = { ...controlRow, border: `1px solid ${themeColors.border}`, borderRadius: "4px", bgcolor: themeColors.controlBg };

  // Model state
  const [title] = useModelState<string>("title");
  const [detRows] = useModelState<number>("det_rows");
  const [detCols] = useModelState<number>("det_cols");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [offlineFrames] = useModelState<DataView>("offline_frames");
  const [frameIdx, setFrameIdx] = useModelState<number>("frame_idx");
  const [nFrames] = useModelState<number>("n_frames");
  const [centerRow, setCenterRow] = useModelState<number>("center_row");
  const [centerCol, setCenterCol] = useModelState<number>("center_col");
  const [bfRadius] = useModelState<number>("bf_radius");
  const [kPixelSize] = useModelState<number>("k_pixel_size");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");
  const [spots] = useModelState<SpotDict[]>("spots");
  const [snapEnabled, setSnapEnabled] = useModelState<boolean>("snap_enabled");
  const [snapRadius] = useModelState<number>("snap_radius");
  const [spotRefine, setSpotRefine] = useModelState<boolean>("spot_refine");
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
  const [dpStats] = useModelState<number[]>("dp_stats");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [panelWidthPx] = useModelState<number>("panel_width_px");

  // Standalone HTML export bridge.
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportPayloadFilename] = useModelState<string>("export_filename");
  const exportCounterRef = React.useRef(0);
  const pendingExportRef = React.useRef<string>("");

  // Center, rings, calibration
  const [centerMode, setCenterMode] = useModelState<string>("center_mode");
  const [rings] = useModelState<RingDict[]>("rings");
  const [calibrationSource] = useModelState<string>("calibration_source");
  const [calibrationRefD] = useModelState<number>("calibration_ref_d");
  const [calibrationRefRadius] = useModelState<number>("calibration_ref_radius");
  const [, setRingUndoRequest] = useModelState<boolean>("_ring_undo_request");
  const [, setRingClearRequest] = useModelState<boolean>("_ring_clear_request");
  const [, setCalibrateFromRingRequest] = useModelState<number[]>("_calibrate_from_ring_request");
  const [, setCalibrateFromSpotRequest] = useModelState<number[]>("_calibrate_from_spot_request");

  // Export spots and rings as CSV or JSON.
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
  const initialCanvasSize = React.useMemo(() => {
    const requested = Number(panelWidthPx);
    return Number.isFinite(requested) && requested > 0 ? Math.max(CANVAS_MIN, Math.round(requested)) : CANVAS_MIN;
  }, [panelWidthPx]);
  const hasResizedCanvasRef = React.useRef(false);
  const [canvasSize, setCanvasSize] = React.useState(initialCanvasSize);
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeCanvasStart, setResizeCanvasStart] = React.useState<{ x: number; y: number; size: number } | null>(null);
  const [dpZoom, setDpZoom] = React.useState(1);
  const [dpPanX, setDpPanX] = React.useState(0);
  const [dpPanY, setDpPanY] = React.useState(0);
  const [dpHistData, setDpHistData] = React.useState<Float32Array | null>(null);
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);
  const [dpExportAnchor, setDpExportAnchor] = React.useState<HTMLElement | null>(null);
  const [dKnown, setDKnown] = React.useState("");

  React.useEffect(() => {
    if (!hasResizedCanvasRef.current) {
      setCanvasSize(initialCanvasSize);
    }
  }, [initialCanvasSize]);

  // Local frame index for smooth scrubbing; commit on release.
  const [localFrame, setLocalFrame] = React.useState(frameIdx);
  React.useEffect(() => { setLocalFrame(frameIdx); }, [frameIdx]);

  // Zoom to the diffraction center.
  const zoomToCenter = React.useCallback(() => {
    const Z = 2.5;
    setDpZoom(Z);
    setDpPanX(canvasSize * Z * (0.5 - centerCol / Math.max(detCols, 1)));
    setDpPanY(canvasSize * Z * (0.5 - centerRow / Math.max(detRows, 1)));
  }, [canvasSize, centerRow, centerCol, detRows, detCols]);

  const dpCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const dpUiRef = React.useRef<HTMLCanvasElement>(null);
  const dpScaleRef = React.useRef<HTMLCanvasElement>(null);
  const dpOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const [dpVersion, setDpVersion] = React.useState(0);
  const dpVminRef = React.useRef(0);
  const dpVmaxRef = React.useRef(1);

  // Drag the corner handle to resize the canvas.
  const handleCanvasResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    hasResizedCanvasRef.current = true;
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

  // Colormap LUT, reversed when Invert is on.
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

  // Offline bakes the whole stack for client-side scrubbing; live streams one frame.
  const activeFrame = React.useMemo<Float32Array | null>(() => {
    const frameLen = detRows * detCols;
    if (offline && offlineFrames && frameLen > 0
        && offlineFrames.byteLength >= frameLen * 4 * nFrames) {
      const stack = extractFloat32(offlineFrames);
      const idx = Math.max(0, Math.min(frameIdx, nFrames - 1));
      if (stack && stack.length >= frameLen * (idx + 1)) {
        return stack.subarray(idx * frameLen, (idx + 1) * frameLen);
      }
    }
    return extractFloat32(frameBytes, frameLen);
  }, [offline, offlineFrames, frameBytes, frameIdx, nFrames, detRows, detCols]);

  // Render the frame: scale then colormap
  React.useEffect(() => {
    const raw = activeFrame;
    if (!raw || raw.length === 0) return;
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
  }, [activeFrame, dpLut, dpScaleMode, dpVminPct, dpVmaxPct, detRows, detCols]);

  // Draw the rendered frame with zoom and pan
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

  // Overlay: center, spots, rings, colorbar
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

    // Spot markers
    if (spots && spots.length > 0) {
      spots.forEach((spot, i) => {
        const sx = offX + spot.col * scX;
        const sy = offY + spot.row * scY;
        const color = spotColorAt(i);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.font = "bold 10px -apple-system, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(`${spot.id}`, sx + 8, sy - 2);
      });
    }

    // Rings
    if (rings && rings.length > 0) {
      ctx.strokeStyle = themeInfo.theme === "dark" ? "#ffb74d" : "#e65100";
      ctx.lineWidth = 1.2;
      for (const ring of rings) {
        ctx.beginPath();
        ctx.arc(cx, cy, ring.radius_px * scX, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }

    drawColorbar(ctx, cssW, cssW, dpLut, dpVminRef.current, dpVmaxRef.current, dpScaleMode === "log");

    if (dpZoom !== 1) {
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${dpZoom.toFixed(1)}×`, 8, cssW - 8);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }, [dpVersion, dpZoom, dpPanX, dpPanY, canvasSize, detRows, detCols, centerRow, centerCol, bfRadius, spots, rings, dpLut, dpScaleMode, themeInfo.theme]);

  // K-space scale bar on its own canvas.
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

  // Mouse handlers
  const dpIsDragging = React.useRef(false);
  const dpDragStart = React.useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  // Canvas pixel to image (row, col).
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
    // Manual mode: click sets the center instead of adding a spot.
    if (centerMode === "manual") {
      setCenterRow(row);
      setCenterCol(col);
      return;
    }
    setSpotAddRequest([row, col]);
  };

  const handleDpMouseMove = (e: React.MouseEvent) => {
    if (dpIsDragging.current) {
      setDpPanX(dpDragStart.current.panX + (e.clientX - dpDragStart.current.x));
      setDpPanY(dpDragStart.current.panY + (e.clientY - dpDragStart.current.y));
      return;
    }
    if (!activeFrame) return;
    const { row, col } = dpToImage(e);
    const ri = Math.round(row), ci = Math.round(col);
    if (ri >= 0 && ri < detRows && ci >= 0 && ci < detCols) {
      const raw = activeFrame;
      setCursorInfo({ row: ri, col: ci, value: raw[ri * detCols + ci] });
    } else {
      setCursorInfo(null);
    }
  };

  const handleDpMouseUp = () => { dpIsDragging.current = false; };
  const handleDpMouseLeave = () => { dpIsDragging.current = false; setCursorInfo(null); };

  // Scroll to zoom.
  const handleDpWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setDpZoom(z => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * delta)));
  };

  const resetDpView = () => { setDpZoom(1); setDpPanX(0); setDpPanY(0); };

  // Wheel scroll prevention
  const dpContainerRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const prevent = (e: WheelEvent) => e.preventDefault();
    const dp = dpContainerRef.current;
    if (dp) dp.addEventListener("wheel", prevent, { passive: false });
    return () => {
      if (dp) dp.removeEventListener("wheel", prevent);
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

  const handleExportPng = () => {
    setDpExportAnchor(null);
    if (!dpCanvasRef.current) return;
    dpCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "showdiffraction_dp.png"); }, "image/png");
  };

  // Request an HTML export; the effect below downloads the payload.
  const handleExportHtml = () => {
    setDpExportAnchor(null);
    exportCounterRef.current += 1;
    const id = `html-${exportCounterRef.current}`;
    const slug = (title || "showdiffraction")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "") || "showdiffraction";
    const filename = `${slug}_${nFrames}x${detRows}x${detCols}.html`;
    pendingExportRef.current = id;
    setExportRequest(JSON.stringify({ mode: "single", download: true, id, filename }));
  };

  // Download the HTML payload once it arrives.
  React.useEffect(() => {
    if (!exportPayloadId || exportPayloadId !== pendingExportRef.current) return;
    const bytes = extractBytes(exportPayload);
    if (bytes.length === 0) return;
    const payload = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
      ? bytes
      : bytes.slice();
    const filename = exportPayloadFilename || "showdiffraction.html";
    downloadBlob(new Blob([payload as BlobPart], { type: "text/html;charset=utf-8" }), filename);
    pendingExportRef.current = "";
    setExportRequest(JSON.stringify({ mode: "clear" }));
  }, [exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest]);

  // Keyboard
  // Skip shortcuts while typing in a field.
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
      case "ArrowLeft":
        if (nFrames > 1) { setFrameIdx(Math.max(0, frameIdx - step)); handled = true; }
        break;
      case "ArrowRight":
        if (nFrames > 1) { setFrameIdx(Math.min(nFrames - 1, frameIdx + step)); handled = true; }
        break;
      case "r":
      case "R":
        resetDpView();
        handled = true;
        break;
      case "z":
      case "Z":
        setSpotUndoRequest(true);
        handled = true;
        break;
    }

    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, [isTypingTarget, frameIdx, nFrames, setFrameIdx, setSpotUndoRequest]);

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
              ["Click", "Add spot (or set center in Manual mode)"],
              ["← →", "Previous / next frame"],
              ["Shift+Arrow", "Move 10 frames"],
              ["Scroll", "Zoom in/out"],
              ["Shift+Drag", "Pan"],
              ["R", "Reset zoom/pan"],
              ["Z", "Undo last spot"],
              ["Double-click", "Reset view"],
            ]} />
          } />
        </Stack>
        <Stack direction="row" spacing={`${SPACING.XS}px`}>
          <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={handleCopyDP}>
            COPY
          </Button>
          <Button
            size="small"
            sx={{ ...compactButton, color: themeColors.accent }}
            onClick={(e) => setDpExportAnchor(e.currentTarget)}
            title={exportStatus || "Export a PNG or a standalone HTML viewer"}
          >
            EXPORT
          </Button>
          <Menu anchorEl={dpExportAnchor} open={Boolean(dpExportAnchor)} onClose={() => setDpExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
            <MenuItem onClick={handleExportPng} sx={{ fontSize: 12 }}>PNG</MenuItem>
            {exportEnabled && <MenuItem onClick={handleExportHtml} sx={{ fontSize: 12 }}>HTML</MenuItem>}
          </Menu>
          {exportEnabled && exportStatus && (
            <Typography
              sx={{
                ...typography.value,
                maxWidth: 160,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                color: exportStatus.startsWith("Export failed") ? "#d32f2f" : themeColors.textMuted,
              }}
              title={exportStatus}
            >
              {exportStatus}
            </Typography>
          )}
        </Stack>
      </Stack>

      {/* DP panel */}
      <Box sx={{ display: "flex", justifyContent: "flex-start" }}>
        <Box>
          {/* Toolbar: general controls above the display */}
          {showControls && (
            <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} useFlexGap sx={{ mb: `${SPACING.XS}px`, minHeight: 28, flexWrap: "wrap", rowGap: `${SPACING.XS}px`, maxWidth: canvasSize, px: 1, py: 0.5, border: `1px solid ${themeColors.border}`, borderRadius: "4px", bgcolor: themeColors.controlBg }}>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={() => setDetectRequest(20)} title="Auto-detect Bragg spots">SPOTS</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={() => setDetectRingsRequest(8)} title="Auto-detect Debye–Scherrer rings">RINGS</Button>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>Center:</Typography>
              <Select size="small" value={centerMode} onChange={(e) => setCenterMode(String(e.target.value))} sx={{ ...themedSelect, minWidth: 80 }} MenuProps={themedMenuProps}>
                <MenuItem value="auto" sx={{ fontSize: 10 }}>Auto</MenuItem>
                <MenuItem value="manual" sx={{ fontSize: 10 }}>Manual</MenuItem>
              </Select>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>Cmap:</Typography>
              <Select size="small" value={dpColormap} onChange={(e) => setDpColormap(e.target.value)} sx={themedSelect} MenuProps={themedMenuProps}>
                {COLORMAP_NAMES.map(n => <MenuItem key={n} value={n} sx={{ fontSize: 10 }}>{n}</MenuItem>)}
              </Select>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
              <Select size="small" value={dpScaleMode} onChange={(e) => setDpScaleMode(e.target.value)} sx={{ ...themedSelect, minWidth: 60 }} MenuProps={themedMenuProps}>
                <MenuItem value="linear" sx={{ fontSize: 10 }}>Linear</MenuItem>
                <MenuItem value="log" sx={{ fontSize: 10 }}>Log</MenuItem>
                <MenuItem value="sqrt" sx={{ fontSize: 10 }}>Sqrt</MenuItem>
              </Select>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>Invert:</Typography>
              <Switch size="small" checked={dpInvert} onChange={(_, v) => setDpInvert(v)} sx={switchStyles.small} />
              {centerMode === "manual" && (
                <Typography sx={{ ...typography.value, color: themeColors.accent }}>click to set</Typography>
              )}
            </Stack>
          )}
          <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mb: `${SPACING.XS}px` }}>
            {nFrames > 1 ? `Frame ${localFrame + 1} / ${nFrames}` : "Diffraction"}
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
            <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
          </Box>

          {/* Frame slider (3D stacks only) */}
          {nFrames > 1 && (
            <Box sx={{ ...controlRow, width: canvasSize }}>
              <Typography sx={typography.label}>Frame:</Typography>
              <Slider
                value={localFrame}
                min={0} max={nFrames - 1} step={1} size="small"
                valueLabelDisplay="auto" valueLabelFormat={(v) => `${v + 1}`}
                onChange={(_, v) => setLocalFrame(v as number)}
                onChangeCommitted={(_, v) => setFrameIdx(v as number)}
                aria-label={`Frame ${localFrame + 1} of ${nFrames}`}
                sx={{ ...sliderStyles.small, flex: 1, minWidth: 40, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }}
              />
              <Typography sx={typography.value}>{localFrame + 1}/{nFrames}</Typography>
            </Box>
          )}

          {/* DP Stats */}
          {showStats && dpStats && dpStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.25, display: "flex", gap: 2 }}>
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
      </Box>

      {/* Spots Table */}
      <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: canvasSize }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px` }}>
            <Typography sx={{ ...typography.label, color: themeColors.text }}>
              Spots ({spots ? spots.length : 0})
            </Typography>
            <Stack direction="row" spacing={`${SPACING.XS}px`} sx={{ p: 0.25, border: `1px solid ${themeColors.border}`, borderRadius: "4px", bgcolor: themeColors.controlBg }}>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={!spots || spots.length === 0}
                onClick={() => exportMeasurements("csv")}
              >
                CSV
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={!spots || spots.length === 0}
                onClick={() => exportMeasurements("json")}
              >
                JSON
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={!spots || spots.length === 0}
                onClick={() => setSpotUndoRequest(true)}
              >
                UNDO
              </Button>
              <Button
                size="small" sx={{ ...compactButton, color: themeColors.accent }}
                disabled={!spots || spots.length === 0}
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
                    <th style={{ padding: "2px 4px" }}>#</th>
                    <th style={{ padding: "2px 6px" }}>d (Å)</th>
                    <th style={{ padding: "2px 6px" }} title="|g| = 1/d in 1/Å · 1/nm">|g| (1/Å·1/nm)</th>
                    <th style={{ padding: "2px 6px" }} title="angle vs reference spot">∠ (°)</th>
                    <th style={{ padding: "2px 6px" }} title="Gaussian fit R²">fit</th>
                    <th style={{ padding: "2px 6px" }}>I</th>
                    <th style={{ padding: "2px 4px" }}></th>
                  </tr>
                </thead>
                <tbody>
                  {spots.map((spot: SpotDict, i: number) => {
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
                        <td style={{ padding: "2px 4px", color, fontWeight: "bold" }}>{spot.id}</td>
                        <td style={{ padding: "2px 6px" }}>{dStr}</td>
                        <td style={{ padding: "2px 6px" }}>{gStr}</td>
                        <td style={{ padding: "2px 6px" }}>{aStr}</td>
                        <td style={{ padding: "2px 6px" }}>{spot.fit_quality != null ? spot.fit_quality.toFixed(2) : "—"}</td>
                        <td style={{ padding: "2px 6px" }}>{formatNumber(spot.intensity)}</td>
                        <td style={{ padding: "1px 4px", textAlign: "center" }}>
                          <span
                            onClick={() => setSpotRemoveRequest(spot.id)}
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

      {/* Rings Table */}
      {rings && rings.length > 0 && (
        <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: canvasSize }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px` }}>
            <Typography sx={{ ...typography.label, color: themeColors.text }}>
              Rings ({rings.length})
            </Typography>
            <Stack direction="row" spacing={`${SPACING.XS}px`} sx={{ p: 0.25, border: `1px solid ${themeColors.border}`, borderRadius: "4px", bgcolor: themeColors.controlBg }}>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={() => setRingUndoRequest(true)}>UNDO</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={() => setRingClearRequest(true)}>CLEAR</Button>
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
                        onClick={() => setRingRemoveRequest(ring.id)}
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

      {/* Fine controls below the display */}
      {showControls && (
        <Box sx={{ mt: `${SPACING.MD}px`, maxWidth: canvasSize }}>
          <Stack direction="row" spacing={`${SPACING.LG}px`} sx={{ flexWrap: "wrap" }}>
            <Box sx={controlBox}>
              <Typography sx={typography.label}>Refine:</Typography>
              <Switch
                size="small" checked={spotRefine}
                onChange={(_, v) => setSpotRefine(v)}
                sx={switchStyles.small}
              />
              <Typography sx={{ ...typography.label, ml: 1 }}>Snap:</Typography>
              <Switch
                size="small" checked={snapEnabled}
                onChange={(_, v) => setSnapEnabled(v)}
                sx={switchStyles.small}
                disabled={spotRefine}
              />
              <Typography sx={typography.label}>r:</Typography>
              <Typography sx={typography.value}>{snapRadius}</Typography>
            </Box>

            <Box sx={controlBox}>
              <Histogram
                data={dpHistData}
                vminPct={dpVminPct}
                vmaxPct={dpVmaxPct}
                onRangeChange={(min, max) => { setDpVminPct(min); setDpVmaxPct(max); }}
                theme={themeInfo.theme}
              />
            </Box>
          </Stack>

          <Box sx={{ ...controlBox, mt: `${SPACING.XS}px` }}>
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
            <Button
              size="small" sx={{ ...compactButton, color: themeColors.accent }}
              onClick={zoomToCenter}
              title="Zoom to the diffraction center"
            >CENTER VIEW</Button>
          </Box>

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
