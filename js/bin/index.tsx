import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { extractFloat32, formatNumber, downloadBlob, downloadDataView } from "../format";
import { applyLogScale, findDataRange, percentileClip, sliderRange } from "../stats";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen } from "../colormaps";
import { computeHistogramFromBytes } from "../histogram";
import { roundToNiceValue, formatScaleLabel, canvasToPDF } from "../scalebar";
import { getWebGPUFFT, WebGPUFFT, fft2d, fftshift, computeMagnitude, autoEnhanceFFT } from "../webgpu-fft";
import { computeToolVisibility } from "../tool-parity";
import { ControlCustomizer } from "../control-customizer";
import "./bin.css";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const CANVAS_SIZE = 250;
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const CLICK_THRESHOLD = 3; // px — below this, mousedown+up = click (not drag)
const ROI_HIT_PX = 8; // CSS pixels — hit detection radius for ROI circles

const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  section: { fontSize: 11, fontWeight: "bold" as const },
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
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};

const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function optionsForFactor(maxValue: number): number[] {
  const maxSafe = Math.max(1, Math.floor(maxValue));
  const powers = [1, 2, 4, 8, 16, 32, 64, 128];
  return powers.filter((v) => v <= maxSafe);
}

function stepOption(options: number[], current: number, delta: -1 | 1): number {
  if (!options.length) return current;
  const sorted = [...options].sort((a, b) => a - b);
  const idx = Math.max(0, sorted.indexOf(current));
  if (delta < 0) return sorted[Math.max(0, idx - 1)];
  return sorted[Math.min(sorted.length - 1, idx + 1)];
}

function isEditableElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  if (target.isContentEditable) return true;
  if (target.getAttribute("role") === "textbox") return true;
  return tag === "input" || tag === "textarea" || tag === "select";
}

function statsLine(stats: number[]): string {
  if (!stats || stats.length < 4) return "";
  const [mean, min, max, std] = stats;
  return `mean ${formatNumber(mean)} | min ${formatNumber(min)} | max ${formatNumber(max)} | std ${formatNumber(std)}`;
}

// ---------------------------------------------------------------------------
// InfoTooltip
// ---------------------------------------------------------------------------

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
            maxWidth: 320,
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

// ---------------------------------------------------------------------------
// KeyboardShortcuts — two-column shortcut table (Show3D pattern)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Histogram — interactive dual-slider histogram (Show3D pattern)
// ---------------------------------------------------------------------------

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
  width = 110, height = 40, theme = "dark",
  dataMin = 0, dataMax = 1,
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
      ctx.fillStyle = (i >= vminBin && i <= vmaxBin) ? colors.barActive : colors.barInactive;
      ctx.fillRect(i * barWidth + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);

  const fmtVal = (pct: number) => {
    const val = dataMin + (pct / 100) * (dataMax - dataMin);
    return val >= 1000 ? val.toExponential(1) : val.toFixed(1);
  };

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
        valueLabelFormat={(pct) => fmtVal(pct)}
        sx={{
          width, py: 0,
          "& .MuiSlider-thumb": { width: 8, height: 8 },
          "& .MuiSlider-rail": { height: 2 },
          "& .MuiSlider-track": { height: 2 },
          "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
        }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{fmtVal(vminPct)}</Typography>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{fmtVal(vmaxPct)}</Typography>
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// InteractivePanel — reusable zoom/pan canvas panel
// ---------------------------------------------------------------------------

interface PanelProps {
  label: string;
  rows: number;
  cols: number;
  bytes: DataView;
  cmap: string;
  logScale: boolean;
  autoContrast?: boolean;
  vminPct?: number;
  vmaxPct?: number;
  canvasSize: number;
  borderColor: string;
  textColor: string;
  mutedColor: string;
  accentColor: string;
  lockView: boolean;
  stats?: number[];
  hideStats?: boolean;
  pixelSize?: number;
  pixelUnit?: string;
  resetKey?: number;
  overrideData?: Float32Array | null;
  overlayRenderer?: (
    ctx: CanvasRenderingContext2D,
    cssW: number,
    cssH: number,
    zoom: number,
    panX: number,
    panY: number,
  ) => void;
  canvasRef?: React.RefObject<HTMLCanvasElement | null>;
  onResizeStart?: (e: React.MouseEvent) => void;
  // Click-to-position: fires when user clicks (not drags) on the panel
  onImageClick?: (row: number, col: number) => void;
  // Position crosshair marker (drawn on overlay)
  positionMarker?: { row: number; col: number } | null;
  // Custom drag handler — intercepts mouse interaction for ROI drag etc.
  customDragHandler?: {
    onDown: (imgRow: number, imgCol: number) => string | null; // returns drag mode or null
    onMove: (imgRow: number, imgCol: number, mode: string) => void;
    onUp: (mode: string) => void;
    getCursor?: (imgRow: number, imgCol: number) => string | null; // custom cursor
    getHoverHandle?: (imgRow: number, imgCol: number) => string | null; // which handle is hovered
  };
  // Callback when mouse leaves the panel (for clearing hover state)
  onMouseLeavePanel?: () => void;
}

function InteractivePanel({
  label,
  rows,
  cols,
  bytes,
  cmap,
  logScale,
  autoContrast,
  vminPct,
  vmaxPct,
  canvasSize: size,
  borderColor,
  textColor,
  mutedColor,
  accentColor,
  lockView,
  stats,
  hideStats,
  pixelSize,
  pixelUnit,
  resetKey,
  overrideData,
  overlayRenderer,
  canvasRef: externalCanvasRef,
  onResizeStart,
  onImageClick,
  positionMarker,
  customDragHandler,
  onMouseLeavePanel,
}: PanelProps) {
  const internalCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const canvasRef = externalCanvasRef || internalCanvasRef;
  const overlayRef = React.useRef<HTMLCanvasElement>(null);

  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [isDragging, setIsDragging] = React.useState(false);
  const dragStart = React.useRef<{ x: number; y: number; pX: number; pY: number } | null>(null);
  // Click detection — track mousedown position to distinguish click vs drag
  const clickDownRef = React.useRef<{ x: number; y: number } | null>(null);
  // Custom drag mode — when customDragHandler captures the event
  const customDragModeRef = React.useRef<string | null>(null);
  // Custom cursor based on what's under the mouse
  const [customCursor, setCustomCursor] = React.useState<string | null>(null);

  // Reset zoom/pan when resetKey changes (R key pressed)
  React.useEffect(() => {
    if (resetKey === undefined || resetKey === 0) return;
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, [resetKey]);

  const parsed = React.useMemo(() => overrideData || extractFloat32(bytes), [overrideData, bytes]);

  // Compute aspect-correct CSS dimensions
  const cssW = React.useMemo(() => {
    if (rows <= 0 || cols <= 0) return size;
    const aspect = cols / rows;
    return aspect >= 1 ? size : Math.round(size * aspect);
  }, [rows, cols, size]);

  const cssH = React.useMemo(() => {
    if (rows <= 0 || cols <= 0) return size;
    const aspect = cols / rows;
    return aspect >= 1 ? Math.round(size / aspect) : size;
  }, [rows, cols, size]);

  // Render colormapped image
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !parsed || parsed.length === 0) return;
    if (rows <= 0 || cols <= 0) return;
    if (parsed.length !== rows * cols) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const data = logScale ? applyLogScale(parsed) : parsed;
    const range = findDataRange(data);
    let vmin: number, vmax: number;
    if (autoContrast) {
      ({ vmin, vmax } = percentileClip(data, 2, 98));
    } else if (vminPct !== undefined && vmaxPct !== undefined && (vminPct !== 0 || vmaxPct !== 100)) {
      ({ vmin, vmax } = sliderRange(range.min, range.max, vminPct, vmaxPct));
    } else {
      vmin = range.min;
      vmax = range.max;
    }
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(data, cols, rows, lut, vmin, vmax);
    if (!offscreen) return;

    canvas.width = cssW;
    canvas.height = cssH;
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.imageSmoothingEnabled = false;

    if (zoom !== 1 || panX !== 0 || panY !== 0) {
      ctx.save();
      ctx.translate(panX, panY);
      ctx.scale(zoom, zoom);
      ctx.drawImage(offscreen, 0, 0, cols, rows, 0, 0, cssW, cssH);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, cols, rows, 0, 0, cssW, cssH);
    }
  }, [parsed, rows, cols, cmap, logScale, autoContrast, vminPct, vmaxPct, cssW, cssH, zoom, panX, panY]);

  // Render overlay (BF/ADF circles on detector panels + scale bar + position crosshair)
  React.useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;

    overlay.width = cssW * DPR;
    overlay.height = cssH * DPR;
    overlay.style.width = `${cssW}px`;
    overlay.style.height = `${cssH}px`;

    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (overlayRenderer) {
      ctx.save();
      ctx.scale(DPR, DPR);
      overlayRenderer(ctx, cssW, cssH, zoom, panX, panY);
      ctx.restore();
    }

    // Position crosshair marker
    if (positionMarker && rows > 0 && cols > 0) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const scaleX = cssW / cols;
      const scaleY = cssH / rows;
      const sx = (positionMarker.col + 0.5) * scaleX * zoom + panX;
      const sy = (positionMarker.row + 0.5) * scaleY * zoom + panY;
      const armLen = 8;
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(255, 255, 0, 0.9)";
      ctx.shadowColor = "rgba(0, 0, 0, 0.7)";
      ctx.shadowBlur = 2;
      ctx.beginPath();
      ctx.moveTo(sx - armLen, sy); ctx.lineTo(sx + armLen, sy);
      ctx.moveTo(sx, sy - armLen); ctx.lineTo(sx, sy + armLen);
      ctx.stroke();
      ctx.restore();
    }

    // Scale bar (drawn without zoom/pan transform — stays fixed in corner)
    if (pixelSize && pixelSize > 0 && pixelUnit && cols > 0) {
      ctx.save();
      ctx.scale(DPR, DPR);

      const scaleX = cssW / cols;
      const effectiveZoom = zoom * scaleX;
      const targetBarPx = 60;
      const barThickness = 5;
      const fontSize = 16;
      const margin = 12;

      const targetPhysical = (targetBarPx / effectiveZoom) * pixelSize;
      const nicePhysical = roundToNiceValue(targetPhysical);
      const barPx = (nicePhysical / pixelSize) * effectiveZoom;

      const barY = cssH - margin;
      const barX = cssW - barPx - margin;

      ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
      ctx.shadowBlur = 2;
      ctx.shadowOffsetX = 1;
      ctx.shadowOffsetY = 1;

      ctx.fillStyle = "white";
      ctx.fillRect(barX, barY, barPx, barThickness);

      const scaleLabel = formatScaleLabel(nicePhysical, pixelUnit as "Å" | "mrad" | "px");
      ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
      ctx.fillStyle = "white";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(scaleLabel, barX + barPx / 2, barY - 4);

      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${zoom.toFixed(1)}×`, margin, cssH - margin + barThickness);

      ctx.restore();
    }
  }, [cssW, cssH, cols, rows, overlayRenderer, zoom, panX, panY, pixelSize, pixelUnit, positionMarker]);

  // Wheel scroll prevention
  React.useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    overlay.addEventListener("wheel", preventDefault, { passive: false });
    return () => overlay.removeEventListener("wheel", preventDefault);
  }, []);

  // Zoom handler (cursor-centered)
  const handleWheel = React.useCallback(
    (e: React.WheelEvent<HTMLCanvasElement>) => {
      if (lockView) return;
      e.preventDefault();
      const overlay = overlayRef.current;
      if (!overlay) return;
      const rect = overlay.getBoundingClientRect();
      const mouseX = (e.clientX - rect.left);
      const mouseY = (e.clientY - rect.top);
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * zoomFactor));
      const zoomRatio = newZoom / zoom;
      setZoom(newZoom);
      setPanX(mouseX - (mouseX - panX) * zoomRatio);
      setPanY(mouseY - (mouseY - panY) * zoomRatio);
    },
    [lockView, zoom, panX, panY],
  );

  // Helper: screen coords → image coords
  const screenToImage = React.useCallback(
    (clientX: number, clientY: number): { row: number; col: number } | null => {
      if (rows <= 0 || cols <= 0) return null;
      const overlay = overlayRef.current;
      if (!overlay) return null;
      const rect = overlay.getBoundingClientRect();
      const mouseX = clientX - rect.left;
      const mouseY = clientY - rect.top;
      const imgX = (mouseX - panX) / zoom;
      const imgY = (mouseY - panY) / zoom;
      const col = (imgX / cssW) * cols;
      const row = (imgY / cssH) * rows;
      return { row, col };
    },
    [rows, cols, cssW, cssH, zoom, panX, panY],
  );

  // Pan handlers (with click detection and custom drag support)
  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      if (lockView) return;
      clickDownRef.current = { x: e.clientX, y: e.clientY };

      // Check custom drag handler first
      if (customDragHandler) {
        const img = screenToImage(e.clientX, e.clientY);
        if (img) {
          const mode = customDragHandler.onDown(img.row, img.col);
          if (mode) {
            customDragModeRef.current = mode;
            e.preventDefault();
            return; // Captured by custom handler — no pan
          }
        }
      }

      setIsDragging(true);
      dragStart.current = { x: e.clientX, y: e.clientY, pX: panX, pY: panY };
    },
    [lockView, panX, panY, customDragHandler, screenToImage],
  );

  const handleMouseUp = React.useCallback(
    (e: React.MouseEvent) => {
      // Custom drag end
      if (customDragModeRef.current) {
        customDragHandler?.onUp(customDragModeRef.current);
        customDragModeRef.current = null;
        clickDownRef.current = null;
        return;
      }

      // Click detection: if mouse didn't move more than threshold, it's a click
      if (onImageClick && clickDownRef.current) {
        const dx = Math.abs(e.clientX - clickDownRef.current.x);
        const dy = Math.abs(e.clientY - clickDownRef.current.y);
        if (dx < CLICK_THRESHOLD && dy < CLICK_THRESHOLD) {
          const img = screenToImage(e.clientX, e.clientY);
          if (img) {
            const r = Math.floor(img.row);
            const c = Math.floor(img.col);
            if (r >= 0 && r < rows && c >= 0 && c < cols) {
              onImageClick(r, c);
            }
          }
        }
      }

      setIsDragging(false);
      dragStart.current = null;
      clickDownRef.current = null;
    },
    [onImageClick, rows, cols, customDragHandler, screenToImage],
  );

  const handleDoubleClick = React.useCallback(() => {
    if (lockView) return;
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, [lockView]);

  const [isHovered, setIsHovered] = React.useState(false);
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);

  // Cursor readout + pan + custom drag move
  const handleCursorMove = React.useCallback(
    (e: React.MouseEvent) => {
      // Custom drag move
      if (customDragModeRef.current && customDragHandler) {
        const img = screenToImage(e.clientX, e.clientY);
        if (img) customDragHandler.onMove(img.row, img.col, customDragModeRef.current);
        return;
      }

      if (isDragging) {
        if (dragStart.current) {
          setPanX(dragStart.current.pX + (e.clientX - dragStart.current.x));
          setPanY(dragStart.current.pY + (e.clientY - dragStart.current.y));
        }
        return;
      }

      // Update custom cursor and hover handle based on what's under the mouse
      if (customDragHandler?.getCursor) {
        const img = screenToImage(e.clientX, e.clientY);
        if (img) {
          setCustomCursor(customDragHandler.getCursor(img.row, img.col));
        }
      }

      if (!parsed || rows <= 0 || cols <= 0) { setCursorInfo(null); return; }
      const img = screenToImage(e.clientX, e.clientY);
      if (!img) { setCursorInfo(null); return; }
      const col = Math.floor(img.col);
      const row = Math.floor(img.row);
      if (row < 0 || row >= rows || col < 0 || col >= cols) { setCursorInfo(null); return; }
      const idx = row * cols + col;
      const value = idx < parsed.length ? parsed[idx] : 0;
      setCursorInfo({ row, col, value });
    },
    [isDragging, parsed, rows, cols, screenToImage, customDragHandler],
  );

  return (
    <Box sx={{ width: cssW }}>
      <Typography
        sx={{ ...typography.labelSmall, color: textColor, mb: "2px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
      >
        {label}
      </Typography>
      <Box
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => { setIsHovered(false); setCursorInfo(null); setCustomCursor(null); setIsDragging(false); dragStart.current = null; clickDownRef.current = null; if (customDragModeRef.current) { customDragHandler?.onUp(customDragModeRef.current); customDragModeRef.current = null; } onMouseLeavePanel?.(); }}
        sx={{
          position: "relative",
          bgcolor: "#000",
          border: `1px solid ${borderColor}`,
          width: cssW,
          height: cssH,
          overflow: "hidden",
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: cssW,
            height: cssH,
            imageRendering: "pixelated",
            display: "block",
          }}
        />
        <canvas
          ref={overlayRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleCursorMove}
          onMouseUp={handleMouseUp}
          onWheel={handleWheel}
          onDoubleClick={handleDoubleClick}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            cursor: customDragModeRef.current
              ? (customDragModeRef.current === "move" ? "grabbing" : "ew-resize")
              : isDragging ? "grabbing"
              : customCursor || (onImageClick ? "crosshair" : lockView ? "default" : "grab"),
          }}
        />
        {/* Cursor readout (top-right) */}
        {cursorInfo && (
          <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.5)", px: 0.5, py: 0.15, pointerEvents: "none" }}>
            <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.8)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
              ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
            </Typography>
          </Box>
        )}
        {/* Stats shown on hover (bottom) */}
        {!hideStats && stats && isHovered && (
          <Box sx={{ position: "absolute", bottom: 0, left: 0, right: 0, bgcolor: "rgba(0,0,0,0.6)", px: 0.5, py: 0.25, pointerEvents: "none" }}>
            <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.85)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {statsLine(stats)}
            </Typography>
          </Box>
        )}
        {/* Resize handle */}
        {onResizeStart && (
          <Box
            onMouseDown={onResizeStart}
            sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.2 : 0.6, pointerEvents: lockView ? "none" : "auto", background: `linear-gradient(135deg, transparent 50%, ${accentColor} 50%)`, "&:hover": { opacity: lockView ? 0.2 : 1 } }}
          />
        )}
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// BinWidget — main component
// ---------------------------------------------------------------------------

function BinWidget() {
  const { themeInfo, colors: themeColors } = useTheme();

  // Shape traits
  const [scanRows] = useModelState<number>("scan_rows");
  const [scanCols] = useModelState<number>("scan_cols");
  const [detRows] = useModelState<number>("det_rows");
  const [detCols] = useModelState<number>("det_cols");

  const [binnedScanRows] = useModelState<number>("binned_scan_rows");
  const [binnedScanCols] = useModelState<number>("binned_scan_cols");
  const [binnedDetRows] = useModelState<number>("binned_det_rows");
  const [binnedDetCols] = useModelState<number>("binned_det_cols");

  const [maxScanBinRow] = useModelState<number>("max_scan_bin_row");
  const [maxScanBinCol] = useModelState<number>("max_scan_bin_col");
  const [maxDetBinRow] = useModelState<number>("max_det_bin_row");
  const [maxDetBinCol] = useModelState<number>("max_det_bin_col");

  // Bin factors
  const [scanBinRow, setScanBinRow] = useModelState<number>("scan_bin_row");
  const [scanBinCol, setScanBinCol] = useModelState<number>("scan_bin_col");
  const [detBinRow, setDetBinRow] = useModelState<number>("det_bin_row");
  const [detBinCol, setDetBinCol] = useModelState<number>("det_bin_col");

  const [binMode, setBinMode] = useModelState<string>("bin_mode");
  const [edgeMode, setEdgeMode] = useModelState<string>("edge_mode");

  // Mask ratios
  const [bfRadiusRatio, setBfRadiusRatio] = useModelState<number>("bf_radius_ratio");
  const [adfInnerRatio, setAdfInnerRatio] = useModelState<number>("adf_inner_ratio");
  const [adfOuterRatio, setAdfOuterRatio] = useModelState<number>("adf_outer_ratio");

  // Calibration
  const [pixelSizeRow] = useModelState<number>("pixel_size_row");
  const [pixelSizeCol] = useModelState<number>("pixel_size_col");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [pixelCalibrated] = useModelState<boolean>("pixel_calibrated");
  const [kPixelSizeRow] = useModelState<number>("k_pixel_size_row");
  const [kPixelSizeCol] = useModelState<number>("k_pixel_size_col");
  const [kUnit] = useModelState<string>("k_unit");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");
  const [binnedPixelSizeRow] = useModelState<number>("binned_pixel_size_row");
  const [binnedPixelSizeCol] = useModelState<number>("binned_pixel_size_col");
  const [binnedKPixelSizeRow] = useModelState<number>("binned_k_pixel_size_row");
  const [binnedKPixelSizeCol] = useModelState<number>("binned_k_pixel_size_col");

  // Scan position (compound [row, col], [-1, -1] = mean DP)
  const [scanPosition, setScanPosition] = useModelState<number[]>("_scan_position");

  // Preview bytes
  const [originalBfBytes] = useModelState<DataView>("original_bf_bytes");
  const [originalAdfBytes] = useModelState<DataView>("original_adf_bytes");
  const [binnedBfBytes] = useModelState<DataView>("binned_bf_bytes");
  const [binnedAdfBytes] = useModelState<DataView>("binned_adf_bytes");
  const [originalMeanDpBytes] = useModelState<DataView>("original_mean_dp_bytes");
  const [binnedMeanDpBytes] = useModelState<DataView>("binned_mean_dp_bytes");

  // Per-position DP bytes
  const [positionDpBytes] = useModelState<DataView>("_position_dp_bytes");
  const [binnedPositionDpBytes] = useModelState<DataView>("_binned_position_dp_bytes");

  // Detector center
  const [centerRow, setCenterRow] = useModelState<number>("center_row");
  const [centerCol, setCenterCol] = useModelState<number>("center_col");
  const [binnedCenterRow] = useModelState<number>("binned_center_row");
  const [binnedCenterCol] = useModelState<number>("binned_center_col");

  // Stats
  const [originalBfStats] = useModelState<number[]>("original_bf_stats");
  const [originalAdfStats] = useModelState<number[]>("original_adf_stats");
  const [binnedBfStats] = useModelState<number[]>("binned_bf_stats");
  const [binnedAdfStats] = useModelState<number[]>("binned_adf_stats");

  // Display
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [showControls] = useModelState<boolean>("show_controls");
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");

  // Export (trait-triggered .npy download)
  const [, setNpyExportRequested] = useModelState<boolean>("_npy_export_requested");
  const [npyExportData] = useModelState<DataView>("_npy_export_data");

  // Status
  const [statusMessage] = useModelState<string>("status_message");
  const [statusLevel] = useModelState<string>("status_level");

  // Local UI state
  const [canvasSize, setCanvasSize] = React.useState(CANVAS_SIZE);
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number } | null>(null);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);
  const [resetKey, setResetKey] = React.useState(0);
  const [imageVminPct, setImageVminPct] = React.useState(0);
  const [imageVmaxPct, setImageVmaxPct] = React.useState(100);
  const [npyExporting, setNpyExporting] = React.useState(false);

  // Hover handle state for detector ROI highlight
  const [origDetHoverHandle, setOrigDetHoverHandle] = React.useState<string | null>(null);
  const [binnedDetHoverHandle, setBinnedDetHoverHandle] = React.useState<string | null>(null);

  // WebGPU FFT
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  const [origBfFftMag, setOrigBfFftMag] = React.useState<Float32Array | null>(null);
  const [origAdfFftMag, setOrigAdfFftMag] = React.useState<Float32Array | null>(null);
  const [binnedBfFftMag, setBinnedBfFftMag] = React.useState<Float32Array | null>(null);
  const [binnedAdfFftMag, setBinnedAdfFftMag] = React.useState<Float32Array | null>(null);

  // Canvas refs for export
  const origBfRef = React.useRef<HTMLCanvasElement | null>(null);
  const origAdfRef = React.useRef<HTMLCanvasElement | null>(null);
  const origDpRef = React.useRef<HTMLCanvasElement | null>(null);
  const binnedBfRef = React.useRef<HTMLCanvasElement | null>(null);
  const binnedAdfRef = React.useRef<HTMLCanvasElement | null>(null);
  const binnedDpRef = React.useRef<HTMLCanvasElement | null>(null);

  // Factor options
  const scanRowOptions = React.useMemo(() => optionsForFactor(maxScanBinRow), [maxScanBinRow]);
  const scanColOptions = React.useMemo(() => optionsForFactor(maxScanBinCol), [maxScanBinCol]);
  const detRowOptions = React.useMemo(() => optionsForFactor(maxDetBinRow), [maxDetBinRow]);
  const detColOptions = React.useMemo(() => optionsForFactor(maxDetBinCol), [maxDetBinCol]);

  // Tool visibility
  const toolVisibility = React.useMemo(
    () => computeToolVisibility("Bin", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideDisplay = toolVisibility.isHidden("display");
  const hideBinning = toolVisibility.isHidden("binning");
  const hideMask = toolVisibility.isHidden("mask");
  const hidePreview = toolVisibility.isHidden("preview");
  const hideStats = toolVisibility.isHidden("stats");
  const hideExport = toolVisibility.isHidden("export");
  const lockDisplay = toolVisibility.isLocked("display");
  const lockBinning = toolVisibility.isLocked("binning");
  const lockMask = toolVisibility.isLocked("mask");
  const lockExport = toolVisibility.isLocked("export");
  const lockView = lockDisplay;

  // Scan position state
  const posRow = scanPosition?.[0] ?? -1;
  const posCol = scanPosition?.[1] ?? -1;
  const hasPosition = posRow >= 0 && posCol >= 0;
  const positionMarker = hasPosition ? { row: posRow, col: posCol } : null;
  // Binned position marker
  const binnedPosRow = hasPosition ? Math.min(Math.floor(posRow / Math.max(1, scanBinRow)), binnedScanRows - 1) : -1;
  const binnedPosCol = hasPosition ? Math.min(Math.floor(posCol / Math.max(1, scanBinCol)), binnedScanCols - 1) : -1;
  const binnedPositionMarker = binnedPosRow >= 0 && binnedPosCol >= 0 ? { row: binnedPosRow, col: binnedPosCol } : null;

  // Click handler for BF/ADF panels — sets scan position
  const handleScanClick = React.useCallback(
    (row: number, col: number) => {
      setScanPosition([row, col]);
    },
    [setScanPosition],
  );
  const handleBinnedScanClick = React.useCallback(
    (row: number, col: number) => {
      // Convert binned position back to original coords
      const origRow = Math.min(row * Math.max(1, scanBinRow), scanRows - 1);
      const origCol = Math.min(col * Math.max(1, scanBinCol), scanCols - 1);
      setScanPosition([origRow, origCol]);
    },
    [setScanPosition, scanBinRow, scanBinCol, scanRows, scanCols],
  );

  // Determine which bytes to show in DP panels (position-specific or mean)
  const origDpBytes = hasPosition && positionDpBytes && positionDpBytes.byteLength > 0
    ? positionDpBytes : originalMeanDpBytes;
  const binnedDpBytes = hasPosition && binnedPositionDpBytes && binnedPositionDpBytes.byteLength > 0
    ? binnedPositionDpBytes : binnedMeanDpBytes;
  const origDpLabel = hasPosition
    ? `DP (${posRow}, ${posCol})` : "Mean DP";
  const binnedDpLabel = hasPosition
    ? `Binned DP (${binnedPosRow}, ${binnedPosCol})` : "Binned Mean DP";

  // Detector ROI drag handler for DP panels
  const makeDetectorDragHandler = React.useCallback(
    (dRows: number, dCols: number, setHoverHandle: (h: string | null) => void) => {
      const detSize = Math.min(dRows, dCols);
      // Convert screen-pixel hit radius to image-pixel space (approximate)
      const hitPxToImg = ROI_HIT_PX * Math.max(dRows, dCols) / canvasSize;

      const getDistToCenter = (imgRow: number, imgCol: number) =>
        Math.sqrt((imgRow - centerRow) ** 2 + (imgCol - centerCol) ** 2);

      const getBfRadius = () => bfRadiusRatio * detSize;
      const getAdfInner = () => adfInnerRatio * detSize;
      const getAdfOuter = () => adfOuterRatio * detSize;

      return {
        onDown: (imgRow: number, imgCol: number): string | null => {
          if (lockMask) return null;
          const d = getDistToCenter(imgRow, imgCol);
          // Check hits: BF edge, ADF inner edge, ADF outer edge, then center
          if (Math.abs(d - getBfRadius()) < hitPxToImg) return "bf-resize";
          if (Math.abs(d - getAdfInner()) < hitPxToImg) return "adf-inner";
          if (Math.abs(d - getAdfOuter()) < hitPxToImg) return "adf-outer";
          if (d < getBfRadius()) return "move";
          return null;
        },
        onMove: (imgRow: number, imgCol: number, mode: string) => {
          if (lockMask) return;
          const d = Math.sqrt((imgRow - centerRow) ** 2 + (imgCol - centerCol) ** 2);
          if (mode === "bf-resize") {
            const newRatio = Math.max(0.02, Math.min(0.5, d / detSize));
            setBfRadiusRatio(newRatio);
          } else if (mode === "adf-inner") {
            const newRatio = Math.max(0.02, Math.min(adfOuterRatio - 0.01, d / detSize));
            setAdfInnerRatio(newRatio);
          } else if (mode === "adf-outer") {
            const newRatio = Math.max(adfInnerRatio + 0.01, Math.min(0.95, d / detSize));
            setAdfOuterRatio(newRatio);
          } else if (mode === "move") {
            const newRow = Math.max(0, Math.min(dRows - 1, imgRow));
            const newCol = Math.max(0, Math.min(dCols - 1, imgCol));
            setCenterRow(newRow);
            setCenterCol(newCol);
          }
        },
        onUp: (_mode: string) => { /* no-op */ },
        getCursor: (imgRow: number, imgCol: number): string | null => {
          if (lockMask) { setHoverHandle(null); return null; }
          const d = getDistToCenter(imgRow, imgCol);
          if (Math.abs(d - getBfRadius()) < hitPxToImg) { setHoverHandle("bf-resize"); return "ew-resize"; }
          if (Math.abs(d - getAdfInner()) < hitPxToImg) { setHoverHandle("adf-inner"); return "ew-resize"; }
          if (Math.abs(d - getAdfOuter()) < hitPxToImg) { setHoverHandle("adf-outer"); return "ew-resize"; }
          if (d < getBfRadius()) { setHoverHandle("move"); return "move"; }
          setHoverHandle(null);
          return null;
        },
      };
    },
    [canvasSize, centerRow, centerCol, bfRadiusRatio, adfInnerRatio, adfOuterRatio,
     lockMask, setBfRadiusRatio, setAdfInnerRatio, setAdfOuterRatio, setCenterRow, setCenterCol],
  );

  const origDetDragHandler = React.useMemo(
    () => makeDetectorDragHandler(detRows, detCols, setOrigDetHoverHandle),
    [makeDetectorDragHandler, detRows, detCols],
  );
  const binnedDetDragHandler = React.useMemo(
    () => {
      const setHoverHandle = setBinnedDetHoverHandle;
      // For binned panel, we need to convert binned coords back to original coords for hit detection
      const detSize = Math.min(binnedDetRows, binnedDetCols);
      const hitPxToImg = ROI_HIT_PX * Math.max(binnedDetRows, binnedDetCols) / canvasSize;

      return {
        onDown: (imgRow: number, imgCol: number): string | null => {
          if (lockMask) return null;
          const d = Math.sqrt((imgRow - binnedCenterRow) ** 2 + (imgCol - binnedCenterCol) ** 2);
          if (Math.abs(d - bfRadiusRatio * detSize) < hitPxToImg) return "bf-resize";
          if (Math.abs(d - adfInnerRatio * detSize) < hitPxToImg) return "adf-inner";
          if (Math.abs(d - adfOuterRatio * detSize) < hitPxToImg) return "adf-outer";
          if (d < bfRadiusRatio * detSize) return "move";
          return null;
        },
        onMove: (imgRow: number, imgCol: number, mode: string) => {
          if (lockMask) return;
          if (mode === "move") {
            // Convert binned coords to original
            const origRow = imgRow * Math.max(1, detBinRow);
            const origCol = imgCol * Math.max(1, detBinCol);
            setCenterRow(Math.max(0, Math.min(detRows - 1, origRow)));
            setCenterCol(Math.max(0, Math.min(detCols - 1, origCol)));
          } else {
            const d = Math.sqrt((imgRow - binnedCenterRow) ** 2 + (imgCol - binnedCenterCol) ** 2);
            if (mode === "bf-resize") {
              setBfRadiusRatio(Math.max(0.02, Math.min(0.5, d / detSize)));
            } else if (mode === "adf-inner") {
              setAdfInnerRatio(Math.max(0.02, Math.min(adfOuterRatio - 0.01, d / detSize)));
            } else if (mode === "adf-outer") {
              setAdfOuterRatio(Math.max(adfInnerRatio + 0.01, Math.min(0.95, d / detSize)));
            }
          }
        },
        onUp: (_mode: string) => { /* no-op */ },
        getCursor: (imgRow: number, imgCol: number): string | null => {
          if (lockMask) { setHoverHandle(null); return null; }
          const d = Math.sqrt((imgRow - binnedCenterRow) ** 2 + (imgCol - binnedCenterCol) ** 2);
          if (Math.abs(d - bfRadiusRatio * detSize) < hitPxToImg) { setHoverHandle("bf-resize"); return "ew-resize"; }
          if (Math.abs(d - adfInnerRatio * detSize) < hitPxToImg) { setHoverHandle("adf-inner"); return "ew-resize"; }
          if (Math.abs(d - adfOuterRatio * detSize) < hitPxToImg) { setHoverHandle("adf-outer"); return "ew-resize"; }
          if (d < bfRadiusRatio * detSize) { setHoverHandle("move"); return "move"; }
          setHoverHandle(null);
          return null;
        },
      };
    },
    [binnedDetRows, binnedDetCols, binnedCenterRow, binnedCenterCol, canvasSize,
     bfRadiusRatio, adfInnerRatio, adfOuterRatio, lockMask, detBinRow, detBinCol,
     detRows, detCols, setBfRadiusRatio, setAdfInnerRatio, setAdfOuterRatio, setCenterRow, setCenterCol],
  );

  // Histogram data (original BF, processed)
  const histogramData = React.useMemo(() => {
    const parsed = extractFloat32(originalBfBytes);
    if (!parsed || parsed.length === 0) return null;
    return logScale ? applyLogScale(parsed) : parsed;
  }, [originalBfBytes, logScale]);

  const histogramDataRange = React.useMemo(() => {
    if (!histogramData) return { min: 0, max: 1 };
    return findDataRange(histogramData);
  }, [histogramData]);

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
      }
    });
  }, []);

  // Async FFT helper: GPU-first with CPU fallback
  const computeFFTAsync = React.useCallback(async (
    inputData: Float32Array, w: number, h: number,
  ): Promise<Float32Array> => {
    const real = inputData.slice();
    const imag = new Float32Array(inputData.length);
    let fReal: Float32Array;
    let fImag: Float32Array;
    if (gpuFFTRef.current && gpuReady) {
      const result = await gpuFFTRef.current.fft2D(real, imag, w, h, false);
      fReal = result.real;
      fImag = result.imag;
    } else {
      fft2d(real, imag, w, h, false);
      fReal = real;
      fImag = imag;
    }
    fftshift(fReal, w, h);
    fftshift(fImag, w, h);
    const mag = computeMagnitude(fReal, fImag);
    autoEnhanceFFT(mag, w, h);
    return mag;
  }, [gpuReady]);

  // FFT for scan-space panels (BF and ADF) — async with WebGPU
  React.useEffect(() => {
    if (!showFft) { setOrigBfFftMag(null); return; }
    const data = extractFloat32(originalBfBytes);
    if (!data || data.length === 0 || scanRows <= 0 || scanCols <= 0) { setOrigBfFftMag(null); return; }
    let cancelled = false;
    computeFFTAsync(data, scanCols, scanRows).then(mag => { if (!cancelled) setOrigBfFftMag(mag); });
    return () => { cancelled = true; };
  }, [showFft, originalBfBytes, scanRows, scanCols, computeFFTAsync]);

  React.useEffect(() => {
    if (!showFft) { setOrigAdfFftMag(null); return; }
    const data = extractFloat32(originalAdfBytes);
    if (!data || data.length === 0 || scanRows <= 0 || scanCols <= 0) { setOrigAdfFftMag(null); return; }
    let cancelled = false;
    computeFFTAsync(data, scanCols, scanRows).then(mag => { if (!cancelled) setOrigAdfFftMag(mag); });
    return () => { cancelled = true; };
  }, [showFft, originalAdfBytes, scanRows, scanCols, computeFFTAsync]);

  React.useEffect(() => {
    if (!showFft) { setBinnedBfFftMag(null); return; }
    const data = extractFloat32(binnedBfBytes);
    if (!data || data.length === 0 || binnedScanRows <= 0 || binnedScanCols <= 0) { setBinnedBfFftMag(null); return; }
    let cancelled = false;
    computeFFTAsync(data, binnedScanCols, binnedScanRows).then(mag => { if (!cancelled) setBinnedBfFftMag(mag); });
    return () => { cancelled = true; };
  }, [showFft, binnedBfBytes, binnedScanRows, binnedScanCols, computeFFTAsync]);

  React.useEffect(() => {
    if (!showFft) { setBinnedAdfFftMag(null); return; }
    const data = extractFloat32(binnedAdfBytes);
    if (!data || data.length === 0 || binnedScanRows <= 0 || binnedScanCols <= 0) { setBinnedAdfFftMag(null); return; }
    let cancelled = false;
    computeFFTAsync(data, binnedScanCols, binnedScanRows).then(mag => { if (!cancelled) setBinnedAdfFftMag(mag); });
    return () => { cancelled = true; };
  }, [showFft, binnedAdfBytes, binnedScanRows, binnedScanCols, computeFFTAsync]);

  // Download .npy when data arrives
  React.useEffect(() => {
    if (!npyExporting || !npyExportData || npyExportData.byteLength === 0) return;
    downloadDataView(npyExportData, "binned_4d.npy");
    setNpyExporting(false);
  }, [npyExportData, npyExporting]);

  // Themed styling
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
    PaperProps: {
      sx: {
        bgcolor: themeColors.controlBg,
        color: themeColors.text,
        border: `1px solid ${themeColors.border}`,
      },
    },
  };

  // Resize handle
  const handleCanvasResizeStart = React.useCallback(
    (e: React.MouseEvent) => {
      if (lockView) return;
      e.stopPropagation();
      e.preventDefault();
      setIsResizingCanvas(true);
      setResizeStart({ x: e.clientX, y: e.clientY, size: canvasSize });
    },
    [lockView, canvasSize],
  );

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      setCanvasSize(Math.max(120, resizeStart.size + delta));
    };
    const handleMouseUp = () => {
      setIsResizingCanvas(false);
      setResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingCanvas, resizeStart]);

  // Detector overlay renderers
  const makeDetectorOverlay = React.useCallback(
    (cRow: number, cCol: number, dRows: number, dCols: number, hoverHandle: string | null) =>
      (ctx: CanvasRenderingContext2D, cssW: number, cssH: number, zm: number, pX: number, pY: number) => {
        if (dRows <= 0 || dCols <= 0) return;
        const scaleX = cssW / dCols;
        const scaleY = cssH / dRows;
        const detSize = Math.min(dRows, dCols);

        // Apply same transform as the image canvas
        ctx.save();
        ctx.translate(pX, pY);
        ctx.scale(zm, zm);

        const cx = cCol * scaleX;
        const cy = cRow * scaleY;

        // BF disk
        const bfRadiusPx = bfRadiusRatio * detSize;
        const bfRx = bfRadiusPx * scaleX;
        const bfRy = bfRadiusPx * scaleY;

        const bfHighlighted = hoverHandle === "bf-resize" || hoverHandle === "move";
        const adfInnerHighlighted = hoverHandle === "adf-inner";
        const adfOuterHighlighted = hoverHandle === "adf-outer";

        ctx.lineWidth = 2 / zm;
        ctx.shadowColor = "rgba(0,0,0,0.5)";
        ctx.shadowBlur = 2 / zm;

        ctx.beginPath();
        ctx.ellipse(cx, cy, bfRx, bfRy, 0, 0, 2 * Math.PI);
        ctx.fillStyle = bfHighlighted ? "rgba(255, 255, 0, 0.18)" : "rgba(0, 255, 0, 0.12)";
        ctx.fill();
        ctx.strokeStyle = bfHighlighted ? "rgba(255, 255, 0, 1)" : "rgba(0, 255, 0, 0.9)";
        if (bfHighlighted) ctx.lineWidth = 3 / zm;
        ctx.stroke();
        ctx.lineWidth = 2 / zm;

        ctx.shadowBlur = 0;
        ctx.font = `bold ${10 / zm}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
        ctx.fillStyle = bfHighlighted ? "rgba(255, 255, 0, 1)" : "rgba(0, 255, 0, 0.9)";
        ctx.textAlign = "center";
        ctx.fillText("BF", cx, cy - bfRy - 4 / zm);

        // ADF annulus
        const adfInnerPx = adfInnerRatio * detSize;
        const adfOuterPx = adfOuterRatio * detSize;
        const adfInnerRx = adfInnerPx * scaleX;
        const adfInnerRy = adfInnerPx * scaleY;
        const adfOuterRx = adfOuterPx * scaleX;
        const adfOuterRy = adfOuterPx * scaleY;

        ctx.shadowBlur = 2 / zm;
        ctx.beginPath();
        ctx.ellipse(cx, cy, adfOuterRx, adfOuterRy, 0, 0, 2 * Math.PI);
        ctx.ellipse(cx, cy, adfInnerRx, adfInnerRy, 0, 2 * Math.PI, 0, true);
        ctx.fillStyle = "rgba(0, 220, 255, 0.12)";
        ctx.fill();

        ctx.beginPath();
        ctx.ellipse(cx, cy, adfOuterRx, adfOuterRy, 0, 0, 2 * Math.PI);
        ctx.strokeStyle = adfOuterHighlighted ? "rgba(255, 255, 0, 1)" : "rgba(0, 220, 255, 0.9)";
        if (adfOuterHighlighted) ctx.lineWidth = 3 / zm;
        ctx.stroke();
        ctx.lineWidth = 2 / zm;

        ctx.beginPath();
        ctx.ellipse(cx, cy, adfInnerRx, adfInnerRy, 0, 0, 2 * Math.PI);
        ctx.strokeStyle = adfInnerHighlighted ? "rgba(255, 255, 0, 1)" : "rgba(0, 220, 255, 0.9)";
        if (adfInnerHighlighted) ctx.lineWidth = 3 / zm;
        ctx.stroke();
        ctx.lineWidth = 2 / zm;

        ctx.shadowBlur = 0;
        ctx.fillStyle = (adfInnerHighlighted || adfOuterHighlighted) ? "rgba(255, 255, 0, 1)" : "rgba(0, 220, 255, 0.9)";
        ctx.fillText("ADF", cx, cy - adfOuterRy - 4 / zm);

        ctx.restore();
      },
    [bfRadiusRatio, adfInnerRatio, adfOuterRatio],
  );

  const origDetOverlay = React.useMemo(
    () => makeDetectorOverlay(centerRow, centerCol, detRows, detCols, origDetHoverHandle),
    [makeDetectorOverlay, centerRow, centerCol, detRows, detCols, origDetHoverHandle],
  );
  const binnedDetOverlay = React.useMemo(
    () => makeDetectorOverlay(binnedCenterRow, binnedCenterCol, binnedDetRows, binnedDetCols, binnedDetHoverHandle),
    [makeDetectorOverlay, binnedCenterRow, binnedCenterCol, binnedDetRows, binnedDetCols, binnedDetHoverHandle],
  );

  // Export
  const createComposite = React.useCallback((): HTMLCanvasElement | null => {
    const refs = [origBfRef, origAdfRef, origDpRef, binnedBfRef, binnedAdfRef, binnedDpRef];
    const canvases = refs.map((r) => r.current).filter((c): c is HTMLCanvasElement => c !== null);
    if (canvases.length < 6) return null;

    const gap = 4;
    // 2 rows × 3 cols: [origBF, origADF, origDP] / [binnedBF, binnedADF, binnedDP]
    const w0 = Math.max(canvases[0].width, canvases[3].width);
    const w1 = Math.max(canvases[1].width, canvases[4].width);
    const w2 = Math.max(canvases[2].width, canvases[5].width);
    const h0 = Math.max(canvases[0].height, canvases[1].height, canvases[2].height);
    const h1 = Math.max(canvases[3].height, canvases[4].height, canvases[5].height);

    const composite = document.createElement("canvas");
    composite.width = w0 + w1 + w2 + gap * 2;
    composite.height = h0 + h1 + gap;
    const ctx = composite.getContext("2d");
    if (!ctx) return null;

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, composite.width, composite.height);
    ctx.drawImage(canvases[0], 0, 0);
    ctx.drawImage(canvases[1], w0 + gap, 0);
    ctx.drawImage(canvases[2], w0 + w1 + gap * 2, 0);
    ctx.drawImage(canvases[3], 0, h0 + gap);
    ctx.drawImage(canvases[4], w0 + gap, h0 + gap);
    ctx.drawImage(canvases[5], w0 + w1 + gap * 2, h0 + gap);
    return composite;
  }, []);

  const handleExportPdf = React.useCallback(async () => {
    if (lockExport) return;
    setExportAnchor(null);
    const composite = createComposite();
    if (!composite) return;
    const pdfBlob = await canvasToPDF(composite);
    downloadBlob(pdfBlob, "bin_export.pdf");
  }, [lockExport, createComposite]);

  const handleExportNpy = React.useCallback(() => {
    if (lockExport) return;
    setExportAnchor(null);
    setNpyExporting(true);
    setNpyExportRequested(true);
  }, [lockExport, setNpyExportRequested]);

  const handleCopy = React.useCallback(async () => {
    if (lockExport) return;
    const composite = createComposite();
    if (!composite) return;
    composite.toBlob(async (blob) => {
      if (!blob) return;
      try {
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
      } catch {
        /* clipboard not available */
      }
    }, "image/png");
  }, [lockExport, createComposite]);

  // Keyboard
  const handleKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      const key = String(event.key || "").toLowerCase();
      if (isEditableElement(event.target)) return;

      if (key === "escape") {
        event.preventDefault();
        setScanPosition([-1, -1]);
        return;
      }
      if (key === "r") {
        event.preventDefault();
        setResetKey((prev) => prev + 1);
        return;
      }
      if (key === "l" && !lockDisplay) {
        event.preventDefault();
        setLogScale((prev) => !prev);
        return;
      }
      if (key === "a" && !lockDisplay) {
        event.preventDefault();
        setAutoContrast((prev) => !prev);
        return;
      }
      if (key === "f" && !lockDisplay) {
        event.preventDefault();
        setShowFft((prev) => !prev);
        return;
      }
      if (key === "c" && !lockDisplay) {
        event.preventDefault();
        const idx = COLORMAP_NAMES.indexOf(cmap);
        const nextIdx = idx < 0 ? 0 : (idx + 1) % COLORMAP_NAMES.length;
        setCmap(COLORMAP_NAMES[nextIdx]);
        return;
      }
      if (key === "m" && !lockBinning) {
        event.preventDefault();
        setBinMode(binMode === "sum" ? "mean" : "sum");
        return;
      }
      if (key === "e" && !lockBinning) {
        event.preventDefault();
        const order = ["crop", "pad", "error"];
        const idx = order.indexOf(edgeMode);
        const nextIdx = idx < 0 ? 0 : (idx + 1) % order.length;
        setEdgeMode(order[nextIdx]);
        return;
      }
      if (key === "]" && !lockBinning) {
        event.preventDefault();
        setScanBinRow(stepOption(scanRowOptions, scanBinRow, +1));
        setScanBinCol(stepOption(scanColOptions, scanBinCol, +1));
        return;
      }
      if (key === "[" && !lockBinning) {
        event.preventDefault();
        setScanBinRow(stepOption(scanRowOptions, scanBinRow, -1));
        setScanBinCol(stepOption(scanColOptions, scanBinCol, -1));
        return;
      }
      if ((key === "=" || key === "+") && !lockBinning) {
        event.preventDefault();
        setDetBinRow(stepOption(detRowOptions, detBinRow, +1));
        setDetBinCol(stepOption(detColOptions, detBinCol, +1));
        return;
      }
      if (key === "-" && !lockBinning) {
        event.preventDefault();
        setDetBinRow(stepOption(detRowOptions, detBinRow, -1));
        setDetBinCol(stepOption(detColOptions, detBinCol, -1));
      }
    },
    [
      binMode, cmap, detBinCol, detBinRow, detColOptions,
      detRowOptions, edgeMode, lockBinning, lockDisplay, scanBinCol, scanBinRow,
      scanColOptions, scanRowOptions, setAutoContrast, setBinMode, setCmap, setDetBinCol,
      setDetBinRow, setEdgeMode, setLogScale, setScanBinCol, setScanBinRow, setShowFft,
      setScanPosition,
    ],
  );

  // Status color
  const statusColor =
    statusLevel === "error" ? "#ff6b6b" : statusLevel === "warn" ? "#ffb84d" : themeColors.textMuted;

  return (
    <Box
      className="bin-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      sx={{
        p: `${SPACING.LG}px`,
        bgcolor: themeColors.bg,
        color: themeColors.text,
        outline: "none",
      }}
    >
      {/* HEADER */}
      <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center" sx={{ mb: `${SPACING.SM}px` }}>
        <Typography sx={{ fontSize: 12, fontWeight: "bold", color: themeColors.accent, flex: 1 }}>
          {title || "Bin"}
          <InfoTooltip
            text={
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Keyboard Shortcuts</Typography>
                <KeyboardShortcuts items={[
                  ["Click", "Select scan position (BF/ADF)"],
                  ["Drag", "Move detector center / resize ROI (DP)"],
                  ["Esc", "Clear position selection"],
                  ["R", "Reset zoom"],
                  ["L", "Toggle log scale"],
                  ["A", "Toggle auto-contrast"],
                  ["F", "Toggle FFT"],
                  ["C", "Cycle colormap"],
                  ["M", "Toggle mean / sum"],
                  ["E", "Cycle edge mode"],
                  ["[ / ]", "Decrease / increase scan bin"],
                  ["- / +", "Decrease / increase det bin"],
                  ["Scroll", "Zoom in / out"],
                  ["Dbl-click", "Reset view"],
                ]} />
              </Box>
            }
            theme={themeInfo.theme === "dark" ? "dark" : "light"}
          />
          <ControlCustomizer
            widgetName="Bin"
            hiddenTools={hiddenTools}
            setHiddenTools={setHiddenTools}
            disabledTools={disabledTools}
            setDisabledTools={setDisabledTools}
            themeColors={themeColors}
          />
        </Typography>
        {!hideExport && (
          <>
            <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={handleCopy}>COPY</Button>
            <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }}>EXPORT</Button>
            <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
              <MenuItem disabled={lockExport} onClick={handleExportPdf} sx={{ fontSize: 12 }}>PDF (grid)</MenuItem>
              <MenuItem disabled={lockExport || npyExporting} onClick={handleExportNpy} sx={{ fontSize: 12 }}>{npyExporting ? "Exporting..." : "NumPy (.npy)"}</MenuItem>
            </Menu>
          </>
        )}
      </Stack>

      {/* 2×3 IMAGE GRID */}
      {!hidePreview && (
        <Box sx={{ position: "relative", width: "fit-content" }}>
          {/* Row 1: Original BF | Original ADF | Original DP */}
          <Stack direction="row" spacing={`${SPACING.SM}px`}>
            <InteractivePanel
              label={showFft ? "Original BF (FFT)" : "Original BF"}
              rows={scanRows}
              cols={scanCols}
              bytes={originalBfBytes}
              overrideData={origBfFftMag}
              cmap={cmap}
              logScale={showFft ? false : logScale}
              autoContrast={autoContrast}
              vminPct={imageVminPct}
              vmaxPct={imageVmaxPct}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              stats={originalBfStats}
              hideStats={hideStats}
              pixelSize={showFft ? undefined : pixelCalibrated ? pixelSizeCol : undefined}
              pixelUnit={showFft ? undefined : pixelCalibrated ? pixelUnit : undefined}
              resetKey={resetKey}
              canvasRef={origBfRef}
              onResizeStart={handleCanvasResizeStart}
              onImageClick={handleScanClick}
              positionMarker={positionMarker}
            />
            <InteractivePanel
              label={showFft ? "Original ADF (FFT)" : "Original ADF"}
              rows={scanRows}
              cols={scanCols}
              bytes={originalAdfBytes}
              overrideData={origAdfFftMag}
              cmap={cmap}
              logScale={showFft ? false : logScale}
              autoContrast={autoContrast}
              vminPct={imageVminPct}
              vmaxPct={imageVmaxPct}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              stats={originalAdfStats}
              hideStats={hideStats}
              pixelSize={showFft ? undefined : pixelCalibrated ? pixelSizeCol : undefined}
              pixelUnit={showFft ? undefined : pixelCalibrated ? pixelUnit : undefined}
              resetKey={resetKey}
              canvasRef={origAdfRef}
              onResizeStart={handleCanvasResizeStart}
              onImageClick={handleScanClick}
              positionMarker={positionMarker}
            />
            <InteractivePanel
              label={origDpLabel}
              rows={detRows}
              cols={detCols}
              bytes={origDpBytes}
              cmap={cmap}
              logScale={logScale}
              autoContrast={autoContrast}
              vminPct={imageVminPct}
              vmaxPct={imageVmaxPct}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              hideStats
              pixelSize={kCalibrated ? kPixelSizeCol : undefined}
              pixelUnit={kCalibrated ? kUnit : undefined}
              resetKey={resetKey}
              overlayRenderer={origDetOverlay}
              canvasRef={origDpRef}
              onResizeStart={handleCanvasResizeStart}
              customDragHandler={origDetDragHandler}
              onMouseLeavePanel={() => setOrigDetHoverHandle(null)}
            />
          </Stack>

          {/* Row 2: Binned BF | Binned ADF | Binned DP */}
          <Stack direction="row" spacing={`${SPACING.SM}px`} sx={{ mt: `${SPACING.SM}px` }}>
            <InteractivePanel
              label={showFft ? "Binned BF (FFT)" : "Binned BF"}
              rows={binnedScanRows}
              cols={binnedScanCols}
              bytes={binnedBfBytes}
              overrideData={binnedBfFftMag}
              cmap={cmap}
              logScale={showFft ? false : logScale}
              autoContrast={autoContrast}
              vminPct={imageVminPct}
              vmaxPct={imageVmaxPct}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              stats={binnedBfStats}
              hideStats={hideStats}
              pixelSize={showFft ? undefined : pixelCalibrated ? binnedPixelSizeCol : undefined}
              pixelUnit={showFft ? undefined : pixelCalibrated ? pixelUnit : undefined}
              resetKey={resetKey}
              canvasRef={binnedBfRef}
              onResizeStart={handleCanvasResizeStart}
              onImageClick={handleBinnedScanClick}
              positionMarker={binnedPositionMarker}
            />
            <InteractivePanel
              label={showFft ? "Binned ADF (FFT)" : "Binned ADF"}
              rows={binnedScanRows}
              cols={binnedScanCols}
              bytes={binnedAdfBytes}
              overrideData={binnedAdfFftMag}
              cmap={cmap}
              logScale={showFft ? false : logScale}
              autoContrast={autoContrast}
              vminPct={imageVminPct}
              vmaxPct={imageVmaxPct}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              stats={binnedAdfStats}
              hideStats={hideStats}
              pixelSize={showFft ? undefined : pixelCalibrated ? binnedPixelSizeCol : undefined}
              pixelUnit={showFft ? undefined : pixelCalibrated ? pixelUnit : undefined}
              resetKey={resetKey}
              canvasRef={binnedAdfRef}
              onResizeStart={handleCanvasResizeStart}
              onImageClick={handleBinnedScanClick}
              positionMarker={binnedPositionMarker}
            />
            <InteractivePanel
              label={binnedDpLabel}
              rows={binnedDetRows}
              cols={binnedDetCols}
              bytes={binnedDpBytes}
              cmap={cmap}
              logScale={logScale}
              autoContrast={autoContrast}
              vminPct={imageVminPct}
              vmaxPct={imageVmaxPct}
              canvasSize={canvasSize}
              borderColor={themeColors.border}
              textColor={themeColors.text}
              mutedColor={themeColors.textMuted}
              accentColor={themeColors.accent}
              lockView={lockView}
              hideStats
              pixelSize={kCalibrated ? binnedKPixelSizeCol : undefined}
              pixelUnit={kCalibrated ? kUnit : undefined}
              resetKey={resetKey}
              overlayRenderer={binnedDetOverlay}
              canvasRef={binnedDpRef}
              onResizeStart={handleCanvasResizeStart}
              customDragHandler={binnedDetDragHandler}
              onMouseLeavePanel={() => setBinnedDetHoverHandle(null)}
            />
          </Stack>

        </Box>
      )}

      {/* CONTROLS (below images) */}
      {showControls && (
        <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: "2px" }}>
          {/* Row 1: Scan + Det bins + Reduce + Edge */}
          {!hideBinning && (
            <Box sx={{ ...controlRow, gap: `${SPACING.SM}px`, width: "auto" }}>
              <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>Scan:</Typography>
              <Select size="small" value={scanBinRow} onChange={(e) => setScanBinRow(Number(e.target.value))} sx={{ ...themedSelect, minWidth: 44 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                {scanRowOptions.map((v) => <MenuItem key={`sr-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
              <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>×</Typography>
              <Select size="small" value={scanBinCol} onChange={(e) => setScanBinCol(Number(e.target.value))} sx={{ ...themedSelect, minWidth: 44 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                {scanColOptions.map((v) => <MenuItem key={`sc-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
              <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>Det:</Typography>
              <Select size="small" value={detBinRow} onChange={(e) => setDetBinRow(Number(e.target.value))} sx={{ ...themedSelect, minWidth: 44 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                {detRowOptions.map((v) => <MenuItem key={`dr-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
              <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>×</Typography>
              <Select size="small" value={detBinCol} onChange={(e) => setDetBinCol(Number(e.target.value))} sx={{ ...themedSelect, minWidth: 44 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                {detColOptions.map((v) => <MenuItem key={`dc-${v}`} value={v}>{v}</MenuItem>)}
              </Select>
              <Select size="small" value={binMode} onChange={(e) => setBinMode(String(e.target.value))} sx={{ ...themedSelect, minWidth: 52 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                <MenuItem value="mean">mean</MenuItem>
                <MenuItem value="sum">sum</MenuItem>
              </Select>
              <Select size="small" value={edgeMode} onChange={(e) => setEdgeMode(String(e.target.value))} sx={{ ...themedSelect, minWidth: 52 }} MenuProps={themedMenuProps} disabled={lockBinning}>
                <MenuItem value="crop">crop</MenuItem>
                <MenuItem value="pad">pad</MenuItem>
                <MenuItem value="error">error</MenuItem>
              </Select>
              <InfoTooltip theme={themeInfo.theme} text={
                <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                  <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Binning Controls</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>Scan:</b> bin factor for scan-space rows × cols</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>Det:</b> bin factor for detector-space rows × cols</Typography>
                  <Typography sx={{ fontSize: 10, mt: 0.5, fontWeight: "bold" }}>Reduce mode</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>mean:</b> average pixel values within each bin</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>sum:</b> sum pixel values within each bin</Typography>
                  <Typography sx={{ fontSize: 10, mt: 0.5, fontWeight: "bold" }}>Edge mode</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>crop:</b> discard remainder pixels at edges</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>pad:</b> zero-pad edges to fill the last bin</Typography>
                  <Typography sx={{ fontSize: 10 }}><b>error:</b> raise error if shape not divisible</Typography>
                </Box>
              } />
            </Box>
          )}

          {/* Row 2: BF / ADF mask sliders */}
          {!hideMask && (
            <Box sx={{ ...controlRow, gap: `${SPACING.SM}px`, width: "auto" }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: "3px", minWidth: 140 }}>
                <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, whiteSpace: "nowrap" }}>BF:{bfRadiusRatio.toFixed(3)}</Typography>
                <Slider value={bfRadiusRatio} min={0.02} max={0.5} step={0.005} onChange={(_, v) => setBfRadiusRatio(v as number)} size="small" sx={{ ...sliderStyles.small, minWidth: 60 }} disabled={lockMask} />
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: "3px", minWidth: 160 }}>
                <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, whiteSpace: "nowrap" }}>ADF in:{adfInnerRatio.toFixed(3)}</Typography>
                <Slider value={adfInnerRatio} min={0.02} max={0.8} step={0.005} onChange={(_, v) => setAdfInnerRatio(v as number)} size="small" sx={{ ...sliderStyles.small, minWidth: 60 }} disabled={lockMask} />
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: "3px", minWidth: 160 }}>
                <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, whiteSpace: "nowrap" }}>ADF out:{adfOuterRatio.toFixed(3)}</Typography>
                <Slider value={adfOuterRatio} min={0.05} max={0.95} step={0.005} onChange={(_, v) => setAdfOuterRatio(v as number)} size="small" sx={{ ...sliderStyles.small, minWidth: 60 }} disabled={lockMask} />
              </Box>
            </Box>
          )}

          {/* Row 3: Display controls + Histogram */}
          {!hideDisplay && (
            <Box sx={{ display: "flex", gap: `${SPACING.SM}px`, alignItems: "flex-start" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: "2px", justifyContent: "center" }}>
                <Box sx={{ ...controlRow, gap: `${SPACING.SM}px`, width: "auto" }}>
                  <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>Color:</Typography>
                  <Select size="small" value={cmap} onChange={(e) => setCmap(String(e.target.value))} sx={{ ...themedSelect, minWidth: 60 }} MenuProps={themedMenuProps} disabled={lockDisplay}>
                    {COLORMAP_NAMES.map((c) => <MenuItem key={c} value={c}>{c}</MenuItem>)}
                  </Select>
                  <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>Log:</Typography>
                  <Switch checked={logScale} onChange={(_, v) => setLogScale(v)} size="small" sx={switchStyles.small} disabled={lockDisplay} />
                  <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>Auto:</Typography>
                  <Switch checked={autoContrast} onChange={(_, v) => setAutoContrast(v)} size="small" sx={switchStyles.small} disabled={lockDisplay} />
                  <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>FFT:</Typography>
                  <Switch checked={showFft} onChange={(_, v) => setShowFft(v)} size="small" sx={switchStyles.small} disabled={lockDisplay} />
                </Box>
              </Box>
              <Histogram
                data={histogramData}
                vminPct={imageVminPct}
                vmaxPct={imageVmaxPct}
                onRangeChange={(min, max) => { if (!lockDisplay) { setImageVminPct(min); setImageVmaxPct(max); } }}
                width={110}
                height={40}
                theme={themeInfo.theme === "dark" ? "dark" : "light"}
                dataMin={histogramDataRange.min}
                dataMax={histogramDataRange.max}
              />
            </Box>
          )}

          {/* Shape + Calibration + Status — table */}
          <Box sx={{ display: "flex", flexDirection: "column", gap: "1px", mt: "2px" }}>
            <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
              shape: ({scanRows}, {scanCols}, {detRows}, {detCols}) → ({binnedScanRows}, {binnedScanCols}, {binnedDetRows}, {binnedDetCols})
              {"  "}({((scanRows * scanCols * detRows * detCols) / Math.max(1, binnedScanRows * binnedScanCols * binnedDetRows * binnedDetCols)).toFixed(1)}× reduction, {(binnedScanRows * binnedScanCols * binnedDetRows * binnedDetCols * 4 / 1024 / 1024).toFixed(1)} MB)
            </Typography>
            <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
              scan:{"  "}({formatNumber(pixelSizeRow, 4)}, {formatNumber(pixelSizeCol, 4)}) → ({formatNumber(binnedPixelSizeRow, 4)}, {formatNumber(binnedPixelSizeCol, 4)}) {pixelUnit}/px
            </Typography>
            <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
              det:{"   "}({formatNumber(kPixelSizeRow, 4)}, {formatNumber(kPixelSizeCol, 4)}) → ({formatNumber(binnedKPixelSizeRow, 4)}, {formatNumber(binnedKPixelSizeCol, 4)}) {kUnit}/px
            </Typography>
            {statusMessage && (
              <Typography sx={{ ...typography.value, color: statusColor }}>
                {statusMessage}
              </Typography>
            )}
          </Box>
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(BinWidget);
