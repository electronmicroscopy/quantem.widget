/**
 * Bin2D - Interactive 2D image binning widget with side-by-side comparison.
 *
 * Layout and styling matches Show2D exactly:
 * - Accent-colored title with InfoTooltip + ControlCustomizer
 * - Top control row: gallery nav + RESET + EXPORT + COPY
 * - Side-by-side Original | Binned canvases with resize handles
 * - Cursor hover readout (top-right, row/col + value)
 * - Stats bar (bgAlt background, "Mean <accent>val</accent>" format)
 * - Control rows with border + controlBg: Bin row + Display row
 * - Histogram with MUI Slider + range labels, right-aligned (110×58)
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { applyLogScale, findDataRange, percentileClip, sliderRange } from "../stats";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen } from "../colormaps";
import { computeHistogramFromBytes } from "../histogram";
import { roundToNiceValue, canvasToPDF } from "../scalebar";
import { computeToolVisibility } from "../tool-parity";
import { ControlCustomizer } from "../control-customizer";
import "./bin2d.css";

// ---------------------------------------------------------------------------
// Constants (matching Show2D / Show3D)
// ---------------------------------------------------------------------------
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const CANVAS_W = 300;
const CANVAS_H = 300;
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
    py: 0,
    "& .MuiSlider-thumb": { width: 8, height: 8 },
    "& .MuiSlider-rail": { height: 2 },
    "& .MuiSlider-track": { height: 2 },
  },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

// ---------------------------------------------------------------------------
// Histogram with Slider + Range Labels (matching Show2D)
// ---------------------------------------------------------------------------
interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
}

function Histogram({ data, vminPct, vmaxPct, onRangeChange, width = 110, height = 40, theme = "dark" }: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);
  const isDark = theme === "dark";
  const colors = isDark
    ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" }
    : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };

  const dataRange = React.useMemo(() => {
    if (!data) return { min: 0, max: 1 };
    return findDataRange(data);
  }, [data]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !bins) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = width * DPR;
    canvas.height = height * DPR;
    ctx.scale(DPR, DPR);
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);
    const n = bins.length;
    const barW = width / n;
    const maxVal = Math.max(...bins, 1e-10);
    const leftEdge = (vminPct / 100) * width;
    const rightEdge = (vmaxPct / 100) * width;
    for (let i = 0; i < n; i++) {
      const x = i * barW;
      const barH = (bins[i] / maxVal) * (height - 2);
      ctx.fillStyle = x >= leftEdge && x <= rightEdge ? colors.barActive : colors.barInactive;
      ctx.fillRect(x, height - barH - 1, barW - 0.5, barH);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);

  const dragging = React.useRef<"left" | "right" | null>(null);
  const onPointerDown = React.useCallback((e: React.PointerEvent) => {
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    dragging.current = Math.abs(x - vminPct) < Math.abs(x - vmaxPct) ? "left" : "right";
    (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
  }, [vminPct, vmaxPct]);
  const onPointerMove = React.useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const x = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
    if (dragging.current === "left") onRangeChange(Math.min(x, vmaxPct - 1), vmaxPct);
    else onRangeChange(vminPct, Math.max(x, vminPct + 1));
  }, [vminPct, vmaxPct, onRangeChange]);
  const onPointerUp = React.useCallback(() => { dragging.current = null; }, []);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, cursor: "ew-resize", border: `1px solid ${colors.border}` }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => {
          const arr = v as number[];
          onRangeChange(arr[0], arr[1]);
        }}
        size="small"
        sx={{ width, ...sliderStyles.small }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>
          {formatNumber(dataRange.min)}
        </Typography>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>
          {formatNumber(dataRange.max)}
        </Typography>
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// InfoTooltip (matching Show2D)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Cursor overlay (matching Show2D: top-right, semi-transparent)
// ---------------------------------------------------------------------------
interface CursorInfo {
  row: number;
  col: number;
  value: number;
}

function CursorReadout({ info, pixelSize }: { info: CursorInfo | null; pixelSize: number }) {
  if (!info) return null;
  const calibrated = pixelSize > 0;
  return (
    <Box sx={{
      position: "absolute", top: 3, right: 3,
      bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15,
      pointerEvents: "none", minWidth: 100, textAlign: "right",
    }}>
      <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
        ({info.row}, {info.col})
        {calibrated && ` = (${(info.row * pixelSize).toFixed(1)}, ${(info.col * pixelSize).toFixed(1)}) Å`}
        {" "}{formatNumber(info.value)}
      </Typography>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Main Widget
// ---------------------------------------------------------------------------
function Bin2DWidget() {
  const { themeInfo, colors: tc } = useTheme();
  const themeColors = { ...tc };

  // Model state
  const [height] = useModelState<number>("height");
  const [width] = useModelState<number>("width");
  const [nImages] = useModelState<number>("n_images");
  const [selectedIdx, setSelectedIdx] = useModelState<number>("selected_idx");
  const [binFactor, setBinFactor] = useModelState<number>("bin_factor");
  const [maxBinFactor] = useModelState<number>("max_bin_factor");
  const [binMode, setBinMode] = useModelState<string>("bin_mode");
  const [edgeMode, setEdgeMode] = useModelState<string>("edge_mode");
  const [binnedHeight] = useModelState<number>("binned_height");
  const [binnedWidth] = useModelState<number>("binned_width");
  const [originalBytesRaw] = useModelState<DataView | null>("original_bytes");
  const [binnedBytesRaw] = useModelState<DataView | null>("binned_bytes");
  const [originalStats] = useModelState<number[]>("original_stats");
  const [binnedStats] = useModelState<number[]>("binned_stats");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [binnedPixelSize] = useModelState<number>("binned_pixel_size");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [labels] = useModelState<string[]>("labels");
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");

  // Tool visibility
  const toolVisibility = React.useMemo(
    () => computeToolVisibility("Bin2D", disabledTools ?? [], hiddenTools ?? []),
    [disabledTools, hiddenTools]
  );
  const hideNavigation = toolVisibility.isHidden("navigation");
  const lockNavigation = toolVisibility.isLocked("navigation");
  const hideStats = toolVisibility.isHidden("stats");
  const lockStats = toolVisibility.isLocked("stats");
  const hideBinning = toolVisibility.isHidden("binning");
  const lockBinning = toolVisibility.isLocked("binning");
  const hideDisplay = toolVisibility.isHidden("display");
  const lockDisplay = toolVisibility.isLocked("display");
  const hideHistogram = toolVisibility.isHidden("histogram");
  const lockHistogram = toolVisibility.isLocked("histogram");
  const hideExport = toolVisibility.isHidden("export");
  const lockExport = toolVisibility.isLocked("export");

  // Extract float32 data
  const originalData = React.useMemo(() => {
    if (!originalBytesRaw) return null;
    return extractFloat32(originalBytesRaw);
  }, [originalBytesRaw]);
  const binnedData = React.useMemo(() => {
    if (!binnedBytesRaw) return null;
    return extractFloat32(binnedBytesRaw);
  }, [binnedBytesRaw]);

  // Histogram state
  const [imageVminPct, setImageVminPct] = React.useState(0);
  const [imageVmaxPct, setImageVmaxPct] = React.useState(100);
  const onHistRange = React.useCallback((min: number, max: number) => {
    if (lockHistogram) return;
    setImageVminPct(min);
    setImageVmaxPct(max);
  }, [lockHistogram]);

  // Auto-contrast
  React.useEffect(() => {
    if (!autoContrast || !originalData) return;
    const { vmin, vmax } = percentileClip(originalData, 2, 98);
    const { min: dmin, max: dmax } = findDataRange(originalData);
    const range = dmax - dmin;
    if (range > 0) {
      setImageVminPct(((vmin - dmin) / range) * 100);
      setImageVmaxPct(((vmax - dmin) / range) * 100);
    }
  }, [autoContrast, originalData]);

  // Canvas refs
  const origCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const binnedCanvasRef = React.useRef<HTMLCanvasElement>(null);

  // Zoom/pan state (synchronized between panels)
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);

  // Cursor hover state
  const [origCursor, setOrigCursor] = React.useState<CursorInfo | null>(null);
  const [binnedCursor, setBinnedCursor] = React.useState<CursorInfo | null>(null);

  // Render a panel
  const renderPanel = React.useCallback(
    (canvas: HTMLCanvasElement | null, data: Float32Array | null, imgW: number, imgH: number) => {
      if (!canvas || !data || imgW === 0 || imgH === 0) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const cssW = canvas.clientWidth || CANVAS_W;
      const cssH = canvas.clientHeight || CANVAS_H;
      canvas.width = cssW * DPR;
      canvas.height = cssH * DPR;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

      ctx.fillStyle = themeColors.canvasBg;
      ctx.fillRect(0, 0, cssW, cssH);

      let processed = new Float32Array(data);
      if (logScale) processed = applyLogScale(processed);
      const { min: dataMin, max: dataMax } = findDataRange(processed);
      const { vmin, vmax } = sliderRange(dataMin, dataMax, imageVminPct, imageVmaxPct);

      const lut = COLORMAPS[cmap] || COLORMAPS["inferno"];
      const offscreen = renderToOffscreen(processed, imgW, imgH, lut, vmin, vmax);

      const scaleX = cssW / imgW;
      const scaleY = cssH / imgH;
      const fitScale = Math.min(scaleX, scaleY);
      const effectiveZoom = fitScale * zoom;
      const offsetX = (cssW - imgW * effectiveZoom) / 2 + panX;
      const offsetY = (cssH - imgH * effectiveZoom) / 2 + panY;

      ctx.imageSmoothingEnabled = zoom < 4;
      ctx.drawImage(offscreen, offsetX, offsetY, imgW * effectiveZoom, imgH * effectiveZoom);

      // Scale bar
      if (pixelSize > 0 || binnedPixelSize > 0) {
        const ps = canvas === origCanvasRef.current ? pixelSize : binnedPixelSize;
        if (ps > 0) {
          const targetBarPx = 60;
          const physicalLength = (targetBarPx / effectiveZoom) * ps;
          const niceLength = roundToNiceValue(physicalLength);
          const barPx = (niceLength / ps) * effectiveZoom;
          const margin = 12;
          const barThickness = 5;
          const barX = cssW - margin - barPx;
          const barY = cssH - margin - barThickness;
          ctx.fillStyle = "white";
          ctx.shadowColor = "rgba(0,0,0,0.5)";
          ctx.shadowBlur = 2;
          ctx.fillRect(barX, barY, barPx, barThickness);
          let unit = "Å";
          let displayVal = niceLength;
          if (niceLength >= 10) { displayVal = niceLength / 10; unit = "nm"; }
          const label = `${displayVal % 1 === 0 ? displayVal.toFixed(0) : displayVal.toFixed(1)} ${unit}`;
          ctx.font = "16px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(label, barX + barPx / 2, barY - 4);
          ctx.shadowColor = "transparent";
          ctx.shadowBlur = 0;
        }
      }

      // Zoom indicator
      ctx.fillStyle = "white";
      ctx.shadowColor = "rgba(0,0,0,0.5)";
      ctx.shadowBlur = 2;
      ctx.font = "14px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(`${zoom.toFixed(1)}×`, 8, cssH - 8);
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
    },
    [zoom, panX, panY, cmap, logScale, imageVminPct, imageVmaxPct, pixelSize, binnedPixelSize, themeColors]
  );

  // Render both panels
  React.useEffect(() => {
    renderPanel(origCanvasRef.current, originalData, width, height);
  }, [renderPanel, originalData, width, height]);

  React.useEffect(() => {
    renderPanel(binnedCanvasRef.current, binnedData, binnedWidth, binnedHeight);
  }, [renderPanel, binnedData, binnedWidth, binnedHeight]);

  // Mouse handlers for zoom/pan + cursor
  const isPanning = React.useRef(false);
  const lastMouse = React.useRef({ x: 0, y: 0 });

  const handleWheel = React.useCallback((e: WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(z => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * delta)));
  }, []);

  const handleMouseDown = React.useCallback((e: React.MouseEvent) => {
    isPanning.current = true;
    lastMouse.current = { x: e.clientX, y: e.clientY };
  }, []);

  const handleMouseUp = React.useCallback(() => { isPanning.current = false; }, []);

  // Cursor computation helper
  const computeCursor = React.useCallback((e: React.MouseEvent, imgW: number, imgH: number, data: Float32Array | null): CursorInfo | null => {
    if (!data) return null;
    const canvas = e.currentTarget as HTMLCanvasElement;
    const rect = canvas.getBoundingClientRect();
    const cssW = rect.width;
    const cssH = rect.height;
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;
    const scaleX = cssW / imgW;
    const scaleY = cssH / imgH;
    const fitScale = Math.min(scaleX, scaleY);
    const effectiveZoom = fitScale * zoom;
    const offsetX = (cssW - imgW * effectiveZoom) / 2 + panX;
    const offsetY = (cssH - imgH * effectiveZoom) / 2 + panY;
    const imgX = (canvasX - offsetX) / effectiveZoom;
    const imgY = (canvasY - offsetY) / effectiveZoom;
    const col = Math.floor(imgX);
    const row = Math.floor(imgY);
    if (row >= 0 && row < imgH && col >= 0 && col < imgW) {
      return { row, col, value: data[row * imgW + col] };
    }
    return null;
  }, [zoom, panX, panY]);

  const handleOrigMouseMove = React.useCallback((e: React.MouseEvent) => {
    if (isPanning.current) {
      const dx = e.clientX - lastMouse.current.x;
      const dy = e.clientY - lastMouse.current.y;
      lastMouse.current = { x: e.clientX, y: e.clientY };
      setPanX(p => p + dx);
      setPanY(p => p + dy);
    }
    setOrigCursor(computeCursor(e, width, height, originalData));
  }, [computeCursor, width, height, originalData]);

  const handleBinnedMouseMove = React.useCallback((e: React.MouseEvent) => {
    if (isPanning.current) {
      const dx = e.clientX - lastMouse.current.x;
      const dy = e.clientY - lastMouse.current.y;
      lastMouse.current = { x: e.clientX, y: e.clientY };
      setPanX(p => p + dx);
      setPanY(p => p + dy);
    }
    setBinnedCursor(computeCursor(e, binnedWidth, binnedHeight, binnedData));
  }, [computeCursor, binnedWidth, binnedHeight, binnedData]);

  const resetView = React.useCallback(() => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, []);

  // Wheel event listeners
  React.useEffect(() => {
    const orig = origCanvasRef.current;
    const binned = binnedCanvasRef.current;
    const opts: AddEventListenerOptions = { passive: false };
    if (orig) orig.addEventListener("wheel", handleWheel, opts);
    if (binned) binned.addEventListener("wheel", handleWheel, opts);
    return () => {
      if (orig) orig.removeEventListener("wheel", handleWheel);
      if (binned) binned.removeEventListener("wheel", handleWheel);
    };
  }, [handleWheel]);

  // Keyboard
  const handleKeyDown = React.useCallback((e: React.KeyboardEvent) => {
    if (e.key === "r" || e.key === "R") resetView();
    if (e.key === "ArrowLeft" && nImages > 1) setSelectedIdx(Math.max(0, selectedIdx - 1));
    if (e.key === "ArrowRight" && nImages > 1) setSelectedIdx(Math.min(nImages - 1, selectedIdx + 1));
  }, [resetView, nImages, selectedIdx, setSelectedIdx]);

  // Canvas resize (matching Show2D pattern)
  const [canvasW, setCanvasW] = React.useState(CANVAS_W);
  const [canvasH, setCanvasH] = React.useState(CANVAS_H);
  const resizeStartRef = React.useRef<{ x: number; y: number; w: number; h: number } | null>(null);

  const handleCanvasResizeStart = React.useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    resizeStartRef.current = { x: e.clientX, y: e.clientY, w: canvasW, h: canvasH };
    const onMove = (ev: MouseEvent) => {
      if (!resizeStartRef.current) return;
      const dx = ev.clientX - resizeStartRef.current.x;
      const dy = ev.clientY - resizeStartRef.current.y;
      const delta = Math.max(dx, dy);
      setCanvasW(Math.max(150, resizeStartRef.current.w + delta));
      setCanvasH(Math.max(150, resizeStartRef.current.h + delta));
    };
    const onUp = () => {
      resizeStartRef.current = null;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [canvasW, canvasH]);

  // Export
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);
  const handleExport = React.useCallback((format: "png" | "pdf") => {
    setExportAnchor(null);
    const canvas = binnedCanvasRef.current;
    if (!canvas) return;
    if (format === "pdf") {
      canvasToPDF(canvas).then(blob => downloadBlob(blob, `bin2d_${binFactor}x.pdf`));
    } else {
      canvas.toBlob(blob => {
        if (blob) downloadBlob(blob, `bin2d_${binFactor}x.png`);
      });
    }
  }, [binFactor]);

  const handleCopy = React.useCallback(() => {
    const canvas = binnedCanvasRef.current;
    if (!canvas) return;
    canvas.toBlob(blob => {
      if (blob) navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    });
  }, []);

  // Bin factor options: powers of 2 only (2, 4, 8, 16)
  const factorOptions = React.useMemo(() => {
    const opts: number[] = [];
    for (let f = 2; f <= Math.min(maxBinFactor, 16); f *= 2) {
      opts.push(f);
    }
    return opts;
  }, [maxBinFactor]);

  // Themed select styling (matching Show2D)
  const themedSelect = {
    fontSize: 10,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };
  const themedMenuProps = {
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } },
    ...upwardMenuProps,
  };

  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;
  const currentLabel = labels && labels.length > selectedIdx ? labels[selectedIdx] : "";

  // File size computation (float32 = 4 bytes per pixel)
  const originalSizeBytes = nImages * height * width * 4;
  const binnedSizeBytes = nImages * binnedHeight * binnedWidth * 4;
  const reductionFactor = originalSizeBytes > 0 && binnedSizeBytes > 0 ? Math.round(originalSizeBytes / binnedSizeBytes) : 1;

  // Resize handle (matching Show2D — with borderRadius)
  const resizeHandle = (
    <Box
      onMouseDown={handleCanvasResizeStart}
      sx={{
        position: "absolute", bottom: 0, right: 0, width: 16, height: 16,
        cursor: "nwse-resize", opacity: 0.6, pointerEvents: "auto",
        background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
        borderRadius: "0 0 4px 0",
        "&:hover": { opacity: 1 },
      }}
    />
  );

  return (
    <Box
      className="bin2d-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, width: "fit-content", outline: "none" }}
    >
      {/* Title row (matching Show2D) */}
      <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block", height: 16, lineHeight: "16px", overflow: "hidden" }}>
        {title || "Bin2D"}
        <InfoTooltip
          text={
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
              Side-by-side 2D binning. Scroll to zoom, double-click to reset. R resets view. Arrow keys navigate gallery.
            </Typography>
          }
          theme={themeInfo.theme}
        />
        <ControlCustomizer
          widgetName="Bin2D"
          hiddenTools={hiddenTools ?? []}
          setHiddenTools={setHiddenTools}
          disabledTools={disabledTools ?? []}
          setDisabledTools={setDisabledTools}
          themeColors={themeColors}
        />
      </Typography>

      {/* Top control row: gallery nav + Reset + Export + Copy (matching Show2D) */}
      <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
        {/* Gallery navigation */}
        {nImages > 1 && !hideNavigation && (
          <>
            <Button size="small" sx={compactButton} disabled={selectedIdx <= 0 || lockNavigation} onClick={() => setSelectedIdx(selectedIdx - 1)}>◀</Button>
            <Typography sx={typography.value}>{selectedIdx + 1}/{nImages}</Typography>
            <Button size="small" sx={compactButton} disabled={selectedIdx >= nImages - 1 || lockNavigation} onClick={() => setSelectedIdx(selectedIdx + 1)}>▶</Button>
            {currentLabel && (
              <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted }}>{currentLabel}</Typography>
            )}
          </>
        )}
        <Box sx={{ flex: 1 }} />
        <Button size="small" sx={compactButton} disabled={!needsReset} onClick={resetView}>Reset</Button>
        {!hideExport && (
          <>
            <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} disabled={lockExport} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }}>Export</Button>
            <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
              <MenuItem disabled={lockExport} onClick={() => handleExport("pdf")} sx={{ fontSize: 12 }}>PDF</MenuItem>
              <MenuItem disabled={lockExport} onClick={() => handleExport("png")} sx={{ fontSize: 12 }}>PNG</MenuItem>
            </Menu>
            <Button size="small" sx={compactButton} disabled={lockExport} onClick={handleCopy}>Copy</Button>
          </>
        )}
      </Stack>

      {/* Side-by-side canvases */}
      <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
        {/* Original panel */}
        <Box>
          <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, mb: `${SPACING.XS}px` }}>
            Original ({height}&times;{width}) &middot; {formatBytes(originalSizeBytes)}
          </Typography>
          <Box sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, width: canvasW, height: canvasH }}>
            <canvas
              ref={origCanvasRef}
              style={{ width: canvasW, height: canvasH, display: "block", cursor: isPanning.current ? "grabbing" : "crosshair", imageRendering: "pixelated" }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleOrigMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={() => { handleMouseUp(); setOrigCursor(null); }}
              onDoubleClick={resetView}
            />
            <CursorReadout info={origCursor} pixelSize={pixelSize} />
            {resizeHandle}
          </Box>
        </Box>
        {/* Binned panel */}
        <Box>
          <Typography sx={{ ...typography.labelSmall, color: themeColors.textMuted, mb: `${SPACING.XS}px` }}>
            Binned &times;{binFactor} ({binnedHeight}&times;{binnedWidth}) &middot; {formatBytes(binnedSizeBytes)}
            {reductionFactor > 1 && <Box component="span" sx={{ color: themeColors.accent, ml: 0.5 }}>&darr;{reductionFactor}&times;</Box>}
            {binMode === "sum" && <Box component="span" sx={{ color: "#f0a040", ml: 0.5 }}>&Sigma;</Box>}
          </Typography>
          <Box sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, width: canvasW, height: canvasH }}>
            <canvas
              ref={binnedCanvasRef}
              style={{ width: canvasW, height: canvasH, display: "block", cursor: isPanning.current ? "grabbing" : "crosshair", imageRendering: "pixelated" }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleBinnedMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={() => { handleMouseUp(); setBinnedCursor(null); }}
              onDoubleClick={resetView}
            />
            <CursorReadout info={binnedCursor} pixelSize={binnedPixelSize} />
            {resizeHandle}
          </Box>
        </Box>
      </Box>

      {/* Stats bar (matching Show2D: bgAlt, "Mean <accent>val</accent>") */}
      {!hideStats && showStats && (
        <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", gap: `${SPACING.SM}px`, opacity: lockStats ? 0.7 : 1 }}>
          {/* Original stats */}
          <Box sx={{ flex: 1, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, whiteSpace: "nowrap", overflow: "hidden" }}>
            {originalStats && originalStats.length === 4 && (
              <>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(originalStats[0])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(originalStats[1])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(originalStats[2])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(originalStats[3])}</Box></Typography>
              </>
            )}
          </Box>
          {/* Binned stats */}
          <Box sx={{ flex: 1, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, whiteSpace: "nowrap", overflow: "hidden" }}>
            {binnedStats && binnedStats.length === 4 && (
              <>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(binnedStats[0])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(binnedStats[1])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(binnedStats[2])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(binnedStats[3])}</Box></Typography>
              </>
            )}
          </Box>
        </Box>
      )}

      {/* Controls: two rows left + histogram right (matching Show2D layout) */}
      {showControls && (
        <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, boxSizing: "border-box" }}>
          <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
            {/* Left: control rows stacked */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: Bin + Mode + Edge */}
              {!hideBinning && (
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockBinning ? 0.5 : 1, pointerEvents: lockBinning ? "none" : "auto" }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Bin:</Typography>
                  <Select disabled={lockBinning} size="small" value={binFactor} onChange={e => setBinFactor(Number(e.target.value))} sx={{ ...themedSelect, minWidth: 50 }} MenuProps={themedMenuProps}>
                    {factorOptions.map(f => (
                      <MenuItem key={f} value={f}>{f}&times;</MenuItem>
                    ))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Mode:</Typography>
                  <Select disabled={lockBinning} size="small" value={binMode} onChange={e => setBinMode(e.target.value)} sx={{ ...themedSelect, minWidth: 55 }} MenuProps={themedMenuProps}>
                    <MenuItem value="mean">Mean</MenuItem>
                    <MenuItem value="sum">Sum</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Edge:</Typography>
                  <Select disabled={lockBinning} size="small" value={edgeMode} onChange={e => setEdgeMode(e.target.value)} sx={{ ...themedSelect, minWidth: 55 }} MenuProps={themedMenuProps}>
                    <MenuItem value="crop">Crop</MenuItem>
                    <MenuItem value="pad">Pad</MenuItem>
                  </Select>
                </Box>
              )}
              {/* Row 2: Scale + Color + Auto */}
              {!hideDisplay && (
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                  <Select disabled={lockDisplay} value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                  <Select disabled={lockDisplay} size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} sx={{ ...themedSelect, minWidth: 60 }} MenuProps={themedMenuProps}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                  <Switch checked={autoContrast} onChange={(e) => { if (!lockDisplay) setAutoContrast(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                </Box>
              )}
            </Box>
            {/* Right: Histogram with slider + range labels (matching Show2D) */}
            {!hideHistogram && (
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
                <Histogram
                  data={originalData}
                  vminPct={imageVminPct}
                  vmaxPct={imageVmaxPct}
                  onRangeChange={onHistRange}
                  width={110}
                  height={58}
                  theme={themeInfo.theme === "dark" ? "dark" : "light"}
                />
              </Box>
            )}
          </Box>
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(Bin2DWidget);
