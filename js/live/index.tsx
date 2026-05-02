/**
 * Live — Real-time incremental image viewer.
 *
 * Layout: filmstrip sidebar (2-col thumbnail grid) on left + main inspect panel on right.
 * Push model: frames arrive via _push_bytes / _batch_bytes, newest frame auto-selected.
 * GPU colormap engine (zero-copy ImageBitmap path) used for main canvas when available.
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawColorbar, exportFigure, canvasToPDF } from "../scalebar";
import Menu from "@mui/material/Menu";
import { getWebGPUFFT, WebGPUFFT, fft2d, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2, applyHannWindow2D } from "../webgpu-fft";
import {
  COLORMAPS,
  COLORMAP_NAMES,
  renderToOffscreen,
  renderToOffscreenReuse,
  GPUColormapEngine,
  getGPUColormapEngine,
} from "../colormaps";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import {
  findDataRange,
  applyLogScale,
  percentileClip,
  sliderRange,
  computeStats,
} from "../stats";
import "./live.css";

// ============================================================================
// Design constants — exact copies from Show2D/Show3D
// ============================================================================
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DPR = window.devicePixelRatio || 1;
const MIN_CANVAS = 256;
const THUMB_SIZE = 64;
const THUMB_COLS = 2;

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
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};
const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

// ============================================================================
// InfoTooltip — exact copy from Show2D
// ============================================================================
function InfoTooltip({
  text,
  theme = "dark",
}: {
  text: React.ReactNode;
  theme?: "light" | "dark";
}) {
  const isDark = theme === "dark";
  const content =
    typeof text === "string" ? (
      <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    ) : (
      text
    );
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

// ============================================================================
// Histogram — exact copy from Show2D (GPU precomputed bins supported)
// ============================================================================
interface HistogramProps {
  data: Float32Array | null;
  precomputedBins?: number[] | null;
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
  data,
  precomputedBins,
  vminPct,
  vmaxPct,
  onRangeChange,
  width = 110,
  height = 40,
  theme = "dark",
  dataMin = 0,
  dataMax = 1,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const cpuBins = React.useMemo(
    () => (precomputedBins ? null : computeHistogramFromBytes(data)),
    [data, precomputedBins],
  );
  const bins = precomputedBins || cpuBins || new Array(256).fill(0);
  const isDark = theme === "dark";
  const colors = isDark
    ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" }
    : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };

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
      ctx.fillStyle =
        i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
      ctx.fillRect(
        i * barWidth + 0.5,
        height - barHeight,
        Math.max(1, barWidth - 1),
        barHeight,
      );
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}` }}
      />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => {
          const [newMin, newMax] = v as number[];
          onRangeChange(
            Math.min(newMin, newMax - 1),
            Math.max(newMax, newMin + 1),
          );
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
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography
          sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}
        >
          {(() => {
            const v = dataMin + (vminPct / 100) * (dataMax - dataMin);
            return v >= 1000 ? v.toExponential(1) : v.toFixed(1);
          })()}
        </Typography>
        <Typography
          sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}
        >
          {(() => {
            const v = dataMin + (vmaxPct / 100) * (dataMax - dataMin);
            return v >= 1000 ? v.toExponential(1) : v.toFixed(1);
          })()}
        </Typography>
      </Box>
    </Box>
  );
}

// ============================================================================
// ROI / Profile helpers
// ============================================================================
const ROI_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];
const RESIZE_HIT_AREA_PX = 10;

type ROIItem = { row: number; col: number; shape: string; radius: number; radius_inner: number; width: number; height: number; color: string; line_width: number; highlight: boolean };

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
// Types
// ============================================================================
interface FrameEntry {
  data: Float32Array;
  height: number;
  width: number;
  label: string;
  thumbCanvas: HTMLCanvasElement | null;
}

// ============================================================================
// Main Component
// ============================================================================
const render = createRender(() => {
  const { themeInfo, colors: tc } = useTheme();
  const themeColors = {
    ...tc,
    accentGreen: themeInfo.theme === "dark" ? "#66bb6a" : "#2e7d32",
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
    PaperProps: {
      sx: {
        bgcolor: themeColors.controlBg,
        color: themeColors.text,
        border: `1px solid ${themeColors.border}`,
      },
    },
  };

  // ─── Model state ─────────────────────────────────────────────────
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [bufferSize] = useModelState<number>("buffer_size");
  const [nFrames] = useModelState<number>("n_frames");
  const [showControls] = useModelState<boolean>("show_controls");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [showStats] = useModelState<boolean>("show_stats");
  const [_scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftWindow] = useModelState<boolean>("fft_window");
  const [roiActive, setRoiActive] = useModelState<boolean>("roi_active");
  const [roiList, setRoiList] = useModelState<{ [key: string]: unknown }[]>("roi_list");
  const [roiSelectedIdx, setRoiSelectedIdx] = useModelState<number>("roi_selected_idx");
  const [profileLine, setProfileLine] = useModelState<{ row: number; col: number }[]>("profile_line");

  // Push / Batch traits
  const [pushBytes] = useModelState<DataView>("_push_bytes");
  const [pushHeight] = useModelState<number>("_push_height");
  const [pushWidth] = useModelState<number>("_push_width");
  const [pushLabel] = useModelState<string>("_push_label");
  const [pushCounter] = useModelState<number>("_push_counter");
  const [batchBytes] = useModelState<DataView>("_batch_bytes");
  const [batchDims] = useModelState<number[][]>("_batch_dims");
  const [batchLabels] = useModelState<string[]>("_batch_labels");
  const [batchCounter] = useModelState<number>("_batch_counter");

  // ─── Local state ─────────────────────────────────────────────────
  const framesRef = React.useRef<FrameEntry[]>([]);
  const [frameCount, setFrameCount] = React.useState(0);
  const [selectedIdx, setSelectedIdx] = React.useState(0);
  const [canvasSize, setCanvasSize] = React.useState(500);

  // Histogram / contrast
  const [vminPct, setVminPct] = React.useState(0);
  const [vmaxPct, setVmaxPct] = React.useState(100);
  const vminPctRef = React.useRef(0);
  const vmaxPctRef = React.useRef(100);
  const sliderRafRef = React.useRef(0);
  const [histData, setHistData] = React.useState<Float32Array | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{
    min: number;
    max: number;
  }>({ min: 0, max: 1 });

  // Stats
  const [statsMean, setStatsMean] = React.useState(0);
  const [statsMin, setStatsMin] = React.useState(0);
  const [statsMax, setStatsMax] = React.useState(0);
  const [statsStd, setStatsStd] = React.useState(0);

  // Cursor
  const [cursorInfo, setCursorInfo] = React.useState<{
    row: number;
    col: number;
    value: number;
  } | null>(null);

  // FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuFftReady, setGpuFftReady] = React.useState(false);
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const fftImgDataRef = React.useRef<ImageData | null>(null);
  const fftMagRef = React.useRef<Float32Array | null>(null);
  const [fftComputing, setFftComputing] = React.useState(false);
  const [_fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const [fftVminPct] = React.useState(0);
  const [fftVmaxPct] = React.useState(100);
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);
  const [profileHeight] = React.useState(60);

  // Colorbar
  const [showColorbar, setShowColorbar] = React.useState(false);
  const colorbarVminRef = React.useRef(0);
  const colorbarVmaxRef = React.useRef(1);

  // Export
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);

  // ROI interaction state
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false);
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const resizeAspectRef = React.useRef<number | null>(null);
  const [newRoiShape, setNewRoiShape] = React.useState<"square" | "rectangle" | "circle" | "annular">("square");
  const clickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [draggingProfileEndpoint, setDraggingProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingProfileLine, setIsDraggingProfileLine] = React.useState(false);
  const profileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);

  // Profile points (synced to model)
  const profilePoints = profileLine || [];
  const setProfilePoints = (pts: { row: number; col: number }[]) => setProfileLine(pts as { row: number; col: number }[]);

  // ROI helpers
  const roiListTyped = (roiList || []) as ROIItem[];
  const selectedRoi = roiSelectedIdx >= 0 && roiSelectedIdx < roiListTyped.length ? roiListTyped[roiSelectedIdx] : null;

  const updateSelectedRoi = (updates: Partial<ROIItem>) => {
    if (roiSelectedIdx < 0 || !roiListTyped) return;
    const newList = [...roiListTyped];
    newList[roiSelectedIdx] = { ...newList[roiSelectedIdx], ...updates };
    setRoiList(newList as { [key: string]: unknown }[]);
  };

  // Zoom / pan
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const zoomRef = React.useRef(1);
  const panXRef = React.useRef(0);
  const panYRef = React.useRef(0);
  zoomRef.current = zoom;
  panXRef.current = panX;
  panYRef.current = panY;
  const panStartRef = React.useRef<{
    x: number;
    y: number;
    pX: number;
    pY: number;
  } | null>(null);
  const dragRef = React.useRef({ wasDrag: false });

  // Canvas resize
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const resizeStartRef = React.useRef<{
    x: number;
    y: number;
    size: number;
  } | null>(null);

  // Refs
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const canvasContainerRef = React.useRef<HTMLDivElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const mainOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const mainImgDataRef = React.useRef<ImageData | null>(null);
  const rootRef = React.useRef<HTMLDivElement>(null);
  const thumbListRef = React.useRef<HTMLDivElement>(null);
  const [offscreenVersion, setOffscreenVersion] = React.useState(0);

  // Ref mirrors for async callbacks
  const logScaleRef = React.useRef(logScale);
  logScaleRef.current = logScale;
  const cmapRef = React.useRef(cmap);
  cmapRef.current = cmap;
  const autoContrastRef = React.useRef(autoContrast);
  autoContrastRef.current = autoContrast;
  const selectedIdxRef = React.useRef(selectedIdx);
  selectedIdxRef.current = selectedIdx;
  const dataRangeRef = React.useRef<{ min: number; max: number }>({
    min: 0,
    max: 1,
  });

  // ─── GPU colormap engine ─────────────────────────────────────────
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const gpuCmapReadyRef = React.useRef(false);

  React.useEffect(() => {
    getGPUColormapEngine().then((engine) => {
      if (engine) {
        gpuCmapRef.current = engine;
        gpuCmapReadyRef.current = true;
        console.log("[Live] WebGPU colormap engine initialized");
      } else {
        console.log("[Live] WebGPU unavailable — using CPU fallback");
      }
    });
    getWebGPUFFT().then((fft) => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuFftReady(true);
        console.log("[Live] WebGPU FFT initialized");
      }
    });
  }, []);

  // ─── Watching indicator: true when new frame arrived in last 3s ──
  const [isWatching, setIsWatching] = React.useState(false);
  const watchTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const markWatching = React.useCallback(() => {
    setIsWatching(true);
    if (watchTimerRef.current) clearTimeout(watchTimerRef.current);
    watchTimerRef.current = setTimeout(() => setIsWatching(false), 3000);
  }, []);

  // ─── Build thumbnail canvas for a frame ─────────────────────────
  const buildThumb = React.useCallback(
    (data: Float32Array, w: number, h: number): HTMLCanvasElement => {
      const lut = COLORMAPS[cmapRef.current] || COLORMAPS.inferno;
      const range = findDataRange(data);
      const src = logScaleRef.current ? applyLogScale(data) : data;
      const logRange = logScaleRef.current
        ? {
            min: Math.log1p(Math.max(range.min, 0)),
            max: Math.log1p(Math.max(range.max, 0)),
          }
        : range;
      return (
        renderToOffscreen(src, w, h, lut, logRange.min, logRange.max) ||
        (() => {
          const c = document.createElement("canvas");
          c.width = w;
          c.height = h;
          return c;
        })()
      );
    },
    [],
  );

  // ─── Accumulate single-push frames ──────────────────────────────
  const prevPushCounterRef = React.useRef(0);
  React.useEffect(() => {
    if (pushCounter === prevPushCounterRef.current) return;
    prevPushCounterRef.current = pushCounter;
    if (!pushBytes || pushHeight === 0 || pushWidth === 0) return;
    const floats = extractFloat32(pushBytes);
    if (!floats || floats.length !== pushHeight * pushWidth) return;
    const data = new Float32Array(floats);
    const thumbCanvas = buildThumb(data, pushWidth, pushHeight);
    const entry: FrameEntry = {
      data,
      height: pushHeight,
      width: pushWidth,
      label: pushLabel || `Frame ${framesRef.current.length + 1}`,
      thumbCanvas,
    };
    // Newest frame at index 0 (top of list)
    framesRef.current = [entry, ...framesRef.current];
    if (framesRef.current.length > bufferSize) {
      framesRef.current = framesRef.current.slice(0, bufferSize);
    }
    setFrameCount(framesRef.current.length);
    setSelectedIdx(0);
    if (thumbListRef.current) thumbListRef.current.scrollTop = 0;
    markWatching();
  }, [pushCounter]);

  // ─── Accumulate batch frames ─────────────────────────────────────
  const prevBatchCounterRef = React.useRef(0);
  React.useEffect(() => {
    if (batchCounter === prevBatchCounterRef.current) return;
    prevBatchCounterRef.current = batchCounter;
    if (!batchBytes || !batchDims || batchDims.length === 0) return;
    const allFloats = extractFloat32(batchBytes);
    if (!allFloats || allFloats.length === 0) return;
    const newEntries: FrameEntry[] = [];
    let offset = 0;
    for (let i = 0; i < batchDims.length; i++) {
      const [h, w] = batchDims[i];
      const count = h * w;
      if (offset + count > allFloats.length) break;
      const slice = allFloats.subarray(offset, offset + count);
      const data = new Float32Array(slice);
      const thumbCanvas = buildThumb(data, w, h);
      const label =
        batchLabels && i < batchLabels.length ? batchLabels[i] : `Frame ${i + 1}`;
      newEntries.push({ data, height: h, width: w, label, thumbCanvas });
      offset += count;
    }
    // Prepend newest-first (batch arrives oldest→newest, reverse so [0]=newest)
    newEntries.reverse();
    framesRef.current = [...newEntries, ...framesRef.current];
    if (framesRef.current.length > bufferSize) {
      framesRef.current = framesRef.current.slice(0, bufferSize);
    }
    setFrameCount(framesRef.current.length);
    setSelectedIdx(0);
    if (thumbListRef.current) thumbListRef.current.scrollTop = 0;
    markWatching();
  }, [batchCounter]);

  // ─── Displayed frame ────────────────────────────────────────────
  const displayIdx = Math.min(selectedIdx, Math.max(0, framesRef.current.length - 1));
  const displayFrame =
    frameCount > 0 && framesRef.current.length > displayIdx
      ? framesRef.current[displayIdx]
      : null;

  // Layout: compute canvas display dimensions preserving aspect ratio
  const displayW = displayFrame ? displayFrame.width : canvasSize;
  const displayH = displayFrame ? displayFrame.height : canvasSize;
  const displayScale =
    canvasSize / Math.max(displayW, displayH, 1);
  const canvasW = Math.round(displayW * displayScale);
  const canvasH = Math.round(displayH * displayScale);

  // ─── Compute stats + histogram on frame/display change ──────────
  React.useEffect(() => {
    if (!displayFrame) return;
    const raw = displayFrame.data;
    const rawRange = findDataRange(raw);
    const processed = logScale ? applyLogScale(raw) : raw;
    const logRange = logScale
      ? {
          min: Math.log1p(Math.max(rawRange.min, 0)),
          max: Math.log1p(Math.max(rawRange.max, 0)),
        }
      : rawRange;
    setImageDataRange(logRange);
    dataRangeRef.current = logRange;
    setHistData(processed);

    // Stats
    const s = computeStats(raw);
    setStatsMean(s.mean);
    setStatsMin(s.min);
    setStatsMax(s.max);
    setStatsStd(s.std);
  }, [displayIdx, frameCount, logScale]);

  // ─── Initialize / resize offscreen canvas ───────────────────────
  React.useEffect(() => {
    if (!displayFrame) return;
    const { width, height } = displayFrame;
    if (
      !mainOffscreenRef.current ||
      mainOffscreenRef.current.width !== width ||
      mainOffscreenRef.current.height !== height
    ) {
      const c = document.createElement("canvas");
      c.width = width;
      c.height = height;
      mainOffscreenRef.current = c;
      mainImgDataRef.current = c.getContext("2d")!.createImageData(width, height);
    }
  }, [displayIdx, frameCount]);

  // ─── GPU upload + render offscreen ───────────────────────────────
  const renderOffscreen = React.useCallback(() => {
    if (!displayFrame) return;
    const { data, width, height } = displayFrame;
    const lut = COLORMAPS[cmapRef.current] || COLORMAPS.inferno;
    const rawRange = findDataRange(data);
    const logRange = {
      min: Math.log1p(Math.max(rawRange.min, 0)),
      max: Math.log1p(Math.max(rawRange.max, 0)),
    };
    const range = logScaleRef.current ? logRange : rawRange;

    let vmin: number, vmax: number;
    if (autoContrastRef.current) {
      const processed = logScaleRef.current ? applyLogScale(data) : data;
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else if (vminPctRef.current > 0 || vmaxPctRef.current < 100) {
      ({ vmin, vmax } = sliderRange(
        range.min,
        range.max,
        vminPctRef.current,
        vmaxPctRef.current,
      ));
    } else {
      vmin = range.min;
      vmax = range.max;
    }

    // Track for colorbar
    colorbarVminRef.current = vmin;
    colorbarVmaxRef.current = vmax;

    // Ensure offscreen
    if (
      !mainOffscreenRef.current ||
      mainOffscreenRef.current.width !== width ||
      mainOffscreenRef.current.height !== height
    ) {
      const c = document.createElement("canvas");
      c.width = width;
      c.height = height;
      mainOffscreenRef.current = c;
      mainImgDataRef.current = c.getContext("2d")!.createImageData(width, height);
    }

    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current) {
      engine.uploadData(0, data, width, height);
      engine.uploadLUT(cmapRef.current, lut);
      const bitmaps = engine.renderSlotsToImageBitmap(
        [0],
        [{ vmin, vmax }],
        logScaleRef.current,
      );
      if (bitmaps && bitmaps[0]) {
        mainOffscreenRef.current.getContext("2d")!.drawImage(bitmaps[0], 0, 0);
        setOffscreenVersion((v) => v + 1);
        return;
      }
      // Fallback within GPU path
    }
    // CPU path
    const processed = logScaleRef.current ? applyLogScale(data) : data;
    if (mainOffscreenRef.current && mainImgDataRef.current) {
      renderToOffscreenReuse(
        processed,
        lut,
        vmin,
        vmax,
        mainOffscreenRef.current,
        mainImgDataRef.current,
      );
    }
    setOffscreenVersion((v) => v + 1);
  }, [displayIdx, frameCount]);

  // Re-render offscreen when display/contrast params change
  React.useEffect(() => {
    renderOffscreen();
  }, [displayIdx, frameCount, cmap, logScale, autoContrast, vminPct, vmaxPct]);

  // ─── Fast slider path (bypass React effect batching) ─────────────
  const handleRangeChange = React.useCallback(
    (newMin: number, newMax: number) => {
      vminPctRef.current = newMin;
      vmaxPctRef.current = newMax;
      setVminPct(newMin);
      setVmaxPct(newMax);
      // Fire on next rAF — coalesces rapid drag events
      cancelAnimationFrame(sliderRafRef.current);
      sliderRafRef.current = requestAnimationFrame(() => {
        renderOffscreen();
      });
    },
    [renderOffscreen],
  );

  // ─── Draw effect: zoom/pan → cheap drawImage from cached offscreen ─
  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const offscreen = mainOffscreenRef.current;
    if (!canvas || !offscreen || !displayFrame) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = canvasW;
    canvas.height = canvasH;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvasW, canvasH);

    if (zoom !== 1 || panX !== 0 || panY !== 0) {
      ctx.save();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);
      ctx.drawImage(
        offscreen,
        0,
        0,
        displayFrame.width,
        displayFrame.height,
        0,
        0,
        canvasW,
        canvasH,
      );
      ctx.restore();
    } else {
      ctx.drawImage(
        offscreen,
        0,
        0,
        displayFrame.width,
        displayFrame.height,
        0,
        0,
        canvasW,
        canvasH,
      );
    }
  }, [offscreenVersion, canvasW, canvasH, zoom, panX, panY, displayFrame]);

  // ─── Overlay: scale bar + colorbar + zoom indicator + ROI + profile
  React.useEffect(() => {
    const uiCanvas = overlayRef.current;
    if (!uiCanvas || !displayFrame) return;
    const ctx = uiCanvas.getContext("2d");
    if (!ctx) return;
    uiCanvas.width = Math.round(canvasW * DPR);
    uiCanvas.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);

    // Scale bar
    if (pixelSize > 0) {
      const unit = "Å" as const;
      drawScaleBarHiDPI(uiCanvas, DPR, zoom, pixelSize, unit, displayFrame.width);
    }

    // Zoom indicator (bottom-left)
    if (zoom !== 1) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const label = `${zoom.toFixed(1)}×`;
      const margin = 10;
      ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.shadowColor = "rgba(0,0,0,0.5)";
      ctx.shadowBlur = 2;
      ctx.fillStyle = "white";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(label, margin, canvasH - margin);
      ctx.restore();
    }

    // Colorbar
    if (showColorbar) {
      const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      drawColorbar(ctx, canvasW, canvasH, lut, colorbarVminRef.current, colorbarVmaxRef.current, logScale);
      ctx.restore();
    }

    // ROI overlay
    if (roiActive && roiListTyped.length > 0) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      for (let ri = 0; ri < roiListTyped.length; ri++) {
        const roi = roiListTyped[ri];
        const isSelected = ri === roiSelectedIdx;
        const screenX = (roi.col * displayScale - cx) * zoom + cx + panX;
        const screenY = (roi.row * displayScale - cy) * zoom + cy + panY;
        const screenRadius = roi.radius * displayScale * zoom;
        const screenW = roi.width * displayScale * zoom;
        const screenH = roi.height * displayScale * zoom;
        const screenRadiusInner = roi.radius_inner * displayScale * zoom;
        const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
        ctx.lineWidth = roi.line_width || 2;
        drawROI(ctx, screenX, screenY, shape, screenRadius, screenW, screenH,
          roi.color || ROI_COLORS[ri % ROI_COLORS.length],
          roi.color || ROI_COLORS[ri % ROI_COLORS.length],
          isSelected && isDraggingROI, screenRadiusInner);
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

    // Profile line overlay
    if (profileActive && profilePoints.length > 0) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const toSX = (ic: number) => (ic * displayScale - cx) * zoom + cx + panX;
      const toSY = (ir: number) => (ir * displayScale - cy) * zoom + cy + panY;
      const ax = toSX(profilePoints[0].col);
      const ay = toSY(profilePoints[0].row);
      ctx.fillStyle = "#4fc3f7";
      ctx.beginPath(); ctx.arc(ax, ay, 4, 0, Math.PI * 2); ctx.fill();
      if (profilePoints.length === 2) {
        const bx = toSX(profilePoints[1].col);
        const by = toSY(profilePoints[1].row);
        ctx.strokeStyle = "#4fc3f7";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#4fc3f7";
        ctx.beginPath(); ctx.arc(bx, by, 4, 0, Math.PI * 2); ctx.fill();
      }
      ctx.restore();
    }
  }, [canvasW, canvasH, zoom, panX, panY, pixelSize, displayFrame, showColorbar, cmap, logScale, roiActive, roiListTyped, roiSelectedIdx, isDraggingROI, displayScale, profileActive, profilePoints, offscreenVersion]);

  // ─── Profile sparkline rendering ────────────────────────────────
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas || !profileActive || !profileData || profileData.length < 2) {
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.offsetWidth || canvasW;
    const cssH = profileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const isDark = themeInfo.theme === "dark";
    const bgColor = isDark ? "#1a1a1a" : "#f0f0f0";
    const lineColor = "#4fc3f7";
    const textColor = isDark ? "#888" : "#666";

    const padLeft = 32, padRight = 8, padTop = 6, padBottom = 18;
    const plotW = cssW - padLeft - padRight;
    const plotH = cssH - padTop - padBottom;

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, cssW, cssH);

    // Compute range
    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < profileData.length; i++) {
      if (profileData[i] < gMin) gMin = profileData[i];
      if (profileData[i] > gMax) gMax = profileData[i];
    }
    if (gMin === gMax) { gMax = gMin + 1; }

    // Calibration
    let totalDist = profileData.length - 1;
    let xUnit = "px";
    if (pixelSize > 0 && profilePoints.length === 2) {
      const dc = profilePoints[1].col - profilePoints[0].col;
      const dr = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(dc * dc + dr * dr);
      const distA = distPx * pixelSize;
      totalDist = distA >= 10 ? distA / 10 : distA;
      xUnit = distA >= 10 ? "nm" : "Å";
    }

    // Axis lines
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Profile line
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < profileData.length; i++) {
      const x = padLeft + (i / (profileData.length - 1)) * plotW;
      const y = padTop + plotH - ((profileData[i] - gMin) / (gMax - gMin)) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // X-axis labels
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = textColor;
    ctx.textBaseline = "top";
    const tickY = padTop + plotH + 2;
    ctx.textAlign = "left";
    ctx.fillText(`0`, padLeft, tickY);
    ctx.textAlign = "right";
    ctx.fillText(`${totalDist.toFixed(1)} ${xUnit}`, padLeft + plotW, tickY);

    // Y-axis labels
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = textColor;
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    const gMaxStr = gMax >= 1000 ? gMax.toExponential(1) : gMax.toFixed(1);
    const gMinStr = gMin >= 1000 ? gMin.toExponential(1) : gMin.toFixed(1);
    ctx.fillText(gMaxStr, padLeft - 3, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(gMinStr, padLeft - 3, padTop + plotH);

    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [profileActive, profileData, profileHeight, canvasW, pixelSize, profilePoints, themeInfo.theme]);

  // ─── FFT computation for current frame ──────────────────────────
  React.useEffect(() => {
    if (!showFft || !displayFrame) {
      fftMagRef.current = null;
      return;
    }
    let cancelled = false;
    const compute = async () => {
      setFftComputing(true);
      const { data, width, height } = displayFrame;
      const fftW = nextPow2(width);
      const fftH = nextPow2(height);
      let inputData = data;
      if (fftW !== width || fftH !== height) {
        const padded = new Float32Array(fftW * fftH);
        for (let y = 0; y < height; y++)
          for (let x = 0; x < width; x++) padded[y * fftW + x] = data[y * width + x];
        inputData = padded;
      }
      const real = inputData.slice();
      const imag = new Float32Array(real.length);
      if (fftWindow) applyHannWindow2D(real, fftW, fftH);
      if (gpuFFTRef.current && gpuFftReady) {
        const result = await gpuFFTRef.current.fft2D(real, imag, fftW, fftH, false);
        if (cancelled) return;
        fftshift(result.real, fftW, fftH);
        fftshift(result.imag, fftW, fftH);
        fftMagRef.current = computeMagnitude(result.real, result.imag);
      } else {
        fft2d(real, imag, fftW, fftH, false);
        if (cancelled) return;
        fftshift(real, fftW, fftH);
        fftshift(imag, fftW, fftH);
        fftMagRef.current = computeMagnitude(real, imag);
      }
      if (cancelled) return;
      // Auto-enhance: DC mask + 99.9th percentile clip (mutates mag in place)
      const mag = fftMagRef.current;
      // Apply log scale for better visualization
      for (let j = 0; j < mag.length; j++) mag[j] = Math.log1p(mag[j]);
      const { min: fftMin, max: fftMax } = autoEnhanceFFT(mag, fftW, fftH);
      setFftDataRange({ min: fftMin, max: fftMax });
      // Render FFT to offscreen canvas using the mutated mag + clipped range
      if (!fftOffscreenRef.current || fftOffscreenRef.current.width !== fftW || fftOffscreenRef.current.height !== fftH) {
        fftOffscreenRef.current = document.createElement("canvas");
        fftOffscreenRef.current.width = fftW;
        fftOffscreenRef.current.height = fftH;
        fftImgDataRef.current = fftOffscreenRef.current.getContext("2d")!.createImageData(fftW, fftH);
      }
      const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
      renderToOffscreenReuse(mag, lut, fftMin, fftMax, fftOffscreenRef.current, fftImgDataRef.current!);
      // Draw to FFT panel canvas (small square, top-right)
      const fftCanvas = fftCanvasRef.current;
      if (fftCanvas) {
        const panelSize = Math.round(canvasH * 0.4);
        const fctx = fftCanvas.getContext("2d");
        if (fctx) {
          fctx.imageSmoothingEnabled = false;
          fctx.clearRect(0, 0, fftCanvas.width, fftCanvas.height);
          fctx.save();
          fctx.scale(DPR, DPR);
          // Fit FFT centered with 3× default zoom (same as Show2D)
          const scale = Math.min(panelSize / fftW, panelSize / fftH) * 3;
          const ox = (panelSize - fftW * scale) / 2;
          const oy = (panelSize - fftH * scale) / 2;
          fctx.translate(ox, oy);
          fctx.scale(scale, scale);
          fctx.drawImage(fftOffscreenRef.current, 0, 0);
          fctx.restore();
        }
      }
      setFftComputing(false);
    };
    compute();
    return () => { cancelled = true; };
  }, [showFft, displayFrame, fftWindow, gpuFftReady, cmap, canvasW, canvasH, fftVminPct, fftVmaxPct]);

  // ─── Prevent page scroll on canvas ──────────────────────────────
  React.useEffect(() => {
    const el = canvasContainerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, []);

  // ─── Zoom (scroll wheel, cursor-centered) ───────────────────────
  // Zoom: same center-based math as Show2D
  const handleWheel = React.useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      // Mouse in canvas pixel coords
      const mouseX = (e.clientX - rect.left) * (canvasW / rect.width);
      const mouseY = (e.clientY - rect.top) * (canvasH / rect.height);
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      // Mouse in image space (inverse of center-based transform)
      const imgX = (mouseX - cx - panXRef.current) / zoomRef.current + cx;
      const imgY = (mouseY - cy - panYRef.current) / zoomRef.current + cy;
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      const nz = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoomRef.current * factor));
      // New pan to keep mouse on same image point
      const newPanX = mouseX - (imgX - cx) * nz - cx;
      const newPanY = mouseY - (imgY - cy) * nz - cy;
      setZoom(nz);
      setPanX(newPanX);
      setPanY(newPanY);
    },
    [canvasW, canvasH],
  );

  // ─── Screen → image coordinate helper ───────────────────────────
  const screenToImg = React.useCallback((e: React.MouseEvent): { imgCol: number; imgRow: number } => {
    const canvas = canvasRef.current;
    if (!canvas) return { imgCol: 0, imgRow: 0 };
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvasW / rect.width);
    const my = (e.clientY - rect.top) * (canvasH / rect.height);
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    return {
      imgCol: ((mx - cx - panXRef.current) / zoomRef.current + cx) / displayScale,
      imgRow: ((my - cy - panYRef.current) / zoomRef.current + cy) / displayScale,
    };
  }, [canvasW, canvasH, displayScale]);

  // ─── ROI hit testing helpers ─────────────────────────────────────
  const isNearEdge = React.useCallback((imgCol: number, imgRow: number, roi: ROIItem): boolean => {
    const hitArea = RESIZE_HIT_AREA_PX / (displayScale * zoomRef.current);
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
  }, [displayScale]);

  const hitTestROI = React.useCallback((imgCol: number, imgRow: number): number => {
    if (!roiActive || !roiListTyped) return -1;
    for (let ri = roiListTyped.length - 1; ri >= 0; ri--) {
      const roi = roiListTyped[ri];
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
  }, [roiActive, roiListTyped]);

  // ─── Pan (click + drag) + ROI + Profile mouse down ───────────────
  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      dragRef.current = { wasDrag: false };
      clickStartRef.current = { x: e.clientX, y: e.clientY };

      const { imgCol, imgRow } = screenToImg(e);

      // ── Profile mode ──
      if (profileActive) {
        if (profilePoints.length === 2) {
          const p0 = profilePoints[0];
          const p1 = profilePoints[1];
          const hitRadius = 10 / (displayScale * zoomRef.current);
          const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
          const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
          if (d0 <= hitRadius || d1 <= hitRadius) {
            setDraggingProfileEndpoint(d0 <= d1 ? 0 : 1);
            return;
          }
          if (pointToSegmentDistance(imgCol, imgRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
            setIsDraggingProfileLine(true);
            profileDragStartRef.current = { row: imgRow, col: imgCol, p0: { row: p0.row, col: p0.col }, p1: { row: p1.row, col: p1.col } };
            return;
          }
        }
        // Fall through to pan
        panStartRef.current = { x: e.clientX, y: e.clientY, pX: panXRef.current, pY: panYRef.current };
        return;
      }

      // ── ROI mode ──
      if (roiActive) {
        // Check resize of inner annular ring
        if (selectedRoi && selectedRoi.shape === "annular") {
          const hitArea = RESIZE_HIT_AREA_PX / (displayScale * zoomRef.current);
          const dist = Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
          if (Math.abs(dist - selectedRoi.radius_inner) < hitArea) {
            setIsDraggingResizeInner(true);
            return;
          }
        }
        // Check resize of selected ROI
        if (selectedRoi && isNearEdge(imgCol, imgRow, selectedRoi)) {
          e.preventDefault();
          resizeAspectRef.current = selectedRoi.shape === "rectangle" && selectedRoi.width > 0 && selectedRoi.height > 0
            ? selectedRoi.width / selectedRoi.height : null;
          setIsDraggingResize(true);
          return;
        }
        // Check edge of any ROI — auto-select + resize
        for (let ri = 0; ri < roiListTyped.length; ri++) {
          if (isNearEdge(imgCol, imgRow, roiListTyped[ri])) {
            e.preventDefault();
            const roi = roiListTyped[ri];
            resizeAspectRef.current = roi.shape === "rectangle" && roi.width > 0 && roi.height > 0 ? roi.width / roi.height : null;
            setRoiSelectedIdx(ri);
            setIsDraggingResize(true);
            return;
          }
        }
        // Hit-test existing ROIs
        const hitIdx = hitTestROI(imgCol, imgRow);
        if (hitIdx >= 0) {
          setRoiSelectedIdx(hitIdx);
          setIsDraggingROI(true);
          return;
        }
        // Click on empty — deselect
        setRoiSelectedIdx(-1);
        // Fall through to pan
      }

      // ── Pan ──
      panStartRef.current = { x: e.clientX, y: e.clientY, pX: panXRef.current, pY: panYRef.current };
      const onMove = (ev: MouseEvent) => {
        if (!panStartRef.current) return;
        const dx = ev.clientX - panStartRef.current.x;
        const dy = ev.clientY - panStartRef.current.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) dragRef.current.wasDrag = true;
        setPanX(panStartRef.current.pX + dx);
        setPanY(panStartRef.current.pY + dy);
      };
      const onUp = () => {
        panStartRef.current = null;
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [profileActive, profilePoints, roiActive, roiListTyped, selectedRoi, displayScale, screenToImg, isNearEdge, hitTestROI],
  );

  // ─── Mouse move: cursor readout + ROI drag + profile drag ────────
  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent) => {
      if (!displayFrame) {
        setCursorInfo(null);
        return;
      }
      const { imgCol, imgRow } = screenToImg(e);

      // Cursor readout
      if (imgCol >= 0 && imgCol < displayFrame.width && imgRow >= 0 && imgRow < displayFrame.height) {
        const value = displayFrame.data[Math.floor(imgRow) * displayFrame.width + Math.floor(imgCol)];
        setCursorInfo({ row: Math.floor(imgRow), col: Math.floor(imgCol), value });
      } else {
        setCursorInfo(null);
      }

      // ── Profile endpoint drag ──
      if (draggingProfileEndpoint !== null && profilePoints.length === 2) {
        const pt = { row: Math.max(0, Math.min(displayFrame.height - 1, imgRow)), col: Math.max(0, Math.min(displayFrame.width - 1, imgCol)) };
        const next = draggingProfileEndpoint === 0 ? [pt, profilePoints[1]] : [profilePoints[0], pt];
        setProfilePoints(next);
        const d = sampleLineProfile(displayFrame.data, displayFrame.width, displayFrame.height, next[0].row, next[0].col, next[1].row, next[1].col);
        setProfileData(d);
        return;
      }

      // ── Profile line drag ──
      if (isDraggingProfileLine && profileDragStartRef.current && profilePoints.length === 2) {
        const drag = profileDragStartRef.current;
        let dRow = imgRow - drag.row;
        let dCol = imgCol - drag.col;
        const minRow = Math.min(drag.p0.row, drag.p1.row);
        const maxRow = Math.max(drag.p0.row, drag.p1.row);
        const minCol = Math.min(drag.p0.col, drag.p1.col);
        const maxCol = Math.max(drag.p0.col, drag.p1.col);
        dRow = Math.max(dRow, -minRow); dRow = Math.min(dRow, (displayFrame.height - 1) - maxRow);
        dCol = Math.max(dCol, -minCol); dCol = Math.min(dCol, (displayFrame.width - 1) - maxCol);
        const next = [{ row: drag.p0.row + dRow, col: drag.p0.col + dCol }, { row: drag.p1.row + dRow, col: drag.p1.col + dCol }];
        setProfilePoints(next);
        const d = sampleLineProfile(displayFrame.data, displayFrame.width, displayFrame.height, next[0].row, next[0].col, next[1].row, next[1].col);
        setProfileData(d);
        return;
      }

      // ── ROI inner annular resize ──
      if (isDraggingResizeInner && selectedRoi) {
        const newR = Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
        updateSelectedRoi({ radius_inner: Math.max(1, Math.min(selectedRoi.radius - 1, Math.round(newR))) });
        return;
      }

      // ── ROI outer resize ──
      if (isDraggingResize && selectedRoi) {
        const shape = selectedRoi.shape || "circle";
        if (shape === "rectangle") {
          let newW = Math.max(2, Math.round(Math.abs(imgCol - selectedRoi.col) * 2));
          let newH = Math.max(2, Math.round(Math.abs(imgRow - selectedRoi.row) * 2));
          if (e.shiftKey && resizeAspectRef.current != null) {
            const aspect = resizeAspectRef.current;
            if (newW / newH > aspect) newH = Math.max(2, Math.round(newW / aspect));
            else newW = Math.max(2, Math.round(newH * aspect));
          }
          updateSelectedRoi({ width: newW, height: newH });
        } else {
          const newR = shape === "square"
            ? Math.max(Math.abs(imgCol - selectedRoi.col), Math.abs(imgRow - selectedRoi.row))
            : Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
          const minR = shape === "annular" ? selectedRoi.radius_inner + 1 : 1;
          updateSelectedRoi({ radius: Math.max(minR, Math.round(newR)) });
        }
        return;
      }

      // ── ROI move drag ──
      if (isDraggingROI && selectedRoi) {
        updateSelectedRoi({
          col: Math.max(0, Math.min(displayFrame.width - 1, Math.floor(imgCol))),
          row: Math.max(0, Math.min(displayFrame.height - 1, Math.floor(imgRow))),
        });
        return;
      }

      // ── Hover: detect ROI resize edge ──
      if (roiActive && !isDraggingROI && !isDraggingResize) {
        if (selectedRoi && selectedRoi.shape === "annular") {
          const hitArea = RESIZE_HIT_AREA_PX / (displayScale * zoomRef.current);
          const dist = Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
          setIsHoveringResizeInner(Math.abs(dist - selectedRoi.radius_inner) < hitArea);
        } else {
          setIsHoveringResizeInner(false);
        }
        setIsHoveringResize(roiListTyped.some(roi => isNearEdge(imgCol, imgRow, roi)));
      }
    },
    [displayFrame, screenToImg, draggingProfileEndpoint, isDraggingProfileLine, isDraggingResizeInner, isDraggingResize, isDraggingROI, selectedRoi, roiActive, roiListTyped, profilePoints, displayScale, isNearEdge],
  );

  // ─── Mouse up: click actions (place profile points, new ROI) ─────
  const handleMouseUp = React.useCallback(
    (e: React.MouseEvent) => {
      const wasDragging = draggingProfileEndpoint !== null || isDraggingProfileLine || isDraggingROI || isDraggingResize || isDraggingResizeInner;
      setDraggingProfileEndpoint(null);
      setIsDraggingProfileLine(false);
      profileDragStartRef.current = null;
      setIsDraggingROI(false);
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      panStartRef.current = null;

      if (wasDragging || !clickStartRef.current || !displayFrame) {
        clickStartRef.current = null;
        return;
      }
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      clickStartRef.current = null;
      if (Math.sqrt(dx * dx + dy * dy) >= 3) return; // was a drag

      const { imgCol, imgRow } = screenToImg(e);
      if (imgCol < 0 || imgCol >= displayFrame.width || imgRow < 0 || imgRow >= displayFrame.height) return;

      // ── Profile: place point ──
      if (profileActive) {
        const pt = { row: imgRow, col: imgCol };
        if (profilePoints.length === 0 || profilePoints.length === 2) {
          setProfilePoints([pt]);
          setProfileData(null);
        } else {
          const p0 = profilePoints[0];
          setProfilePoints([p0, pt]);
          const d = sampleLineProfile(displayFrame.data, displayFrame.width, displayFrame.height, p0.row, p0.col, pt.row, pt.col);
          setProfileData(d);
        }
        return;
      }

      // ── ROI: place new ROI on empty space click ──
      if (roiActive && roiSelectedIdx < 0) {
        const defaultR = Math.round(Math.min(displayFrame.width, displayFrame.height) * 0.1);
        const newRoi: ROIItem = {
          row: imgRow, col: imgCol,
          shape: newRoiShape,
          radius: defaultR, radius_inner: Math.round(defaultR * 0.5),
          width: defaultR * 2, height: defaultR * 2,
          color: ROI_COLORS[roiListTyped.length % ROI_COLORS.length],
          line_width: 2, highlight: false,
        };
        const newList = [...roiListTyped, newRoi] as { [key: string]: unknown }[];
        setRoiList(newList);
        setRoiSelectedIdx(roiListTyped.length);
      }
    },
    [profileActive, profilePoints, roiActive, roiSelectedIdx, roiListTyped, newRoiShape, displayFrame, screenToImg, draggingProfileEndpoint, isDraggingProfileLine, isDraggingROI, isDraggingResize, isDraggingResizeInner],
  );

  // ─── Reset zoom ──────────────────────────────────────────────────
  const resetZoom = React.useCallback(() => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, []);

  // ─── Copy to clipboard ───────────────────────────────────────────
  const handleCopy = React.useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/png"),
      );
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch {
      canvas.toBlob(
        (b) => {
          if (b)
            downloadBlob(
              b,
              `live_${displayFrame?.label ?? "frame"}.png`,
            );
        },
        "image/png",
      );
    }
  }, [displayFrame]);

  // ─── Export figure ───────────────────────────────────────────────
  const handleExportFigure = React.useCallback((withColorbarOpt: boolean, asPNG: boolean) => {
    setExportAnchor(null);
    if (!displayFrame) return;
    const { data, width, height } = displayFrame;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const rawRange = findDataRange(data);
    const logRange = {
      min: Math.log1p(Math.max(rawRange.min, 0)),
      max: Math.log1p(Math.max(rawRange.max, 0)),
    };
    const range = logScale ? logRange : rawRange;
    let vmin: number, vmax: number;
    if (autoContrast) {
      const processed = logScale ? applyLogScale(data) : data;
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else if (vminPctRef.current > 0 || vmaxPctRef.current < 100) {
      ({ vmin, vmax } = sliderRange(range.min, range.max, vminPctRef.current, vmaxPctRef.current));
    } else {
      vmin = range.min;
      vmax = range.max;
    }
    const processed = logScale ? applyLogScale(data) : data;
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
      showColorbar: withColorbarOpt,
      showScaleBar: pixelSize > 0,
      drawAnnotations: (ctx) => {
        if (roiActive && roiListTyped.length > 0) {
          for (const roi of roiListTyped) {
            const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
            ctx.lineWidth = roi.line_width || 2;
            drawROI(ctx, roi.col, roi.row, shape, roi.radius, roi.width, roi.height, roi.color, roi.color, false, roi.radius_inner);
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
      },
    });

    const filename = `live_${displayFrame.label || "frame"}`;
    if (asPNG) {
      figCanvas.toBlob((b) => { if (b) downloadBlob(b, `${filename}.png`); }, "image/png");
    } else {
      canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, `${filename}.pdf`));
    }
  }, [displayFrame, cmap, logScale, autoContrast, pixelSize, title, roiActive, roiListTyped, profileActive, profilePoints]);

  // ─── Canvas resize handle ────────────────────────────────────────
  const handleResizeMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsResizingCanvas(true);
      resizeStartRef.current = { x: e.clientX, y: e.clientY, size: canvasSize };
    },
    [canvasSize],
  );

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    let rafId = 0;
    let latestSize = resizeStartRef.current ? resizeStartRef.current.size : canvasSize;

    const onMove = (e: MouseEvent) => {
      if (!resizeStartRef.current) return;
      const delta = Math.max(
        e.clientX - resizeStartRef.current.x,
        e.clientY - resizeStartRef.current.y,
      );
      latestSize = Math.max(MIN_CANVAS, resizeStartRef.current.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setCanvasSize(latestSize);
        });
      }
    };
    const onUp = () => {
      cancelAnimationFrame(rafId);
      setCanvasSize(latestSize);
      setIsResizingCanvas(false);
      resizeStartRef.current = null;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, [isResizingCanvas]);

  // ─── Keyboard shortcuts ──────────────────────────────────────────
  const handleKeyDown = React.useCallback(
    (e: React.KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT"
      )
        return;
      switch (e.key) {
        case "ArrowLeft":
          e.preventDefault();
          e.stopPropagation();
          setSelectedIdx((prev) =>
            Math.min(prev + 1, framesRef.current.length - 1),
          );
          break;
        case "ArrowRight":
          e.preventDefault();
          e.stopPropagation();
          setSelectedIdx((prev) => Math.max(prev - 1, 0));
          break;
        case "r":
        case "R":
          e.preventDefault();
          resetZoom();
          break;
        default:
          break;
      }
    },
    [resetZoom],
  );

  // ─── Derived display values ──────────────────────────────────────
  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;
  const displayLabel = displayFrame?.label ?? "—";
  const displayDims = displayFrame
    ? `${displayFrame.height}×${displayFrame.width}`
    : "—";
  const calibratedUnit =
    pixelSize > 0
      ? Math.max(displayH, displayW) * pixelSize >= 10
        ? "nm"
        : "Å"
      : "";
  const calibratedFactor =
    calibratedUnit === "nm" ? pixelSize / 10 : pixelSize;

  // Thumb panel width: based on 2 thumbnails + gap + padding
  const thumbPanelWidth = THUMB_COLS * THUMB_SIZE + (THUMB_COLS - 1) * SPACING.XS + 2 * SPACING.SM + 2;

  // ─── Render ──────────────────────────────────────────────────────
  return (
    <Box
      ref={rootRef}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={() => rootRef.current?.focus()}
      sx={{
        p: 2,
        bgcolor: themeColors.bg,
        color: themeColors.text,
        width: "fit-content",
        outline: "none",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      }}
    >
      {/* ── Title row ─────────────────────────────────────────────── */}
      <Stack
        direction="row"
        alignItems="center"
        spacing={`${SPACING.SM}px`}
        sx={{ mb: `${SPACING.XS}px`, height: 16 }}
      >
        <Typography
          variant="caption"
          sx={{
            ...typography.label,
            color: themeColors.accent,
            lineHeight: "16px",
            overflow: "hidden",
          }}
        >
          {title || "Live"}
        </Typography>
        <InfoTooltip
          text={
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>
                Live Controls
              </Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                Frames are pushed in real-time from Python. Newest frame is
                always shown first (index 0).
              </Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>
                Keyboard
              </Typography>
              <Box
                component="table"
                sx={{
                  borderCollapse: "collapse",
                  "& td": {
                    py: 0.25,
                    fontSize: 11,
                    lineHeight: 1.3,
                    verticalAlign: "top",
                  },
                  "& td:first-of-type": {
                    pr: 1.5,
                    opacity: 0.7,
                    fontFamily: "monospace",
                    fontSize: 10,
                    whiteSpace: "nowrap",
                  },
                }}
              >
                <tbody>
                  {(
                    [
                      ["← / →", "Older / newer frame"],
                      ["R", "Reset zoom"],
                      ["Scroll", "Zoom in/out"],
                      ["Dbl-click", "Reset view"],
                    ] as [string, string][]
                  ).map(([k, d], i) => (
                    <tr key={i}>
                      <td>{k}</td>
                      <td>{d}</td>
                    </tr>
                  ))}
                </tbody>
              </Box>
            </Box>
          }
          theme={themeInfo.theme}
        />
        <Box sx={{ flex: 1 }} />
        {/* Frame count */}
        <Typography
          sx={{
            ...typography.labelSmall,
            color: themeColors.textMuted,
            fontFamily: "monospace",
          }}
        >
          {frameCount} frame{frameCount !== 1 ? "s" : ""}
          {nFrames > frameCount ? ` / ${nFrames}` : ""}
        </Typography>
        {/* LIVE indicator */}
        {isWatching && (
          <Typography
            sx={{
              fontSize: 10,
              fontFamily: "monospace",
              color: themeColors.accentGreen,
              fontWeight: "bold",
              "@keyframes livePulse": {
                "0%,100%": { opacity: 1 },
                "50%": { opacity: 0.4 },
              },
              animation: "livePulse 1.2s ease-in-out infinite",
            }}
          >
            ● LIVE
          </Typography>
        )}
      </Stack>

      {/* ── Toggle row ──────────────────────────────────────────── */}
      {showControls && (
        <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center" sx={{ mb: `${SPACING.XS}px` }}>
          <Typography sx={{ ...typography.labelSmall, fontSize: 10 }}>FFT:</Typography>
          <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
          <Typography sx={{ ...typography.labelSmall, fontSize: 10 }}>ROI:</Typography>
          <Switch checked={roiActive} onChange={(e) => setRoiActive(e.target.checked)} size="small" sx={switchStyles.small} />
          <Typography sx={{ ...typography.labelSmall, fontSize: 10 }}>Profile:</Typography>
          <Switch checked={profileActive} onChange={(e) => setProfileActive(e.target.checked)} size="small" sx={switchStyles.small} />
          <Typography sx={{ ...typography.labelSmall, fontSize: 10 }}>Colorbar:</Typography>
          <Switch checked={showColorbar} onChange={(e) => setShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} />
          <Box sx={{ flex: 1 }} />
          <Typography sx={{ ...typography.labelSmall, fontSize: 10, color: themeColors.accent, cursor: "pointer", "&:hover": { opacity: 0.7 } }}
            onClick={resetZoom}>RESET</Typography>
          <Typography sx={{ ...typography.labelSmall, fontSize: 10, color: themeColors.accent, cursor: "pointer", "&:hover": { opacity: 0.7 } }}
            onClick={handleCopy}>COPY</Typography>
        </Stack>
      )}

      {/* ── Body: filmstrip + inspect panel ───────────────────────── */}
      <Stack direction="row" spacing={`${SPACING.LG}px`} alignItems="flex-start">

        {/* ══ LEFT: Filmstrip ══════════════════════════════════════ */}
        <Box sx={{ width: thumbPanelWidth, flexShrink: 0 }}>
          <Box
            ref={thumbListRef}
            sx={{
              border: `1px solid ${themeColors.border}`,
              bgcolor: themeColors.controlBg,
              p: `${SPACING.SM}px`,
              maxHeight: canvasH + 150,
              overflowY: "auto",
              overflowX: "hidden",
            }}
          >
            <Typography
              sx={{
                ...typography.labelSmall,
                color: themeColors.textMuted,
                mb: `${SPACING.XS}px`,
                display: "block",
              }}
            >
              Frames
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: `repeat(${THUMB_COLS}, ${THUMB_SIZE}px)`,
                gap: `${SPACING.XS}px`,
              }}
            >
              {framesRef.current.slice(0, frameCount).map((frame, i) => (
                <Box
                  key={`thumb-${frameCount}-${i}`}
                  onClick={() => setSelectedIdx(i)}
                  sx={{
                    border: `2px solid ${i === selectedIdx ? themeColors.accent : "transparent"}`,
                    cursor: "pointer",
                    opacity: i === selectedIdx ? 1 : 0.65,
                    "&:hover": { opacity: 1 },
                    position: "relative",
                  }}
                >
                  {frame.thumbCanvas ? (
                    <canvas
                      ref={(el) => {
                        if (el && frame.thumbCanvas) {
                          el.width = THUMB_SIZE;
                          el.height = THUMB_SIZE;
                          const ctx = el.getContext("2d");
                          if (ctx) {
                            ctx.imageSmoothingEnabled = true;
                            ctx.drawImage(
                              frame.thumbCanvas,
                              0,
                              0,
                              el.width,
                              el.height,
                            );
                          }
                        }
                      }}
                      style={{
                        width: THUMB_SIZE,
                        height: THUMB_SIZE,
                        display: "block",
                        imageRendering: "auto",
                      }}
                    />
                  ) : (
                    <Box
                      sx={{
                        width: THUMB_SIZE,
                        height: THUMB_SIZE,
                        bgcolor: themeColors.border,
                      }}
                    />
                  )}
                  <Typography
                    sx={{
                      position: "absolute",
                      bottom: 0,
                      left: 0,
                      right: 0,
                      fontSize: 8,
                      textAlign: "center",
                      bgcolor: "rgba(0,0,0,0.55)",
                      color: "#fff",
                      lineHeight: 1.3,
                      px: 0.25,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {frame.label}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Box>
        </Box>

        {/* ══ RIGHT: Inspect Panel ═════════════════════════════════ */}
        <Box
          sx={{
            width: showFft ? canvasW + Math.round(canvasH * 0.4) + SPACING.SM : canvasW,
            flexShrink: 0,
            display: "flex",
            flexDirection: "column",
            gap: 0,
          }}
        >
          {/* Controls row: Reset + Copy + Export */}
          <Stack
            direction="row"
            alignItems="center"
            spacing={`${SPACING.SM}px`}
            sx={{ mb: `${SPACING.XS}px`, height: 28 }}
          >
            <Box sx={{ flex: 1 }} />
            <Button
              size="small"
              sx={compactButton}
              disabled={!needsReset}
              onClick={resetZoom}
            >
              Reset
            </Button>
            <Button size="small" sx={compactButton} onClick={handleCopy}>
              Copy
            </Button>
            <Button
              size="small"
              sx={compactButton}
              onClick={(e) => setExportAnchor(e.currentTarget)}
              disabled={!displayFrame}
            >
              Export
            </Button>
            <Menu
              anchorEl={exportAnchor}
              open={Boolean(exportAnchor)}
              onClose={() => setExportAnchor(null)}
              {...upwardMenuProps}
              PaperProps={{ sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } }}
            >
              <MenuItem sx={{ fontSize: 11 }} onClick={() => handleExportFigure(true, false)}>PDF + colorbar</MenuItem>
              <MenuItem sx={{ fontSize: 11 }} onClick={() => handleExportFigure(false, false)}>PDF</MenuItem>
              <MenuItem sx={{ fontSize: 11 }} onClick={() => handleExportFigure(false, true)}>PNG</MenuItem>
            </Menu>
          </Stack>

          {/* Canvas + FFT side by side */}
          <Stack direction="row" spacing={`${SPACING.SM}px`}>
          <Box
            ref={canvasContainerRef}
            sx={{
              position: "relative",
              bgcolor: "#000",
              border: `1px solid ${themeColors.border}`,
              width: canvasW,
              height: canvasH,
              minWidth: canvasW,
              flexShrink: 0,
              cursor: isHoveringResize || isHoveringResizeInner ? "nwse-resize" : profileActive ? "crosshair" : roiActive ? "crosshair" : "grab",
              overflow: "hidden",
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => { setCursorInfo(null); setIsHoveringResize(false); setIsHoveringResizeInner(false); }}
            onWheel={handleWheel}
            onDoubleClick={resetZoom}
          >
            {frameCount === 0 ? (
              <Box
                sx={{
                  width: "100%",
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Typography
                  sx={{ ...typography.label, color: themeColors.textMuted }}
                >
                  Waiting for data…
                </Typography>
              </Box>
            ) : (
              <canvas
                ref={canvasRef}
                width={canvasW}
                height={canvasH}
                style={{
                  width: canvasW,
                  height: canvasH,
                  display: "block",
                  imageRendering: "pixelated",
                }}
              />
            )}

            {/* HiDPI overlay: scale bar */}
            <canvas
              ref={overlayRef}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: canvasW,
                height: canvasH,
                pointerEvents: "none",
              }}
            />

            {/* Cursor readout (top-right) */}
            {cursorInfo && (
              <Box
                sx={{
                  position: "absolute",
                  top: 3,
                  right: 3,
                  bgcolor: "rgba(0,0,0,0.35)",
                  px: 0.5,
                  py: 0.15,
                  pointerEvents: "none",
                  minWidth: 100,
                  textAlign: "right",
                }}
              >
                <Typography
                  sx={{
                    fontSize: 9,
                    fontFamily: "monospace",
                    color: "rgba(255,255,255,0.7)",
                    whiteSpace: "nowrap",
                    lineHeight: 1.2,
                  }}
                >
                  ({cursorInfo.row}, {cursorInfo.col})
                  {pixelSize > 0
                    ? ` = (${(cursorInfo.row * calibratedFactor).toFixed(1)}, ${(cursorInfo.col * calibratedFactor).toFixed(1)} ${calibratedUnit})`
                    : ""}{" "}
                  {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}

            {/* (FFT panel is outside this box, to the right) */}

            {/* Resize handle (bottom-right triangle) */}
            <Box
              onMouseDown={handleResizeMouseDown}
              sx={{
                position: "absolute",
                bottom: 0,
                right: 0,
                width: 16,
                height: 16,
                cursor: "nwse-resize",
                opacity: 0.6,
                pointerEvents: "auto",
                background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
                "&:hover": { opacity: 1 },
              }}
            />
          </Box>

          {/* FFT panel — small square, top-right, ~1/4 of canvas height */}
          {showFft && (() => {
            const fftSize = Math.round(canvasH * 0.4);
            return (
            <Box sx={{
              position: "relative",
              border: `1px solid ${themeColors.border}`,
              bgcolor: "#000",
              overflow: "hidden",
              width: fftSize,
              height: fftSize,
              minWidth: fftSize,
              flexShrink: 0,
              alignSelf: "flex-start",
            }}>
              <canvas
                ref={fftCanvasRef}
                width={fftSize * DPR}
                height={fftSize * DPR}
                style={{ width: fftSize, height: fftSize, imageRendering: "pixelated" }}
              />
              {fftComputing && (
                <Box sx={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)", bgcolor: "rgba(0,0,0,0.7)", px: 1, py: 0.25 }}>
                  <Typography sx={{ fontSize: 9, color: "#fff" }}>FFT...</Typography>
                </Box>
              )}
              <Typography sx={{ position: "absolute", top: 2, left: 4, fontSize: 9, fontWeight: 600, color: "rgba(255,255,255,0.7)", pointerEvents: "none", textShadow: "0 0 3px rgba(0,0,0,0.8)" }}>FFT</Typography>
            </Box>
            );
          })()}
          </Stack>

          {/* Stats bar — same format as Show2D */}
          {showStats && frameCount > 0 && (
            <Box
              sx={{
                mt: `${SPACING.XS}px`,
                px: 1,
                py: 0.5,
                bgcolor: themeColors.bgAlt,
                display: "flex",
                gap: 2,
                alignItems: "center",
                boxSizing: "border-box",
                overflow: "hidden",
                whiteSpace: "nowrap",
              }}
            >
              <Typography
                sx={{ fontSize: 11, color: themeColors.textMuted }}
              >
                Mean{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsMean)}
                </Box>
              </Typography>
              <Typography
                sx={{ fontSize: 11, color: themeColors.textMuted }}
              >
                Min{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsMin)}
                </Box>
              </Typography>
              <Typography
                sx={{ fontSize: 11, color: themeColors.textMuted }}
              >
                Max{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsMax)}
                </Box>
              </Typography>
              <Typography
                sx={{ fontSize: 11, color: themeColors.textMuted }}
              >
                Std{" "}
                <Box component="span" sx={{ color: themeColors.accent }}>
                  {formatNumber(statsStd)}
                </Box>
              </Typography>
            </Box>
          )}

          {/* Controls + Histogram */}
          {showControls && (
            <Box
              sx={{
                mt: `${SPACING.SM}px`,
                display: "flex",
                flexDirection: "column",
                gap: `${SPACING.XS}px`,
                boxSizing: "border-box",
              }}
            >
              <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
                {/* Left: two control rows */}
                <Box
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    gap: `${SPACING.XS}px`,
                    flex: 1,
                    justifyContent: "center",
                  }}
                >
                  {/* Row 1: Scale + Color + Colorbar */}
                  <Box
                    sx={{
                      ...controlRow,
                      border: `1px solid ${themeColors.border}`,
                      bgcolor: themeColors.controlBg,
                    }}
                  >
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>
                      Scale:
                    </Typography>
                    <Select
                      value={logScale ? "log" : "linear"}
                      onChange={(e) => setLogScale(e.target.value === "log")}
                      size="small"
                      sx={{ ...themedSelect, minWidth: 45 }}
                      MenuProps={themedMenuProps}
                    >
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>
                      Color:
                    </Typography>
                    <Select
                      size="small"
                      value={cmap}
                      onChange={(e) => setCmap(e.target.value)}
                      MenuProps={themedMenuProps}
                      sx={{ ...themedSelect, minWidth: 60 }}
                    >
                      {COLORMAP_NAMES.map((name) => (
                        <MenuItem key={name} value={name}>
                          {name.charAt(0).toUpperCase() + name.slice(1)}
                        </MenuItem>
                      ))}
                    </Select>
                  </Box>

                  {/* Row 2: Auto + zoom indicator */}
                  <Box
                    sx={{
                      ...controlRow,
                      border: `1px solid ${themeColors.border}`,
                      bgcolor: themeColors.controlBg,
                    }}
                  >
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>
                      Auto:
                    </Typography>
                    <Switch
                      checked={autoContrast}
                      onChange={() => {
                        const next = !autoContrast;
                        autoContrastRef.current = next;
                        setAutoContrast(next);
                      }}
                      size="small"
                      sx={switchStyles.small}
                    />
                    {/* Label: current frame info */}
                    {displayFrame && (
                      <Typography
                        sx={{
                          ...typography.value,
                          color: themeColors.textMuted,
                          ml: 0.5,
                          maxWidth: 140,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {displayLabel}
                      </Typography>
                    )}
                    {displayFrame && (
                      <Typography
                        sx={{
                          ...typography.value,
                          color: themeColors.textMuted,
                          ml: 0.5,
                        }}
                      >
                        {displayDims}
                      </Typography>
                    )}
                  </Box>
                </Box>

                {/* Right: Histogram */}
                {histData && (
                  <Box
                    sx={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "flex-end",
                      justifyContent: "center",
                    }}
                  >
                    <Histogram
                      data={histData}
                      vminPct={vminPct}
                      vmaxPct={vmaxPct}
                      onRangeChange={handleRangeChange}
                      width={110}
                      height={58}
                      theme={themeInfo.theme === "dark" ? "dark" : "light"}
                      dataMin={imageDataRange.min}
                      dataMax={imageDataRange.max}
                    />
                  </Box>
                )}
              </Box>
            </Box>
          )}

          {/* ROI controls */}
          {showControls && roiActive && (
            <Box
              sx={{
                mt: `${SPACING.XS}px`,
                ...controlRow,
                border: `1px solid ${themeColors.border}`,
                bgcolor: themeColors.controlBg,
              }}
            >
              <Typography sx={{ ...typography.label, fontSize: 10 }}>Shape:</Typography>
              <Select
                size="small"
                value={newRoiShape}
                onChange={(e) => setNewRoiShape(e.target.value as "square" | "rectangle" | "circle" | "annular")}
                sx={{ ...themedSelect, minWidth: 80 }}
                MenuProps={themedMenuProps}
              >
                <MenuItem value="square">Square</MenuItem>
                <MenuItem value="rectangle">Rectangle</MenuItem>
                <MenuItem value="circle">Circle</MenuItem>
                <MenuItem value="annular">Annular</MenuItem>
              </Select>
              <Button
                size="small"
                variant="outlined"
                sx={{ ...compactButton, borderColor: themeColors.border, color: themeColors.text }}
                onClick={() => {
                  if (!displayFrame) return;
                  const defaultR = Math.round(Math.min(displayFrame.width, displayFrame.height) * 0.1);
                  const cx = displayFrame.width / 2;
                  const cy = displayFrame.height / 2;
                  const newRoi: ROIItem = {
                    row: cy, col: cx,
                    shape: newRoiShape,
                    radius: defaultR, radius_inner: Math.round(defaultR * 0.5),
                    width: defaultR * 2, height: defaultR * 2,
                    color: ROI_COLORS[roiListTyped.length % ROI_COLORS.length],
                    line_width: 2, highlight: false,
                  };
                  const newList = [...roiListTyped, newRoi] as { [key: string]: unknown }[];
                  setRoiList(newList);
                  setRoiSelectedIdx(roiListTyped.length);
                }}
              >
                Add
              </Button>
              <Button
                size="small"
                variant="outlined"
                sx={{ ...compactButton, borderColor: themeColors.border, color: themeColors.text }}
                onClick={() => { setRoiList([]); setRoiSelectedIdx(-1); }}
                disabled={roiListTyped.length === 0}
              >
                Clear
              </Button>
              {roiListTyped.length > 0 && (
                <Typography sx={{ ...typography.value, color: themeColors.textMuted }}>
                  {roiListTyped.length} ROI{roiListTyped.length !== 1 ? "s" : ""}
                  {roiSelectedIdx >= 0 && ` · #${roiSelectedIdx + 1} selected`}
                </Typography>
              )}
            </Box>
          )}

          {/* Profile sparkline */}
          {showControls && profileActive && profileData && profileData.length >= 2 && (
            <Box sx={{ mt: `${SPACING.XS}px` }}>
              <canvas
                ref={profileCanvasRef}
                style={{
                  width: canvasW,
                  height: profileHeight,
                  display: "block",
                  border: `1px solid ${themeColors.border}`,
                  boxSizing: "border-box",
                }}
              />
            </Box>
          )}
          {showControls && profileActive && profilePoints.length === 1 && (
            <Typography sx={{ ...typography.value, color: themeColors.textMuted, mt: `${SPACING.XS}px`, fontSize: 10 }}>
              Click a second point to complete the profile line.
            </Typography>
          )}
          {showControls && profileActive && profilePoints.length === 0 && (
            <Typography sx={{ ...typography.value, color: themeColors.textMuted, mt: `${SPACING.XS}px`, fontSize: 10 }}>
              Click two points on the image to draw a profile line.
            </Typography>
          )}
        </Box>
      </Stack>
    </Box>
  );
});

export default { render };
