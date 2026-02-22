/// <reference types="@webgpu/types" />
/**
 * Show3DVolume - Orthogonal slice viewer for 3D volumetric data.
 *
 * Three side-by-side canvases showing XY, XZ, YZ planes with sliders.
 * All slicing done in JS from raw float32 volume data for instant response.
 *
 * Self-contained widget with all utilities inlined (matching Show3D pattern).
 */
import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Menu from "@mui/material/Menu";
import Tooltip from "@mui/material/Tooltip";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import FastForwardIcon from "@mui/icons-material/FastForward";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import StopIcon from "@mui/icons-material/Stop";
import "./show3dvolume.css";
import { useTheme } from "../theme";
import { VolumeRenderer, CameraState, DEFAULT_CAMERA } from "../webgl-volume";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar, exportFigure, canvasToPDF } from "../scalebar";
import { extractFloat32, formatNumber, downloadBlob, downloadDataView } from "../format";
import { computeHistogramFromBytes } from "../histogram";
import { findDataRange, applyLogScale, percentileClip, sliderRange } from "../stats";
import { ControlCustomizer } from "../control-customizer";
import { computeToolVisibility } from "../tool-parity";

// ============================================================================
// UI Styles (matching Show3D exactly)
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
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
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};

import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen } from "../colormaps";

import { WebGPUFFT, getWebGPUFFT, fft2d, fftshift, nextPow2, computeMagnitude, autoEnhanceFFT } from "../webgpu-fft";

// ============================================================================
// Zoom constants (matching Show3D)
// ============================================================================
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

// ============================================================================
// Slice extraction from flat float32 buffer
// ============================================================================
function extractXY(vol: Float32Array, nx: number, ny: number, _nz: number, z: number): Float32Array {
  const start = z * ny * nx;
  return vol.subarray(start, start + ny * nx);
}

function extractXZ(vol: Float32Array, nx: number, ny: number, nz: number, y: number): Float32Array {
  const out = new Float32Array(nz * nx);
  for (let z = 0; z < nz; z++) {
    const srcOffset = z * ny * nx + y * nx;
    for (let x = 0; x < nx; x++) out[z * nx + x] = vol[srcOffset + x];
  }
  return out;
}

function extractYZ(vol: Float32Array, nx: number, ny: number, nz: number, x: number): Float32Array {
  const out = new Float32Array(nz * ny);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) out[z * ny + y] = vol[z * ny * nx + y * nx + x];
  }
  return out;
}

// ============================================================================
// Constants
// ============================================================================
type ZoomState = { zoom: number; panX: number; panY: number };
const DEFAULT_ZOOM: ZoomState = { zoom: 1, panX: 0, panY: 0 };
const CANVAS_TARGET = 400;
const AXES = ["xy", "xz", "yz"] as const;
const DPR = window.devicePixelRatio || 1;

// ============================================================================
// InfoTooltip
// ============================================================================
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
  const colors = theme === "dark" ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" } : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };

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

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas ref={canvasRef} style={{ width, height, border: `1px solid ${colors.border}` }} />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => { const [newMin, newMax] = v as number[]; onRangeChange(Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1)); }}
        min={0} max={100} size="small" valueLabelDisplay="auto"
        valueLabelFormat={(pct) => { const val = dataMin + (pct / 100) * (dataMax - dataMin); return val >= 1000 ? val.toExponential(1) : val.toFixed(1); }}
        sx={{ width, py: 0, "& .MuiSlider-thumb": { width: 8, height: 8 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vminPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vmaxPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography></Box>
    </Box>
  );
}

// ============================================================================
// Main Component
// ============================================================================
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

function Show3DVolume() {
  // Theme detection
  const { themeInfo, colors: baseColors } = useTheme();
  const tc = {
    ...baseColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#b08800",
  };

  const themedSelect = {
    ...controlPanel.select,
    bgcolor: tc.controlBg,
    color: tc.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: tc.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: tc.accent },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: tc.controlBg, color: tc.text, border: `1px solid ${tc.border}` } },
  };

  // Model state
  const [nx] = useModelState<number>("nx");
  const [ny] = useModelState<number>("ny");
  const [nz] = useModelState<number>("nz");
  const [volumeBytes] = useModelState<DataView>("volume_bytes");
  const [sliceX, setSliceX] = useModelState<number>("slice_x");
  const [sliceY, setSliceY] = useModelState<number>("slice_y");
  const [sliceZ, setSliceZ] = useModelState<number>("slice_z");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [showControls] = useModelState<boolean>("show_controls");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showCrosshair, setShowCrosshair] = useModelState<boolean>("show_crosshair");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [disabledTools, setDisabledTools] = useModelState<string[]>("disabled_tools");
  const [hiddenTools, setHiddenTools] = useModelState<string[]>("hidden_tools");
  const [dimLabels] = useModelState<string[]>("dim_labels");
  const [statsMean] = useModelState<number[]>("stats_mean");
  const [statsMin] = useModelState<number[]>("stats_min");
  const [statsMax] = useModelState<number[]>("stats_max");
  const [statsStd] = useModelState<number[]>("stats_std");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");

  const toolVisibility = React.useMemo(
    () => computeToolVisibility("Show3DVolume", disabledTools, hiddenTools),
    [disabledTools, hiddenTools],
  );
  const hideDisplay = toolVisibility.isHidden("display");
  const hideHistogram = toolVisibility.isHidden("histogram");
  const hideStats = toolVisibility.isHidden("stats");
  const hidePlayback = toolVisibility.isHidden("playback") || toolVisibility.isHidden("navigation");
  const hideView = toolVisibility.isHidden("view");
  const hideExport = toolVisibility.isHidden("export");
  const hideVolume = toolVisibility.isHidden("volume");

  const lockDisplay = toolVisibility.isLocked("display");
  const lockHistogram = toolVisibility.isLocked("histogram");
  const lockStats = toolVisibility.isLocked("stats");
  const lockPlayback = toolVisibility.isLocked("playback") || toolVisibility.isLocked("navigation");
  const lockView = toolVisibility.isLocked("view");
  const lockExport = toolVisibility.isLocked("export");
  const lockVolume = toolVisibility.isLocked("volume");
  const effectiveShowFft = showFft && !hideDisplay;

  // Initialize WebGPU FFT
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) { gpuFFTRef.current = fft; setGpuReady(true); }
    });
  }, []);

  // Canvas refs
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const uiRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);

  // FFT state
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftLogScale, setFftLogScale] = React.useState(false);
  const [fftAuto, setFftAuto] = React.useState(true);
  const [fftZooms, setFftZooms] = React.useState<ZoomState[]>([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
  const [fftDragAxis, setFftDragAxis] = React.useState<number | null>(null);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);

  // FFT d-spacing measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    axis: number; row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number; axis: number } | null>(null);
  const fftCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftOverlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftOffscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftMagCacheRefs = React.useRef<(Float32Array | null)[]>([null, null, null]);
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);

  // Zoom/pan per axis
  const [zooms, setZooms] = React.useState<ZoomState[]>([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
  const [dragAxis, setDragAxis] = React.useState<number | null>(null);
  const [dragStart, setDragStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);

  // Canvas resize (matching Show2D pattern)
  const [canvasTarget, setCanvasTarget] = React.useState(CANVAS_TARGET);
  const [isResizing, setIsResizing] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number } | null>(null);

  // Playback state (synced with Python)
  const [playing, setPlaying] = useModelState<boolean>("playing");
  const [playAxis, setPlayAxis] = useModelState<number>("play_axis");
  const [reverse, setReverse] = useModelState<boolean>("reverse");
  const [fps, setFps] = useModelState<number>("fps");
  const [loop, setLoop] = useModelState<boolean>("loop");
  const playIntervalRef = React.useRef<number | null>(null);
  const [boomerang, setBoomerang] = useModelState<boolean>("boomerang");
  const bounceDirRef = React.useRef<1 | -1>(1);
  const [loopStarts, setLoopStarts] = React.useState([0, 0, 0]);
  const [loopEnds, setLoopEnds] = React.useState([-1, -1, -1]);

  // 3D volume renderer state
  const volumeCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const volumeRendererRef = React.useRef<VolumeRenderer | null>(null);
  const [camera, setCamera] = React.useState<CameraState>(DEFAULT_CAMERA);
  const [volumeDrag, setVolumeDrag] = React.useState<{
    button: number; x: number; y: number; yaw: number; pitch: number; panX: number; panY: number;
  } | null>(null);
  const [webglSupported, setWebglSupported] = React.useState(true);
  const [volumeOpacity, setVolumeOpacity] = React.useState(0.5);
  const [volumeBrightness, setVolumeBrightness] = React.useState(1.0);
  const [volumeCanvasSize, setVolumeCanvasSize] = React.useState(300);
  const [volumeResizing, setVolumeResizing] = React.useState(false);
  const volumeResizeStartRef = React.useRef<{ x: number; y: number; size: number } | null>(null);
  const [showSlicePlanes, setShowSlicePlanes] = React.useState(true);

  // Histogram state
  const [imageVminPct, setImageVminPct] = React.useState(0);
  const [imageVmaxPct, setImageVmaxPct] = React.useState(100);
  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });

  // Cached offscreen canvases for slice rendering (avoids recomputing colormap on zoom/pan)
  const sliceOffscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);

  // Colorbar state
  const [showColorbar, setShowColorbar] = React.useState(false);

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number; view: string } | null>(null);

  // Export state
  const [, setExportAxis] = useModelState<number>("_export_axis");
  const [, setGifExportRequested] = useModelState<boolean>("_gif_export_requested");
  const [gifData] = useModelState<DataView>("_gif_data");
  const [gifMetadataJson] = useModelState<string>("_gif_metadata_json");
  const [, setZipExportRequested] = useModelState<boolean>("_zip_export_requested");
  const [zipData] = useModelState<DataView>("_zip_data");
  const [exporting, setExporting] = React.useState(false);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);

  // Parse volume data
  const allFloats = React.useMemo(() => extractFloat32(volumeBytes), [volumeBytes]);

  // Slice dimensions: [xy: ny x nx], [xz: nz x nx], [yz: nz x ny]
  const sliceDims: [number, number][] = React.useMemo(() => [[ny, nx], [nz, nx], [nz, ny]], [nx, ny, nz]);

  // Canvas sizes
  const canvasSizes = React.useMemo(() => {
    return sliceDims.map(([h, w]) => {
      const scale = canvasTarget / Math.max(w, h);
      return { w: Math.round(w * scale), h: Math.round(h * scale), scale };
    });
  }, [sliceDims, canvasTarget]);

  React.useEffect(() => {
    if (hideDisplay && showFft) {
      setShowFft(false);
    }
  }, [hideDisplay, showFft, setShowFft]);

  React.useEffect(() => {
    if (lockPlayback && playing) {
      setPlaying(false);
    }
  }, [lockPlayback, playing, setPlaying]);

  // Prevent page scroll on canvases
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    canvasRefs.current.forEach(c => c?.addEventListener("wheel", preventDefault, { passive: false }));
    fftCanvasRefs.current.forEach(c => c?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => {
      canvasRefs.current.forEach(c => c?.removeEventListener("wheel", preventDefault));
      fftCanvasRefs.current.forEach(c => c?.removeEventListener("wheel", preventDefault));
    };
  }, [allFloats, effectiveShowFft]);

  // Compute histogram from XY slice (primary view)
  React.useEffect(() => {
    if (!allFloats || allFloats.length === 0) return;
    const xySlice = extractXY(allFloats, nx, ny, nz, sliceZ);
    const processed = logScale ? applyLogScale(xySlice) : xySlice;
    setImageHistogramData(processed);
    setImageDataRange(findDataRange(processed));
  }, [allFloats, sliceZ, nx, ny, nz, logScale]);

  // Download GIF when data arrives from Python
  React.useEffect(() => {
    if (!gifData || gifData.byteLength === 0) return;
    downloadDataView(gifData, "show3dvolume_animation.gif", "image/gif");
    const metaText = (gifMetadataJson || "").trim();
    if (metaText) {
      downloadBlob(new Blob([metaText], { type: "application/json" }), "show3dvolume_animation.json");
    }
    setExporting(false);
  }, [gifData, gifMetadataJson]);

  // Download ZIP when data arrives from Python
  React.useEffect(() => {
    if (!zipData || zipData.byteLength === 0) return;
    downloadDataView(zipData, "show3dvolume_slices.zip", "application/zip");
    setExporting(false);
  }, [zipData]);

  // Sync boomerang direction ref with reverse state
  React.useEffect(() => {
    bounceDirRef.current = reverse ? -1 : 1;
  }, [reverse]);

  // -------------------------------------------------------------------------
  // 3D Volume Renderer — init, upload, render
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = volumeCanvasRef.current;
    if (!canvas) return;
    if (!VolumeRenderer.isSupported()) { setWebglSupported(false); return; }
    try {
      const renderer = new VolumeRenderer(canvas);
      volumeRendererRef.current = renderer;
    } catch { setWebglSupported(false); }
    return () => { volumeRendererRef.current?.dispose(); volumeRendererRef.current = null; };
  }, []);

  // Upload volume data
  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer || !allFloats || allFloats.length === 0) return;
    renderer.uploadVolume(allFloats, nx, ny, nz);
  }, [allFloats, nx, ny, nz]);

  // Upload colormap
  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer) return;
    renderer.uploadColormap(COLORMAPS[cmap] || COLORMAPS.inferno);
  }, [cmap]);

  // Render 3D volume
  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer || !allFloats || allFloats.length === 0) return;
    // Parse background color from theme
    const bgHex = tc.bg;
    const r = parseInt(bgHex.slice(1, 3), 16) / 255;
    const g = parseInt(bgHex.slice(3, 5), 16) / 255;
    const b = parseInt(bgHex.slice(5, 7), 16) / 255;
    renderer.render({
      sliceX, sliceY, sliceZ, nx, ny, nz,
      opacity: volumeOpacity, brightness: volumeBrightness,
      showSlicePlanes,
    }, camera, [r, g, b]);
  }, [allFloats, sliceX, sliceY, sliceZ, nx, ny, nz, cmap, camera, volumeOpacity, volumeBrightness, volumeCanvasSize, tc.bg, showSlicePlanes]);

  // Prevent scroll on volume canvas
  React.useEffect(() => {
    const canvas = volumeCanvasRef.current;
    if (!canvas || !webglSupported) return;
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    canvas.addEventListener("wheel", preventDefault, { passive: false });
    return () => canvas.removeEventListener("wheel", preventDefault);
  }, [webglSupported]);

  // -------------------------------------------------------------------------
  // 3D Volume mouse handlers
  // -------------------------------------------------------------------------
  const handleVolumeMouseDown = (e: React.MouseEvent) => {
    setVolumeDrag({
      button: e.button, x: e.clientX, y: e.clientY,
      yaw: camera.yaw, pitch: camera.pitch, panX: camera.panX, panY: camera.panY,
    });
    e.preventDefault();
  };

  const handleVolumeMouseMove = (e: React.MouseEvent) => {
    if (!volumeDrag) return;
    const dx = e.clientX - volumeDrag.x;
    const dy = e.clientY - volumeDrag.y;
    if (volumeDrag.button === 0 && !e.shiftKey) {
      // Left drag = rotate
      setCamera(prev => ({
        ...prev,
        yaw: volumeDrag.yaw + dx * 0.005,
        pitch: Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, volumeDrag.pitch - dy * 0.005)),
      }));
    } else {
      // Right drag or shift+drag = pan
      const sens = 0.003 * camera.distance;
      setCamera(prev => ({
        ...prev,
        panX: volumeDrag.panX + dx * sens,
        panY: volumeDrag.panY - dy * sens,
      }));
    }
  };

  const handleVolumeMouseUp = () => setVolumeDrag(null);

  const handleVolumeWheel = (e: React.WheelEvent) => {
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    setCamera(prev => ({ ...prev, distance: Math.max(0.5, Math.min(10, prev.distance * factor)) }));
  };

  const handleVolumeDoubleClick = () => setCamera(DEFAULT_CAMERA);

  // -------------------------------------------------------------------------
  // 3D Volume canvas resize
  // -------------------------------------------------------------------------
  const handleVolumeResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation(); e.preventDefault();
    setVolumeResizing(true);
    volumeResizeStartRef.current = { x: e.clientX, y: e.clientY, size: volumeCanvasSize };
  };

  React.useEffect(() => {
    if (!volumeResizing) return;
    const onMove = (e: MouseEvent) => {
      const start = volumeResizeStartRef.current;
      if (!start) return;
      const delta = Math.max(e.clientX - start.x, e.clientY - start.y);
      setVolumeCanvasSize(Math.max(300, Math.min(800, start.size + delta)));
    };
    const onUp = () => { setVolumeResizing(false); volumeResizeStartRef.current = null; };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => { document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
  }, [volumeResizing]);

  const cameraChanged = camera.yaw !== DEFAULT_CAMERA.yaw || camera.pitch !== DEFAULT_CAMERA.pitch || camera.distance !== DEFAULT_CAMERA.distance || camera.panX !== DEFAULT_CAMERA.panX || camera.panY !== DEFAULT_CAMERA.panY;

  // Any zoom active?
  const needsReset = zooms.some(z => z.zoom !== 1 || z.panX !== 0 || z.panY !== 0) || fftZooms.some(z => z.zoom !== 1 || z.panX !== 0 || z.panY !== 0) || cameraChanged;

  // -------------------------------------------------------------------------
  // Build colormapped offscreen canvases (expensive: log scale, percentile, colormap LUT)
  // Excludes zoom/pan so dragging only triggers the cheap redraw below.
  // useLayoutEffect so offscreens are ready before the draw useLayoutEffect runs.
  // -------------------------------------------------------------------------
  React.useLayoutEffect(() => {
    if (!allFloats || allFloats.length === 0) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const sliceData = [
      extractXY(allFloats, nx, ny, nz, sliceZ),
      extractXZ(allFloats, nx, ny, nz, sliceY),
      extractYZ(allFloats, nx, ny, nz, sliceX),
    ];
    for (let a = 0; a < 3; a++) {
      const [sliceH, sliceW] = sliceDims[a];
      const processed = logScale ? applyLogScale(sliceData[a]) : sliceData[a];
      let vmin: number, vmax: number;
      if (autoContrast) {
        ({ vmin, vmax } = percentileClip(processed, 2, 98));
      } else if (imageVminPct > 0 || imageVmaxPct < 100) {
        const { min: dMin, max: dMax } = findDataRange(processed);
        ({ vmin, vmax } = sliderRange(dMin, dMax, imageVminPct, imageVmaxPct));
      } else {
        const r = findDataRange(processed);
        vmin = r.min;
        vmax = r.max;
      }
      sliceOffscreenRefs.current[a] = renderToOffscreen(processed, sliceW, sliceH, lut, vmin, vmax);
    }
  }, [allFloats, sliceX, sliceY, sliceZ, nx, ny, nz, cmap, logScale, autoContrast, sliceDims, imageVminPct, imageVmaxPct]);

  // -------------------------------------------------------------------------
  // Redraw slices with zoom/pan (cheap: just drawImage from cached offscreen)
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  // -------------------------------------------------------------------------
  React.useLayoutEffect(() => {
    for (let a = 0; a < 3; a++) {
      const canvas = canvasRefs.current[a];
      const offscreen = sliceOffscreenRefs.current[a];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      const [sliceH, sliceW] = sliceDims[a];
      const { w: cw, h: ch } = canvasSizes[a];
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, cw, ch);
      const zs = zooms[a];
      if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
        ctx.save();
        const cx = cw / 2, cy = ch / 2;
        ctx.translate(cx + zs.panX, cy + zs.panY);
        ctx.scale(zs.zoom, zs.zoom);
        ctx.translate(-cx, -cy);
        ctx.drawImage(offscreen, 0, 0, sliceW, sliceH, 0, 0, cw, ch);
        ctx.restore();
      } else {
        ctx.drawImage(offscreen, 0, 0, sliceW, sliceH, 0, 0, cw, ch);
      }
    }
  }, [allFloats, sliceX, sliceY, sliceZ, nx, ny, nz, cmap, logScale, autoContrast, zooms, sliceDims, canvasSizes, imageVminPct, imageVmaxPct]);

  // -------------------------------------------------------------------------
  // Render overlays (crosshair lines)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!allFloats) return;
    const crossPositions: [number, number][] = [
      [sliceX, sliceY],
      [sliceX, sliceZ],
      [sliceY, sliceZ],
    ];
    for (let a = 0; a < 3; a++) {
      const overlay = overlayRefs.current[a];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      const { w: cw, h: ch, scale } = canvasSizes[a];
      ctx.clearRect(0, 0, cw, ch);
      if (showCrosshair) {
        const zs = zooms[a];
        const [dataX, dataY] = crossPositions[a];
        const cx = cw / 2, cy = ch / 2;
        let canvasX = dataX * scale;
        let canvasY = dataY * scale;
        if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
          canvasX = (canvasX - cx) * zs.zoom + cx + zs.panX;
          canvasY = (canvasY - cy) * zs.zoom + cy + zs.panY;
        }
        ctx.strokeStyle = tc.accentYellow + "80";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(canvasX, 0); ctx.lineTo(canvasX, ch); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, canvasY); ctx.lineTo(cw, canvasY); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }, [allFloats, sliceX, sliceY, sliceZ, zooms, showCrosshair, tc, sliceDims, canvasSizes]);

  // -------------------------------------------------------------------------
  // Scale bar (HiDPI UI overlay)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    for (let a = 0; a < 3; a++) {
      const uiCanvas = uiRefs.current[a];
      if (!uiCanvas) continue;
      const { w: cw, h: ch } = canvasSizes[a];
      uiCanvas.width = Math.round(cw * DPR);
      uiCanvas.height = Math.round(ch * DPR);
      const uiCtx = uiCanvas.getContext("2d");
      if (!uiCtx) continue;
      uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
      if (scaleBarVisible) {
        const pxSize = pixelSize || 0;
        const sliceW = sliceDims[a][1];
        const unit = pxSize > 0 ? "Å" : "px";
        const size = pxSize > 0 ? pxSize : 1;
        drawScaleBarHiDPI(uiCanvas, DPR, zooms[a].zoom, size, unit, sliceW);
      }

      if (showColorbar) {
        const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
        const { vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct);
        const cssW = uiCanvas.width / DPR;
        const cssH = uiCanvas.height / DPR;
        uiCtx.save();
        uiCtx.scale(DPR, DPR);
        drawColorbar(uiCtx, cssW, cssH, lut, vmin, vmax, logScale);
        uiCtx.restore();
      }
    }
  }, [pixelSize, scaleBarVisible, zooms, canvasSizes, sliceDims, showColorbar, cmap, imageDataRange, imageVminPct, imageVmaxPct, logScale, themeInfo.theme]);

  // -------------------------------------------------------------------------
  // FFT computation and caching
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!effectiveShowFft || !allFloats || allFloats.length === 0) return;

    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;

    const computeAllFFTs = async () => {
      const sliceData = [
        extractXY(allFloats, nx, ny, nz, sliceZ),
        extractXZ(allFloats, nx, ny, nz, sliceY),
        extractYZ(allFloats, nx, ny, nz, sliceX),
      ];
      const dims: [number, number][] = [[ny, nx], [nz, nx], [nz, ny]];

      for (let a = 0; a < 3; a++) {
        const data = sliceData[a];
        const [sliceH, sliceW] = dims[a];

        // Full-slice FFT
        const pw = nextPow2(sliceW);
        ph = nextPow2(sliceH);
        const paddedSize = pw * ph;
        let real: Float32Array, imag: Float32Array;

        if (gpuReady && gpuFFTRef.current) {
          // Pad input manually so GPU FFT doesn't crop
          const padReal = new Float32Array(paddedSize);
          const padImag = new Float32Array(paddedSize);
          for (let y = 0; y < sliceH; y++) for (let x = 0; x < sliceW; x++) padReal[y * pw + x] = data[y * sliceW + x];
          const result = await gpuFFTRef.current.fft2D(padReal, padImag, pw, ph, false);
          real = result.real; imag = result.imag;
        } else {
          real = new Float32Array(paddedSize);
          imag = new Float32Array(paddedSize);
          for (let y = 0; y < sliceH; y++) for (let x = 0; x < sliceW; x++) real[y * pw + x] = data[y * sliceW + x];
          fft2d(real, imag, pw, ph, false);
        }

        fftshift(real, pw, ph);
        fftshift(imag, pw, ph);

        const mag = computeMagnitude(real, imag);
        fftMagCacheRefs.current[a] = mag;

        let displayMin: number, displayMax: number;
        if (fftAuto) {
          ({ min: displayMin, max: displayMax } = autoEnhanceFFT(mag, pw, ph));
        } else {
          ({ min: displayMin, max: displayMax } = findDataRange(mag));
        }

        const displayData = fftLogScale ? applyLogScale(mag) : mag;
        if (fftLogScale) { displayMin = Math.log1p(displayMin); displayMax = Math.log1p(displayMax); }

        const offscreen = renderToOffscreen(displayData, pw, ph, lut, displayMin, displayMax);
        if (!offscreen) continue;
        fftOffscreenRefs.current[a] = offscreen;

        const canvas = fftCanvasRefs.current[a];
        if (canvas) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            const { w: cw, h: ch } = canvasSizes[a];
            ctx.imageSmoothingEnabled = false;
            ctx.clearRect(0, 0, cw, ch);
            const zs = fftZooms[a];
            if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
              ctx.save();
              const cx = cw / 2, cy = ch / 2;
              ctx.translate(cx + zs.panX, cy + zs.panY); ctx.scale(zs.zoom, zs.zoom); ctx.translate(-cx, -cy);
              ctx.drawImage(offscreen, 0, 0, pw, ph, 0, 0, cw, ch);
              ctx.restore();
            } else {
              ctx.drawImage(offscreen, 0, 0, pw, ph, 0, 0, cw, ch);
            }
          }
        }
      }
    };

    computeAllFFTs();
  }, [effectiveShowFft, allFloats, sliceX, sliceY, sliceZ, nx, ny, nz, fftColormap, fftLogScale, fftAuto, gpuReady, canvasSizes, fftZooms]);

  // Redraw cached FFT with zoom/pan (cheap -- no recomputation)
  React.useLayoutEffect(() => {
    if (!effectiveShowFft) return;
    for (let a = 0; a < 3; a++) {
      const canvas = fftCanvasRefs.current[a];
      const offscreen = fftOffscreenRefs.current[a];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      const { w: cw, h: ch } = canvasSizes[a];
      const ow = offscreen.width, oh = offscreen.height;
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, cw, ch);
      const zs = fftZooms[a];
      if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
        ctx.save();
        const cx = cw / 2, cy = ch / 2;
        ctx.translate(cx + zs.panX, cy + zs.panY); ctx.scale(zs.zoom, zs.zoom); ctx.translate(-cx, -cy);
        ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
        ctx.restore();
      } else {
        ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
      }
    }
  }, [effectiveShowFft, fftZooms, canvasSizes]);

  // Render FFT overlays (reciprocal-space scale bars + d-spacing crosshair per axis)
  React.useEffect(() => {
    if (!effectiveShowFft) return;
    const dims: [number, number][] = [[ny, nx], [nz, nx], [nz, ny]];
    for (let a = 0; a < 3; a++) {
      const overlay = fftOverlayRefs.current[a];
      if (!overlay) continue;
      const { w: cw, h: ch } = canvasSizes[a];
      overlay.width = Math.round(cw * DPR);
      overlay.height = Math.round(ch * DPR);
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      // FFT scale bar (only when calibrated)
      if (pixelSize > 0) {
        const [, sliceW] = dims[a];
        const pw = nextPow2(sliceW);
        const fftPixelSize = 1 / (pw * pixelSize);
        drawFFTScaleBarHiDPI(overlay, DPR, fftZooms[a].zoom, fftPixelSize, pw);
      }

      // D-spacing crosshair on clicked FFT panel
      if (fftClickInfo && fftClickInfo.axis === a) {
        const [sliceH, sliceW] = dims[a];
        const fftW = nextPow2(sliceW);
        const fftH = nextPow2(sliceH);

        ctx.save();
        ctx.scale(DPR, DPR);
        const zs = fftZooms[a];
        const cx = cw / 2, cy = ch / 2;
        const rawX = fftClickInfo.col / fftW * cw;
        const rawY = fftClickInfo.row / fftH * ch;
        const screenX = (rawX - cx) * zs.zoom + cx + zs.panX;
        const screenY = (rawY - cy) * zs.zoom + cy + zs.panY;

        // Draw crosshair with gap in center
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

        // D-spacing label
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
    }
  }, [effectiveShowFft, fftZooms, canvasSizes, pixelSize, nx, ny, nz, fftClickInfo]);

  // -------------------------------------------------------------------------
  // Playback logic (matching Show3D pattern)
  // -------------------------------------------------------------------------
  const sliceSettersRef = React.useRef<((v: number) => void)[]>([setSliceZ, setSliceY, setSliceX]);
  sliceSettersRef.current = [setSliceZ, setSliceY, setSliceX];
  const effectiveLoopEnds = React.useMemo(
    () => loopEnds.map((end, i) => {
      const max = [nz - 1, ny - 1, nx - 1][i];
      return end < 0 ? max : Math.min(end, max);
    }),
    [loopEnds, nx, ny, nz],
  );
  React.useEffect(() => {
    if (!playing) return;
    const intervalMs = 1000 / fps;

    if (playAxis === 3) {
      // "All" mode: advance all 3 axes simultaneously
      playIntervalRef.current = window.setInterval(() => {
        const dir = boomerang ? bounceDirRef.current : (reverse ? -1 : 1);
        // Check if any axis would go out of range
        let shouldBounce = false;
        for (let a = 0; a < 3; a++) {
          const next = sliceValuesRef.current[a] + dir;
          if (next > effectiveLoopEnds[a] || next < loopStarts[a]) { shouldBounce = true; break; }
        }
        if (boomerang && shouldBounce) {
          bounceDirRef.current = (-bounceDirRef.current) as 1 | -1;
        }
        const finalDir = boomerang ? bounceDirRef.current : dir;
        for (let a = 0; a < 3; a++) {
          const start = loopStarts[a];
          const end = effectiveLoopEnds[a];
          let next = sliceValuesRef.current[a] + finalDir;
          if (next > end) next = loop || boomerang ? start : end;
          else if (next < start) next = loop || boomerang ? end : start;
          sliceSettersRef.current[a](next);
          sliceValuesRef.current[a] = next;
        }
        if (!loop && !boomerang && shouldBounce) setPlaying(false);
      }, intervalMs);
    } else {
      // Single axis mode
      const axis = playAxis;
      const start = loopStarts[axis];
      const end = effectiveLoopEnds[axis];
      const setter = sliceSettersRef.current[axis];
      playIntervalRef.current = window.setInterval(() => {
        const prev = sliceValuesRef.current[axis];
        let next = prev;
        if (boomerang) {
          const candidate = prev + bounceDirRef.current;
          if (candidate > end) {
            bounceDirRef.current = -1;
            next = prev - 1 >= start ? prev - 1 : prev;
          } else if (candidate < start) {
            bounceDirRef.current = 1;
            next = prev + 1 <= end ? prev + 1 : prev;
          } else {
            next = candidate;
          }
        } else {
          next = prev + (reverse ? -1 : 1);
          if (reverse) {
            if (next < start) {
              if (!loop) setPlaying(false);
              next = loop ? end : start;
            }
          } else if (next > end) {
            if (!loop) setPlaying(false);
            next = loop ? start : end;
          }
        }
        setter(next);
        sliceValuesRef.current[axis] = next;
      }, intervalMs);
    }
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
    };
  }, [playing, fps, reverse, boomerang, loop, playAxis, loopStarts, effectiveLoopEnds]);

  // -------------------------------------------------------------------------
  // Zoom/Pan handlers (matching Show3D)
  // -------------------------------------------------------------------------
  const handleWheel = (e: React.WheelEvent, axis: number) => {
    const canvas = canvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const zs = zooms[axis];
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const imgX = (mouseX - cx - zs.panX) / zs.zoom + cx;
    const imgY = (mouseY - cy - zs.panY) / zs.zoom + cy;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * factor));
    const newPanX = mouseX - (imgX - cx) * newZoom - cx;
    const newPanY = mouseY - (imgY - cy) * newZoom - cy;
    setZooms(prev => { const next = [...prev]; next[axis] = { zoom: newZoom, panX: newPanX, panY: newPanY }; return next; });
  };

  const handleDoubleClick = (axis: number) => {
    setZooms(prev => { const next = [...prev]; next[axis] = DEFAULT_ZOOM; return next; });
  };

  const handleMouseDown = (e: React.MouseEvent, axis: number) => {
    const zs = zooms[axis];
    setDragAxis(axis);
    setDragStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
  };

  const handleMouseMove = (e: React.MouseEvent, axis: number) => {
    // Fast-path: skip cursor readout during pan drag for 60fps
    if (dragAxis === axis && dragStart) {
      const canvas = canvasRefs.current[axis];
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dx = (e.clientX - dragStart.x) * (canvas.width / rect.width);
      const dy = (e.clientY - dragStart.y) * (canvas.height / rect.height);
      setZooms(prev => { const next = [...prev]; next[axis] = { ...prev[axis], panX: dragStart.pX + dx, panY: dragStart.pY + dy }; return next; });
      return;
    }

    // Cursor readout (only when not dragging)
    const cursorCanvas = canvasRefs.current[axis];
    if (cursorCanvas && allFloats && allFloats.length > 0) {
      const rect = cursorCanvas.getBoundingClientRect();
      const canvasX = (e.clientX - rect.left) * (cursorCanvas.width / rect.width);
      const canvasY = (e.clientY - rect.top) * (cursorCanvas.height / rect.height);
      const { w: cw, h: ch, scale } = canvasSizes[axis];
      const zs = zooms[axis];
      const cx = cw / 2, cy = ch / 2;
      // Reverse zoom/pan transform to get image pixel coordinates
      let imgX: number, imgY: number;
      if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
        imgX = ((canvasX - cx - zs.panX) / zs.zoom + cx) / scale;
        imgY = ((canvasY - cy - zs.panY) / zs.zoom + cy) / scale;
      } else {
        imgX = canvasX / scale;
        imgY = canvasY / scale;
      }
      const px = Math.floor(imgX);
      const py = Math.floor(imgY);
      const [sliceH, sliceW] = sliceDims[axis];
      if (px >= 0 && px < sliceW && py >= 0 && py < sliceH) {
        // Look up voxel value from volume data at the appropriate 3D coordinate
        let value: number;
        if (axis === 0) {
          // XY view: x maps to x, y maps to y, slice along Z
          value = allFloats[sliceZ * ny * nx + py * nx + px];
        } else if (axis === 1) {
          // XZ view: x maps to x, y maps to z, slice along Y
          value = allFloats[py * ny * nx + sliceY * nx + px];
        } else {
          // YZ view: x maps to y, y maps to z, slice along X
          value = allFloats[py * ny * nx + px * nx + sliceX];
        }
        setCursorInfo({ row: py, col: px, value, view: ["XY", "XZ", "YZ"][axis] });
      } else {
        setCursorInfo(null);
      }
    }
  };

  const handleMouseUp = () => { setDragAxis(null); setDragStart(null); };

  const handleMouseLeave = () => { setDragAxis(null); setDragStart(null); setCursorInfo(null); };

  const handleResetAll = () => {
    if (!lockView) {
      setZooms([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
      setFftZooms([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
    }
    if (!lockVolume) {
      setCamera(DEFAULT_CAMERA);
      setVolumeOpacity(0.5);
      setVolumeBrightness(1.0);
    }
    if (!lockPlayback) {
      setLoopStarts([0, 0, 0]);
      setLoopEnds([-1, -1, -1]);
    }
    if (!lockDisplay) {
      setImageVminPct(0);
      setImageVmaxPct(100);
    }
  };

  // -------------------------------------------------------------------------
  // Keyboard shortcuts
  // -------------------------------------------------------------------------
  const handleKeyDown = (e: React.KeyboardEvent) => {
    const axisSetters = [setSliceZ, setSliceY, setSliceX];
    const axisValues = [sliceZ, sliceY, sliceX];
    const axisMaxes = [nz - 1, ny - 1, nx - 1];
    const activeAxis = playAxis < 3 ? playAxis : 0;
    switch (e.key) {
      case " ":
        if (!lockPlayback) {
          e.preventDefault();
          setPlaying(!playing);
        }
        break;
      case "ArrowLeft":
        if (!lockPlayback) {
          e.preventDefault();
          axisSetters[activeAxis](Math.max(0, axisValues[activeAxis] - 1));
        }
        break;
      case "ArrowRight":
        if (!lockPlayback) {
          e.preventDefault();
          axisSetters[activeAxis](Math.min(axisMaxes[activeAxis], axisValues[activeAxis] + 1));
        }
        break;
      case "Home":
        if (!lockPlayback) {
          e.preventDefault();
          axisSetters[activeAxis](0);
        }
        break;
      case "End":
        if (!lockPlayback) {
          e.preventDefault();
          axisSetters[activeAxis](axisMaxes[activeAxis]);
        }
        break;
      case "r":
      case "R":
        handleResetAll();
        break;
    }
  };

  // -------------------------------------------------------------------------
  // Export handlers
  // -------------------------------------------------------------------------
  const handleExportPng = () => {
    if (lockExport) return;
    setExportAnchor(null);
    // Export all 3 slice canvases as individual PNGs
    for (let a = 0; a < 3; a++) {
      const canvas = canvasRefs.current[a];
      if (!canvas) continue;
      canvas.toBlob((blob) => {
        if (!blob) return;
        downloadBlob(blob, `show3dvolume_${AXES[a]}.png`);
      }, "image/png");
    }
  };

  const handleExportGif = () => {
    if (lockExport) return;
    setExportAnchor(null);
    setExporting(true);
    setExportAxis(playAxis < 3 ? playAxis : 0);
    setGifExportRequested(true);
  };

  const handleExportZip = () => {
    if (lockExport) return;
    setExportAnchor(null);
    setExporting(true);
    setExportAxis(playAxis < 3 ? playAxis : 0);
    setZipExportRequested(true);
  };

  const handleExportFigure = (withColorbar: boolean) => {
    if (lockExport) return;
    setExportAnchor(null);
    if (!allFloats || allFloats.length === 0) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const sliceData = [
      extractXY(allFloats, nx, ny, nz, sliceZ),
      extractXZ(allFloats, nx, ny, nz, sliceY),
      extractYZ(allFloats, nx, ny, nz, sliceX),
    ];
    for (let a = 0; a < 3; a++) {
      const [sliceH, sliceW] = sliceDims[a];
      const processed = logScale ? applyLogScale(sliceData[a]) : sliceData[a];
      let vmin: number, vmax: number;
      if (autoContrast) {
        ({ vmin, vmax } = percentileClip(processed, 2, 98));
      } else if (imageVminPct > 0 || imageVmaxPct < 100) {
        const { min: dMin, max: dMax } = findDataRange(processed);
        ({ vmin, vmax } = sliderRange(dMin, dMax, imageVminPct, imageVmaxPct));
      } else {
        const r = findDataRange(processed);
        vmin = r.min;
        vmax = r.max;
      }
      const offscreen = renderToOffscreen(processed, sliceW, sliceH, lut, vmin, vmax);
      if (!offscreen) continue;
      const axisLabel = AXES[a].toUpperCase();
      const sliceIndices = [sliceZ, sliceY, sliceX];
      const figCanvas = exportFigure({
        imageCanvas: offscreen,
        title: `${title || "Volume"} — ${axisLabel} slice ${sliceIndices[a]}`,
        lut,
        vmin,
        vmax,
        logScale,
        pixelSize: pixelSize > 0 ? pixelSize : undefined,
        showColorbar: withColorbar,
        showScaleBar: pixelSize > 0,
      });
      canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, `show3dvolume_figure_${AXES[a]}.pdf`));
    }
  };

  // -------------------------------------------------------------------------
  // FFT Zoom/Pan handlers
  // -------------------------------------------------------------------------
  const handleFftWheel = (e: React.WheelEvent, axis: number) => {
    const canvas = fftCanvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const zs = fftZooms[axis];
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const imgX = (mouseX - cx - zs.panX) / zs.zoom + cx;
    const imgY = (mouseY - cy - zs.panY) / zs.zoom + cy;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * factor));
    const newPanX = mouseX - (imgX - cx) * newZoom - cx;
    const newPanY = mouseY - (imgY - cy) * newZoom - cy;
    setFftZooms(prev => { const next = [...prev]; next[axis] = { zoom: newZoom, panX: newPanX, panY: newPanY }; return next; });
  };

  const handleFftDoubleClick = (axis: number) => {
    setFftZooms(prev => { const next = [...prev]; next[axis] = DEFAULT_ZOOM; return next; });
  };

  const handleFftMouseDown = (e: React.MouseEvent, axis: number) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY, axis };
    const zs = fftZooms[axis];
    setFftDragAxis(axis);
    setFftDragStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
  };

  const handleFftMouseMove = (e: React.MouseEvent, axis: number) => {
    if (fftDragAxis !== axis || !fftDragStart) return;
    const canvas = fftCanvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dx = (e.clientX - fftDragStart.x) * (canvas.width / rect.width);
    const dy = (e.clientY - fftDragStart.y) * (canvas.height / rect.height);
    setFftZooms(prev => { const next = [...prev]; next[axis] = { ...prev[axis], panX: fftDragStart.pX + dx, panY: fftDragStart.pY + dy }; return next; });
  };

  const handleFftMouseUp = (e: React.MouseEvent, axis: number) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current && fftClickStartRef.current.axis === axis) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = fftCanvasRefs.current[axis];
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const { w: cw, h: ch } = canvasSizes[axis];
          const zs = fftZooms[axis];

          // Determine FFT dimensions for this axis
          const dims: [number, number][] = [[ny, nx], [nz, nx], [nz, ny]];
          const [sliceH, sliceW] = dims[axis];
          const fftW = nextPow2(sliceW);
          const fftH = nextPow2(sliceH);

          const mouseX = (e.clientX - rect.left) * (cw / rect.width);
          const mouseY = (e.clientY - rect.top) * (ch / rect.height);
          const cx = cw / 2, cy = ch / 2;
          const imgX = (mouseX - cx - zs.panX) / zs.zoom + cx;
          const imgY = (mouseY - cy - zs.panY) / zs.zoom + cy;
          let imgCol = imgX / cw * fftW;
          let imgRow = imgY / ch * fftH;

          // Snap to nearest Bragg spot
          const cachedMag = fftMagCacheRefs.current[axis];
          if (cachedMag && imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
            const snapped = findFFTPeak(cachedMag, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
            imgCol = snapped.col;
            imgRow = snapped.row;
          }

          if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
            const dcCol = imgCol - fftW / 2;
            const dcRow = imgRow - fftH / 2;
            const distPx = Math.sqrt(dcCol * dcCol + dcRow * dcRow);

            if (distPx < 1) {
              setFftClickInfo(null);
            } else {
              let spatialFreq: number | null = null;
              let dSpacing: number | null = null;
              if (pixelSize > 0) {
                const paddedW = fftW;
                const paddedH = fftH;
                const freqC = dcCol / paddedW / pixelSize;
                const freqR = dcRow / paddedH / pixelSize;
                spatialFreq = Math.sqrt(freqC * freqC + freqR * freqR);
                dSpacing = spatialFreq > 0 ? 1 / spatialFreq : null;
              }
              setFftClickInfo({ axis, row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
            }
          }
        }
      }
    }
    fftClickStartRef.current = null;
    setFftDragAxis(null);
    setFftDragStart(null);
  };

  const handleFftResetAll = () => { setFftZooms([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]); setFftClickInfo(null); };

  const fftNeedsReset = fftZooms.some(z => z.zoom !== 1 || z.panX !== 0 || z.panY !== 0);

  // -------------------------------------------------------------------------
  // Canvas resize (matching Show2D)
  // -------------------------------------------------------------------------
  const handleResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: canvasTarget });
  };

  React.useEffect(() => {
    if (!isResizing) return;
    let rafId = 0;
    let latestSize = resizeStart ? resizeStart.size : canvasTarget;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      latestSize = Math.max(300, Math.min(800, resizeStart.size + delta));
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setCanvasTarget(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
      setCanvasTarget(latestSize);
      setIsResizing(false);
      setResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, resizeStart]);

  // -------------------------------------------------------------------------
  // Labels and setters
  // -------------------------------------------------------------------------
  const dl = dimLabels || ["Z", "Y", "X"];
  const axisLabels = [
    `${dl[1]}${dl[2]} (${dl[0]}=${sliceZ})`,
    `${dl[0]}${dl[2]} (${dl[1]}=${sliceY})`,
    `${dl[0]}${dl[1]} (${dl[2]}=${sliceX})`,
  ];
  const sliceValues = [sliceZ, sliceY, sliceX];
  const sliceValuesRef = React.useRef(sliceValues);
  sliceValuesRef.current = sliceValues;
  const sliceMaxes = [nz - 1, ny - 1, nx - 1];
  const sliceSetters = [
    (_: Event, v: number | number[]) => setSliceZ(v as number),
    (_: Event, v: number | number[]) => setSliceY(v as number),
    (_: Event, v: number | number[]) => setSliceX(v as number),
  ];

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <Box className="show3dvolume-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ ...container.root, bgcolor: tc.bg, color: tc.text }}>
      {/* 3D Volume Renderer */}
      {!hideVolume && (
      <Box sx={{ mb: `${SPACING.LG}px` }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
          <Typography variant="caption" sx={{ ...typography.label }}>
            {title || "Volume 3D"}<InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum (Fourier transform) below each slice.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile-based contrast (2nd-98th percentile). FFT Auto masks DC + clips to 99.9th.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Cross: Show crosshair lines indicating orthogonal slice positions.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Colorbar: Display colorbar overlay on each slice canvas.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop: Loop playback. Drag end markers on slider for loop range.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Bounce: Ping-pong playback — alternates forward and reverse.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Opacity/Bright/Planes: 3D volume renderer controls.</Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
              <KeyboardShortcuts items={[["Space", "Play / Pause"], ["← / →", "Prev / Next slice"], ["Home / End", "First / Last slice"], ["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
            </Box>} theme={themeInfo.theme} />
            <ControlCustomizer
              widgetName="Show3DVolume"
              hiddenTools={hiddenTools}
              setHiddenTools={setHiddenTools}
              disabledTools={disabledTools}
              setDisabledTools={setDisabledTools}
              themeColors={tc}
            />
          </Typography>
          <Stack direction="row" alignItems="center" spacing={0.5}>
            {!hideDisplay && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>FFT:</Typography>
                <Switch
                  checked={showFft}
                  onChange={(e) => { if (!lockDisplay) setShowFft(e.target.checked); }}
                  disabled={lockDisplay}
                  size="small"
                  sx={switchStyles.small}
                />
              </>
            )}
            {!hideExport && (
              <Button size="small" sx={{ ...compactButton, color: tc.accent }} disabled={lockExport} onClick={async () => {
              const canvas = canvasRefs.current[0];
              if (!canvas) return;
              try {
                const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/png"));
                if (!blob) return;
                await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
              } catch {
                canvas.toBlob((b) => { if (b) downloadBlob(b, "show3dvolume_xy.png"); }, "image/png");
              }
              }}>Copy</Button>
            )}
            {!hideExport && (
              <>
                <Button size="small" sx={{ ...compactButton, color: tc.accent }} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }} disabled={lockExport || exporting}>{exporting ? "Exporting..." : "Export"}</Button>
                <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                  <MenuItem disabled={lockExport} onClick={() => handleExportFigure(true)} sx={{ fontSize: 12 }}>Figure + colorbar</MenuItem>
                  <MenuItem disabled={lockExport} onClick={() => handleExportFigure(false)} sx={{ fontSize: 12 }}>Figure</MenuItem>
                  <MenuItem disabled={lockExport} onClick={handleExportPng} sx={{ fontSize: 12 }}>PNG (current slices)</MenuItem>
                  <MenuItem disabled={lockExport} onClick={handleExportGif} sx={{ fontSize: 12 }}>GIF (animation)</MenuItem>
                  <MenuItem disabled={lockExport} onClick={handleExportZip} sx={{ fontSize: 12 }}>ZIP (all slices)</MenuItem>
                </Menu>
              </>
            )}
            {!hideView && (
              <Button size="small" sx={compactButton} disabled={lockView || lockVolume || !cameraChanged} onClick={() => { if (!lockView && !lockVolume) handleVolumeDoubleClick(); }}>Reset View</Button>
            )}
          </Stack>
        </Stack>
        {webglSupported ? (
          <Box
            sx={{
              ...container.imageBox,
              width: volumeCanvasSize,
              height: volumeCanvasSize,
              cursor: lockVolume ? "default" : (volumeDrag ? "grabbing" : "grab"),
            }}
            onMouseDown={(e) => { if (!lockVolume) handleVolumeMouseDown(e); }}
            onMouseMove={(e) => { if (!lockVolume) handleVolumeMouseMove(e); }}
            onMouseUp={() => { if (!lockVolume) handleVolumeMouseUp(); }}
            onMouseLeave={() => { if (!lockVolume) handleVolumeMouseUp(); }}
            onWheel={(e) => { if (!lockVolume) handleVolumeWheel(e); }}
            onDoubleClick={() => { if (!lockVolume && !lockView) handleVolumeDoubleClick(); }}
            onContextMenu={(e) => e.preventDefault()}
          >
            <canvas
              ref={volumeCanvasRef}
              width={volumeCanvasSize}
              height={volumeCanvasSize}
              style={{ width: volumeCanvasSize, height: volumeCanvasSize, display: "block" }}
            />
            {/* Resize handle */}
            <Box
              onMouseDown={(e) => { if (!lockVolume) handleVolumeResizeStart(e); }}
              sx={{
                position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                cursor: lockVolume ? "default" : "nwse-resize", opacity: lockVolume ? 0.2 : 0.4,
                background: `linear-gradient(135deg, transparent 50%, ${tc.textMuted} 50%)`,
                "&:hover": { opacity: 1 },
              }}
            />
          </Box>
        ) : (
          <Box sx={{
            ...container.imageBox, width: volumeCanvasSize, height: 80,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Typography sx={{ ...typography.label, color: tc.textMuted, px: 2, textAlign: "center" }}>
              WebGL 2 not available. 3D volume rendering requires a WebGL 2 capable browser.
            </Typography>
          </Box>
        )}
        {/* Volume rendering controls */}
        {webglSupported && (
          <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
            <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Opacity:</Typography>
            <Slider
              value={volumeOpacity} min={0} max={1} step={0.01}
              onChange={(_, v) => setVolumeOpacity(v as number)}
              disabled={lockVolume}
              size="small" sx={{ ...sliderStyles.small, width: 60 }}
            />
            <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 24 }}>{volumeOpacity.toFixed(2)}</Typography>
            <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Bright:</Typography>
            <Slider
              value={volumeBrightness} min={0.1} max={3} step={0.1}
              onChange={(_, v) => setVolumeBrightness(v as number)}
              disabled={lockVolume}
              size="small" sx={{ ...sliderStyles.small, width: 60 }}
            />
            <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 24 }}>{volumeBrightness.toFixed(1)}</Typography>
            <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Planes:</Typography>
            <Switch checked={showSlicePlanes} onChange={(e) => setShowSlicePlanes(e.target.checked)} disabled={lockVolume} size="small" sx={switchStyles.small} />
          </Box>
        )}
      </Box>
      )}
      {/* Slice canvases row */}
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {AXES.map((_, a) => {
          const { w: cw, h: ch } = canvasSizes[a];
          return (
            <Box key={a} sx={{ minWidth: cw }}>
              {/* Header row matching Show3D */}
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
                <Typography variant="caption" sx={{ ...typography.label }}>{a === 0 && title ? title : axisLabels[a]}</Typography>
                {a === 0 && (
                  <Button size="small" sx={compactButton} disabled={lockView || !needsReset} onClick={() => { if (!lockView) handleResetAll(); }}>Reset</Button>
                )}
              </Stack>
              {/* Canvas with plane-colored border */}
              <Box
                sx={{ ...container.imageBox, width: cw, height: ch, cursor: "grab", borderColor: ["#4d80ff", "#4dff66", "#ff4d4d"][a] }}
                onMouseDown={(e) => { if (!lockView) handleMouseDown(e, a); }}
                onMouseMove={(e) => handleMouseMove(e, a)}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseLeave}
                onWheel={(e) => { if (!lockView) handleWheel(e, a); }}
                onDoubleClick={() => { if (!lockView) handleDoubleClick(a); }}
              >
                <canvas
                  ref={(el) => { canvasRefs.current[a] = el; }}
                  width={cw}
                  height={ch}
                  style={{ width: cw, height: ch, imageRendering: "pixelated" }}
                />
                <canvas
                  ref={(el) => { overlayRefs.current[a] = el; }}
                  width={cw}
                  height={ch}
                  style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
                />
                <canvas
                  ref={(el) => { uiRefs.current[a] = el; }}
                  width={Math.round(cw * DPR)}
                  height={Math.round(ch * DPR)}
                  style={{ position: "absolute", top: 0, left: 0, width: cw, height: ch, pointerEvents: "none" }}
                />
                {/* Cursor readout overlay */}
                {cursorInfo && cursorInfo.view === ["XY", "XZ", "YZ"][a] && (
                  <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                    <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                      ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                    </Typography>
                  </Box>
                )}
                {/* Resize handle */}
                <Box
                  onMouseDown={(e) => { if (!lockView) handleResizeStart(e); }}
                  sx={{
                    position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                    cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.2 : 0.4,
                    background: `linear-gradient(135deg, transparent 50%, ${tc.textMuted} 50%)`,
                    "&:hover": { opacity: 1 },
                  }}
                />
              </Box>
              {/* Stats bar */}
              {showStats && !hideStats && (
                <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: tc.bgAlt, display: "flex", gap: 2, alignItems: "center", overflow: "hidden", whiteSpace: "nowrap", width: cw, boxSizing: "border-box", opacity: lockStats ? 0.6 : 1 }}>
                  {[
                    { label: "Mean", value: statsMean?.[a] },
                    { label: "Min", value: statsMin?.[a] },
                    { label: "Max", value: statsMax?.[a] },
                    { label: "Std", value: statsStd?.[a] },
                  ].map(({ label, value }) => (
                    <Typography key={label} sx={{ fontSize: 11, color: tc.textMuted, whiteSpace: "nowrap" }}>
                      {label} <Box component="span" sx={{ color: tc.accent, fontFamily: "monospace", fontSize: 10 }}>{value !== undefined ? formatNumber(value) : "-"}</Box>
                    </Typography>
                  ))}
                </Box>
              )}
              {/* FFT canvas (inline, below stats) */}
              {effectiveShowFft && (
                <Box sx={{ mt: `${SPACING.SM}px` }}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 20 }}>
                    <Stack direction="row" alignItems="center" sx={{ overflow: "hidden" }}>
                      <Typography variant="caption" sx={{ ...typography.label, fontSize: 10, flexShrink: 0 }}>
                        {`FFT ${["XY", "XZ", "YZ"][a]} ${gpuReady ? "(GPU)" : "(CPU)"}`}
                      </Typography>
                      {fftClickInfo && fftClickInfo.axis === a && (
                        <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: tc.textMuted, ml: 1, whiteSpace: "nowrap" }}>
                          {fftClickInfo.dSpacing != null ? (
                            <>d=<Box component="span" sx={{ color: tc.accent, fontWeight: "bold" }}>{fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} \u00C5`}</Box>{" |g|="}<Box component="span" sx={{ color: tc.accent }}>{fftClickInfo.spatialFreq!.toFixed(4)} \u00C5\u207B\u00B9</Box></>
                          ) : (
                            <>dist=<Box component="span" sx={{ color: tc.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                          )}
                        </Typography>
                      )}
                    </Stack>
                    {a === 0 && fftNeedsReset && (
                      <Button size="small" sx={compactButton} disabled={lockView} onClick={() => { if (!lockView) handleFftResetAll(); }}>Reset</Button>
                    )}
                  </Stack>
                  <Box
                    sx={{ ...container.imageBox, width: cw, height: ch, cursor: "grab", borderColor: ["#4d80ff", "#4dff66", "#ff4d4d"][a] }}
                    onMouseDown={(e) => { if (!lockView) handleFftMouseDown(e, a); }}
                    onMouseMove={(e) => { if (!lockView) handleFftMouseMove(e, a); }}
                    onMouseUp={(e) => { if (!lockView) handleFftMouseUp(e, a); }}
                    onMouseLeave={() => { if (!lockView) { fftClickStartRef.current = null; setFftDragAxis(null); setFftDragStart(null); } }}
                    onWheel={(e) => { if (!lockView) handleFftWheel(e, a); }}
                    onDoubleClick={() => { if (!lockView) handleFftDoubleClick(a); }}
                  >
                    <canvas
                      ref={(el) => { fftCanvasRefs.current[a] = el; }}
                      width={cw}
                      height={ch}
                      style={{ width: cw, height: ch, imageRendering: "pixelated" }}
                    />
                    <canvas
                      ref={(el) => { fftOverlayRefs.current[a] = el; }}
                      width={Math.round(cw * DPR)}
                      height={Math.round(ch * DPR)}
                      style={{ position: "absolute", top: 0, left: 0, width: cw, height: ch, pointerEvents: "none" }}
                    />
                  </Box>
                  {fftZooms[a].zoom !== 1 && (
                    <Typography sx={{ ...typography.label, fontSize: 10, color: tc.accent, fontWeight: "bold", mt: 0.25, textAlign: "right" }}>
                      {fftZooms[a].zoom.toFixed(1)}x
                    </Typography>
                  )}
                </Box>
              )}
              {/* Slider row (always at bottom) — 3-thumb when loop on, single when off */}
              {!hidePlayback && (
              <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, width: cw, maxWidth: cw, boxSizing: "border-box" }}>
                <Typography sx={{ ...typography.labelSmall, color: tc.textMuted, flexShrink: 0 }}>{dl[a]}</Typography>
                {loop ? (
                  <Slider
                    value={[loopStarts[a], sliceValues[a], effectiveLoopEnds[a]]}
                    onChange={(_, v) => {
                      const vals = v as number[];
                      setLoopStarts(prev => { const next = [...prev]; next[a] = vals[0]; return next; });
                      [setSliceZ, setSliceY, setSliceX][a](vals[1]);
                      setLoopEnds(prev => { const next = [...prev]; next[a] = vals[2]; return next; });
                    }}
                    disableSwap
                    min={0}
                    max={sliceMaxes[a]}
                    disabled={lockPlayback}
                    size="small"
                    valueLabelDisplay="auto"
                    valueLabelFormat={(v) => `${v}`}
                    sx={{
                      ...sliderStyles.small,
                      flex: 1,
                      minWidth: 40,
                      "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: tc.textMuted },
                      "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 },
                      "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: tc.textMuted },
                      "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
                    }}
                  />
                ) : (
                  <Slider
                    value={sliceValues[a]}
                    min={0}
                    max={sliceMaxes[a]}
                    onChange={sliceSetters[a]}
                    disabled={lockPlayback}
                    size="small"
                    sx={{ ...sliderStyles.small, flex: 1, minWidth: 40 }}
                  />
                )}
                <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 28, textAlign: "right", flexShrink: 0 }}>
                  {sliceValues[a]}/{sliceMaxes[a]}
                </Typography>
                {zooms[a].zoom !== 1 && (
                  <Typography sx={{ ...typography.label, fontSize: 10, color: tc.accent, fontWeight: "bold" }}>{zooms[a].zoom.toFixed(1)}x</Typography>
                )}
              </Box>
              )}
            </Box>
          );
        })}
      </Stack>
      {/* FFT controls row */}
      {effectiveShowFft && (
        <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
          <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>FFT Scale:</Typography>
          <Select disabled={lockDisplay} value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps}>
            <MenuItem value="linear">Lin</MenuItem>
            <MenuItem value="log">Log</MenuItem>
          </Select>
          <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Color:</Typography>
          <Select disabled={lockDisplay} value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps}>
            {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
          </Select>
          <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Auto:</Typography>
          <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} disabled={lockDisplay} size="small" sx={switchStyles.small} />
        </Box>
      )}
      {/* Controls row with histogram on right */}
      {showControls && (!hideDisplay || !hideHistogram) && (
        <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "fit-content" }}>
          {!hideDisplay && (
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, justifyContent: "center" }}>
              <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Scale:</Typography>
                <Select disabled={lockDisplay} value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Color:</Typography>
                <Select disabled={lockDisplay} size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }}>
                  {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                </Select>
              </Box>
              <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Auto:</Typography>
                <Switch checked={autoContrast} onChange={(e) => { if (!lockDisplay) setAutoContrast(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Cross:</Typography>
                <Switch checked={showCrosshair} onChange={(e) => { if (!lockDisplay) setShowCrosshair(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
                <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>Colorbar:</Typography>
                <Switch checked={showColorbar} onChange={(e) => { if (!lockDisplay) setShowColorbar(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} />
              </Box>
            </Box>
          )}
          {!hideHistogram && (
            <Box sx={{ display: "flex", flexDirection: "column", justifyContent: "center", opacity: lockHistogram ? 0.6 : 1 }}>
              <Histogram
                data={imageHistogramData}
                vminPct={imageVminPct}
                vmaxPct={imageVmaxPct}
                onRangeChange={(min, max) => {
                  if (!lockHistogram) {
                    setImageVminPct(min);
                    setImageVmaxPct(max);
                  }
                }}
                width={110}
                height={58}
                theme={themeInfo.theme}
                dataMin={imageDataRange.min}
                dataMax={imageDataRange.max}
              />
            </Box>
          )}
        </Box>
      )}
      {/* Playback: transport + axis selector + fps + loop + bounce */}
      {!hidePlayback && (
      <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg }}>
        <Select
          value={playAxis}
          onChange={(e) => { if (!lockPlayback) { setPlaying(false); setPlayAxis(e.target.value as number); } }}
          disabled={lockPlayback}
          size="small"
          sx={{ ...themedSelect, minWidth: 40, fontSize: 10 }}
          MenuProps={themedMenuProps}
        >
          <MenuItem value={0}>{dl[0]}</MenuItem>
          <MenuItem value={1}>{dl[1]}</MenuItem>
          <MenuItem value={2}>{dl[2]}</MenuItem>
          <MenuItem value={3}>All</MenuItem>
        </Select>
        <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
          <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) { setReverse(true); setPlaying(true); } }} sx={{ color: reverse && playing ? tc.accent : tc.textMuted, p: 0.25 }}>
            <FastRewindIcon sx={{ fontSize: 18 }} />
          </IconButton>
          <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) setPlaying(!playing); }} sx={{ color: tc.accent, p: 0.25 }}>
            {playing ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
          </IconButton>
          <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) { setReverse(false); setPlaying(true); } }} sx={{ color: !reverse && playing ? tc.accent : tc.textMuted, p: 0.25 }}>
            <FastForwardIcon sx={{ fontSize: 18 }} />
          </IconButton>
          <IconButton size="small" disabled={lockPlayback} onClick={() => {
            if (!lockPlayback) {
              setPlaying(false);
              if (playAxis === 3) {
                for (let a = 0; a < 3; a++) sliceSettersRef.current[a](loopStarts[a]);
              } else {
                sliceSettersRef.current[playAxis](loopStarts[playAxis]);
              }
            }
          }} sx={{ color: tc.textMuted, p: 0.25 }}>
            <StopIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Stack>
        <Typography sx={{ ...typography.label, color: tc.textMuted, flexShrink: 0 }}>fps</Typography>
        <Slider disabled={lockPlayback} value={fps} min={1} max={30} step={1} onChange={(_, v) => setFps(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35, flexShrink: 0 }} />
        <Typography sx={{ ...typography.label, color: tc.textMuted, minWidth: 14, flexShrink: 0 }}>{Math.round(fps)}</Typography>
        <Typography sx={{ ...typography.label, color: tc.textMuted, flexShrink: 0 }}>Loop</Typography>
        <Switch size="small" checked={loop} onChange={() => { if (!lockPlayback) setLoop(!loop); }} disabled={lockPlayback} sx={{ ...switchStyles.small, flexShrink: 0 }} />
        <Typography sx={{ ...typography.label, color: tc.textMuted, flexShrink: 0 }}>Bounce</Typography>
        <Switch size="small" checked={boomerang} onChange={() => { if (!lockPlayback) setBoomerang(!boomerang); }} disabled={lockPlayback} sx={{ ...switchStyles.small, flexShrink: 0 }} />
      </Box>
      )}
    </Box>
  );
}

export const render = createRender(Show3DVolume);
