/**
 * ShowPtycho: interactive ptychography aberration explorer widget.
 *
 * Two panels (Phase + optional FFT) following Show2D/Live design.
 * Uses the full bright-field set automatically during interaction and saving.
 * Features: zoom/pan, scale bar, histogram contrast, colormap selector,
 * pin system, resize handle.
 */

import * as React from "react";
import { useLocalTiltWorkspace } from "./localTilt";
import { idleDiagnostic, latestFrame } from "./interaction";
import { FFTMagnitudeDisplay } from "./fftDisplay";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Switch from "@mui/material/Switch";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Popover from "@mui/material/Popover";
import Badge from "@mui/material/Badge";
import Tooltip from "@mui/material/Tooltip";
import LinearProgress from "@mui/material/LinearProgress";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import PushPinIcon from "@mui/icons-material/PushPin";
import SaveIcon from "@mui/icons-material/Save";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import { useTheme, WIDGET_UI_FONT, WIDGET_TYPOGRAPHY as typography, type ThemeColors } from "../theme";
import { SPACING, switchStyles } from "../controlStyles";
import { downloadBlob, extractFloat32, formatNumber } from "../format";
import { computeHistogramFromBytes, percentileClip, percentileWindow } from "../display/stats";
import { DisplayStatsWorker, frameStats, type FrameStats } from "../display/displayStats";
import { createPanelFrames, type PanelFrame } from "./panelFrames";
import { useChangedState, useStoreValue, type ValueStore } from "../shared/valueStore";
import { RenderPathBadge } from "../shared/RenderPathBadge";
import { shouldIgnoreWidgetShortcut } from "../shared/widgetShortcuts";
import { COLORMAPS, COLORMAP_NAMES, applyColormap, renderToOffscreen, GPUColormapEngine, createGPUColormapEngine } from "../display/colormaps";
import { reciprocalCoordinatesFromShiftedOffset, getDisplayFFT, type DisplayFFT } from "../display/fft";
import { findFFTPeakBrowser } from "../display/geometry";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, strokePeakMarker } from "../figure";
import { degreesWithin, wrapDegrees } from "../units";
import { type SampleValues, SNAPSHOTS_SCHEMA, higherOrderAzimuthKey, higherOrderMagnitudeKey, reconstruct } from "./engineAdapter";
import { WebGPUSSBBackend, deleteSSBFolderFile, readSSBFolderBytes, readSSBFolderJson, setSSBLocalDirectory, setSSBLocalFiles, ssbFolderWritable, ssbNeedsLocalSource, writeSSBFolderFile, type WebGPULoadProgress, type WebGPUSSBResult } from "../.generated/engine/ssb/webgpu/backend";

/* ================================================================
   Design tokens (matching Live / Show2D)
   ================================================================ */

// Portaled menus use the same UI font as the other viewers; values stay monospace.
const widgetTheme = createTheme({ typography: { fontFamily: WIDGET_UI_FONT } });

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: WIDGET_UI_FONT, overflow: "visible" },
  imageBox: { overflow: "hidden", position: "relative" as const },
};

function compactButton(tc: ThemeColors) {
  return {
    ...typography.labelSmall,
    textTransform: "none",
    py: 0.25,
    px: 1,
    minWidth: 0,
    boxSizing: "border-box",
    whiteSpace: "nowrap",
    flexShrink: 0,
    "&.Mui-disabled": { color: tc.textMuted, borderColor: tc.border },
  };
}

function compactIconButton(tc: ThemeColors, color?: string) {
  return {
    width: 26,
    height: 26,
    minWidth: 26,
    flex: "0 0 26px",
    boxSizing: "border-box",
    p: 0.25,
    border: `1px solid ${color || tc.border}`,
    borderRadius: "4px",
    color: color || tc.textMuted,
    bgcolor: tc.bgAlt,
    transition: "none",
    transform: "none",
    "&:active": { transform: "none" },
    "& .MuiTouchRipple-root": { display: "none" },
    "&.Mui-focusVisible": {
      outline: `1px solid ${color || tc.accent}`,
      outlineOffset: 0,
    },
    "&:hover": {
      bgcolor: tc.controlBg,
      borderColor: color || tc.accent,
      color: color || tc.accent,
    },
    "&.Mui-disabled": {
      color: tc.textMuted,
      borderColor: tc.border,
      opacity: 0.45,
    },
  };
}

function compactBareIconButton(tc: ThemeColors, color?: string) {
  return {
    width: 22,
    height: 22,
    p: 0,
    border: "none",
    borderRadius: "50%",
    color: color || tc.textMuted,
    bgcolor: "transparent",
    "&:hover": {
      bgcolor: tc.controlBg,
      color: color || tc.accent,
    },
    "&.Mui-disabled": {
      color: tc.textMuted,
      opacity: 0.32,
    },
  };
}

function controlBand(tc: ThemeColors) {
  return {
    display: "flex",
    alignItems: "center",
    gap: `${SPACING.SM}px`,
    px: 1,
    py: 0.5,
    border: `1px solid ${tc.border}`,
    bgcolor: tc.bgAlt,
    flexWrap: "wrap",
    rowGap: `${SPACING.XS}px`,
  };
}

function statChip(tc: ThemeColors, emphasis = false) {
  return {
    px: 0,
    py: 0,
    border: "none",
    bgcolor: "transparent",
    color: emphasis ? tc.accent : tc.textMuted,
    whiteSpace: "nowrap",
  };
}

function webgpuProgressLabel(progress: WebGPULoadProgress): string {
  const parts = [progress.message];
  if (progress.detail) parts.push(progress.detail);
  if (progress.current != null && progress.total != null && progress.total > 0) {
    parts.push(`${Math.round(progress.current)}/${Math.round(progress.total)}`);
  }
  if (progress.elapsedMs != null && Number.isFinite(progress.elapsedMs)) {
    parts.push(`${(progress.elapsedMs / 1000).toFixed(1)} s`);
  }
  return parts.join(" · ");
}

function webgpuProgressPercent(progress: WebGPULoadProgress): number | null {
  if (progress.percent != null && Number.isFinite(progress.percent)) {
    return Math.max(0, Math.min(100, progress.percent));
  }
  if (progress.current != null && progress.total != null && progress.total > 0) {
    return Math.max(0, Math.min(100, (progress.current / progress.total) * 100));
  }
  return null;
}

function compactRuntimeStatus(status: string): string {
  return status
    .replace(/^apple\s+metal-\d+\s+/i, "")
    .replace(/^WebGPU folder /, "WebGPU ")
    .replace(/\((\d+\/\d+) BF, C10=([^)]+?) nm, rot=([^)]+?)°([^)]*)\)/, "$1 BF · C10 $2 nm · rot $3°$4")
    .replace(/,\s+HO=/, " · HO ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Status line after a browser WebGPU reconstruction, naming the adapter (and whether it is software), the path, the BF
 *  count and the aberrations it ran with, so a slow or wrong frame can be traced to its configuration. */
function webgpuRunStatus(result: WebGPUSSBResult, modeLabel: string, totalBf: number, c10Nm: number, hoLabel: string): string {
  return `${result.adapterInfo}${result.softwareAdapter ? " software" : ""} WebGPU folder ${modeLabel} (${result.numBrightFieldPixels}/${totalBf} BF, C10=${c10Nm.toFixed(1)} nm, rot=${result.scanDetectorRotationDeg.toFixed(1)}°${hoLabel})`;
}

function compactPathLabel(path: string, depth = 2): string {
  const parts = path.split("/").filter(Boolean);
  if (parts.length <= depth) return path;
  return parts.slice(-depth).join("/");
}

function formatStat(value: number): string {
  if (!isFinite(value)) return "--";
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 1e4 || magnitude < 0.01) return value.toExponential(2);
  if (magnitude >= 100) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

/* ================================================================
   Constants
   ================================================================ */

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 20;
const ZOOM_FACTOR = 1.15;
const MIN_PANEL = 200;
const DEFAULT_PANEL = 800;
// C12 (nm) and phi12 (degrees) slider spans: the SSB fit's default search range
const C12_RANGE: [number, number] = [-10, 10];
const PHI12_RANGE: [number, number] = [-90, 90];
const MAX_PINS = 20;
// Frames in one PLAY sweep from range start to end (then it bounces back).
const SWEEP_STEPS = 40;
// Thumb bitmap is rendered at the largest size and CSS-scaled down, so it keeps its quality
// when the user flips to a larger preset without re-making the thumbnail.
const THUMB_BITMAP_PX = 128;
type ThumbSize = "S" | "M" | "L";
const THUMB_SIZE_PX: Record<ThumbSize, number> = { S: 44, M: 72, L: 120 };
const PANEL_GAP = 0;
const ACTION_CONTROL_HEIGHT = 26;
const ABERRATION_SLIDER_WIDTH = 144;
// Overview slider requests the kernel may hold at once: two keep it computing while a phase is in transit.
const PREVIEW_REQUESTS_IN_FLIGHT = 2;
const ROTATION_SLIDER_WIDTH = 170;

// Semantic status colors, used only for loss delta (better/worse) and the play
// indicator (active sweep).  Kept as constants so there's one place to retune.
const STATUS_GOOD = "#4caf50";
const STATUS_BAD = "#f44336";

type ZoomState = { zoom: number; panX: number; panY: number };
const ZOOM_RESET: ZoomState = { zoom: 1, panX: 0, panY: 0 };
// Amplitude and complex views exist only for an object-wave array: SSB recovers the phase alone.
type ExtraRealViewMode = "amp" | "complex";
type FFTPlacement = "panel" | "inset";
type FFTInsetBox = { x: number; y: number };
type GPUColormapSlot = 0 | 1 | 2;

interface PinnedEntry {
  id: number;
  C10_nm: number;
  C12_nm: number;
  phi12_deg: number;
  scan_detector_rotation_deg: number;
  display_negate_phase: boolean;
  loss: number;
  starred: boolean;
  generated_at_utc: string;
  phaseData: Float32Array;
  w: number;
  h: number;
  thumb: HTMLCanvasElement | null;
}

/* ================================================================
   Rendering helpers
   ================================================================ */

function renderPhaseOffscreen(
  data: Float32Array, w: number, h: number,
  lut: Uint8Array, pctLo: number, pctHi: number,
): { canvas: HTMLCanvasElement | null; min: number; max: number } {
  const { vmin, vmax, min, max } = percentileClip(data, pctLo, pctHi);
  return { canvas: renderToOffscreen(data, w, h, lut, vmin, vmax), min, max };
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

function renderComplexPhaseOffscreen(
  phase: Float32Array,
  w: number,
  h: number,
  pctLo: number,
  pctHi: number,
  amp: Float32Array,
): { canvas: HTMLCanvasElement | null; min: number; max: number } {
  // hue = phase, value = the object wave's amplitude
  const { vmin, vmax, min, max } = percentileClip(amp, pctLo, pctHi);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return { canvas: null, min, max };
  const image = ctx.createImageData(w, h);
  const range = vmax > vmin ? vmax - vmin : 1;
  for (let i = 0; i < phase.length; i++) {
    const hue = ((phase[i] + Math.PI) / (2 * Math.PI) % 1 + 1) % 1;
    const clipped = Math.max(vmin, Math.min(vmax, amp[i]));
    const value = Math.max(0, Math.min(1, (clipped - vmin) / range));
    const [r, g, b] = hsvToRgb(hue, 0.92, value);
    const j = i * 4;
    image.data[j] = Math.round(r * 255);
    image.data[j + 1] = Math.round(g * 255);
    image.data[j + 2] = Math.round(b * 255);
    image.data[j + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return { canvas, min, max };
}

// Radius (px) searched when an FFT click snaps to the nearest peak, as Show2D's d-spacing snap does.
const FFT_SNAP_RADIUS = 5;

/** Display FFT magnitude (WebGPU or the JS reference) on the complete padded grid, DC centred and
 *  log-scaled, so the scale bar and d-spacing picks use the padded grid's reciprocal sampling. */
async function computeFFTMagnitude(
  fft: DisplayFFT, magnitudeDisplay: FFTMagnitudeDisplay,
  data: Float32Array, w: number, h: number,
): Promise<{ mag: Float32Array; pw: number; ph: number; stats: FrameStats }> {
  // Windowing, padding, the magnitude and its display statistics run in the display worker; the
  // input thread only uploads the interleaved grid and reads the spectrum back, so phase frames
  // keep arriving.
  const { complex, pw, ph } = await magnitudeDisplay.prepare(data, w, h);
  const spectrum = await fft.fft2DInterleaved(complex, pw, ph, false);
  const { magnitude, stats } = await magnitudeDisplay.magnitudeInterleaved(spectrum, pw, ph);
  return { mag: magnitude, pw, ph, stats };
}

function makeThumbnail(data: Float32Array, w: number, h: number, cmapName: string) {
  const lut = COLORMAPS[cmapName as keyof typeof COLORMAPS] || COLORMAPS.viridis;
  const { canvas: full } = renderPhaseOffscreen(data, w, h, lut, 1, 99);
  if (!full) return null;
  const thumb = document.createElement("canvas");
  thumb.width = THUMB_BITMAP_PX; thumb.height = THUMB_BITMAP_PX;
  thumb.getContext("2d")!.drawImage(full, 0, 0, THUMB_BITMAP_PX, THUMB_BITMAP_PX);
  return thumb;
}

function slugPart(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_+|_+$/g, "") || "pin";
}

function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Could not encode pinned image as PNG."));
    }, "image/png");
  });
}

/* ================================================================
   drawCanvas: DPR-aware (matching Live)
   ================================================================ */

function drawCanvas(
  canvas: HTMLCanvasElement | null,
  offscreen: HTMLCanvasElement | null,
  size: number,
  zoom: ZoomState,
  bgColor: string,
  textColor: string,
  smooth: boolean = false,
) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Only resize when the size actually changes. Assigning canvas.width/height
  // clears the canvas to transparent, so doing it every draw (e.g. on every
  // histogram-drag tick) wipes the visible image before the redraw lands, which
  // reads as flicker. When the size is unchanged, repaint in place instead.
  // read at draw time: moving the notebook to another monitor or zooming the page changes it
  const dpr = window.devicePixelRatio || 1;
  const targetW = Math.round(size * dpr);
  const targetH = Math.round(size * dpr);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, size, size);

  if (!offscreen) {
    ctx.fillStyle = textColor;
    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Waiting…", size / 2, size / 2);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return;
  }

  const { zoom: zoomLevel, panX, panY } = zoom;
  const baseFit = Math.min(size / offscreen.width, size / offscreen.height);
  const scale = baseFit * zoomLevel;
  const drawW = offscreen.width * scale, drawH = offscreen.height * scale;

  // Smooth on: bilinear via canvas2d. Smooth off: nearest-neighbor, so individual
  // pixels stay visible when zoomed in.
  if (smooth) {
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
  } else {
    ctx.imageSmoothingEnabled = false;
  }
  ctx.drawImage(offscreen, (size - drawW) / 2 + panX, (size - drawH) / 2 + panY, drawW, drawH);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

/** Paint completed pixels immediately; React still owns layout and controls. */
function useCanvasFrame(offscreen: HTMLCanvasElement | null, draw: () => void) {
  const drawRef = React.useRef(draw);
  React.useLayoutEffect(() => { drawRef.current = draw; draw(); }, [draw]);
  React.useLayoutEffect(() => {
    const paint = () => drawRef.current();
    offscreen?.addEventListener("quantem-frame", paint);
    return () => offscreen?.removeEventListener("quantem-frame", paint);
  }, [offscreen]);
}

/** Map a client-space mouse position to fractional image (row, col) through the same fit-to-box
 *  scale, zoom and pan that drawCanvas uses, so the cursor readout, crop drag and FFT peak pick
 *  all land on the pixel drawn under the mouse. */
function clientToImage(
  canvas: HTMLCanvasElement, clientX: number, clientY: number,
  image: HTMLCanvasElement, size: number, zoom: ZoomState,
): { row: number; col: number } {
  const rect = canvas.getBoundingClientRect();
  const mouseX = clientX - rect.left, mouseY = clientY - rect.top;
  const baseFit = Math.min(size / image.width, size / image.height);
  const scale = baseFit * zoom.zoom;
  const xOffset = (size - image.width * scale) / 2 + zoom.panX;
  const yOffset = (size - image.height * scale) / 2 + zoom.panY;
  return { row: (mouseY - yOffset) / scale, col: (mouseX - xOffset) / scale };
}

type FFTPeakMark = { row: number; col: number; dSpacing: number | null };

/** Mark a snapped FFT peak with its d-spacing, from the peak's offset to the DC pixel that fftshift
 *  puts at (fftH/2, fftW/2). A pick on DC itself has no lattice spacing, so it returns null and the
 *  marker clears; dSpacing stays null without a real-space pixel size. */
function measureFFTPeak(row: number, col: number, fftW: number, fftH: number, pixelSize?: number): FFTPeakMark | null {
  const dcol = col - Math.floor(fftW / 2);
  const drow = row - Math.floor(fftH / 2);
  if (Math.sqrt(dcol * dcol + drow * drow) < 1) return null;
  let dSpacing: number | null = null;
  if (pixelSize && pixelSize > 0) {
    ({ dSpacing } = reciprocalCoordinatesFromShiftedOffset(drow, dcol, fftH, fftW, pixelSize, pixelSize));
  }
  return { row, col, dSpacing };
}

/** Stroke the d-spacing marker: an open circle inside a cross whose arms stop 3 px short of the
 *  centre, so the picked peak itself stays visible under the marker. */
/* ================================================================
   Histogram (copied from Live)
   ================================================================ */

interface HistogramProps {
  /** The painted frame of the panel this histogram shows; it re-renders on that panel's frames only. */
  frame: ValueStore<PanelFrame>;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  tc: ThemeColors;
}

function Histogram({
  frame: frameStore, vminPct, vmaxPct, onRangeChange,
  width = 110, height = 40, tc,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const frame = useStoreValue(frameStore);
  const { min: dataMin, max: dataMax } = frame;
  // The display worker bins each frame once; a frame painted without them is binned here.
  const bins = React.useMemo(() => frame.displayBins ?? computeHistogramFromBytes(frame.data), [frame]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = tc.bgAlt;
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
      const inRange = i >= vminBin && i <= vmaxBin;
      ctx.fillStyle = inRange ? tc.textMuted : tc.border;
      ctx.fillRect(i * barWidth + 0.5, height - barH, Math.max(1, barWidth - 1), barH);
    }
  }, [bins, vminPct, vmaxPct, width, height, tc]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        onDoubleClick={() => onRangeChange(1, 99)}
        style={{ width, height, border: `1px solid ${tc.border}`, cursor: "pointer" }}
      />
      <Slider
        value={[vminPct, vmaxPct]}
        disableSwap
        onChange={(_, value) => {
          const [lo, hi] = value as number[];
          onRangeChange(Math.min(lo, hi - 1), Math.max(hi, lo + 1));
        }}
        min={0} max={100} size="small"
        valueLabelDisplay="auto"
        valueLabelFormat={(pct) => {
          const val = dataMin + (pct / 100) * (dataMax - dataMin);
          return formatStat(val);
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
          {formatStat(dataMin + (vminPct / 100) * (dataMax - dataMin))}
        </Typography>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>
          {formatStat(dataMin + (vmaxPct / 100) * (dataMax - dataMin))}
        </Typography>
      </Box>
    </Box>
  );
}

/* ================================================================
   ImagePanel: canvas with zoom/pan, cursor readout and scale bar overlay
   ================================================================ */

/** A panel's scalar values: the array itself, or a getter for the frame its canvas shows now
 *  (frames repaint without re-rendering the panel). */
type RawPanelData = Float32Array | null | undefined | (() => Float32Array | null | undefined);
const readRawPanelData = (raw: RawPanelData) => (typeof raw === "function" ? raw() : raw);

function ImagePanel({
  offscreen, label, size, tc, zoom, onZoomChange,
  showResize, onResizeStart,
  pixelSize, realSpacePixelSize, imageWidth, isFFT,
  rawData, smooth, inset, cropRegion, cropSelecting, onCropChange,
}: {
  offscreen: HTMLCanvasElement | null;
  offscreenVersion?: number;
  label: string;
  size: number;
  tc: ThemeColors;
  zoom: ZoomState;
  onZoomChange: (z: ZoomState) => void;
  showResize?: boolean;
  onResizeStart?: (e: React.MouseEvent) => void;
  pixelSize?: number;
  realSpacePixelSize?: number;
  smooth?: boolean;
  imageWidth?: number;
  isFFT?: boolean;
  rawData?: RawPanelData;
  inset?: React.ReactNode;
  cropRegion?: [number, number, number, number] | null;
  cropSelecting?: boolean;
  onCropChange?: (startRow: number, startCol: number, endRow: number, endCol: number) => void;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const zoomRef = React.useRef(zoom); zoomRef.current = zoom;
  const dragState = React.useRef({ on: false, x: 0, y: 0, panX0: 0, panY0: 0 });
  const cropDragRef = React.useRef<{ row: number; col: number } | null>(null);
  const cropRafRef = React.useRef<number | null>(null);
  const pendingCropRef = React.useRef<{ startRow: number; startCol: number; endRow: number; endCol: number } | null>(null);
  const [cursor, setCursor] = React.useState<{ row: number; col: number; val: number } | null>(null);
  // FFT d-spacing measurement (only used when isFFT)
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [fftClickInfo, setFftClickInfo] = React.useState<FFTPeakMark | null>(null);

  useCanvasFrame(offscreen, React.useCallback(() => {
    drawCanvas(canvasRef.current, offscreen, size, zoom, tc.bgAlt, tc.textMuted, !!smooth);
  }, [offscreen, size, zoom, tc, smooth]));

  // Scale bar overlay
  React.useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay || !pixelSize || pixelSize <= 0 || !imageWidth || imageWidth <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    overlay.width = size * dpr;
    overlay.height = size * dpr;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (isFFT) {
      drawFFTScaleBarHiDPI(overlay, dpr, zoom.zoom, pixelSize, imageWidth, "Å⁻¹");
      // d-spacing marker, Show3D style: open circle + gap-cross, white with
      // shadow so it reads on any background. Label flips Å → nm at ≥10 Å.
      if (fftClickInfo && offscreen) {
        const fftW = offscreen.width, fftH = offscreen.height;
        const baseFit = Math.min(size / fftW, size / fftH);
        const scale = baseFit * zoom.zoom;
        const xOff = (size - fftW * scale) / 2 + zoom.panX;
        const yOff = (size - fftH * scale) / 2 + zoom.panY;
        const cssX = xOff + fftClickInfo.col * scale;
        const cssY = yOff + fftClickInfo.row * scale;
        ctx.save();
        ctx.scale(dpr, dpr);
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.shadowColor = "rgba(0,0,0,0.6)";
        ctx.shadowBlur = 2;
        ctx.lineWidth = 1.5;
        strokePeakMarker(ctx, cssX, cssY, 8, 4);
        if (fftClickInfo.dSpacing != null) {
          const d = fftClickInfo.dSpacing;
          const label = d >= 10 ? `d = ${(d / 10).toFixed(2)} nm` : `d = ${d.toFixed(2)} Å`;
          ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
          ctx.fillStyle = "white";
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(label, cssX + 10, cssY - 4);
        }
        ctx.restore();
      }
    } else {
      drawScaleBarHiDPI(overlay, dpr, zoom.zoom, pixelSize, "Å", imageWidth);
    }
  }, [zoom, size, pixelSize, imageWidth, isFFT, fftClickInfo, offscreen]);

  // Native wheel listener with { passive: false }: React's synthetic onWheel
  // is passive by default since React 17, so e.preventDefault() is silently
  // ignored and the Jupyter cell scrolls along with the zoom gesture.
  const sizeRef = React.useRef(size); sizeRef.current = size;
  const offscreenRef = React.useRef(offscreen); offscreenRef.current = offscreen;
  const onZoomChangeRef = React.useRef(onZoomChange); onZoomChangeRef.current = onZoomChange;
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      if (!offscreenRef.current) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left, mouseY = e.clientY - rect.top;
      const prev = zoomRef.current;
      const factor = e.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, prev.zoom * factor));
      const ratio = newZoom / prev.zoom;
      const centerX = sizeRef.current / 2 + prev.panX;
      const centerY = sizeRef.current / 2 + prev.panY;
      onZoomChangeRef.current({
        zoom: newZoom,
        panX: prev.panX - (mouseX - centerX) * (ratio - 1),
        panY: prev.panY - (mouseY - centerY) * (ratio - 1),
      });
    };
    canvas.addEventListener("wheel", handler, { passive: false });
    return () => canvas.removeEventListener("wheel", handler);
  }, []);

  const imageCoordinate = React.useCallback((clientX: number, clientY: number) => {
    if (!canvasRef.current || !offscreen) return null;
    const point = clientToImage(canvasRef.current, clientX, clientY, offscreen, size, zoomRef.current);
    const col = Math.floor(point.col);
    const row = Math.floor(point.row);
    if (row < 0 || row >= offscreen.height || col < 0 || col >= offscreen.width) return null;
    return { row, col };
  }, [offscreen, size]);
  const publishCrop = React.useCallback((startRow: number, startCol: number, endRow: number, endCol: number) => {
    pendingCropRef.current = { startRow, startCol, endRow, endCol };
    if (cropRafRef.current != null) return;
    cropRafRef.current = requestAnimationFrame(() => {
      cropRafRef.current = null;
      const crop = pendingCropRef.current;
      if (crop) onCropChange?.(crop.startRow, crop.startCol, crop.endRow, crop.endCol);
    });
  }, [onCropChange]);
  React.useEffect(() => () => {
    if (cropRafRef.current != null) cancelAnimationFrame(cropRafRef.current);
  }, []);

  const onDown = React.useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    if (cropSelecting && !isFFT) {
      const point = imageCoordinate(e.clientX, e.clientY);
      if (point) {
        cropDragRef.current = point;
        publishCrop(point.row, point.col, point.row, point.col);
      }
      return;
    }
    dragState.current = {
      on: true,
      x: e.clientX, y: e.clientY,
      panX0: zoomRef.current.panX, panY0: zoomRef.current.panY,
    };
    if (isFFT) fftClickStartRef.current = { x: e.clientX, y: e.clientY };
  }, [cropSelecting, imageCoordinate, isFFT, publishCrop]);
  // Mouse up handler on the FFT canvas: detect short click (no drag) and
  // compute d-spacing at the clicked peak, snapped to the nearest local-max.
  const onFFTUp = React.useCallback(async (e: React.MouseEvent) => {
    const values = readRawPanelData(rawData);
    if (!isFFT || !fftClickStartRef.current || !offscreen || !values) return;
    const dx = e.clientX - fftClickStartRef.current.x;
    const dy = e.clientY - fftClickStartRef.current.y;
    fftClickStartRef.current = null;
    if (Math.sqrt(dx * dx + dy * dy) >= 3) return;
    const point = clientToImage(canvasRef.current!, e.clientX, e.clientY, offscreen, size, zoomRef.current);
    const fftW = offscreen.width, fftH = offscreen.height;
    if (point.col < 0 || point.col >= fftW || point.row < 0 || point.row >= fftH) return;
    // Snap to the nearest local maximum, as Show3D's findFFTPeak does.
    let snapped: { row: number; col: number };
    try {
      snapped = await findFFTPeakBrowser(values, fftW, fftH, point.col, point.row, FFT_SNAP_RADIUS);
    } catch (error) {
      console.error("[ShowPtycho] WebGPU FFT peak refinement failed", error);
      return;
    }
    setFftClickInfo(measureFFTPeak(snapped.row, snapped.col, fftW, fftH, realSpacePixelSize));
  }, [isFFT, offscreen, rawData, size, realSpacePixelSize]);
  const onLeaveFFT = React.useCallback(() => { fftClickStartRef.current = null; }, []);

  // Cursor readout: map mouse position through zoom/pan to raw-data indices
  const onMove = React.useCallback((e: React.MouseEvent) => {
    const values = readRawPanelData(rawData);
    if (!canvasRef.current || !offscreen || !values) return;
    const point = clientToImage(canvasRef.current, e.clientX, e.clientY, offscreen, size, zoomRef.current);
    const col = Math.floor(point.col);
    const row = Math.floor(point.row);
    if (row < 0 || row >= offscreen.height || col < 0 || col >= offscreen.width) {
      setCursor(null);
      return;
    }
    setCursor({ row, col, val: values[row * offscreen.width + col] });
    if (cropSelecting && cropDragRef.current && !isFFT) {
      publishCrop(cropDragRef.current.row, cropDragRef.current.col, row, col);
    }
  }, [cropSelecting, isFFT, offscreen, publishCrop, size, rawData]);
  const onUp = React.useCallback((e: React.MouseEvent) => {
    if (cropSelecting && cropDragRef.current && !isFFT) {
      const point = imageCoordinate(e.clientX, e.clientY);
      if (point) publishCrop(cropDragRef.current.row, cropDragRef.current.col, point.row, point.col);
      cropDragRef.current = null;
      return;
    }
    if (isFFT) onFFTUp(e);
  }, [cropSelecting, imageCoordinate, isFFT, onFFTUp, publishCrop]);
  const onLeave = React.useCallback(() => setCursor(null), []);

  React.useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (!dragState.current.on) return;
      onZoomChange({
        ...zoomRef.current,
        panX: dragState.current.panX0 + e.clientX - dragState.current.x,
        panY: dragState.current.panY0 + e.clientY - dragState.current.y,
      });
    };
    const handleUp = () => { dragState.current.on = false; };
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => { window.removeEventListener("mousemove", handleMove); window.removeEventListener("mouseup", handleUp); };
  }, [onZoomChange]);

  const cropOverlay = React.useMemo(() => {
    if (!cropRegion || !offscreen || isFFT) return null;
    const [rowStart, rowEnd, colStart, colEnd] = cropRegion;
    const baseFit = Math.min(size / offscreen.width, size / offscreen.height);
    const scale = baseFit * zoom.zoom;
    return {
      left: (size - offscreen.width * scale) / 2 + zoom.panX + colStart * scale,
      top: (size - offscreen.height * scale) / 2 + zoom.panY + rowStart * scale,
      width: (colEnd - colStart) * scale,
      height: (rowEnd - rowStart) * scale,
    };
  }, [cropRegion, isFFT, offscreen, size, zoom]);

  return (
    <Box sx={{ width: size, flexShrink: 0, display: "flex", flexDirection: "column" }}>
      <Box sx={{ ...container.imageBox, bgcolor: tc.bgAlt, border: `1px solid ${tc.border}`, width: size, height: size }}>
        <canvas
          data-quantem-scientific-output={label}
          data-source-width={offscreen?.width}
          data-source-height={offscreen?.height}
          aria-label={label}
          ref={canvasRef}
          onMouseDown={onDown}
          onMouseMove={onMove}
          onMouseUp={onUp}
          onMouseLeave={() => { onLeave(); if (isFFT) onLeaveFFT(); }}
          onDoubleClick={() => { onZoomChange(ZOOM_RESET); if (isFFT) setFftClickInfo(null); }}
          style={{ width: size, height: size, cursor: isFFT || cropSelecting ? "crosshair" : "grab", display: "block", imageRendering: smooth ? "auto" : "pixelated" }}
        />
        <Typography sx={{ position: "absolute", top: 4, left: 5, px: 0.5,
          fontSize: 10, fontFamily: "monospace", color: "#fff",
          bgcolor: "rgba(0,0,0,0.5)", pointerEvents: "none" }}>{label}</Typography>
        {cropOverlay && (
          <Box sx={{
            position: "absolute", pointerEvents: "none",
            left: cropOverlay.left, top: cropOverlay.top,
            width: cropOverlay.width, height: cropOverlay.height,
            border: "3px solid #54e36a",
            bgcolor: "rgba(84, 227, 106, 0.07)",
            boxShadow: "0 0 0 1px rgba(2, 24, 8, 0.88), 0 0 0 9999px rgba(2, 8, 5, 0.46)",
            boxSizing: "border-box",
            zIndex: 2,
          }} />
        )}
        {(cursor || zoom.zoom !== 1) && (
          <Box sx={{
            position: "absolute", top: 4, right: 4, px: 0.5, py: 0.15,
            bgcolor: "rgba(0,0,0,0.55)", color: "#fff",
            fontFamily: "monospace", fontSize: 10, pointerEvents: "none",
          }}>
            {cursor ? `(${cursor.row},${cursor.col}) ${formatStat(cursor.val)}` : ""}
            {zoom.zoom !== 1 ? `${cursor ? " · " : ""}${zoom.zoom.toFixed(1)}x` : ""}
          </Box>
        )}
        {/* Scale bar overlay */}
        {pixelSize != null && pixelSize > 0 && (
          <canvas
            ref={overlayRef}
            style={{
              position: "absolute", top: 0, left: 0, width: size, height: size,
              pointerEvents: "none",
            }}
          />
        )}
        {inset}
        {showResize && onResizeStart && (
          <Box onMouseDown={onResizeStart} sx={{
            position: "absolute", bottom: 0, right: 0, width: 16, height: 16,
            cursor: "nwse-resize", opacity: 0.6,
            background: `linear-gradient(135deg, transparent 50%, ${tc.accent} 50%)`,
            "&:hover": { opacity: 1 },
          }} />
        )}
      </Box>
    </Box>
  );
}

function FFTInset({
  offscreen, rawData, size, panelSize, box, onBoxChange, tc, smooth, pixelSize,
}: {
  offscreen: HTMLCanvasElement | null;
  offscreenVersion?: number;
  rawData?: RawPanelData;
  size: number;
  panelSize: number;
  box: FFTInsetBox;
  onBoxChange: React.Dispatch<React.SetStateAction<FFTInsetBox>>;
  tc: ThemeColors;
  smooth?: boolean;
  pixelSize?: number;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const [fftClickInfo, setFftClickInfo] = React.useState<FFTPeakMark | null>(null);

  useCanvasFrame(offscreen, React.useCallback(() => {
    drawCanvas(canvasRef.current, offscreen, size, ZOOM_RESET, "#000", "#fff", !!smooth);
  }, [offscreen, size, smooth]));

  React.useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const dpr = window.devicePixelRatio || 1;
    overlay.width = size * dpr;
    overlay.height = size * dpr;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!fftClickInfo || !offscreen) return;
    const scale = size / Math.max(offscreen.width, offscreen.height);
    const cssX = (size - offscreen.width * scale) / 2 + fftClickInfo.col * scale;
    const cssY = (size - offscreen.height * scale) / 2 + fftClickInfo.row * scale;
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.shadowColor = "rgba(0,0,0,0.7)";
    ctx.shadowBlur = 2;
    ctx.lineWidth = 1.4;
    strokePeakMarker(ctx, cssX, cssY, 7, 3.5);
    if (fftClickInfo.dSpacing != null) {
      const d = fftClickInfo.dSpacing;
      const label = d >= 10 ? `${(d / 10).toFixed(2)} nm` : `${d.toFixed(2)} Å`;
      ctx.font = "bold 10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = "white";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(label, Math.min(size - 54, cssX + 8), Math.max(14, cssY - 4));
    }
    ctx.restore();
  }, [fftClickInfo, offscreen, size]);

  const measurePeak = React.useCallback(async (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    const values = readRawPanelData(rawData);
    if (!canvas || !offscreen || !values) return;
    // the inset draws unzoomed, so the panel mapping at ZOOM_RESET is the inset's own
    const point = clientToImage(canvas, clientX, clientY, offscreen, size, ZOOM_RESET);
    const fftW = offscreen.width;
    const fftH = offscreen.height;
    if (point.col < 0 || point.col >= fftW || point.row < 0 || point.row >= fftH) return;
    let snapped: { row: number; col: number };
    try {
      snapped = await findFFTPeakBrowser(values, fftW, fftH, point.col, point.row, FFT_SNAP_RADIUS);
    } catch (error) {
      console.error("[ShowPtycho] WebGPU inset peak refinement failed", error);
      return;
    }
    setFftClickInfo(measureFFTPeak(snapped.row, snapped.col, fftW, fftH, pixelSize));
  }, [offscreen, rawData, size, pixelSize]);

  const onMouseDown = React.useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const start = { x: e.clientX, y: e.clientY, box };
    let moved = false;
    const handleMove = (moveEvent: MouseEvent) => {
      const dx = moveEvent.clientX - start.x;
      const dy = moveEvent.clientY - start.y;
      if (Math.sqrt(dx * dx + dy * dy) > 3) moved = true;
      const sizeFrac = size / Math.max(1, panelSize);
      onBoxChange({
        ...start.box,
        x: Math.max(0, Math.min(1 - sizeFrac, start.box.x + dx / Math.max(1, panelSize))),
        y: Math.max(0, Math.min(1 - sizeFrac, start.box.y + dy / Math.max(1, panelSize))),
      });
      moveEvent.preventDefault();
    };
    const handleUp = (upEvent: MouseEvent) => {
      document.removeEventListener("mousemove", handleMove);
      document.removeEventListener("mouseup", handleUp);
      if (!moved) void measurePeak(upEvent.clientX, upEvent.clientY);
    };
    document.addEventListener("mousemove", handleMove);
    document.addEventListener("mouseup", handleUp);
  }, [box, measurePeak, onBoxChange, panelSize, size]);

  const sizeFrac = size / Math.max(1, panelSize);
  const leftPct = Math.max(0, Math.min(1 - sizeFrac, box.x)) * 100;
  const topPct = Math.max(0, Math.min(1 - sizeFrac, box.y)) * 100;

  return (
    <Box
      onMouseDown={onMouseDown}
      onDoubleClick={(e) => { e.stopPropagation(); setFftClickInfo(null); }}
      sx={{
        position: "absolute",
        top: `${topPct}%`,
        left: `${leftPct}%`,
        width: size,
        height: size,
        bgcolor: "rgba(0,0,0,0.82)",
        border: `1px solid ${tc.border}`,
        cursor: "move",
        pointerEvents: "auto",
        userSelect: "none",
      }}
      title="Drag FFT inset. Click a peak to measure d-spacing. Double-click to clear marker."
    >
      <canvas
        ref={canvasRef}
        aria-label="FFT inset"
        style={{ width: size, height: size, display: "block", imageRendering: smooth ? "auto" : "pixelated", pointerEvents: "none" }}
      />
      <canvas
        ref={overlayRef}
        aria-label="FFT inset peak marker"
        style={{ position: "absolute", inset: 0, width: size, height: size, pointerEvents: "none" }}
      />
      <Typography
        sx={{
          position: "absolute",
          top: 2,
          left: 4,
          fontSize: 9,
          fontFamily: "monospace",
          color: "#fff",
          textShadow: "0 1px 2px #000",
          lineHeight: 1,
          pointerEvents: "none",
        }}
      >
        FFT
      </Typography>
      <Box
        sx={{
          position: "absolute",
          top: 0,
          right: 0,
          width: 16,
          height: 16,
          bgcolor: "rgba(0,0,0,0.48)",
          borderLeft: "1px solid rgba(255,255,255,0.22)",
          borderBottom: "1px solid rgba(255,255,255,0.22)",
          color: "#fff",
          fontSize: 10,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          pointerEvents: "none",
        }}
      >
        ↕
      </Box>
    </Box>
  );
}

/* ================================================================
   Higher-order aberration panel (n=2..5)
   ================================================================ */

// Aberration layout.  Magnitude ranges are empirical: they cover typical
// Cs-corrected 300 kV aberration budgets (C3 ~1 μm, C5 ~1 mm).  Users on
// uncorrected scopes will live at the low end of each range.
type HOEntry = {
  name: string;          // Krivanek name (C21, C23, ...)
  hasAngle: boolean;     // false for rotationally-symmetric (m=0) terms
  mag_max: number;       // nm (internal storage; engine convention)
  step_nm: number;       // slider step in nm, tuned so snapping lands on "nice" display values
  unit_display: string;  // native unit shown in UI (nm, μm, or mm)
  display_scale: number; // factor to divide nm by for display value (1, 1000, 1000000)
  tooltip: string;
};

// Slider ranges and units copied from bobleesj's Ronchigram reference
// (github.com/bobleesj/electron-microscopy-website, Ronchigram.jsx).  The
// ranges are validated against real 300 kV STEM residuals.  Internal
// values are stored in nm (engine convention); `display_scale` converts
// to the native display unit so the slider label reads in nm/μm/mm.
const HO_BY_ORDER: Record<number, HOEntry[]> = {
  2: [
    { name: "C21", hasAngle: true,  mag_max: 100,          step_nm: 2,       unit_display: "nm", display_scale: 1,
      tooltip: "C21: axial coma.  Comet-like tails on features.  Range ±100 nm, step 2 nm." },
    { name: "C23", hasAngle: true,  mag_max: 100,          step_nm: 2,       unit_display: "nm", display_scale: 1,
      tooltip: "C23: 3-fold astigmatism.  3-lobed probe shape.  Range ±100 nm, step 2 nm." },
  ],
  3: [
    { name: "C30", hasAngle: false, mag_max: 100000,       step_nm: 1000,    unit_display: "μm", display_scale: 1000,
      tooltip: "C30: spherical aberration (Cs).  Primary resolution limiter.  Range ±100 μm, step 1 μm." },
    { name: "C32", hasAngle: true,  mag_max: 100000,       step_nm: 1000,    unit_display: "μm", display_scale: 1000,
      tooltip: "C32: star aberration.  2-fold star pattern.  Range ±100 μm, step 1 μm." },
    { name: "C34", hasAngle: true,  mag_max: 1000,         step_nm: 10,      unit_display: "μm", display_scale: 1000,
      tooltip: "C34: 4-fold astigmatism.  Range ±1 μm, step 0.01 μm (10 nm)." },
  ],
  4: [
    { name: "C41", hasAngle: true,  mag_max: 100000,       step_nm: 1000,    unit_display: "μm", display_scale: 1000,
      tooltip: "C41: 4th order coma.  Subtle directional blur.  Range ±100 μm, step 1 μm." },
    { name: "C43", hasAngle: true,  mag_max: 100000,       step_nm: 1000,    unit_display: "μm", display_scale: 1000,
      tooltip: "C43: 3-lobe aberration (4th order).  Range ±100 μm, step 1 μm." },
    { name: "C45", hasAngle: true,  mag_max: 100000,       step_nm: 1000,    unit_display: "μm", display_scale: 1000,
      tooltip: "C45: 5-fold astigmatism (4th order).  Range ±100 μm, step 1 μm." },
  ],
  5: [
    { name: "C50", hasAngle: false, mag_max: 100000000,    step_nm: 100000,  unit_display: "mm", display_scale: 1000000,
      tooltip: "C50: 5th order spherical.  Dominant on Cs-corrected scopes.  Range ±100 mm, step 0.1 mm." },
    { name: "C52", hasAngle: true,  mag_max: 100000000,    step_nm: 1000000, unit_display: "mm", display_scale: 1000000,
      tooltip: "C52: 5th order star.  Range ±100 mm, step 1 mm." },
    { name: "C54", hasAngle: true,  mag_max: 100000000,    step_nm: 1000000, unit_display: "mm", display_scale: 1000000,
      tooltip: "C54: rosette aberration; 4-fold rosette pattern.  Range ±100 mm, step 1 mm." },
    { name: "C56", hasAngle: true,  mag_max: 100000000,    step_nm: 1000000, unit_display: "mm", display_scale: 1000000,
      tooltip: "C56: 6-fold astigmatism (5th order).  Range ±100 mm, step 1 mm." },
  ],
};

function formatHOValue(valueNm: number, maxNm: number, displayScale: number): string {
  // Convert nm → native display unit (μm divides by 1000, mm by 1e6)
  const value = valueNm / displayScale;
  const max = maxNm / displayScale;
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  // Choose precision based on range in display units.  Small ranges
  // (e.g. C34, max 1 μm, step 0.01 μm) need extra decimals so one slider
  // step stays visible instead of rounding away.
  if (max >= 100) return value.toFixed(0);
  if (max >= 10) return value.toFixed(1);
  if (max >= 1) return value.toFixed(2);
  if (max >= 0.1) return value.toFixed(3);
  return value.toFixed(4);
}

/* Sample panel: thick-sample SSB.  A crystal tilted by theta changes how strongly each
   bright-field pixel carries each spatial frequency; the kernel averages every pixel's correction
   over a depth spread with the columns leaning by the tilt.  Depth spread 0 = standard SSB.
   Joint fit solves C10, C12, phi12, tilt and depth spread together in the SSB session. */
const SAMPLE_ZERO: SampleValues = { tiltRowMrad: 0, tiltColMrad: 0, depthSpreadNm: 0 };

function SamplePanel({
  tc, open, onToggle, values, setValues, onCommit, onFit, onReset, fitBusy, fitStatus, fitAvailable = true, disabled = false,
}: {
  tc: ThemeColors;
  open: boolean;
  onToggle: () => void;
  values: SampleValues;
  setValues: React.Dispatch<React.SetStateAction<SampleValues>>;
  onCommit?: () => void;
  onFit: () => void;
  onReset: () => void;
  fitBusy: boolean;
  fitStatus: string;
  fitAvailable?: boolean;
  disabled?: boolean;
}) {
  const active = values.depthSpreadNm > 0;
  const sliders: { key: keyof SampleValues; label: string; unit: string; min: number; max: number; step: number; tip: string }[] = [
    { key: "tiltRowMrad", label: "tilt row", unit: "mrad", min: -25, max: 25, step: 0.1,
      tip: "Sample tilt along the scan rows.  Only acts when the depth spread > 0." },
    { key: "tiltColMrad", label: "tilt col", unit: "mrad", min: -25, max: 25, step: 0.1,
      tip: "Sample tilt along the scan columns.  Only acts when the depth spread > 0." },
    { key: "depthSpreadNm", label: "depth spread", unit: "nm", min: 0, max: 60, step: 0.5,
      tip: "Depth over which the correction is averaged.  0 = standard SSB.  A model depth spread, not a measured sample thickness." },
  ];
  return (
    <Box sx={{
      display: "inline-flex", flexDirection: "column", width: "fit-content", maxWidth: "100%",
      border: `1px solid ${tc.border}`, bgcolor: tc.controlBg,
    }}>
      <Box
        sx={{
          display: "inline-flex", alignItems: "center", gap: `${SPACING.MD}px`, px: 1, py: 0.5,
          cursor: "pointer", userSelect: "none", borderBottom: open ? `1px solid ${tc.border}` : "none",
        }}
        onClick={onToggle}
      >
        <Typography sx={{ ...typography.label, color: tc.accent }}>
          {open ? "▾" : "▸"} Sample tilt (thick SSB)
        </Typography>
        {active && (
          <Box sx={{ fontSize: 10, px: 0.8, py: 0.1, borderRadius: 0, bgcolor: STATUS_GOOD, color: "#fff", fontFamily: "monospace" }}>
            ({values.tiltRowMrad.toFixed(1)}, {values.tiltColMrad.toFixed(1)}) mrad · {values.depthSpreadNm.toFixed(1)} nm
          </Box>
        )}
        <Tooltip title="Restore the jointly fitted optics, sample tilt and depth." placement="top" arrow>
          <span>
            <IconButton
              size="small" aria-label="Restore joint fit" disabled={fitBusy}
              onClick={(e) => { e.stopPropagation(); onReset(); }}
              sx={compactBareIconButton(tc)}
            >
              <RestartAltIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </span>
        </Tooltip>
      </Box>
      {open && (
        <Box sx={{ px: 1, py: 0.5, minWidth: 420 }}>
          {sliders.map(slider => (
            <Box key={slider.key} sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, py: 0.25 }}>
              <Tooltip title={slider.tip} placement="right" arrow>
                <Typography sx={{ ...typography.value, color: active ? tc.accent : tc.textMuted, minWidth: 64 }}>{slider.label}</Typography>
              </Tooltip>
              <Box sx={{ flex: 1, maxWidth: 200, minWidth: 120 }}>
                <Slider
                  aria-label={slider.label} disabled={disabled}
                  value={values[slider.key]} min={slider.min} max={slider.max} step={slider.step}
                  onChange={(_, value) => setValues(prev => ({ ...prev, [slider.key]: value as number }))}
                  onChangeCommitted={() => onCommit?.()}
                  size="small" sx={{ py: 0.5 }}
                />
              </Box>
              <Typography sx={{ ...typography.value, color: active ? tc.accent : tc.textMuted, minWidth: 72, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                {values[slider.key].toFixed(1)} {slider.unit}
              </Typography>
            </Box>
          ))}
          {fitAvailable ? (
            <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.MD}px`, mt: 0.5 }}>
              <Button size="small" variant="outlined" onClick={onFit} disabled={fitBusy} sx={{ ...compactButton(tc), color: STATUS_GOOD, borderColor: tc.border }}>
                {fitBusy ? "Fitting…" : "Joint fit"}
              </Button>
              <Typography sx={{ ...typography.labelSmall, color: /failed|ignored/.test(fitStatus) ? STATUS_BAD : tc.textMuted, maxWidth: 520 }}>
                {fitStatus || "The default fit, model-based and tilt-corrected: defocus, astigmatism, tilt and depth together. C10 is the mid-depth defocus."}
              </Typography>
            </Box>
          ) : (
            <Typography sx={{ ...typography.labelSmall, color: tc.textMuted, mt: 0.5, maxWidth: 520 }}>
              Sliders run in the browser.  Fitting tilt needs the Python session (ShowPtycho in a notebook).
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
}

function HigherOrderPanel({
  tc, open, onToggle, activeCount, values, setValues, onCommit, disabled = false,
}: {
  tc: ThemeColors;
  open: boolean;
  onToggle: () => void;
  activeCount: number;
  onCommit: () => void;
  disabled?: boolean;
  values: Record<string, number>;
  setValues: React.Dispatch<React.SetStateAction<Record<string, number>>>;
}) {
  const updateMag = React.useCallback((name: string, value: number) => {
    setValues(prev => ({ ...prev, [name]: value }));
  }, [setValues]);
  const updateAngle = React.useCallback((name: string, deg: number) => {
    setValues(prev => ({ ...prev, [higherOrderAzimuthKey(name)]: deg }));
  }, [setValues]);
  const resetAll = React.useCallback(() => { onCommit(); setValues({}); }, [setValues, onCommit]);
  const resetMag = React.useCallback((name: string) => {
    onCommit();
    setValues(prev => {
      const next = { ...prev };
      delete next[higherOrderMagnitudeKey(name)];
      return next;
    });
  }, [setValues]);
  const resetAngle = React.useCallback((name: string) => {
    onCommit();
    setValues(prev => {
      const next = { ...prev };
      delete next[higherOrderAzimuthKey(name)];
      return next;
    });
  }, [setValues]);
  const orderColumns = React.useMemo(() => [[2, 3], [4, 5]], []);
  const renderEntry = React.useCallback((entry: HOEntry) => {
    const magKey = higherOrderMagnitudeKey(entry.name);
    const mag = values[magKey] ?? 0;
    const angle = entry.hasAngle ? (values[higherOrderAzimuthKey(entry.name)] ?? 0) : 0;
    const isActive = Math.abs(mag) > 0;
    const hasAngleOffset = entry.hasAngle && Math.abs(angle) > 0;
    return (
      <Box
        key={entry.name}
        sx={{
          display: "flex", alignItems: "center", gap: `${SPACING.SM}px`,
          py: 0.25,
        }}
      >
        <Tooltip title={entry.tooltip} placement="right" arrow>
          <Typography
            sx={{
              ...typography.value,
              color: isActive ? tc.accent : tc.textMuted,
              minWidth: 34,
            }}
          >
            {entry.name}
          </Typography>
        </Tooltip>
        {/* Cap slider width so rows stay compact in the two-column panel. */}
        <Box sx={{ flex: 1, maxWidth: 140, minWidth: 80 }}>
          <Slider
            disabled={disabled}
            value={mag} min={-entry.mag_max} max={entry.mag_max} step={entry.step_nm}
            aria-label={`${entry.name} magnitude`}
            onChange={(_, value) => updateMag(magKey, value as number)}
            onChangeCommitted={onCommit}
            size="small" sx={{ py: 0.5 }}
          />
        </Box>
        <Typography sx={{
          ...typography.value, color: isActive ? tc.accent : tc.textMuted,
          minWidth: 58, textAlign: "right",
        }}>
          {formatHOValue(mag, entry.mag_max, entry.display_scale)} {entry.unit_display}
        </Typography>
        <IconButton
          size="small"
          aria-label={`Reset ${entry.name} magnitude`}
          disabled={!isActive}
          onClick={() => resetMag(entry.name)}
          sx={compactBareIconButton(tc)}
        >
          <RestartAltIcon sx={{ fontSize: 13 }} />
        </IconButton>
        {entry.hasAngle && (
          <>
            <Box sx={{ flex: 1, maxWidth: 110, minWidth: 60 }}>
              <Slider
                value={angle} min={-180} max={180} step={1}
                aria-label={`${entry.name} angle`}
                onChange={(_, value) => updateAngle(entry.name, value as number)}
                onChangeCommitted={onCommit}
                size="small" sx={{ py: 0.5 }}
                disabled={disabled || !isActive}
              />
            </Box>
            <Typography sx={{
              ...typography.value,
              color: hasAngleOffset ? tc.accent : tc.textMuted,
              minWidth: 38, textAlign: "right",
            }}>
              {angle.toFixed(0)}°
            </Typography>
            <IconButton
              size="small"
              aria-label={`Reset ${entry.name} angle`}
              disabled={!hasAngleOffset}
              onClick={() => resetAngle(entry.name)}
              sx={compactBareIconButton(tc)}
            >
              <RestartAltIcon sx={{ fontSize: 13 }} />
            </IconButton>
          </>
        )}
        {!entry.hasAngle && (
          // Filler to keep column alignment consistent across rows.
          <Box sx={{ flex: 1, maxWidth: 176, minWidth: 0 }} />
        )}
      </Box>
    );
  }, [disabled, resetAngle, resetMag, tc, updateAngle, updateMag, values]);

  return (
    <Box sx={{
      display: "inline-flex",
      flexDirection: "column",
      width: open ? "max-content" : "fit-content",
      maxWidth: open ? "none" : "100%",
      border: `1px solid ${tc.border}`,
      bgcolor: tc.controlBg,
    }}>
      {/* Header row: click to expand/collapse, shows badge when active. */}
      <Box
        sx={{
          display: "inline-flex", alignItems: "center", gap: `${SPACING.MD}px`,
          px: 1, py: 0.5, cursor: "pointer", userSelect: "none",
          borderBottom: open ? `1px solid ${tc.border}` : "none",
        }}
        onClick={onToggle}
      >
        <Typography sx={{ ...typography.label, color: tc.accent }}>
          {open ? "▾" : "▸"} Higher-order (n=2..5)
        </Typography>
        {activeCount > 0 && (
          <Tooltip title={`${activeCount} higher-order magnitude(s) non-zero.  The reconstruction and its loss use the full 14-coefficient kernel; the automatic fit searches only C10, C12 and phi12.`} placement="top" arrow>
            <Box sx={{
              fontSize: 10, px: 0.8, py: 0.1, borderRadius: 0,
              bgcolor: STATUS_GOOD, color: "#fff", fontFamily: "monospace", cursor: "help",
            }}>
              {activeCount} active
            </Box>
          </Tooltip>
        )}
        <Tooltip title="Reset all higher-order sliders to 0.  Returns reconstruction to the fast C10/C12/phi12 path." placement="top" arrow>
          <span>
            <IconButton
              size="small"
              aria-label="Reset all higher-order aberrations"
              disabled={activeCount === 0}
              onClick={(e) => { e.stopPropagation(); resetAll(); }}
              sx={compactBareIconButton(tc)}
            >
              <RestartAltIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </span>
        </Tooltip>
      </Box>

      {open && (
        <Box sx={{ px: 1, py: 0.5, overflowX: "auto", maxWidth: "calc(100vw - 32px)" }}>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "repeat(2, max-content)",
              columnGap: `${SPACING.LG}px`,
              rowGap: `${SPACING.SM}px`,
              alignItems: "start",
              width: "fit-content",
            }}
          >
            {orderColumns.map((orders) => (
              <Box key={orders.join("-")} sx={{ minWidth: 420 }}>
                {orders.map((order, idx) => (
                  <Box key={order} sx={{ mb: 0.5 }}>
                    <Typography sx={{
                      ...typography.labelSmall,
                      color: tc.textMuted, fontFamily: "monospace",
                      mt: idx === 0 ? 0 : 0.5, mb: 0.25,
                    }}>
                      n = {order}
                    </Typography>
                    {HO_BY_ORDER[order].map(renderEntry)}
                  </Box>
                ))}
              </Box>
            ))}
          </Box>
          <Typography sx={{
            ...typography.labelSmall, color: tc.textMuted, mt: 0.5, fontFamily: "monospace",
          }}>
            Use reset icons to zero individual magnitudes or angles.  Magnitudes in nm, angles in deg.
          </Typography>
        </Box>
      )}
    </Box>
  );
}

/* ================================================================
   Main component
   ================================================================ */

function Explore() {
  const model = useModel();
  const fftMagnitudeDisplayRef = React.useRef<FFTMagnitudeDisplay | null>(null);
  React.useEffect(() => {
    const display = new FFTMagnitudeDisplay();
    fftMagnitudeDisplayRef.current = display;
    return () => { fftMagnitudeDisplayRef.current = null; display.dispose(); };
  }, []);
  const [localTiltJson] = useModelState<string>("local_tilt_json");
  const [localTiltEnabled,setLocalTiltEnabled] = useModelState<boolean>("local_tilt_enabled");
  const [maxColumns,setMaxColumns] = useModelState<number>("local_tilt_max_columns");
  const [activeRegionId] = useModelState<string>("local_tilt_active_region");
  const [localSelectionJson] = useModelState<string>("local_tilt_selection_json");
  const [localReplyJson] = useModelState<string>("local_tilt_reply_json");
  const [localTiltStale] = useModelState<boolean>("local_tilt_stale");
  const regionalMode = localTiltEnabled && !!activeRegionId && !!model.get('ssb_session');
  const localConfig = React.useMemo(() => JSON.parse(localTiltJson || '{}'), [localTiltJson]);
  const activeFit = regionalMode ? localConfig.fits?.find((fit:any) => fit.id === activeRegionId) : null;
  const selectedRegions = JSON.parse(localSelectionJson || 'null') || localConfig.regions || [];
  const activeBounds = selectedRegions.find((region:any) => region.id === activeRegionId)?.bounds;
  const regionalReady = !localTiltStale && !!activeFit && JSON.stringify(activeFit.bounds) === JSON.stringify(activeBounds);
  const regionFitBusy = JSON.parse(localReplyJson || '{}').status === 'running';
  // <fieldset disabled> stops native inputs only; MUI sliders take this explicitly.
  const controlsLocked = regionalMode && (!regionalReady || regionFitBusy);
  const regionContextRef = React.useRef<any>(null);
  regionContextRef.current = regionalMode ? {id:activeRegionId,bounds:activeBounds,ready:regionalReady} : null;
  const [localImagesJson] = useModelState<string>("local_tilt_images_json");
  const { colors: tc } = useTheme();

  /* --- Model state (traitlets) --- */
  const [c10Min] = useModelState<number>("C10_min_nm");
  const [c10Max] = useModelState<number>("C10_max_nm");
  const [rotationMin] = useModelState<number>("scan_detector_rotation_min_deg");
  const [rotationMax] = useModelState<number>("scan_detector_rotation_max_deg");
  const [rotationDeg, setRotationDeg] = useModelState<number>("scan_detector_rotation_deg");
  const [displayNegatePhase, setDisplayNegatePhase] = useModelState<boolean>("display_negate_phase");

  const [overviewAutoC10] = useModelState<number>("fitted_C10_nm");
  const [overviewAutoC12] = useModelState<number>("fitted_C12_nm");
  const [overviewAutoPhi12] = useModelState<number>("fitted_phi12_deg");
  const [fittedSampleJson] = useModelState<string>("fitted_sample_json");
  const regionalReference = activeFit?.fitted_parameters || activeFit;
  const autoC10 = regionalReference?.aberrations?.C10_nm ?? overviewAutoC10;
  const autoC12 = regionalReference?.aberrations?.C12_nm ?? overviewAutoC12;
  const autoPhi12 = regionalReference?.aberrations?.phi12_deg ?? overviewAutoPhi12;
  const [autoRotation] = useModelState<number>("fitted_scan_detector_rotation_deg");
  const [autoLoss] = useModelState<number>("fitted_loss");
  // an array carries the pixel size of its image; a session the scan step, which the output sampling divides
  const [objectSamplingA] = useModelState<number[]>("object_sampling_A");
  const [scanSamplingA] = useModelState<number[]>("scan_sampling_A");
  const [sampling] = useModelState<number | "auto">("upsampling_factor");
  const [autoFactor] = useModelState<number>("auto_upsampling_factor");
  const upsamplingFactor = sampling === "auto" ? (autoFactor || 1) : (sampling || 1);
  const upsamplingFactorRef = React.useRef(upsamplingFactor || 1); upsamplingFactorRef.current = upsamplingFactor || 1;

  const [phaseWidth] = useModelState<number>("phase_width");
  const [phaseHeight] = useModelState<number>("phase_height");

  const [, setPinJson] = useModelState<string>("pin_json");
  const [dragBfTrait, setDragBfTrait] = useModelState<number>("num_drag_bright_field_pixels");
  const [totalBf] = useModelState<number>("num_bright_field_pixels");
  const [starsPath] = useModelState<string>("stars_path");
  const [saveTrigger, setSaveTrigger] = useModelState<number>("save_trigger");
  const [notes, setNotes] = useModelState<string>("notes");
  const [calibrationPath] = useModelState<string>("calibration_path");
  const [calibrationSavedAt] = useModelState<string>("calibration_saved_at");
  const [calibrationStatus] = useModelState<string>("calibration_status");
  const [trialsJson] = useModelState<string>("trials_json");
  const [initialFftOn] = useModelState<boolean>("initial_fft_on");
  const [higherOrderJsonModel, setHigherOrderJson] = useModelState<string>("higher_order_aberrations_json");
  const [webgpuPreviewEnabled] = useModelState<boolean>("webgpu_preview_enabled");
  const [webgpuCalJson] = useModelState<string>("webgpu_cal_json");
  const [webgpuH5SourceJson] = useModelState<string>("webgpu_h5_source_json");
  const [webgpuPreviewStatus] = useModelState<string>("webgpu_preview_status");
  const [webgpuStandalone] = useModelState<boolean>("webgpu_standalone");
  const [scanRows] = useModelState<number>("scan_rows");
  const [scanCols] = useModelState<number>("scan_cols");
  const [cropRefitAvailable] = useModelState<boolean>("crop_refit_available");
  const [cropRefitStatus] = useModelState<string>("crop_refit_status");
  const [, setCropRefitRequestJson] = useModelState<string>("crop_refit_request_json");
  const [sampleJsonModel, setSampleJson] = useModelState<string>("sample_json");
  const [tiltCorrection, setTiltCorrection] = useModelState<boolean>("tilt_correction");
  const [sampleAvailable] = useModelState<boolean>("sample_available");
  const [sampleFitRequest, setSampleFitRequest] = useModelState<number>("sample_fit_request");
  const [sampleFitStatus] = useModelState<string>("sample_fit_status");
  const [sampleFitJson] = useModelState<string>("sample_fit_json");
  const [ssbSession] = useModelState<boolean>("ssb_session");
  const [sessionNote] = useModelState<string>("session_note");
  const [amplitudeBytes] = useModelState<DataView>("amplitude_bytes");
  // An array input has no SSB session: only the views, FFT and display controls apply.
  const tuning = ssbSession;
  const objectAmplitude = React.useMemo(
    () => (amplitudeBytes && amplitudeBytes.byteLength > 0 ? extractFloat32(amplitudeBytes) : null),
    [amplitudeBytes],
  );
  const objectAmplitudeRef = React.useRef(objectAmplitude); objectAmplitudeRef.current = objectAmplitude;

  /* --- Local state --- */
  const [c10, setC10] = React.useState(0);
  const [c12, setC12] = React.useState(0);
  const [phi12, setPhi12] = React.useState(0);
  type AberKey = "c10" | "c12" | "phi12" | "rot";
  type MainRanges = Record<AberKey, [number, number]>;
  const [uiRanges, setUiRanges] = React.useState<MainRanges>({
    c10: [-30, 30],
    c12: [-10, 10],
    phi12: [-180, 180],
    rot: [-180, 180],
  });
  const uiRangesSeededRef = React.useRef(false);
  React.useEffect(() => {
    if (uiRangesSeededRef.current) return;
    uiRangesSeededRef.current = true;
    setUiRanges({
      c10: [c10Min, c10Max],
      c12: [Math.min(C12_RANGE[0], Math.floor(autoC12)), Math.max(C12_RANGE[1], Math.ceil(autoC12))],
      phi12: PHI12_RANGE,
      rot: [Math.floor(rotationMin), Math.ceil(rotationMax)],
    });
  }, [c10Min, c10Max, autoC12, rotationMin, rotationMax]);
  React.useEffect(() => {
    // A new joint fit or region can exceed the ranges of an unfitted session.
    // Keep its actual value reachable without discarding the user's wider ranges.
    setUiRanges(previous => {
      let next = previous;
      for (const [key, value] of [["c10", c10], ["c12", c12], ["phi12", phi12]] as const) {
        const [low, high] = previous[key];
        if (Number.isFinite(value) && (value < low || value > high)) {
          next = { ...next, [key]: [Math.min(low, Math.floor(value)), Math.max(high, Math.ceil(value))] };
        }
      }
      return next;
    });
  }, [c10, c12, phi12]);
  const c10UiMin = uiRanges.c10[0];
  const c10UiMax = uiRanges.c10[1];
  const c12UiMin = uiRanges.c12[0];
  const c12UiMax = uiRanges.c12[1];
  const phi12UiMin = uiRanges.phi12[0];
  const phi12UiMax = uiRanges.phi12[1];
  const rotationUiMin = uiRanges.rot[0];
  const rotationUiMax = uiRanges.rot[1];

  const adoptingPanelRef = React.useRef(false);

  /* --- Higher-order aberration state.  Keys: C21_nm/phi21_deg, ..., C56_nm/phi56_deg,
         with C30_nm/C50_nm alone (no azimuth).  All values default to 0. */
  const [higherOrder, setHigherOrder] = React.useState<Record<string, number>>(() => {
    try { return JSON.parse(higherOrderJsonModel || "{}"); } catch { return {}; }
  });
  React.useEffect(() => {
    if (inflightDragRef.current || pendingDragRef.current) return;
    let next: Record<string, number>;
    try { next = JSON.parse(higherOrderJsonModel || "{}"); } catch { return; }
    setHigherOrder(previous => {
      if (JSON.stringify(previous) === JSON.stringify(next)) return previous;
      adoptingPanelRef.current = true;
      return next;
    });
  }, [higherOrderJsonModel]);
  const [hoOpen, setHoOpen] = React.useState(false);
  // Count non-zero magnitudes so the UI can show "active" + user knows when
  // the loss readout stops reflecting the 3-param auto reference.
  const hoActiveCount = React.useMemo(() => {
    let count = 0;
    for (const key of Object.keys(higherOrder)) {
      if (key.endsWith("_nm") && Math.abs(higherOrder[key] || 0) > 0) count += 1;
    }
    return count;
  }, [higherOrder]);
  const hoActive = hoActiveCount > 0;
  const hoActiveRef = React.useRef(false);
  hoActiveRef.current = hoActive;
  const hoActiveCountRef = React.useRef(0);
  hoActiveCountRef.current = hoActiveCount;
  const higherOrderRef = React.useRef<Record<string, number>>({});
  higherOrderRef.current = higherOrder;
  const [loss, setLoss] = React.useState<number | null>(null);
  // Set on every request and reply; an unchanged value must not re-render the widget per frame.
  const [busy, setBusy] = useChangedState(false);
  const webgpuCalLogicalBf = React.useMemo(() => {
    if (!webgpuCalJson) return 0;
    try {
      const cal = JSON.parse(webgpuCalJson);
      const numBf = Number(cal?.num_bright_field_pixels || 0);
      return Number.isFinite(numBf) && numBf > 0 ? Math.round(numBf) : 0;
    } catch {
      return 0;
    }
  }, [webgpuCalJson]);
  // The engine's BF count is a prefix into the full, row-major detector
  // coordinate list. Aperture-active pixels are not a contiguous prefix, so
  // replacing this logical total with the active count silently drops valid BF
  // evidence near the bottom/right of the disk. Memory compaction happens after
  // collectActiveBfIndices() has inspected the complete requested prefix.
  const effectiveTotalBf = webgpuStandalone && webgpuCalLogicalBf > 0
    ? webgpuCalLogicalBf
    : Math.max(0, Math.round(totalBf || 0));
  // Use the full logical BF set for both live and exported reconstruction.
  // Synchronize older saved widgets that still carry an explicit drag subset.
  const dragBfRef = React.useRef(effectiveTotalBf);
  dragBfRef.current = effectiveTotalBf;
  React.useEffect(() => {
    if (effectiveTotalBf > 0 && dragBfTrait !== effectiveTotalBf) {
      setDragBfTrait(effectiveTotalBf);
    }
  }, [dragBfTrait, effectiveTotalBf, setDragBfTrait]);
  const [pinned, setPinned] = React.useState<PinnedEntry[]>([]);
  const [viewPin, setViewPin] = React.useState<number | null>(null);
  const viewPinRef = React.useRef(viewPin); viewPinRef.current = viewPin;
  const [exportStatus, setExportStatus] = React.useState("");
  const [settingsAnchor, setSettingsAnchor] = React.useState<HTMLElement | null>(null);
  const settingsPopover = React.useRef<{ updatePosition: () => void } | null>(null);
  const [dragOverPin, setDragOverPin] = React.useState<number | null>(null);
  const [draggingPin, setDraggingPin] = React.useState<number | null>(null);
  const [newPinId, setNewPinId] = React.useState<number | null>(null);
  const didDragPinRef = React.useRef(false);
  const pinPointerDragRef = React.useRef<{ id: number; x: number; y: number; active: boolean } | null>(null);
  const pinLayoutBeforeRef = React.useRef<Map<number, DOMRect> | null>(null);

  const rawPhaseRef = React.useRef<{ data: Float32Array; w: number; h: number } | null>(null);
  // the horizontal scale bar measures columns: an array's own pixel, or the scan step shared out over the output columns
  const pixelSize = objectSamplingA?.length === 2 ? objectSamplingA[1]
    : scanSamplingA?.length === 2 ? scanSamplingA[1] * (scanCols || phaseWidth || 1) / (rawPhaseRef.current?.w || phaseWidth || scanCols || 1)
    : 0;
  const [phaseZoom, setPhaseZoom] = React.useState(ZOOM_RESET);
  const [ampZoom, setAmpZoom] = React.useState(ZOOM_RESET);
  const [complexZoom, setComplexZoom] = React.useState(ZOOM_RESET);
  const [fftZoom, setFFTZoom] = React.useState(ZOOM_RESET);
  // Every phase clears the panels it does not show; an unchanged null must not re-render the widget.
  const [phaseOff, setPhaseOff] = useChangedState<HTMLCanvasElement | null>(null);
  const [ampOff, setAmpOff] = useChangedState<HTMLCanvasElement | null>(null);
  const [complexOff, setComplexOff] = useChangedState<HTMLCanvasElement | null>(null);
  const [fftOff, setFFTOff] = useChangedState<HTMLCanvasElement | null>(null);
  const [cropSelecting, setCropSelecting] = React.useState(false);
  const [scanCrop, setScanCrop] = React.useState<[number, number, number, number] | null>(null);
  const displayedScanCrop: [number, number, number, number] | null = scanCrop ? [
    scanCrop[0] * (rawPhaseRef.current?.h || phaseHeight) / (scanRows || phaseHeight),
    scanCrop[1] * (rawPhaseRef.current?.h || phaseHeight) / (scanRows || phaseHeight),
    scanCrop[2] * (rawPhaseRef.current?.w || phaseWidth) / (scanCols || phaseWidth),
    scanCrop[3] * (rawPhaseRef.current?.w || phaseWidth) / (scanCols || phaseWidth),
  ] : null;
  const [cropRefitPending, setCropRefitPending] = React.useState(false);
  React.useEffect(() => {
    const rows = Math.round(scanRows || phaseHeight || 0);
    const cols = Math.round(scanCols || phaseWidth || 0);
    setScanCrop(previous => {
      if (!previous) return null;
      const [rowStart, rowEnd, colStart, colEnd] = previous;
      if (rowStart >= 0 && colStart >= 0 && rowEnd <= rows && colEnd <= cols) return previous;
      return null;
    });
  }, [phaseHeight, phaseWidth, scanCols, scanRows]);
  React.useEffect(() => {
    if (/^Refit complete:/.test(cropRefitStatus || "")) {
      setCropRefitPending(false);
      setCropSelecting(false);
      setScanCrop(null);
    } else if (/^Crop refit failed:/.test(cropRefitStatus || "")) {
      setCropRefitPending(false);
    }
  }, [cropRefitStatus]);
  const chooseCropRectangle = React.useCallback((startRow: number, startCol: number, endRow: number, endCol: number) => {
    const rows = Math.round(scanRows || phaseHeight || 0);
    const cols = Math.round(scanCols || phaseWidth || 0);
    if (!rows || !cols) return;
    const scale = (rawPhaseRef.current?.w || cols) / cols;
    startRow = Math.floor(startRow / scale); endRow = Math.floor(endRow / scale);
    startCol = Math.floor(startCol / scale); endCol = Math.floor(endCol / scale);
    const minSpan = Math.min(32, rows, cols);
    let rowStart = Math.max(0, Math.min(startRow, endRow));
    let rowEnd = Math.min(rows, Math.max(startRow, endRow) + 1);
    let colStart = Math.max(0, Math.min(startCol, endCol));
    let colEnd = Math.min(cols, Math.max(startCol, endCol) + 1);
    if (rowEnd - rowStart < minSpan) rowEnd = Math.min(rows, rowStart + minSpan);
    if (rowEnd - rowStart < minSpan) rowStart = Math.max(0, rowEnd - minSpan);
    if (colEnd - colStart < minSpan) colEnd = Math.min(cols, colStart + minSpan);
    if (colEnd - colStart < minSpan) colStart = Math.max(0, colEnd - minSpan);
    setScanCrop([rowStart, rowEnd, colStart, colEnd]);
  }, [phaseHeight, phaseWidth, scanCols, scanRows]);
  const resetCrop = React.useCallback(() => {
    setScanCrop(null);
    setCropSelecting(false);
  }, []);
  const requestCropRefit = React.useCallback(() => {
    if (!scanCrop || !cropRefitAvailable) return;
    setCropSelecting(false);
    setCropRefitPending(true);
    setCropRefitRequestJson(JSON.stringify({
      id: Date.now(), scan_region: scanCrop, num_trials: 200,
    }));
  }, [cropRefitAvailable, scanCrop, setCropRefitRequestJson]);
  const cropSummary = React.useMemo(() => {
    if (!scanCrop) return "Draw a region";
    const [rowStart, rowEnd, colStart, colEnd] = scanCrop;
    return `r ${rowStart}:${rowEnd}  c ${colStart}:${colEnd}  ${rowEnd - rowStart}x${colEnd - colStart}`;
  }, [scanCrop]);

  // Cached FFT magnitude, so contrast and colormap changes re-render without recomputing the FFT.
  const fftMagRef = React.useRef<{ mag: Float32Array; w: number; h: number } | null>(null);

  // UI toggles: FFT defaults OFF so the drag path is phase-render-only (fastest).
  // The ``fft_on=`` kwarg from Python overrides the default via a
  // one-shot effect below; useState's lazy initializer can't see the trait
  // value yet because anywidget traits arrive asynchronously on mount.
  const [showFFT, setShowFFT] = React.useState<boolean>(false);
  const [fftPlacement, setFFTPlacement] = React.useState<FFTPlacement>("panel");
  const [fftInsetBox, setFFTInsetBox] = React.useState<FFTInsetBox>({ x: 0.72, y: 0.02 });
  const [extraRealViews, setExtraRealViews] = React.useState<Record<ExtraRealViewMode, boolean>>({ amp: false, complex: false });
  const [smooth, setSmooth] = React.useState<boolean>(false);
  const [cmap, setCmap] = React.useState("inferno");
  const [fftCmap, setFftCmap] = React.useState("inferno");
  const [contrastRange, setContrastRange] = React.useState<[number, number]>([1, 99]);
  const [fftContrastRange, setFftContrastRange] = React.useState<[number, number]>([1, 99]);
  // Auto-contrast toggles (persistent, Show2D-style).  When ON, every new
  // reconstruction pins the histogram clip to 1-99 percentile.  Dragging the
  // histogram slider flips auto OFF so the user's manual range sticks.
  const [autoContrast, setAutoContrast] = React.useState(true);
  const [fftAutoContrast, setFftAutoContrast] = React.useState(true);
  const [thumbSize, setThumbSize] = React.useState<ThumbSize>("M");
  const cmapRef = React.useRef(cmap); cmapRef.current = cmap;
  const fftCmapRef = React.useRef(fftCmap); fftCmapRef.current = fftCmap;
  const contrastRef = React.useRef(contrastRange); contrastRef.current = contrastRange;
  const fftContrastRef = React.useRef(fftContrastRange); fftContrastRef.current = fftContrastRange;
  const autoContrastRef = React.useRef(autoContrast); autoContrastRef.current = autoContrast;
  const fftAutoContrastRef = React.useRef(fftAutoContrast); fftAutoContrastRef.current = fftAutoContrast;
  const showFFTRef = React.useRef(showFFT); showFFTRef.current = showFFT;
  const extraRealViewsRef = React.useRef(extraRealViews); extraRealViewsRef.current = extraRealViews;
  const rotationDegRef = React.useRef(rotationDeg ?? 0); rotationDegRef.current = rotationDeg ?? 0;
  const negatePhaseRef = React.useRef(displayNegatePhase);
  negatePhaseRef.current = !!displayNegatePhase;
  const ampDataRef = React.useRef<{ data: Float32Array; w: number; h: number } | null>(null);

  const [panel, setPanel] = React.useState<number>(DEFAULT_PANEL);
  const shownPanelRef = React.useRef(DEFAULT_PANEL);
  // content width of the widget root (its parent's width less the root padding)
  const rootRef = React.useRef<HTMLDivElement>(null);
  const [availableWidth, setAvailableWidth] = React.useState(0);
  React.useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      const style = window.getComputedStyle(root);
      setAvailableWidth(Math.max(0, root.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(root);
    return () => observer.disconnect();
  }, []);

  // One-shot seed from the Python kwarg ``fft_on=``. Runs once (guarded by a
  // ref) after the trait arrives from the kernel, so the FFT switch then
  // belongs to the user.
  const kwargsAppliedRef = React.useRef(false);
  React.useEffect(() => {
    if (kwargsAppliedRef.current) return;
    if (initialFftOn == null) return;
    kwargsAppliedRef.current = true;
    if (initialFftOn) setShowFFT(true);
  }, [initialFftOn]);
  const resizeDrag = React.useRef({ on: false, startY: 0, startSize: 0 });
  const requestIdRef = React.useRef(0);
  // Browser-side reconstruction entry points; false means the WebGPU engine did not take the request, so it goes to Python.
  type FrontendRun = (c10Val: number, c12Val: number, phi12Val: number, rotationVal: number) => boolean;
  const frontendPreviewRef = React.useRef<FrontendRun | null>(null);
  const frontendFullRef = React.useRef<FrontendRun | null>(null);
  const shouldCommitOnReleaseRef = React.useRef(false);
  const [webgpuRuntimeStatus, setWebgpuRuntimeStatus] = React.useState("");
  const [webgpuLoadProgress, setWebgpuLoadProgress] = React.useState<WebGPULoadProgress | null>(null);
  const webgpuSsbRef = React.useRef<WebGPUSSBBackend | null>(null);
  const webgpuInFlightRef = React.useRef(false);
  const webgpuPendingRef = React.useRef<[number, number, number, number] | null>(null);
  const webgpuPendingFullRef = React.useRef(false);
  const webgpuPhaseKeyRef = React.useRef<string | null>(null);
  const webgpuDiagnostic = React.useMemo(() => idleDiagnostic(), []);
  React.useEffect(() => () => webgpuDiagnostic.cancel(), [webgpuDiagnostic]);
  const fullBrightFieldCount = React.useCallback(
    () => Math.max(1, effectiveTotalBf || 1), [effectiveTotalBf]);

  // No-server mode: on file:// the sibling data files cannot be fetch()ed, so
  // the folder must be granted once (picker or webkitdirectory input) before
  // the engine is created. HTTP-served folders skip this entirely.
  const [localSourceGranted, setLocalSourceGranted] = React.useState(() => !ssbNeedsLocalSource());
  const [localSourceError, setLocalSourceError] = React.useState("");
  const localDirInputRef = React.useRef<HTMLInputElement | null>(null);
  // Name of the folder this HTML lives in, so the banner can say exactly what
  // to select in the picker (the exported folder name).
  const localFolderName = React.useMemo(() => {
    try {
      const parts = decodeURIComponent(globalThis.location?.pathname || "").split("/").filter(Boolean);
      return parts.length >= 2 ? parts[parts.length - 2] : "";
    } catch {
      return "";
    }
  }, []);
  const grantLocalDirectory = React.useCallback(async () => {
    const picker = (globalThis as {
      showDirectoryPicker?: (options?: { startIn?: string }) => Promise<FileSystemDirectoryHandle>;
    }).showDirectoryPicker;
    if (picker) {
      try {
        const handle = await picker.call(globalThis, { startIn: "downloads", mode: "readwrite" } as { startIn: string });
        // The calibration is embedded in the page; looking for snapshots/cal.json only catches a
        // wrong folder selection before source/saves reads fail later.
        try {
          const snapshots = await handle.getDirectoryHandle("snapshots");
          await snapshots.getFileHandle("cal.json");
        } catch {
          setLocalSourceError(
            `That folder has no calibration snapshot - select the folder named "${localFolderName || "the one containing this page"}" (the folder this index.html lives in).`,
          );
          return;
        }
        setSSBLocalDirectory(handle);
        setLocalSourceError("");
        setLocalSourceGranted(true);
        return;
      } catch { /* canceled - fall through to input */ }
    }
    localDirInputRef.current?.click();
  }, [localFolderName]);
  // Folder-persisted snapshots: JPEG + entry in snapshots/snapshots.json inside
  // the export folder. Survive relaunch from CLI serve AND double-click; loadable,
  // deletable, downloadable from the strip below the action row.
  type FolderSaveRecord = {
    id: string;
    generated_at_utc: string;
    C10_nm: number;
    C12_nm: number;
    phi12_deg: number;
    scan_detector_rotation_deg: number;
    display_negate_phase: boolean;
    loss: number | null;
    num_bright_field_pixels: number;
    notes: string;
    image: string;
  };
  const [folderSaves, setFolderSaves] = React.useState<FolderSaveRecord[]>([]);
  const [folderSaveStatus, setFolderSaveStatus] = React.useState("");
  const refreshFolderSaves = React.useCallback(async () => {
    const current = await readSSBFolderJson<{ schema?: string; snapshots?: FolderSaveRecord[] }>("snapshots/snapshots.json");
    const readable = current?.schema === SNAPSHOTS_SCHEMA && Array.isArray(current.snapshots);
    // a file of another layout (an earlier release) is not converted: it is left on disk and reported
    if (current && !readable) setFolderSaveStatus(`snapshots/snapshots.json is not ${SNAPSHOTS_SCHEMA}; its snapshots are not listed`);
    setFolderSaves(readable ? current.snapshots! : []);
  }, []);
  React.useEffect(() => {
    if (webgpuStandalone && localSourceGranted) void refreshFolderSaves();
  }, [webgpuStandalone, localSourceGranted, refreshFolderSaves]);
  const onLocalDirInput = React.useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const names = new Set(Array.from(files).flatMap((file) => [
        file.name,
        (file as File & { webkitRelativePath?: string }).webkitRelativePath || "",
      ]));
      const hasCalibration = Array.from(names).some((name) => name.endsWith("snapshots/cal.json"));
      if (!hasCalibration) {
        setLocalSourceError(
          `That folder has no calibration snapshot - select the folder named "${localFolderName || "the one containing this page"}".`,
        );
        return;
      }
      setSSBLocalFiles(Array.from(files));
      setLocalSourceError("");
      setLocalSourceGranted(true);
    }
  }, [localFolderName]);
  React.useEffect(() => {
    webgpuPhaseKeyRef.current = null;
    webgpuDiagnostic.cancel();
    const hasH5Source = !!webgpuH5SourceJson && webgpuH5SourceJson.trim() !== "" && webgpuH5SourceJson.trim() !== "{}";
    if (!webgpuPreviewEnabled || !webgpuCalJson || !hasH5Source) {
      webgpuSsbRef.current = null;
      setWebgpuRuntimeStatus(webgpuPreviewStatus || "");
      setWebgpuLoadProgress(null);
      return;
    }
    if (!localSourceGranted) {
      webgpuSsbRef.current = null;
      setWebgpuRuntimeStatus("No-server mode: grant the folder containing this HTML to start");
      setWebgpuLoadProgress(null);
      return;
    }
    try {
      // the engine reads the calibration and the bright-field source as ShowPtycho.export wrote them
      const engine = new WebGPUSSBBackend(webgpuCalJson, { json: webgpuH5SourceJson });
      engine.setProgressHandler((progress) => {
        setWebgpuLoadProgress(progress.stage === "ready" ? null : progress);
        setWebgpuRuntimeStatus(webgpuProgressLabel(progress));
      });
      webgpuSsbRef.current = engine;
      setWebgpuRuntimeStatus(engine.readyLabel);
      setWebgpuLoadProgress(null);
      return () => {
        webgpuDiagnostic.cancel();
        engine.setProgressHandler(null);
        if (webgpuSsbRef.current === engine) webgpuSsbRef.current = null;
        engine.close();
      };
    } catch (err) {
      webgpuSsbRef.current = null;
      setWebgpuRuntimeStatus(err instanceof Error ? err.message : String(err));
      setWebgpuLoadProgress({
        stage: "error",
        message: "WebGPU ptychography unavailable",
        detail: err instanceof Error ? err.message : String(err),
        percent: 0,
      });
    }
  }, [localSourceGranted, webgpuPreviewEnabled, webgpuCalJson, webgpuH5SourceJson, webgpuPreviewStatus, webgpuDiagnostic]);

  // Display FFT: WebGPU on a hardware adapter, the JS reference otherwise.
  const displayFFTRef = React.useRef<DisplayFFT | null>(null);
  // Generation counter: discards stale async FFT results when newer data arrives.
  const fftGenRef = React.useRef(0);
  // The display FFT in progress, and the newest phase waiting for the next one.
  const fftBusyRef = React.useRef(false);
  const fftQueuedRef = React.useRef<{ data: Float32Array; w: number; h: number } | null>(null);
  React.useEffect(() => {
    let cancelled = false;
    getDisplayFFT().then(fft => {
      if (!cancelled) displayFFTRef.current = fft;
    }).catch(error => {
      if (!cancelled) setWebgpuRuntimeStatus(error instanceof Error ? error.message : String(error));
    });
    return () => { cancelled = true; };
  }, []);

  // WebGPU colormap engine.
  // Each panel binds an explicitly named LUT through applySingleWithLut, so
  // phase/FFT/amplitude submissions cannot leak colormaps across slots.
  // Generation counter discards stale async GPU results if newer data lands first.
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const gpuCmapGenRef = React.useRef<Record<GPUColormapSlot, number>>({ 0: 0, 1: 0, 2: 0 });
  const gpuSlotCanvasRef = React.useRef<Record<GPUColormapSlot, HTMLCanvasElement | null>>({ 0: null, 1: null, 2: null });
  const gpuCmapBusyRef = React.useRef<Record<GPUColormapSlot, boolean>>({ 0: false, 1: false, 2: false });
  const gpuCmapPendingRef = React.useRef<Record<GPUColormapSlot, (() => void) | null>>({ 0: null, 1: null, 2: null });
  // The frame each panel shows, read by the histograms and the cursor readouts: painting a new
  // phase re-renders those, not this component.
  const panelFrames = React.useMemo(createPanelFrames, []);
  const phaseFrameData = React.useCallback(() => panelFrames[0].get().data, [panelFrames]);
  const fftFrameData = React.useCallback(() => panelFrames[1].get().data, [panelFrames]);
  const ampFrameData = React.useCallback(() => panelFrames[2].get().data, [panelFrames]);
  // Percentile histograms and histogram bins of each slot's newest frame, from the display worker.
  const statsWorkerRef = React.useRef<DisplayStatsWorker | null>(null);
  const slotStatsRef = React.useRef<Record<GPUColormapSlot, { data: Float32Array; stats: FrameStats } | null>>({ 0: null, 1: null, 2: null });
  const paintedGenRef = React.useRef<Record<GPUColormapSlot, number>>({ 0: 0, 1: 0, 2: 0 });
  React.useEffect(() => {
    const worker = new DisplayStatsWorker();
    statsWorkerRef.current = worker;
    return () => { worker.dispose(); statsWorkerRef.current = null; };
  }, []);
  const gpuCmapReadyRef = React.useRef(false);
  // Track the data uploaded to each slot so we only re-upload on real data change.
  const gpuSlotDataRef = React.useRef<Record<GPUColormapSlot, Float32Array | null>>({ 0: null, 1: null, 2: null });
  // This view's own engine, so its slots 0-2 never hold another view's images; destroyed on unmount.
  React.useEffect(() => {
    let cancelled = false;
    createGPUColormapEngine().then(engine => {
      if (!engine) return;
      if (cancelled) { engine.destroy(); return; }
      gpuCmapRef.current = engine;
      gpuCmapReadyRef.current = true;
    });
    return () => {
      cancelled = true;
      gpuCmapRef.current?.destroy();
      gpuCmapRef.current = null;
      gpuCmapReadyRef.current = false;
      gpuSlotDataRef.current = { 0: null, 1: null, 2: null };
    };
  }, []);

  /* --- Init sliders from auto --- */
  const initRef = React.useRef(false);
  React.useEffect(() => {
    if (!initRef.current && autoC10 !== undefined) {
      let current: Record<string, number> = {};
      try { current = JSON.parse(model.get("result_json") || "{}"); } catch { /* Use the fitted reference. */ }
      setC10(current.C10_nm ?? autoC10); setC12(current.C12_nm ?? autoC12); setPhi12(current.phi12_deg ?? autoPhi12);
      initRef.current = true;
    }
  }, [autoC10, autoC12, autoPhi12]);

  /* --- Sample panel (thick SSB): push slider values to Python; apply a finished tilt fit to the
         aberration sliders and the panel (the sample_json push then re-renders). --- */
  // Seed once from Python (a restored calibration sets sample_json before the widget mounts); afterwards the panel owns it.
  const [sample, setSample] = React.useState<SampleValues>(() => {
    try {
      const parsed = JSON.parse(sampleJsonModel || "{}");
      return parsed && Number(parsed.depth_spread_nm) > 0
        ? { tiltRowMrad: Number(parsed.sample_tilt_mrad?.[0]) || 0, tiltColMrad: Number(parsed.sample_tilt_mrad?.[1]) || 0, depthSpreadNm: Number(parsed.depth_spread_nm) }
        : SAMPLE_ZERO;
    } catch { return SAMPLE_ZERO; }
  });
  React.useEffect(() => {
    if (inflightDragRef.current || pendingDragRef.current) return;
    const parsed = JSON.parse(sampleJsonModel || "{}");
    const next = {
      tiltRowMrad: Number(parsed.sample_tilt_mrad?.[0]) || 0,
      tiltColMrad: Number(parsed.sample_tilt_mrad?.[1]) || 0,
      depthSpreadNm: Number(parsed.depth_spread_nm) || 0,
    };
    setSample(previous => {
      if (previous.tiltRowMrad === next.tiltRowMrad && previous.tiltColMrad === next.tiltColMrad
          && previous.depthSpreadNm === next.depthSpreadNm) return previous;
      adoptingPanelRef.current = true;
      return next;
    });
  }, [sampleJsonModel]);
  const sampleRef = React.useRef<SampleValues>(SAMPLE_ZERO);
  sampleRef.current = sample;
  const samplePreviewRef = React.useRef<SampleValues>(SAMPLE_ZERO);
  samplePreviewRef.current = tiltCorrection === false
    ? { ...sample, tiltRowMrad: 0, tiltColMrad: 0 }
    : sample;
  const [sampleOpen, setSampleOpen] = React.useState(false);
  const [sampleFitBusy, setSampleFitBusy] = React.useState(false);
  React.useEffect(() => {
    if (/^Tilt fit(:| failed)/.test(sampleFitStatus || "")) setSampleFitBusy(false);
  }, [sampleFitStatus]);
  const requestSampleFit = React.useCallback(() => {
    if (regionContextRef.current) {
      const regions = JSON.parse(model.get('local_tilt_selection_json') || 'null') || JSON.parse(model.get('local_tilt_json')).regions;
      model.set('local_tilt_request_json', JSON.stringify({id:Date.now(),action:'fit',regions,
        signature:JSON.stringify(regions),only_changed:true,force_region:regionContextRef.current.id}));
      model.save_changes();
      return;
    }
    setSampleFitBusy(true);
    setSampleFitRequest((sampleFitRequest || 0) + 1);
  }, [sampleFitRequest, setSampleFitRequest]);

  /* --- Request plumbing ---
   * Slider events use every available bright-field pixel.
   * In-flight throttling: at most PREVIEW_REQUESTS_IN_FLIGHT overview requests (one for a region)
   * are outstanding; newer slider values wait as one latest-wins pending request. The kernel answers
   * requests in order, so with two outstanding it starts the next reconstruction while the previous
   * phase is still travelling to the browser, instead of idling for the round trip. */
  const inflightDragRef = React.useRef(false);
  const pendingDragRef = React.useRef<{ values: [number, number, number]; committed: boolean } | null>(null);
  // The outstanding requests, oldest first: a reply releases its own request and any older one.
  const sentRequestsRef = React.useRef<{ id: number; regionId: string | null }[]>([]);
  const [previewError, setPreviewError] = useChangedState('');
  const requestCapacity = () => (regionContextRef.current ? 1 : PREVIEW_REQUESTS_IN_FLIGHT);

  const fireRequest = React.useCallback((c10Val: number, c12Val: number, phi12Val: number, committed: boolean) => {
    const region = regionContextRef.current;
    if (region && !region.ready) {
      inflightDragRef.current = sentRequestsRef.current.length > 0;
      setBusy(inflightDragRef.current);
      return;
    }
    const id = ++requestIdRef.current;
    sentRequestsRef.current = [...sentRequestsRef.current, { id, regionId: region ? region.id : null }];
    inflightDragRef.current = true;
    setBusy(true);
    if (viewPinRef.current !== null) setViewPin(null);
    const sample = sampleRef.current;
    // Our bounded in-flight queue already supplies backpressure. A custom command
    // can start as soon as a previous result arrives, without the notebook's
    // generic trait throttle waiting for a later kernel-idle notification.
    model.send({ type: "ssb-preview", request: { id, C10_nm: c10Val, C12_nm: c12Val, phi12_deg: phi12Val, committed,
      ...(region ? {region_id:region.id,region_bounds:region.bounds} : {}),
      sample: { sample_tilt_mrad: [sample.tiltRowMrad, sample.tiltColMrad], depth_spread_nm: sample.depthSpreadNm },
      higher_order: higherOrderRef.current,
    }});
  }, [model]);

  const sendDrag = React.useCallback((c10Val: number, c12Val: number, phi12Val: number, rotationVal = rotationDegRef.current) => {
    if (!regionContextRef.current && frontendPreviewRef.current?.(c10Val, c12Val, phi12Val, rotationVal)) return;
    if (sentRequestsRef.current.length >= requestCapacity()) {
      pendingDragRef.current = { values: [c10Val, c12Val, phi12Val], committed: false };
      return;
    }
    fireRequest(c10Val, c12Val, phi12Val, false);
  }, [fireRequest]);

  const sendCommit = React.useCallback((c10Val: number, c12Val: number, phi12Val: number, rotationVal = rotationDegRef.current) => {
    pendingDragRef.current = null;
    if (!regionContextRef.current && frontendFullRef.current?.(c10Val, c12Val, phi12Val, rotationVal)) return;
    if (sentRequestsRef.current.length >= requestCapacity()) {
      pendingDragRef.current = { values: [c10Val, c12Val, phi12Val], committed: true };
      return;
    }
    fireRequest(c10Val, c12Val, phi12Val, true);
  }, [fireRequest]);

  /* --- Slider values ref (so callbacks read fresh values without re-creating) --- */
  const sliderVals = React.useRef({ c10: 0, c12: 0, phi12: 0 });
  // A dragged aberration thumb and its label re-render once per animation frame; every pointer
  // event still sends its request with its own value, and the ref always holds the newest values.
  const sliderPendingRef = React.useRef<Partial<Record<"c10" | "c12" | "phi12", number>>>({});
  sliderVals.current = {
    c10: sliderPendingRef.current.c10 ?? c10,
    c12: sliderPendingRef.current.c12 ?? c12,
    phi12: sliderPendingRef.current.phi12 ?? phi12,
  };
  const flushSliderState = React.useCallback(() => {
    const next = sliderPendingRef.current;
    sliderPendingRef.current = {};
    if (next.c10 !== undefined) setC10(next.c10);
    if (next.c12 !== undefined) setC12(next.c12);
    if (next.phi12 !== undefined) setPhi12(next.phi12);
  }, []);
  const sliderStateFrame = React.useMemo(() => latestFrame(flushSliderState), [flushSliderState]);
  React.useEffect(() => () => sliderStateFrame.cancel(), [sliderStateFrame]);
  const dragSliderValue = React.useCallback((key: "c10" | "c12" | "phi12", value: number) => {
    sliderPendingRef.current[key] = value;
    sliderVals.current = { ...sliderVals.current, [key]: value };
    sliderStateFrame.schedule();
  }, [sliderStateFrame]);
  // A release shows its committed value at once.
  const commitSliderValue = React.useCallback((key: "c10" | "c12" | "phi12", value: number) => {
    sliderStateFrame.cancel();
    sliderPendingRef.current[key] = value;
    flushSliderState();
  }, [sliderStateFrame, flushSliderState]);
  /* Panel changes share the one-in-flight request path with the primary sliders.
     Frames read the latest values; continuous input never restarts a debounce. */
  const sampleCommitPendingRef = React.useRef(false);
  const firstSampleEffectRef = React.useRef(true);
  const panelRunRef = React.useRef(() => {});
  panelRunRef.current = () => {
    if (!initRef.current) return;
    const commit = sampleCommitPendingRef.current;
    sampleCommitPendingRef.current = false;
    const current = sliderVals.current;
    if (webgpuStandalone) {
      const value = sampleRef.current;
      setSampleJson(JSON.stringify({ sample_tilt_mrad: [value.tiltRowMrad, value.tiltColMrad], depth_spread_nm: value.depthSpreadNm }));
      setHigherOrderJson(JSON.stringify(higherOrderRef.current));
    }
    (commit ? sendCommit : sendDrag)(current.c10, current.c12, current.phi12, rotationDegRef.current);
  };
  const panelFrame = React.useMemo(() => latestFrame(() => panelRunRef.current()), []);
  React.useEffect(() => () => panelFrame.cancel(), [panelFrame]);
  React.useEffect(() => {
    if (firstSampleEffectRef.current) { firstSampleEffectRef.current = false; return; }
    if (adoptingPanelRef.current) { adoptingPanelRef.current = false; return; }
    panelFrame.schedule();
  }, [sample, higherOrder, panelFrame]);
  const previousTiltCorrectionRef = React.useRef(tiltCorrection);
  React.useEffect(() => {
    if (previousTiltCorrectionRef.current === tiltCorrection) return;
    previousTiltCorrectionRef.current = tiltCorrection;
    // Python handles its own preview-only trait; standalone exports run here.
    if (webgpuStandalone) panelFrame.schedule();
  }, [tiltCorrection, webgpuStandalone, panelFrame]);
  const commitSample = React.useCallback(() => {
    sampleCommitPendingRef.current = true;
    setSample(prev => ({ ...prev }));
  }, []);

  /* --- Play control: sweep one parameter over its range ---
     Like Show3D playback: pick which aberration to sweep, set FPS, hit play.
     Each tick advances the sweep index and fires a normal drag/commit request
     so the image updates with the same pipeline users see when they drag a slider.
     Drag and sweep requests use the same full bright-field set. */
  // SweepParam: one of the main-slider keys, or a higher-order key of the form
  // "ho:<hoKey>" where <hoKey> is either a magnitude key (e.g. "C32_nm",
  // "C30_nm") or an azimuth key ("phi32_deg").
  type SweepParam = string;
  const [playing, setPlaying] = React.useState(false);
  const [playFps, setPlayFps] = React.useState(10);

  /* --- Optuna trials panel state.  Parsed once from the trait, renders as
     a loss-ranked strip the user can click through to preview any trial.
     Collapsed by default when there are many trials so the widget doesn't
     dominate the notebook scroll. --- */
  // objective: what the trial was scored on, quantem.gpu's names: "agreement" (the default model fit; loss is the
  // negative agreement) or "phase_variance" (the backup search); losses rank only within one objective
  interface OptunaTrial {
    rank: number; C10_nm: number; C12_nm: number; phi12_deg: number; loss: number;
    objective?: "agreement" | "phase_variance"; sample_tilt_mrad?: [number, number]; depth_spread_nm?: number;
  }
  const trials: OptunaTrial[] = React.useMemo(() => {
    if (!trialsJson) return [];
    try {
      const parsed = JSON.parse(trialsJson);
      return Array.isArray(parsed) ? parsed : [];
    } catch { return []; }
  }, [trialsJson]);
  // the objectives the strip lists, in their order, and a trial's own score: agreement is the negative loss
  const trialObjectiveText = Array.from(new Set(trials.map(trial => trial.objective ?? "phase_variance")))
    .map(objective => objective === "agreement" ? "model agreement" : "phase variance").join(", ");
  const trialScoreText = (trial: OptunaTrial) => trial.objective === "agreement"
    ? `agr ${formatNumber(-trial.loss, 4)}` : `pv ${formatNumber(trial.loss, 8)}`;
  const [trialsExpanded, setTrialsExpanded] = React.useState(false);
  const [activeTrialRank, setActiveTrialRank] = React.useState<number | null>(null);

  const doViewTrial = React.useCallback((trial: OptunaTrial) => {
    setActiveTrialRank(trial.rank);
    setViewPin(null);   // trials + pins are mutually exclusive selections
    if (trial.sample_tilt_mrad && trial.depth_spread_nm != null && !regionContextRef.current) {
      // a trial is its optics with its own sample model (tilt 0 and no depth for a tilt-0 or phase-variance trial)
      const trialSample = { tiltRowMrad: trial.sample_tilt_mrad[0], tiltColMrad: trial.sample_tilt_mrad[1],
                            depthSpreadNm: trial.depth_spread_nm };
      sampleRef.current = trialSample;
      adoptingPanelRef.current = true;
      setSample(trialSample);
    }
    setC10(trial.C10_nm); setC12(trial.C12_nm); setPhi12(trial.phi12_deg);
    sendCommit(trial.C10_nm, trial.C12_nm, trial.phi12_deg);
  }, [sendCommit]);
  const [sweepParam, setSweepParam] = React.useState<SweepParam>("c10");
  const sweepIdxRef = React.useRef(0);
  const sweepDirRef = React.useRef<1 | -1>(1);  // boomerang direction
  const sweepMainKeys = React.useCallback((param: SweepParam): AberKey[] => {
    switch (param) {
      case "c10":
      case "c12":
      case "phi12":
      case "rot":
        return [param];
      case "bundle:c10c12":
        return ["c10", "c12"];
      case "bundle:c10c12phi12":
        return ["c10", "c12", "phi12"];
      case "bundle:c10c12phi12rot":
        return ["c10", "c12", "phi12", "rot"];
      default:
        return [];
    }
  }, []);

  // Full range [lo, hi] for the currently-selected sweep parameter.  Used
  // both as the bounds of the range slider and the auto-reset target when the
  // user changes the sweep parameter.
  const getFullSweepRange = React.useCallback((param: SweepParam): [number, number] => {
    if (param.startsWith("bundle:")) return [0, 1];
    if (param.startsWith("ho:")) {
      const hoKey = param.slice(3);
      if (hoKey.startsWith("phi")) return [-180, 180];
      const baseName = hoKey.slice(0, -3);
      for (const order of [2, 3, 4, 5]) {
        const entry = HO_BY_ORDER[order].find(candidate => candidate.name === baseName);
        if (entry) return [-entry.mag_max, entry.mag_max];
      }
      return [0, 0];
    }
    switch (param) {
      case "c10":   return [c10UiMin, c10UiMax];
      case "c12":   return [c12UiMin, c12UiMax];
      case "phi12": return [phi12UiMin, phi12UiMax];
      case "rot":   return [rotationUiMin, rotationUiMax];
    }
    return [0, 0];  // any other key has no sweep range
  }, [c10UiMin, c10UiMax, c12UiMin, c12UiMax, phi12UiMin, phi12UiMax, rotationUiMin, rotationUiMax]);

  // Per-parameter sweep range.  Each aberration slider has its own stopper
  // brackets; switching the sweep param doesn't reset the others' windows.
  type SweepRanges = Record<AberKey, [number, number]>;
  const [sweepRanges, setSweepRanges] = React.useState<SweepRanges>(() => ({
    c10: getFullSweepRange("c10"),
    c12: getFullSweepRange("c12"),
    phi12: getFullSweepRange("phi12"),
    rot: getFullSweepRange("rot"),
  }));
  // Trait-backed min/max values arrive asynchronously from Python after mount.
  // The useState initializer above sees the Float(0.0) defaults, so without this
  // effect the rotation stoppers lock at [0, 0] and disableSwap pins the middle
  // thumb at zero, so rotation drag looks dead until the ranges refresh.  Gate
  // with a ref so user-narrowed windows don't get stomped on later re-renders.
  const sweepRangesSeededRef = React.useRef(false);
  React.useEffect(() => {
    if (sweepRangesSeededRef.current) return;
    const full: SweepRanges = {
      c10: [c10UiMin, c10UiMax],
      c12: [c12UiMin, c12UiMax],
      phi12: [phi12UiMin, phi12UiMax],
      rot: [rotationUiMin, rotationUiMax],
    };
    // Only consider traits "arrived" once at least one min/max is non-default.
    const arrived =
      c10UiMin !== 0 || c10UiMax !== 0 ||
      phi12UiMin !== 0 || phi12UiMax !== 0 ||
      rotationUiMin !== 0 || rotationUiMax !== 0;
    if (!arrived) return;
    sweepRangesSeededRef.current = true;
    setSweepRanges(full);
  }, [c10UiMin, c10UiMax, c12UiMin, c12UiMax, phi12UiMin, phi12UiMax, rotationUiMin, rotationUiMax]);
  const updateSweepRange = React.useCallback((key: AberKey, range: [number, number]) => {
    setSweepRanges(prev => ({ ...prev, [key]: range }));
  }, []);
  const previousUiRangesRef = React.useRef(uiRanges);
  React.useEffect(() => {
    const old = previousUiRangesRef.current;
    setSweepRanges(prev => {
      const next = { ...prev };
      (Object.keys(uiRanges) as AberKey[]).forEach(key => {
        const [lo, hi] = uiRanges[key];
        const oldFull = old[key];
        const current = prev[key];
        const wasFull = current[0] === oldFull[0] && current[1] === oldFull[1];
        next[key] = wasFull
          ? [lo, hi]
          : [Math.max(lo, Math.min(hi, current[0])), Math.max(lo, Math.min(hi, current[1]))];
      });
      return next;
    });
    previousUiRangesRef.current = uiRanges;
  }, [uiRanges]);
  const sweepRangesRef = React.useRef(sweepRanges);
  sweepRangesRef.current = sweepRanges;

  // Resolve the active PLAY range from the selected sweep parameter.
  const sweepRange: [number, number] = React.useMemo(() => {
    if (sweepParam.startsWith("bundle:")) return [0, 1];
    if (sweepParam.startsWith("ho:")) return getFullSweepRange(sweepParam);
    return sweepRanges[sweepParam as AberKey];
  }, [sweepParam, sweepRanges, getFullSweepRange]);
  const sweepRangeRef = React.useRef(sweepRange);
  sweepRangeRef.current = sweepRange;
  React.useEffect(() => {
    if (!playing) return;
    const tick = () => {
      // Target value for this frame: interpolate through the user's sub-range
      // (defaults to the full slider range for the selected sweep param).
      const [lo, hi] = sweepRangeRef.current;
      let hoKey: string | null = null;
      if (sweepParam.startsWith("ho:")) hoKey = sweepParam.slice(3);
      const frac = sweepIdxRef.current / (SWEEP_STEPS - 1);
      const value = lo + frac * (hi - lo);
      // Apply the value, using the same code path the sliders would
      if (hoKey) {
        // The next animation frame sends the latest higher-order correction.
        setHigherOrder(prev => ({ ...prev, [hoKey!]: value }));
      } else if (sweepParam.startsWith("bundle:")) {
        const keys = sweepMainKeys(sweepParam);
        const nextC10 = keys.includes("c10")
          ? sweepRangesRef.current.c10[0] + frac * (sweepRangesRef.current.c10[1] - sweepRangesRef.current.c10[0])
          : sliderVals.current.c10;
        const nextC12 = keys.includes("c12")
          ? sweepRangesRef.current.c12[0] + frac * (sweepRangesRef.current.c12[1] - sweepRangesRef.current.c12[0])
          : sliderVals.current.c12;
        const nextPhi12 = keys.includes("phi12")
          ? sweepRangesRef.current.phi12[0] + frac * (sweepRangesRef.current.phi12[1] - sweepRangesRef.current.phi12[0])
          : sliderVals.current.phi12;
        const nextRot = keys.includes("rot")
          ? sweepRangesRef.current.rot[0] + frac * (sweepRangesRef.current.rot[1] - sweepRangesRef.current.rot[0])
          : rotationDegRef.current;
        if (keys.includes("c10")) setC10(nextC10);
        if (keys.includes("c12")) setC12(nextC12);
        if (keys.includes("phi12")) setPhi12(nextPhi12);
        if (keys.includes("rot")) setRotationDeg(nextRot);
        sendDrag(nextC10, nextC12, nextPhi12, nextRot);
      } else if (sweepParam === "c10") {
        setC10(value); sendDrag(value, sliderVals.current.c12, sliderVals.current.phi12);
      } else if (sweepParam === "c12") {
        setC12(value); sendDrag(sliderVals.current.c10, value, sliderVals.current.phi12);
      } else if (sweepParam === "phi12") {
        setPhi12(value); sendDrag(sliderVals.current.c10, sliderVals.current.c12, value);
      } else if (sweepParam === "rot") {
        setRotationDeg(value);
        if (webgpuSsbRef.current) sendDrag(sliderVals.current.c10, sliderVals.current.c12, sliderVals.current.phi12, value);
      }
      // Advance index (boomerang: bounces between 0 and SWEEP_STEPS-1)
      sweepIdxRef.current += sweepDirRef.current;
      if (sweepIdxRef.current >= SWEEP_STEPS - 1) {
        sweepIdxRef.current = SWEEP_STEPS - 1;
        sweepDirRef.current = -1;
      } else if (sweepIdxRef.current <= 0) {
        sweepIdxRef.current = 0;
        sweepDirRef.current = 1;
      }
    };
    tick();  // fire once immediately so user sees motion without a delay
    const interval = window.setInterval(tick, Math.max(1000 / 60, 1000 / playFps));
    return () => window.clearInterval(interval);
    // sendDrag, setRotationDeg and the slider setters are stable callbacks, and
    // c10Min etc. come from traits that only change at init, so only the play
    // state restarts the sweep timer.
  }, [playing, playFps, sweepParam]);

  /* --- Process result from Python --- */
  React.useEffect(() => {
    const receive = (regional = false) => {
      const resultJson = model.get(regional ? "local_tilt_preview_json" : "result_json");
      if (!resultJson) return;
      let result: { id?: number; region_id?: string; error?: string; loss?: number | null };
      try { result = JSON.parse(resultJson); } catch { return; }
      // Replies name their request and, when regional, their region. A late reply
      // from another view must not set this view's loss or release its request.
      const regionId = regional ? (result.region_id ?? null) : null;
      const shown = regionContextRef.current?.id ?? null;
      const sent = sentRequestsRef.current;
      const answered = sent.findIndex(request => request.id === result.id && request.regionId === regionId);
      if (sent.length && answered < 0) {
        if (regionId === shown && !regional && result.loss != null) setLoss(result.loss);  // a Python-side redraw
        return;
      }
      // Replies arrive in request order: this one also retires any older request.
      sentRequestsRef.current = sent.slice(answered + 1);
      inflightDragRef.current = sentRequestsRef.current.length > 0;
      setBusy(inflightDragRef.current);
      if (result.error) {
        // Queued values were relative to the failed request; drop them and say why.
        pendingDragRef.current = null;
        setPreviewError(result.error);
        return;
      }
      setPreviewError('');
      if (result.loss != null && regionId === shown) setLoss(result.loss);

      // Drain pending drag value if user moved the slider while we were busy.
      const pending = pendingDragRef.current;
      if (pending && sentRequestsRef.current.length < requestCapacity()) {
        pendingDragRef.current = null;
        fireRequest(...pending.values, pending.committed);
      }
    };
    // Dispatch the next reconstruction at comm arrival, before React paints
    // labels, histograms and controls for the completed frame.
    const overviewResult = () => receive(false);
    const regionResult = () => receive(true);
    model.on("change:result_json", overviewResult);
    model.on("change:local_tilt_preview_json", regionResult);
    receive();
    return () => { model.off("change:result_json", overviewResult); model.off("change:local_tilt_preview_json", regionResult); };
  }, [model, fireRequest]);

  /** The display statistics of `data` in `slot`: the ones given (the FFT worker bins its own
   *  magnitude), the cached ones of the frame the slot already holds (a contrast or colormap
   *  change), else the display worker's. */
  const slotStats = React.useCallback((slot: GPUColormapSlot, data: Float32Array, given?: FrameStats): FrameStats | Promise<FrameStats> => {
    const keep = (stats: FrameStats) => { slotStatsRef.current[slot] = { data, stats }; return stats; };
    if (given) return keep(given);
    const cached = slotStatsRef.current[slot];
    if (cached && cached.data === data) return cached.stats;
    const worker = statsWorkerRef.current;
    return worker ? worker.stats(data).then(keep) : keep(frameStats(data));
  }, []);

  /** Colormap one panel into its canvas: WebGPU when a hardware adapter exists, else the
   *  JS applyColormap with the same float32 window math, so both paths paint the same pixels.
   *  The percentile window comes from the frame's histogram, binned once per frame in the
   *  display worker; painting publishes the frame to its histogram without a widget render. */
  const renderPanel = React.useCallback((data: Float32Array, w: number, h: number, slot: GPUColormapSlot, cmapName: string, pctLo: number, pctHi: number, given?: FrameStats) => {
    const gen = ++gpuCmapGenRef.current[slot];
    const paint = (pixels: Uint8ClampedArray | ImageBitmap, min: number, max: number, stats: FrameStats) => {
      // Launches run one at a time per slot, so a later generation never paints before an earlier one.
      if (gen < paintedGenRef.current[slot]) return;
      paintedGenRef.current[slot] = gen;
      let canvas = gpuSlotCanvasRef.current[slot];
      const fresh = !canvas || canvas.width !== w || canvas.height !== h;
      if (fresh) {
        canvas = document.createElement("canvas");
        canvas.width = w; canvas.height = h;
        gpuSlotCanvasRef.current[slot] = canvas;
      }
      const ctx = canvas!.getContext("2d");
      if (!ctx) return;
      if (pixels instanceof ImageBitmap) {
        ctx.drawImage(pixels, 0, 0);
      } else {
        const image = ctx.createImageData(w, h);
        image.data.set(pixels);
        ctx.putImageData(image, 0, 0);
      }
      panelFrames[slot].set({ data, min, max, displayBins: stats.displayBins });
      canvas!.dispatchEvent(new Event("quantem-frame"));
      // A new canvas (first frame or new dimensions) re-renders the widget for its panels and labels.
      if (fresh) {
        if (slot === 0) setPhaseOff(canvas!);
        else if (slot === 1) setFFTOff(canvas!);
        else setAmpOff(canvas!);
      }
    };
    const engine = gpuCmapRef.current;
    const lut = COLORMAPS[cmapName as keyof typeof COLORMAPS] || COLORMAPS.viridis;
    if (!engine || !gpuCmapReadyRef.current) {
      const cpuPaint = (stats: FrameStats) => {
        const { vmin, vmax, min, max } = percentileWindow(stats.histogram, pctLo, pctHi);
        const rgba = new Uint8ClampedArray(w * h * 4);
        applyColormap(data, rgba, lut, vmin, vmax);
        paint(rgba, min, max, stats);
      };
      const stats = slotStats(slot, data, given);
      if (stats instanceof Promise) void stats.then(cpuPaint);
      else cpuPaint(stats);
      return;
    }
    const launch = () => {
      gpuCmapBusyRef.current[slot] = true;
      // Upload data only if it actually changed (new reconstruction), not on every contrast tick.
      const prev = gpuSlotDataRef.current[slot];
      if (prev !== data) {
        engine.uploadData(slot, data, w, h);
        gpuSlotDataRef.current[slot] = data;
      }
      // Keep one pass per slot and replace pending contrast changes with the latest; the
      // completed GPU image transfers directly, without reading RGBA into JS.
      Promise.resolve(slotStats(slot, data, given)).then(stats => {
        const { vmin, vmax, min, max } = percentileWindow(stats.histogram, pctLo, pctHi);
        return engine.renderSlotsToImageBitmapAsync([slot], [{ vmin, vmax }], false, cmapName, lut).then(bitmaps => {
          const bitmap = bitmaps?.[0];
          if (!bitmap) return;
          try { paint(bitmap, min, max, stats); } finally { bitmap.close(); }
        });
      }).catch((error) => {
        if (gen === gpuCmapGenRef.current[slot]) {
          setWebgpuRuntimeStatus(`WebGPU phase colormap failed: ${error instanceof Error ? error.message : String(error)}`);
        }
      }).finally(() => {
        gpuCmapBusyRef.current[slot] = false;
        const pending = gpuCmapPendingRef.current[slot];
        gpuCmapPendingRef.current[slot] = null;
        pending?.();
      });
    };
    if (gpuCmapBusyRef.current[slot]) {
      gpuCmapPendingRef.current[slot] = launch;
    } else {
      launch();
    }
  }, [panelFrames, slotStats]);

  /* --- Re-render real-space displays when additive views/contrast/cmap changes. --- */
  const renderRealDisplay = React.useCallback((phase: Float32Array, w: number, h: number) => {
    renderPanel(phase, w, h, 0, cmapRef.current, contrastRange[0], contrastRange[1]);
    const extras = extraRealViewsRef.current;
    const objectAmp = objectAmplitudeRef.current;
    // only an object-wave array carries an amplitude (SSB recovers the phase alone)
    const amp = objectAmp && objectAmp.length === phase.length && (extras.amp || extras.complex) ? objectAmp : null;
    ampDataRef.current = amp ? { data: amp, w, h } : null;
    if (amp && extras.amp) {
      renderPanel(amp, w, h, 2, cmapRef.current, contrastRange[0], contrastRange[1]);
    } else {
      setAmpOff(null);
    }
    setComplexOff(amp && extras.complex ? renderComplexPhaseOffscreen(phase, w, h, contrastRange[0], contrastRange[1], amp).canvas : null);
  }, [contrastRange, renderPanel]);

  /* --- Re-render FFT when its contrast/cmap changes (no FFT recompute).
         Uses the user-selected FFT colormap
         (defaults to inferno but follows the dropdown state). --- */
  const rerenderFFT = React.useCallback(() => {
    const fft = fftMagRef.current;
    if (!fft) return;
    // Slot 1 holds the FFT magnitude.
    renderPanel(fft.mag, fft.w, fft.h, 1, fftCmap, fftContrastRange[0], fftContrastRange[1]);
  }, [fftContrastRange, fftCmap, renderPanel]);

  /* --- Full render: real-space view synchronous, FFT only when requested. --- */
  const renderPreviewPhase = React.useCallback((data: Float32Array, w: number, h: number) => {
    rawPhaseRef.current = { data, w, h };
    if (autoContrastRef.current) {
      const range = contrastRef.current;
      if (range[0] !== 1 || range[1] !== 99) setContrastRange([1, 99]);
    }
    renderRealDisplay(data, w, h);
  }, [renderRealDisplay]);

  const renderAll = React.useCallback((data: Float32Array, w: number, h: number) => {
    rawPhaseRef.current = { data, w, h };
    // Auto-contrast: new data snaps clip back to 1-99 percentile on every
    // reconstruction.  The user can override via the histogram slider, which
    // flips auto OFF until they re-enable it.
    if (autoContrastRef.current) {
      const range = contrastRef.current;
      if (range[0] !== 1 || range[1] !== 99) setContrastRange([1, 99]);
    }
    if (showFFTRef.current && fftAutoContrastRef.current) {
      const fftRange = fftContrastRef.current;
      if (fftRange[0] !== 1 || fftRange[1] !== 99) setFftContrastRange([1, 99]);
    }
    renderRealDisplay(data, w, h);

    // Skip the FFT entirely when hidden: saves 30-80 ms/frame.
    if (!showFFTRef.current) {
      fftMagRef.current = null;
      return;
    }

    const applyFFT = (fftResult: { mag: Float32Array; pw: number; ph: number; stats: FrameStats }) => {
      fftMagRef.current = { mag: fftResult.mag, w: fftResult.pw, h: fftResult.ph };
      const fftRange = fftContrastRef.current;
      const fftCmapName = fftCmapRef.current;
      renderPanel(fftResult.mag, fftResult.pw, fftResult.ph, 1, fftCmapName, fftRange[0], fftRange[1], fftResult.stats);
    };

    // One display FFT at a time; phases that arrive meanwhile leave only the newest queued. Starting
    // an FFT per phase and discarding all but the last starved the panel once phases arrived faster
    // than a 1024^2 display FFT completes.
    const launchFFT = (phase: Float32Array, width: number, height: number) => {
      fftBusyRef.current = true;
      const gen = ++fftGenRef.current;
      void (displayFFTRef.current
        ? Promise.resolve(displayFFTRef.current)
        : getDisplayFFT()).then(fft => {
        displayFFTRef.current = fft;
        const display = fftMagnitudeDisplayRef.current;
        if (!display) throw new Error("FFT display is closed.");
        return computeFFTMagnitude(fft, display, phase, width, height);
      }).then(fftResult => {
        if (gen !== fftGenRef.current) return;  // stale: newer data already rendered
        applyFFT(fftResult);
      }).catch(error => {
        if (gen !== fftGenRef.current) return;
        setFFTOff(null);
        setWebgpuRuntimeStatus(error instanceof Error ? error.message : String(error));
      }).finally(() => {
        fftBusyRef.current = false;
        const queued = fftQueuedRef.current;
        fftQueuedRef.current = null;
        if (queued && showFFTRef.current) launchFFT(queued.data, queued.w, queued.h);
      });
    };
    if (fftBusyRef.current) {
      fftQueuedRef.current = { data, w, h };
      return;
    }
    launchFFT(data, w, h);
  }, [renderPanel, renderRealDisplay]);

  const renderAllRef = React.useRef(renderAll);
  renderAllRef.current = renderAll;
  const pendingWebgpuDisplay = React.useRef<{
    engine: WebGPUSSBBackend; phase: Float32Array; width: number; height: number;
  } | null>(null);
  const webgpuDisplayFrame = React.useMemo(() => latestFrame(() => {
    const next = pendingWebgpuDisplay.current;
    pendingWebgpuDisplay.current = null;
    if (!next || webgpuSsbRef.current !== next.engine) return;
    if (negatePhaseRef.current) {
      for (let i = 0; i < next.phase.length; i++) next.phase[i] = -next.phase[i];
    }
    renderAllRef.current(next.phase, next.width, next.height);
  }), []);
  React.useEffect(() => () => webgpuDisplayFrame.cancel(), [webgpuDisplayFrame]);

  // One WebGPU request runs at a time; when it settles, free the slot and run only the newest queued
  // request, so a drag that outpaces the engine renders its latest position, not every step in between.
  // Reads refs and a state setter only, so the callbacks below may hold it from any render.
  const drainWebgpuQueue = () => {
    webgpuInFlightRef.current = false;
    setBusy(false);
    const pending = webgpuPendingRef.current;
    if (pending) {
      const pendingFull = webgpuPendingFullRef.current;
      webgpuPendingRef.current = null;
      webgpuPendingFullRef.current = false;
      (pendingFull ? frontendFullRef.current : frontendPreviewRef.current)?.(pending[0], pending[1], pending[2], pending[3]);
    }
  };
  const runFrontendPreview = React.useCallback((c10Val: number, c12Val: number, phi12Val: number, rotationVal: number): boolean => {
    const engine = webgpuSsbRef.current;
    if (!engine) return false;
    webgpuDiagnostic.cancel();
    setLoss(null);
    if (webgpuInFlightRef.current) {
      webgpuPendingRef.current = [c10Val, c12Val, phi12Val, rotationVal];
      webgpuPendingFullRef.current = false;
      return true;
    }
    webgpuInFlightRef.current = true;
    setBusy(true);
    setViewPin(null);
    const bfCount = fullBrightFieldCount();
    const total = Math.max(1, effectiveTotalBf || bfCount);
    const isFull = bfCount >= total;
    const request = {
      aberrations: { C10_nm: c10Val, C12_nm: c12Val, phi12_deg: phi12Val },
      higherOrderAberrations: higherOrderRef.current,
      scanDetectorRotationDeg: rotationVal,
      upsamplingFactor: upsamplingFactorRef.current,
      numBrightFieldPixels: bfCount,
      reducedBrightField: !isFull,
      computeLoss: false,
      sample: samplePreviewRef.current,
    };
    const phaseKey = JSON.stringify(request);
    reconstruct(engine, request).then(result => {
      if (webgpuSsbRef.current !== engine) return;
      webgpuPhaseKeyRef.current = phaseKey;
      if (result.loss != null) {
        setLoss(result.loss);
      } else {
        setLoss(null);
      }
      const modeLabel = isFull ? "full BF phase" : "drag";
      const hoLabel = hoActiveRef.current ? `, HO=${hoActiveCountRef.current}` : "";
      setWebgpuLoadProgress(null);
      setWebgpuRuntimeStatus(webgpuRunStatus(result, modeLabel, total, c10Val, hoLabel));
      if (isFull || showFFTRef.current) {
        // Let the next GPU request start before full-grid display statistics
        // run on the main thread. Paint only the latest completed phase.
        pendingWebgpuDisplay.current = { engine, phase: result.phase, width: result.width, height: result.height };
        webgpuDisplayFrame.schedule();
      } else {
        const phase = result.phase;
        if (negatePhaseRef.current) {
          for (let i = 0; i < phase.length; i++) phase[i] = -phase[i];
        }
        renderPreviewPhase(phase, result.width, result.height);
      }
    }).catch(err => {
      if (webgpuSsbRef.current !== engine) return;
      const message = err instanceof Error ? err.message : String(err);
      console.error("[showptycho] WebGPU preview failed", err);
      if (webgpuStandalone) {
        setWebgpuLoadProgress({
          stage: "error",
          message: "WebGPU ptychography failed",
          detail: message,
          percent: 0,
        });
        setWebgpuRuntimeStatus(`WebGPU folder failed: ${message}`);
      } else {
        webgpuSsbRef.current = null;
        setWebgpuLoadProgress({ stage: "error", message: "WebGPU ptychography failed", detail: message, percent: 0 });
        setWebgpuRuntimeStatus(`WebGPU preview failed: ${message}`);
      }
    }).finally(drainWebgpuQueue);
    return true;
  }, [effectiveTotalBf, fireRequest, renderAll, renderPreviewPhase, fullBrightFieldCount, webgpuStandalone]);

  frontendPreviewRef.current = runFrontendPreview;
  const runFrontendFull = React.useCallback((c10Val: number, c12Val: number, phi12Val: number, rotationVal: number): boolean => {
    const engine = webgpuSsbRef.current;
    if (!webgpuStandalone || !engine) return false;
    webgpuDiagnostic.cancel();
    if (webgpuInFlightRef.current) {
      webgpuPendingRef.current = [c10Val, c12Val, phi12Val, rotationVal];
      webgpuPendingFullRef.current = true;
      return true;
    }
    const bfCount = fullBrightFieldCount();
    const total = Math.max(1, effectiveTotalBf || bfCount);
    const isFull = bfCount >= total;
    const request = {
      aberrations: { C10_nm: c10Val, C12_nm: c12Val, phi12_deg: phi12Val },
      higherOrderAberrations: higherOrderRef.current,
      scanDetectorRotationDeg: rotationVal,
      upsamplingFactor: upsamplingFactorRef.current,
      numBrightFieldPixels: bfCount,
      reducedBrightField: !isFull,
      computeLoss: false,
      sample: samplePreviewRef.current,
    };
    const phaseKey = JSON.stringify(request);
    if (webgpuPhaseKeyRef.current !== phaseKey) {
      // Show the last full-quality image before starting its diagnostic. If a
      // preview already produced this image, release does not reconstruct it.
      webgpuPendingRef.current = [c10Val, c12Val, phi12Val, rotationVal];
      webgpuPendingFullRef.current = true;
      runFrontendPreview(c10Val, c12Val, phi12Val, rotationVal);
      return true;
    }
    if (!isFull) return true;
    webgpuDiagnostic.schedule(signal => {
      if (webgpuSsbRef.current !== engine) return;
      webgpuInFlightRef.current = true;
      setBusy(true);
      reconstruct(engine, { ...request, computeLoss: true, diagnosticOnly: true, signal }).then(result => {
        if (signal.aborted || webgpuSsbRef.current !== engine || webgpuPhaseKeyRef.current !== phaseKey) return;
        setLoss(result.loss);
      }).catch(err => {
        if (signal.aborted) return;
        const message = err instanceof Error ? err.message : String(err);
        setWebgpuRuntimeStatus(`WebGPU loss failed: ${message}`);
      }).finally(drainWebgpuQueue);
    });
    return true;
  }, [effectiveTotalBf, runFrontendPreview, fullBrightFieldCount, webgpuStandalone, webgpuDiagnostic]);

  frontendFullRef.current = runFrontendFull;
  const previousUpsamplingFactorRef = React.useRef(upsamplingFactor);
  React.useEffect(() => {
    if (previousUpsamplingFactorRef.current === upsamplingFactor) return;
    previousUpsamplingFactorRef.current = upsamplingFactor;
    if (webgpuStandalone) frontendPreviewRef.current?.(sliderVals.current.c10, sliderVals.current.c12, sliderVals.current.phi12, rotationDegRef.current);
  }, [upsamplingFactor, webgpuStandalone]);
  shouldCommitOnReleaseRef.current = dragBfRef.current > 0 || !!webgpuSsbRef.current;
  const standaloneInitialRenderRef = React.useRef(false);
  React.useEffect(() => {
    if (!webgpuStandalone || standaloneInitialRenderRef.current) return;
    if (!webgpuSsbRef.current || !initRef.current) return;
    if (rawPhaseRef.current) return;
    standaloneInitialRenderRef.current = true;
    const total = Math.max(1, effectiveTotalBf || 1);
    const count = fullBrightFieldCount();
    setWebgpuLoadProgress({
      stage: "device",
      message: "Starting browser-side ptychography",
      detail: `Loading detector counts and preparing ${count}/${total} BF pixels`,
      current: 0,
      total: count,
      percent: 0,
      activeBf: count,
      totalBf: total,
    });
    setWebgpuRuntimeStatus("Preparing detector counts on WebGPU");
    frontendPreviewRef.current?.(autoC10, autoC12, autoPhi12, autoRotation ?? rotationDegRef.current);
  }, [autoC10, autoC12, autoPhi12, autoRotation, effectiveTotalBf, fullBrightFieldCount, webgpuRuntimeStatus, webgpuStandalone]);

  /* --- Re-render real-space display when mode/contrast/cmap changes.
         Coalesce to one render per animation frame: the histogram slider fires
         many times per drag, and each render kicks an async GPU colormap pass;
         without throttling those pile up and arrive interleaved, which makes the
         phase and histogram flicker/jitter while dragging. --- */
  const contrastRafRef = React.useRef<number | null>(null);
  const renderRealDisplayRef = React.useRef(renderRealDisplay);
  renderRealDisplayRef.current = renderRealDisplay;
  React.useEffect(() => {
    if (!rawPhaseRef.current || contrastRafRef.current !== null) return;
    contrastRafRef.current = requestAnimationFrame(() => {
      contrastRafRef.current = null;
      const latest = rawPhaseRef.current;
      if (latest) renderRealDisplayRef.current(latest.data, latest.w, latest.h);
    });
  }, [cmap, contrastRange, extraRealViews]);
  React.useEffect(() => {
    return () => {
      if (contrastRafRef.current !== null) cancelAnimationFrame(contrastRafRef.current);
    };
  }, []);

  /* --- Re-render FFT when its contrast or colormap changes (same rAF
         coalescing so the FFT histogram drag does not flicker). --- */
  const fftContrastRafRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    if (fftContrastRafRef.current !== null) cancelAnimationFrame(fftContrastRafRef.current);
    fftContrastRafRef.current = requestAnimationFrame(() => {
      fftContrastRafRef.current = null;
      rerenderFFT();
    });
    return () => {
      if (fftContrastRafRef.current !== null) cancelAnimationFrame(fftContrastRafRef.current);
    };
  }, [fftContrastRange, fftCmap, rerenderFFT]);

  /* --- When user turns FFT on or moves it between panel/inset, compute once. --- */
  React.useEffect(() => {
    if (!showFFT) return;
    if (fftMagRef.current) return;  // already have a current FFT
    const latest = rawPhaseRef.current;
    if (!latest) return;
    // Re-render for the same data, now with FFT path enabled (showFFTRef is already true).
    renderAll(latest.data, latest.w, latest.h);
  }, [showFFT, fftPlacement, renderAll]);

  /* --- Flip is a display-sign convention, not a new reconstruction.
         The live notebook path also receives a Python trait update, but the
         exported WebGPU HTML has no kernel observer. Negate the currently
         displayed phase immediately so both paths feel identical. --- */
  const flipInitializedRef = React.useRef(false);
  React.useEffect(() => {
    if (!flipInitializedRef.current) {
      flipInitializedRef.current = true;
      return;
    }
    const latest = rawPhaseRef.current;
    if (!latest) return;
    const flipped = new Float32Array(latest.data.length);
    for (let i = 0; i < latest.data.length; i++) flipped[i] = -latest.data[i];
    renderAllRef.current(flipped, latest.w, latest.h);
  }, [displayNegatePhase]);

  /* --- Render when new data arrives --- */
  React.useEffect(() => {
    let previous: DataView | null = null;
    let previousWidth = 0, previousHeight = 0;
    const receive = () => {
      const bytes = model.get("phase_bytes");
      const width = model.get("phase_width"), height = model.get("phase_height");
      if (!bytes || !width || !height) return;
      if (bytes === previous && width === previousWidth && height === previousHeight) return;
      const data = extractFloat32(bytes);
      // Keep the last complete image until the new dimensions and bytes agree.
      if (!data || data.length !== width * height) return;
      previous = bytes; previousWidth = width; previousHeight = height;
      renderAllRef.current(data, width, height);
    };
    model.on("change:phase_bytes change:phase_width change:phase_height", receive);
    receive();
    return () => model.off("change:phase_bytes change:phase_width change:phase_height", receive);
  }, [model]);

  /* --- Apply a finished tilt fit: aberration sliders + Sample panel.  The fit's C10 is the
         mid-depth defocus, so it only makes sense together with the fitted tilt/depth spread. --- */
  // An existing fit is the reset reference, not a new fit to apply on mount.
  // Reopening must preserve later manual edits in result_json/sample_json.
  const appliedSampleFitRef = React.useRef(model.get("sample_fit_json") || "");
  React.useEffect(() => {
    if (regionContextRef.current) return;
    if (!sampleFitJson || sampleFitJson === appliedSampleFitRef.current) return;
    appliedSampleFitRef.current = sampleFitJson;
    let fit: { C10_nm: number; C12_nm: number; phi12_deg: number; sample_tilt_mrad: [number, number]; depth_spread_nm: number };
    try { fit = JSON.parse(sampleFitJson); } catch { return; }
    const fittedSample = { tiltRowMrad: fit.sample_tilt_mrad[0], tiltColMrad: fit.sample_tilt_mrad[1], depthSpreadNm: fit.depth_spread_nm };
    sampleRef.current = fittedSample;
    adoptingPanelRef.current = true;
    setSample(fittedSample);
    setSampleOpen(true);
    setC10(fit.C10_nm); setC12(fit.C12_nm); setPhi12(fit.phi12_deg);
    sendCommit(fit.C10_nm, fit.C12_nm, fit.phi12_deg);
  }, [sampleFitJson, sendCommit, regionalMode]);

  const fitIdentity = JSON.stringify(activeFit?.fit_record || null);
  React.useEffect(() => {
    const context = regionalMode ? activeRegionId : '';
    const overview = JSON.parse(model.get('result_json') || '{}');
    const overviewSample = JSON.parse(model.get('sample_json') || '{}');
    setPlaying(false); pendingDragRef.current=null; setPreviewError('');
    if (context) setSweepParam('c10');
    const values = context ? (activeFit ? {
      c10:activeFit.aberrations.C10_nm,c12:activeFit.aberrations.C12_nm,phi12:activeFit.aberrations.phi12_deg,
      sample:{tiltRowMrad:activeFit.sample_tilt_mrad[0],tiltColMrad:activeFit.sample_tilt_mrad[1],depthSpreadNm:activeFit.depth_spread_nm},
      higher:{},loss:null,
    } : null) : {
      c10:overview.C10_nm ?? overviewAutoC10,c12:overview.C12_nm ?? overviewAutoC12,
      phi12:overview.phi12_deg ?? overviewAutoPhi12,
      sample:{tiltRowMrad:overviewSample.sample_tilt_mrad?.[0] || 0,
        tiltColMrad:overviewSample.sample_tilt_mrad?.[1] || 0,depthSpreadNm:overviewSample.depth_spread_nm || 0},
      higher:JSON.parse(model.get('higher_order_aberrations_json') || '{}'),loss:overview.loss ?? autoLoss,
    };
    if (!values) return;
    adoptingPanelRef.current=true;
    setUiRanges(previous => ({...previous,
      c10:[Math.min(previous.c10[0],Math.floor(values.c10)),Math.max(previous.c10[1],Math.ceil(values.c10))],
      c12:[Math.min(previous.c12[0],Math.floor(values.c12)),Math.max(previous.c12[1],Math.ceil(values.c12))],
      phi12:[Math.min(previous.phi12[0],Math.floor(values.phi12)),Math.max(previous.phi12[1],Math.ceil(values.phi12))],
    }));
    setC10(values.c10); setC12(values.c12); setPhi12(values.phi12);
    setSample(values.sample); setHigherOrder(values.higher); setLoss(values.loss);
  }, [regionalMode, activeRegionId, fitIdentity]);

  const doReset = () => {
    // auto values arrive asynchronously from Python; skip until init landed
    if (!initRef.current) return;
    setC10(autoC10); setC12(autoC12); setPhi12(autoPhi12);
    if (regionalMode && regionalReference) {
      const restored={tiltRowMrad:regionalReference.sample_tilt_mrad[0],tiltColMrad:regionalReference.sample_tilt_mrad[1],depthSpreadNm:regionalReference.depth_spread_nm};
      sampleRef.current=restored; adoptingPanelRef.current=true; setSample(restored);
    } else {
      const fittedSample=JSON.parse(fittedSampleJson || '{}');
      if(fittedSample.sample_tilt_mrad) {
        const restored={tiltRowMrad:fittedSample.sample_tilt_mrad[0],tiltColMrad:fittedSample.sample_tilt_mrad[1],depthSpreadNm:fittedSample.depth_spread_nm};
        sampleRef.current=restored; adoptingPanelRef.current=true; setSample(restored);
      }
      if(autoRotation != null) setRotationDeg(autoRotation);
    }
    sendCommit(autoC10, autoC12, autoPhi12, autoRotation ?? rotationDegRef.current);
  };
  const resetZoom = () => {
    setPlaying(false);
    setViewPin(null);
    resetCrop();
    setPhaseZoom(ZOOM_RESET);
    setAmpZoom(ZOOM_RESET);
    setComplexZoom(ZOOM_RESET);
    setFFTZoom(ZOOM_RESET);
    localTilt?.resetView();
  };

  const capturePinLayout = React.useCallback(() => {
    const rects = new Map<number, DOMRect>();
    rootRef.current!.querySelectorAll<HTMLElement>("[data-showptycho-pin-id]").forEach(element => {
      const id = Number(element.getAttribute("data-showptycho-pin-id"));
      if (Number.isFinite(id)) rects.set(id, element.getBoundingClientRect());
    });
    pinLayoutBeforeRef.current = rects;
  }, []);

  React.useLayoutEffect(() => {
    const before = pinLayoutBeforeRef.current;
    if (!before) return;
    pinLayoutBeforeRef.current = null;
    window.requestAnimationFrame(() => {
      rootRef.current?.querySelectorAll<HTMLElement>("[data-showptycho-pin-id]").forEach(element => {
        const id = Number(element.getAttribute("data-showptycho-pin-id"));
        const previous = before.get(id);
        if (!previous && id === newPinId) {
          element.animate(
            [
              { opacity: 0, transform: "scale(0.82)" },
              { opacity: 1, transform: "scale(1.06)" },
              { opacity: 1, transform: "scale(1)" },
            ],
            { duration: 320, easing: "cubic-bezier(0.2, 0, 0, 1)" },
          );
          return;
        }
        if (!previous) return;
        const next = element.getBoundingClientRect();
        const dx = previous.left - next.left;
        const dy = previous.top - next.top;
        if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) return;
        element.animate(
          [
            { transform: `translate(${dx}px, ${dy}px) scale(1)`, zIndex: 2 },
            { transform: "translate(0, 0) scale(1)", zIndex: 2 },
          ],
          { duration: 180, easing: "cubic-bezier(0.2, 0, 0, 1)" },
        );
      });
    });
  }, [newPinId, pinned]);

  /* --- Pin --- */
  const doPin = React.useCallback(() => {
    if (regionContextRef.current) return;
    const phase = rawPhaseRef.current;
    if (!phase || busy) return;
    capturePinLayout();
    const entry: PinnedEntry = {
      id: Date.now(),
      C10_nm: sliderVals.current.c10,
      C12_nm: sliderVals.current.c12,
      phi12_deg: sliderVals.current.phi12,
      // the pin message and the manifest hold the physical angle in [0, 360); the slider may run below 0
      scan_detector_rotation_deg: wrapDegrees(rotationDeg ?? 0),
      display_negate_phase: !!displayNegatePhase,
      loss: loss ?? 0,
      starred: false,
      generated_at_utc: new Date().toISOString(),
      phaseData: new Float32Array(phase.data),
      w: phase.w, h: phase.h,
      thumb: makeThumbnail(phase.data, phase.w, phase.h, cmapRef.current),
    };
    setPinned(prev => {
      const next = [...prev, entry];
      if (next.length > MAX_PINS) next.shift();
      return next;
    });
    setNewPinId(entry.id);
    window.setTimeout(() => setNewPinId(current => current === entry.id ? null : current), 700);
    setPinJson(JSON.stringify({
      action: "pin", id: entry.id,
      C10_nm: entry.C10_nm, C12_nm: entry.C12_nm, phi12_deg: entry.phi12_deg,
      scan_detector_rotation_deg: entry.scan_detector_rotation_deg, display_negate_phase: entry.display_negate_phase,
      loss: entry.loss, generated_at_utc: entry.generated_at_utc,
    }));
  }, [busy, capturePinLayout, loss, rotationDeg, displayNegatePhase, setPinJson]);

  React.useEffect(() => {
    setPinned(prev => {
      if (prev.length === 0) return prev;
      return prev.map(pin => ({
        ...pin,
        thumb: makeThumbnail(pin.phaseData, pin.w, pin.h, cmap),
      }));
    });
  }, [cmap]);

  const doFolderSave = React.useCallback(async () => {
    const phase = rawPhaseRef.current;
    if (!phase || busy) return;
    try {
      const lut = COLORMAPS[cmapRef.current as keyof typeof COLORMAPS] || COLORMAPS.viridis;
      const { canvas } = renderPhaseOffscreen(phase.data, phase.w, phase.h, lut, 1, 99);
      if (!canvas) throw new Error("no phase canvas");
      const jpeg: Blob = await new Promise((resolve, reject) => {
        canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("JPEG encode failed"))), "image/jpeg", 0.92);
      });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const record: FolderSaveRecord = {
        id: stamp,
        generated_at_utc: new Date().toISOString(),
        C10_nm: sliderVals.current.c10,
        C12_nm: sliderVals.current.c12,
        phi12_deg: sliderVals.current.phi12,
        scan_detector_rotation_deg: wrapDegrees(rotationDeg ?? 0),
        display_negate_phase: !!displayNegatePhase,
        loss: loss ?? null,
        num_bright_field_pixels: Math.round(dragBfRef.current || 0),
        notes: String(notes ?? ""),
        image: `snapshots/snapshot_${stamp}_C10_${slugPart(sliderVals.current.c10.toFixed(0))}.jpg`,
      };
      const next = [...folderSaves, record];
      if (ssbFolderWritable()) {
        await writeSSBFolderFile(record.image, jpeg);
        await writeSSBFolderFile("snapshots/snapshots.json", JSON.stringify({ schema: SNAPSHOTS_SCHEMA, snapshots: next }, null, 2));
        setFolderSaves(next);
        setFolderSaveStatus(`Snapshot saved (${next.length})`);
      } else {
        setFolderSaveStatus("Folder is read-only here - use Open data folder for persistent snapshots");
      }
    } catch (err) {
      setFolderSaveStatus(`Save failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [busy, folderSaves, loss, notes, rotationDeg, displayNegatePhase]);
  const loadFolderSave = React.useCallback((record: FolderSaveRecord) => {
    setC10(record.C10_nm);
    setC12(record.C12_nm);
    setPhi12(record.phi12_deg);
    // the file holds the angle in [0, 360); the slider takes the turn of it inside its own range
    const rotation = degreesWithin(record.scan_detector_rotation_deg, rotationUiMin, rotationUiMax);
    setRotationDeg(rotation);
    setDisplayNegatePhase(record.display_negate_phase);
    sliderVals.current = { c10: record.C10_nm, c12: record.C12_nm, phi12: record.phi12_deg };
    frontendFullRef.current?.(record.C10_nm, record.C12_nm, record.phi12_deg, rotation);
  }, [setRotationDeg, setDisplayNegatePhase, rotationUiMin, rotationUiMax]);
  const downloadFolderSave = React.useCallback(async (record: FolderSaveRecord) => {
    try {
      const bytes = await readSSBFolderBytes(record.image);
      downloadBlob(new Blob([bytes as unknown as BlobPart], { type: "image/jpeg" }), record.image.split("/").pop() || "save.jpg");
    } catch (err) {
      setFolderSaveStatus(`Download failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, []);
  const deleteFolderSave = React.useCallback(async (record: FolderSaveRecord) => {
    try {
      const next = folderSaves.filter((save) => save.id !== record.id);
      await writeSSBFolderFile("snapshots/snapshots.json", JSON.stringify({ schema: SNAPSHOTS_SCHEMA, snapshots: next }, null, 2));
      try { await deleteSSBFolderFile(record.image); } catch { /* image may already be gone */ }
      setFolderSaves(next);
      setFolderSaveStatus(`Deleted snapshot (${next.length} left)`);
    } catch (err) {
      setFolderSaveStatus(`Delete failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [folderSaves]);

  const doExportPinned = React.useCallback(async () => {
    if (pinned.length === 0) return;
    setExportStatus(`Exporting ${pinned.length}`);
    try {
      const lut = COLORMAPS[cmapRef.current as keyof typeof COLORMAPS] || COLORMAPS.viridis;
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const manifest = {
        exported_at: new Date().toISOString(),
        colormap: cmapRef.current,
        contrast_percentiles: [1, 99],
        count: pinned.length,
        pins: pinned.map((pin, i) => ({
          index: i + 1,
          filename: `showptycho_pin_${String(i + 1).padStart(2, "0")}_C10_${slugPart(pin.C10_nm.toFixed(0))}_C12_${slugPart(pin.C12_nm.toFixed(0))}_phi_${slugPart(pin.phi12_deg.toFixed(0))}.png`,
          C10_nm: pin.C10_nm,
          C12_nm: pin.C12_nm,
          phi12_deg: pin.phi12_deg,
          scan_detector_rotation_deg: wrapDegrees(pin.scan_detector_rotation_deg),
          display_negate_phase: pin.display_negate_phase,
          loss: pin.loss,
          starred: pin.starred,
          generated_at_utc: pin.generated_at_utc,
          shape: [pin.h, pin.w],
        })),
      };

      let exported = 0;
      for (const [i, pin] of pinned.entries()) {
        const { canvas } = renderPhaseOffscreen(pin.phaseData, pin.w, pin.h, lut, 1, 99);
        if (!canvas) continue;
        const blob = await canvasToPngBlob(canvas);
        downloadBlob(blob, manifest.pins[i].filename);
        exported += 1;
        await new Promise(resolve => window.setTimeout(resolve, 80));
      }
      const manifestName = `showptycho_pins_${stamp}.json`;
      downloadBlob(
        new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" }),
        manifestName,
      );
      setExportStatus(`Exported ${exported} PNG + JSON`);
    } catch (err) {
      setExportStatus(`Export failed ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [pinned]);

  const reorderPinned = React.useCallback((fromId: number, toId: number) => {
    if (fromId === toId) return;
    capturePinLayout();
    setPinned(prev => {
      const from = prev.findIndex(pin => pin.id === fromId);
      const to = prev.findIndex(pin => pin.id === toId);
      if (from < 0 || to < 0 || from === to) return prev;
      const next = [...prev];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });
  }, [capturePinLayout]);

  const pinIdFromPoint = React.useCallback((clientX: number, clientY: number) => {
    const element = document.elementFromPoint(clientX, clientY)?.closest("[data-showptycho-pin-id]");
    const raw = element?.getAttribute("data-showptycho-pin-id");
    const id = raw ? Number(raw) : NaN;
    return Number.isFinite(id) ? id : null;
  }, []);

  const beginPinDrag = React.useCallback((id: number, clientX: number, clientY: number) => {
    pinPointerDragRef.current = { id, x: clientX, y: clientY, active: false };
  }, []);

  const updatePinDrag = React.useCallback((id: number, clientX: number, clientY: number) => {
    const drag = pinPointerDragRef.current;
    if (!drag || drag.id !== id) return false;
    const dx = clientX - drag.x;
    const dy = clientY - drag.y;
    if (!drag.active && Math.hypot(dx, dy) < 6) return false;
    drag.active = true;
    setDraggingPin(id);
    didDragPinRef.current = true;
    setDragOverPin(pinIdFromPoint(clientX, clientY));
    return true;
  }, [pinIdFromPoint]);

  const finishPinDrag = React.useCallback((id: number, clientX: number, clientY: number) => {
    const drag = pinPointerDragRef.current;
    if (!drag || drag.id !== id) return;
    const overId = pinIdFromPoint(clientX, clientY);
    if (drag.active && overId != null) reorderPinned(drag.id, overId);
    pinPointerDragRef.current = null;
    setDraggingPin(null);
    setDragOverPin(null);
    if (drag.active) window.setTimeout(() => { didDragPinRef.current = false; }, 0);
  }, [pinIdFromPoint, reorderPinned]);

  const cancelPinDrag = React.useCallback(() => {
    pinPointerDragRef.current = null;
    setDraggingPin(null);
    setDragOverPin(null);
    window.setTimeout(() => { didDragPinRef.current = false; }, 0);
  }, []);

  const doUnpin = React.useCallback((id: number) => {
    capturePinLayout();
    setPinned(prev => prev.filter(pin => pin.id !== id));
    setViewPin(current => current === id ? null : current);
    setPinJson(JSON.stringify({ action: "unpin", id }));
  }, [capturePinLayout, setPinJson]);

  // Toggle star on a pin; Python persists the new state to disk. The new state comes from the
  // rendered pins, not from inside a setPinned updater, which React may run later than this call.
  const doToggleStar = React.useCallback((id: number) => {
    const starred = !pinned.find(entry => entry.id === id)?.starred;
    setPinned(prev => prev.map(entry => entry.id === id ? { ...entry, starred } : entry));
    setPinJson(JSON.stringify({ action: starred ? "star" : "unstar", id }));
  }, [pinned, setPinJson]);

  const doViewPin = React.useCallback((entry: PinnedEntry) => {
    setViewPin(entry.id);
    setActiveTrialRank(null);  // pins + trials are mutually exclusive selections
    setC10(entry.C10_nm); setC12(entry.C12_nm); setPhi12(entry.phi12_deg);
    setLoss(entry.loss);
    renderAll(entry.phaseData, entry.w, entry.h);
  }, [renderAll]);

  /* --- Keyboard: shortcuts act only while this widget has focus --- */
  // Scoped to the root (as in Show2D/Show3D) so typing in notebook cells, CodeMirror
  // editors and other widgets is never intercepted.
  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (shouldIgnoreWidgetShortcut(e.target, e.key)) return;
    let handled = true;
    if (e.key === "r" || e.key === "R") resetZoom();
    // pins, sweeps and trials belong to an SSB session
    else if (!tuning) handled = false;
    // P pins the current snapshot (the oldest is evicted once MAX_PINS is hit)
    else if (e.key === "p" || e.key === "P") doPin();
    else if (e.key === " ") setPlaying(wasPlaying => !wasPlaying);
    else if (e.key === "Escape") { setViewPin(null); setActiveTrialRank(null); }
    // S stars or unstars the viewed pin (Python persists it to disk)
    else if ((e.key === "s" || e.key === "S") && viewPin != null) doToggleStar(viewPin);
    else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && (trials.length > 0 || pinned.length > 0)) {
      // arrows step through whichever browser was touched last: clicking a trial tile sets
      // activeTrialRank, clicking a pin sets viewPin, and each clears the other
      const step = e.key === "ArrowRight" ? 1 : -1;
      if (activeTrialRank != null && trials.length > 0) {
        const trialIndex = trials.findIndex(trial => trial.rank === activeTrialRank);
        doViewTrial(trials[trialIndex < 0 ? (step > 0 ? 0 : trials.length - 1) : (trialIndex + step + trials.length) % trials.length]);
        setTrialsExpanded(true);  // so the selected tile is visible
      } else if (pinned.length > 0) {
        const pinIndex = viewPin ? pinned.findIndex(pin => pin.id === viewPin) : -1;
        doViewPin(pinned[pinIndex < 0 ? (step > 0 ? 0 : pinned.length - 1) : (pinIndex + step + pinned.length) % pinned.length]);
      } else {
        doViewTrial(trials[0]);
        setTrialsExpanded(true);
      }
    } else handled = false;
    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  const handleRootMouseDownCapture = (e: React.MouseEvent<HTMLDivElement>) => {
    if (shouldIgnoreWidgetShortcut(e.target)) return;
    rootRef.current?.focus({ preventScroll: true });
  };

  /* --- Resize handle --- */
  React.useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (resizeDrag.current.on) {
        setPanel(Math.max(MIN_PANEL, resizeDrag.current.startSize + e.clientY - resizeDrag.current.startY));
      }
    };
    const handleUp = () => { resizeDrag.current.on = false; };
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, []);

  const startResize = React.useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    resizeDrag.current = { on: true, startY: e.clientY, startSize: shownPanelRef.current };
  }, []);

  /* --- Scale bar: reciprocal pixel size for FFT --- */
  const fftPixelSize = React.useMemo(() => {
    const fft = fftMagRef.current;
    if (!fft || !pixelSize || pixelSize <= 0) return 0;
    // Reciprocal pixel size of the complete padded Fourier grid.
    return 1.0 / (fft.w * pixelSize);
  }, [pixelSize, fftOff]);

  /* --- Derived --- */
  const delta = loss != null && autoLoss > 0 ? loss - autoLoss : null;
  const fftAsPanel = showFFT && fftPlacement === "panel";
  const fftAsInset = showFFT && fftPlacement === "inset";
  const realPanelCount = 1 + (extraRealViews.amp ? 1 : 0) + (extraRealViews.complex ? 1 : 0);
  const localPanelCount = localTiltEnabled ? JSON.parse(localImagesJson || "[]").filter((image:any)=>{
    const fit=localConfig.fits?.find((fit:any)=>`${fit.label} · SSB`===image.label);
    return !fit||selectedRegions.some((region:any)=>region.id===fit.id);
  }).length : 0;
  const visiblePanelCount = realPanelCount + (fftAsPanel ? 1 : 0) + localPanelCount;
  // Panels shrink to share the notebook column (the resize handle sets the largest size);
  // below MIN_PANEL each they wrap onto the next row instead.
  const fitWidth = availableWidth > 0 ? availableWidth : Infinity;
  // each image box draws a 1 px border outside its size
  const columns = localTiltEnabled ? Math.max(1,Math.min(maxColumns||3,visiblePanelCount,
    Math.floor((fitWidth+PANEL_GAP)/(MIN_PANEL+2+PANEL_GAP)))) : visiblePanelCount;
  const perPanel = Math.floor((fitWidth - PANEL_GAP * Math.max(0, columns - 1)) / columns) - 2;
  const shownPanel = Math.max(Math.min(MIN_PANEL, fitWidth), Math.min(panel, perPanel));
  shownPanelRef.current = shownPanel;
  const totalW = (shownPanel+2) * columns + PANEL_GAP * Math.max(0, columns - 1);
  const fftInsetSize = Math.max(96, Math.min(180, Math.round(shownPanel * 0.28)));
  const loadProgressPercent = webgpuLoadProgress ? webgpuProgressPercent(webgpuLoadProgress) : null;
  const loadProgressIsError = webgpuLoadProgress?.stage === "error";
  const runtimeStatusLabel = webgpuRuntimeStatus ? compactRuntimeStatus(webgpuRuntimeStatus) : "";

  // Themed select styling
  const themedSelect = {
    color: tc.text,
    "& .MuiSelect-icon": { color: tc.textMuted },
    "& .MuiSelect-select, & .MuiSelect-select.MuiSelect-outlined": {
      textOverflow: "clip !important",
      overflow: "visible !important",
    },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: tc.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: tc.accent },
    "&.Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: tc.accent },
  };
  const actionSelect = {
    ...themedSelect,
    ...typography.labelSmall,
    height: ACTION_CONTROL_HEIGHT,
    "& .MuiSelect-select": {
      ...themedSelect["& .MuiSelect-select, & .MuiSelect-select.MuiSelect-outlined"],
      minHeight: 0,
      py: 0,
      display: "flex",
      alignItems: "center",
    },
  };
  const themedMenuPaper = {
    sx: {
      bgcolor: tc.bgAlt,
      color: tc.text,
      border: `1px solid ${tc.border}`,
      "& .MuiMenuItem-root": {
        ...typography.label,
        minHeight: 28,
      },
    },
  };
  const themedMenuProps = { slotProps: { paper: themedMenuPaper } };
  const rangeInputSx = {
    width: 66,
    height: 22,
    boxSizing: "border-box",
    fontFamily: "monospace",
    fontSize: 10,
    color: tc.text,
    bgcolor: tc.bg,
    border: `1px solid ${tc.border}`,
    borderRadius: 0,
    px: 0.5,
    outline: "none",
    "&:focus": { borderColor: tc.accent },
  };
  const defaultUiRanges: MainRanges = {
    c10: [c10Min, c10Max],
    c12: [Math.min(C12_RANGE[0], Math.floor(autoC12)), Math.max(C12_RANGE[1], Math.ceil(autoC12))],
    phi12: PHI12_RANGE,
    rot: [Math.floor(rotationMin), Math.ceil(rotationMax)],
  };
  const rangeStep = React.useCallback((key: AberKey) => (key === "rot" ? 0.1 : 1), []);
  const setUiRangeBound = React.useCallback((key: AberKey, side: 0 | 1, raw: string) => {
    const value = Number(raw);
    if (!Number.isFinite(value)) return;
    setUiRanges(prev => {
      const next: MainRanges = { ...prev };
      const current: [number, number] = [...prev[key]];
      const gap = rangeStep(key);
      if (side === 0) current[0] = Math.min(value, current[1] - gap);
      else current[1] = Math.max(value, current[0] + gap);
      next[key] = current;
      return next;
    });
  }, [rangeStep]);
  const resetUiRange = React.useCallback((key: AberKey) => {
    setUiRanges(prev => ({ ...prev, [key]: defaultUiRanges[key] }));
  }, [defaultUiRanges]);
  const resetAllUiRanges = React.useCallback(() => {
    setUiRanges(defaultUiRanges);
  }, [defaultUiRanges]);
  const editedRangeCount = (Object.keys(uiRanges) as AberKey[]).filter(key => (
    uiRanges[key][0] !== defaultUiRanges[key][0] || uiRanges[key][1] !== defaultUiRanges[key][1]
  )).length;
  const renderRangeEditor = (key: AberKey, label: string, unit: string, decimals = 0) => (
    <Box
      key={key}
      sx={{
        display: "grid",
        gridTemplateColumns: "42px auto auto auto",
        gap: 0.75,
        alignItems: "center",
      }}
    >
      <Typography sx={{ ...typography.labelSmall, color: tc.text, fontWeight: 700 }}>{label}</Typography>
      <Box component="input" type="number" step={rangeStep(key)} value={uiRanges[key][0].toFixed(decimals)} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUiRangeBound(key, 0, e.target.value)} sx={rangeInputSx} aria-label={`${label} minimum`} />
      <Box component="input" type="number" step={rangeStep(key)} value={uiRanges[key][1].toFixed(decimals)} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUiRangeBound(key, 1, e.target.value)} sx={rangeInputSx} aria-label={`${label} maximum`} />
      <IconButton size="small" onClick={() => resetUiRange(key)} disabled={uiRanges[key][0] === defaultUiRanges[key][0] && uiRanges[key][1] === defaultUiRanges[key][1]} sx={compactBareIconButton(tc)} aria-label={`Reset ${label} range`}>
        <RestartAltIcon sx={{ fontSize: 14 }} />
      </IconButton>
      <Typography sx={{ ...typography.labelSmall, color: tc.textMuted, gridColumn: "2 / span 2", mt: -0.75 }}>
        {unit}
      </Typography>
    </Box>
  );

  const localTilt = useLocalTiltWorkspace({ Panel: ImagePanel, tc,
    ui: { button: compactButton(tc), select: actionSelect, menu: themedMenuProps, typography },
    cmap, smooth, contrastRange, maxColumns:maxColumns||3, setMaxColumns,
    overviewSize: shownPanel, zoom: phaseZoom, setZoom: setPhaseZoom,
    onEnable: () => { setCropSelecting(false); setFFTPlacement("inset"); },
    closeSettings: () => setSettingsAnchor(null),
    available: !tuning || !localTiltJson || JSON.stringify(JSON.parse(localTiltJson).scan_shape) === JSON.stringify([scanRows, scanCols]) });

  return (
    <Box
      ref={rootRef}
      className="showptycho-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
      sx={{ ...container.root, bgcolor: tc.bg, color: tc.text, outline: "none" }}
    >
      {/* ---- Top toolbar: views, image dimensions and display settings ---- */}
      <Stack
        direction="row"
        alignItems="center"
        spacing={`${SPACING.SM}px`}
        sx={{
          mb: `${SPACING.XS}px`, minHeight: 28, height: "auto", width: "100%",
          flexWrap: "wrap", rowGap: `${SPACING.XS}px`,
        }}
      >
        {tuning && (sampleAvailable || webgpuStandalone) && (
          <Tooltip title="Compare with zero sample tilt while retaining the fitted depth spread and aberrations. Save keeps the fitted tilt for reconstruction." placement="top" arrow>
            <Box sx={{ display: "inline-flex", alignItems: "center", whiteSpace: "nowrap", gap: 0.5 }}>
              <Typography sx={{ ...typography.label, color: tc.text }}>Tilt correction</Typography>
              <Switch
                checked={tiltCorrection !== false}
                disabled={busy || sample.depthSpreadNm <= 0}
                onChange={e => setTiltCorrection(e.target.checked)}
                slotProps={{ input: { "aria-label": "Tilt correction" } }}
                size="small" sx={switchStyles.small}
              />
              <Typography sx={{ ...typography.value, color: tc.textMuted }}>
                {sample.depthSpreadNm <= 0 ? "Not fitted" : tiltCorrection === false ? "Off · preview" : "On"}
              </Typography>
            </Box>
          </Tooltip>
        )}
        {objectAmplitude && <>
        <Button
          size="small"
          onClick={() => setExtraRealViews(prev => ({ ...prev, amp: !prev.amp }))}
          aria-pressed={extraRealViews.amp}
          title="Toggle amplitude view"
          sx={{ ...compactButton(tc), color: extraRealViews.amp ? tc.accent : tc.text }}
        >
          Amp
        </Button>
        <Button
          size="small"
          onClick={() => setExtraRealViews(prev => ({ ...prev, complex: !prev.complex }))}
          aria-pressed={extraRealViews.complex}
          title="Toggle complex view"
          sx={{ ...compactButton(tc), color: extraRealViews.complex ? tc.accent : tc.text }}
        >
          Complex
        </Button>
        </>}
        {localTilt?.toggle}
        <Box sx={{ flex: 1 }} />
        {(rawPhaseRef.current?.w || phaseWidth) > 0 && (rawPhaseRef.current?.h || phaseHeight) > 0 && (
          <Typography sx={{ ...typography.value, color: tc.textMuted, opacity: 0.7 }}>
            {rawPhaseRef.current?.w || phaseWidth}×{rawPhaseRef.current?.h || phaseHeight}
          </Typography>
        )}
        {(cropSelecting || scanCrop) && (
          <Typography sx={{ ...typography.value, fontVariantNumeric: "tabular-nums", color: scanCrop ? STATUS_GOOD : tc.textMuted }}>
            {cropSummary}
          </Typography>
        )}
        {(cropRefitStatus || "").startsWith("Refit complete:") && (
          <Typography sx={{ ...typography.value, color: STATUS_GOOD }}>SSB refit complete</Typography>
        )}
        <Badge variant="dot" invisible={!showFFT && !cropSelecting && !scanCrop && editedRangeCount === 0}
          sx={{ "& .MuiBadge-badge": { bgcolor: tc.accent, right: 4, top: 4 } }}>
          <IconButton size="small" aria-label="View settings" title="View settings"
            aria-expanded={Boolean(settingsAnchor)} aria-haspopup="dialog"
            onClick={(event) => setSettingsAnchor(event.currentTarget)}
            sx={{ ...compactIconButton(tc, showFFT || cropSelecting || scanCrop || editedRangeCount > 0 ? tc.accent : tc.textMuted), width: 28, height: 28 }}>
            <SettingsOutlinedIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Badge>
        <Popover open={Boolean(settingsAnchor)} anchorEl={settingsAnchor} action={settingsPopover}
          onClose={() => setSettingsAnchor(null)}
          anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
          transformOrigin={{ vertical: "top", horizontal: "right" }}
          slotProps={{ paper: { ...themedMenuPaper, sx: { ...themedMenuPaper.sx, p: 1.25, maxWidth: "calc(100vw - 32px)" } } }}>
          <Box role="dialog" aria-label="View settings"
            sx={{ display: "flex", flexDirection: "column", gap: 1, fontFamily: WIDGET_UI_FONT, width: 280, maxWidth: "100%" }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <Box component="label" sx={{ ...typography.label, display: "inline-flex", alignItems: "center", gap: .5 }}>
                Show FFT
                <Switch checked={showFFT} onChange={(_, checked) => setShowFFT(checked)}
                  slotProps={{ input: { "aria-label": "Show FFT" } }} size="small" sx={switchStyles.small} />
              </Box>
              <Select value={fftPlacement} onChange={(event) => setFFTPlacement(event.target.value as FFTPlacement)}
                disabled={!showFFT} size="small" sx={{ ...actionSelect, minWidth: 116 }}
                MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT placement" }}>
                <MenuItem value="inset">Inset</MenuItem>
                <MenuItem value="panel">Side by Side</MenuItem>
              </Select>
            </Box>
            {showFFT && <>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Box component="label" sx={{ ...typography.label, display: "inline-flex", alignItems: "center", gap: .5 }}>
                  Color
                  <Select value={fftCmap} onChange={(event) => setFftCmap(event.target.value)}
                    size="small" sx={{ ...themedSelect, ...typography.labelSmall, minWidth: 92 }}
                    MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT color" }}>
                    {COLORMAP_NAMES.map(name => <MenuItem key={name} value={name}>
                      {name.charAt(0).toUpperCase() + name.slice(1)}
                    </MenuItem>)}
                  </Select>
                </Box>
                <Box component="label" sx={{ ...typography.label, display: "inline-flex", alignItems: "center", gap: .5 }}>
                  Auto
                  <Switch checked={fftAutoContrast}
                    onChange={(_, checked) => { setFftAutoContrast(checked); if (checked) setFftContrastRange([1, 99]); }}
                    slotProps={{ input: { "aria-label": "FFT auto contrast" } }} size="small" sx={switchStyles.small} />
                </Box>
              </Box>
              <Histogram frame={panelFrames[1]}
                vminPct={fftContrastRange[0]} vmaxPct={fftContrastRange[1]}
                onRangeChange={(lo, hi) => { setFftContrastRange([lo, hi]); setFftAutoContrast(false); }}
                width={240} height={48} tc={tc} />
            </>}
            {localTilt?.enabled && <Box component="details" onToggle={() => settingsPopover.current?.updatePosition()}
              sx={{ borderTop: `1px solid ${tc.border}`, pt: 1 }}>
              <Box component="summary" sx={{ ...typography.label, cursor: "pointer", color: tc.text }}>Local Tilt</Box>
              {localTilt.settings}
            </Box>}
            {tuning && <Box component="details" onToggle={() => settingsPopover.current?.updatePosition()}
              sx={{ borderTop: `1px solid ${tc.border}`, pt: 1 }}>
              <Box component="summary" sx={{ ...typography.label, cursor: "pointer", color: editedRangeCount ? tc.accent : tc.text }}>Sweep Ranges{editedRangeCount ? ` (${editedRangeCount})` : ""}</Box>
              <Box sx={{ display: "flex", flexDirection: "column", gap: .75, pt: 1 }}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <Button size="small" onClick={resetAllUiRanges} disabled={editedRangeCount === 0} sx={{ ...compactButton(tc), minHeight: 22 }}>
                  Reset
                </Button>
              </Stack>
              <Box sx={{ display: "grid", gridTemplateColumns: "42px 66px 66px 22px", gap: 0.75, alignItems: "center" }}>
                <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Param</Typography>
                <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Min</Typography>
                <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Max</Typography>
              </Box>
              {renderRangeEditor("c10", "C10", "nm")}
              {renderRangeEditor("c12", "C12", "nm")}
              {renderRangeEditor("phi12", "φ₁₂", "deg")}
              {renderRangeEditor("rot", "rot", "deg", 1)}
              </Box>
            </Box>}
            {tuning && cropRefitAvailable && <Box component="details" onToggle={() => settingsPopover.current?.updatePosition()}
              sx={{ borderTop: `1px solid ${tc.border}`, pt: 1 }}>
              <Box component="summary" sx={{ ...typography.label, cursor: "pointer", color: cropSelecting || scanCrop ? tc.accent : tc.text }}>SSB Crop</Box>
              <Box component="label" sx={{ ...typography.label, display: "flex", alignItems: "center", justifyContent: "space-between", mt: 1 }}>
                Draw Crop Region
                <Switch checked={cropSelecting} size="small" sx={switchStyles.small}
                  slotProps={{ input: { "aria-label": "Draw crop region" } }}
                  onChange={() => { setCropSelecting(value => !value); setLocalTiltEnabled(false); setSettingsAnchor(null); }} />
              </Box>
              {(cropSelecting || scanCrop) && <Typography sx={{ ...typography.value, color: tc.textMuted }}>{cropSummary}</Typography>}
              <Box sx={{ display: "flex", gap: 1, mt: .5 }}>
                <Button size="small" sx={compactButton(tc)} disabled={!scanCrop && !cropSelecting}
                  onClick={() => { resetCrop(); setSettingsAnchor(null); }}>Crop Reset</Button>
                <Button size="small" sx={compactButton(tc)} disabled={busy || cropRefitPending || !scanCrop}
                  title="Refit the crop with the default fit, model-based and tilt-corrected"
                  onClick={() => { requestCropRefit(); setSettingsAnchor(null); }}>Refit SSB (200 trials)</Button>
              </Box>
            </Box>}
          </Box>
        </Popover>
        <Button
          size="small"
          variant="outlined"
          onClick={resetZoom}
          title="Return to the full field of view without changing the fitted model."
          disabled={!localTilt?.canReset && !scanCrop && !cropSelecting && viewPin == null &&
            [phaseZoom,ampZoom,complexZoom,fftZoom].every(view=>view.zoom===1 && view.panX===0 && view.panY===0)}
          sx={{ ...compactButton(tc), width: 52, minWidth: 52, color: tc.textMuted, borderColor: tc.border }}
        >
          Reset
        </Button>
      </Stack>

      {localTilt?.toolbar}

      {/* ---- Two panels ---- */}
      <Box data-showptycho-images sx={{display:localTiltEnabled?'grid':'flex',flexWrap:'wrap',gap:`${PANEL_GAP}px`,
        gridTemplateColumns:`repeat(${columns}, ${shownPanel+2}px)`,maxWidth:'100%'}}>
        <ImagePanel
          offscreen={phaseOff} label={localTiltEnabled ? "Overview" : "Phase"} size={shownPanel} tc={tc}
          zoom={phaseZoom} onZoomChange={setPhaseZoom}
          showResize onResizeStart={startResize}
          pixelSize={pixelSize} imageWidth={rawPhaseRef.current?.w}
          rawData={phaseFrameData}
          smooth={smooth}
          cropRegion={cropRefitAvailable ? displayedScanCrop : null}
          cropSelecting={cropRefitAvailable && cropSelecting}
          onCropChange={chooseCropRectangle}
          inset={<>{localTilt?.overlay}{fftAsInset ? (
            <FFTInset
              offscreen={fftOff}
              rawData={fftFrameData}
              size={fftInsetSize}
              panelSize={shownPanel}
              box={fftInsetBox}
              onBoxChange={setFFTInsetBox}
              tc={tc}
              smooth={smooth}
              pixelSize={pixelSize}
            />
          ) : null}</>}
        />
        {extraRealViews.amp && (
          <ImagePanel
            offscreen={ampOff} label="Amplitude" size={shownPanel} tc={tc}
            zoom={ampZoom} onZoomChange={setAmpZoom}
            showResize onResizeStart={startResize}
            pixelSize={pixelSize} imageWidth={rawPhaseRef.current?.w}
            rawData={ampFrameData}
            smooth={smooth}
          />
        )}
        {extraRealViews.complex && (
          <ImagePanel
            offscreen={complexOff} label="Complex" size={shownPanel} tc={tc}
            zoom={complexZoom} onZoomChange={setComplexZoom}
            showResize onResizeStart={startResize}
            pixelSize={pixelSize} imageWidth={rawPhaseRef.current?.w}
            rawData={ampDataRef.current?.data}
            smooth={smooth}
          />
        )}
        {fftAsPanel && (
          <ImagePanel
            offscreen={fftOff} label="FFT" size={shownPanel} tc={tc}
            zoom={fftZoom} onZoomChange={setFFTZoom}
            showResize onResizeStart={startResize}
            pixelSize={fftPixelSize} realSpacePixelSize={pixelSize}
            imageWidth={fftMagRef.current?.w} isFFT
            rawData={fftFrameData}
          />
        )}
        {localTilt?.comparison}
      </Box>

      {/* ---- Controls ---- */}
      <Box component="fieldset" disabled={controlsLocked}
        sx={{ border:0, p:0, minWidth:0, width: totalW, maxWidth: "100%", mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>
        {localTiltEnabled && <Typography sx={typography.label}>
          Controls · {regionalMode ? (activeFit?.label || 'Selected region') : 'Overview'}
          {regionalMode && !regionalReady ? ' · refit this region to edit' : ''}
          {previewError && <Box component="span" role="alert" sx={{ color: "#ef4444" }}> · {previewError}</Box>}
        </Typography>}

        {/* Controls + Histogram row */}
        <Box sx={{ display: "flex", gap: `${SPACING.SM}px`, width: "100%" }}>
          {/* Left: stats + controls */}
          <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
            {/* Stats row: each readout gets a hover tooltip so new users know what the numbers mean */}
            <Box sx={controlBand(tc)}>
              {tuning && <>
              <Tooltip title="Defocus in nanometers.  Primary aberration: sets the probe's axial focus point relative to the sample.  Usually the first knob to get right." placement="top" arrow>
                <Typography sx={{ ...typography.label, ...statChip(tc, Math.abs(c10) > 0), cursor: "help" }}>
                  C10 <Box component="span" sx={{ color: tc.accent }}>{c10.toFixed(1)}</Box> nm
                </Typography>
              </Tooltip>
              <Tooltip title="2-fold astigmatism magnitude in nanometers.  How asymmetric the probe is.  Paired with φ₁₂ (angle)." placement="top" arrow>
                <Typography sx={{ ...typography.label, ...statChip(tc, Math.abs(c12) > 0), cursor: "help" }}>
                  C12 <Box component="span" sx={{ color: tc.accent }}>{c12.toFixed(1)}</Box> nm
                </Typography>
              </Tooltip>
              <Tooltip title="Angle of 2-fold astigmatism in degrees.  Only meaningful when C12 ≠ 0." placement="top" arrow>
                <Typography sx={{ ...typography.label, ...statChip(tc, Math.abs(phi12) > 0), cursor: "help" }}>
                  φ₁₂ <Box component="span" sx={{ color: tc.accent }}>{phi12.toFixed(1)}</Box>°
                </Typography>
              </Tooltip>
              {loss != null && !regionalMode && (
                <Tooltip title={hoActive
                  ? "BF-disk phase variance from the full 14-coef kernel, a diagnostic: the default fit maximises the tilt-corrected model's agreement with the data, not this number.  Drag a higher-order slider and watch this number: lower = sharper reconstruction.  Compare to the HO=0 baseline to confirm your tuning actually helps."
                  : "Variance of the reconstructed phase inside the BF (brightfield) disk, the central region of the diffraction pattern where the direct beam lands.  SSB uses that region's scattered intensity to recover phase; lower loss ≈ sharper reconstruction.  A diagnostic: the default fit maximises the tilt-corrected model's agreement with the data instead."} placement="top" arrow>
                  <Typography sx={{ ...typography.label, ...statChip(tc, true), cursor: "help" }}>
                    loss <Box component="span" sx={{ color: tc.accent }}>{formatNumber(loss, 8)}</Box>
                    {delta != null && (
                      <Box component="span" sx={{ color: delta < 0 ? STATUS_GOOD : STATUS_BAD, ml: 0.5 }}>
                        {delta > 0 ? "+" : ""}{delta.toFixed(8)}
                      </Box>
                    )}
                    {hoActive && (
                      <Box component="span" sx={{ color: STATUS_GOOD, ml: 0.5, fontFamily: "monospace" }}>
                        ({hoActiveCount} HO)
                      </Box>
                    )}
                  </Typography>
                </Tooltip>
              )}
              {/* Inline readout of every non-zero higher-order coefficient.
                  Shows magnitude (with unit) and angle (if any) so the user can
                  see what they've dialed in without opening the HO panel. */}
              {[2, 3, 4, 5].flatMap(order =>
                HO_BY_ORDER[order]
                  .map(entry => {
                    const mag = higherOrder[higherOrderMagnitudeKey(entry.name)] ?? 0;
                    if (Math.abs(mag) === 0) return null;
                    const angle = entry.hasAngle ? (higherOrder[higherOrderAzimuthKey(entry.name)] ?? 0) : null;
                    const label = `${formatHOValue(mag, entry.mag_max, entry.display_scale)} ${entry.unit_display}${angle != null ? ` / ${angle.toFixed(0)}°` : ""}`;
                    return (
                      <Tooltip key={entry.name} title={entry.tooltip} placement="top" arrow>
                        <Typography sx={{ ...typography.label, color: tc.textMuted, fontFamily: "monospace", cursor: "help" }}>
                          {entry.name} <Box component="span" sx={{ color: tc.accent }}>{label}</Box>
                        </Typography>
                      </Tooltip>
                    );
                  })
                  .filter(Boolean)
              )}
              </>}
              {busy && <Box component="span" sx={{ color: tc.accent }}>●</Box>}
              <RenderPathBadge colors={tc} />
              {!tuning && !localTiltJson && sessionNote && (
                <Typography sx={{ ...typography.value, color: tc.textMuted }} data-showptycho-session-note>
                  {sessionNote}
                </Typography>
              )}
              {runtimeStatusLabel && (
                <Typography
                  sx={{
                    ...typography.value,
                    ...statChip(tc, !!webgpuSsbRef.current && !/failed|unavailable/i.test(webgpuRuntimeStatus)),
                    maxWidth: 520,
                    whiteSpace: "nowrap",
                    overflow: "visible",
                  }}
                  title={webgpuRuntimeStatus}
                >
                  {runtimeStatusLabel}
                </Typography>
              )}
              <Box sx={{ flex: 1 }} />
              {tuning && (
                <Typography sx={{ ...typography.value, color: tc.textMuted, opacity: 0.7 }}>
                  auto {autoC10?.toFixed(1)} / {autoC12?.toFixed(1)} / {autoPhi12?.toFixed(0)}°{!regionalMode ? ` = ${autoLoss?.toFixed(8)}` : ''}
                </Typography>
              )}
            </Box>

            {webgpuPreviewEnabled && !localSourceGranted && (
              <Box sx={{ border: `1px solid ${tc.border}`, bgcolor: tc.bgAlt, px: 1, py: 0.75, display: "flex", alignItems: "center", gap: 1 }}>
                <Typography sx={{ ...typography.value, color: tc.text }}>
                  {localFolderName
                    ? `No server needed - click Open data folder, then select the folder "${localFolderName}" (the one this page lives in, usually in Downloads).`
                    : "No server needed - grant this page access to its own folder to load the data."}
                </Typography>
                {localSourceError && (
                  <Typography sx={{ ...typography.value, color: STATUS_BAD }}>
                    {localSourceError}
                  </Typography>
                )}
                <Typography sx={{ ...typography.value, color: tc.textMuted, fontSize: 11 }}>
                  Alternative: serve the folder over HTTP (`python -m http.server`) and open index.html.
                </Typography>
                <Button size="small" variant="outlined" onClick={grantLocalDirectory} sx={{ textTransform: "none", fontSize: 12 }} data-showptycho-open-folder>
                  Open data folder
                </Button>
                <input
                  ref={localDirInputRef}
                  type="file"
                  style={{ display: "none" }}
                  onChange={onLocalDirInput}
                  {...({ webkitdirectory: "", directory: "", multiple: true } as object)}
                />
              </Box>
            )}
            {webgpuLoadProgress && (
              <Box
                sx={{
                  border: `1px solid ${loadProgressIsError ? STATUS_BAD : tc.border}`,
                  bgcolor: loadProgressIsError ? `${STATUS_BAD}18` : tc.bgAlt,
                  px: 1,
                  py: 0.75,
                  display: "flex",
                  flexDirection: "column",
                  gap: 0.5,
                }}
                title={webgpuProgressLabel(webgpuLoadProgress)}
              >
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, minWidth: 0 }}>
                  <Typography sx={{ ...typography.label, color: loadProgressIsError ? STATUS_BAD : tc.accent, fontWeight: 700, whiteSpace: "nowrap" }}>
                    {loadProgressIsError ? "WebGPU blocked" : "Loading WebGPU"}
                  </Typography>
                  <Typography sx={{ ...typography.value, color: tc.text, minWidth: 0, overflowWrap: "anywhere" }}>
                    {webgpuLoadProgress.message}
                  </Typography>
                  {webgpuLoadProgress.activeBf != null && webgpuLoadProgress.totalBf != null && (
                    <Typography sx={{ ...typography.value, color: tc.textMuted, whiteSpace: "nowrap" }}>
                      active BF {webgpuLoadProgress.activeBf}/{webgpuLoadProgress.totalBf}
                    </Typography>
                  )}
                  {webgpuLoadProgress.sourceFrames != null && (
                    <Typography sx={{ ...typography.value, color: tc.textMuted, whiteSpace: "nowrap" }}>
                      frames {webgpuLoadProgress.sourceFrames}
                    </Typography>
                  )}
                  {webgpuLoadProgress.elapsedMs != null && (
                    <Typography sx={{ ...typography.value, color: tc.textMuted, whiteSpace: "nowrap" }}>
                      {(webgpuLoadProgress.elapsedMs / 1000).toFixed(1)} s
                    </Typography>
                  )}
                </Box>
                {webgpuLoadProgress.detail && (
                  <Typography sx={{ ...typography.value, color: tc.textMuted, overflowWrap: "anywhere" }}>
                    {webgpuLoadProgress.detail}
                  </Typography>
                )}
                <LinearProgress
                  variant={loadProgressPercent == null || loadProgressIsError ? "indeterminate" : "determinate"}
                  value={loadProgressPercent ?? 0}
                  sx={{
                    height: 4,
                    bgcolor: tc.controlBg,
                    "& .MuiLinearProgress-bar": { bgcolor: loadProgressIsError ? STATUS_BAD : tc.accent },
                  }}
                />
              </Box>
            )}

            {/* Display controls: color/contrast controls stay below the stats; primary
                real-space view mode lives in the top toolbar next to BF. */}
            <Box sx={{ ...controlBand(tc), width: "fit-content" }}>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>Color</Typography>
              <Select
                value={cmap} onChange={(e) => setCmap(e.target.value)}
                size="small" sx={{ ...themedSelect, minWidth: 78, fontSize: 10 }}
                MenuProps={themedMenuProps}
              >
                {COLORMAP_NAMES.map(n => (
                  <MenuItem key={n} value={n}>{n.charAt(0).toUpperCase() + n.slice(1)}</MenuItem>
                ))}
              </Select>
              <Typography sx={{ ...typography.label, fontSize: 10 }} title="Auto-contrast: snap histogram clip to 1–99 percentile on every new reconstruction. Dragging the histogram slider flips this OFF.">Auto</Typography>
              <Switch
                checked={autoContrast}
                onChange={(e) => { const on = e.target.checked; setAutoContrast(on); if (on) setContrastRange([1, 99]); }}
                size="small" sx={switchStyles.small}
              />
              <Typography sx={{ ...typography.label, fontSize: 10 }} title="CSS bilinear interpolation on the phase canvas. Same data, browser softens pixel grid. Useful when upscaling small phase images on a large canvas.">Smooth</Typography>
              <Switch
                checked={smooth}
                onChange={(_, checked) => setSmooth(checked)}
                size="small" sx={switchStyles.small}
              />
            </Box>
          </Box>

          {/* Phase contrast stays beside the image controls; FFT contrast lives in Settings. */}
          <Box sx={{ display: "flex", flexDirection: "row", alignItems: "flex-end", gap: 1, flexShrink: 0 }}>
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0.25 }}>
              <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>
                Phase
              </Typography>
              <Histogram
                frame={panelFrames[0]}
                vminPct={contrastRange[0]}
                vmaxPct={contrastRange[1]}
                onRangeChange={(lo, hi) => { setContrastRange([lo, hi]); setAutoContrast(false); }}
                width={110}
                height={48}
                tc={tc}
              />
            </Box>
          </Box>
        </Box>

        {/* Sample tilt first: for thick, tilted crystals it decides what the lattice looks like before any aberration does. */}
        {(sampleAvailable || webgpuStandalone) && (
          <SamplePanel
            tc={tc}
            open={sampleOpen}
            onToggle={() => setSampleOpen(open => !open)}
            values={sample}
            setValues={setSample}
            onCommit={commitSample}
            onFit={requestSampleFit}
            onReset={doReset}
            fitBusy={regionalMode ? regionFitBusy : sampleFitBusy}
            fitStatus={regionalMode ? '' : (sampleFitStatus || "")}
            fitAvailable={!webgpuStandalone}
            disabled={controlsLocked}
          />
        )}
        {tuning && <>
        {/* Aberration sliders.  Each is a 3-thumb slider: outer thumbs = PLAY
            sweep bounds, middle thumb = current value.  CSS below dims the outer
            thumbs (fully opaque only for the active sweep target) so the
            current-value thumb stays visually dominant: the range markers are
            meant to look like brackets, not competing handles. */}
        <Box sx={{
          display: "flex",
          alignItems: "center",
          gap: `${SPACING.MD}px`,
          px: 1,
          py: 0.5,
          border: `1px solid ${tc.border}`,
          bgcolor: tc.controlBg,
          width: "fit-content",
          maxWidth: "100%",
          flexWrap: "wrap",
        }}>
          {([
            {
              key: "c10" as AberKey, label: "C10", unit: "nm", value: c10, displayPrec: 1,
              tipFull: "C10: defocus.  Positive = overfocus, negative = underfocus.  The dominant aberration.",
              min: c10UiMin, max: c10UiMax, step: 0.1,
              setValue: (value: number) => { dragSliderValue("c10", value); sendDrag(value, sliderVals.current.c12, sliderVals.current.phi12); },
              commitValue: (value: number) => { commitSliderValue("c10", value); if (shouldCommitOnReleaseRef.current) sendCommit(value, sliderVals.current.c12, sliderVals.current.phi12); },
            },
            {
              key: "c12" as AberKey, label: "C12", unit: "nm", value: c12, displayPrec: 1,
              tipFull: "C12: 2-fold astigmatism magnitude.  Paired with φ₁₂.",
              min: c12UiMin, max: c12UiMax, step: 0.1,
              setValue: (value: number) => { dragSliderValue("c12", value); sendDrag(sliderVals.current.c10, value, sliderVals.current.phi12); },
              commitValue: (value: number) => { commitSliderValue("c12", value); if (shouldCommitOnReleaseRef.current) sendCommit(sliderVals.current.c10, value, sliderVals.current.phi12); },
            },
            {
              key: "phi12" as AberKey, label: "φ₁₂", unit: "°", value: phi12, displayPrec: 0,
              tipFull: "φ₁₂: angle of 2-fold astigmatism.  Meaningful when C12 ≠ 0.",
              min: phi12UiMin, max: phi12UiMax, step: 1,
              setValue: (value: number) => { dragSliderValue("phi12", value); sendDrag(sliderVals.current.c10, sliderVals.current.c12, value); },
              commitValue: (value: number) => { commitSliderValue("phi12", value); if (shouldCommitOnReleaseRef.current) sendCommit(sliderVals.current.c10, sliderVals.current.c12, value); },
            },
            {
              key: "rot" as AberKey, label: "scan-det rot", unit: "°", value: rotationDeg ?? 0, displayPrec: 1,
              tipFull: "Scan-detector rotation: angle between scan axes and detector axes.  Sweep to find the sharpest reconstruction.",
              min: rotationUiMin, max: rotationUiMax, step: 0.1,
              setValue: (value: number) => {
                setRotationDeg(value);
                if (webgpuSsbRef.current) sendDrag(sliderVals.current.c10, sliderVals.current.c12, sliderVals.current.phi12, value);
              },
              commitValue: (value: number) => {
                if (shouldCommitOnReleaseRef.current) sendCommit(sliderVals.current.c10, sliderVals.current.c12, sliderVals.current.phi12, value);
              },
            },
          ] as const).filter(aberration => !regionalMode || aberration.key !== 'rot').map(aberration => {
            const range = sweepRanges[aberration.key];
            const isActiveSweep = sweepMainKeys(sweepParam).includes(aberration.key);
            const isAtFull = range[0] === aberration.min && range[1] === aberration.max;
            const sliderWidth = aberration.key === "rot" ? ROTATION_SLIDER_WIDTH : ABERRATION_SLIDER_WIDTH;
            const visibleValue = Math.max(aberration.min, Math.min(aberration.max, aberration.value));
            return (
              <Box key={aberration.key} sx={{ flex: `0 0 ${sliderWidth}px`, width: sliderWidth, minWidth: 0 }}>
                <Tooltip title={aberration.tipFull} placement="top" arrow>
                  <Typography sx={{ ...typography.labelSmall, color: tc.textMuted, mb: -0.5, cursor: "help" }}>
                    {aberration.label} <b>{aberration.value.toFixed(aberration.displayPrec)}</b> {aberration.unit}
                    {!isAtFull && (
                      <span style={{ color: isActiveSweep ? tc.accent : tc.textMuted, opacity: 0.65 }}>
                        {" ["}{range[0].toFixed(aberration.displayPrec)}, {range[1].toFixed(aberration.displayPrec)}{"]"}
                      </span>
                    )}
                  </Typography>
                </Tooltip>
                {/* Show3D-style 3-thumb slider: outer thumbs = sweep stoppers,
                    middle thumb = current value.  disableSwap keeps order stable;
                    to push the value outside the stopper window, drag the relevant
                    stopper first. */}
                <Slider
                  value={[range[0], visibleValue, range[1]]}
                  min={aberration.min} max={aberration.max} step={aberration.step}
                  disabled={controlsLocked}
                  disableSwap
                  onChange={(_, value, activeThumb) => {
                    const thumbs = value as number[];
                    if (activeThumb === 1) {
                      aberration.setValue(thumbs[1]);
                    } else {
                      updateSweepRange(aberration.key, [thumbs[0], thumbs[2]]);
                    }
                  }}
                  onChangeCommitted={(_, value) => {
                    const thumbs = value as number[];
                    aberration.commitValue(thumbs[1]);
                  }}
                  size="small"
                  sx={{
                    py: 0.5,
                    "& .MuiSlider-thumb[data-index='0']": {
                      width: 8, height: 8, bgcolor: tc.textMuted, opacity: isActiveSweep ? 1 : 0.7,
                    },
                    "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 },
                    "& .MuiSlider-thumb[data-index='2']": {
                      width: 8, height: 8, bgcolor: tc.textMuted, opacity: isActiveSweep ? 1 : 0.7,
                    },
                  }}
                />
              </Box>
            );
          })}
          <Tooltip title="Negate the displayed phase.  SSB's phase is defined only up to ± (sign-ambiguous).  Use this to match the expected contrast of your sample.  Display only: it does not re-run reconstruction or change any fitted number (display_negate_phase)." placement="top" arrow>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, cursor: "help" }}>
              <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Negate phase (display)</Typography>
              <Switch disabled={regionalMode} checked={!!displayNegatePhase} onChange={(_, checked) => setDisplayNegatePhase(checked)} size="small" />
            </Box>
          </Tooltip>
          <Tooltip title="Reset all aberrations to the automatic fitted result and restore every sweep-stopper to its full range." placement="top" arrow>
            <IconButton
              size="small"
              onClick={() => {
                doReset();
                updateSweepRange("c10", [c10UiMin, c10UiMax]);
                updateSweepRange("c12", [c12UiMin, c12UiMax]);
                updateSweepRange("phi12", [phi12UiMin, phi12UiMax]);
                updateSweepRange("rot", [rotationUiMin, rotationUiMax]);
              }}
              sx={compactIconButton(tc, tc.textMuted)}
              aria-label="Reset aberrations"
            >
              <RestartAltIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Higher-order aberration panel (collapsible).  Shows 11 Krivanek
            coefficients (C21 through C56), grouped by order.  When any
            magnitude is non-zero the engine routes through the 14-coef
            kernel. */}
        {!regionalMode && <HigherOrderPanel
          tc={tc}
          open={hoOpen}
          onToggle={() => setHoOpen(open => !open)}
          activeCount={hoActiveCount}
          values={higherOrder}
          setValues={setHigherOrder}
          onCommit={commitSample}
        />}

        {/* Single compact action row: PLAY + PIN + SAVE CALIBRATION.
            Dropdowns live inline.  Wraps to a second line only
            on narrow widgets (below ~520 px panel size). */}
        <Box sx={controlBand(tc)}>
          <Tooltip title={playing ? "Pause sweep (Space)" : "Play: sweep the selected aberration (Space)"} placement="top" arrow>
            <IconButton
              size="small"
              disableRipple
              disableFocusRipple
              disableTouchRipple
              onClick={() => setPlaying(wasPlaying => !wasPlaying)}
              sx={{ ...compactIconButton(tc, playing ? STATUS_GOOD : tc.accent), height: ACTION_CONTROL_HEIGHT }}
              aria-label={playing ? "Pause sweep" : "Play sweep"}
            >
              {playing ? <PauseIcon sx={{ fontSize: 16 }} /> : <PlayArrowIcon sx={{ fontSize: 16 }} />}
            </IconButton>
          </Tooltip>
          <Select
            value={sweepParam}
            onChange={(e) => setSweepParam(e.target.value as SweepParam)}
            size="small"
            sx={{ ...actionSelect, width: 118, minWidth: 118, flex: "0 0 118px" }}
            title="Sweep parameter"
          >
            <MenuItem value="c10">C10</MenuItem>
            <MenuItem value="c12">C12</MenuItem>
            <MenuItem value="phi12">φ₁₂</MenuItem>
            {!regionalMode && <MenuItem value="rot">rot</MenuItem>}
            <MenuItem value="bundle:c10c12">C10+C12</MenuItem>
            <MenuItem value="bundle:c10c12phi12">C10+C12+φ₁₂</MenuItem>
            {!regionalMode && <MenuItem value="bundle:c10c12phi12rot">C10+C12+φ₁₂+rot</MenuItem>}
            {Object.keys(higherOrder)
              .filter(hoKey => (hoKey.startsWith("phi")
                ? (higherOrder[hoKey] !== undefined)
                : Math.abs(higherOrder[hoKey] || 0) > 0))
              .sort()
              .map(hoKey => (
                <MenuItem key={`ho:${hoKey}`} value={`ho:${hoKey}`}>
                  {hoKey.startsWith("phi") ? `C${hoKey.slice(3, -4)} φ` : `${hoKey.slice(0, -3)} m`}
                </MenuItem>
              ))}
          </Select>
          <Select
            value={playFps}
            onChange={(e) => setPlayFps(Number(e.target.value))}
            size="small"
            sx={{ ...actionSelect, width: 52, minWidth: 52, flex: "0 0 52px" }}
            title="Target sweep frames per second"
          >
            {[1, 3, 5, 10, 15, 30, 60].map(fps => (<MenuItem key={fps} value={fps}>{fps}</MenuItem>))}
          </Select>
          {/* Divider: PLAY group | PIN */}
          <Box sx={{ width: "1px", height: 20, bgcolor: tc.border, mx: 0.5, alignSelf: "center" }} />
          <Tooltip title="Pin the current snapshot (P)" placement="top" arrow>
            <IconButton
              size="small"
              disableRipple
              disableFocusRipple
              disableTouchRipple
              onClick={doPin}
              disabled={regionalMode || busy || !rawPhaseRef.current}
              sx={{ ...compactIconButton(tc, tc.accent), height: ACTION_CONTROL_HEIGHT }}
              aria-label="Pin current snapshot"
            >
              <PushPinIcon sx={{ fontSize: 15 }} />
            </IconButton>
          </Tooltip>
          <Box sx={{ width: "1px", height: 20, bgcolor: tc.border, mx: 0.5, alignSelf: "center" }} />

          <Tooltip title="Save calibration.json in this ShowPtycho project for later sessions and downstream analysis." placement="top" arrow>
            <Button
              size="small" variant="outlined"
              startIcon={<SaveIcon sx={{ fontSize: 14 }} />}
              onClick={() => { if (webgpuStandalone) { void doFolderSave(); } else { setSaveTrigger((saveTrigger ?? 0) + 1); } }}
              sx={{ ...compactButton(tc), height: ACTION_CONTROL_HEIGHT, color: STATUS_GOOD, borderColor: STATUS_GOOD, width: 76, minWidth: 76 }}
            >
              Save
            </Button>
          </Tooltip>
          <Box
            component="input"
            type="text"
            value={notes ?? ""}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNotes(e.target.value)}
            placeholder="notes (optional)"
            sx={{
              height: ACTION_CONTROL_HEIGHT,
              boxSizing: "border-box",
              ...typography.label, flex: 1, minWidth: 120,
              bgcolor: tc.bgAlt, color: tc.text,
              border: `1px solid ${tc.border}`, borderRadius: 0,
              px: 0.75, py: 0, outline: "none",
              "&:focus": { borderColor: tc.accent },
            }}
          />
          {(calibrationPath || calibrationStatus) && (
            <Tooltip title={`${calibrationPath || ""} · ${calibrationSavedAt || ""}`} placement="top" arrow>
              <Typography sx={{
                ...typography.label, color: calibrationPath ? STATUS_GOOD : tc.textMuted,
                maxWidth: 260, overflowWrap: "anywhere",
              }}>
                {calibrationStatus || `✓ ${compactPathLabel(calibrationPath, 2)}`}
              </Typography>
            </Tooltip>
          )}
        </Box>
        </>}

        {/* Folder-persisted snapshots - live in snapshots/ inside the export folder,
            so they reappear on relaunch from CLI serve or double-click. */}
        {webgpuStandalone && (folderSaves.length > 0 || folderSaveStatus) && (
          <Box sx={{ pt: `${SPACING.XS}px`, borderTop: `1px solid ${tc.border}`, display: "flex", flexDirection: "column", gap: 0.25 }}>
            {folderSaveStatus && (
              <Typography sx={{ ...typography.value, color: /failed|read-only/i.test(folderSaveStatus) ? STATUS_BAD : STATUS_GOOD }}>
                {folderSaveStatus}
              </Typography>
            )}
            {folderSaves.length > 0 && (
              <Typography sx={{ ...typography.label, color: tc.textMuted }}>Snapshots ({folderSaves.length})</Typography>
            )}
            {folderSaves.map((record) => (
              <Box key={record.id} sx={{ display: "flex", alignItems: "center", gap: 1, fontFamily: "monospace", fontSize: 11 }}>
                <Typography sx={{ ...typography.value, color: tc.text, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
                  C10 {record.C10_nm.toFixed(1)} · C12 {record.C12_nm.toFixed(1)} · φ12 {record.phi12_deg.toFixed(0)}° · rot {record.scan_detector_rotation_deg.toFixed(1)}°
                  {record.loss != null ? ` · loss ${record.loss.toFixed(6)}` : ""}
                  {` · ${record.generated_at_utc.slice(0, 16).replace("T", " ")}`}
                  {record.notes ? ` · ${record.notes}` : ""}
                </Typography>
                <Button size="small" onClick={() => loadFolderSave(record)} sx={{ ...compactButton(tc), minWidth: 44 }}>Load</Button>
                <Button size="small" onClick={() => void downloadFolderSave(record)} sx={{ ...compactButton(tc), minWidth: 30 }}>⬇</Button>
                <Button size="small" onClick={() => void deleteFolderSave(record)} sx={{ ...compactButton(tc), minWidth: 30, color: STATUS_BAD }}>✕</Button>
              </Box>
            ))}
          </Box>
        )}

        {/* Optuna trials strip: scrollable, loss-ranked.  Click any tile to
            preview that trial's aberrations.  Best-loss tile is highlighted so
            the user can immediately verify that Nelder-Mead's refined point is near
            (or past) Optuna's best. */}
        {trials.length > 0 && (
          <Box sx={{ pt: `${SPACING.XS}px`, borderTop: `1px solid ${tc.border}` }}>
            <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} sx={{ mb: `${SPACING.XS}px` }}>
              <Box
                onClick={() => setTrialsExpanded(expanded => !expanded)}
                sx={{ cursor: "pointer", userSelect: "none", color: tc.textMuted, fontSize: 10, fontFamily: "monospace" }}
              >
                {trialsExpanded ? "▼" : "▶"} Search trials ({trials.length}) · {trialObjectiveText} · best first · click to preview
              </Box>
              <Box sx={{ flex: 1 }} />
              {trials.length > 0 && (
                <Tooltip title={`The search scored ${trials.length} trials on ${trialObjectiveText}, then the local solver refined the best one. The refined fit's loss is the phase-variance diagnostic (${autoLoss?.toFixed(8) ?? '-'}); a trial shows its own objective's score: agreement (higher is better) for the default model fit, phase variance (lower is better) for the backup.`} placement="top" arrow>
                  <Typography sx={{ ...typography.value, color: tc.textMuted, cursor: "help" }}>
                    Refined fit {autoLoss?.toFixed(8) ?? '-'}  ·  best trial {trialScoreText(trials[0])}
                  </Typography>
                </Tooltip>
              )}
            </Stack>
            {trialsExpanded && (
              <Stack direction="row" spacing={`${SPACING.XS}px`} sx={{ overflowX: "auto", pb: 0.5 }}>
                {/* First tile: the Nelder-Mead refined result.  This is what
                    ssb.aberrations contains after ssb.refine() and what the
                    widget's "auto" reference is set to.  Visually distinct from
                    the Optuna tiles so users know it's the refined winner, not
                    a raw trial. */}
                {autoC10 != null && autoC12 != null && autoPhi12 != null && (
                  <Tooltip title="The local solver refined the best Optuna trial.  Click to jump sliders back to this result.  This is what 'auto' in the stats bar refers to." placement="top" arrow>
                    <Box
                      onClick={() => {
                        setActiveTrialRank(null);
                        setViewPin(null);
                        setC10(autoC10); setC12(autoC12); setPhi12(autoPhi12);
                        sendCommit(autoC10, autoC12, autoPhi12);
                      }}
                      sx={{
                        flexShrink: 0, cursor: "pointer", userSelect: "none",
                        minWidth: 100, px: 0.75, py: 0.4,
                        border: `1px solid ${STATUS_GOOD}`,
                        borderLeft: `3px solid ${STATUS_GOOD}`,
                        bgcolor: tc.bgAlt,
                        "&:hover": { bgcolor: tc.controlBg },
                      }}
                    >
                      <Stack direction="row" alignItems="baseline" spacing={0.5}>
                        <Typography sx={{ fontSize: 9, color: STATUS_GOOD, fontFamily: "monospace", fontWeight: "bold" }}>
                          Refined fit
                        </Typography>
                        <Typography sx={{ fontSize: 11, color: STATUS_GOOD, fontFamily: "monospace", fontWeight: "bold" }}>
                          {autoLoss?.toFixed(8) ?? '-'}
                        </Typography>
                      </Stack>
                      <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: tc.textMuted, whiteSpace: "nowrap" }}>
                        {autoC10.toFixed(0)} / {autoC12.toFixed(0)} / {autoPhi12.toFixed(0)}°
                      </Typography>
                    </Box>
                  </Tooltip>
                )}
                {trials.map((trial, i) => {
                  // Color ramp: best trial = accent, worst trial = border.  Interpolate
                  // via opacity so the CSS reads simply from a single accent token.
                  const frac = trials.length > 1 ? i / (trials.length - 1) : 0;
                  const isActive = activeTrialRank === trial.rank;
                  return (
                    <Box
                      key={trial.rank}
                      onClick={() => doViewTrial(trial)}
                      sx={{
                        flexShrink: 0, cursor: "pointer", userSelect: "none",
                        minWidth: 92, px: 0.75, py: 0.4,
                        border: `1px solid ${isActive ? tc.accent : tc.border}`,
                        bgcolor: tc.bgAlt,
                        borderLeft: `3px solid ${tc.accent}`,
                        opacity: 1 - 0.55 * frac,
                        "&:hover": { borderColor: tc.accent, opacity: 1 },
                      }}
                    >
                      <Stack direction="row" alignItems="baseline" spacing={0.5}>
                        <Typography sx={{ fontSize: 9, color: tc.textMuted, fontFamily: "monospace" }}>
                          #{trial.rank + 1}
                        </Typography>
                        <Typography sx={{ fontSize: 11, color: tc.accent, fontFamily: "monospace", fontWeight: "bold" }}
                          title={trial.objective === "agreement" ? "Model agreement (the default fit), higher is better" : "Phase variance (the backup search), lower is better"}>
                          {trialScoreText(trial)}
                        </Typography>
                      </Stack>
                      <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: tc.textMuted, whiteSpace: "nowrap" }}>
                        {trial.C10_nm.toFixed(0)} / {trial.C12_nm.toFixed(0)} / {trial.phi12_deg.toFixed(0)}°
                        {trial.depth_spread_nm ? ` · tilt ${trial.sample_tilt_mrad?.[0].toFixed(1)}, ${trial.sample_tilt_mrad?.[1].toFixed(1)}` : ""}
                      </Typography>
                    </Box>
                  );
                })}
              </Stack>
            )}
          </Box>
        )}

        {/* Pin viewing indicator */}
        {viewPin && (
          <Typography sx={{ ...typography.labelSmall, color: tc.accent, textAlign: "center" }}>
            Viewing pinned: drag a slider or press Esc to return
          </Typography>
        )}

        {/* Pinned strip */}
        {pinned.length > 0 && (
          <Box sx={{ pt: `${SPACING.XS}px`, borderTop: `1px solid ${tc.border}` }}>
            <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} sx={{ mb: `${SPACING.XS}px`, flexWrap: "wrap" }}>
              <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>
                Pinned ({pinned.length}/{MAX_PINS}) · P pin · ← → view · S star · Esc exit
              </Typography>
              <Tooltip title="Export every pinned phase image as PNG plus a JSON manifest with aberration metadata." placement="top" arrow>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<FileDownloadIcon sx={{ fontSize: 14 }} />}
                  onClick={() => { void doExportPinned(); }}
                  sx={{ ...compactButton(tc), height: ACTION_CONTROL_HEIGHT, color: tc.accent, borderColor: tc.accent, minWidth: 78 }}
                >
                  Export
                </Button>
              </Tooltip>
              {exportStatus && (
                <Typography sx={{ ...typography.value, color: exportStatus.startsWith("Export failed") ? STATUS_BAD : tc.textMuted }}>
                  {exportStatus}
                </Typography>
              )}
              <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>size</Typography>
              <Select
                value={thumbSize}
                onChange={(e) => setThumbSize(e.target.value as ThumbSize)}
                size="small"
                sx={{ ...themedSelect, minWidth: 58, fontSize: 10 }}
              >
                <MenuItem value="S">S</MenuItem>
                <MenuItem value="M">M</MenuItem>
                <MenuItem value="L">L</MenuItem>
              </Select>
              {starsPath && pinned.some(pin => pin.starred) && (
                <Tooltip title={`Starred snapshots auto-save to ${starsPath}.  Load in the next cell with json.load(open(path)).`} placement="top" arrow>
                  <Typography sx={{ ...typography.value, color: tc.textMuted, opacity: 0.65, cursor: "help", overflowWrap: "anywhere" }}>
                    ★ → {compactPathLabel(starsPath, 2)}
                  </Typography>
                </Tooltip>
              )}
            </Stack>
            <Stack direction="row" spacing={`${SPACING.XS}px`} sx={{ overflowX: "auto", pb: 0.5 }}>
              {pinned.map(pin => {
                const thumbPx = THUMB_SIZE_PX[thumbSize];
                const isDraggingPin = draggingPin === pin.id;
                const isNewPin = newPinId === pin.id;
                return (
                  <Box
                    key={pin.id}
                    data-showptycho-pin-id={pin.id}
                    data-showptycho-pin-state={isDraggingPin ? "dragging" : isNewPin ? "new" : undefined}
                    title="Drag to reorder pinned images. Click to view."
                    onClick={() => {
                      if (didDragPinRef.current) {
                        didDragPinRef.current = false;
                        return;
                      }
                      doViewPin(pin);
                    }}
                    onPointerDown={(e) => {
                      if (e.button !== 0) return;
                      beginPinDrag(pin.id, e.clientX, e.clientY);
                      e.currentTarget.setPointerCapture(e.pointerId);
                    }}
                    onPointerMove={(e) => {
                      if (updatePinDrag(pin.id, e.clientX, e.clientY)) e.preventDefault();
                    }}
                    onPointerUp={(e) => {
                      finishPinDrag(pin.id, e.clientX, e.clientY);
                    }}
                    onPointerCancel={cancelPinDrag}
                    sx={{
                    position: "relative", cursor: "grab", flexShrink: 0,
                    border: viewPin === pin.id
                      ? `2px solid ${tc.accent}`
                      : pin.starred ? `2px solid ${STATUS_GOOD}` : "2px solid transparent",
                    outline: dragOverPin === pin.id ? `2px solid ${tc.accent}` : "none",
                    outlineOffset: 1,
                    transform: isDraggingPin ? "scale(1.045)" : "scale(1)",
                    transition: "border-color 120ms ease, transform 120ms ease, outline-color 120ms ease",
                    animation: isNewPin ? "showptycho-pin-add 320ms cubic-bezier(0.2, 0, 0, 1)" : "none",
                    overflow: "hidden",
                    userSelect: "none",
                    touchAction: "none",
                    zIndex: isDraggingPin ? 3 : 1,
                    "&:active": { cursor: "grabbing" },
                    "&:hover": { borderColor: tc.accent },
                    "@keyframes showptycho-pin-add": {
                      "0%": { opacity: 0, transform: "scale(0.82)" },
                      "70%": { opacity: 1, transform: "scale(1.06)" },
                      "100%": { opacity: 1, transform: "scale(1)" },
                    },
                  }}>
                    <canvas
                      key={`${pin.id}-${cmap}`}
                      data-cmap={cmap}
                      ref={el => {
                        if (el && pin.thumb) {
                          el.width = THUMB_BITMAP_PX; el.height = THUMB_BITMAP_PX;
                          el.getContext("2d")!.drawImage(pin.thumb, 0, 0);
                        }
                      }}
                      style={{ width: thumbPx, height: thumbPx, display: "block", pointerEvents: "none" }}
                    />
                    {/* Bottom metadata strip, sized so small thumbs stay readable. */}
                    <Box sx={{
                      position: "absolute", bottom: 0, left: 0, right: 0,
                      bgcolor: "rgba(0,0,0,0.65)",
                      px: 0.3, py: thumbSize === "S" ? 0 : 0.15,
                    }}>
                      <Typography sx={{
                        fontSize: thumbSize === "L" ? 10 : thumbSize === "M" ? 9 : 7,
                        lineHeight: 1.15, fontFamily: "monospace", color: "#e0e0e0",
                        whiteSpace: "nowrap",
                      }}>
                        {pin.C10_nm.toFixed(0)}/{pin.C12_nm.toFixed(0)}/{pin.phi12_deg.toFixed(0)}°
                      </Typography>
                    </Box>
                    {/* Star toggle (top-left).  Gold when active, muted otherwise. */}
                    <Tooltip title={pin.starred ? "Unstar (remove from time-series JSON)" : "Star: save to JSON time-series"} placement="top" arrow>
                      <Box
                        onClick={e => { e.stopPropagation(); doToggleStar(pin.id); }}
                        sx={{
                          position: "absolute", top: 0, left: 0,
                          width: thumbSize === "L" ? 20 : 16, height: thumbSize === "L" ? 20 : 16,
                          bgcolor: "rgba(0,0,0,0.6)",
                          display: "flex", alignItems: "center", justifyContent: "center",
                          cursor: "pointer", borderRadius: "0 0 4px 0",
                          fontSize: thumbSize === "L" ? 13 : 11,
                          color: pin.starred ? "#ffd54a" : tc.textMuted,
                          "&:hover": { color: "#ffd54a" },
                        }}
                      >{pin.starred ? "★" : "☆"}</Box>
                    </Tooltip>
                    {/* Unpin (top-right) */}
                    <Box onClick={e => { e.stopPropagation(); doUnpin(pin.id); }} sx={{
                      position: "absolute", top: 0, right: 0,
                      width: thumbSize === "L" ? 18 : 14, height: thumbSize === "L" ? 18 : 14,
                      bgcolor: "rgba(0,0,0,0.6)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      cursor: "pointer", borderRadius: "0 0 0 4px",
                      fontSize: thumbSize === "L" ? 11 : 9, color: tc.textMuted,
                      "&:hover": { color: STATUS_BAD },
                    }}>✕</Box>
                  </Box>
                );
              })}
            </Stack>
          </Box>
        )}
      </Box>
    </Box>
  );
}

const render = createRender(() => <ThemeProvider theme={widgetTheme}><Explore /></ThemeProvider>);

export default { render };
