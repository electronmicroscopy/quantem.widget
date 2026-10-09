/// <reference types="@webgpu/types" />
/**
 * Show3DSlices - Orthogonal slice viewer for 3D volumetric data.
 *
 * Top plus arbitrary-angle vertical slice panels with synchronized sliders and a 3D orientation view.
 * All slicing done in JS from raw float32 volume data for instant response.
 *
 * Ptycho-focused single-object workflow; tomography/comparison flows belong in
 * Show3DVolume.
 */
import * as React from "react";
import { sliderStyles } from "../controlStyles";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import StopIcon from "@mui/icons-material/Stop";
import { useTheme } from "../theme";
import { VolumeRenderer, CameraState, DEFAULT_CAMERA } from "../webgpu-volume";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar } from "../figure";
import { downloadBlob, extractBytes, extractFloat32, formatNumber, preserveRestoredWidgetModelsOnSave } from "../format";
import { findDataRange, applyLogScale, percentileClip, signedLog1p, sliderRange } from "../display/stats";
import { MetadataSection } from "../widgetInfo";
import { dequantizeUint8 } from "../display/quantization";
import { InfoTooltip } from "../shared/InfoTooltip";
import { RenderPathBadge } from "../shared/RenderPathBadge";
import { KeyboardShortcuts } from "../shared/KeyboardShortcuts";
import { exportTitleSlug, formatEstimatedHtmlSize, formatSavedBytes, isAbortLikeError } from "../shared/exportFormat";
import { pointToSegmentDistance } from "../shared/geometry";
import { shouldIgnoreWidgetShortcut } from "../shared/widgetShortcuts";
import { resolveDisplayBounds } from "../shared/displayRange";
import { Histogram } from "../shared/Histogram";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen, renderToOffscreenReuse, createGPUColormapEngine, GPUColormapEngine, VOLUME_SLICE_MIN_ZOOM } from "../display/colormaps";
import { DisplayFFT, getDisplayFFT, getGPUDevice, fft2d, fftshift, nextPow2, computeMagnitude, autoEnhanceFFT, applyHannWindow2D, reciprocalCoordinatesFromShiftedOffset } from "../display/fft";
import { findFFTPeakBrowser } from "../display/geometry";
import { estimateSliceAlignment } from "../sliceAlignment";

const MAX_PLAYBACK_FPS = 30;
const PAGE_PLAY_FPS_OPTIONS = [1, 2, 4, 8] as const;

const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const PLANE_KEYS = ["xy", "oblique"] as const;
const PLANE_LABELS = ["Top", "Side"] as const;
const PLANE_COLORS = ["#4d80ff", "#4dff66"] as const;
const OBLIQUE_PROFILE_EDGE_INSET = 1;
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
  whiteSpace: "nowrap" as const,
};
const compactButton = {
  borderRadius: 0,
  fontSize: 10,
  textTransform: "none" as const,
  letterSpacing: 0,
  whiteSpace: "nowrap" as const,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};
const planeToggleButtonSx = {
  minWidth: 30,
  height: 18,
  px: 0.7,
  py: 0.1,
  fontSize: 10,
  lineHeight: 1,
  color: "primary.main",
  borderColor: "divider",
  textTransform: "none",
  letterSpacing: 0,
  "&.Mui-selected": {
    color: "primary.contrastText",
    bgcolor: "primary.main",
    "&:hover": { bgcolor: "primary.dark" },
  },
  "&:hover": { bgcolor: "action.hover" },
} as const;
const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};
// MUI 7 Switch drops `inputProps` (its own input slot props win), so the
// accessible name goes through slotProps.input, which also replaces the default
// role and so restates it.
function switchInputSlot(ariaLabel: string) {
  return { input: { role: "switch", "aria-label": ariaLabel } };
}

const typographyLabel = {
  fontSize: 10,
  textTransform: "none" as const,
  letterSpacing: 0,
};
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

type Show3DSlicesWritableFile = {
  write: (data: BlobPart) => Promise<void>;
  close: () => Promise<void>;
};

type Show3DSlicesFileHandle = {
  createWritable: () => Promise<Show3DSlicesWritableFile>;
};

type Show3DSlicesSavePickerOptions = {
  suggestedName?: string;
  types?: { description: string; accept: Record<string, string[]> }[];
};

type Show3DSlicesWindow = Window & typeof globalThis & {
  showSaveFilePicker?: (options?: Show3DSlicesSavePickerOptions) => Promise<Show3DSlicesFileHandle>;
};

function extractXY(volume: Float32Array, nx: number, ny: number, nz: number, z: number): Float32Array {
  if (z < 0 || z >= nz) return new Float32Array(ny * nx);
  const start = z * ny * nx;
  return volume.subarray(start, start + ny * nx);
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function obliqueLineEndpoints(
  nx: number,
  ny: number,
  cx: number,
  cy: number,
  angleDeg: number,
): [{ x: number; y: number }, { x: number; y: number }] {
  const theta = (angleDeg * Math.PI) / 180;
  const dx = Math.cos(theta);
  const dy = Math.sin(theta);
  const candidates: number[] = [];
  const maxX = Math.max(0, nx - 1);
  const maxY = Math.max(0, ny - 1);
  if (Math.abs(dx) > 1e-8) {
    const t0 = (0 - cx) / dx;
    const y0 = cy + t0 * dy;
    if (y0 >= 0 && y0 <= maxY) candidates.push(t0);
    const t1 = (maxX - cx) / dx;
    const y1 = cy + t1 * dy;
    if (y1 >= 0 && y1 <= maxY) candidates.push(t1);
  }
  if (Math.abs(dy) > 1e-8) {
    const t0 = (0 - cy) / dy;
    const x0 = cx + t0 * dx;
    if (x0 >= 0 && x0 <= maxX) candidates.push(t0);
    const t1 = (maxY - cy) / dy;
    const x1 = cx + t1 * dx;
    if (x1 >= 0 && x1 <= maxX) candidates.push(t1);
  }
  if (candidates.length < 2) {
    return [
      { x: clampNumber(cx, 0, maxX), y: clampNumber(cy, 0, maxY) },
      { x: clampNumber(cx, 0, maxX), y: clampNumber(cy, 0, maxY) },
    ];
  }
  const minT = Math.min(...candidates);
  const maxT = Math.max(...candidates);
  return [
    { x: clampNumber(cx + minT * dx, 0, maxX), y: clampNumber(cy + minT * dy, 0, maxY) },
    { x: clampNumber(cx + maxT * dx, 0, maxX), y: clampNumber(cy + maxT * dy, 0, maxY) },
  ];
}

function obliqueNormal(angleDeg: number): { x: number; y: number } {
  const theta = (angleDeg * Math.PI) / 180;
  return { x: -Math.sin(theta), y: Math.cos(theta) };
}

function obliqueSegmentOffsetBounds(
  nx: number,
  ny: number,
  angleDeg: number,
  start: { x: number; y: number },
  stop: { x: number; y: number },
  inset: number = 0,
): [number, number] {
  const normal = obliqueNormal(angleDeg);
  const points = [start, stop];
  const xMin = Math.min(inset, Math.max(0, nx - 1));
  const yMin = Math.min(inset, Math.max(0, ny - 1));
  const xMax = Math.max(xMin, Math.max(1, nx) - 1 - inset);
  const yMax = Math.max(yMin, Math.max(1, ny) - 1 - inset);
  let minDelta = -Infinity;
  let maxDelta = Infinity;
  for (const point of points) {
    if (Math.abs(normal.x) > 1e-8) {
      const d0 = (xMin - point.x) / normal.x;
      const d1 = (xMax - point.x) / normal.x;
      minDelta = Math.max(minDelta, Math.min(d0, d1));
      maxDelta = Math.min(maxDelta, Math.max(d0, d1));
    }
    if (Math.abs(normal.y) > 1e-8) {
      const d0 = (yMin - point.y) / normal.y;
      const d1 = (yMax - point.y) / normal.y;
      minDelta = Math.max(minDelta, Math.min(d0, d1));
      maxDelta = Math.min(maxDelta, Math.max(d0, d1));
    }
  }
  if (!Number.isFinite(minDelta) || !Number.isFinite(maxDelta) || minDelta > maxDelta) return [0, 0];
  return [Math.ceil(minDelta), Math.floor(maxDelta)];
}

function obliqueCenterOffset(
  nx: number,
  ny: number,
  angleDeg: number,
  start: { x: number; y: number },
  stop: { x: number; y: number },
): number {
  const normal = obliqueNormal(angleDeg);
  const ox = (Math.max(1, nx) - 1) / 2;
  const oy = (Math.max(1, ny) - 1) / 2;
  const cx = (start.x + stop.x) / 2;
  const cy = (start.y + stop.y) / 2;
  return (cx - ox) * normal.x + (cy - oy) * normal.y;
}

function profilePointFromAny(value: unknown): { x: number; y: number } | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const row = Number(record.row);
  const col = Number(record.col);
  if (!Number.isFinite(row) || !Number.isFinite(col)) return null;
  return { x: col, y: row };
}

function profileLinePayload(start: { x: number; y: number }, stop: { x: number; y: number }): { row: number; col: number }[] {
  return [
    { row: start.y, col: start.x },
    { row: stop.y, col: stop.x },
  ];
}

function clampPointToImage(point: { x: number; y: number }, nx: number, ny: number, inset: number = 0): { x: number; y: number } {
  const xInset = Math.min(inset, Math.max(0, nx - 1) / 2);
  const yInset = Math.min(inset, Math.max(0, ny - 1) / 2);
  return {
    x: clampNumber(point.x, xInset, Math.max(xInset, Math.max(0, nx - 1) - xInset)),
    y: clampNumber(point.y, yInset, Math.max(yInset, Math.max(0, ny - 1) - yInset)),
  };
}

function translateSegmentInsideImage(
  start: { x: number; y: number },
  stop: { x: number; y: number },
  dxRaw: number,
  dyRaw: number,
  nx: number,
  ny: number,
  inset: number = 0,
): { start: { x: number; y: number }; stop: { x: number; y: number } } {
  const xMin = Math.min(inset, Math.max(0, nx - 1));
  const yMin = Math.min(inset, Math.max(0, ny - 1));
  const xMax = Math.max(xMin, Math.max(1, nx) - 1 - inset);
  const yMax = Math.max(yMin, Math.max(1, ny) - 1 - inset);
  const dxMin = Math.max(xMin - start.x, xMin - stop.x);
  const dxMax = Math.min(xMax - start.x, xMax - stop.x);
  const dyMin = Math.max(yMin - start.y, yMin - stop.y);
  const dyMax = Math.min(yMax - start.y, yMax - stop.y);
  const dx = clampNumber(dxRaw, dxMin, dxMax);
  const dy = clampNumber(dyRaw, dyMin, dyMax);
  return {
    start: { x: start.x + dx, y: start.y + dy },
    stop: { x: stop.x + dx, y: stop.y + dy },
  };
}

function segmentWidth(start: { x: number; y: number }, stop: { x: number; y: number }): number {
  return Math.max(1, Math.ceil(Math.hypot(stop.x - start.x, stop.y - start.y)) + 1);
}

function sampleVolumeBilinear(
  volume: Float32Array,
  nx: number,
  ny: number,
  nz: number,
  z: number,
  x: number,
  y: number,
): number {
  if (z < 0 || z >= nz || x < 0 || y < 0 || x > nx - 1 || y > ny - 1) return Number.NaN;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(nx - 1, x0 + 1);
  const y1 = Math.min(ny - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;
  const base = z * ny * nx;
  const v00 = volume[base + y0 * nx + x0];
  const v10 = volume[base + y0 * nx + x1];
  const v01 = volume[base + y1 * nx + x0];
  const v11 = volume[base + y1 * nx + x1];
  return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
}

function extractOblique(
  volume: Float32Array,
  nx: number,
  ny: number,
  nz: number,
  start: { x: number; y: number },
  stop: { x: number; y: number },
): Float32Array {
  const width = segmentWidth(start, stop);
  const out = new Float32Array(nz * width);
  const denom = Math.max(1, width - 1);
  for (let z = 0; z < nz; z++) {
    for (let col = 0; col < width; col++) {
      const t = col / denom;
      const x = start.x + (stop.x - start.x) * t;
      const y = start.y + (stop.y - start.y) * t;
      const value = sampleVolumeBilinear(volume, nx, ny, nz, z, x, y);
      out[z * width + col] = Number.isFinite(value) ? value : 0;
    }
  }
  return out;
}

// Copies a row-major slice into the top-left of a zero-filled power-of-two
// grid, the input layout the FFT needs.
function zeroPadSlice(data: Float32Array, sliceW: number, sliceH: number, paddedW: number, paddedH: number): Float32Array {
  const padded = new Float32Array(paddedW * paddedH);
  for (let row = 0; row < sliceH; row++) for (let col = 0; col < sliceW; col++) padded[row * paddedW + col] = data[row * sliceW + col];
  return padded;
}

function extractVolumeFloat32(
  dataView: DataView | ArrayBuffer | Uint8Array,
  offline: boolean,
  offlineMin: number,
  offlineMax: number,
  nx: number,
  ny: number,
  nz: number,
  panelCount = 1,
): Float32Array | null {
  const panels = Math.max(1, Math.floor(panelCount || 1));
  const count = Math.max(0, Math.floor(nx) * Math.floor(ny) * Math.floor(nz) * panels);
  if (!offline) return extractFloat32(dataView, count);
  const bytes = extractBytes(dataView);
  if (bytes.length === 0 || count === 0) return null;
  const usable = Math.min(count, bytes.length);
  const lo = Number.isFinite(offlineMin) ? offlineMin : 0;
  const hi = Number.isFinite(offlineMax) ? offlineMax : lo;
  const out = new Float32Array(count);
  dequantizeUint8(bytes.subarray(0, usable), lo, hi, out);
  if (usable < count) out.fill(lo, usable);
  return out;
}

function makeExportFilename(title: string, nz: number, ny: number, nx: number, mode: string): string {
  const slug = exportTitleSlug(title, "show3dslices");
  const suffix = mode === "quantized" ? "quantized" : "exact";
  return `${slug}_${nz}x${ny}x${nx}_${suffix}.html`;
}

function reverseLut(lut: Uint8Array): Uint8Array {
  const out = new Uint8Array(lut.length);
  const entryCount = lut.length / 3;
  for (let i = 0; i < entryCount; i++) {
    const src = (entryCount - 1 - i) * 3;
    const dst = i * 3;
    out[dst + 0] = lut[src + 0];
    out[dst + 1] = lut[src + 1];
    out[dst + 2] = lut[src + 2];
  }
  return out;
}

function maybeFlip(data: Float32Array, flip: boolean): Float32Array {
  if (!flip) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = -data[i];
  return out;
}

/**
 * The contrast window every slice is painted with, on WebGPU and on Canvas2D:
 * the image_vmin_pct to image_vmax_pct share of the stack-wide display range
 * ([vmin, vmax] when given; otherwise Auto snaps the percentiles to the
 * volume's 2/98 percentiles), negated when the display is flipped. One window
 * for the whole volume keeps contrast fixed while scrubbing and makes both
 * render paths pick the same LUT entry.
 */
function volumeDisplayWindow(
  range: { min: number; max: number },
  vminPct: number,
  vmaxPct: number,
  flip: boolean,
): { vmin: number; vmax: number } {
  const window = vminPct > 0 || vmaxPct < 100
    ? sliderRange(range.min, range.max, vminPct, vmaxPct)
    : { vmin: range.min, vmax: range.max };
  return flip ? { vmin: -window.vmax, vmax: -window.vmin } : window;
}

function makeHistogramSample(data: Float32Array | null, target = 1_000_000): Float32Array | null {
  if (!data || data.length === 0) return null;
  if (data.length <= target) return data;
  const stride = Math.ceil(data.length / target);
  const out = new Float32Array(Math.ceil(data.length / stride));
  for (let src = 0, dst = 0; src < data.length; src += stride, dst++) out[dst] = data[src];
  return out;
}

function clampCanvasTarget(value: number): number {
  return Math.max(MIN_CANVAS_TARGET, Math.min(MAX_CANVAS_TARGET, Math.round(value)));
}

// Raster size of one slice panel before z stretch: the top panel fits its longer
// side to the target, a depth panel takes the full target width and one canvas
// row per slice so its height does not follow the oblique profile length.
function panelRasterSize([sliceH, sliceW]: [number, number], isDepth: boolean, target: number) {
  const width = isDepth ? target : Math.round(sliceW * (target / Math.max(sliceW, sliceH)));
  const height = isDepth ? Math.max(1, sliceH) : Math.round(sliceH * (target / Math.max(sliceW, sliceH)));
  return { w: width, h: height, scaleX: width / Math.max(1, sliceW), scaleY: height / Math.max(1, sliceH) };
}

function transformDisplaySample(data: Float32Array | null, logScale: boolean, flip: boolean): Float32Array | null {
  if (!data) return null;
  if (!logScale && !flip) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const value = logScale ? signedLog1p(data[i]) : data[i];
    out[i] = flip ? -value : value;
  }
  return out;
}

interface LiveNumberSliderProps {
  value: number;
  min: number;
  max: number;
  step: number;
  onLiveChange: (value: number) => void;
  onCommit: (value: number) => void;
  size?: "small" | "medium";
  valueLabelDisplay?: "auto" | "on" | "off";
  sx?: React.ComponentProps<typeof Slider>["sx"];
  ariaLabel: string;
}

const LiveNumberSlider = React.memo(function LiveNumberSlider({
  value, min, max, step, onLiveChange, onCommit, size = "small", valueLabelDisplay = "auto", sx, ariaLabel,
}: LiveNumberSliderProps) {
  const [liveValue, setLiveValue] = React.useState(value);
  React.useEffect(() => { setLiveValue(value); }, [value]);
  return (
    <Slider
      value={liveValue}
      min={min}
      max={max}
      step={step}
      onChange={(_, sliderValue) => {
        const next = sliderValue as number;
        setLiveValue(next);
        onLiveChange(next);
      }}
      onChangeCommitted={(_, sliderValue) => {
        const next = sliderValue as number;
        setLiveValue(next);
        onCommit(next);
      }}
      size={size}
      valueLabelDisplay={valueLabelDisplay}
      sx={sx}
      aria-label={ariaLabel}
    />
  );
});

interface NumberCommitInputProps {
  value: number;
  min: number;
  max: number;
  step?: number;
  onLiveChange?: (value: number) => void;
  onCommit: (value: number) => void;
  ariaLabel: string;
}

// Status line for a fitted shift, e.g. "Aligned row +0.120 px/slice, col -0.050 px/slice (WebGPU)".
function alignedStatusText(rowShift: number, colShift: number, source: string): string {
  return `Aligned row ${rowShift >= 0 ? "+" : ""}${rowShift.toFixed(3)} px/slice, `
    + `col ${colShift >= 0 ? "+" : ""}${colShift.toFixed(3)} px/slice (${source})`;
}

function formatNumberInput(value: number): string {
  if (!Number.isFinite(value)) return "0";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(3);
}

const NumberCommitInput = React.memo(function NumberCommitInput({
  value, min, max, step = 0.05, onLiveChange, onCommit, ariaLabel,
}: NumberCommitInputProps) {
  const [draft, setDraft] = React.useState(formatNumberInput(value));
  React.useEffect(() => { setDraft(formatNumberInput(value)); }, [value]);
  const commitDraft = () => {
    const rawNext = Number(draft);
    if (!Number.isFinite(rawNext)) {
      setDraft(formatNumberInput(value));
      return;
    }
    const next = clampNumber(rawNext, min, max);
    setDraft(formatNumberInput(next));
    onCommit(next);
  };
  return (
    <Box
      component="input"
      type="number"
      value={draft}
      min={min}
      max={max}
      step={step}
      aria-label={ariaLabel}
      onChange={(event) => {
        const nextDraft = event.currentTarget.value;
        setDraft(nextDraft);
        const next = Number(nextDraft);
        if (Number.isFinite(next)) onLiveChange?.(clampNumber(next, min, max));
      }}
      onBlur={commitDraft}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        } else if (event.key === "Escape") {
          setDraft(formatNumberInput(value));
          event.currentTarget.blur();
        }
      }}
      sx={{
        width: 48,
        height: 20,
        boxSizing: "border-box",
        px: 0.5,
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 0.5,
        bgcolor: "background.paper",
        color: "text.primary",
        fontSize: 10,
        fontFamily: "monospace",
        textAlign: "right",
        "&:focus": { outline: "1px solid", outlineColor: "primary.main" },
      }}
    />
  );
});

const controlLabel = { ...typography.label, ...typographyLabel };
const clickableControlLabel = {
  ...controlLabel,
  cursor: "pointer",
  userSelect: "none",
} as const;

const controlPanel = {
  // flexShrink 0 + overflow visible so the view label ("Top"/"Side") is never
  // compressed below its width and truncated to "S..." in a narrow column.
  select: { minWidth: 96, flexShrink: 0, fontSize: 11, "& .MuiSelect-select": { py: 0.5, textOverflow: "clip", overflow: "visible" } },
};

const container = {
  // overflowX:auto so panels stay reachable via horizontal scroll on narrow
  // viewport instead of being silently clipped past the cell edge.
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", overflowX: "auto", overflowY: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

// The volume-slice shader caps its per-pixel average at this zoom, so the view never zooms out further.
const MIN_ZOOM = VOLUME_SLICE_MIN_ZOOM;
const MAX_ZOOM = 30;
type ZoomState = { zoom: number; panX: number; panY: number };
const DEFAULT_ZOOM: ZoomState = { zoom: 1, panX: 0, panY: 0 };
const DEFAULT_FFT_ZOOM: ZoomState = { zoom: 2, panX: 0, panY: 0 };

// Zooms one notch about the pointer so the image pixel under it stays put.
function wheelZoomAboutPointer(canvas: HTMLCanvasElement, event: React.WheelEvent, view: ZoomState): ZoomState {
  const rect = canvas.getBoundingClientRect();
  const mouseX = (event.clientX - rect.left) * (canvas.width / rect.width);
  const mouseY = (event.clientY - rect.top) * (canvas.height / rect.height);
  const centerX = canvas.width / 2, centerY = canvas.height / 2;
  const imageX = (mouseX - centerX - view.panX) / view.zoom + centerX;
  const imageY = (mouseY - centerY - view.panY) / view.zoom + centerY;
  const factor = event.deltaY > 0 ? 0.9 : 1.1;
  const zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, view.zoom * factor));
  return { zoom, panX: mouseX - (imageX - centerX) * zoom - centerX, panY: mouseY - (imageY - centerY) * zoom - centerY };
}

// Blits a cached raster onto its panel canvas under the panel's zoom and pan,
// scaling about the canvas centre. The source rect is the offscreen's actual
// size: the GPU path can render it smaller than the slice, and reading the
// slice dimensions instead would sample a partly-empty buffer.
function drawZoomedOffscreen(
  ctx: CanvasRenderingContext2D,
  offscreen: HTMLCanvasElement,
  canvasW: number,
  canvasH: number,
  view: ZoomState,
  smooth: boolean,
): void {
  ctx.imageSmoothingEnabled = smooth;
  ctx.clearRect(0, 0, canvasW, canvasH);
  if (view.zoom !== 1 || view.panX !== 0 || view.panY !== 0) {
    ctx.save();
    const centerX = canvasW / 2, centerY = canvasH / 2;
    ctx.translate(centerX + view.panX, centerY + view.panY);
    ctx.scale(view.zoom, view.zoom);
    ctx.translate(-centerX, -centerY);
    ctx.drawImage(offscreen, 0, 0, offscreen.width, offscreen.height, 0, 0, canvasW, canvasH);
    ctx.restore();
  } else {
    ctx.drawImage(offscreen, 0, 0, offscreen.width, offscreen.height, 0, 0, canvasW, canvasH);
  }
}
const CANVAS_TARGET = 480;
const MIN_CANVAS_TARGET = 300;
const MAX_CANVAS_TARGET = 800;
const SLICE_PANEL_TOP_ALIGN_PX = 20;
const AXES = ["xy", "oblique"] as const;
const PANEL_NAMES = ["XY", "Oblique"] as const;
// Show3DSlices opens in the same orientation as the main top slice panel:
// x/columns left-to-right and y/rows top-to-bottom.
const SHOW3DSLICES_DEFAULT_CAMERA: CameraState = {
  ...DEFAULT_CAMERA,
  yaw: Math.PI,
  pitch: 0,
  roll: Math.PI,
};
const VOLUME_VIEW_PRESETS = [
  { value: "xy", label: "Top", description: "top (XY) view" },
  { value: "side", label: "Side", description: "oblique vertical plane view" },
] as const;
const DPR = window.devicePixelRatio || 1;
const SHOW3DSLICES_FFT_RESULT_CACHE_MAX_BYTES = 192 * 1024 * 1024;
const SHOW3DSLICES_FFT_RESULT_CACHE_MAX_ENTRIES = 24;

const FFT_SNAP_RADIUS = 5;

function Show3DSlices() {
  const model = useModel();
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);

  // Theme detection (offline HTML exports force a light/white background)
  const [offlineForTheme] = useModelState<boolean>("_export_light");
  const { themeInfo, colors: baseColors } = useTheme(offlineForTheme);
  const themeColors = {
    ...baseColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#b08800",
  };

  const themedSelect = {
    ...controlPanel.select,
    borderRadius: 0,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderRadius: 0, borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { borderRadius: 0, bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } },
  };

  const [nx] = useModelState<number>("nx");
  const [ny] = useModelState<number>("ny");
  const [nz] = useModelState<number>("nz");
  const [panelCount] = useModelState<number>("panel_count");
  const [activePanel, setActivePanel] = useModelState<number>("active_panel");
  const [panelTitles] = useModelState<string[]>("panel_titles");
  const [nPages] = useModelState<number>("n_pages");
  const [pageIdx, setPageIdx] = useModelState<number>("page_idx");
  const [panelsPerPage] = useModelState<number>("panels_per_page");
  const [pageLabels] = useModelState<string[]>("page_labels");
  const [pagePlaying, setPagePlaying] = React.useState(false);
  const [pagePlayFps, setPagePlayFps] = React.useState<number>(2);
  const [volumeBytes] = useModelState<DataView>("volume_bytes");
  const [offline] = useModelState<boolean>("offline");
  const [offlineMin] = useModelState<number>("_offline_min");
  const [offlineMax] = useModelState<number>("_offline_max");
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportPayloadFilename] = useModelState<string>("export_filename");
  const [sliceX, setSliceX] = useModelState<number>("slice_x");
  const [sliceY, setSliceY] = useModelState<number>("slice_y");
  const [sliceZ, setSliceZ] = useModelState<number>("slice_z");
  const [obliqueAngle, setObliqueAngle] = useModelState<number>("oblique_angle");
  const [obliqueProfileLine, setObliqueProfileLine] = useModelState<{ row: number; col: number }[]>("oblique_profile_line");
  const [title] = useModelState<string>("title");
  const [showTitle] = useModelState<boolean>("show_title");
  const safePanelCount = Math.max(1, Math.floor(panelCount || 1));
  const safeNPages = Math.max(1, Math.floor(nPages || 1));
  const safePanelsPerPage = Math.max(0, Math.floor(panelsPerPage || 0));
  const isPaged = safeNPages > 1 && safePanelsPerPage > 0;
  const safePageIdx = Math.max(0, Math.min(Math.floor(pageIdx || 0), safeNPages - 1));
  const pageStart = isPaged ? safePageIdx * safePanelsPerPage : 0;
  const pageEnd = isPaged
    ? Math.min(safePanelCount, pageStart + safePanelsPerPage)
    : safePanelCount;
  const pagePanelIndices = React.useMemo(
    () => Array.from({ length: Math.max(0, pageEnd - pageStart) }, (_, idx) => pageStart + idx),
    [pageEnd, pageStart]
  );
  const absoluteActivePanel = Math.max(0, Math.min(Math.floor(activePanel || 0), safePanelCount - 1));
  const activePanelSlot = isPaged
    ? Math.max(0, Math.min(absoluteActivePanel % safePanelsPerPage, Math.max(0, pagePanelIndices.length - 1)))
    : absoluteActivePanel;
  const safeActivePanel = isPaged ? pageStart + activePanelSlot : absoluteActivePanel;
  const effectivePanelTitles = Array.isArray(panelTitles) ? panelTitles : [];
  const activePanelTitle = effectivePanelTitles[safeActivePanel] || `Panel ${safeActivePanel + 1}`;
  const effectivePageLabels = Array.isArray(pageLabels) ? pageLabels : [];
  const activePageLabel = effectivePageLabels[safePageIdx] || `Page ${safePageIdx + 1}`;
  const pageStatus = `${activePageLabel} ${safePageIdx + 1}/${safeNPages}`;
  const displayTitle = isPaged
    ? `${title || "Volume 3D"} · ${activePageLabel}${pagePanelIndices.length > 1 ? ` · ${activePanelTitle}` : ""}`
    : safePanelCount > 1
      ? `${title || "Volume 3D"}: ${activePanelTitle}`
      : (title || "Volume 3D");
  React.useEffect(() => {
    if (pageIdx !== safePageIdx) setPageIdx(safePageIdx);
    if (activePanel !== safeActivePanel) setActivePanel(safeActivePanel);
  }, [activePanel, pageIdx, safeActivePanel, safePageIdx, setActivePanel, setPageIdx]);
  const showPage = React.useCallback((value: number) => {
    const next = Math.max(0, Math.min(Math.round(value), safeNPages - 1));
    setPagePlaying(false);
    setPageIdx(next);
    if (isPaged) {
      setActivePanel(Math.min(safePanelCount - 1, next * safePanelsPerPage + activePanelSlot));
    }
  }, [activePanelSlot, isPaged, safeNPages, safePanelCount, safePanelsPerPage, setActivePanel, setPageIdx]);
  React.useEffect(() => {
    if (!isPaged || safeNPages <= 1) setPagePlaying(false);
  }, [isPaged, safeNPages]);
  React.useEffect(() => {
    if (!pagePlaying || !isPaged || safeNPages <= 1) return;
    const timeout = window.setTimeout(() => {
      const next = (safePageIdx + 1) % safeNPages;
      setPageIdx(next);
      setActivePanel(Math.min(safePanelCount - 1, next * safePanelsPerPage + activePanelSlot));
    }, 1000 / Math.max(1, pagePlayFps));
    return () => window.clearTimeout(timeout);
  }, [activePanelSlot, isPaged, pagePlayFps, pagePlaying, safeNPages, safePageIdx, safePanelCount, safePanelsPerPage, setActivePanel, setPageIdx]);
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [traitVmin] = useModelState<number | null>("vmin");
  const [traitVmax] = useModelState<number | null>("vmax");
  const [showControls] = useModelState<boolean>("show_controls");
  const [controlsCollapsed, setControlsCollapsed] = useModelState<boolean>("controls_collapsed");
  const controlsVisible = showControls && !controlsCollapsed;
  const [showCrosshair] = useModelState<boolean>("show_crosshair");
  const [sliceAlignment, setSliceAlignment] = useModelState<string>("slice_alignment");
  const [rowShiftPxPerSlice, setRowShiftPxPerSlice] = useModelState<number>("row_shift_px_per_slice");
  const [colShiftPxPerSlice, setColShiftPxPerSlice] = useModelState<number>("col_shift_px_per_slice");
  const [sliceAlignmentCached, setSliceAlignmentCached] = useModelState<boolean>("slice_alignment_cached");
  const [sliceAlignmentStatus] = useModelState<string>("slice_alignment_status");
  const [, setSliceAlignmentRequest] = useModelState<string>("_slice_alignment_request");
  const [panelWidthPx] = useModelState<number>("panel_width_px");
  type Show3DSlicesViewState = {
    zooms?: Partial<ZoomState>[];
    fft_zooms?: Partial<ZoomState>[];
    camera?: Partial<CameraState>;
    canvas_target?: number;
    side_canvas_target?: number;
    volume_canvas_size?: number;
  };
  type Show3DSlicesViewSizes = {
    canvasTarget: number;
    sideCanvasTarget: number;
    volumeCanvasSize: number;
  };
  const [viewState, setViewState] = useModelState<Show3DSlicesViewState>("view_state");
  const readNumber = (value: unknown, fallback: number): number => (
    typeof value === "number" && Number.isFinite(value) ? value : fallback
  );
  const readCanvasTarget = (value: unknown, fallback: number): number => clampCanvasTarget(readNumber(value, fallback));
  const normalizeZoomState = (value: Partial<ZoomState> | undefined, fallback: ZoomState): ZoomState => ({
    zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, readNumber(value?.zoom, fallback.zoom))),
    panX: readNumber(value?.panX, readNumber((value as { pan_x?: unknown } | undefined)?.pan_x, fallback.panX)),
    panY: readNumber(value?.panY, readNumber((value as { pan_y?: unknown } | undefined)?.pan_y, fallback.panY)),
  });
  const normalizeCameraState = (value: Partial<CameraState> | undefined): CameraState => ({
    ...SHOW3DSLICES_DEFAULT_CAMERA,
    yaw: readNumber(value?.yaw, SHOW3DSLICES_DEFAULT_CAMERA.yaw),
    pitch: Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, readNumber(value?.pitch, SHOW3DSLICES_DEFAULT_CAMERA.pitch))),
    roll: readNumber(value?.roll, SHOW3DSLICES_DEFAULT_CAMERA.roll ?? 0),
    distance: Math.max(0.5, Math.min(10, readNumber(value?.distance, SHOW3DSLICES_DEFAULT_CAMERA.distance))),
    panX: readNumber(value?.panX, readNumber((value as { pan_x?: unknown } | undefined)?.pan_x, SHOW3DSLICES_DEFAULT_CAMERA.panX)),
    panY: readNumber(value?.panY, readNumber((value as { pan_y?: unknown } | undefined)?.pan_y, SHOW3DSLICES_DEFAULT_CAMERA.panY)),
  });
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [orthographic, setOrthographic] = useModelState<boolean>("orthographic");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const [flip, setFlip] = useModelState<boolean>("flip");
  const [dimLabels] = useModelState<string[]>("dim_labels");
  const [pixelSize] = useModelState<number>("pixel_size");
  // Per-axis sampling [pz, py, px] for anisotropic data; falls back to [pixelSize]*3.
  const [pixelSizeAxes] = useModelState<number[]>("pixel_size_axes");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [modelZStretch, setModelZStretch] = useModelState<number>("z_stretch");
  const [zStretch, setZStretch] = React.useState(modelZStretch);
  const pendingZStretchRef = React.useRef(modelZStretch);
  const zStretchLiveDirtyRef = React.useRef(false);
  const zStretchRafRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    zStretchLiveDirtyRef.current = false;
    pendingZStretchRef.current = modelZStretch;
    setZStretch(modelZStretch);
  }, [modelZStretch]);
  React.useEffect(() => {
    return () => {
      if (zStretchRafRef.current != null) cancelAnimationFrame(zStretchRafRef.current);
    };
  }, []);

  React.useEffect(() => {
    let disposed = false;
    getDisplayFFT().then(fft => {
      if (fft) { gpuFFTRef.current = fft; setGpuReady(true); }
    });
    // Colormap engine: volume-resident GPU slice + colormap (no CPU per-scrub work).
    createGPUColormapEngine().then(engine => {
      if (disposed) { engine?.destroy(); return; }
      if (engine) { gpuCmapRef.current = engine; setCmapReady(true); }
    });
    return () => { disposed = true; gpuCmapRef.current?.destroy(); gpuCmapRef.current = null; volUploadedKeyRef.current = null; };
  }, []);

  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const uiRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const imageBoxRefs = React.useRef<(HTMLDivElement | null)[]>([null, null, null]);

  const [fftColormap, setFftColormap] = useModelState<string>("fft_colormap");
  const [fftLogScale, setFftLogScale] = useModelState<boolean>("fft_log_scale");
  const [fftAuto, setFftAuto] = useModelState<boolean>("fft_auto");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const savedSliceZooms = Array.from({ length: 3 }, (_, i) => normalizeZoomState(viewState?.zooms?.[i], DEFAULT_ZOOM));
  const savedFftZooms = Array.from({ length: 3 }, (_, i) => normalizeZoomState(viewState?.fft_zooms?.[i], DEFAULT_FFT_ZOOM));
  const [fftZooms, setFftZooms] = React.useState<ZoomState[]>(() => savedFftZooms);
  const [fftDragAxis, setFftDragAxis] = React.useState<number | null>(null);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);

  // FFT d-spacing measurement
  type FftClickInfo = {
    axis: number; row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  };
  const [fftClickInfo, setFftClickInfo] = React.useState<FftClickInfo | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number; axis: number } | null>(null);
  const fftCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftOverlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftOffscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftImgDataRefs = React.useRef<(ImageData | null)[]>([null, null, null]);
  const fftMagCacheRefs = React.useRef<(Float32Array | null)[]>([null, null, null]);
  type FftResultCacheEntry = {
    mag: Float32Array;
    displayData: Float32Array;
    displayMin: number;
    displayMax: number;
    width: number;
    height: number;
    bytes: number;
    lastUsed: number;
  };
  const fftResultCacheRef = React.useRef<Map<string, FftResultCacheEntry>>(new Map());
  const fftResultCacheSeqRef = React.useRef(0);
  const fftResultCacheBytesRef = React.useRef(0);
  const fftDataObjectRef = React.useRef<Float32Array | null>(null);
  const fftDataTokenRef = React.useRef(0);
  const gpuFFTRef = React.useRef<DisplayFFT | null>(null);
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const [cmapReady, setCmapReady] = React.useState(false);
  const volUploadedKeyRef = React.useRef<Float32Array | null>(null);
  const gpuVolReadyRef = React.useRef(false);
  // Live params snapshot for direct-paint (slider handler bypasses React).
  const paintParamsRef = React.useRef<{
    cmap: string; logScale: boolean; flip: boolean;
    imageVminPct: number; imageVmaxPct: number; imageDataRange: { min: number; max: number };
    traitVmin: number | null; traitVmax: number | null;
    zooms: { zoom: number; panX: number; panY: number }[]; canvasSizes: { w: number; h: number }[]; smooth: boolean;
    alignment?: { rowShift: number; colShift: number; segment: { start: { x: number; y: number }; stop: { x: number; y: number } } };
  } | null>(null);
  const fftComputeGenerationRef = React.useRef(0);
  const [gpuReady, setGpuReady] = React.useState(false);
  // Counter to trigger FFT redraw after async compute finishes
  const [fftVersion, setFftVersion] = React.useState(0);

  const [zooms, setZooms] = React.useState<ZoomState[]>(() => savedSliceZooms);
  const [dragAxis, setDragAxis] = React.useState<number | null>(null);
  const [dragStart, setDragStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);
  // rAF bypass: keep live zoom in ref during drag, sync to React state on mouseup.
  // Only sync ref from state when NOT dragging - otherwise an unrelated re-render
  // (playback tick, cursor update) would clobber in-flight pan values.
  const liveZoomsRef = React.useRef<ZoomState[]>(savedSliceZooms);
  const liveZoomDirtyRef = React.useRef(false);
  if (dragAxis === null && !liveZoomDirtyRef.current) liveZoomsRef.current = zooms;
  const zoomRafRef = React.useRef<number>(0);
  const zoomCommitTimeoutRef = React.useRef<number | null>(null);
  const liveFftZoomsRef = React.useRef<ZoomState[]>(savedFftZooms);
  const liveFftZoomDirtyRef = React.useRef(false);
  if (fftDragAxis === null && !liveFftZoomDirtyRef.current) liveFftZoomsRef.current = fftZooms;
  const fftZoomRafRef = React.useRef<number>(0);
  const fftZoomCommitTimeoutRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    return () => {
      if (zoomCommitTimeoutRef.current != null) window.clearTimeout(zoomCommitTimeoutRef.current);
      if (fftZoomCommitTimeoutRef.current != null) window.clearTimeout(fftZoomCommitTimeoutRef.current);
    };
  }, []);

  const initialPanelDefault = panelWidthPx > 0 ? panelWidthPx : CANVAS_TARGET;
  const initialCanvasTarget = readCanvasTarget(viewState?.canvas_target, initialPanelDefault);
  const initialSideCanvasTarget = readCanvasTarget(viewState?.side_canvas_target, initialPanelDefault);
  const initialVolumeCanvasSize = readCanvasTarget(viewState?.volume_canvas_size, initialPanelDefault);
  const [canvasTarget, setCanvasTarget] = React.useState(initialCanvasTarget);
  const [sideCanvasTarget, setSideCanvasTarget] = React.useState(initialSideCanvasTarget);
  const canvasTargetRef = React.useRef(initialCanvasTarget);
  const sideCanvasTargetRef = React.useRef(initialSideCanvasTarget);
  canvasTargetRef.current = canvasTarget;
  sideCanvasTargetRef.current = sideCanvasTarget;
  React.useEffect(() => {
    if (panelWidthPx > 0) {
      if (viewState?.canvas_target == null) setCanvasTarget(clampCanvasTarget(panelWidthPx));
      if (viewState?.side_canvas_target == null) setSideCanvasTarget(clampCanvasTarget(panelWidthPx));
    }
  }, [panelWidthPx, viewState?.canvas_target, viewState?.side_canvas_target]);
  const [isResizing, setIsResizing] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number; target: "primary" | "side" } | null>(null);

  const [playing, setPlaying] = useModelState<boolean>("playing");
  const [playAxis, setPlayAxis] = useModelState<number>("play_axis");
  const playbackAxis = playAxis === 0 || playAxis === 3 ? playAxis : 1;
  React.useEffect(() => {
    if (playAxis !== playbackAxis) setPlayAxis(playbackAxis);
  }, [playAxis, playbackAxis, setPlayAxis]);
  const [reverse, setReverse] = useModelState<boolean>("reverse");
  const [modelFps, setModelFps] = useModelState<number>("fps");
  const [fps, setFps] = React.useState(() => Math.max(1, Math.min(MAX_PLAYBACK_FPS, modelFps)));
  const fpsRef = React.useRef(Math.max(1, Math.min(MAX_PLAYBACK_FPS, modelFps)));
  React.useEffect(() => {
    const capped = Math.max(1, Math.min(MAX_PLAYBACK_FPS, modelFps));
    fpsRef.current = capped;
    setFps(capped);
  }, [modelFps]);
  const [loop, setLoop] = useModelState<boolean>("loop");
  const playRafRef = React.useRef<number | null>(null);
  const lastPlayTsRef = React.useRef<number | null>(null);
  const playAccumulatorRef = React.useRef(0);
  const [boomerang, setBoomerang] = useModelState<boolean>("boomerang");
  const bounceDirRef = React.useRef<1 | -1>(1);
  const [loopStarts, setLoopStarts] = React.useState([0, 0, 0]);
  const [loopEnds, setLoopEnds] = React.useState([-1, -1, -1]);
  const loopStartsRef = React.useRef(loopStarts);
  const loopEndsRef = React.useRef(loopEnds);
  const pendingLoopRangeRef = React.useRef<{ starts: number[]; ends: number[] } | null>(null);
  const loopRangeRafRef = React.useRef<number | null>(null);
  React.useEffect(() => { loopStartsRef.current = loopStarts; }, [loopStarts]);
  React.useEffect(() => { loopEndsRef.current = loopEnds; }, [loopEnds]);
  React.useEffect(() => () => {
    if (loopRangeRafRef.current != null) cancelAnimationFrame(loopRangeRafRef.current);
  }, []);
  const fastTrackSliceRef = React.useRef<((axis: number, value: number) => void) | null>(null);
  const commitSliceValuesRef = React.useRef<() => void>(() => {});
  const pausePlaybackForEdit = React.useCallback(() => {
    if (playRafRef.current != null) {
      cancelAnimationFrame(playRafRef.current);
      playRafRef.current = null;
    }
    lastPlayTsRef.current = null;
    playAccumulatorRef.current = 0;
    if (playing) setPlaying(false);
  }, [playing, setPlaying]);

  const volumeCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const volumeRendererRef = React.useRef<VolumeRenderer | null>(null);
  const [camera, setCamera] = React.useState<CameraState>(() => normalizeCameraState(viewState?.camera));
  const [volumeDrag, setVolumeDrag] = React.useState<{
    button: number; x: number; y: number; yaw: number; pitch: number; panX: number; panY: number;
  } | null>(null);
  const [webgpuSupported, setWebgpuSupported] = React.useState(true);
  const [volumeInitError, setVolumeInitError] = React.useState<string>("");
  const [rendererReady, setRendererReady] = React.useState(0);
  const [volumeCanvasSize, setVolumeCanvasSize] = React.useState(initialVolumeCanvasSize);
  const volumeCanvasSizeRef = React.useRef(initialVolumeCanvasSize);
  volumeCanvasSizeRef.current = volumeCanvasSize;
  React.useEffect(() => {
    if (panelWidthPx > 0 && viewState?.volume_canvas_size == null) setVolumeCanvasSize(clampCanvasTarget(panelWidthPx));
  }, [panelWidthPx, viewState?.volume_canvas_size]);
  const [volumeResizing, setVolumeResizing] = React.useState(false);
  const volumeResizeStartRef = React.useRef<{ x: number; y: number; size: number } | null>(null);
  const [showSlicePlanes, setShowSlicePlanes] = useModelState<boolean | undefined>("show_slice_planes");
  const [planeVisibility, setPlaneVisibility] = useModelState<boolean[] | undefined>("plane_visibility");
  const normalizedPlaneVisibility = PLANE_KEYS.map((_, i) => Boolean(planeVisibility?.[i] ?? showSlicePlanes ?? true));
  const visiblePlanes = PLANE_KEYS.filter((_, i) => normalizedPlaneVisibility[i]);
  const slicePlaneMask = normalizedPlaneVisibility.reduce((mask, visible, i) => (
    visible ? mask | (1 << i) : mask
  ), 0);
  const anySlicePlaneVisible = slicePlaneMask !== 0;

  const [imageVminPct, setImageVminPct] = useModelState<number>("image_vmin_pct");
  const [imageVmaxPct, setImageVmaxPct] = useModelState<number>("image_vmax_pct");
  const manualImageRangeBeforeAutoRef = React.useRef<{ min: number; max: number } | null>(null);
  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);

  const [opacityA, setOpacityA] = useModelState<number>("volume_opacity");
  const [slicePlaneOpacity, setSlicePlaneOpacity] = useModelState<number>("slice_plane_opacity");
  const pendingVolumeControlsRef = React.useRef({ opacity: opacityA, slicePlaneOpacity });
  const volumeControlsRafRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    return () => {
      if (volumeControlsRafRef.current != null) cancelAnimationFrame(volumeControlsRafRef.current);
    };
  }, []);

  // Cached offscreen canvases for slice rendering (avoids recomputing colormap on zoom/pan)
  const sliceOffscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  // Reusable ImageData per axis to avoid GC churn (allocated once per dimension change)
  const sliceImgDataRefs = React.useRef<(ImageData | null)[]>([null, null, null]);

  const [showColorbar, setShowColorbar] = useModelState<boolean>("show_colorbar");
  const [exportMenuAnchor, setExportMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [exportBusy, setExportBusy] = React.useState(false);
  const [localExportStatus, setLocalExportStatus] = React.useState("");
  const [advancedControlsOpen, setAdvancedControlsOpen] = React.useState(false);
  const [localAlignmentStatus, setLocalAlignmentStatus] = React.useState("");
  const [liveRowShift, setLiveRowShift] = React.useState(rowShiftPxPerSlice || 0);
  const [liveColShift, setLiveColShift] = React.useState(colShiftPxPerSlice || 0);
  const pendingAlignmentRef = React.useRef<{ rowShift: number; colShift: number } | null>(null);
  const alignmentRafRef = React.useRef<number | null>(null);
  const alignmentPreviewRef = React.useRef<((rowShift: number, colShift: number) => void) | null>(null);
  const sliceAlignmentModeRef = React.useRef(sliceAlignment || "off");
  const sliceAlignmentCachedRef = React.useRef(!!sliceAlignmentCached);
  const autoAlignmentRef = React.useRef({
    cached: !!sliceAlignmentCached || Math.abs(rowShiftPxPerSlice || 0) >= 1e-12 || Math.abs(colShiftPxPerSlice || 0) >= 1e-12,
    rowShift: rowShiftPxPerSlice || 0,
    colShift: colShiftPxPerSlice || 0,
  });
  const pendingExportRef = React.useRef<{
    id: string;
    filename: string;
    mode: string;
    handle: Show3DSlicesFileHandle | null;
  } | null>(null);

  React.useEffect(() => {
    if (!exportStatus) return;
    const preparing = exportStatus.startsWith("Preparing ") || exportStatus.startsWith("Exporting ");
    if (preparing) {
      setExportBusy(true);
    } else if (!pendingExportRef.current) {
      setExportBusy(false);
    }
  }, [exportStatus]);
  React.useEffect(() => {
    setLiveRowShift(rowShiftPxPerSlice || 0);
  }, [rowShiftPxPerSlice]);
  React.useEffect(() => {
    setLiveColShift(colShiftPxPerSlice || 0);
  }, [colShiftPxPerSlice]);
  React.useEffect(() => {
    sliceAlignmentModeRef.current = sliceAlignment || "off";
  }, [sliceAlignment]);
  React.useEffect(() => {
    sliceAlignmentCachedRef.current = !!sliceAlignmentCached;
  }, [sliceAlignmentCached]);
  React.useEffect(() => {
    if (sliceAlignment === "auto" && sliceAlignmentCached) {
      autoAlignmentRef.current = {
        cached: true,
        rowShift: rowShiftPxPerSlice || 0,
        colShift: colShiftPxPerSlice || 0,
      };
    }
  }, [sliceAlignment, sliceAlignmentCached, rowShiftPxPerSlice, colShiftPxPerSlice]);
  React.useEffect(() => () => {
    if (alignmentRafRef.current != null) cancelAnimationFrame(alignmentRafRef.current);
  }, []);
  React.useEffect(() => {
    if (!sliceAlignmentStatus) return;
    if (/^(Aligned|Cached) row/.test(sliceAlignmentStatus)) {
      setLocalAlignmentStatus("");
      return;
    }
    setLocalAlignmentStatus(sliceAlignmentStatus);
  }, [sliceAlignmentStatus]);

  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number; view: string } | null>(null);
  const [obliqueHoverTarget, setObliqueHoverTarget] = React.useState<"endpoint" | "line" | null>(null);
  const cursorInfoRef = React.useRef<typeof cursorInfo>(null);
  const pendingCursorInfoRef = React.useRef<typeof cursorInfo>(null);
  const cursorRafRef = React.useRef<number | null>(null);
  const setCursorInfoThrottled = (next: typeof cursorInfo) => {
    pendingCursorInfoRef.current = next;
    if (cursorRafRef.current != null) return;
    cursorRafRef.current = requestAnimationFrame(() => {
      cursorRafRef.current = null;
      const pending = pendingCursorInfoRef.current;
      const prev = cursorInfoRef.current;
      const same = prev === pending || (!!prev && !!pending &&
        prev.row === pending.row && prev.col === pending.col && prev.view === pending.view && prev.value === pending.value);
      if (!same) {
        cursorInfoRef.current = pending;
        setCursorInfo(pending);
      }
    });
  };
  React.useEffect(() => () => {
    if (cursorRafRef.current != null) cancelAnimationFrame(cursorRafRef.current);
  }, []);

  // Parse volume data. Live notebooks receive exact float32 bytes; offline
  // reports receive uint8 bytes plus global min/max metadata to reduce HTML size.
  const allPanelFloats = React.useMemo(
    () => extractVolumeFloat32(volumeBytes, offline, offlineMin, offlineMax, nx, ny, nz, safePanelCount),
    [volumeBytes, offline, offlineMin, offlineMax, nx, ny, nz, safePanelCount],
  );
  const panelFloats = React.useMemo(() => {
    if (!allPanelFloats || allPanelFloats.length === 0) return null;
    const onePanelCount = Math.max(0, Math.floor(nx) * Math.floor(ny) * Math.floor(nz));
    if (onePanelCount === 0) return null;
    if (safePanelCount <= 1) return allPanelFloats;
    const start = Math.min(safeActivePanel * onePanelCount, allPanelFloats.length);
    const stop = Math.min(start + onePanelCount, allPanelFloats.length);
    if (stop <= start) return null;
    return allPanelFloats.subarray(start, stop);
  }, [allPanelFloats, safePanelCount, safeActivePanel, nx, ny, nz]);
  const alignmentMode = (sliceAlignment || "off").toLowerCase();
  const alignmentActive = alignmentMode !== "off";
  React.useEffect(() => {
    if (fftDataObjectRef.current === panelFloats) return;
    fftDataObjectRef.current = panelFloats;
    fftDataTokenRef.current += 1;
    fftResultCacheRef.current.clear();
    fftResultCacheBytesRef.current = 0;
  }, [panelFloats]);
  React.useEffect(() => {
    if (cursorInfoRef.current !== null) {
      cursorInfoRef.current = null;
      setCursorInfo(null);
    }
  }, [safeActivePanel]);
  const obliqueSegment = React.useMemo(() => {
    const startFromState = profilePointFromAny(obliqueProfileLine?.[0]);
    const stopFromState = profilePointFromAny(obliqueProfileLine?.[1]);
    if (startFromState && stopFromState) {
      return {
        start: clampPointToImage(startFromState, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
        stop: clampPointToImage(stopFromState, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
        explicit: true,
      };
    }
    const [start, stop] = obliqueLineEndpoints(nx, ny, sliceX, sliceY, obliqueAngle);
    return {
      start: clampPointToImage(start, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
      stop: clampPointToImage(stop, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
      explicit: false,
    };
  }, [obliqueProfileLine, nx, ny, sliceX, sliceY, obliqueAngle]);
  // The oblique panel's geometry must reach the GPU slice shader on EVERY render,
  // not only while alignment is on: the axis-3 shader walks from segment start to
  // stop, so a missing segment collapses every output column onto voxel (0, 0, z)
  // and the panel paints flat bands that ignore the angle and position sliders.
  // Depth shifts stay zero unless alignment is active.
  const gpuSliceParams = React.useMemo(
    () => ({
      rowShift: alignmentActive ? liveRowShift : 0,
      colShift: alignmentActive ? liveColShift : 0,
      segment: { start: obliqueSegment.start, stop: obliqueSegment.stop },
    }),
    [alignmentActive, liveRowShift, liveColShift, obliqueSegment],
  );
  // SYNCHRONOUS data range (useMemo, not useState+effect). If this lands a frame
  // late, the first render uses the default {0,1} range so a value-based contrast
  // (vmin/vmax) converts to the wrong percent -> secondary planes paint with the
  // wrong contrast ("blue") until a scrub recomputes. Inline makes frame 1 correct.
  const imageDataRange = React.useMemo(
    () => (panelFloats && panelFloats.length > 0 ? findDataRange(panelFloats) : { min: 0, max: 1 }),
    [panelFloats],
  );
  const voxelCount = Math.max(0, Math.floor(nx) * Math.floor(ny) * Math.floor(nz));
  const syncedVoxelCount = voxelCount * safePanelCount;
  const exactExportSize = formatEstimatedHtmlSize(syncedVoxelCount * 4);
  const quantizedExportSize = formatEstimatedHtmlSize(syncedVoxelCount);
  const handleExportMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setExportMenuAnchor(event.currentTarget);
  };
  const handleExportMenuClose = () => {
    setExportMenuAnchor(null);
  };
  const handleExportSelect = async (mode: string) => {
    setExportMenuAnchor(null);
    if (mode !== "exact" && mode !== "quantized") return;
    const filename = makeExportFilename(title, nz, ny, nx, mode);
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setExportBusy(true);
    setLocalExportStatus("Choose export location...");
    const picker = (window as Show3DSlicesWindow).showSaveFilePicker;
    let handle: Show3DSlicesFileHandle | null = null;
    if (picker) {
      try {
        handle = await picker({
          suggestedName: filename,
          types: [{ description: "Standalone HTML", accept: { "text/html": [".html"] } }],
        });
      } catch (err) {
        if (isAbortLikeError(err)) {
          setExportBusy(false);
          setLocalExportStatus("Export canceled");
          return;
        }
        setExportBusy(false);
        setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
    }
    pendingExportRef.current = { id, filename, mode, handle };
    setLocalExportStatus(`Preparing ${filename}...`);
    setExportRequest(JSON.stringify({ mode, id, filename, download: true }));
  };
  const requestSliceAlignmentEstimate = () => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setLocalAlignmentStatus("Estimating slice alignment...");
    setSliceAlignmentRequest(JSON.stringify({ mode: "estimate", id, panel: safeActivePanel }));
  };
  const alignmentEstimateBusyRef = React.useRef(false);
  /**
   * Fit the global row/col drift from the volume already loaded in the browser.
   * Runs the same algorithm as the kernel estimator, so an exported HTML can
   * align a stack with no Python attached; falls back to the kernel only when
   * there is no volume in hand to fit.
   */
  const runBrowserSliceAlignmentEstimate = async () => {
    if (alignmentEstimateBusyRef.current) return;
    if (!panelFloats || panelFloats.length === 0 || nz < 2) {
      if (!offline) requestSliceAlignmentEstimate();
      return;
    }
    alignmentEstimateBusyRef.current = true;
    setLocalAlignmentStatus("Estimating slice alignment...");
    try {
      const estimate = await estimateSliceAlignment(
        panelFloats, nx, ny, nz, gpuFFTRef.current, await getGPUDevice(),
      );
      autoAlignmentRef.current = {
        cached: true,
        rowShift: estimate.rowShiftPxPerSlice,
        colShift: estimate.colShiftPxPerSlice,
      };
      setLiveRowShift(estimate.rowShiftPxPerSlice);
      setLiveColShift(estimate.colShiftPxPerSlice);
      alignmentPreviewRef.current?.(estimate.rowShiftPxPerSlice, estimate.colShiftPxPerSlice);
      setRowShiftPxPerSlice(estimate.rowShiftPxPerSlice);
      setColShiftPxPerSlice(estimate.colShiftPxPerSlice);
      setSliceAlignmentCached(true);
      sliceAlignmentCachedRef.current = true;
      sliceAlignmentModeRef.current = "auto";
      // Name the backend: a silent CPU fallback is many times slower and would
      // otherwise look identical to a WebGPU fit in the toolbar.
      setLocalAlignmentStatus(alignedStatusText(
        estimate.rowShiftPxPerSlice,
        estimate.colShiftPxPerSlice,
        estimate.backend === "webgpu" ? "WebGPU" : "CPU fallback",
      ));
    } catch (err) {
      setLocalAlignmentStatus(`Alignment estimate failed: ${err instanceof Error ? err.message : String(err)}`);
      setSliceAlignment("off");
    } finally {
      alignmentEstimateBusyRef.current = false;
    }
  };
  // Paint a fitted or hand-set shift at once and sync it to the model, noting
  // which mode owns the Row/Col sliders.
  const applySliceAlignment = (rowShift: number, colShift: number, mode: "auto" | "manual") => {
    setLiveRowShift(rowShift);
    setLiveColShift(colShift);
    alignmentPreviewRef.current?.(rowShift, colShift);
    setRowShiftPxPerSlice(rowShift);
    setColShiftPxPerSlice(colShift);
    setSliceAlignmentCached(true);
    setSliceAlignment(mode);
    sliceAlignmentCachedRef.current = true;
    sliceAlignmentModeRef.current = mode;
  };
  const handleSliceAlignmentToggle = (on: boolean) => {
    if (!on) {
      setSliceAlignment("off");
      setLocalAlignmentStatus("");
      return;
    }
    const auto = autoAlignmentRef.current;
    if (auto.cached) {
      applySliceAlignment(auto.rowShift, auto.colShift, "auto");
      // Re-enabling reuses the fit instead of paying for it again, but say so:
      // a blank toolbar reads as "nothing happened".
      setLocalAlignmentStatus(alignedStatusText(auto.rowShift, auto.colShift, "cached"));
      return;
    }
    setSliceAlignment("auto");
    // Estimate in the browser. The volume is already here for slice rendering,
    // so WebGPU can fit the drift without a kernel - which is the only option in
    // an exported HTML, and saves a comm round-trip when a kernel IS attached.
    runBrowserSliceAlignmentEstimate();
  };
  const commitManualSliceAlignment = (rowShift: number, colShift: number) => {
    if (alignmentRafRef.current != null) {
      cancelAnimationFrame(alignmentRafRef.current);
      alignmentRafRef.current = null;
    }
    pendingAlignmentRef.current = null;
    applySliceAlignment(rowShift, colShift, "manual");
    setLocalAlignmentStatus("");
  };
  const stageManualSliceAlignment = React.useCallback((rowShift: number, colShift: number) => {
    pendingAlignmentRef.current = { rowShift, colShift };
    if (sliceAlignmentModeRef.current !== "manual") {
      sliceAlignmentModeRef.current = "manual";
      setSliceAlignment("manual");
    }
    if (!sliceAlignmentCachedRef.current) {
      sliceAlignmentCachedRef.current = true;
      setSliceAlignmentCached(true);
    }
    setLocalAlignmentStatus("");
    if (alignmentRafRef.current != null) return;
    alignmentRafRef.current = requestAnimationFrame(() => {
      alignmentRafRef.current = null;
      const pending = pendingAlignmentRef.current;
      pendingAlignmentRef.current = null;
      if (!pending) return;
      React.startTransition(() => {
        setLiveRowShift(pending.rowShift);
        setLiveColShift(pending.colShift);
      });
      alignmentPreviewRef.current?.(pending.rowShift, pending.colShift);
    });
  }, [setSliceAlignment, setSliceAlignmentCached]);
  const resetSliceAlignment = () => {
    if (alignmentRafRef.current != null) {
      cancelAnimationFrame(alignmentRafRef.current);
      alignmentRafRef.current = null;
    }
    pendingAlignmentRef.current = null;
    if (autoAlignmentRef.current.cached) {
      const auto = autoAlignmentRef.current;
      applySliceAlignment(auto.rowShift, auto.colShift, "auto");
      setLocalAlignmentStatus("");
      return;
    }
    setLiveRowShift(0);
    setLiveColShift(0);
    alignmentPreviewRef.current?.(0, 0);
    setRowShiftPxPerSlice(0);
    setColShiftPxPerSlice(0);
    setSliceAlignmentCached(false);
    setSliceAlignment("off");
    sliceAlignmentCachedRef.current = false;
    sliceAlignmentModeRef.current = "off";
    setLocalAlignmentStatus("");
    if (offline) return;
    setSliceAlignmentRequest(JSON.stringify({ mode: "reset", id: `${Date.now()}-${Math.random().toString(36).slice(2)}` }));
  };

  React.useEffect(() => {
    const pending = pendingExportRef.current;
    if (!pending || exportPayloadId !== pending.id) return;
    const bytes = extractBytes(exportPayload);
    if (bytes.length === 0) return;
    let canceled = false;
    const save = async () => {
      const payload = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? bytes
        : bytes.slice();
      const filename = exportPayloadFilename || pending.filename;
      const blob = new Blob([payload as BlobPart], { type: "text/html;charset=utf-8" });
      try {
        if (pending.handle) {
          setLocalExportStatus(`Saving ${filename}...`);
          const writable = await pending.handle.createWritable();
          await writable.write(blob);
          await writable.close();
        } else {
          downloadBlob(blob, filename);
        }
        if (canceled) return;
        pendingExportRef.current = null;
        setExportBusy(false);
        setLocalExportStatus(`Saved ${filename} (${formatSavedBytes(bytes.byteLength)})`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      } catch (err) {
        if (canceled) return;
        pendingExportRef.current = null;
        setExportBusy(false);
        setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      }
    };
    void save();
    return () => { canceled = true; };
  }, [exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest]);

  // Slice dimensions: [xy: ny x nx], [oblique: nz x diagonal]
  const sliceDims = React.useMemo<[number, number][]>(
    () => [[ny, nx], [nz, segmentWidth(obliqueSegment.start, obliqueSegment.stop)]],
    [ny, nx, nz, obliqueSegment],
  );
  type LiveFftSegment = { start: { x: number; y: number }; stop: { x: number; y: number } };
  const liveFftSchedulerRef = React.useRef<{
    pending: { axes: number[]; sliceZ: number; segment: LiveFftSegment } | null;
    inFlight: boolean;
    timer: number | null;
    lastStartMs: number;
  }>({ pending: null, inFlight: false, timer: null, lastStartMs: 0 });

  const makeFftResultCacheKey = (
    axis: number,
    sliceZValue: number,
    segment: LiveFftSegment,
    sliceW: number,
    sliceH: number,
  ) => {
    const geometry = axis === 0
      ? `z=${Math.round(sliceZValue)}`
      : `line=${segment.start.x.toFixed(2)},${segment.start.y.toFixed(2)}:${segment.stop.x.toFixed(2)},${segment.stop.y.toFixed(2)}`;
    return [
      `data=${fftDataTokenRef.current}`,
      `axis=${axis}`,
      geometry,
      `dims=${sliceW}x${sliceH}`,
      `window=${fftWindow ? 1 : 0}`,
      `log=${fftLogScale ? 1 : 0}`,
      `auto=${fftAuto ? 1 : 0}`,
    ].join("|");
  };

  const trimFftResultCache = () => {
    const cache = fftResultCacheRef.current;
    while (
      cache.size > SHOW3DSLICES_FFT_RESULT_CACHE_MAX_ENTRIES ||
      fftResultCacheBytesRef.current > SHOW3DSLICES_FFT_RESULT_CACHE_MAX_BYTES
    ) {
      let oldestKey = "";
      let oldestUsed = Number.POSITIVE_INFINITY;
      cache.forEach((entry, key) => {
        if (entry.lastUsed < oldestUsed) {
          oldestUsed = entry.lastUsed;
          oldestKey = key;
        }
      });
      if (!oldestKey) break;
      const entry = cache.get(oldestKey);
      if (entry) fftResultCacheBytesRef.current -= entry.bytes;
      cache.delete(oldestKey);
    }
  };

  const applyFftResultCacheEntry = (
    axis: number,
    entry: FftResultCacheEntry,
    magCache: React.MutableRefObject<(Float32Array | null)[]>,
    offscreenCache: React.MutableRefObject<(HTMLCanvasElement | null)[]>,
    imgDataCache: React.MutableRefObject<(ImageData | null)[]>,
  ) => {
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    magCache.current[axis] = entry.mag;
    // Reuse the offscreen when the size matches: saves a ~4 MB ImageData per axis.
    const existingOff = offscreenCache.current[axis];
    const existingImg = imgDataCache.current[axis];
    if (existingOff && existingImg && existingOff.width === entry.width && existingOff.height === entry.height) {
      renderToOffscreenReuse(entry.displayData, lut, entry.displayMin, entry.displayMax, existingOff, existingImg);
    } else {
      const offscreen = renderToOffscreen(entry.displayData, entry.width, entry.height, lut, entry.displayMin, entry.displayMax);
      if (!offscreen) return false;
      offscreenCache.current[axis] = offscreen;
      const ctx = offscreen.getContext("2d");
      imgDataCache.current[axis] = ctx ? ctx.getImageData(0, 0, entry.width, entry.height) : null;
    }
    return true;
  };

  const putFftResultCacheEntry = (
    key: string,
    mag: Float32Array,
    displayData: Float32Array,
    displayMin: number,
    displayMax: number,
    width: number,
    height: number,
  ): FftResultCacheEntry => {
    const existing = fftResultCacheRef.current.get(key);
    if (existing) fftResultCacheBytesRef.current -= existing.bytes;
    const bytes = mag.byteLength + (displayData === mag ? 0 : displayData.byteLength);
    const entry = {
      mag,
      displayData,
      displayMin,
      displayMax,
      width,
      height,
      bytes,
      lastUsed: ++fftResultCacheSeqRef.current,
    };
    fftResultCacheRef.current.set(key, entry);
    fftResultCacheBytesRef.current += bytes;
    trimFftResultCache();
    return entry;
  };

  // Centre the transformed slice, take its magnitude and display window (Auto,
  // Log), and cache it so revisiting the slice skips the transform.
  const storeFftResult = (
    key: string,
    real: Float32Array,
    imag: Float32Array,
    paddedW: number,
    paddedH: number,
  ): FftResultCacheEntry => {
    fftshift(real, paddedW, paddedH);
    fftshift(imag, paddedW, paddedH);
    const mag = computeMagnitude(real, imag);
    let displayMin: number;
    let displayMax: number;
    if (fftAuto) {
      ({ min: displayMin, max: displayMax } = autoEnhanceFFT(mag, paddedW, paddedH));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(mag));
    }
    const displayData = fftLogScale ? applyLogScale(mag) : mag;
    if (fftLogScale) {
      displayMin = Math.log1p(displayMin);
      displayMax = Math.log1p(displayMax);
    }
    return putFftResultCacheEntry(key, mag, displayData, displayMin, displayMax, paddedW, paddedH);
  };

  const computeLiveFftAxis = async (
    axis: number,
    sliceZValue: number,
    segment: LiveFftSegment,
  ): Promise<boolean> => {
    if (!showFft || !panelFloats || panelFloats.length === 0) return false;
    const [sliceH, sliceW] = axis === 0 ? [ny, nx] : [nz, segmentWidth(segment.start, segment.stop)];
    const cacheKey = makeFftResultCacheKey(axis, sliceZValue, segment, sliceW, sliceH);
    const cached = fftResultCacheRef.current.get(cacheKey);
    if (cached) {
      cached.lastUsed = ++fftResultCacheSeqRef.current;
      return applyFftResultCacheEntry(
        axis,
        cached,
        fftMagCacheRefs,
        fftOffscreenRefs,
        fftImgDataRefs,
      );
    }
    const extracted = axis === 0
      ? extractXY(panelFloats, nx, ny, nz, sliceZValue)
      : extractOblique(panelFloats, nx, ny, nz, segment.start, segment.stop);
    const data = fftWindow ? new Float32Array(extracted) : extracted;
    if (fftWindow) applyHannWindow2D(data, sliceW, sliceH);
    const paddedW = nextPow2(sliceW);
    const paddedH = nextPow2(sliceH);
    let real: Float32Array;
    let imag: Float32Array;
    if (gpuReady && gpuFFTRef.current) {
      const result = await gpuFFTRef.current.fft2D(zeroPadSlice(data, sliceW, sliceH, paddedW, paddedH), new Float32Array(paddedW * paddedH), paddedW, paddedH, false);
      real = result.real;
      imag = result.imag;
    } else {
      real = zeroPadSlice(data, sliceW, sliceH, paddedW, paddedH);
      imag = new Float32Array(paddedW * paddedH);
      fft2d(real, imag, paddedW, paddedH, false);
    }
    const entry = storeFftResult(cacheKey, real, imag, paddedW, paddedH);
    return applyFftResultCacheEntry(axis, entry, fftMagCacheRefs, fftOffscreenRefs, fftImgDataRefs);
  };

  const runLiveFftScheduler = () => {
    const state = liveFftSchedulerRef.current;
    state.timer = null;
    if (state.inFlight || !state.pending) return;
    const now = performance.now();
    const minIntervalMs = 16;
    const waitMs = Math.max(0, minIntervalMs - (now - state.lastStartMs));
    if (waitMs > 0) {
      state.timer = window.setTimeout(runLiveFftScheduler, waitMs);
      return;
    }
    const pending = state.pending;
    state.pending = null;
    state.inFlight = true;
    state.lastStartMs = now;
    Promise.all(pending.axes.map(axis => computeLiveFftAxis(axis, pending.sliceZ, pending.segment)))
      .then(results => {
        if (results.some(Boolean)) setFftVersion((version) => version + 1);
      })
      .finally(() => {
        state.inFlight = false;
        if (state.pending && state.timer == null) runLiveFftScheduler();
      });
  };

  const scheduleLiveFft = (
    axes: number[],
    options: { sliceZ?: number; segment?: LiveFftSegment } = {},
  ) => {
    if (!showFft || !panelFloats || panelFloats.length === 0) return;
    const state = liveFftSchedulerRef.current;
    const prevAxes = state.pending?.axes ?? [];
    const mergedAxes = Array.from(new Set([...prevAxes, ...axes])).filter(axis => axis === 0 || axis === 1);
    if (mergedAxes.length === 0) return;
    state.pending = {
      axes: mergedAxes,
      sliceZ: options.sliceZ ?? sliceZ,
      segment: options.segment ?? { start: { ...obliqueSegment.start }, stop: { ...obliqueSegment.stop } },
    };
    if (!state.inFlight && state.timer == null) runLiveFftScheduler();
  };

  React.useEffect(() => () => {
    const state = liveFftSchedulerRef.current;
    if (state.timer != null) window.clearTimeout(state.timer);
    state.timer = null;
    state.pending = null;
  }, []);
  React.useEffect(() => {
    sliceOffscreenRefs.current = [null, null, null];
    sliceImgDataRefs.current = [null, null, null];
    fftOffscreenRefs.current = [null, null, null];
    fftImgDataRefs.current = [null, null, null];
    fftMagCacheRefs.current = [null, null, null];
    liveFftSchedulerRef.current.pending = null;
    volUploadedKeyRef.current = null;
    gpuVolReadyRef.current = false;
    setFftVersion((version) => version + 1);
  }, [safeActivePanel, volumeBytes]);

  // Canvas sizes. For depth panels, keep the Z scale independent from the
  // oblique profile length; otherwise shortening the profile would secretly
  // magnify Z before the explicit z_stretch slider is applied.
  // smooth=true → CSS bilinear (auto); smooth=false → nearest-neighbor (pixelated).
  // Overlay canvases (crosshair, scale bar, colorbar, FFT scale bar) use displayH
  // for their pixel buffer to avoid distortion under CSS stretch.
  const canvasSizes = React.useMemo(() => sliceDims.map((dims, axis) => {
    const isDepth = axis > 0;
    const target = isDepth ? sideCanvasTarget : canvasTarget;
    const raster = panelRasterSize(dims, isDepth, target);
    const displayH = isDepth ? Math.min(target, Math.round(raster.h * Math.max(1, zStretch))) : raster.h;
    return { ...raster, displayH };
  }), [sliceDims, sideCanvasTarget, canvasTarget, zStretch]);
  const dataPointToCanvas = React.useCallback((axis: number, x: number, y: number): { x: number; y: number } => {
    const { w: canvasW, h: canvasH, displayH, scaleX, scaleY } = canvasSizes[axis];
    const stretchY = displayH / canvasH;
    const view = liveZoomsRef.current[axis];
    const centerX = canvasW / 2, centerY = displayH / 2;
    let canvasX = x * scaleX;
    let canvasY = y * scaleY * stretchY;
    if (view.zoom !== 1 || view.panX !== 0 || view.panY !== 0) {
      canvasX = (canvasX - centerX) * view.zoom + centerX + view.panX;
      canvasY = (canvasY - centerY) * view.zoom + centerY + view.panY * stretchY;
    }
    return { x: canvasX, y: canvasY };
  }, [canvasSizes]);
  const rasterCanvasSizes = React.useMemo(
    () => sliceDims.map((dims, axis) => panelRasterSize(dims, axis > 0, axis > 0 ? sideCanvasTarget : canvasTarget)),
    [sliceDims, sideCanvasTarget, canvasTarget],
  );

  // Pre-allocate reusable offscreen canvases + ImageData per axis (avoids GC churn)
  React.useEffect(() => {
    for (let axis = 0; axis < sliceDims.length; axis++) {
      const [sliceH, sliceW] = sliceDims[axis];
      const existing = sliceOffscreenRefs.current[axis];
      if (!existing || existing.width !== sliceW || existing.height !== sliceH) {
        const offscreen = document.createElement("canvas");
        offscreen.width = sliceW; offscreen.height = sliceH;
        sliceOffscreenRefs.current[axis] = offscreen;
        sliceImgDataRefs.current[axis] = new ImageData(sliceW, sliceH);
      }
    }
  }, [sliceDims]);

  // Prevent page scroll on canvases
  React.useEffect(() => {
    const preventDefault = (event: WheelEvent) => event.preventDefault();
    canvasRefs.current.forEach(c => c?.addEventListener("wheel", preventDefault, { passive: false }));
    fftCanvasRefs.current.forEach(c => c?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => {
      canvasRefs.current.forEach(c => c?.removeEventListener("wheel", preventDefault));
      fftCanvasRefs.current.forEach(c => c?.removeEventListener("wheel", preventDefault));
    };
  }, [panelFloats, showFft]);

  // Keep the exact full volume resident on the GPU. Hot display toggles (flip,
  // log, auto) must not allocate or upload a transformed 45M-voxel volume.
  const volumeFloats = panelFloats;
  const histogramSample = React.useMemo(() => makeHistogramSample(panelFloats), [panelFloats]);
  const displayHistogramSample = React.useMemo(
    () => transformDisplaySample(histogramSample, logScale, false),
    [histogramSample, logScale],
  );

  // Compute UI histogram and auto-contrast from a deterministic sample. The
  // rendered slice pixels still come from the exact full-resolution GPU volume.
  React.useEffect(() => {
    if (!displayHistogramSample || displayHistogramSample.length === 0) return;
    setImageHistogramData(displayHistogramSample);
  }, [displayHistogramSample]);

  const displayDataRange = React.useMemo(() => {
    return resolveDisplayBounds(
      imageDataRange.min,
      imageDataRange.max,
      traitVmin,
      traitVmax,
      logScale,
    );
  }, [imageDataRange, traitVmin, traitVmax, logScale]);
  // Explicit vmin/vmax are the window as given: Auto never narrows them to
  // percentiles, so it is only active while both limits are unset.
  const hasExplicitLimits = traitVmin != null || traitVmax != null;
  const autoActive = autoContrast && !hasExplicitLimits;

  const handleAutoContrastChange = (on: boolean) => {
    if (on) {
      manualImageRangeBeforeAutoRef.current = { min: imageVminPct, max: imageVmaxPct };
    }
    setAutoContrast(on);
    if (on && imageHistogramData) {
      const { vmin: pmin, vmax: pmax } = percentileClip(imageHistogramData, 2, 98);
      const span = displayDataRange.max - displayDataRange.min;
      if (span > 0) {
        setImageVminPct(Math.max(0, Math.min(100, ((pmin - displayDataRange.min) / span) * 100)));
        setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - displayDataRange.min) / span) * 100)));
      }
    } else {
      const restore = manualImageRangeBeforeAutoRef.current;
      if (restore) {
        setImageVminPct(restore.min);
        setImageVmaxPct(restore.max);
        manualImageRangeBeforeAutoRef.current = null;
      } else {
        setImageVminPct(0);
        setImageVmaxPct(100);
      }
    }
  };

  // Auto owns the sliders while it is on (a histogram drag turns it off): snap
  // them to the 2/98 percentiles of the displayed values whenever those change,
  // e.g. after a log toggle, so every slice keeps the volume's own window.
  // Percentiles restored from Python at mount (not 0/100) are kept until then.
  const autoSnapInputsRef = React.useRef<{ sample: Float32Array; range: { min: number; max: number } } | null>(null);
  React.useEffect(() => {
    if (!autoActive || !displayHistogramSample) {
      autoSnapInputsRef.current = null;
      return;
    }
    const last = autoSnapInputsRef.current;
    autoSnapInputsRef.current = { sample: displayHistogramSample, range: displayDataRange };
    if (!last && (imageVminPct !== 0 || imageVmaxPct !== 100)) return;
    if (last && last.sample === displayHistogramSample && last.range === displayDataRange) return;
    const { vmin: pmin, vmax: pmax } = percentileClip(displayHistogramSample, 2, 98);
    const span = displayDataRange.max - displayDataRange.min;
    if (span > 0) {
      setImageVminPct(Math.max(0, Math.min(100, ((pmin - displayDataRange.min) / span) * 100)));
      setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - displayDataRange.min) / span) * 100)));
    }
  }, [autoActive, displayHistogramSample, displayDataRange]);

  React.useEffect(() => {
    bounceDirRef.current = reverse ? -1 : 1;
  }, [reverse]);

  // -------------------------------------------------------------------------
  // 3D Volume Renderer - init, upload, render
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = volumeCanvasRef.current;
    if (!canvas) return;
    if (!VolumeRenderer.isSupported()) { setVolumeInitError("navigator.gpu missing"); setWebgpuSupported(false); return; }
    let disposed = false;
    VolumeRenderer.create(canvas).then(renderer => {
      if (disposed) { renderer.dispose(); return; }
      volumeRendererRef.current = renderer;
      setRendererReady((count) => count + 1);
    }).catch((err) => {
      // Surface the REAL reason - a swallowed error here used to show a generic
      // "WebGPU not available" even when the adapter was fine but the volume
      // pipeline/3D-texture init failed, making the bug undebuggable.
      const message = String(err?.message || err);
      const unavailable = /WebGPU not available|requestAdapter|navigator\.gpu/i.test(message);
      const logger = unavailable ? console.warn : console.error;
      logger("[Show3DSlices] 3D volume renderer init failed:", err);
      setVolumeInitError(message);
      setWebgpuSupported(false);
    });
    return () => { disposed = true; volumeRendererRef.current?.dispose(); volumeRendererRef.current = null; };
  }, []);

  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer || !volumeFloats || volumeFloats.length === 0) return;
    renderer.uploadVolume(volumeFloats, nx, ny, nz);
  }, [volumeFloats, nx, ny, nz, rendererReady]);

  // Upload colormap. When flip, reverse the LUT entry order so the 3D volume
  // inverts contrast the same way slice panels do (slices negate the data and
  // swap vmin/vmax, equivalent to reversing the colormap lookup). LUT is
  // 256 RGB triplets (768 bytes); reverse per-entry, not per-byte.
  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    renderer.uploadColormap(flip ? reverseLut(lut) : lut);
  }, [cmap, rendererReady, flip]);

  // Map slider %s + optional traitVmin/Vmax to the texture's [0,1] normalized space.
  // The 3D context texture is uploaded from raw data only once; log/flip are
  // hot display toggles handled by the exact slice shader and LUT reversal, not
  // by re-uploading a transformed volume.
  const volumeTextureRangeForPercent = (minPct: number, maxPct: number) => {
    const span = imageDataRange.max - imageDataRange.min;
    if (span <= 0) return { vmin: 0, vmax: 1 };
    const subMinData = imageDataRange.min + span * (minPct / 100);
    const subMaxData = imageDataRange.min + span * (maxPct / 100);
    const subMin = (subMinData - imageDataRange.min) / span;
    const subMax = (subMaxData - imageDataRange.min) / span;
    return { vmin: subMin, vmax: subMax };
  };
  const volTexRange = volumeTextureRangeForPercent(imageVminPct, imageVmaxPct);
  // Keep live slice positions separate from committed model traits. Slider drag
  // updates these refs every frame; model traits sync only on release.
  const liveSliceParamsRef = React.useRef({ sliceX, sliceY, sliceZ });
  const committedSliceParamsRef = React.useRef({ sliceX, sliceY, sliceZ });
  const committedSliceParams = committedSliceParamsRef.current;
  if (
    committedSliceParams.sliceX !== sliceX ||
    committedSliceParams.sliceY !== sliceY ||
    committedSliceParams.sliceZ !== sliceZ
  ) {
    const next = { sliceX, sliceY, sliceZ };
    committedSliceParamsRef.current = next;
    liveSliceParamsRef.current = next;
  }

  // Keep render params in ref for direct rAF rendering (bypasses React during drag)
  const volumeRenderParams = {
    ...liveSliceParamsRef.current, nx, ny, nz,
    opacity: opacityA, brightness: 1.0, slicePlaneMask, slicePlaneOpacity,
    obliqueAngleDeg: obliqueAngle,
    obliqueStartX: obliqueSegment.start.x,
    obliqueStartY: obliqueSegment.start.y,
    obliqueEndX: obliqueSegment.stop.x,
    obliqueEndY: obliqueSegment.stop.y,
    rowShiftPxPerSlice: gpuSliceParams.rowShift,
    colShiftPxPerSlice: gpuSliceParams.colShift,
    vmin: volTexRange.vmin, vmax: volTexRange.vmax,
  };
  const volumeRenderParamsRef = React.useRef(volumeRenderParams);
  volumeRenderParamsRef.current = volumeRenderParams;
  const bgColorRef = React.useRef<[number, number, number]>([0, 0, 0]);
  React.useEffect(() => {
    const r = parseInt(themeColors.bg.slice(1, 3), 16) / 255;
    const g = parseInt(themeColors.bg.slice(3, 5), 16) / 255;
    const b = parseInt(themeColors.bg.slice(5, 7), 16) / 255;
    bgColorRef.current = [r, g, b];
  }, [themeColors.bg]);

  // Render 3D volume (non-interactive: triggered by React state changes)
  React.useEffect(() => {
    if (volumeDrag) return; // Skip during drag - rAF handles it directly
    const renderer = volumeRendererRef.current;
    if (!renderer || !volumeFloats || volumeFloats.length === 0) return;
    renderer.render(volumeRenderParamsRef.current, camera, bgColorRef.current, undefined, undefined, zStretch, orthographic);
  }, [volumeFloats, sliceX, sliceY, sliceZ, obliqueAngle, obliqueSegment, nx, ny, nz, cmap, camera, volumeCanvasSize, themeColors.bg, slicePlaneMask, slicePlaneOpacity, volumeDrag, rendererReady, volTexRange, opacityA, zStretch, orthographic, flip, gpuSliceParams]);

  // First-frame paint guard: the very first synchronous render after the renderer
  // mounts can land before the canvas swapchain is ready (flush race) and commit a
  // BLACK frame - the volume then stayed blank until the user dragged. Re-render on
  // the next animation frame (once the context is configured + data uploaded) so the
  // volume is visible on mount, no interaction needed.
  React.useEffect(() => {
    if (!rendererReady) return;
    const id = requestAnimationFrame(() => {
      const renderer = volumeRendererRef.current;
      if (renderer && volumeFloats && volumeFloats.length > 0) {
        renderer.render(volumeRenderParamsRef.current, camera, bgColorRef.current, undefined, undefined, zStretch, orthographic);
      }
    });
    return () => cancelAnimationFrame(id);
  }, [rendererReady, volumeFloats]);

  // Prevent scroll on volume canvas
  React.useEffect(() => {
    const canvas = volumeCanvasRef.current;
    if (!canvas || !webgpuSupported) return;
    const preventDefault = (event: WheelEvent) => event.preventDefault();
    canvas.addEventListener("wheel", preventDefault, { passive: false });
    return () => canvas.removeEventListener("wheel", preventDefault);
  }, [webgpuSupported]);

  // -------------------------------------------------------------------------
  // 3D Volume mouse handlers - document-level listeners for robust drag
  // -------------------------------------------------------------------------
  const volumeRafRef = React.useRef<number>(0);
  const liveCameraRef = React.useRef<CameraState>(camera);
  const persistViewState = React.useCallback((
    nextZooms: ZoomState[] = liveZoomsRef.current,
    nextFftZooms: ZoomState[] = liveFftZoomsRef.current,
    nextCamera: CameraState = liveCameraRef.current,
    nextSizes: Partial<Show3DSlicesViewSizes> = {},
  ) => {
    setViewState({
      zooms: nextZooms.map((view) => ({ ...view })),
      fft_zooms: nextFftZooms.map((view) => ({ ...view })),
      camera: { ...nextCamera },
      canvas_target: clampCanvasTarget(nextSizes.canvasTarget ?? canvasTargetRef.current),
      side_canvas_target: clampCanvasTarget(nextSizes.sideCanvasTarget ?? sideCanvasTargetRef.current),
      volume_canvas_size: clampCanvasTarget(nextSizes.volumeCanvasSize ?? volumeCanvasSizeRef.current),
    });
  }, [setViewState]);
  // Live z_stretch ref for rAF drag path - keeps latest value without re-binding closure.
  const zStretchRef = React.useRef(zStretch);
  if (!zStretchLiveDirtyRef.current) zStretchRef.current = zStretch;
  const applyDepthPanelHeight = (value: number) => {
    for (let axis = 1; axis < sliceDims.length; axis++) {
      const base = rasterCanvasSizes[axis];
      if (!base) continue;
      const displayH = Math.min(sideCanvasTarget, Math.round(base.h * Math.max(1, value)));
      const height = `${displayH}px`;
      const box = imageBoxRefs.current[axis];
      const canvas = canvasRefs.current[axis];
      const overlay = overlayRefs.current[axis];
      const ui = uiRefs.current[axis];
      if (box) box.style.height = height;
      if (canvas) canvas.style.height = height;
      if (overlay) overlay.style.height = height;
      if (ui) ui.style.height = height;
    }
  };
  React.useEffect(() => { applyDepthPanelHeight(zStretch); }, [zStretch, rasterCanvasSizes, sideCanvasTarget]);
  const handleZStretchChange = (value: number) => {
    zStretchLiveDirtyRef.current = true;
    pendingZStretchRef.current = value;
    zStretchRef.current = value;
    if (zStretchRafRef.current != null) return;
    zStretchRafRef.current = requestAnimationFrame(() => {
      zStretchRafRef.current = null;
      const next = pendingZStretchRef.current;
      applyDepthPanelHeight(next);
      const renderer = volumeRendererRef.current;
      if (renderer && volumeFloats && volumeFloats.length > 0) {
        renderer.render(volumeRenderParamsRef.current, liveCameraRef.current, bgColorRef.current, undefined, undefined, next, orthographic);
      }
    });
  };
  const handleZStretchCommit = (value: number) => {
    if (zStretchRafRef.current != null) {
      cancelAnimationFrame(zStretchRafRef.current);
      zStretchRafRef.current = null;
    }
    pendingZStretchRef.current = value;
    zStretchRef.current = value;
    applyDepthPanelHeight(value);
    zStretchLiveDirtyRef.current = false;
    setZStretch(value);
    setModelZStretch(value);
  };
  const handleVolumeControlChange = (key: "opacity" | "slicePlaneOpacity", value: number) => {
    pendingVolumeControlsRef.current = { ...pendingVolumeControlsRef.current, [key]: value };
    if (volumeControlsRafRef.current != null) return;
    volumeControlsRafRef.current = requestAnimationFrame(() => {
      volumeControlsRafRef.current = null;
      const next = pendingVolumeControlsRef.current;
      volumeRenderParamsRef.current = {
        ...volumeRenderParamsRef.current,
        opacity: next.opacity,
        slicePlaneOpacity: next.slicePlaneOpacity,
      };
      const renderer = volumeRendererRef.current;
      if (renderer && volumeFloats && volumeFloats.length > 0) {
        renderer.render(
          volumeRenderParamsRef.current,
          liveCameraRef.current,
          bgColorRef.current,
          undefined,
          undefined,
          zStretchRef.current,
          orthographic,
        );
      }
    });
  };
  const handleVolumeControlCommit = (key: "opacity" | "slicePlaneOpacity", value: number) => {
    pendingVolumeControlsRef.current = { ...pendingVolumeControlsRef.current, [key]: value };
    const next = pendingVolumeControlsRef.current;
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      opacity: next.opacity,
      slicePlaneOpacity: next.slicePlaneOpacity,
    };
    setOpacityA(next.opacity);
    setSlicePlaneOpacity(next.slicePlaneOpacity);
  };
  const handlePlaneVisibilityChange = (_event: React.MouseEvent<HTMLElement>, nextPlanes: string[]) => {
    const nextVisibility = PLANE_KEYS.map((key) => nextPlanes.includes(key));
    const nextMask = nextVisibility.reduce((mask, visible, i) => (
      visible ? mask | (1 << i) : mask
    ), 0);
    setPlaneVisibility(nextVisibility);
    setShowSlicePlanes(nextMask !== 0);
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      slicePlaneMask: nextMask,
    };
    const renderer = volumeRendererRef.current;
    if (renderer && volumeFloats && volumeFloats.length > 0) {
      renderer.render(
        volumeRenderParamsRef.current,
        liveCameraRef.current,
        bgColorRef.current,
        undefined,
        undefined,
        zStretchRef.current,
        orthographic,
      );
    }
  };

  const handleObliqueAngleChange = (_event: Event, value: number | number[]) => {
    const nextAngle = Array.isArray(value) ? value[0] : value;
    const center = {
      x: (obliqueSegment.start.x + obliqueSegment.stop.x) / 2,
      y: (obliqueSegment.start.y + obliqueSegment.stop.y) / 2,
    };
    const length = Math.max(1, Math.hypot(
      obliqueSegment.stop.x - obliqueSegment.start.x,
      obliqueSegment.stop.y - obliqueSegment.start.y,
    ));
    const theta = (nextAngle * Math.PI) / 180;
    const halfDx = Math.cos(theta) * length / 2;
    const halfDy = Math.sin(theta) * length / 2;
    const start = clampPointToImage({ x: center.x - halfDx, y: center.y - halfDy }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    const stop = clampPointToImage({ x: center.x + halfDx, y: center.y + halfDy }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    setObliqueAngle(nextAngle);
    setObliqueProfileLine(profileLinePayload(start, stop));
    setObliquePositionBounds(null);
    // Pass the freshly computed segment: the `obliqueSegment` default is a
    // trait-derived memo that still holds the pre-drag geometry on this tick.
    updateObliqueCenter((start.x + stop.x) / 2, (start.y + stop.y) / 2, nextAngle, { start, stop });
    scheduleLiveFft([1], { segment: { start, stop } });
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      obliqueAngleDeg: nextAngle,
      obliqueStartX: start.x,
      obliqueStartY: start.y,
      obliqueEndX: stop.x,
      obliqueEndY: stop.y,
    };
    const renderer = volumeRendererRef.current;
    if (renderer && volumeFloats && volumeFloats.length > 0) {
      renderer.render(
        volumeRenderParamsRef.current,
        liveCameraRef.current,
        bgColorRef.current,
        undefined,
        undefined,
        zStretchRef.current,
        orthographic,
      );
    }
  };
  if (!volumeDrag) liveCameraRef.current = camera;
  const volumeDragDataRef = React.useRef<{ button: number; x: number; y: number; yaw: number; pitch: number; panX: number; panY: number } | null>(null);

  const handleVolumeMouseDown = (event: React.MouseEvent) => {
    const dragData = {
      button: event.button, x: event.clientX, y: event.clientY,
      yaw: camera.yaw, pitch: camera.pitch, panX: camera.panX, panY: camera.panY,
    };
    volumeDragDataRef.current = dragData;
    setVolumeDrag(dragData);
    event.preventDefault();
  };

  React.useEffect(() => {
    if (!volumeDrag) return;
    const onMove = (event: MouseEvent) => {
      const drag = volumeDragDataRef.current;
      if (!drag) return;
      const deltaX = event.clientX - drag.x;
      const deltaY = event.clientY - drag.y;
      let next: CameraState;
      if (drag.button === 0 && !event.shiftKey) {
        next = {
          ...liveCameraRef.current,
          yaw: drag.yaw + deltaX * 0.005,
          pitch: Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, drag.pitch - deltaY * 0.005)),
        };
      } else {
        const panPerPixel = 0.003 * liveCameraRef.current.distance;
        next = {
          ...liveCameraRef.current,
          panX: drag.panX + deltaX * panPerPixel,
          panY: drag.panY - deltaY * panPerPixel,
        };
      }
      liveCameraRef.current = next;
      if (!volumeRafRef.current) {
        volumeRafRef.current = requestAnimationFrame(() => {
          volumeRafRef.current = 0;
          const liveCamera = liveCameraRef.current;
          const params = volumeRenderParamsRef.current;
          const background = bgColorRef.current;
          const renderer = volumeRendererRef.current;
          if (renderer) {
            renderer.render(params, liveCamera, background, undefined, undefined, zStretchRef.current, orthographic);
          }
        });
      }
    };
    const onUp = () => {
      const nextCamera = liveCameraRef.current;
      setCamera(nextCamera);
      persistViewState(undefined, undefined, nextCamera);
      setVolumeDrag(null);
      volumeDragDataRef.current = null;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => { document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
  }, [volumeDrag, orthographic]);

  const handleVolumeWheel = (event: React.WheelEvent) => {
    const factor = event.deltaY > 0 ? 1.1 : 0.9;
    const next = { ...liveCameraRef.current, distance: Math.max(0.5, Math.min(10, liveCameraRef.current.distance * factor)) };
    liveCameraRef.current = next;
    const renderer = volumeRendererRef.current;
    if (renderer) {
      renderer.render(volumeRenderParamsRef.current, next, bgColorRef.current, undefined, undefined, zStretchRef.current, orthographic);
    }
    setCamera(next);
    persistViewState(undefined, undefined, next);
  };

  const handleVolumeDoubleClick = () => {
    liveCameraRef.current = SHOW3DSLICES_DEFAULT_CAMERA;
    setCamera(SHOW3DSLICES_DEFAULT_CAMERA);
    persistViewState(undefined, undefined, SHOW3DSLICES_DEFAULT_CAMERA);
  };

  const setVolumeView = (view: "xy" | "side") => {
    const distance = liveCameraRef.current.distance || camera.distance || SHOW3DSLICES_DEFAULT_CAMERA.distance;
    // Match the 2D slice panels rather than mathematical world-up:
    // Top: x right, row/y down. Side: x right, z down.
    const presets: Record<"xy" | "side", Pick<CameraState, "yaw" | "pitch" | "roll">> = {
      xy: { yaw: Math.PI, pitch: 0, roll: Math.PI },
      side: { yaw: 0, pitch: Math.PI * 0.49, roll: 0 },
    };
    const next = { ...SHOW3DSLICES_DEFAULT_CAMERA, ...presets[view], distance, panX: 0, panY: 0 };
    liveCameraRef.current = next;
    setCamera(next);
    persistViewState(undefined, undefined, next);
  };

  const rollVolumeView = (direction: -1 | 1) => {
    const current = liveCameraRef.current;
    const next = { ...current, roll: (current.roll ?? 0) + direction * Math.PI / 2 };
    liveCameraRef.current = next;
    setCamera(next);
    persistViewState(undefined, undefined, next);
  };

  // -------------------------------------------------------------------------
  // 3D Volume canvas resize
  // -------------------------------------------------------------------------
  const volumeResizeRafRef = React.useRef(0);

  const handleVolumeResizeStart = (event: React.MouseEvent) => {
    event.stopPropagation(); event.preventDefault();
    setVolumeResizing(true);
    volumeResizeStartRef.current = { x: event.clientX, y: event.clientY, size: volumeCanvasSize };
  };

  React.useEffect(() => {
    if (!volumeResizing) return;
    let latestSize = volumeCanvasSize;
    const onMove = (event: MouseEvent) => {
      const start = volumeResizeStartRef.current;
      if (!start) return;
      const delta = Math.max(event.clientX - start.x, event.clientY - start.y);
      const newSize = clampCanvasTarget(start.size + delta);
      latestSize = newSize;
      // Throttle canvas resize to rAF for smooth drag
      if (!volumeResizeRafRef.current) {
        volumeResizeRafRef.current = requestAnimationFrame(() => {
          volumeResizeRafRef.current = 0;
          setVolumeCanvasSize(latestSize);
        });
      }
    };
    const onUp = () => {
      if (volumeResizeRafRef.current) { cancelAnimationFrame(volumeResizeRafRef.current); volumeResizeRafRef.current = 0; }
      const start = volumeResizeStartRef.current;
      if (start) {
        const nextSize = clampCanvasTarget(latestSize);
        setVolumeCanvasSize(nextSize);
        persistViewState(undefined, undefined, undefined, { volumeCanvasSize: nextSize });
      }
      setVolumeResizing(false);
      volumeResizeStartRef.current = null;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => { document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
  }, [persistViewState, volumeCanvasSize, volumeResizing]);

  const cameraChanged = camera.yaw !== SHOW3DSLICES_DEFAULT_CAMERA.yaw || camera.pitch !== SHOW3DSLICES_DEFAULT_CAMERA.pitch || (camera.roll ?? 0) !== (SHOW3DSLICES_DEFAULT_CAMERA.roll ?? 0) || camera.distance !== SHOW3DSLICES_DEFAULT_CAMERA.distance || camera.panX !== SHOW3DSLICES_DEFAULT_CAMERA.panX || camera.panY !== SHOW3DSLICES_DEFAULT_CAMERA.panY;

  // Reset Zoom is intentionally narrow: slice and FFT zoom/pan only. Camera,
  // contrast, colormap, and playback loop state have their own controls/state.
  const anyZoomDirty = zooms.some((view) => view.zoom !== 1 || view.panX !== 0 || view.panY !== 0)
    || fftZooms.some((view) => view.zoom !== DEFAULT_FFT_ZOOM.zoom || view.panX !== DEFAULT_FFT_ZOOM.panX || view.panY !== DEFAULT_FFT_ZOOM.panY);

  // -------------------------------------------------------------------------
  // Build colormapped offscreen canvases (expensive: log scale, percentile, colormap LUT)
  // Per-panel: XY depends on sliceZ; oblique depends on the XY center and angle.
  // Excludes zoom/pan so dragging only triggers the cheap redraw below.
  // useLayoutEffect so offscreens are ready before the draw useLayoutEffect runs.
  // -------------------------------------------------------------------------
  const prevCacheRef = React.useRef<{
    sliceX: number; sliceY: number; sliceZ: number;
    cmap: string; logScale: boolean;
    imageVminPct: number; imageVmaxPct: number;
    imageRangeMin: number; imageRangeMax: number;
    panelFloats: Float32Array | null;
    nx: number; ny: number; nz: number;
    traitVmin: number | null; traitVmax: number | null;
    flip: boolean;
    alignRowShift: number; alignColShift: number;
    alignStartX: number; alignStartY: number; alignStopX: number; alignStopY: number;
  }>({ sliceX: -1, sliceY: -1, sliceZ: -1, cmap: "", logScale: false, imageVminPct: -1, imageVmaxPct: -1, imageRangeMin: Number.NaN, imageRangeMax: Number.NaN, panelFloats: null, nx: 0, ny: 0, nz: 0, traitVmin: null, traitVmax: null, flip: false, alignRowShift: Number.NaN, alignColShift: Number.NaN, alignStartX: Number.NaN, alignStartY: Number.NaN, alignStopX: Number.NaN, alignStopY: Number.NaN });

  React.useLayoutEffect(() => {
    if (!panelFloats || panelFloats.length === 0) return;

    const prev = prevCacheRef.current;
    const alignRowShift = gpuSliceParams.rowShift;
    const alignColShift = gpuSliceParams.colShift;
    const alignStartX = gpuSliceParams.segment.start.x;
    const alignStartY = gpuSliceParams.segment.start.y;
    const alignStopX = gpuSliceParams.segment.stop.x;
    const alignStopY = gpuSliceParams.segment.stop.y;
    const globalChanged = panelFloats !== prev.panelFloats || cmap !== prev.cmap ||
      logScale !== prev.logScale ||
      imageVminPct !== prev.imageVminPct || imageVmaxPct !== prev.imageVmaxPct ||
      displayDataRange.min !== prev.imageRangeMin || displayDataRange.max !== prev.imageRangeMax ||
      traitVmin !== prev.traitVmin || traitVmax !== prev.traitVmax ||
      flip !== prev.flip ||
      alignRowShift !== prev.alignRowShift || alignColShift !== prev.alignColShift ||
      alignStartX !== prev.alignStartX || alignStartY !== prev.alignStartY ||
      alignStopX !== prev.alignStopX || alignStopY !== prev.alignStopY ||
      nx !== prev.nx || ny !== prev.ny || nz !== prev.nz;
    const axisChanged = [
      globalChanged || sliceZ !== prev.sliceZ,
      true,
    ];

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const extractors = [
      () => extractXY(panelFloats, nx, ny, nz, sliceZ),
      () => extractOblique(panelFloats, nx, ny, nz, obliqueSegment.start, obliqueSegment.stop),
    ];
    // GPU path: upload the whole volume ONCE; each scrub only slices + colormaps on
    // the GPU (no CPU extract / re-upload), so scrubbing stays buffer-smooth even on
    // a 1688x1688x16 volume. CPU path is the fallback (no engine / volume too big).
    const engine = gpuCmapRef.current;
    let gpuVolReady = false;
    if (cmapReady && engine && panelFloats) {
      if (volUploadedKeyRef.current !== panelFloats) {
        gpuVolReady = engine.uploadVolume(panelFloats, nx, ny, nz);
        volUploadedKeyRef.current = gpuVolReady ? panelFloats : null;
      } else {
        gpuVolReady = true;
      }
      if (gpuVolReady) engine.uploadLUT(cmap, lut);
    }
    gpuVolReadyRef.current = gpuVolReady;
    const { vmin, vmax } = volumeDisplayWindow(displayDataRange, imageVminPct, imageVmaxPct, flip);
    for (let axis = 0; axis < sliceDims.length; axis++) {
      if (!axisChanged[axis]) continue;
      const [sliceH, sliceW] = sliceDims[axis];
      if (gpuVolReady && engine) {
        // Always cache the native slice raster. The displayed panel may be
        // smaller, but zoom/pan must reveal source pixels instead of magnifying
        // a display-resolution scrub proxy.
        const gpuAxis = axis === 0 ? 0 : 3;
        const bitmap = engine.renderVolumeSliceToImageBitmap(
          gpuAxis,
          axis === 0 ? sliceZ : 0,
          { vmin, vmax },
          logScale,
          flip,
          undefined,
          undefined,
          gpuSliceParams,
        );
        if (bitmap) {
          let offscreen = sliceOffscreenRefs.current[axis];
          if (!offscreen || offscreen.width !== bitmap.width || offscreen.height !== bitmap.height) {
            offscreen = document.createElement("canvas");
            offscreen.width = bitmap.width; offscreen.height = bitmap.height;
            sliceOffscreenRefs.current[axis] = offscreen;
            sliceImgDataRefs.current[axis] = null;
          }
          const offscreenCtx = offscreen.getContext("2d");
          if (offscreenCtx) { offscreenCtx.clearRect(0, 0, offscreen.width, offscreen.height); offscreenCtx.drawImage(bitmap, 0, 0); }
          bitmap.close();
          continue;
        }
      }
      // CPU fallback
      const processed = maybeFlip(logScale ? applyLogScale(extractors[axis]()) : extractors[axis](), flip);
      const offscreen = sliceOffscreenRefs.current[axis];
      const imgData = sliceImgDataRefs.current[axis];
      if (offscreen && imgData && offscreen.width === sliceW && offscreen.height === sliceH) {
        renderToOffscreenReuse(processed, lut, vmin, vmax, offscreen, imgData);
      } else {
        sliceOffscreenRefs.current[axis] = renderToOffscreen(processed, sliceW, sliceH, lut, vmin, vmax);
      }
    }
    prevCacheRef.current = { sliceX, sliceY, sliceZ, cmap, logScale, imageVminPct, imageVmaxPct, imageRangeMin: displayDataRange.min, imageRangeMax: displayDataRange.max, panelFloats, nx, ny, nz, traitVmin, traitVmax, flip, alignRowShift, alignColShift, alignStartX, alignStartY, alignStopX, alignStopY };
  }, [panelFloats, sliceX, sliceY, sliceZ, obliqueAngle, obliqueSegment, nx, ny, nz, cmap, logScale, sliceDims, imageVminPct, imageVmaxPct, displayDataRange, traitVmin, traitVmax, flip, cmapReady, gpuSliceParams]);

  // Snapshot of everything direct-paint needs, refreshed every render so the
  // slider handler (which fires faster than React commits) reads current values.
  React.useEffect(() => {
    paintParamsRef.current = {
      cmap, logScale, flip, imageVminPct, imageVmaxPct, imageDataRange: displayDataRange,
      traitVmin, traitVmax, zooms, canvasSizes, smooth, alignment: gpuSliceParams,
    };
  });

  // DIRECT PAINT (Show3D's 60fps-at-4k trick): paint ONE plane straight to its
  // visible canvas via the resident-volume GPU slice path, bypassing React. The
  // slice sliders are anywidget model traits (slice_x/y/z) whose setter does a
  // comm round-trip (model.set + save_changes) that React BATCHES during a drag,
  // so the render effect keyed on them doesn't fire per drag-frame -> the lag.
  // The slider onChange calls this for an INSTANT image, then sets the trait for
  // crosshair/title/state to catch up. The shader samples the float32 resident
  // volume and area-averages every source pixel covered by the displayed pixel.
  const directPaintPlane = React.useCallback((axis: number, idx: number): boolean => {
    const engine = gpuCmapRef.current;
    const paint = paintParamsRef.current;
    if (axis !== 0 && axis !== 1) return false;
    if (!engine || !gpuVolReadyRef.current || !paint) return false;
    const canvas = canvasRefs.current[axis];
    if (!canvas) return false;
    const canvasSize = paint.canvasSizes[axis]; const view = paint.zooms[axis];
    if (!canvasSize) return false;
    const { vmin, vmax } = volumeDisplayWindow(paint.imageDataRange, paint.imageVminPct, paint.imageVmaxPct, paint.flip);
    engine.uploadLUT(paint.cmap, COLORMAPS[paint.cmap] || COLORMAPS.inferno);
    const canvasW = canvasSize.w, canvasH = canvasSize.h;
    const renderAxis = axis === 0 ? 0 : 3;
    const renderIndex = axis === 0 ? idx : 0;
    const bitmap = engine.renderVolumeSliceToImageBitmap(
      renderAxis,
      renderIndex,
      { vmin, vmax },
      paint.logScale,
      paint.flip,
      undefined,
      { zoom: view?.zoom || 1, panX: view?.panX || 0, panY: view?.panY || 0, canvasW, canvasH },
      paint.alignment,
    );
    if (!bitmap) return false;
    const ctx = canvas.getContext("2d");
    if (!ctx) { bitmap.close(); return false; }
    ctx.imageSmoothingEnabled = paint.smooth;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, 0, 0, canvasW, canvasH);
    bitmap.close();
    return true;
  }, []);

  const renderVolumePlanesLive = React.useCallback(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer || !volumeFloats || volumeFloats.length === 0) return;
    const params = { ...volumeRenderParamsRef.current, ...liveSliceParamsRef.current };
    volumeRenderParamsRef.current = params;
    renderer.render(
      params,
      liveCameraRef.current,
      bgColorRef.current,
      1,
      32,
      zStretchRef.current,
      orthographic,
    );
  }, [orthographic, volumeFloats]);

  const previewSliceAlignment = React.useCallback((rowShift: number, colShift: number) => {
    const current = paintParamsRef.current;
    const alignment = {
      rowShift,
      colShift,
      segment: { start: obliqueSegment.start, stop: obliqueSegment.stop },
    };
    if (current) paintParamsRef.current = { ...current, alignment };
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      rowShiftPxPerSlice: rowShift,
      colShiftPxPerSlice: colShift,
    };
    const slices = liveSliderRef.current;
    directPaintPlane(0, slices[0]);
    directPaintPlane(1, 0);
    renderVolumePlanesLive();
  }, [directPaintPlane, obliqueSegment, renderVolumePlanesLive]);

  alignmentPreviewRef.current = previewSliceAlignment;

  // -------------------------------------------------------------------------
  // Redraw slices with zoom/pan (cheap: just drawImage from cached offscreen)
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  // -------------------------------------------------------------------------
  React.useLayoutEffect(() => {
    for (let axis = 0; axis < sliceDims.length; axis++) {
      const canvas = canvasRefs.current[axis];
      const offscreen = sliceOffscreenRefs.current[axis];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      const { w: canvasW, h: canvasH } = canvasSizes[axis];
      drawZoomedOffscreen(ctx, offscreen, canvasW, canvasH, zooms[axis], smooth);
    }
    // gpuSliceParams belongs here: toggling Align re-renders the offscreens with
    // new depth shifts, but without this dependency the blit never re-runs and
    // the visible canvases keep the previous alignment.
  }, [panelFloats, sliceX, sliceY, sliceZ, obliqueAngle, nx, ny, nz, cmap, logScale, autoContrast, zooms, sliceDims, canvasSizes, imageVminPct, imageVmaxPct, smooth, flip, gpuSliceParams]);

  // -------------------------------------------------------------------------
  // Render crosshair lines for the orthogonal slice intersections.
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!panelFloats) return;
    const crossPositions: [number, number][] = [
      [sliceX, sliceY],
      [(sliceDims[1]?.[1] ?? 1) / 2, sliceZ],
    ];
    for (let axis = 0; axis < sliceDims.length; axis++) {
      const overlay = overlayRefs.current[axis];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      const { w: canvasW, h: canvasH, displayH, scaleX, scaleY } = canvasSizes[axis];
      const stretchY = displayH / canvasH;
      ctx.clearRect(0, 0, canvasW, displayH);
      if (axis === 0) {
        const { start, stop } = obliqueSegment;
        const startCanvas = dataPointToCanvas(axis, start.x, start.y);
        const stopCanvas = dataPointToCanvas(axis, stop.x, stop.y);
        ctx.save();
        ctx.strokeStyle = themeColors.accent;
        ctx.fillStyle = themeColors.accent;
        ctx.lineWidth = obliqueHoverTarget === "line" || obliqueHandleDragRef.current?.mode === "line" ? 2.5 : 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(startCanvas.x, startCanvas.y);
        ctx.lineTo(stopCanvas.x, stopCanvas.y);
        ctx.stroke();
        ctx.setLineDash([]);
        const drawHandle = (point: { x: number; y: number }) => {
          const activeEndpoint = obliqueHoverTarget === "endpoint" || obliqueHandleDragRef.current?.mode === "endpoint";
          const radius = activeEndpoint ? 6 : 4;
          const x = clampNumber(point.x, radius + 2, canvasW - radius - 2);
          const y = clampNumber(point.y, radius + 2, displayH - radius - 2);
          if (activeEndpoint) {
            ctx.save();
            ctx.strokeStyle = themeColors.bg;
            ctx.lineWidth = 2;
          }
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.fill();
          if (activeEndpoint) {
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x - radius - 4, y);
            ctx.lineTo(x - radius - 1, y);
            ctx.moveTo(x + radius + 1, y);
            ctx.lineTo(x + radius + 4, y);
            ctx.moveTo(x, y - radius - 4);
            ctx.lineTo(x, y - radius - 1);
            ctx.moveTo(x, y + radius + 1);
            ctx.lineTo(x, y + radius + 4);
            ctx.stroke();
            ctx.restore();
          }
        };
        drawHandle(startCanvas);
        drawHandle(stopCanvas);
        ctx.restore();
      }
      if (!showCrosshair) continue;
      const view = zooms[axis];
      const [dataX, dataY] = crossPositions[axis];
      const centerX = canvasW / 2, centerY = displayH / 2;
      let canvasX = dataX * scaleX;
      let canvasY = dataY * scaleY * stretchY;
      if (view.zoom !== 1 || view.panX !== 0 || view.panY !== 0) {
        canvasX = (canvasX - centerX) * view.zoom + centerX + view.panX;
        canvasY = (canvasY - centerY) * view.zoom + centerY + view.panY * stretchY;
      }
      ctx.strokeStyle = themeColors.accentYellow + "80";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(canvasX, 0); ctx.lineTo(canvasX, displayH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, canvasY); ctx.lineTo(canvasW, canvasY); ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [panelFloats, sliceX, sliceY, sliceZ, obliqueAngle, obliqueSegment, obliqueHoverTarget, zooms, showCrosshair, themeColors, canvasSizes, sliceDims, nx, ny, dataPointToCanvas]);

  // -------------------------------------------------------------------------
  // Scale bar (HiDPI UI overlay)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    for (let axis = 0; axis < sliceDims.length; axis++) {
      const uiCanvas = uiRefs.current[axis];
      if (!uiCanvas) continue;
      const { w: canvasW, displayH } = canvasSizes[axis];
      uiCanvas.width = Math.round(canvasW * DPR);
      uiCanvas.height = Math.round(displayH * DPR);
      const uiCtx = uiCanvas.getContext("2d");
      if (!uiCtx) continue;
      uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
      if (scaleBarVisible) {
        const axes = pixelSizeAxes && pixelSizeAxes.length === 3 ? pixelSizeAxes : null;
        const theta = (obliqueAngle * Math.PI) / 180;
        const obliquePx = axes
          ? Math.hypot(Math.cos(theta) * axes[2], Math.sin(theta) * axes[1])
          : (pixelSize || 0);
        const pxSize = axis === 0 ? (axes ? axes[2] : (pixelSize || 0)) : obliquePx;
        const sliceW = sliceDims[axis][1];
        const unit = pxSize > 0 ? "Å" : "px";
        const size = pxSize > 0 ? pxSize : 1;
        drawScaleBarHiDPI(uiCanvas, DPR, zooms[axis].zoom, size, unit, sliceW);
      }

      if (showColorbar) {
        const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
        // The window the slices are painted with, so the labels follow Flip.
        const { vmin, vmax } = volumeDisplayWindow(displayDataRange, imageVminPct, imageVmaxPct, flip);
        const cssW = uiCanvas.width / DPR;
        const cssH = uiCanvas.height / DPR;
        uiCtx.save();
        uiCtx.scale(DPR, DPR);
        drawColorbar(uiCtx, cssW, cssH, lut, vmin, vmax, logScale);
        uiCtx.restore();
      }
    }
  }, [pixelSize, pixelSizeAxes, scaleBarVisible, zooms, canvasSizes, sliceDims, showColorbar, cmap, displayDataRange, imageVminPct, imageVmaxPct, flip, obliqueAngle, themeInfo.theme]);

  // -------------------------------------------------------------------------
  // FFT computation and caching (per-axis: only recompute changed axes)
  // -------------------------------------------------------------------------
  const prevFFTCacheRef = React.useRef<{
    sliceX: number; sliceY: number; sliceZ: number;
    panelFloats: Float32Array | null;
    fftColormap: string; fftLogScale: boolean; fftAuto: boolean; fftWindow: boolean; gpuReady: boolean;
    showFft: boolean;
  }>({ sliceX: -1, sliceY: -1, sliceZ: -1, panelFloats: null, fftColormap: "", fftLogScale: false, fftAuto: false, fftWindow: false, gpuReady: false, showFft: false });

  React.useEffect(() => {
    if (!showFft || !panelFloats || panelFloats.length === 0) {
      // Release FFT caches when toggling off (each is up to 64 MB per axis).
      if (prevFFTCacheRef.current.showFft && !showFft) {
        for (let axis = 0; axis < sliceDims.length; axis++) {
          fftMagCacheRefs.current[axis] = null;
          fftOffscreenRefs.current[axis] = null;
          fftImgDataRefs.current[axis] = null;
        }
        prevFFTCacheRef.current.showFft = false;
      }
      return;
    }

    const prevFFT = prevFFTCacheRef.current;
    const globalFFTChanged = panelFloats !== prevFFT.panelFloats || fftColormap !== prevFFT.fftColormap ||
      fftLogScale !== prevFFT.fftLogScale || fftAuto !== prevFFT.fftAuto ||
      fftWindow !== prevFFT.fftWindow ||
      gpuReady !== prevFFT.gpuReady || !prevFFT.showFft;
    const fftAxisChanged = [
      globalFFTChanged || sliceZ !== prevFFT.sliceZ,
      true,
    ];

    const generation = ++fftComputeGenerationRef.current;
    let cancelled = false;

    const computeFFTsForVolume = async (
      floats: Float32Array,
      magCache: React.MutableRefObject<(Float32Array | null)[]>,
      offscreenCache: React.MutableRefObject<(HTMLCanvasElement | null)[]>,
      imgDataCache: React.MutableRefObject<(ImageData | null)[]>,
      forceAll: boolean,
    ) => {
      const extractors = [
        () => extractXY(floats, nx, ny, nz, sliceZ),
        () => extractOblique(floats, nx, ny, nz, obliqueSegment.start, obliqueSegment.stop),
      ];
      const dims = sliceDims;

      for (let axis = 0; axis < sliceDims.length; axis++) {
        if (!forceAll && !fftAxisChanged[axis]) continue;
        const [sliceH, sliceW] = dims[axis];
        const cacheKey = makeFftResultCacheKey(axis, sliceZ, obliqueSegment, sliceW, sliceH);
        const cached = fftResultCacheRef.current.get(cacheKey);
        if (cached) {
          cached.lastUsed = ++fftResultCacheSeqRef.current;
          applyFftResultCacheEntry(axis, cached, magCache, offscreenCache, imgDataCache);
          continue;
        }
        const extracted = extractors[axis]();
        const data = fftWindow ? new Float32Array(extracted) : extracted;
        if (fftWindow) applyHannWindow2D(data, sliceW, sliceH);
        const paddedW = nextPow2(sliceW);
        const paddedH = nextPow2(sliceH);
        let real: Float32Array;
        let imag: Float32Array;
        if (gpuReady && gpuFFTRef.current) {
          const result = await gpuFFTRef.current.fft2D(zeroPadSlice(data, sliceW, sliceH, paddedW, paddedH), new Float32Array(paddedW * paddedH), paddedW, paddedH, false);
          real = result.real;
          imag = result.imag;
        } else {
          real = zeroPadSlice(data, sliceW, sliceH, paddedW, paddedH);
          imag = new Float32Array(paddedW * paddedH);
          fft2d(real, imag, paddedW, paddedH, false);
        }
        const entry = storeFftResult(cacheKey, real, imag, paddedW, paddedH);
        // Paint the offscreen only; the cheap redraw effect below blits it.
        applyFftResultCacheEntry(axis, entry, magCache, offscreenCache, imgDataCache);
      }
    };

    const computeAllFFTs = async () => {
      const localMagCache = { current: fftMagCacheRefs.current.map((value, axis) => fftAxisChanged[axis] ? null : value) } as React.MutableRefObject<(Float32Array | null)[]>;
      const localOffscreenCache = { current: fftOffscreenRefs.current.map((value, axis) => fftAxisChanged[axis] ? null : value) } as React.MutableRefObject<(HTMLCanvasElement | null)[]>;
      const localImgDataCache = { current: fftImgDataRefs.current.map((value, axis) => fftAxisChanged[axis] ? null : value) } as React.MutableRefObject<(ImageData | null)[]>;
      await computeFFTsForVolume(panelFloats, localMagCache, localOffscreenCache, localImgDataCache, false);
      if (cancelled || generation !== fftComputeGenerationRef.current) return false;
      fftMagCacheRefs.current = localMagCache.current;
      fftOffscreenRefs.current = localOffscreenCache.current;
      fftImgDataRefs.current = localImgDataCache.current;
      prevFFTCacheRef.current = { sliceX, sliceY, sliceZ, panelFloats, fftColormap, fftLogScale, fftAuto, fftWindow, gpuReady, showFft };
      return true;
    };

    // Debounce FFT compute during slider scrubbing: defer 80 ms so a 60 Hz drag
    // collapses to ~12 Hz, freeing the main thread for image redraws. Oblique
    // line edits are a direct visual inspection path, so keep the side FFT close
    // to frame rate while the endpoint or line body is actively dragged.
    const liveObliqueEditing = obliqueHandleDragRef.current !== null;
    const debounceMs = liveObliqueEditing ? 16 : 80;
    const timeoutId = setTimeout(() => {
      if (cancelled) return;
      computeAllFFTs().then((committed) => { if (committed) setFftVersion((version) => version + 1); });
    }, debounceMs);
    return () => { cancelled = true; clearTimeout(timeoutId); };
  }, [showFft, panelFloats, sliceX, sliceY, sliceZ, obliqueAngle, obliqueSegment, nx, ny, nz, sliceDims, fftColormap, fftLogScale, fftAuto, fftWindow, gpuReady]);

  // Redraw cached FFT with zoom/pan (cheap -- no recomputation)
  React.useLayoutEffect(() => {
    if (!showFft) return;
    for (let axis = 0; axis < sliceDims.length; axis++) {
      const canvas = fftCanvasRefs.current[axis];
      const offscreen = fftOffscreenRefs.current[axis];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      const { w: canvasW, h: canvasH } = canvasSizes[axis];
      drawZoomedOffscreen(ctx, offscreen, canvasW, canvasH, fftZooms[axis], smooth);
    }
  }, [showFft, fftZooms, canvasSizes, sliceDims, fftVersion, smooth]);

  // Render FFT overlays (reciprocal-space scale bars + d-spacing crosshair per axis)
  React.useEffect(() => {
    if (!showFft) return;
    const dims = sliceDims;
    for (let axis = 0; axis < sliceDims.length; axis++) {
      const overlay = fftOverlayRefs.current[axis];
      if (!overlay) continue;
      const { w: canvasW, h: canvasH, displayH } = canvasSizes[axis];
      const stretchY = displayH / canvasH;
      overlay.width = Math.round(canvasW * DPR);
      overlay.height = Math.round(displayH * DPR);
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      const axes = pixelSizeAxes && pixelSizeAxes.length === 3 ? pixelSizeAxes : null;
      const theta = (obliqueAngle * Math.PI) / 180;
      const obliquePx = axes
        ? Math.hypot(Math.cos(theta) * axes[2], Math.sin(theta) * axes[1])
        : pixelSize;
      const realPx = axis === 0 ? (axes ? axes[2] : pixelSize) : obliquePx;
      if (realPx > 0) {
        const [, sliceW] = dims[axis];
        const paddedW = nextPow2(sliceW);
        const fftPixelSize = 1 / (paddedW * realPx);
        drawFFTScaleBarHiDPI(overlay, DPR, fftZooms[axis].zoom, fftPixelSize, paddedW, "Å⁻¹");
      }

      if (fftClickInfo && fftClickInfo.axis === axis) {
        const [sliceH, sliceW] = dims[axis];
        const fftW = nextPow2(sliceW);
        const fftH = nextPow2(sliceH);

        ctx.save();
        ctx.scale(DPR, DPR);
        const view = fftZooms[axis];
        const centerX = canvasW / 2, centerY = displayH / 2;
        const rawX = fftClickInfo.col / fftW * canvasW;
        const rawY = fftClickInfo.row / fftH * displayH;
        const screenX = (rawX - centerX) * view.zoom + centerX + view.panX;
        const screenY = (rawY - centerY) * view.zoom + centerY + view.panY * stretchY;

        ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
        ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
        ctx.shadowBlur = 2;
        ctx.lineWidth = 1.5;
        const crossRadius = 8;
        ctx.beginPath();
        ctx.moveTo(screenX - crossRadius, screenY); ctx.lineTo(screenX - 3, screenY);
        ctx.moveTo(screenX + 3, screenY); ctx.lineTo(screenX + crossRadius, screenY);
        ctx.moveTo(screenX, screenY - crossRadius); ctx.lineTo(screenX, screenY - 3);
        ctx.moveTo(screenX, screenY + 3); ctx.lineTo(screenX, screenY + crossRadius);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
        ctx.stroke();

        if (fftClickInfo.dSpacing != null) {
          const dSpacing = fftClickInfo.dSpacing;
          const label = dSpacing >= 10 ? `d = ${(dSpacing / 10).toFixed(2)} nm` : `d = ${dSpacing.toFixed(2)} \u00C5`;
          ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
          ctx.fillStyle = "white";
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(label, screenX + 10, screenY - 4);
        }
        ctx.restore();
      }
    }
  }, [showFft, fftZooms, canvasSizes, pixelSize, pixelSizeAxes, nx, ny, nz, fftClickInfo]);

  // -------------------------------------------------------------------------
  // Playback logic (matching Show3D pattern)
  // -------------------------------------------------------------------------
  const sliceSettersRef = React.useRef<((value: number) => void)[]>([setSliceZ, setSliceY, setSliceX]);
  sliceSettersRef.current = [setSliceZ, setSliceY, setSliceX];
  const obliquePlaybackStateRef = React.useRef({ current: 0, start: 0, end: 0 });
  const effectiveLoopEnds = React.useMemo(() => loopEnds.map((end, i) => {
    const max = [nz - 1, ny - 1, nx - 1][i];
    return end < 0 ? max : Math.min(end, max);
  }), [loopEnds, nz, ny, nx]);
  React.useEffect(() => {
    if (!playing) return;
    let cancelled = false;
    let hiddenPaused = false;

    const clearPlayFrame = () => {
      if (playRafRef.current != null) {
        cancelAnimationFrame(playRafRef.current);
        playRafRef.current = null;
      }
    };

    const playbackAxes = playbackAxis === 3 ? [0, 1] : [playbackAxis];
    const axisBounds = (axis: number) => {
      if (axis === 0) return { start: loopStarts[0], end: effectiveLoopEnds[0], current: sliceValuesRef.current[0] };
      const state = obliquePlaybackStateRef.current;
      return { start: state.start, end: state.end, current: state.current };
    };
    const setPlaybackAxisFast = (axis: number, value: number) => {
      if (axis === 0) {
        if (fastTrackSliceRef.current) fastTrackSliceRef.current(0, value);
        else sliceSettersRef.current[0](value);
        sliceValuesRef.current[0] = value;
        return;
      }
      obliquePlaybackStateRef.current = { ...obliquePlaybackStateRef.current, current: value };
      updateObliqueFromNormalOffset(value);
    };

    const advanceAllAxes = (): boolean => {
      const dir = boomerang ? bounceDirRef.current : (reverse ? -1 : 1);
      let wouldHitEdge = false;
      for (const axis of playbackAxes) {
        const { start, end, current } = axisBounds(axis);
        const next = current + dir;
        if (next > end || next < start) {
          wouldHitEdge = true;
          break;
        }
      }
      if (boomerang && wouldHitEdge) {
        bounceDirRef.current = (-bounceDirRef.current) as 1 | -1;
      }
      const finalDir = boomerang ? bounceDirRef.current : dir;
      for (const axis of playbackAxes) {
        const { start, end, current } = axisBounds(axis);
        let next = current + finalDir;
        if (next > end) next = loop || boomerang ? start : end;
        else if (next < start) next = loop || boomerang ? end : start;
        setPlaybackAxisFast(axis, next);
      }
      return !loop && !boomerang && wouldHitEdge;
    };

    const advanceSingleAxis = (): boolean => {
      const axis = playbackAxis === 3 ? 0 : playbackAxis;
      const { start, end, current: prev } = axisBounds(axis);
      let next = prev;
      let hitStop = false;
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
        if (reverse && next < start) {
          hitStop = !loop;
          next = loop ? end : start;
        } else if (!reverse && next > end) {
          hitStop = !loop;
          next = loop ? start : end;
        }
      }
      setPlaybackAxisFast(axis, next);
      return hitStop;
    };

    const advanceOnce = () => (playbackAxis === 3 ? advanceAllAxes() : advanceSingleAxis());

    const tick = (ts: number) => {
      if (cancelled) return;
      const fpsSafe = Math.max(1, Math.min(MAX_PLAYBACK_FPS, Math.round(fpsRef.current || 1)));
      const intervalMs = 1000 / fpsSafe;
      const lastTs = lastPlayTsRef.current;
      lastPlayTsRef.current = ts;
      if (lastTs != null) {
        playAccumulatorRef.current += ts - lastTs;
        if (playAccumulatorRef.current > intervalMs * 4) {
          playAccumulatorRef.current = intervalMs;
        }
      }

      let steps = 0;
      while (playAccumulatorRef.current >= intervalMs && steps < 3) {
        playAccumulatorRef.current -= intervalMs;
        steps += 1;
        if (advanceOnce()) {
          setPlaying(false);
          return;
        }
      }
      playRafRef.current = requestAnimationFrame(tick);
    };

    const startFrameLoop = () => {
      if (playRafRef.current != null) return;
      lastPlayTsRef.current = null;
      playAccumulatorRef.current = 0;
      playRafRef.current = requestAnimationFrame(tick);
    };

    startFrameLoop();

    const onVis = () => {
      if (document.hidden) {
        hiddenPaused = playRafRef.current != null;
        clearPlayFrame();
      } else if (hiddenPaused) {
        hiddenPaused = false;
        startFrameLoop();
      }
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVis);
      clearPlayFrame();
      commitSliceValuesRef.current();
    };
  }, [playing, reverse, boomerang, loop, playbackAxis, loopStarts, effectiveLoopEnds]);

  // -------------------------------------------------------------------------
  // Direct canvas draw (bypasses React state for 60fps pan during drag)
  // -------------------------------------------------------------------------
  const drawSliceDirect = (axis: number) => {
    const view = liveZoomsRef.current[axis];
    const { w: canvasW, h: canvasH } = canvasSizes[axis];
    const canvas = canvasRefs.current[axis];
    const offscreen = sliceOffscreenRefs.current[axis];
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawZoomedOffscreen(ctx, offscreen, canvasW, canvasH, view, smooth);
  };

  const drawFftDirect = (axis: number) => {
    const view = liveFftZoomsRef.current[axis];
    const { w: canvasW, h: canvasH } = canvasSizes[axis];
    const canvas = fftCanvasRefs.current[axis];
    const offscreen = fftOffscreenRefs.current[axis];
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawZoomedOffscreen(ctx, offscreen, canvasW, canvasH, view, smooth);
  };

  // -------------------------------------------------------------------------
  // Zoom/Pan handlers (matching Show3D)
  // -------------------------------------------------------------------------
  const commitLiveZoomsNow = () => {
    if (zoomCommitTimeoutRef.current != null) {
      window.clearTimeout(zoomCommitTimeoutRef.current);
      zoomCommitTimeoutRef.current = null;
    }
    liveZoomDirtyRef.current = false;
    const next = liveZoomsRef.current;
    setZooms(next);
    persistViewState(next, undefined, undefined);
  };
  const commitLiveZoomsSoon = () => {
    liveZoomDirtyRef.current = true;
    if (zoomCommitTimeoutRef.current != null) window.clearTimeout(zoomCommitTimeoutRef.current);
    zoomCommitTimeoutRef.current = window.setTimeout(commitLiveZoomsNow, 120);
  };
  const commitLiveFftZoomsNow = () => {
    if (fftZoomCommitTimeoutRef.current != null) {
      window.clearTimeout(fftZoomCommitTimeoutRef.current);
      fftZoomCommitTimeoutRef.current = null;
    }
    liveFftZoomDirtyRef.current = false;
    const next = liveFftZoomsRef.current;
    setFftZooms(next);
    persistViewState(undefined, next, undefined);
  };
  const commitLiveFftZoomsSoon = () => {
    liveFftZoomDirtyRef.current = true;
    if (fftZoomCommitTimeoutRef.current != null) window.clearTimeout(fftZoomCommitTimeoutRef.current);
    fftZoomCommitTimeoutRef.current = window.setTimeout(commitLiveFftZoomsNow, 120);
  };
  const handleWheel = (event: React.WheelEvent, axis: number) => {
    const canvas = canvasRefs.current[axis];
    if (!canvas) return;
    // The native passive:false listener above already blocks page scrolling.
    // React delegates wheel events through a passive root listener in Chrome.
    const next = [...liveZoomsRef.current];
    next[axis] = wheelZoomAboutPointer(canvas, event, liveZoomsRef.current[axis]);
    liveZoomsRef.current = next;
    if (!zoomRafRef.current) {
      zoomRafRef.current = requestAnimationFrame(() => {
        zoomRafRef.current = 0;
        drawSliceDirect(axis);
      });
    }
    commitLiveZoomsSoon();
  };

  const clickJumpTimerRef = React.useRef<number | null>(null);

  const handleDoubleClick = (axis: number) => {
    if (clickJumpTimerRef.current !== null) {
      window.clearTimeout(clickJumpTimerRef.current);
      clickJumpTimerRef.current = null;
    }
    const next = [...liveZoomsRef.current];
    next[axis] = DEFAULT_ZOOM;
    liveZoomsRef.current = next;
    commitLiveZoomsNow();
  };

  // Synchronous click-detection ref: synthetic events (CDP, automation) fire
  // mousedown→mouseup back-to-back before React commits setDragStart. The ref
  // is always current, so handleMouseUp can detect a stationary click even
  // when dragStart state hasn't been flushed yet.
  const clickStartRef = React.useRef<{ x: number; y: number; axis: number } | null>(null);
  const obliqueHandleDragRef = React.useRef<{
    mode: "endpoint";
    handle: "start" | "stop";
    opposite: { x: number; y: number };
  } | {
    mode: "line";
    angleDeg: number;
    origin: { x: number; y: number };
    start: { x: number; y: number };
    stop: { x: number; y: number };
  } | null>(null);
  const obliquePositionDragRef = React.useRef<{
    angleDeg: number;
    currentOffset: number;
    minOffset: number;
    maxOffset: number;
    start: { x: number; y: number };
    stop: { x: number; y: number };
  } | null>(null);
  const [liveObliqueOffset, setLiveObliqueOffset] = React.useState<number | null>(null);
  const [obliqueAngleBounds, setObliqueAngleBounds] = React.useState<[number, number] | null>(null);
  const [obliquePositionBounds, setObliquePositionBounds] = React.useState<[number, number] | null>(null);
  const imagePointFromEvent = (event: React.MouseEvent | MouseEvent, axis: number): { col: number; row: number } | null => {
    const canvas = canvasRefs.current?.[axis];
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const canvasX = (event.clientX - rect.left) * (canvas.width / rect.width);
    const canvasY = (event.clientY - rect.top) * (canvas.height / rect.height);
    const { w: canvasW, h: canvasH, scaleX, scaleY } = canvasSizes[axis];
    const view = liveZoomsRef.current[axis];
    const centerX = canvasW / 2, centerY = canvasH / 2;
    const col = ((canvasX - centerX - view.panX) / view.zoom + centerX) / scaleX;
    const row = ((canvasY - centerY - view.panY) / view.zoom + centerY) / scaleY;
    return { col, row };
  };
  const updateObliqueCenter = (
    cx: number,
    cy: number,
    angleDeg = obliqueAngle,
    segment: { start: { x: number; y: number }; stop: { x: number; y: number } } = obliqueSegment,
  ) => {
    const x = clampNumber(Math.round(cx), 0, nx - 1);
    const y = clampNumber(Math.round(cy), 0, ny - 1);
    setSliceX(x);
    setSliceY(y);
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      sliceX: x,
      sliceY: y,
      obliqueAngleDeg: angleDeg,
      obliqueStartX: segment.start.x,
      obliqueStartY: segment.start.y,
      obliqueEndX: segment.stop.x,
      obliqueEndY: segment.stop.y,
    };
    // Repaint the oblique panel straight from the resident GPU volume. The
    // segment lives in anywidget model traits whose setters round-trip through
    // the comm, and React batches those during a drag, so waiting for the render
    // effect would only move the panel once the drag ENDS. Seeding the paint
    // params with the segment we just computed keeps the panel live per frame.
    const paint = paintParamsRef.current;
    if (paint?.alignment) {
      paintParamsRef.current = {
        ...paint,
        alignment: { ...paint.alignment, segment: { start: segment.start, stop: segment.stop } },
      };
    }
    directPaintPlane(1, 0);
    const renderer = volumeRendererRef.current;
    if (renderer && volumeFloats && volumeFloats.length > 0) {
      renderer.render(
        volumeRenderParamsRef.current,
        liveCameraRef.current,
        bgColorRef.current,
        undefined,
        undefined,
        zStretchRef.current,
        orthographic,
      );
    }
  };
  const updateObliqueFromEndpoints = (moving: { x: number; y: number }, opposite: { x: number; y: number }) => {
    const start = clampPointToImage(moving, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    const stop = clampPointToImage(opposite, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    const cx = clampNumber(Math.round((start.x + stop.x) / 2), 0, nx - 1);
    const cy = clampNumber(Math.round((start.y + stop.y) / 2), 0, ny - 1);
    const rawAngle = (Math.atan2(start.y - stop.y, start.x - stop.x) * 180) / Math.PI;
    const nextAngle = ((rawAngle % 180) + 180) % 180;
    setObliqueAngle(nextAngle);
    setObliqueProfileLine(profileLinePayload(start, stop));
    setObliquePositionBounds(null);
    updateObliqueCenter(cx, cy, nextAngle, { start, stop });
    scheduleLiveFft([1], { segment: { start, stop } });
  };
  const updateObliqueFromNormalOffset = (offset: number) => {
    const dragBasis = obliquePositionDragRef.current;
    const angleDeg = dragBasis?.angleDeg ?? obliqueAngle;
    const baseStart = dragBasis?.start ?? obliqueSegment.start;
    const baseStop = dragBasis?.stop ?? obliqueSegment.stop;
    const currentOffset = dragBasis?.currentOffset ?? obliqueCenterOffset(nx, ny, angleDeg, baseStart, baseStop);
    const [minDelta, maxDelta] = dragBasis
      ? [dragBasis.minOffset - dragBasis.currentOffset, dragBasis.maxOffset - dragBasis.currentOffset]
      : obliqueSegmentOffsetBounds(nx, ny, angleDeg, baseStart, baseStop, OBLIQUE_PROFILE_EDGE_INSET);
    const normal = obliqueNormal(angleDeg);
    const nextOffset = clampNumber(offset, currentOffset + minDelta, currentOffset + maxDelta);
    const delta = nextOffset - currentOffset;
    const dx = normal.x * delta;
    const dy = normal.y * delta;
    const start = clampPointToImage({ x: baseStart.x + dx, y: baseStart.y + dy }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    const stop = clampPointToImage({ x: baseStop.x + dx, y: baseStop.y + dy }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    obliquePlaybackStateRef.current = {
      ...obliquePlaybackStateRef.current,
      current: Math.round(nextOffset),
    };
    setLiveObliqueOffset(Math.round(nextOffset));
    setObliqueProfileLine(profileLinePayload(start, stop));
    updateObliqueCenter((start.x + stop.x) / 2, (start.y + stop.y) / 2, angleDeg, { start, stop });
    scheduleLiveFft([1], { segment: { start, stop } });
  };
  const updateObliqueFromLineDrag = (
    drag: {
      angleDeg: number;
      origin: { x: number; y: number };
      start: { x: number; y: number };
      stop: { x: number; y: number };
    },
    point: { col: number; row: number },
  ) => {
    const dxRaw = point.col - drag.origin.x;
    const dyRaw = point.row - drag.origin.y;
    const { start, stop } = translateSegmentInsideImage(
      drag.start,
      drag.stop,
      dxRaw,
      dyRaw,
      nx,
      ny,
      OBLIQUE_PROFILE_EDGE_INSET,
    );
    setLiveObliqueOffset(null);
    setObliquePositionBounds(null);
    setObliqueProfileLine(profileLinePayload(start, stop));
    updateObliqueCenter((start.x + stop.x) / 2, (start.y + stop.y) / 2, drag.angleDeg, { start, stop });
    scheduleLiveFft([1], { segment: { start, stop } });
  };
  // What the pointer is over on the Top panel's oblique line: the nearer
  // endpoint handle (screen-space radius around the drawn, edge-clamped handle)
  // or the line body (image-space distance). Null off the Top panel or image.
  const obliqueHitTest = (event: React.MouseEvent, axis: number): { target: "start" | "stop" | "line" | null; point: { col: number; row: number } } | null => {
    if (axis !== 0) return null;
    const canvas = canvasRefs.current?.[axis];
    const point = imagePointFromEvent(event, axis);
    if (!canvas || !point) return null;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (event.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (event.clientY - rect.top) * (canvas.height / rect.height);
    const { start, stop } = obliqueSegment;
    const { w: hitW, displayH: hitH, scaleX, scaleY } = canvasSizes[axis];
    const view = liveZoomsRef.current[axis];
    const imageHitRadius = (screenPx: number) => screenPx / Math.max(1e-6, Math.min(scaleX, scaleY) * view.zoom);
    const visualRadius = 8;
    const startCanvas = dataPointToCanvas(axis, start.x, start.y);
    const stopCanvas = dataPointToCanvas(axis, stop.x, stop.y);
    const hitStart = {
      x: clampNumber(startCanvas.x, visualRadius + 2, hitW - visualRadius - 2),
      y: clampNumber(startCanvas.y, visualRadius + 2, hitH - visualRadius - 2),
    };
    const hitStop = {
      x: clampNumber(stopCanvas.x, visualRadius + 2, hitW - visualRadius - 2),
      y: clampNumber(stopCanvas.y, visualRadius + 2, hitH - visualRadius - 2),
    };
    const startDist = Math.hypot(mouseX - hitStart.x, mouseY - hitStart.y);
    const stopDist = Math.hypot(mouseX - hitStop.x, mouseY - hitStop.y);
    const handleRadius = 20 * (window.devicePixelRatio || 1);
    if (startDist <= handleRadius || stopDist <= handleRadius) return { target: startDist <= stopDist ? "start" : "stop", point };
    const lineDist = pointToSegmentDistance(point.col, point.row, start.x, start.y, stop.x, stop.y);
    return { target: lineDist <= imageHitRadius(32 * (window.devicePixelRatio || 1)) ? "line" : null, point };
  };
  const handleMouseDown = (event: React.MouseEvent, axis: number) => {
    if (clickJumpTimerRef.current !== null) {
      window.clearTimeout(clickJumpTimerRef.current);
      clickJumpTimerRef.current = null;
    }
    if (axis === 0 && playing && (playbackAxis === 1 || playbackAxis === 3)) {
      pausePlaybackForEdit();
    }
    const hit = obliqueHitTest(event, axis);
    if (hit?.target) {
      const { start, stop } = obliqueSegment;
      event.preventDefault();
      event.stopPropagation();
      pausePlaybackForEdit();
      if (hit.target === "line") {
        obliquePositionDragRef.current = null;
        setLiveObliqueOffset(null);
        setObliquePositionBounds(null);
        obliqueHandleDragRef.current = {
          mode: "line",
          angleDeg: obliqueAngle,
          origin: { x: hit.point.col, y: hit.point.row },
          start,
          stop,
        };
      } else {
        obliqueHandleDragRef.current = hit.target === "start"
          ? { mode: "endpoint", handle: "start", opposite: stop }
          : { mode: "endpoint", handle: "stop", opposite: start };
      }
      clickStartRef.current = null;
      setDragAxis(null);
      setDragStart(null);
      return;
    }
    const view = liveZoomsRef.current[axis];
    setDragAxis(axis);
    setDragStart({ x: event.clientX, y: event.clientY, pX: view.panX, pY: view.panY });
    clickStartRef.current = { x: event.clientX, y: event.clientY, axis };
    liveZoomDirtyRef.current = true;
  };
  React.useEffect(() => () => {
    if (clickJumpTimerRef.current !== null) window.clearTimeout(clickJumpTimerRef.current);
  }, []);

  const handleMouseMove = (event: React.MouseEvent, axis: number) => {
    if (axis === 0 && obliqueHandleDragRef.current) {
      setObliqueHoverTarget(obliqueHandleDragRef.current.mode === "endpoint" ? "endpoint" : "line");
      const point = imagePointFromEvent(event, axis);
      if (!point) return;
      const drag = obliqueHandleDragRef.current;
      if (drag.mode === "endpoint") {
        updateObliqueFromEndpoints(
          clampPointToImage({ x: point.col, y: point.row }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
          drag.opposite,
        );
      } else {
        updateObliqueFromLineDrag(drag, point);
      }
      return;
    }
    if (axis === 0) {
      const target = obliqueHitTest(event, axis)?.target ?? null;
      setObliqueHoverTarget(target === "start" || target === "stop" ? "endpoint" : target);
    } else if (obliqueHoverTarget !== null) {
      setObliqueHoverTarget(null);
    }
    if (dragAxis === axis && dragStart) {
      const canvas = canvasRefs.current?.[axis];
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const deltaX = (event.clientX - dragStart.x) * (canvas.width / rect.width);
      const deltaY = (event.clientY - dragStart.y) * (canvas.height / rect.height);
      const newZoom = { ...liveZoomsRef.current[axis], panX: dragStart.pX + deltaX, panY: dragStart.pY + deltaY };
      const next = [...liveZoomsRef.current]; next[axis] = newZoom;
      liveZoomsRef.current = next;
      if (!zoomRafRef.current) {
        zoomRafRef.current = requestAnimationFrame(() => {
          zoomRafRef.current = 0;
          drawSliceDirect(axis);
        });
      }
      return;
    }
    const cursorCanvas = canvasRefs.current?.[axis];
    if (!cursorCanvas || !panelFloats || panelFloats.length === 0) return;
    const rect = cursorCanvas.getBoundingClientRect();
    const canvasX = (event.clientX - rect.left) * (cursorCanvas.width / rect.width);
    const canvasY = (event.clientY - rect.top) * (cursorCanvas.height / rect.height);
    const { w: canvasW, h: canvasH, scaleX, scaleY } = canvasSizes[axis];
    const view = liveZoomsRef.current[axis];
    const centerX = canvasW / 2, centerY = canvasH / 2;
    let imgCol: number, imgRow: number;
    if (view.zoom !== 1 || view.panX !== 0 || view.panY !== 0) {
      imgCol = ((canvasX - centerX - view.panX) / view.zoom + centerX) / scaleX;
      imgRow = ((canvasY - centerY - view.panY) / view.zoom + centerY) / scaleY;
    } else {
      imgCol = canvasX / scaleX;
      imgRow = canvasY / scaleY;
    }
    const pixelCol = Math.floor(imgCol);
    const pixelRow = Math.floor(imgRow);
    const [sliceH, sliceW] = sliceDims[axis];
    if (pixelCol < 0 || pixelCol >= sliceW || pixelRow < 0 || pixelRow >= sliceH) {
      setCursorInfoThrottled(null);
      return;
    }
    // 3D voxel lookup. XY is a Z slice; oblique is a vertical plane through
    // the current XY center, rotated about Z.
    let value: number;
    if (axis === 0) {
      value = panelFloats[sliceZ * ny * nx + pixelRow * nx + pixelCol];
    } else {
      const denom = Math.max(1, sliceW - 1);
      const t = pixelCol / denom;
      const x = obliqueSegment.start.x + (obliqueSegment.stop.x - obliqueSegment.start.x) * t;
      const y = obliqueSegment.start.y + (obliqueSegment.stop.y - obliqueSegment.start.y) * t;
      value = sampleVolumeBilinear(panelFloats, nx, ny, nz, pixelRow, x, y);
    }
    setCursorInfoThrottled({
      row: pixelRow,
      col: pixelCol,
      value: Number.isFinite(value) ? value : Number.NaN,
      view: PANEL_NAMES[axis] ?? "Slice",
    });
  };

  React.useEffect(() => {
    const handleDocumentMove = (event: MouseEvent) => {
      const drag = obliqueHandleDragRef.current;
      if (!drag) return;
      const point = imagePointFromEvent(event, 0);
      if (!point) return;
      if (drag.mode === "endpoint") {
        updateObliqueFromEndpoints(
          clampPointToImage({ x: point.col, y: point.row }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
          drag.opposite,
        );
      } else {
        updateObliqueFromLineDrag(drag, point);
      }
    };
    const handleDocumentUp = () => {
      obliqueHandleDragRef.current = null;
      setObliqueHoverTarget(null);
      endObliquePositionDrag();
    };
    document.addEventListener("mousemove", handleDocumentMove);
    document.addEventListener("mouseup", handleDocumentUp);
    return () => {
      document.removeEventListener("mousemove", handleDocumentMove);
      document.removeEventListener("mouseup", handleDocumentUp);
    };
  });

  // Stationary click on a slice panel = jump-to-voxel: the image pixel under
  // the click sets the matching volume indices.
  const handleMouseUp = (event?: React.MouseEvent, axis?: number) => {
    if (zoomRafRef.current) { cancelAnimationFrame(zoomRafRef.current); zoomRafRef.current = 0; }
    commitLiveZoomsNow();
    const wasDraggingObliqueHandle = obliqueHandleDragRef.current !== null;
    obliqueHandleDragRef.current = null;
    setObliqueHoverTarget(null);
    endObliquePositionDrag();
    const click = clickStartRef.current;
    if (!wasDraggingObliqueHandle && event && axis !== undefined && click && click.axis === axis) {
      const moved = Math.abs(event.clientX - click.x) + Math.abs(event.clientY - click.y);
      if (moved < 4) {
        const point = imagePointFromEvent(event, axis);
        if (point) {
          const pixelCol = Math.floor(point.col), pixelRow = Math.floor(point.row);
          const [sliceH, sliceW] = sliceDims[axis];
          if (pixelCol >= 0 && pixelCol < sliceW && pixelRow >= 0 && pixelRow < sliceH) {
            if (clickJumpTimerRef.current !== null) {
              window.clearTimeout(clickJumpTimerRef.current);
            }
            clickJumpTimerRef.current = window.setTimeout(() => {
              if (axis === 0) {
                setSliceY(pixelRow);
                setSliceX(pixelCol);
              } else {
                const denom = Math.max(1, sliceW - 1);
                const t = pixelCol / denom;
                const x = obliqueSegment.start.x + (obliqueSegment.stop.x - obliqueSegment.start.x) * t;
                const y = obliqueSegment.start.y + (obliqueSegment.stop.y - obliqueSegment.start.y) * t;
                const nextX = Math.round(x);
                const nextY = Math.round(y);
                if (nextX >= 0 && nextX < nx && nextY >= 0 && nextY < ny) {
                  setSliceZ(pixelRow);
                  setSliceX(nextX);
                  setSliceY(nextY);
                }
              }
              clickJumpTimerRef.current = null;
            }, 220);
          }
        }
      }
    }
    clickStartRef.current = null;
    setDragAxis(null); setDragStart(null);
  };
  // Don't kill the drag when the cursor briefly leaves the panel - users routinely
  // drag past the edge while panning. Only clear the cursor readout overlay.
  const handleMouseLeave = () => {
    setCursorInfoThrottled(null);
    if (!obliqueHandleDragRef.current) setObliqueHoverTarget(null);
  };

  // Global mouseup ensures drag ends even if the user releases the mouse outside
  // any slice or FFT canvas (e.g. they drag onto the volume panel and let go).
  // Without this the dragAxis state stays pinned and the next mouseMove on ANY
  // panel pans it - very confusing.
  React.useEffect(() => {
    if (dragAxis === null && fftDragAxis === null) return;
    const onUp = () => {
      if (zoomRafRef.current) { cancelAnimationFrame(zoomRafRef.current); zoomRafRef.current = 0; }
      if (fftZoomRafRef.current) { cancelAnimationFrame(fftZoomRafRef.current); fftZoomRafRef.current = 0; }
      commitLiveZoomsNow();
      commitLiveFftZoomsNow();
      setDragAxis(null); setDragStart(null);
      setFftDragAxis(null); setFftDragStart(null);
      fftClickStartRef.current = null;
    };
    document.addEventListener("mouseup", onUp);
    return () => document.removeEventListener("mouseup", onUp);
  }, [dragAxis, fftDragAxis]);

  const handleResetSlices = () => {
    const resetZooms = [DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM];
    const resetFftZooms = [DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM];
    liveZoomsRef.current = resetZooms;
    liveFftZoomsRef.current = resetFftZooms;
    liveZoomDirtyRef.current = false;
    liveFftZoomDirtyRef.current = false;
    setZooms(resetZooms);
    setFftZooms(resetFftZooms);
    persistViewState(resetZooms, resetFftZooms, undefined);
    setFftClickInfo(null);
  };

  // -------------------------------------------------------------------------
  // Keyboard shortcuts
  // -------------------------------------------------------------------------
  // Arrow Left/Right  : prev/next active transport axis (slice or plane)
  // Arrow Up/Down     : decrease/increase oblique angle
  // Home / End        : first / last on active transport axis
  // Space             : play/pause
  // r / R             : reset slice/FFT zoom and pan
  const handleKeyDown = (event: React.KeyboardEvent) => {
    // Keep native keyboard behavior for sliders/selects/buttons.
    if (shouldIgnoreWidgetShortcut(event.target)) return;
    const activeAxis = playbackAxis === 3 ? 0 : playbackAxis;
    const advanceTransportAxis = (axis: number, delta: number) => {
      event.preventDefault();
      if (axis === 0) {
        setSliceZ(Math.max(0, Math.min(nz - 1, sliceZ + delta)));
        return;
      }
      const state = obliquePlaybackStateRef.current;
      updateObliqueFromNormalOffset(clampNumber(state.current + delta, state.start, state.end));
    };
    const jumpTransportAxis = (axis: number, toEnd: boolean) => {
      event.preventDefault();
      if (axis === 0) {
        setSliceZ(toEnd ? nz - 1 : 0);
        return;
      }
      const state = obliquePlaybackStateRef.current;
      updateObliqueFromNormalOffset(toEnd ? state.end : state.start);
    };
    switch (event.key) {
      case " ":
        event.preventDefault();
        setPlaying(!playing);
        break;
      case "ArrowLeft":
        advanceTransportAxis(activeAxis, -1);
        break;
      case "ArrowRight":
        advanceTransportAxis(activeAxis, 1);
        break;
      case "ArrowUp":
        event.preventDefault();
        updateObliqueAngleWithinBounds(Math.round(obliqueAngle) - 1);
        break;
      case "ArrowDown":
        event.preventDefault();
        updateObliqueAngleWithinBounds(Math.round(obliqueAngle) + 1);
        break;
      case "Home":
        jumpTransportAxis(activeAxis, false);
        break;
      case "End":
        jumpTransportAxis(activeAxis, true);
        break;
      case "r":
      case "R":
        // Only handle 'r' when no modifier so we don't shadow Ctrl+R / Cmd+R reload.
        if (!event.ctrlKey && !event.metaKey && !event.altKey) {
          event.preventDefault();
          handleResetSlices();
        }
        break;
    }
  };

  // -------------------------------------------------------------------------
  // FFT Zoom/Pan handlers
  // -------------------------------------------------------------------------
  const handleFftWheel = (event: React.WheelEvent, axis: number) => {
    const canvas = fftCanvasRefs.current[axis];
    if (!canvas) return;
    // The canvas-level passive:false listener owns scroll suppression.
    const next = [...liveFftZoomsRef.current];
    next[axis] = wheelZoomAboutPointer(canvas, event, liveFftZoomsRef.current[axis]);
    liveFftZoomsRef.current = next;
    if (!fftZoomRafRef.current) {
      fftZoomRafRef.current = requestAnimationFrame(() => {
        fftZoomRafRef.current = 0;
        drawFftDirect(axis);
      });
    }
    commitLiveFftZoomsSoon();
  };

  const handleFftDoubleClick = (axis: number) => {
    const next = [...liveFftZoomsRef.current];
    next[axis] = DEFAULT_FFT_ZOOM;
    liveFftZoomsRef.current = next;
    commitLiveFftZoomsNow();
  };

  const handleFftMouseDown = (event: React.MouseEvent, axis: number) => {
    fftClickStartRef.current = { x: event.clientX, y: event.clientY, axis };
    const view = liveFftZoomsRef.current[axis];
    setFftDragAxis(axis);
    setFftDragStart({ x: event.clientX, y: event.clientY, pX: view.panX, pY: view.panY });
    liveFftZoomDirtyRef.current = true;
  };

  const handleFftMouseMove = (event: React.MouseEvent, axis: number) => {
    if (fftDragAxis !== axis || !fftDragStart) return;
    const canvas = fftCanvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const deltaX = (event.clientX - fftDragStart.x) * (canvas.width / rect.width);
    const deltaY = (event.clientY - fftDragStart.y) * (canvas.height / rect.height);
    const newZoom = { ...liveFftZoomsRef.current[axis], panX: fftDragStart.pX + deltaX, panY: fftDragStart.pY + deltaY };
    const next = [...liveFftZoomsRef.current]; next[axis] = newZoom;
    liveFftZoomsRef.current = next;
    if (!fftZoomRafRef.current) {
      fftZoomRafRef.current = requestAnimationFrame(() => {
        fftZoomRafRef.current = 0;
        drawFftDirect(axis);
      });
    }
  };

  const handleFftMouseUp = async (event: React.MouseEvent, axis: number) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current && fftClickStartRef.current.axis === axis) {
      const deltaX = event.clientX - fftClickStartRef.current.x;
      const deltaY = event.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(deltaX * deltaX + deltaY * deltaY) < 3) {
        const canvas = fftCanvasRefs.current[axis];
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const { w: canvasW, h: canvasH } = canvasSizes[axis];
          const view = liveFftZoomsRef.current[axis];

          const [sliceH, sliceW] = sliceDims[axis];
          const fftW = nextPow2(sliceW);
          const fftH = nextPow2(sliceH);

          const mouseX = (event.clientX - rect.left) * (canvasW / rect.width);
          const mouseY = (event.clientY - rect.top) * (canvasH / rect.height);
          const centerX = canvasW / 2, centerY = canvasH / 2;
          const imgX = (mouseX - centerX - view.panX) / view.zoom + centerX;
          const imgY = (mouseY - centerY - view.panY) / view.zoom + centerY;
          let imgCol = imgX / canvasW * fftW;
          let imgRow = imgY / canvasH * fftH;

          const cachedMag = fftMagCacheRefs.current[axis];
          if (cachedMag && imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
            let snapped: { row: number; col: number };
            try {
              snapped = await findFFTPeakBrowser(cachedMag, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
            } catch (error) {
              console.error("[Show3DSlices] WebGPU FFT peak refinement failed", error);
              return;
            }
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
              const axes = pixelSizeAxes && pixelSizeAxes.length === 3 ? pixelSizeAxes : null;
              const theta = (obliqueAngle * Math.PI) / 180;
              const obliqueSpacing = axes
                ? Math.hypot(Math.cos(theta) * axes[2], Math.sin(theta) * axes[1])
                : pixelSize;
              const rowSpacing = axis === 0 ? (axes ? axes[1] : pixelSize) : (axes ? axes[0] : pixelSize);
              const colSpacing = axis === 0 ? (axes ? axes[2] : pixelSize) : obliqueSpacing;
              if (rowSpacing > 0 && colSpacing > 0) {
                const paddedW = fftW;
                const paddedH = fftH;
                ({ spatialFrequency: spatialFreq, dSpacing } = reciprocalCoordinatesFromShiftedOffset(
                  dcRow,
                  dcCol,
                  paddedH,
                  paddedW,
                  rowSpacing,
                  colSpacing,
                ));
              }
              setFftClickInfo({ axis, row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
            }
          }
        }
      }
    }
    fftClickStartRef.current = null;
    if (fftZoomRafRef.current) { cancelAnimationFrame(fftZoomRafRef.current); fftZoomRafRef.current = 0; }
    commitLiveFftZoomsNow();
    setFftDragAxis(null);
    setFftDragStart(null);
  };

  const handleFftResetAxis = (axis: number) => {
    const next = [...liveFftZoomsRef.current];
    next[axis] = DEFAULT_FFT_ZOOM;
    liveFftZoomsRef.current = next;
    commitLiveFftZoomsNow();
    if (fftClickInfo && fftClickInfo.axis === axis) setFftClickInfo(null);
  };

  const fftNeedsResetAxis = (axis: number) => {
    const view = fftZooms[axis];
    return view.zoom !== DEFAULT_FFT_ZOOM.zoom || view.panX !== DEFAULT_FFT_ZOOM.panX || view.panY !== DEFAULT_FFT_ZOOM.panY;
  };

  // -------------------------------------------------------------------------
  // Canvas resize (matching Show2D)
  // -------------------------------------------------------------------------
  const handleResizeStart = (event: React.MouseEvent, axis: number = 0) => {
    event.stopPropagation();
    event.preventDefault();
    const target = axis > 0 ? "side" : "primary";
    setIsResizing(true);
    setResizeStart({ x: event.clientX, y: event.clientY, size: target === "side" ? sideCanvasTarget : canvasTarget, target });
  };

  React.useEffect(() => {
    if (!isResizing || !resizeStart) return;
    let rafId = 0;
    let latestSize = resizeStart.size;
    const handleMouseMove = (event: MouseEvent) => {
      const delta = Math.max(event.clientX - resizeStart.x, event.clientY - resizeStart.y);
      latestSize = clampCanvasTarget(resizeStart.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          if (resizeStart.target === "side") setSideCanvasTarget(latestSize);
          else setCanvasTarget(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
      const nextSize = clampCanvasTarget(latestSize);
      if (resizeStart?.target === "side") {
        setSideCanvasTarget(nextSize);
        persistViewState(undefined, undefined, undefined, { sideCanvasTarget: nextSize });
      } else {
        setCanvasTarget(nextSize);
        persistViewState(undefined, undefined, undefined, { canvasTarget: nextSize });
      }
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
  }, [isResizing, persistViewState, resizeStart]);

  // -------------------------------------------------------------------------
  // Labels and setters
  // -------------------------------------------------------------------------
  // Default mirrors Python's dim_labels default ["slice", "row", "col"]: axis
  // 0 is the slice (multislice depth), axis 1 is row, axis 2 is col. Fallback
  // fires only when the trait is briefly undefined (initial mount race).
  const axisLabels = dimLabels || ["slice", "row", "col"];
  const sliceValues = [sliceZ, sliceY, sliceX];
  // Mirror of slice values for the playback loop to read between renders.
  // The loop's `sliceValuesRef.current[0] = value` writes are load-bearing
  // at high fps (>~20): React batches setSliceZ/Y/X so two ticks can fire
  // before the next render reassigns this ref to the new [sliceZ,sliceY,sliceX].
  // Without the mutation the second tick reads the stale value and computes the
  // same `next`, freezing playback.
  const sliceValuesRef = React.useRef(sliceValues);
  if (!playing) sliceValuesRef.current = sliceValues;
  const sliceMaxes = [nz - 1, ny - 1, nx - 1];
  // Live thumb mirror: updates per drag-frame so the thumb tracks, WITHOUT touching
  // the model traits (whose change re-runs the heavy render/layout/crosshair effects).
  // Those effects key on sliceX/Y/Z, so during a drag (traits unchanged) they don't
  // run - only the slider JSX re-renders + directPaintPlane paints the GPU image.
  const [liveSlider, setLiveSlider] = React.useState<number[]>([sliceZ, sliceY, sliceX]);
  const liveSliderRef = React.useRef<number[]>([sliceZ, sliceY, sliceX]);
  const pendingPaintRef = React.useRef<Map<number, number>>(new Map());
  const sliderPaintRafRef = React.useRef<number | null>(null);
  const pendingContrastRangeRef = React.useRef<[number, number] | null>(null);
  const contrastPaintRafRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    const next = [sliceZ, sliceY, sliceX];
    liveSliderRef.current = next;
    setLiveSlider(next);
  }, [sliceZ, sliceY, sliceX]);
  React.useEffect(() => {
    return () => {
      if (sliderPaintRafRef.current != null) cancelAnimationFrame(sliderPaintRafRef.current);
      if (contrastPaintRafRef.current != null) cancelAnimationFrame(contrastPaintRafRef.current);
    };
  }, []);
  // Without the WebGPU engine the drag paints the top plane on the CPU, with the
  // steps of the committed paint (extract, log, flip, colormap, zoomed blit), so
  // the slice follows the thumb instead of appearing on release. Its own
  // offscreen keeps the committed one a cache of the committed slice.
  const previewOffscreenRef = React.useRef<{ canvas: HTMLCanvasElement; image: ImageData } | null>(null);
  const previewTopPlane = (sliceIndex: number) => {
    const canvas = canvasRefs.current[0];
    const ctx = canvas?.getContext("2d");
    const paint = paintParamsRef.current;
    if (!ctx || !paint || !panelFloats) return;
    const [sliceH, sliceW] = sliceDims[0];
    let preview = previewOffscreenRef.current;
    if (!preview || preview.canvas.width !== sliceW || preview.canvas.height !== sliceH) {
      const offscreen = document.createElement("canvas");
      offscreen.width = sliceW; offscreen.height = sliceH;
      preview = { canvas: offscreen, image: new ImageData(sliceW, sliceH) };
      previewOffscreenRef.current = preview;
    }
    const slice = extractXY(panelFloats, nx, ny, nz, sliceIndex);
    const { vmin, vmax } = volumeDisplayWindow(paint.imageDataRange, paint.imageVminPct, paint.imageVmaxPct, paint.flip);
    renderToOffscreenReuse(maybeFlip(paint.logScale ? applyLogScale(slice) : slice, paint.flip), COLORMAPS[paint.cmap] || COLORMAPS.inferno, vmin, vmax, preview.canvas, preview.image);
    const { w: canvasW, h: canvasH } = paint.canvasSizes[0];
    drawZoomedOffscreen(ctx, preview.canvas, canvasW, canvasH, paint.zooms[0], paint.smooth);
  };
  // DURING DRAG: only direct-paint (GPU, off React) - do NOT set the model trait,
  // which would re-render the whole component per drag-frame (the 39->stuck cap).
  // ON RELEASE (onChangeCommitted): set the trait once so crosshair/title/state sync.
  const paintAndTrackRef = React.useRef<((axis: number, value: number) => void) | null>(null);
  paintAndTrackRef.current = (axis: number, value: number) => {
    if (liveSliderRef.current[axis] === value) return;
    const next = [...liveSliderRef.current];
    next[axis] = value;
    liveSliderRef.current = next;
    sliceValuesRef.current = next;
    liveSliceParamsRef.current = { sliceZ: next[0], sliceY: next[1], sliceX: next[2] };
    volumeRenderParamsRef.current = { ...volumeRenderParamsRef.current, ...liveSliceParamsRef.current };
    if (axis === 0) {
      scheduleLiveFft([0], { sliceZ: next[0] });
    } else if (!obliqueSegment.explicit) {
      const [rawStart, rawStop] = obliqueLineEndpoints(nx, ny, next[2], next[1], obliqueAngle);
      const start = clampPointToImage(rawStart, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
      const stop = clampPointToImage(rawStop, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
      scheduleLiveFft([1], { segment: { start, stop } });
    }
    pendingPaintRef.current.set(axis, value);
    if (sliderPaintRafRef.current != null) return;
    sliderPaintRafRef.current = requestAnimationFrame(() => {
      sliderPaintRafRef.current = null;
      const pending = pendingPaintRef.current;
      pendingPaintRef.current = new Map();
      for (const [pendingAxis, pendingValue] of pending) {
        if (!directPaintPlane(pendingAxis, pendingValue) && pendingAxis === 0) previewTopPlane(pendingValue);
      }
      renderVolumePlanesLive();
      setLiveSlider(liveSliderRef.current);
    });
  };
  fastTrackSliceRef.current = (axis: number, value: number) => {
    paintAndTrackRef.current?.(axis, value);
  };
  const paintContrastRange = (min: number, max: number) => {
    pendingContrastRangeRef.current = [min, max];
    const paint = paintParamsRef.current;
    if (paint) paintParamsRef.current = { ...paint, imageVminPct: min, imageVmaxPct: max };
    const nextVolRange = volumeTextureRangeForPercent(min, max);
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      vmin: nextVolRange.vmin,
      vmax: nextVolRange.vmax,
    };
    if (contrastPaintRafRef.current != null) return;
    contrastPaintRafRef.current = requestAnimationFrame(() => {
      contrastPaintRafRef.current = null;
      const pending = pendingContrastRangeRef.current;
      pendingContrastRangeRef.current = null;
      if (!pending) return;
      const [pendingMin, pendingMax] = pending;
      const current = paintParamsRef.current;
      if (current) paintParamsRef.current = { ...current, imageVminPct: pendingMin, imageVmaxPct: pendingMax };
      const pendingVolRange = volumeTextureRangeForPercent(pendingMin, pendingMax);
      volumeRenderParamsRef.current = {
        ...volumeRenderParamsRef.current,
        vmin: pendingVolRange.vmin,
        vmax: pendingVolRange.vmax,
      };
      const slices = liveSliderRef.current;
      for (let axis = 0; axis < sliceDims.length; axis++) directPaintPlane(axis, slices[axis]);
      renderVolumePlanesLive();
    });
  };
  commitSliceValuesRef.current = () => {
    const [liveSliceZ, liveSliceY, liveSliceX] = liveSliderRef.current;
    if (sliceZ !== liveSliceZ) setSliceZ(liveSliceZ);
    if (sliceY !== liveSliceY) setSliceY(liveSliceY);
    if (sliceX !== liveSliceX) setSliceX(liveSliceX);
  };
  const stopPlaybackAndRewind = () => {
    setPlaying(false);
    const axes = playbackAxis === 3 ? [0, 1] : [playbackAxis];
    const next = [...sliceValuesRef.current];
    for (const axis of axes) {
      if (axis === 0) {
        const start = Math.max(0, Math.min(loopStarts[0], sliceMaxes[0]));
        next[0] = start;
        paintAndTrackRef.current?.(0, start);
        sliceSettersRef.current[0](start);
      } else {
        const start = obliquePlaybackStateRef.current.start;
        obliquePlaybackStateRef.current = { ...obliquePlaybackStateRef.current, current: start };
        updateObliqueFromNormalOffset(start);
      }
    }
    sliceValuesRef.current = next;
  };
  const sliceSetters = [
    (_: Event, value: number | number[]) => paintAndTrackRef.current!(0, value as number),
    (_: Event, value: number | number[]) => paintAndTrackRef.current!(1, value as number),
    (_: Event, value: number | number[]) => paintAndTrackRef.current!(2, value as number),
  ];
  const sliceCommitters = [
    (_: unknown, value: number | number[]) => setSliceZ(value as number),
    (_: unknown, value: number | number[]) => setSliceY(value as number),
    (_: unknown, value: number | number[]) => setSliceX(value as number),
  ];
  const loopSliderValues = (axis: number) => {
    return [loopStarts[axis], liveSlider[axis], effectiveLoopEnds[axis]];
  };
  const handleLoopSliderChange = (axis: number, values: number[]) => {
    paintAndTrackRef.current?.(axis, values[1]);
    if (values[0] === loopStartsRef.current[axis] && values[2] === loopEndsRef.current[axis]) return;
    const nextStarts = [...loopStartsRef.current];
    const nextEnds = [...loopEndsRef.current];
    nextStarts[axis] = values[0];
    nextEnds[axis] = values[2];
    loopStartsRef.current = nextStarts;
    loopEndsRef.current = nextEnds;
    pendingLoopRangeRef.current = { starts: nextStarts, ends: nextEnds };
    if (loopRangeRafRef.current == null) {
      loopRangeRafRef.current = requestAnimationFrame(() => {
        loopRangeRafRef.current = null;
        const pending = pendingLoopRangeRef.current;
        pendingLoopRangeRef.current = null;
        if (!pending) return;
        setLoopStarts(pending.starts);
        setLoopEnds(pending.ends);
      });
    }
  };
  const handleLoopSliderCommit = (axis: number, values: number[]) => {
    if (loopRangeRafRef.current != null) {
      cancelAnimationFrame(loopRangeRafRef.current);
      loopRangeRafRef.current = null;
    }
    const startsChanged = values[0] !== loopStartsRef.current[axis];
    const endsChanged = values[2] !== loopEndsRef.current[axis];
    pendingLoopRangeRef.current = null;
    if (startsChanged || endsChanged) {
      const nextStarts = [...loopStartsRef.current];
      const nextEnds = [...loopEndsRef.current];
      nextStarts[axis] = values[0];
      nextEnds[axis] = values[2];
      loopStartsRef.current = nextStarts;
      loopEndsRef.current = nextEnds;
      setLoopStarts(nextStarts);
      setLoopEnds(nextEnds);
    }
    [setSliceZ, setSliceY, setSliceX][axis](values[1]);
  };
  const handleLoopSliderPointerDownCapture = (axis: number, event: React.PointerEvent<HTMLSpanElement>) => {
    if (event.button !== 0) return;
    const target = event.target as HTMLElement;
    if (target.closest(".MuiSlider-thumb")) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const max = sliceMaxes[axis];
    const valueFromClientX = (clientX: number) => {
      const pct = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
      return Math.max(0, Math.min(max, Math.round(pct * max)));
    };
    const moveCurrent = (clientX: number, commit: boolean) => {
      const next = valueFromClientX(clientX);
      paintAndTrackRef.current?.(axis, next);
      if (commit) [setSliceZ, setSliceY, setSliceX][axis](next);
    };
    event.preventDefault();
    event.stopPropagation();
    event.nativeEvent.stopImmediatePropagation();
    moveCurrent(event.clientX, false);
    const onMove = (ev: PointerEvent) => {
      ev.preventDefault();
      moveCurrent(ev.clientX, false);
    };
    const onUp = (ev: PointerEvent) => {
      ev.preventDefault();
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      moveCurrent(ev.clientX, true);
    };
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
  };
  // Over-clip detection: user dragged hist thumbs past data peak → image goes black.
  // Compute effective vmin/vmax in data units, compare against 1st/99th percentile of histogram.
  // If vmin > 99% of data OR vmax < 1% of data, no visible content.
  const imageClipBounds = React.useMemo(() => {
    if (!imageHistogramData || imageHistogramData.length === 0) return null;
    return percentileClip(imageHistogramData, 1, 99);
  }, [imageHistogramData]);
  const isOverClipped = (() => {
    if (autoActive) return false;
    if (imageVminPct <= 0 && imageVmaxPct >= 100) return false;
    if (!imageClipBounds) return false;
    const span = displayDataRange.max - displayDataRange.min;
    if (span <= 0) return false;
    const vmin = displayDataRange.min + (imageVminPct / 100) * span;
    const vmax = displayDataRange.min + (imageVmaxPct / 100) * span;
    return vmin >= imageClipBounds.vmax || vmax <= imageClipBounds.vmin;
  })();

  // Thin-Z layout: depth axis much smaller than lateral. Show the top panel
  // beside the single oblique depth panel.
  const panelTotalW = (canvasSizes[0]?.w ?? CANVAS_TARGET) + (canvasSizes[1]?.w ?? 0) + SPACING.SM;
  const sliceColumnOffsetPx = (webgpuSupported ? volumeCanvasSize : 220) + SPACING.SM;
  const sideBySideMinWidth = sliceColumnOffsetPx + panelTotalW;
  const obliqueAngleSliderMin = 0;
  const obliqueAngleSliderMax = 179;
  const [rawObliqueAngleMinBound, rawObliqueAngleMaxBound] = obliqueAngleBounds ?? [
    obliqueAngleSliderMin,
    obliqueAngleSliderMax,
  ];
  const obliqueAngleMinBound = clampNumber(rawObliqueAngleMinBound, obliqueAngleSliderMin, obliqueAngleSliderMax);
  const obliqueAngleMaxBound = clampNumber(rawObliqueAngleMaxBound, obliqueAngleMinBound, obliqueAngleSliderMax);
  const boundedObliqueAngle = clampNumber(obliqueAngle, obliqueAngleMinBound, obliqueAngleMaxBound);
  const obliqueAngleSliderValues = [
    obliqueAngleMinBound,
    boundedObliqueAngle,
    obliqueAngleMaxBound,
  ];
  const updateObliqueAngleWithinBounds = (angle: number, minBound = obliqueAngleMinBound, maxBound = obliqueAngleMaxBound) => {
    const nextAngle = clampNumber(angle, minBound, maxBound);
    if (Math.round(nextAngle) !== Math.round(obliqueAngle)) handleObliqueAngleChange(new Event("change"), nextAngle);
  };
  const handleObliqueAngleSliderChange = (values: number[]) => {
    setObliqueAngleBounds([values[0], values[2]]);
    updateObliqueAngleWithinBounds(values[1], values[0], values[2]);
  };
  const handleObliqueAnglePointerDownCapture = (event: React.PointerEvent<HTMLSpanElement>) => {
    if (event.button !== 0) return;
    pausePlaybackForEdit();
    const target = event.target as HTMLElement;
    if (target.closest(".MuiSlider-thumb")) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const valueFromClientX = (clientX: number) => {
      const pct = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
      const full = obliqueAngleSliderMin + pct * (obliqueAngleSliderMax - obliqueAngleSliderMin);
      return Math.round(clampNumber(full, obliqueAngleMinBound, obliqueAngleMaxBound));
    };
    const moveCurrent = (clientX: number) => updateObliqueAngleWithinBounds(valueFromClientX(clientX));
    event.preventDefault();
    event.stopPropagation();
    event.nativeEvent.stopImmediatePropagation();
    moveCurrent(event.clientX);
    const onMove = (ev: PointerEvent) => {
      ev.preventDefault();
      moveCurrent(ev.clientX);
    };
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      moveCurrent(ev.clientX);
    };
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
  };
  const obliqueCurrentOffset = obliqueCenterOffset(nx, ny, obliqueAngle, obliqueSegment.start, obliqueSegment.stop);
  // Plane center in image coordinates. Position measures along the plane normal,
  // which turns with Angle, so its number moves on rotation even when the cut
  // does not; the center is stated in fixed image pixels instead.
  const obliqueCenterCol = (obliqueSegment.start.x + obliqueSegment.stop.x) / 2;
  const obliqueCenterRow = (obliqueSegment.start.y + obliqueSegment.stop.y) / 2;
  const [obliqueDeltaMin, obliqueDeltaMax] = obliqueSegmentOffsetBounds(
    nx,
    ny,
    obliqueAngle,
    obliqueSegment.start,
    obliqueSegment.stop,
    OBLIQUE_PROFILE_EDGE_INSET,
  );
  const obliqueOffsetMin = Math.ceil(obliqueCurrentOffset + obliqueDeltaMin);
  const obliqueOffsetMax = Math.floor(obliqueCurrentOffset + obliqueDeltaMax);
  const obliqueOffset = liveObliqueOffset ?? clampNumber(Math.round(obliqueCurrentOffset), obliqueOffsetMin, obliqueOffsetMax);
  const beginObliquePositionDrag = () => {
    pausePlaybackForEdit();
    const currentOffset = obliqueCenterOffset(nx, ny, obliqueAngle, obliqueSegment.start, obliqueSegment.stop);
    const [minDelta, maxDelta] = obliqueSegmentOffsetBounds(nx, ny, obliqueAngle, obliqueSegment.start, obliqueSegment.stop, OBLIQUE_PROFILE_EDGE_INSET);
    obliquePositionDragRef.current = {
      angleDeg: obliqueAngle,
      currentOffset,
      minOffset: Math.ceil(currentOffset + minDelta),
      maxOffset: Math.floor(currentOffset + maxDelta),
      start: { ...obliqueSegment.start },
      stop: { ...obliqueSegment.stop },
    };
    setLiveObliqueOffset(Math.round(currentOffset));
  };
  const endObliquePositionDrag = () => {
    obliquePositionDragRef.current = null;
    setLiveObliqueOffset(null);
  };
  const obliquePositionSliderMin = obliquePositionDragRef.current?.minOffset ?? obliqueOffsetMin;
  const obliquePositionSliderMax = obliquePositionDragRef.current?.maxOffset ?? obliqueOffsetMax;
  const [rawObliquePositionMinBound, rawObliquePositionMaxBound] = obliquePositionBounds ?? [
    obliquePositionSliderMin,
    obliquePositionSliderMax,
  ];
  const obliquePositionMinBound = clampNumber(
    rawObliquePositionMinBound,
    obliquePositionSliderMin,
    obliquePositionSliderMax,
  );
  const obliquePositionMaxBound = clampNumber(
    rawObliquePositionMaxBound,
    obliquePositionMinBound,
    obliquePositionSliderMax,
  );
  const boundedObliqueOffset = clampNumber(obliqueOffset, obliquePositionMinBound, obliquePositionMaxBound);
  const obliquePositionSliderValues = [
    obliquePositionMinBound,
    boundedObliqueOffset,
    obliquePositionMaxBound,
  ];
  obliquePlaybackStateRef.current = {
    current: boundedObliqueOffset,
    start: obliquePositionMinBound,
    end: obliquePositionMaxBound,
  };
  const slicePanelCursor = (axis: number) => {
    if (axis === 0) {
      if (obliqueHandleDragRef.current || dragAxis === axis) return "grabbing";
      if (obliqueHoverTarget === "endpoint" || obliqueHoverTarget === "line") return "grab";
    }
    return dragAxis === axis ? "grabbing" : "crosshair";
  };
  const handleObliquePositionChange = (values: number[]) => {
    setObliquePositionBounds([values[0], values[2]]);
    updateObliqueFromNormalOffset(clampNumber(values[1], values[0], values[2]));
  };
  const handleObliquePositionCommit = (values: number[]) => {
    setObliquePositionBounds([values[0], values[2]]);
    updateObliqueFromNormalOffset(clampNumber(values[1], values[0], values[2]));
    endObliquePositionDrag();
  };
  const handleObliquePositionPointerDownCapture = (event: React.PointerEvent<HTMLSpanElement>) => {
    beginObliquePositionDrag();
    if (event.button !== 0) return;
    const target = event.target as HTMLElement;
    if (target.closest(".MuiSlider-thumb")) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const valueFromClientX = (clientX: number) => {
      const pct = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
      const full = obliquePositionSliderMin + pct * (obliquePositionSliderMax - obliquePositionSliderMin);
      return Math.round(clampNumber(full, obliquePositionMinBound, obliquePositionMaxBound));
    };
    const moveCurrent = (clientX: number, commit: boolean) => {
      updateObliqueFromNormalOffset(valueFromClientX(clientX));
      if (commit) endObliquePositionDrag();
    };
    event.preventDefault();
    event.stopPropagation();
    event.nativeEvent.stopImmediatePropagation();
    moveCurrent(event.clientX, false);
    const onMove = (ev: PointerEvent) => {
      ev.preventDefault();
      moveCurrent(ev.clientX, false);
    };
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      moveCurrent(ev.clientX, true);
    };
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
  };
  const controlRowHeight = 28;
  const denseControlRow = {
    ...controlRow,
    minHeight: controlRowHeight,
    py: 0.25,
    boxSizing: "border-box" as const,
  };
  const panelControlRow = {
    ...denseControlRow,
    border: `1px solid ${themeColors.border}`,
    bgcolor: themeColors.controlBg,
    boxSizing: "border-box" as const,
  };
  const inlineVolumeControlRow = {
    ...denseControlRow,
    px: 0,
    py: 0,
    minHeight: 22,
    width: "fit-content",
    maxWidth: "none",
    flexWrap: "nowrap" as const,
    alignSelf: "flex-start",
  };
  const denseSelect = {
    ...themedSelect,
    height: 22,
    fontSize: 10,
    "& .MuiSelect-select": { py: 0.25, px: 1 },
  };
  // Loop range, oblique angle and position sliders: small outer bounds thumbs
  // around the current-value thumb.
  const rangeSliderSx = {
    ...sliderStyles.small,
    flex: 1,
    minWidth: 40,
    "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: themeColors.textMuted },
    "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 },
    "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: themeColors.textMuted },
    "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
  };
  const resizeHandleSx = {
    position: "absolute", bottom: 0, right: 0, width: 16, height: 16,
    cursor: "nwse-resize", opacity: 0.6,
    background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
    touchAction: "none",
    zIndex: 5,
    "&:hover": { opacity: 1 },
  };
  const contentControlRow = {
    ...panelControlRow,
    width: "fit-content",
    maxWidth: "none",
    flexWrap: "wrap" as const,
    alignSelf: "flex-start",
  };
  const alignmentShiftLimit = Math.max(
    2,
    Math.ceil(Math.max(nx, ny) / Math.max(1, nz - 1)),
  );
  const alignmentStatusText = localAlignmentStatus || (
    !sliceAlignmentCached && alignmentActive ? "Estimating..." : ""
  );
  const topRightActions = (
    <Box sx={{
      ...controlRow,
      mb: 0,
      py: 0,
      minHeight: 24,
      boxSizing: "border-box" as const,
      width: "fit-content",
      maxWidth: "100%",
      flexWrap: "wrap" as const,
    }}>
      {isPaged && (
        <>
          <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, minWidth: 0 }}>
            <Typography sx={{ ...controlLabel }}>Page</Typography>
            <Typography
              title={pageStatus}
              sx={{
                ...controlLabel,
                color: themeColors.accent,
                flex: "0 1 16ch",
                minWidth: "8ch",
                maxWidth: "18ch",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {pageStatus}
            </Typography>
            <Slider
              value={safePageIdx}
              min={0}
              max={safeNPages - 1}
              step={1}
              onChange={(_, value) => showPage(Array.isArray(value) ? value[0] : value)}
              size="small"
              sx={{ ...sliderStyles.small, width: 96, flex: "0 0 96px", color: themeColors.accent }}
              aria-label="Show3DSlices page"
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => effectivePageLabels[value] || `Page ${value + 1}`}
            />
          </Box>
          <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px` }}>
            <IconButton
              size="small"
              onClick={() => setPagePlaying((value) => !value)}
              title={pagePlaying ? "Pause page playback" : "Play pages"}
              aria-label={pagePlaying ? "Pause page playback" : "Play pages"}
              sx={{ width: 24, height: 24, p: 0, color: themeColors.accent }}
            >
              {pagePlaying ? <PauseIcon sx={{ fontSize: 16 }} /> : <PlayArrowIcon sx={{ fontSize: 16 }} />}
            </IconButton>
            <Select
              value={String(pagePlayFps)}
              onChange={(event) => setPagePlayFps(Number(event.target.value) || 2)}
              size="small"
              sx={{ ...denseSelect, minWidth: 48 }}
              MenuProps={themedMenuProps}
              inputProps={{ "aria-label": "Show3DSlices page playback frames per second" }}
              title="Page playback speed"
            >
              {PAGE_PLAY_FPS_OPTIONS.map((value) => (
                <MenuItem key={value} value={String(value)}>{value} fps</MenuItem>
              ))}
            </Select>
          </Box>
        </>
      )}
      {pagePanelIndices.length > 1 && (
        <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, minWidth: 0 }}>
          <Typography sx={{ ...controlLabel }}>Panel</Typography>
          <Select
            value={safeActivePanel}
            onChange={(event) => setActivePanel(Number(event.target.value))}
            size="small"
            sx={{ ...denseSelect, minWidth: 150, maxWidth: 220 }}
            MenuProps={themedMenuProps}
            inputProps={{ "aria-label": "Select Show3DSlices panel" }}
          >
            {pagePanelIndices.map((i) => (
              <MenuItem key={i} value={i}>
                {effectivePanelTitles[i] || `Panel ${isPaged ? i - pageStart + 1 : i + 1}`}
              </MenuItem>
            ))}
          </Select>
        </Box>
      )}
      {showControls && (
        <Button
          size="small"
          sx={compactButton}
          onClick={() => setControlsCollapsed(!controlsCollapsed)}
          aria-label={controlsCollapsed ? "Show controls" : "Hide controls"}
        >
          {controlsCollapsed ? "Controls" : "Hide"}
        </Button>
      )}
      {controlsVisible && (
        <>
      <Typography sx={{ ...controlLabel }}>FFT</Typography>
      <Switch checked={showFft} onChange={(event) => setShowFft(event.target.checked)} size="small" sx={switchStyles.small} slotProps={switchInputSlot("Toggle FFT power spectrum panels")} />
      {exportEnabled && (
        <>
        <Button
          size="small"
          sx={compactButton}
          disabled={exportBusy}
          onClick={handleExportMenuOpen}
          aria-label="Export standalone HTML"
          aria-controls={exportMenuAnchor ? "show3dslices-export-menu" : undefined}
          aria-expanded={exportMenuAnchor ? "true" : undefined}
          aria-haspopup="menu"
          title={localExportStatus || exportStatus || "Export standalone HTML with a save dialog"}
        >
          {exportBusy ? "Exporting" : "Export"}
        </Button>
        <Menu
          id="show3dslices-export-menu"
          anchorEl={exportMenuAnchor}
          open={Boolean(exportMenuAnchor)}
          onClose={handleExportMenuClose}
          MenuListProps={{ "aria-label": "Export standalone HTML options" }}
          {...themedMenuProps}
        >
          <MenuItem onClick={() => handleExportSelect("exact")}>Exact float32 ({exactExportSize})</MenuItem>
          <MenuItem onClick={() => handleExportSelect("quantized")}>Quantized uint8 ({quantizedExportSize})</MenuItem>
        </Menu>
        </>
      )}
      {exportEnabled && (localExportStatus || exportStatus) && (
        <Typography
          sx={{
            ...controlLabel,
            maxWidth: 120,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            color: (localExportStatus || exportStatus).startsWith("Export failed") ? "#d32f2f" : themeColors.textMuted,
          }}
          title={localExportStatus || exportStatus}
        >
          {localExportStatus || exportStatus}
        </Typography>
      )}
      <Button
        size="small"
        sx={compactButton}
        disabled={!anyZoomDirty}
        onClick={handleResetSlices}
        title="Reset slice and FFT zoom/pan only"
        aria-label="Reset slice and FFT zoom/pan"
      >
        Reset Zoom
      </Button>
        </>
      )}
    </Box>
  );

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <Box className="show3dslices-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ ...container.root, position: "relative", bgcolor: themeColors.bg, color: themeColors.text, outline: "none", "&:focus::after": { content: '""', position: "absolute", inset: 0, pointerEvents: "none", zIndex: 20, boxShadow: "inset 0 0 0 2px #0af" }, "& canvas": { display: "block" } }}>
      {/* 3D volume on the LEFT, slice toolbar + projected slice panels on the RIGHT.
          Side-by-side layout keeps the whole widget within a 13" laptop viewport. */}
      <Box sx={{ display: "flex", flexDirection: { xs: "column", md: "row" }, flexWrap: "wrap", alignItems: "flex-start", gap: `${SPACING.SM}px`, width: "100%" }}>
      {/* 3D Volume Renderer (left column) */}
      <Box sx={{ mb: 0, flexShrink: 0, width: { xs: "100%", md: webgpuSupported ? volumeCanvasSize : 220 }, maxWidth: "100%", overflow: "visible" }}>
        {showTitle && <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block", minHeight: 16, maxWidth: webgpuSupported ? volumeCanvasSize : 220, lineHeight: "16px", whiteSpace: "normal", overflowWrap: "anywhere" }}>
          {displayTitle}<RenderPathBadge colors={themeColors} /><InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
            <MetadataSection rows={[
              ...(isPaged ? [
                ["Pages", String(safeNPages)] as [string, React.ReactNode],
                ["Active page", `${safePageIdx + 1}: ${activePageLabel}`] as [string, React.ReactNode],
              ] : []),
              ...(pagePanelIndices.length > 1 ? [
                ["Panels per page", String(pagePanelIndices.length)] as [string, React.ReactNode],
                ["Active panel", `${activePanelSlot + 1}: ${activePanelTitle}`] as [string, React.ReactNode],
              ] : []),
              ["Shape", `${nz} x ${ny} x ${nx}`] as [string, React.ReactNode],
              ["Axes", Array.isArray(dimLabels) && dimLabels.length ? dimLabels.join(", ") : "slice, row, col"] as [string, React.ReactNode],
              ["Sampling", Array.isArray(pixelSizeAxes) && pixelSizeAxes.length >= 3
                ? pixelSizeAxes.map((spacing) => formatNumber(spacing)).join(" x ")
                : pixelSize > 0 ? `${formatNumber(pixelSize)} /px` : ""] as [string, React.ReactNode],
              ["Display", `z stretch ${formatNumber(zStretch)}, ${orthographic ? "orthographic" : "perspective"}`] as [string, React.ReactNode],
            ]} />
            <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
            {isPaged && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Page switches comparable volumes while preserving the same top slice, side cut, zoom, and display settings.</Typography>}
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT shows the power spectrum below each slice.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto uses percentile-based contrast (2nd-98th percentile). FFT Auto masks DC + clips to 99.9th.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Colorbar displays a colorbar overlay on each slice canvas.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop repeats playback. Drag end markers on slider for loop range.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Bounce alternates forward and reverse playback.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Planes toggles the Top and angled vertical slice planes, in the 3D volume view and as 2D panels.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Align corrects a global tilt through depth: it fits one straight row/col drift per slice by registering adjacent slices, then shifts deeper slices back by that slope. Use it when a reconstruction leans through the stack so vertical columns look sheared in the side view. It is display-only - the stored volume is never modified - and Row/Col let you override the fitted slope by hand.</Typography>
            <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
            <KeyboardShortcuts items={[["Space", "Play / Pause"], ["← / →", "Active axis -/+"], ["↑ / ↓", "Angle -/+"], ["Home / End", "First / Last on active axis"], ["R", "Reset zoom"], ["Click panel", "Jump to voxel"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
          </Box>} theme={themeInfo.theme} />
        </Typography>}
        {webgpuSupported ? (
          <Stack direction="row" spacing={`${SPACING.SM}px`}>
            <Box>
              {controlsVisible && (
              <Box sx={{ ...inlineVolumeControlRow, mb: `${SPACING.XS}px` }}>
                <Typography sx={{ ...controlLabel }}>Planes</Typography>
                <ToggleButtonGroup
                  size="small"
                  value={visiblePlanes}
                  onChange={handlePlaneVisibilityChange}
                  aria-label="Slice plane visibility"
                  sx={{ height: 18, "& .MuiToggleButtonGroup-grouped": { m: 0 } }}
                >
                  {PLANE_KEYS.map((key, i) => (
                    <ToggleButton
                      key={key}
                      value={key}
                      aria-label={`${PLANE_LABELS[i]} plane`}
                      sx={planeToggleButtonSx}
                    >
                      {PLANE_LABELS[i]}
                    </ToggleButton>
                  ))}
                </ToggleButtonGroup>
                <Typography sx={{ ...controlLabel }}>Ortho</Typography>
                <Switch checked={orthographic} onChange={(event) => setOrthographic(event.target.checked)} size="small" sx={switchStyles.small} slotProps={switchInputSlot("Toggle orthographic 3D projection")} />
                {anySlicePlaneVisible && (
                  <>
                    <Typography sx={{ ...controlLabel }}>Opacity</Typography>
                    <LiveNumberSlider value={slicePlaneOpacity} min={0.05} max={1} step={0.05} onLiveChange={(value) => handleVolumeControlChange("slicePlaneOpacity", value)} onCommit={(value) => handleVolumeControlCommit("slicePlaneOpacity", value)} sx={{ ...sliderStyles.small, width: 50 }} ariaLabel="Slice plane opacity" />
                  </>
                )}
                <Typography sx={{ ...controlLabel }}>Vol Strength</Typography>
                <LiveNumberSlider value={opacityA} min={0} max={1} step={0.05} onLiveChange={(value) => handleVolumeControlChange("opacity", value)} onCommit={(value) => handleVolumeControlCommit("opacity", value)} sx={{ ...sliderStyles.small, width: 50 }} ariaLabel="Volume strength" />
              </Box>
              )}
              <Box
                sx={{
                  ...container.imageBox,
                  border: `1px solid ${themeColors.border}`,
                  width: volumeCanvasSize,
                  height: volumeCanvasSize,
                  cursor: volumeDrag ? "grabbing" : "grab",
                }}
                onMouseDown={handleVolumeMouseDown}
                onWheel={handleVolumeWheel}
                onDoubleClick={handleVolumeDoubleClick}
                onContextMenu={(event) => event.preventDefault()}
              >
                <canvas
                  data-quantem-scientific-output="show3dslices-volume"
                  ref={volumeCanvasRef}
                  style={{ width: volumeCanvasSize, height: volumeCanvasSize, display: "block" }}
                  role="img"
                  aria-label={`3D volume rendering: ${displayTitle} (${nx} by ${ny} by ${nz} voxels). Drag to rotate, wheel to zoom.`}
                />
                {cameraChanged && (
                  <Button
                    size="small"
                    sx={{ ...compactButton, position: "absolute", top: 4, right: 4, minWidth: 0, px: 0.75, bgcolor: "rgba(255,255,255,0.75)", "&:hover": { bgcolor: "rgba(255,255,255,0.9)" } }}
                    onClick={(event) => { event.stopPropagation(); liveCameraRef.current = SHOW3DSLICES_DEFAULT_CAMERA; setCamera(SHOW3DSLICES_DEFAULT_CAMERA); persistViewState(undefined, undefined, SHOW3DSLICES_DEFAULT_CAMERA); }}
                    aria-label="Reset 3D camera view"
                    title="Reset 3D camera view"
                  >
                    Reset View
                  </Button>
                )}
                <Box
                  onMouseDown={handleVolumeResizeStart}
                  sx={resizeHandleSx}
                />
              </Box>
              <Box sx={{ ...inlineVolumeControlRow, mt: 0 }}>
                <Typography sx={{ ...controlLabel }} title="Align the 3D camera to a slice plane.">View</Typography>
                {VOLUME_VIEW_PRESETS.map(({ value, label, description }) => (
                  <Button
                    key={value}
                    size="small"
                    sx={{ ...compactButton, minWidth: label === "Top" ? 28 : 30, px: 0.5 }}
                    onClick={() => setVolumeView(value)}
                    aria-label={`Set 3D view to ${description}`}
                    title={`Set 3D view to ${description}`}
                  >
                    {label}
                  </Button>
                ))}
                <Button
                  size="small"
                  sx={{ ...compactButton, minWidth: 28, px: 0.5, fontSize: 13 }}
                  onClick={() => rollVolumeView(1)}
                  aria-label="Roll 3D camera view counterclockwise 90 degrees"
                  title="Roll view counterclockwise 90 degrees"
                >
                  ↺90
                </Button>
                <Button
                  size="small"
                  sx={{ ...compactButton, minWidth: 28, px: 0.5, fontSize: 13 }}
                  onClick={() => rollVolumeView(-1)}
                  aria-label="Roll 3D camera view clockwise 90 degrees"
                  title="Roll view clockwise 90 degrees"
                >
                  ↻90
                </Button>
              </Box>
            </Box>
          </Stack>
        ) : (
          <Box sx={{
            width: 220, py: 1.5, px: 1.5, alignSelf: "flex-start",
            display: "flex", flexDirection: "column", gap: 0.5,
          }}>
            <Typography sx={{ ...typography.label, color: themeColors.text, fontWeight: "bold", fontSize: 11 }}>
              3D volume needs WebGPU
            </Typography>
            <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: 11, lineHeight: 1.4 }}>
              The slice panels work without it. To enable the 3D view, turn on hardware
              acceleration in your browser (Settings - System) and reload.
            </Typography>
            {volumeInitError && (
              <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: 9, opacity: 0.7, mt: 0.5, wordBreak: "break-word" }}>
                {volumeInitError}
              </Typography>
            )}
          </Box>
        )}
      </Box>
      {/* Right column: slice toolbar + projected slice panels (grouped so they
          sit beside the 3D volume rather than below it). */}
      <Box sx={{ display: "flex", flexDirection: "column", flex: "1 1 auto", flexBasis: { xs: "100%", md: panelTotalW }, minWidth: 0, width: { xs: "100%", md: "auto" }, maxWidth: "100%", alignItems: "flex-start" }}>
      <Box sx={{ display: "flex", justifyContent: { xs: "flex-start", md: "flex-end" }, alignItems: "center", width: { xs: "100%", md: panelTotalW }, maxWidth: "100%", minHeight: 24, mb: `${SPACING.XS}px`, pointerEvents: "none" }}>
        <Box sx={{ pointerEvents: "auto" }}>
        {topRightActions}
        </Box>
      </Box>
      {(() => {
        const panels = AXES.map((_, axis) => {
          const { w: canvasW, h: canvasH, displayH } = canvasSizes[axis];
          const panelName = PANEL_NAMES[axis] ?? "Slice";
          return (
            // Hidden with display:none rather than unmounted so the canvas refs
            // and their painted contents survive - remounting would drop the
            // direct-paint targets and force a full repaint on every toggle.
            <Box
              key={axis}
              sx={{ minWidth: canvasW, gridArea: `a${axis}`, display: normalizedPlaneVisibility[axis] ? undefined : "none" }}
            >
              {/* displayH is the z-stretched height on depth panels. */}
              <Box
                ref={(el: HTMLDivElement | null) => { imageBoxRefs.current[axis] = el; }}
                sx={{ ...container.imageBox, width: canvasW, height: displayH, cursor: slicePanelCursor(axis), borderColor: PLANE_COLORS[axis] }}
                onMouseDown={(event) => handleMouseDown(event, axis)}
                onMouseMove={(event) => handleMouseMove(event, axis)}
                onMouseUp={(event) => handleMouseUp(event, axis)}
                onMouseLeave={handleMouseLeave}
                onWheel={(event) => handleWheel(event, axis)}
                onDoubleClick={() => handleDoubleClick(axis)}
              >
                <canvas
                  data-quantem-scientific-output={`show3dslices-slice-${axis}`}
                  ref={(el) => { canvasRefs.current[axis] = el; }}
                  width={canvasW}
                  height={canvasH}
                  style={{ width: canvasW, height: displayH, imageRendering: smooth ? "auto" : "pixelated" }}
                  role="img"
                  aria-label={axis === 0
                    ? `XY slice ${liveSlider[0] + 1} of ${nz} along ${axisLabels[0]} axis${title ? `: ${title}` : ""} (${canvasW} by ${canvasH} pixels)`
                    : `Oblique vertical slice at ${obliqueAngle.toFixed(1)} degrees, position ${Math.round(obliqueCurrentOffset)}${title ? `: ${title}` : ""} (${canvasW} by ${canvasH} pixels)`}
                />
                <canvas
                  ref={(el) => { overlayRefs.current[axis] = el; }}
                  width={canvasW}
                  height={displayH}
                  style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: displayH, pointerEvents: "none" }}
                  aria-hidden="true"
                />
                <canvas
                  ref={(el) => { uiRefs.current[axis] = el; }}
                  width={Math.round(canvasW * DPR)}
                  height={Math.round(displayH * DPR)}
                  style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: displayH, pointerEvents: "none" }}
                  aria-hidden="true"
                />
                {cursorInfo && cursorInfo.view === panelName && (
                  <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                    <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                      ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                    </Typography>
                  </Box>
                )}
                {/* Over-clip warning: image is mostly black because histogram thumbs sit outside data range */}
                {isOverClipped && axis === 0 && (
                  <Box sx={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", bgcolor: "rgba(255, 180, 0, 0.85)", color: "#000", px: 1, py: 0.5, fontSize: 11, fontWeight: "bold", borderRadius: 0.5, textAlign: "center", lineHeight: 1.3, pointerEvents: "none", maxWidth: canvasW - 20 }}>
                    No data visible<br/>
                    <span style={{ fontSize: 9, fontWeight: "normal" }}>Adjust contrast range or enable Auto</span>
                  </Box>
                )}
                <Box
                  onMouseDown={(event) => handleResizeStart(event, axis)}
                  sx={resizeHandleSx}
                />
              </Box>
              {showFft && (
                <Box sx={{ mt: `${SPACING.SM}px` }}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 20 }}>
                    <Stack direction="row" alignItems="center" sx={{ overflow: "hidden" }}>
                      <Typography variant="caption" sx={{ ...typography.label, fontSize: 10, flexShrink: 0 }}>
                        {`FFT ${axis === 0 ? `${axisLabels[1]}${axisLabels[2]}` : `oblique ${obliqueAngle.toFixed(1)}°`} ${gpuReady ? "" : " (CPU fallback)"}`}
                      </Typography>
                      {fftClickInfo && fftClickInfo.axis === axis && (
                        <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: themeColors.textMuted, ml: 1, whiteSpace: "nowrap" }}>
                          {fftClickInfo.dSpacing != null ? (
                            <>d=<Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} \u00C5`}</Box>{" |g|="}<Box component="span" sx={{ color: themeColors.accent }}>{`${fftClickInfo.spatialFreq!.toFixed(4)} \u00C5\u207B\u00B9`}</Box></>
                          ) : (
                            <>dist=<Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                          )}
                        </Typography>
                      )}
                    </Stack>
                    <Button size="small" sx={compactButton} disabled={!fftNeedsResetAxis(axis)} onClick={() => handleFftResetAxis(axis)} aria-label={`Reset ${panelName} FFT zoom and pan`}>Reset</Button>
                  </Stack>
                  <Box
                    sx={{ ...container.imageBox, width: canvasW, height: displayH, cursor: "grab", borderColor: PLANE_COLORS[axis] }}
                    onMouseDown={(event) => handleFftMouseDown(event, axis)}
                    onMouseMove={(event) => handleFftMouseMove(event, axis)}
                    onMouseUp={(event) => handleFftMouseUp(event, axis)}
                    onMouseLeave={() => { fftClickStartRef.current = null; setFftDragAxis(null); setFftDragStart(null); }}
                    onWheel={(event) => handleFftWheel(event, axis)}
                    onDoubleClick={() => handleFftDoubleClick(axis)}
                  >
                    <canvas
                      data-quantem-scientific-output={`show3dslices-fft-${axis}`}
                      ref={(el) => { fftCanvasRefs.current[axis] = el; }}
                      width={canvasW}
                      height={canvasH}
                      style={{ width: canvasW, height: displayH, imageRendering: smooth ? "auto" : "pixelated" }}
                      role="img"
                      aria-label={`FFT power spectrum of ${panelName} slice (reciprocal space, ${canvasW} by ${canvasH} pixels)`}
                    />
                    <canvas
                      ref={(el) => { fftOverlayRefs.current[axis] = el; }}
                      width={Math.round(canvasW * DPR)}
                      height={Math.round(displayH * DPR)}
                      style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: displayH, pointerEvents: "none" }}
                      aria-hidden="true"
                    />
                  </Box>
                </Box>
              )}
              <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, width: canvasW, maxWidth: canvasW, boxSizing: "border-box", ...(axis === 1 ? { flexDirection: "column", alignItems: "stretch", gap: `${SPACING.XS}px` } : {}) }}>
                {axis === 1 ? (
                  <>
                    <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 18 }}>
                      <Typography sx={{ ...controlLabel, color: themeColors.textMuted, flexShrink: 0, minWidth: 42 }}>Angle</Typography>
                      <Slider
                        value={obliqueAngleSliderValues}
                        min={obliqueAngleSliderMin}
                        max={obliqueAngleSliderMax}
                        step={1}
                        onPointerDownCapture={handleObliqueAnglePointerDownCapture}
                        onChange={(_, value) => handleObliqueAngleSliderChange(value as number[])}
                        onChangeCommitted={(_, value) => handleObliqueAngleSliderChange(value as number[])}
                        disableSwap
                        size="small"
                        sx={rangeSliderSx}
                        aria-label={`Oblique plane angle ${Math.round(boundedObliqueAngle)} degrees within ${obliqueAngleMinBound} to ${obliqueAngleMaxBound}`}
                        valueLabelDisplay="off"
                        valueLabelFormat={(value) => `${value as number}°`}
                      />
                      <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 36, textAlign: "right", flexShrink: 0 }}>
                        {Math.round(boundedObliqueAngle)}°
                      </Typography>
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 18 }}>
                      <Typography sx={{ ...controlLabel, color: themeColors.textMuted, flexShrink: 0, minWidth: 42 }}>Position</Typography>
                      <Slider
                        value={obliquePositionSliderValues}
                        min={obliquePositionSliderMin}
                        max={obliquePositionSliderMax}
                        step={1}
                        onPointerDownCapture={handleObliquePositionPointerDownCapture}
                        onChange={(_, value) => handleObliquePositionChange(value as number[])}
                        onChangeCommitted={(_, value) => {
                          handleObliquePositionCommit(value as number[]);
                        }}
                        disableSwap
                        size="small"
                        sx={rangeSliderSx}
                        aria-label={`Oblique plane position ${boundedObliqueOffset} within ${obliquePositionMinBound} to ${obliquePositionMaxBound}`}
                        valueLabelDisplay="off"
                        valueLabelFormat={(value) => `${value as number}`}
                      />
                      <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 36, textAlign: "right", flexShrink: 0 }}>
                        {boundedObliqueOffset}
                      </Typography>
                    </Box>
                    {/* Position is measured along the plane normal, an axis that
                        turns with Angle, so it shifts on rotation even when the cut
                        has not moved. The center reports the same plane in fixed
                        image pixels. It holds steady while a rotation leaves the
                        chord inside the image; near an edge the endpoint clamp in
                        handleObliqueAngleChange shortens the segment and does move
                        the midpoint, so the center shifts there too. */}
                    <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 14 }}>
                      <Typography
                        sx={{ ...controlLabel, color: themeColors.textMuted, flexShrink: 0, minWidth: 42 }}
                        title="Plane center in image pixels. Steady under rotation unless the cut is clamped at an image edge."
                      >
                        Center
                      </Typography>
                      <Typography sx={{ ...typography.value, color: themeColors.textMuted, flexShrink: 0 }}>
                        {`(${Math.round(obliqueCenterRow)}, ${Math.round(obliqueCenterCol)})`}
                      </Typography>
                    </Box>
                  </>
                ) : (
                  <>
                <Typography sx={{ ...controlLabel, color: themeColors.textMuted, flexShrink: 0 }}>{axisLabels[0]}</Typography>
                {loop ? (
                  <Slider
                    value={loopSliderValues(axis)}
                    onPointerDownCapture={(event) => handleLoopSliderPointerDownCapture(axis, event)}
                    onChange={(_, value) => {
                      handleLoopSliderChange(axis, value as number[]);
                    }}
                    onChangeCommitted={(_, value) => {
                      handleLoopSliderCommit(axis, value as number[]);
                    }}
                    disableSwap
                    min={0}
                    max={sliceMaxes[axis]}
                    size="small"
                    valueLabelDisplay="off"
                    sx={rangeSliderSx}
                    aria-label={`Loop range and current ${axisLabels[axis]} slice (${liveSlider[axis] + 1} of ${sliceMaxes[axis] + 1}, loop ${loopStarts[axis] + 1} to ${effectiveLoopEnds[axis] + 1})`}
                    valueLabelFormat={(value) => `${value as number}`}
                  />
                ) : (
                  <Slider
                    value={liveSlider[axis]}
                    min={0}
                    max={sliceMaxes[axis]}
                    onChange={sliceSetters[axis]}
                    onChangeCommitted={sliceCommitters[axis]}
                    size="small"
                    sx={{ ...sliderStyles.small, flex: 1, minWidth: 40 }}
                    aria-label={`${axisLabels[axis]} slice ${liveSlider[axis] + 1} of ${sliceMaxes[axis] + 1}`}
                    valueLabelDisplay="off"
                    valueLabelFormat={(value) => `${value as number}`}
                  />
                )}
                {axis === 0 && (
                  <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 28, textAlign: "right", flexShrink: 0 }}>
                    {liveSlider[axis]}/{sliceMaxes[axis]}
                  </Typography>
                )}
                  </>
                )}
              </Box>
            </Box>
          );
        });
        return (
          <Box sx={{ display: "flex", flexDirection: { xs: "column", sm: "row" }, flexWrap: "wrap", alignItems: "flex-start", gap: `${SPACING.SM}px`, justifyContent: "flex-start", mt: `${SLICE_PANEL_TOP_ALIGN_PX}px`, maxWidth: "100%" }}>
            {panels}
          </Box>
        );
      })()}
      </Box> {/* end right column (toolbar + slices) */}
      </Box> {/* end side-by-side row (3D volume + slices) */}
      {showFft && (
        <Box sx={{
          ...panelControlRow,
          mt: `${SPACING.SM}px`,
          ml: { xs: 0, md: `${sliceColumnOffsetPx}px` },
          [`@media (max-width:${sideBySideMinWidth - 1}px)`]: { ml: 0 },
          width: "fit-content",
          maxWidth: panelTotalW,
          flexWrap: "wrap",
        }}>
          <Typography sx={{ ...controlLabel }}>FFT Scale</Typography>
          <Select value={fftLogScale ? "log" : "linear"} onChange={(event) => setFftLogScale(event.target.value === "log")} size="small" sx={{ ...denseSelect, minWidth: 45 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT intensity scale (linear or logarithmic)" }}>
            <MenuItem value="linear">Lin</MenuItem>
            <MenuItem value="log">Log</MenuItem>
          </Select>
          <Typography sx={{ ...controlLabel }}>FFT Color</Typography>
          <Select value={fftColormap} onChange={(event) => setFftColormap(String(event.target.value))} size="small" sx={{ ...denseSelect, minWidth: 60 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT colormap" }}>
            {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
          </Select>
          <Typography sx={{ ...controlLabel }}>FFT Auto</Typography>
          <Switch checked={fftAuto} onChange={(event) => setFftAuto(event.target.checked)} size="small" sx={switchStyles.small} slotProps={switchInputSlot("Toggle automatic FFT contrast")} />
          <Typography sx={{ ...controlLabel }} title="Apply a Hann window before zero-padding each slice FFT to reduce edge leakage.">Window</Typography>
          <Switch checked={!!fftWindow} onChange={(event) => setFftWindow(event.target.checked)} size="small" sx={switchStyles.small} slotProps={switchInputSlot("Toggle Hann window before FFT")} />
        </Box>
      )}
      {/* Controls row with histogram anchored to the slice panel columns. */}
      {controlsVisible && (() => {
        const histogramW = 110;
        const histogramH = controlRowHeight * 2 + SPACING.XS;
        return (
        <Box sx={{
          mt: `${SPACING.SM}px`,
          display: "flex",
          gap: `${SPACING.SM}px`,
          alignItems: "flex-start",
          width: "fit-content",
          maxWidth: { xs: "100%", md: panelTotalW },
          boxSizing: "border-box",
          flexWrap: "wrap",
        }}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, justifyContent: "flex-start", minWidth: 0 }}>
            <Box sx={contentControlRow}>
              <Typography sx={{ ...controlLabel }}>Color</Typography>
              <Select size="small" value={cmap} onChange={(event) => setCmap(event.target.value)} MenuProps={themedMenuProps} sx={{ ...denseSelect, minWidth: 60 }} inputProps={{ "aria-label": "Image colormap" }}>
                {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
              </Select>
              <Typography sx={{ ...controlLabel }}>Colorbar</Typography>
              <Switch checked={showColorbar} onChange={(event) => setShowColorbar(event.target.checked)} size="small" sx={switchStyles.small} slotProps={switchInputSlot("Toggle colorbar overlay")} />
              <Typography sx={{ ...controlLabel }} title="CSS bilinear interpolation on image canvas. Off = pixelated.">Smooth</Typography>
              <Switch checked={smooth} onChange={(event) => setSmooth(event.target.checked)} size="small" sx={switchStyles.small} slotProps={switchInputSlot("Toggle bilinear smoothing")} />
            </Box>
            <Box sx={contentControlRow}>
              <Typography sx={{ ...controlLabel }} title="Depth-axis display height multiplier (1-50x). CSS-only stretch; data unchanged. Useful when nz << nxy (e.g. multislice ptycho).">Z stretch</Typography>
              <LiveNumberSlider value={zStretch} min={1} max={50} step={0.5} onLiveChange={handleZStretchChange} onCommit={handleZStretchCommit} sx={{ ...sliderStyles.small, width: 80, mr: 1, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }} ariaLabel="Depth axis display stretch multiplier" />
              <Typography sx={clickableControlLabel} title="Negate displayed values. Useful when phase sign is inverted." onClick={() => setFlip(!flip)}>Flip</Typography>
              <Switch checked={flip} onChange={(event) => setFlip(event.target.checked)} size="small" sx={switchStyles.small} slotProps={switchInputSlot("Flip (negate) displayed values")} />
              <Typography sx={{ ...controlLabel }} title="Log scale (signed log1p). Useful for high-dynamic-range volumes.">Log</Typography>
              <Switch checked={logScale} onChange={(event) => setLogScale(event.target.checked)} size="small" sx={switchStyles.small} slotProps={switchInputSlot("Toggle log scale (signed log1p) display")} />
              <Typography sx={{ ...controlLabel }} title={hasExplicitLimits ? "Off while vmin or vmax is set: the window is used as given" : undefined}>Auto</Typography>
              <Switch checked={autoActive} disabled={hasExplicitLimits} onChange={(event) => handleAutoContrastChange(event.target.checked)} size="small" sx={switchStyles.small} slotProps={switchInputSlot("Toggle automatic percentile-based contrast")} />
              <Button
                size="small"
                sx={{ ...compactButton, color: advancedControlsOpen ? themeColors.accent : themeColors.textMuted }}
                onClick={() => setAdvancedControlsOpen(!advancedControlsOpen)}
                aria-expanded={advancedControlsOpen}
                aria-controls="show3dslices-advanced-controls"
              >
                Advanced
              </Button>
            </Box>
            {advancedControlsOpen && <Box id="show3dslices-advanced-controls" sx={contentControlRow}>
              <Typography sx={{ ...controlLabel }} title="Display-only global post-alignment through depth. Raw volume data is unchanged.">Align</Typography>
              <Switch
                checked={alignmentActive}
                onChange={(event) => handleSliceAlignmentToggle(event.target.checked)}
                size="small"
                sx={switchStyles.small}
                slotProps={switchInputSlot("Align slices with automatic global slice alignment")}
              />
              {alignmentActive && (
                <>
                  <Typography sx={{ ...controlLabel, color: themeColors.textMuted }}>Row</Typography>
                  <LiveNumberSlider
                    value={liveRowShift}
                    min={-alignmentShiftLimit}
                    max={alignmentShiftLimit}
                    step={0.05}
                    onLiveChange={(value) => {
                      stageManualSliceAlignment(value, liveColShift);
                    }}
                    onCommit={(value) => commitManualSliceAlignment(value, liveColShift)}
                    sx={{ ...sliderStyles.small, width: 58, flexShrink: 0 }}
                    ariaLabel={`Row shift per slice ${liveRowShift.toFixed(3)} pixels`}
                  />
                  <NumberCommitInput
                    value={liveRowShift}
                    min={-alignmentShiftLimit}
                    max={alignmentShiftLimit}
                    step={0.05}
                    onLiveChange={(value) => {
                      stageManualSliceAlignment(value, liveColShift);
                    }}
                    onCommit={(value) => commitManualSliceAlignment(value, liveColShift)}
                    ariaLabel="Edit row shift per slice"
                  />
                  <Typography sx={{ ...controlLabel, color: themeColors.textMuted }}>Col</Typography>
                  <LiveNumberSlider
                    value={liveColShift}
                    min={-alignmentShiftLimit}
                    max={alignmentShiftLimit}
                    step={0.05}
                    onLiveChange={(value) => {
                      stageManualSliceAlignment(liveRowShift, value);
                    }}
                    onCommit={(value) => commitManualSliceAlignment(liveRowShift, value)}
                    sx={{ ...sliderStyles.small, width: 58, flexShrink: 0 }}
                    ariaLabel={`Column shift per slice ${liveColShift.toFixed(3)} pixels`}
                  />
                  <NumberCommitInput
                    value={liveColShift}
                    min={-alignmentShiftLimit}
                    max={alignmentShiftLimit}
                    step={0.05}
                    onLiveChange={(value) => {
                      stageManualSliceAlignment(liveRowShift, value);
                    }}
                    onCommit={(value) => commitManualSliceAlignment(liveRowShift, value)}
                    ariaLabel="Edit column shift per slice"
                  />
                  <Button
                    size="small"
                    sx={{ ...compactButton, color: themeColors.accent }}
                    disabled={!sliceAlignmentCached}
                    onClick={resetSliceAlignment}
                    aria-label="Reset slice alignment"
                    title={offline ? "Restore the exported alignment estimate" : "Discard the cached alignment estimate"}
                  >
                    Reset
                  </Button>
                </>
              )}
              {alignmentStatusText && (
                <Typography
                  sx={{
                    ...controlLabel,
                    maxWidth: 160,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: alignmentStatusText.startsWith("Slice alignment failed") ? "#d32f2f" : themeColors.textMuted,
                  }}
                  title={alignmentStatusText}
                >
                  {alignmentStatusText}
                </Typography>
              )}
            </Box>}
          </Box>
          <Box sx={{ display: "flex", flexDirection: "row", gap: `${SPACING.SM}px`, alignItems: "flex-start", justifyContent: "flex-start" }}>
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start" }}>
              <Histogram
                data={imageHistogramData}
                vminPct={imageVminPct}
                vmaxPct={imageVmaxPct}
                onRangeChange={(min, max) => {
                  paintContrastRange(min, max);
                }}
                onRangeCommit={(min, max) => {
                  // User drag overrides Auto. Commit once on release so dragging stays local.
                  if (autoContrast) {
                    manualImageRangeBeforeAutoRef.current = null;
                    setAutoContrast(false);
                  }
                  setImageVminPct(min);
                  setImageVmaxPct(max);
                }}
                width={histogramW}
                height={histogramH}
                theme={themeInfo.theme === "dark" ? "dark" : "light"}
                dataMin={displayDataRange.min}
                dataMax={displayDataRange.max}
                pinBinsToRange={false}
                ariaHidden
              />
            </Box>
          </Box>
        </Box>
        );
      })()}
      {controlsVisible && <Box sx={{ ...contentControlRow, mt: `${SPACING.SM}px`, flexWrap: "nowrap" }}>
        <Select
          value={playbackAxis}
          onChange={(event) => { setPlaying(false); setPlayAxis(Number(event.target.value)); }}
          size="small"
          sx={{ ...denseSelect, minWidth: 40 }}
          MenuProps={themedMenuProps}
          inputProps={{ "aria-label": "Playback axis (Top, Side, or All)" }}
        >
          <MenuItem value={0}>Top</MenuItem>
          <MenuItem value={1}>Side</MenuItem>
          <MenuItem value={3}>All</MenuItem>
        </Select>
        <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
          <IconButton size="small" onClick={() => setReverse(!reverse)} sx={{ color: reverse ? themeColors.accent : themeColors.textMuted, p: 0.25 }} aria-label={reverse ? "Playback direction reverse" : "Playback direction forward"} aria-pressed={reverse} title={reverse ? "Direction: reverse" : "Direction: forward"}>
            <FastRewindIcon sx={{ fontSize: 18, transform: reverse ? "none" : "scaleX(-1)" }} />
          </IconButton>
          <IconButton size="small" onClick={() => setPlaying(!playing)} sx={{ color: themeColors.accent, p: 0.3 }} aria-label={playing ? "Pause playback" : "Play"} title={playing ? "Pause (Space)" : "Play (Space)"}>
            {playing ? <PauseIcon sx={{ fontSize: 20 }} /> : <PlayArrowIcon sx={{ fontSize: 20 }} />}
          </IconButton>
          <IconButton size="small" onClick={stopPlaybackAndRewind} sx={{ color: themeColors.textMuted, p: 0.25 }} aria-label="Stop and rewind to loop start" title="Stop">
            <StopIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Stack>
        <Typography sx={{ ...controlLabel, color: themeColors.textMuted, flexShrink: 0 }}>fps</Typography>
        <LiveNumberSlider
          value={fps}
          min={1}
          max={MAX_PLAYBACK_FPS}
          step={1}
          onLiveChange={(value) => {
            fpsRef.current = value;
            setFps(value);
          }}
          onCommit={(value) => {
            fpsRef.current = value;
            setFps(value);
            setModelFps(value);
          }}
          sx={{ ...sliderStyles.small, width: 35, flexShrink: 0 }}
          ariaLabel={`Playback frames per second (${Math.round(fps)})`}
        />
        <Typography sx={{ ...controlLabel, color: themeColors.textMuted, flexShrink: 0 }}>Loop</Typography>
        <Switch size="small" checked={loop} onChange={() => setLoop(!loop)} sx={{ ...switchStyles.small, flexShrink: 0 }} slotProps={switchInputSlot("Toggle loop playback")} />
        <Typography sx={{ ...controlLabel, color: themeColors.textMuted, flexShrink: 0 }}>Bounce</Typography>
        <Switch size="small" checked={boomerang} onChange={() => setBoomerang(!boomerang)} sx={{ ...switchStyles.small, flexShrink: 0 }} slotProps={switchInputSlot("Toggle bounce (ping-pong) playback")} />
      </Box>}
    </Box>
  );
}

// anywidget v0.9+ deprecates `export render` in favor of `export default { render }`.
const render = createRender(Show3DSlices);
export default { render };
