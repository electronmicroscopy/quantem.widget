/// <reference types="@webgpu/types" />
/**
 * Show3DSlices - slice viewer for 3D volumetric data.
 *
 * A Top (row, col) slice and an arbitrary-angle vertical (oblique) slice with
 * synchronized sliders, beside a ray-cast 3D view of the volume. Slices are
 * cut from the float32 volume resident on WebGPU; without WebGPU they are cut
 * and colormapped in JavaScript.
 */
import * as React from "react";
import { sliderStyles, SPACING, switchStyles, upwardMenuProps } from "../controlStyles";
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
import { VolumeRenderer, CameraState, DEFAULT_CAMERA } from "./webgpu-volume";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar } from "../figure";
import { extractBytes, extractFloat32, formatNumber, preserveRestoredWidgetModelsOnSave } from "../format";
import { findDataRange, applyLogScale, percentileClip, signedLog1p, sliderRange } from "../display/stats";
import { MetadataSection } from "../widgetInfo";
import { dequantizeUint8 } from "../display/quantization";
import { InfoTooltip } from "../shared/InfoTooltip";
import { RenderPathBadge } from "../shared/RenderPathBadge";
import { KeyboardShortcuts } from "../shared/KeyboardShortcuts";
import { exportTitleSlug, formatEstimatedHtmlSize } from "../shared/exportFormat";
import { HTML_PICKER_TYPE, type PendingSave, pickSaveHandle, useSaveExportPayload } from "../shared/exportSave";
import { pointToSegmentDistance } from "../shared/geometry";
import { shouldIgnoreWidgetShortcut } from "../shared/widgetShortcuts";
import { resolveDisplayBounds } from "../shared/displayRange";
import { Histogram } from "../shared/Histogram";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen, renderToOffscreenReuse, createGPUColormapEngine, GPUColormapEngine, VOLUME_SLICE_MIN_ZOOM } from "../display/colormaps";
import { getHardwareGPUDevice, type RenderPath } from "../display/device";
import { DisplayFFT, getDisplayFFT, fft2d, fftshift, nextPow2, computeMagnitude, autoEnhanceFFT, applyHannWindow2D, reciprocalCoordinatesFromShiftedOffset, zeroPadTopLeft } from "../display/fft";
import { findFFTPeakBrowser } from "../display/geometry";
import { estimateSliceAlignment } from "./sliceAlignment";

const MAX_PLAYBACK_FPS = 30;
const PAGE_PLAY_FPS_OPTIONS = [1, 2, 4, 8] as const;

const PLANE_KEYS = ["top", "oblique"] as const;
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

// A position in the volume is (row, col) in voxels; x and y are kept for canvas and pointer positions.
type ImagePoint = { row: number; col: number };

function extractTopSlice(volume: Float32Array, numCols: number, numRows: number, numSlices: number, sliceIndex: number): Float32Array {
  if (sliceIndex < 0 || sliceIndex >= numSlices) return new Float32Array(numRows * numCols);
  const start = sliceIndex * numRows * numCols;
  return volume.subarray(start, start + numRows * numCols);
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function obliqueLineEndpoints(
  numCols: number,
  numRows: number,
  centerCol: number,
  centerRow: number,
  angleDeg: number,
): [ImagePoint, ImagePoint] {
  const theta = (angleDeg * Math.PI) / 180;
  const dirCol = Math.cos(theta);
  const dirRow = Math.sin(theta);
  const candidates: number[] = [];
  const maxCol = Math.max(0, numCols - 1);
  const maxRow = Math.max(0, numRows - 1);
  if (Math.abs(dirCol) > 1e-8) {
    const t0 = (0 - centerCol) / dirCol;
    const row0 = centerRow + t0 * dirRow;
    if (row0 >= 0 && row0 <= maxRow) candidates.push(t0);
    const t1 = (maxCol - centerCol) / dirCol;
    const row1 = centerRow + t1 * dirRow;
    if (row1 >= 0 && row1 <= maxRow) candidates.push(t1);
  }
  if (Math.abs(dirRow) > 1e-8) {
    const t0 = (0 - centerRow) / dirRow;
    const col0 = centerCol + t0 * dirCol;
    if (col0 >= 0 && col0 <= maxCol) candidates.push(t0);
    const t1 = (maxRow - centerRow) / dirRow;
    const col1 = centerCol + t1 * dirCol;
    if (col1 >= 0 && col1 <= maxCol) candidates.push(t1);
  }
  if (candidates.length < 2) {
    return [
      { row: clampNumber(centerRow, 0, maxRow), col: clampNumber(centerCol, 0, maxCol) },
      { row: clampNumber(centerRow, 0, maxRow), col: clampNumber(centerCol, 0, maxCol) },
    ];
  }
  const minT = Math.min(...candidates);
  const maxT = Math.max(...candidates);
  return [
    { row: clampNumber(centerRow + minT * dirRow, 0, maxRow), col: clampNumber(centerCol + minT * dirCol, 0, maxCol) },
    { row: clampNumber(centerRow + maxT * dirRow, 0, maxRow), col: clampNumber(centerCol + maxT * dirCol, 0, maxCol) },
  ];
}

function obliqueNormal(angleDeg: number): ImagePoint {
  const theta = (angleDeg * Math.PI) / 180;
  return { row: Math.cos(theta), col: -Math.sin(theta) };
}

function obliqueSegmentOffsetBounds(
  numCols: number,
  numRows: number,
  angleDeg: number,
  start: ImagePoint,
  stop: ImagePoint,
  inset: number = 0,
): [number, number] {
  const normal = obliqueNormal(angleDeg);
  const points = [start, stop];
  const colMin = Math.min(inset, Math.max(0, numCols - 1));
  const rowMin = Math.min(inset, Math.max(0, numRows - 1));
  const colMax = Math.max(colMin, Math.max(1, numCols) - 1 - inset);
  const rowMax = Math.max(rowMin, Math.max(1, numRows) - 1 - inset);
  let minDelta = -Infinity;
  let maxDelta = Infinity;
  for (const point of points) {
    if (Math.abs(normal.col) > 1e-8) {
      const d0 = (colMin - point.col) / normal.col;
      const d1 = (colMax - point.col) / normal.col;
      minDelta = Math.max(minDelta, Math.min(d0, d1));
      maxDelta = Math.min(maxDelta, Math.max(d0, d1));
    }
    if (Math.abs(normal.row) > 1e-8) {
      const d0 = (rowMin - point.row) / normal.row;
      const d1 = (rowMax - point.row) / normal.row;
      minDelta = Math.max(minDelta, Math.min(d0, d1));
      maxDelta = Math.min(maxDelta, Math.max(d0, d1));
    }
  }
  if (!Number.isFinite(minDelta) || !Number.isFinite(maxDelta) || minDelta > maxDelta) return [0, 0];
  return [Math.ceil(minDelta), Math.floor(maxDelta)];
}

function obliqueCenterOffset(
  numCols: number,
  numRows: number,
  angleDeg: number,
  start: ImagePoint,
  stop: ImagePoint,
): number {
  const normal = obliqueNormal(angleDeg);
  const originCol = (Math.max(1, numCols) - 1) / 2;
  const originRow = (Math.max(1, numRows) - 1) / 2;
  const centerCol = (start.col + stop.col) / 2;
  const centerRow = (start.row + stop.row) / 2;
  return (centerCol - originCol) * normal.col + (centerRow - originRow) * normal.row;
}

function profilePointFromAny(value: unknown): ImagePoint | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const row = Number(record.row);
  const col = Number(record.col);
  if (!Number.isFinite(row) || !Number.isFinite(col)) return null;
  return { row, col };
}

function profileLinePayload(start: ImagePoint, stop: ImagePoint): ImagePoint[] {
  return [
    { row: start.row, col: start.col },
    { row: stop.row, col: stop.col },
  ];
}

function clampPointToImage(point: ImagePoint, numCols: number, numRows: number, inset: number = 0): ImagePoint {
  const colInset = Math.min(inset, Math.max(0, numCols - 1) / 2);
  const rowInset = Math.min(inset, Math.max(0, numRows - 1) / 2);
  return {
    row: clampNumber(point.row, rowInset, Math.max(rowInset, Math.max(0, numRows - 1) - rowInset)),
    col: clampNumber(point.col, colInset, Math.max(colInset, Math.max(0, numCols - 1) - colInset)),
  };
}

function translateSegmentInsideImage(
  start: ImagePoint,
  stop: ImagePoint,
  colDeltaRaw: number,
  rowDeltaRaw: number,
  numCols: number,
  numRows: number,
  inset: number = 0,
): { start: ImagePoint; stop: ImagePoint } {
  const colMin = Math.min(inset, Math.max(0, numCols - 1));
  const rowMin = Math.min(inset, Math.max(0, numRows - 1));
  const colMax = Math.max(colMin, Math.max(1, numCols) - 1 - inset);
  const rowMax = Math.max(rowMin, Math.max(1, numRows) - 1 - inset);
  const colDeltaMin = Math.max(colMin - start.col, colMin - stop.col);
  const colDeltaMax = Math.min(colMax - start.col, colMax - stop.col);
  const rowDeltaMin = Math.max(rowMin - start.row, rowMin - stop.row);
  const rowDeltaMax = Math.min(rowMax - start.row, rowMax - stop.row);
  const colDelta = clampNumber(colDeltaRaw, colDeltaMin, colDeltaMax);
  const rowDelta = clampNumber(rowDeltaRaw, rowDeltaMin, rowDeltaMax);
  return {
    start: { row: start.row + rowDelta, col: start.col + colDelta },
    stop: { row: stop.row + rowDelta, col: stop.col + colDelta },
  };
}

function segmentWidth(start: ImagePoint, stop: ImagePoint): number {
  return Math.max(1, Math.ceil(Math.hypot(stop.col - start.col, stop.row - start.row)) + 1);
}

function sampleVolumeBilinear(
  volume: Float32Array,
  numCols: number,
  numRows: number,
  numSlices: number,
  sliceIndex: number,
  col: number,
  row: number,
): number {
  if (sliceIndex < 0 || sliceIndex >= numSlices || col < 0 || row < 0 || col > numCols - 1 || row > numRows - 1) return Number.NaN;
  const col0 = Math.floor(col);
  const row0 = Math.floor(row);
  const col1 = Math.min(numCols - 1, col0 + 1);
  const row1 = Math.min(numRows - 1, row0 + 1);
  const colFrac = col - col0;
  const rowFrac = row - row0;
  const base = sliceIndex * numRows * numCols;
  const v00 = volume[base + row0 * numCols + col0];
  const v10 = volume[base + row0 * numCols + col1];
  const v01 = volume[base + row1 * numCols + col0];
  const v11 = volume[base + row1 * numCols + col1];
  return (v00 * (1 - colFrac) + v10 * colFrac) * (1 - rowFrac) + (v01 * (1 - colFrac) + v11 * colFrac) * rowFrac;
}

function extractOblique(
  volume: Float32Array,
  numCols: number,
  numRows: number,
  numSlices: number,
  start: ImagePoint,
  stop: ImagePoint,
): Float32Array {
  const width = segmentWidth(start, stop);
  const out = new Float32Array(numSlices * width);
  const denom = Math.max(1, width - 1);
  for (let sliceIndex = 0; sliceIndex < numSlices; sliceIndex++) {
    for (let panelCol = 0; panelCol < width; panelCol++) {
      const t = panelCol / denom;
      const col = start.col + (stop.col - start.col) * t;
      const row = start.row + (stop.row - start.row) * t;
      const value = sampleVolumeBilinear(volume, numCols, numRows, numSlices, sliceIndex, col, row);
      out[sliceIndex * width + panelCol] = Number.isFinite(value) ? value : 0;
    }
  }
  return out;
}

// Copies a row-major slice into the top-left of a zero-filled power-of-two
// grid, the input layout the FFT needs.
function extractVolumeFloat32(
  dataView: DataView | ArrayBuffer | Uint8Array,
  offline: boolean,
  offlineMin: number,
  offlineMax: number,
  numCols: number,
  numRows: number,
  numSlices: number,
  panelCount = 1,
): Float32Array | null {
  const panels = Math.max(1, Math.floor(panelCount || 1));
  const count = Math.max(0, Math.floor(numCols) * Math.floor(numRows) * Math.floor(numSlices) * panels);
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

function makeExportFilename(title: string, numSlices: number, numRows: number, numCols: number, encoding: "full" | "uint8"): string {
  const slug = exportTitleSlug(title, "show3dslices");
  const suffix = encoding === "uint8" ? "quantized" : "exact";
  return `${slug}_${numSlices}x${numRows}x${numCols}_${suffix}.html`;
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

// Raster size of one slice panel before slice stretch: the top panel fits its longer
// side to the target, a depth panel takes the full target width and one canvas
// row per slice so its height does not follow the oblique profile length.
function panelRasterSize([sliceH, sliceW]: [number, number], isDepth: boolean, target: number) {
  const width = isDepth ? target : Math.round(sliceW * (target / Math.max(sliceW, sliceH)));
  const height = isDepth ? Math.max(1, sliceH) : Math.round(sliceH * (target / Math.max(sliceW, sliceH)));
  return { w: width, h: height, scaleX: width / Math.max(1, sliceW), scaleY: height / Math.max(1, sliceH) };
}

/**
 * Real-space sampling (Å per px) along each slice panel's rows and columns,
 * from the volume's [slice, row, col]; zero when uncalibrated. The oblique panel's
 * rows are depth slices and its columns step along the cut at `angleDeg` in
 * the row/col plane, so their spacing mixes the row and col samplings.
 */
function panelSampling(axes: number[], angleDeg: number): { row: number; col: number }[] {
  const theta = (angleDeg * Math.PI) / 180;
  return [
    { row: axes[1], col: axes[2] },
    { row: axes[0], col: Math.hypot(Math.cos(theta) * axes[2], Math.sin(theta) * axes[1]) },
  ];
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

/**
 * A press on a slider's track (not a thumb) moves the current value to the
 * pointer and follows it until release. MUI would instead move the nearest
 * thumb, which on the range sliders here is a loop or angle bound.
 * `apply(value, released)` receives every value, `released` on the last.
 */
function trackSliderPress(
  event: React.PointerEvent<HTMLSpanElement>,
  valueAt: (fraction: number) => number,
  apply: (value: number, released: boolean) => void,
): void {
  if ((event.target as HTMLElement).closest(".MuiSlider-thumb")) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const valueFromClientX = (clientX: number) => valueAt(rect.width > 0 ? (clientX - rect.left) / rect.width : 0);
  event.preventDefault();
  event.stopPropagation();
  event.nativeEvent.stopImmediatePropagation();
  apply(valueFromClientX(event.clientX), false);
  const onMove = (moveEvent: PointerEvent) => {
    moveEvent.preventDefault();
    apply(valueFromClientX(moveEvent.clientX), false);
  };
  const onUp = (upEvent: PointerEvent) => {
    upEvent.preventDefault();
    window.removeEventListener("pointermove", onMove, true);
    window.removeEventListener("pointerup", onUp, true);
    apply(valueFromClientX(upEvent.clientX), true);
  };
  window.addEventListener("pointermove", onMove, true);
  window.addEventListener("pointerup", onUp, true);
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
const AXES = ["top", "oblique"] as const;
const PANEL_NAMES = ["XY", "Oblique"] as const;
// Show3DSlices opens in the same orientation as the main top slice panel:
// columns left-to-right and rows top-to-bottom.
const SHOW3DSLICES_DEFAULT_CAMERA: CameraState = {
  ...DEFAULT_CAMERA,
  yaw: Math.PI,
  pitch: 0,
  roll: Math.PI,
};
const VOLUME_VIEW_PRESETS = [
  { value: "top", label: "Top", description: "top (XY) view" },
  { value: "side", label: "Side", description: "oblique vertical plane view" },
] as const;
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
    slotProps: { paper: { sx: { borderRadius: 0, bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } } },
  };

  const [volumeShape] = useModelState<number[]>("volume_shape");
  const [numSlices, numRows, numCols] = volumeShape;
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
  const [colIndex, setColIndex] = useModelState<number>("col_index");
  const [rowIndex, setRowIndex] = useModelState<number>("row_index");
  const [sliceIndex, setSliceIndex] = useModelState<number>("slice_index");
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
    panX: readNumber(value?.panX, fallback.panX),
    panY: readNumber(value?.panY, fallback.panY),
  });
  const normalizeCameraState = (value: Partial<CameraState> | undefined): CameraState => ({
    ...SHOW3DSLICES_DEFAULT_CAMERA,
    yaw: readNumber(value?.yaw, SHOW3DSLICES_DEFAULT_CAMERA.yaw),
    pitch: Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, readNumber(value?.pitch, SHOW3DSLICES_DEFAULT_CAMERA.pitch))),
    roll: readNumber(value?.roll, SHOW3DSLICES_DEFAULT_CAMERA.roll ?? 0),
    distance: Math.max(0.5, Math.min(10, readNumber(value?.distance, SHOW3DSLICES_DEFAULT_CAMERA.distance))),
    panX: readNumber(value?.panX, SHOW3DSLICES_DEFAULT_CAMERA.panX),
    panY: readNumber(value?.panY, SHOW3DSLICES_DEFAULT_CAMERA.panY),
  });
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [orthographic, setOrthographic] = useModelState<boolean>("orthographic");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const [flip, setFlip] = useModelState<boolean>("flip");
  const [dimLabels] = useModelState<string[]>("dim_labels");
  // Per-axis sampling [slice, row, col] in Angstrom (zeros when uncalibrated).
  const [voxelSizeA] = useModelState<number[]>("voxel_size_A");
  const [showScaleBar] = useModelState<boolean>("show_scale_bar");
  const [modelSliceStretch, setModelSliceStretch] = useModelState<number>("slice_stretch");
  const [sliceStretch, setSliceStretch] = React.useState(modelSliceStretch);
  const pendingSliceStretchRef = React.useRef(modelSliceStretch);
  const sliceStretchLiveDirtyRef = React.useRef(false);
  const sliceStretchRafRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    sliceStretchLiveDirtyRef.current = false;
    pendingSliceStretchRef.current = modelSliceStretch;
    setSliceStretch(modelSliceStretch);
  }, [modelSliceStretch]);
  React.useEffect(() => {
    return () => {
      if (sliceStretchRafRef.current != null) cancelAnimationFrame(sliceStretchRafRef.current);
    };
  }, []);

  React.useEffect(() => {
    let disposed = false;
    getDisplayFFT().then(fft => {
      displayFFTRef.current = fft;
      setFftPath(fft.path);
    });
    // Colormap engine: volume-resident GPU slice + colormap (no CPU per-scrub work).
    createGPUColormapEngine().then(engine => {
      if (disposed) { engine?.destroy(); return; }
      if (engine) { gpuCmapRef.current = engine; setCmapReady(true); }
    });
    return () => { disposed = true; gpuCmapRef.current?.destroy(); gpuCmapRef.current = null; volUploadedKeyRef.current = null; };
  }, []);

  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null]);
  const uiRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null]);
  const imageBoxRefs = React.useRef<(HTMLDivElement | null)[]>([null, null]);

  const [fftColormap, setFftColormap] = useModelState<string>("fft_colormap");
  const [fftLogScale, setFftLogScale] = useModelState<boolean>("fft_log_scale");
  const [fftAuto, setFftAuto] = useModelState<boolean>("fft_auto");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const savedSliceZooms = AXES.map((_, i) => normalizeZoomState(viewState?.zooms?.[i], DEFAULT_ZOOM));
  const savedFftZooms = AXES.map((_, i) => normalizeZoomState(viewState?.fft_zooms?.[i], DEFAULT_FFT_ZOOM));
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
  const fftCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null]);
  const fftOverlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null]);
  const fftOffscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null]);
  const fftImgDataRefs = React.useRef<(ImageData | null)[]>([null, null]);
  const fftMagCacheRefs = React.useRef<(Float32Array | null)[]>([null, null]);
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
  const displayFFTRef = React.useRef<DisplayFFT | null>(null);
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
    alignment?: { rowShift: number; colShift: number; segment: { start: ImagePoint; stop: ImagePoint } };
  } | null>(null);
  const fftComputeGenerationRef = React.useRef(0);
  // Which display FFT the panels use (WebGPU or the CPU reference), once it resolves.
  const [fftPath, setFftPath] = React.useState<RenderPath | null>(null);
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
  const sliceOffscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null]);
  // Reusable ImageData per axis to avoid GC churn (allocated once per dimension change)
  const sliceImgDataRefs = React.useRef<(ImageData | null)[]>([null, null]);

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
  const pendingExportRef = React.useRef<PendingSave | null>(null);

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
    () => extractVolumeFloat32(volumeBytes, offline, offlineMin, offlineMax, numCols, numRows, numSlices, safePanelCount),
    [volumeBytes, offline, offlineMin, offlineMax, numCols, numRows, numSlices, safePanelCount],
  );
  const panelFloats = React.useMemo(() => {
    if (!allPanelFloats || allPanelFloats.length === 0) return null;
    const onePanelCount = Math.max(0, Math.floor(numCols) * Math.floor(numRows) * Math.floor(numSlices));
    if (onePanelCount === 0) return null;
    if (safePanelCount <= 1) return allPanelFloats;
    const start = Math.min(safeActivePanel * onePanelCount, allPanelFloats.length);
    const stop = Math.min(start + onePanelCount, allPanelFloats.length);
    if (stop <= start) return null;
    return allPanelFloats.subarray(start, stop);
  }, [allPanelFloats, safePanelCount, safeActivePanel, numCols, numRows, numSlices]);
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
        start: clampPointToImage(startFromState, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET),
        stop: clampPointToImage(stopFromState, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET),
        explicit: true,
      };
    }
    const [start, stop] = obliqueLineEndpoints(numCols, numRows, colIndex, rowIndex, obliqueAngle);
    return {
      start: clampPointToImage(start, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET),
      stop: clampPointToImage(stop, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET),
      explicit: false,
    };
  }, [obliqueProfileLine, numCols, numRows, colIndex, rowIndex, obliqueAngle]);
  // The oblique panel's geometry must reach the GPU slice shader on EVERY render,
  // not only while alignment is on: the axis-3 shader walks from segment start to
  // stop, so a missing segment collapses every output column onto voxel (slice, 0, 0)
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
  const voxelCount = Math.max(0, Math.floor(numCols) * Math.floor(numRows) * Math.floor(numSlices));
  const syncedVoxelCount = voxelCount * safePanelCount;
  const exactExportSize = formatEstimatedHtmlSize(syncedVoxelCount * 4);
  const quantizedExportSize = formatEstimatedHtmlSize(syncedVoxelCount);
  const handleExportMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setExportMenuAnchor(event.currentTarget);
  };
  const handleExportMenuClose = () => {
    setExportMenuAnchor(null);
  };
  const handleExportSelect = async (encoding: "full" | "uint8") => {
    setExportMenuAnchor(null);
    const filename = makeExportFilename(title, numSlices, numRows, numCols, encoding);
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setExportBusy(true);
    setLocalExportStatus("Choose export location...");
    const handle = await pickSaveHandle(filename, HTML_PICKER_TYPE, setExportBusy, setLocalExportStatus);
    if (handle === undefined) return;
    pendingExportRef.current = { id, filename, handle };
    setLocalExportStatus(`Preparing ${filename}...`);
    setExportRequest(JSON.stringify({ encoding, id, filename, download: true }));
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
    if (!panelFloats || panelFloats.length === 0 || numSlices < 2) {
      if (!offline) requestSliceAlignmentEstimate();
      return;
    }
    alignmentEstimateBusyRef.current = true;
    setLocalAlignmentStatus("Estimating slice alignment...");
    try {
      const estimate = await estimateSliceAlignment(panelFloats, numCols, numRows, numSlices, await getHardwareGPUDevice());
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

  useSaveExportPayload(pendingExportRef, exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest, setExportBusy, setLocalExportStatus);

  // Slice dimensions: [top: numRows x numCols], [oblique: numSlices x diagonal]
  const sliceDims = React.useMemo<[number, number][]>(
    () => [[numRows, numCols], [numSlices, segmentWidth(obliqueSegment.start, obliqueSegment.stop)]],
    [numRows, numCols, numSlices, obliqueSegment],
  );
  type LiveFftSegment = { start: ImagePoint; stop: ImagePoint };
  const liveFftSchedulerRef = React.useRef<{
    pending: { axes: number[]; sliceIndex: number; segment: LiveFftSegment } | null;
    inFlight: boolean;
    timer: number | null;
    lastStartMs: number;
  }>({ pending: null, inFlight: false, timer: null, lastStartMs: 0 });

  const makeFftResultCacheKey = (
    axis: number,
    sliceIndexValue: number,
    segment: LiveFftSegment,
    sliceW: number,
    sliceH: number,
  ) => {
    const geometry = axis === 0
      ? `slice=${Math.round(sliceIndexValue)}`
      : `line=${segment.start.col.toFixed(2)},${segment.start.row.toFixed(2)}:${segment.stop.col.toFixed(2)},${segment.stop.row.toFixed(2)}`;
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
    sliceIndexValue: number,
    segment: LiveFftSegment,
  ): Promise<boolean> => {
    if (!showFft || !panelFloats || panelFloats.length === 0) return false;
    const [sliceH, sliceW] = axis === 0 ? [numRows, numCols] : [numSlices, segmentWidth(segment.start, segment.stop)];
    const cacheKey = makeFftResultCacheKey(axis, sliceIndexValue, segment, sliceW, sliceH);
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
      ? extractTopSlice(panelFloats, numCols, numRows, numSlices, sliceIndexValue)
      : extractOblique(panelFloats, numCols, numRows, numSlices, segment.start, segment.stop);
    const data = fftWindow ? new Float32Array(extracted) : extracted;
    if (fftWindow) applyHannWindow2D(data, sliceW, sliceH);
    const paddedW = nextPow2(sliceW);
    const paddedH = nextPow2(sliceH);
    let real: Float32Array;
    let imag: Float32Array;
    if (displayFFTRef.current) {
      const result = await displayFFTRef.current.fft2D(zeroPadTopLeft(data, sliceW, sliceH, paddedW, paddedH), new Float32Array(paddedW * paddedH), paddedW, paddedH, false);
      real = result.real;
      imag = result.imag;
    } else {
      real = zeroPadTopLeft(data, sliceW, sliceH, paddedW, paddedH);
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
    Promise.all(pending.axes.map(axis => computeLiveFftAxis(axis, pending.sliceIndex, pending.segment)))
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
    options: { sliceIndex?: number; segment?: LiveFftSegment } = {},
  ) => {
    if (!showFft || !panelFloats || panelFloats.length === 0) return;
    const state = liveFftSchedulerRef.current;
    const prevAxes = state.pending?.axes ?? [];
    const mergedAxes = Array.from(new Set([...prevAxes, ...axes])).filter(axis => axis === 0 || axis === 1);
    if (mergedAxes.length === 0) return;
    state.pending = {
      axes: mergedAxes,
      sliceIndex: options.sliceIndex ?? sliceIndex,
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
    sliceOffscreenRefs.current = [null, null];
    sliceImgDataRefs.current = [null, null];
    fftOffscreenRefs.current = [null, null];
    fftImgDataRefs.current = [null, null];
    fftMagCacheRefs.current = [null, null];
    liveFftSchedulerRef.current.pending = null;
    volUploadedKeyRef.current = null;
    gpuVolReadyRef.current = false;
    setFftVersion((version) => version + 1);
  }, [safeActivePanel, volumeBytes]);

  // Canvas sizes. For depth panels, keep the slice-axis scale independent from the
  // oblique profile length; otherwise shortening the profile would secretly
  // magnify the slice axis before the explicit slice_stretch slider is applied.
  // smooth=true → CSS bilinear (auto); smooth=false → nearest-neighbor (pixelated).
  // Overlay canvases (crosshair, scale bar, colorbar, FFT scale bar) use displayH
  // for their pixel buffer to avoid distortion under CSS stretch.
  const canvasSizes = React.useMemo(() => sliceDims.map((dims, axis) => {
    const isDepth = axis > 0;
    const target = isDepth ? sideCanvasTarget : canvasTarget;
    const raster = panelRasterSize(dims, isDepth, target);
    const displayH = isDepth ? Math.min(target, Math.round(raster.h * Math.max(1, sliceStretch))) : raster.h;
    return { ...raster, displayH };
  }), [sliceDims, sideCanvasTarget, canvasTarget, sliceStretch]);
  const dataPointToCanvas = React.useCallback((axis: number, col: number, row: number): { x: number; y: number } => {
    const { w: canvasW, h: canvasH, displayH, scaleX, scaleY } = canvasSizes[axis];
    const stretchY = displayH / canvasH;
    const view = liveZoomsRef.current[axis];
    const centerX = canvasW / 2, centerY = displayH / 2;
    let canvasX = col * scaleX;
    let canvasY = row * scaleY * stretchY;
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
    if (!renderer || !panelFloats || panelFloats.length === 0) return;
    renderer.uploadVolume(panelFloats, numCols, numRows, numSlices);
  }, [panelFloats, numCols, numRows, numSlices, rendererReady]);

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
  const liveSliceParamsRef = React.useRef({ colIndex, rowIndex, sliceIndex });
  const committedSliceParamsRef = React.useRef({ colIndex, rowIndex, sliceIndex });
  const committedSliceParams = committedSliceParamsRef.current;
  if (
    committedSliceParams.colIndex !== colIndex ||
    committedSliceParams.rowIndex !== rowIndex ||
    committedSliceParams.sliceIndex !== sliceIndex
  ) {
    const next = { colIndex, rowIndex, sliceIndex };
    committedSliceParamsRef.current = next;
    liveSliceParamsRef.current = next;
  }

  // Keep render params in ref for direct rAF rendering (bypasses React during drag)
  const volumeRenderParams = {
    ...liveSliceParamsRef.current, numCols, numRows, numSlices,
    opacity: opacityA, brightness: 1.0, slicePlaneMask, slicePlaneOpacity,
    obliqueAngleDeg: obliqueAngle,
    obliqueStartCol: obliqueSegment.start.col,
    obliqueStartRow: obliqueSegment.start.row,
    obliqueEndCol: obliqueSegment.stop.col,
    obliqueEndRow: obliqueSegment.stop.row,
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
    if (!renderer || !panelFloats || panelFloats.length === 0) return;
    renderer.render(volumeRenderParamsRef.current, camera, bgColorRef.current, { sliceStretch, orthographic });
  }, [panelFloats, colIndex, rowIndex, sliceIndex, obliqueAngle, obliqueSegment, numCols, numRows, numSlices, cmap, camera, volumeCanvasSize, themeColors.bg, slicePlaneMask, slicePlaneOpacity, volumeDrag, rendererReady, volTexRange, opacityA, sliceStretch, orthographic, flip, gpuSliceParams]);

  // First-frame paint guard: the very first synchronous render after the renderer
  // mounts can land before the canvas swapchain is ready (flush race) and commit a
  // BLACK frame - the volume then stayed blank until the user dragged. Re-render on
  // the next animation frame (once the context is configured + data uploaded) so the
  // volume is visible on mount, no interaction needed.
  React.useEffect(() => {
    if (!rendererReady) return;
    const id = requestAnimationFrame(() => {
      const renderer = volumeRendererRef.current;
      if (renderer && panelFloats && panelFloats.length > 0) {
        renderer.render(volumeRenderParamsRef.current, camera, bgColorRef.current, { sliceStretch, orthographic });
      }
    });
    return () => cancelAnimationFrame(id);
  }, [rendererReady, panelFloats]);

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
  // Live slice_stretch ref for rAF drag path - keeps latest value without re-binding closure.
  const sliceStretchRef = React.useRef(sliceStretch);
  if (!sliceStretchLiveDirtyRef.current) sliceStretchRef.current = sliceStretch;
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
  React.useEffect(() => { applyDepthPanelHeight(sliceStretch); }, [sliceStretch, rasterCanvasSizes, sideCanvasTarget]);
  const handleSliceStretchChange = (value: number) => {
    sliceStretchLiveDirtyRef.current = true;
    pendingSliceStretchRef.current = value;
    sliceStretchRef.current = value;
    if (sliceStretchRafRef.current != null) return;
    sliceStretchRafRef.current = requestAnimationFrame(() => {
      sliceStretchRafRef.current = null;
      const next = pendingSliceStretchRef.current;
      applyDepthPanelHeight(next);
      const renderer = volumeRendererRef.current;
      if (renderer && panelFloats && panelFloats.length > 0) {
        renderer.render(volumeRenderParamsRef.current, liveCameraRef.current, bgColorRef.current, { sliceStretch: next, orthographic });
      }
    });
  };
  const handleSliceStretchCommit = (value: number) => {
    if (sliceStretchRafRef.current != null) {
      cancelAnimationFrame(sliceStretchRafRef.current);
      sliceStretchRafRef.current = null;
    }
    pendingSliceStretchRef.current = value;
    sliceStretchRef.current = value;
    applyDepthPanelHeight(value);
    sliceStretchLiveDirtyRef.current = false;
    setSliceStretch(value);
    setModelSliceStretch(value);
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
      if (renderer && panelFloats && panelFloats.length > 0) {
        renderer.render(volumeRenderParamsRef.current, liveCameraRef.current, bgColorRef.current, { sliceStretch: sliceStretchRef.current, orthographic });
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
    if (renderer && panelFloats && panelFloats.length > 0) {
      renderer.render(volumeRenderParamsRef.current, liveCameraRef.current, bgColorRef.current, { sliceStretch: sliceStretchRef.current, orthographic });
    }
  };

  // Rotate the cut to `nextAngle` about its centre, keeping its length inside the image.
  const applyObliqueAngle = (nextAngle: number) => {
    const center = {
      row: (obliqueSegment.start.row + obliqueSegment.stop.row) / 2,
      col: (obliqueSegment.start.col + obliqueSegment.stop.col) / 2,
    };
    const length = Math.max(1, Math.hypot(
      obliqueSegment.stop.col - obliqueSegment.start.col,
      obliqueSegment.stop.row - obliqueSegment.start.row,
    ));
    const theta = (nextAngle * Math.PI) / 180;
    const halfColSpan = Math.cos(theta) * length / 2;
    const halfRowSpan = Math.sin(theta) * length / 2;
    const start = clampPointToImage({ row: center.row - halfRowSpan, col: center.col - halfColSpan }, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET);
    const stop = clampPointToImage({ row: center.row + halfRowSpan, col: center.col + halfColSpan }, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET);
    setObliqueAngle(nextAngle);
    setObliqueProfileLine(profileLinePayload(start, stop));
    setObliquePositionBounds(null);
    // Pass the freshly computed segment: the `obliqueSegment` default is a
    // trait-derived memo that still holds the pre-drag geometry on this tick.
    updateObliqueCenter((start.col + stop.col) / 2, (start.row + stop.row) / 2, nextAngle, { start, stop });
    scheduleLiveFft([1], { segment: { start, stop } });
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      obliqueAngleDeg: nextAngle,
      obliqueStartCol: start.col,
      obliqueStartRow: start.row,
      obliqueEndCol: stop.col,
      obliqueEndRow: stop.row,
    };
    const renderer = volumeRendererRef.current;
    if (renderer && panelFloats && panelFloats.length > 0) {
      renderer.render(volumeRenderParamsRef.current, liveCameraRef.current, bgColorRef.current, { sliceStretch: sliceStretchRef.current, orthographic });
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
            renderer.render(params, liveCamera, background, { sliceStretch: sliceStretchRef.current, orthographic });
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
      renderer.render(volumeRenderParamsRef.current, next, bgColorRef.current, { sliceStretch: sliceStretchRef.current, orthographic });
    }
    setCamera(next);
    persistViewState(undefined, undefined, next);
  };

  const handleVolumeDoubleClick = () => {
    liveCameraRef.current = SHOW3DSLICES_DEFAULT_CAMERA;
    setCamera(SHOW3DSLICES_DEFAULT_CAMERA);
    persistViewState(undefined, undefined, SHOW3DSLICES_DEFAULT_CAMERA);
  };

  const setVolumeView = (view: "top" | "side") => {
    const distance = liveCameraRef.current.distance || camera.distance || SHOW3DSLICES_DEFAULT_CAMERA.distance;
    // Match the 2D slice panels rather than mathematical world-up:
    // Top: col right, row down. Side: col right, slice down.
    const presets: Record<"top" | "side", Pick<CameraState, "yaw" | "pitch" | "roll">> = {
      top: { yaw: Math.PI, pitch: 0, roll: Math.PI },
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
  // Per-panel: Top depends on sliceIndex; oblique depends on the (row, col) center and angle.
  // Excludes zoom/pan so dragging only triggers the cheap redraw below.
  // useLayoutEffect so offscreens are ready before the draw useLayoutEffect runs.
  // -------------------------------------------------------------------------
  const prevCacheRef = React.useRef<{
    sliceIndex: number;
    cmap: string; logScale: boolean;
    imageVminPct: number; imageVmaxPct: number;
    imageRangeMin: number; imageRangeMax: number;
    panelFloats: Float32Array | null;
    numCols: number; numRows: number; numSlices: number;
    traitVmin: number | null; traitVmax: number | null;
    flip: boolean;
    alignRowShift: number; alignColShift: number;
    alignStartCol: number; alignStartRow: number; alignStopCol: number; alignStopRow: number;
  }>({ sliceIndex: -1, cmap: "", logScale: false, imageVminPct: -1, imageVmaxPct: -1, imageRangeMin: Number.NaN, imageRangeMax: Number.NaN, panelFloats: null, numCols: 0, numRows: 0, numSlices: 0, traitVmin: null, traitVmax: null, flip: false, alignRowShift: Number.NaN, alignColShift: Number.NaN, alignStartCol: Number.NaN, alignStartRow: Number.NaN, alignStopCol: Number.NaN, alignStopRow: Number.NaN });

  React.useLayoutEffect(() => {
    if (!panelFloats || panelFloats.length === 0) return;

    const prev = prevCacheRef.current;
    const alignRowShift = gpuSliceParams.rowShift;
    const alignColShift = gpuSliceParams.colShift;
    const alignStartCol = gpuSliceParams.segment.start.col;
    const alignStartRow = gpuSliceParams.segment.start.row;
    const alignStopCol = gpuSliceParams.segment.stop.col;
    const alignStopRow = gpuSliceParams.segment.stop.row;
    const globalChanged = panelFloats !== prev.panelFloats || cmap !== prev.cmap ||
      logScale !== prev.logScale ||
      imageVminPct !== prev.imageVminPct || imageVmaxPct !== prev.imageVmaxPct ||
      displayDataRange.min !== prev.imageRangeMin || displayDataRange.max !== prev.imageRangeMax ||
      traitVmin !== prev.traitVmin || traitVmax !== prev.traitVmax ||
      flip !== prev.flip ||
      alignRowShift !== prev.alignRowShift || alignColShift !== prev.alignColShift ||
      alignStartCol !== prev.alignStartCol || alignStartRow !== prev.alignStartRow ||
      alignStopCol !== prev.alignStopCol || alignStopRow !== prev.alignStopRow ||
      numCols !== prev.numCols || numRows !== prev.numRows || numSlices !== prev.numSlices;
    const axisChanged = [
      globalChanged || sliceIndex !== prev.sliceIndex,
      true,
    ];

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const extractors = [
      () => extractTopSlice(panelFloats, numCols, numRows, numSlices, sliceIndex),
      () => extractOblique(panelFloats, numCols, numRows, numSlices, obliqueSegment.start, obliqueSegment.stop),
    ];
    // GPU path: upload the whole volume ONCE; each scrub only slices + colormaps on
    // the GPU (no CPU extract / re-upload), so scrubbing stays buffer-smooth even on
    // a 1688x1688x16 volume. CPU path is the fallback (no engine / volume too big).
    const engine = gpuCmapRef.current;
    let gpuVolReady = false;
    if (cmapReady && engine && panelFloats) {
      if (volUploadedKeyRef.current !== panelFloats) {
        gpuVolReady = engine.uploadVolume(panelFloats, numCols, numRows, numSlices);
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
          axis === 0 ? sliceIndex : 0,
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
    prevCacheRef.current = { sliceIndex, cmap, logScale, imageVminPct, imageVmaxPct, imageRangeMin: displayDataRange.min, imageRangeMax: displayDataRange.max, panelFloats, numCols, numRows, numSlices, traitVmin, traitVmax, flip, alignRowShift, alignColShift, alignStartCol, alignStartRow, alignStopCol, alignStopRow };
  }, [panelFloats, colIndex, rowIndex, sliceIndex, obliqueAngle, obliqueSegment, numCols, numRows, numSlices, cmap, logScale, sliceDims, imageVminPct, imageVmaxPct, displayDataRange, traitVmin, traitVmax, flip, cmapReady, gpuSliceParams]);

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
  // slice sliders are anywidget model traits (col_index, row_index, slice_index) whose setter does a
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
    if (!renderer || !panelFloats || panelFloats.length === 0) return;
    const params = { ...volumeRenderParamsRef.current, ...liveSliceParamsRef.current };
    volumeRenderParamsRef.current = params;
    renderer.render(params, liveCameraRef.current, bgColorRef.current, {
      sliceStretch: sliceStretchRef.current, orthographic, draft: { dpr: 1, numSteps: 32 },
    });
  }, [orthographic, panelFloats]);

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
  }, [panelFloats, colIndex, rowIndex, sliceIndex, obliqueAngle, numCols, numRows, numSlices, cmap, logScale, autoContrast, zooms, sliceDims, canvasSizes, imageVminPct, imageVmaxPct, smooth, flip, gpuSliceParams]);

  // -------------------------------------------------------------------------
  // Render crosshair lines for the orthogonal slice intersections.
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!panelFloats) return;
    const crossPositions: [number, number][] = [
      [colIndex, rowIndex],
      [(sliceDims[1]?.[1] ?? 1) / 2, sliceIndex],
    ];
    // Drawn in CSS pixels on a device-pixel backing store, so lines and handles stay sharp on HiDPI screens.
    const dpr = window.devicePixelRatio || 1;
    for (let axis = 0; axis < sliceDims.length; axis++) {
      const overlay = overlayRefs.current[axis];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      const { w: canvasW, h: canvasH, displayH, scaleX, scaleY } = canvasSizes[axis];
      const stretchY = displayH / canvasH;
      const backingW = Math.round(canvasW * dpr), backingH = Math.round(displayH * dpr);
      if (overlay.width !== backingW || overlay.height !== backingH) {
        overlay.width = backingW;
        overlay.height = backingH;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, canvasW, displayH);
      if (axis === 0) {
        const { start, stop } = obliqueSegment;
        const startCanvas = dataPointToCanvas(axis, start.col, start.row);
        const stopCanvas = dataPointToCanvas(axis, stop.col, stop.row);
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
      const [panelCol, panelRow] = crossPositions[axis];
      const centerX = canvasW / 2, centerY = displayH / 2;
      let canvasX = panelCol * scaleX;
      let canvasY = panelRow * scaleY * stretchY;
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
  }, [panelFloats, colIndex, rowIndex, sliceIndex, obliqueAngle, obliqueSegment, obliqueHoverTarget, zooms, showCrosshair, themeColors, canvasSizes, sliceDims, numCols, numRows, dataPointToCanvas]);

  // -------------------------------------------------------------------------
  // Scale bar (HiDPI UI overlay)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const dpr = window.devicePixelRatio || 1;
    const sampling = panelSampling(voxelSizeA, obliqueAngle);
    for (let axis = 0; axis < sliceDims.length; axis++) {
      const uiCanvas = uiRefs.current[axis];
      if (!uiCanvas) continue;
      const { w: canvasW, displayH } = canvasSizes[axis];
      uiCanvas.width = Math.round(canvasW * dpr);
      uiCanvas.height = Math.round(displayH * dpr);
      const uiCtx = uiCanvas.getContext("2d");
      if (!uiCtx) continue;
      if (showScaleBar) {
        const spacing = sampling[axis].col;
        drawScaleBarHiDPI(uiCanvas, dpr, zooms[axis].zoom, spacing > 0 ? spacing : 1, spacing > 0 ? "Å" : "px", sliceDims[axis][1]);
      }
      if (showColorbar) {
        const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
        // The window the slices are painted with, so the labels follow Flip.
        const { vmin, vmax } = volumeDisplayWindow(displayDataRange, imageVminPct, imageVmaxPct, flip);
        uiCtx.save();
        uiCtx.scale(dpr, dpr);
        drawColorbar(uiCtx, canvasW, displayH, lut, vmin, vmax, logScale);
        uiCtx.restore();
      }
    }
  }, [voxelSizeA, showScaleBar, zooms, canvasSizes, sliceDims, showColorbar, cmap, displayDataRange, imageVminPct, imageVmaxPct, flip, obliqueAngle, themeInfo.theme]);

  // -------------------------------------------------------------------------
  // FFT computation and caching (per-axis: only recompute changed axes)
  // -------------------------------------------------------------------------
  const prevFFTCacheRef = React.useRef<{
    sliceIndex: number;
    panelFloats: Float32Array | null;
    fftColormap: string; fftLogScale: boolean; fftAuto: boolean; fftWindow: boolean; fftPath: RenderPath | null;
    showFft: boolean;
  }>({ sliceIndex: -1, panelFloats: null, fftColormap: "", fftLogScale: false, fftAuto: false, fftWindow: false, fftPath: null, showFft: false });

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
      fftPath !== prevFFT.fftPath || !prevFFT.showFft;
    const fftAxisChanged = [
      globalFFTChanged || sliceIndex !== prevFFT.sliceIndex,
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
        () => extractTopSlice(floats, numCols, numRows, numSlices, sliceIndex),
        () => extractOblique(floats, numCols, numRows, numSlices, obliqueSegment.start, obliqueSegment.stop),
      ];
      for (let axis = 0; axis < sliceDims.length; axis++) {
        if (!forceAll && !fftAxisChanged[axis]) continue;
        const [sliceH, sliceW] = sliceDims[axis];
        const cacheKey = makeFftResultCacheKey(axis, sliceIndex, obliqueSegment, sliceW, sliceH);
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
        if (displayFFTRef.current) {
          const result = await displayFFTRef.current.fft2D(zeroPadTopLeft(data, sliceW, sliceH, paddedW, paddedH), new Float32Array(paddedW * paddedH), paddedW, paddedH, false);
          real = result.real;
          imag = result.imag;
        } else {
          real = zeroPadTopLeft(data, sliceW, sliceH, paddedW, paddedH);
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
      prevFFTCacheRef.current = { sliceIndex, panelFloats, fftColormap, fftLogScale, fftAuto, fftWindow, fftPath, showFft };
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
  }, [showFft, panelFloats, colIndex, rowIndex, sliceIndex, obliqueAngle, obliqueSegment, numCols, numRows, numSlices, sliceDims, fftColormap, fftLogScale, fftAuto, fftWindow, fftPath]);

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
    const dpr = window.devicePixelRatio || 1;
    const sampling = panelSampling(voxelSizeA, obliqueAngle);
    for (let axis = 0; axis < sliceDims.length; axis++) {
      const overlay = fftOverlayRefs.current[axis];
      if (!overlay) continue;
      const { w: canvasW, h: canvasH, displayH } = canvasSizes[axis];
      const stretchY = displayH / canvasH;
      overlay.width = Math.round(canvasW * dpr);
      overlay.height = Math.round(displayH * dpr);
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      if (sampling[axis].col > 0) {
        const paddedW = nextPow2(sliceDims[axis][1]);
        drawFFTScaleBarHiDPI(overlay, dpr, fftZooms[axis].zoom, 1 / (paddedW * sampling[axis].col), paddedW, "Å⁻¹");
      }

      if (fftClickInfo && fftClickInfo.axis === axis) {
        const [sliceH, sliceW] = sliceDims[axis];
        const fftW = nextPow2(sliceW);
        const fftH = nextPow2(sliceH);

        ctx.save();
        ctx.scale(dpr, dpr);
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
  }, [showFft, fftZooms, canvasSizes, sliceDims, voxelSizeA, obliqueAngle, fftClickInfo]);

  // -------------------------------------------------------------------------
  // Playback logic (matching Show3D pattern)
  // -------------------------------------------------------------------------
  const obliquePlaybackStateRef = React.useRef({ current: 0, start: 0, end: 0 });
  const effectiveLoopEnds = React.useMemo(() => loopEnds.map((end, i) => {
    const max = [numSlices - 1, numRows - 1, numCols - 1][i];
    return end < 0 ? max : Math.min(end, max);
  }), [loopEnds, numSlices, numRows, numCols]);
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
        paintAndTrackRef.current!(0, value);
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
    opposite: ImagePoint;
  } | {
    mode: "line";
    angleDeg: number;
    origin: ImagePoint;
    start: ImagePoint;
    stop: ImagePoint;
  } | null>(null);
  const obliquePositionDragRef = React.useRef<{
    angleDeg: number;
    currentOffset: number;
    minOffset: number;
    maxOffset: number;
    start: ImagePoint;
    stop: ImagePoint;
  } | null>(null);
  const [liveObliqueOffset, setLiveObliqueOffset] = React.useState<number | null>(null);
  const [obliqueAngleBounds, setObliqueAngleBounds] = React.useState<[number, number] | null>(null);
  const [obliquePositionBounds, setObliquePositionBounds] = React.useState<[number, number] | null>(null);
  const imagePointFromEvent = (event: React.MouseEvent | MouseEvent, axis: number): ImagePoint | null => {
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
    return { row, col };
  };
  const updateObliqueCenter = (
    centerCol: number,
    centerRow: number,
    angleDeg = obliqueAngle,
    segment: { start: ImagePoint; stop: ImagePoint } = obliqueSegment,
  ) => {
    const nextCol = clampNumber(Math.round(centerCol), 0, numCols - 1);
    const nextRow = clampNumber(Math.round(centerRow), 0, numRows - 1);
    setColIndex(nextCol);
    setRowIndex(nextRow);
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      colIndex: nextCol,
      rowIndex: nextRow,
      obliqueAngleDeg: angleDeg,
      obliqueStartCol: segment.start.col,
      obliqueStartRow: segment.start.row,
      obliqueEndCol: segment.stop.col,
      obliqueEndRow: segment.stop.row,
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
    if (renderer && panelFloats && panelFloats.length > 0) {
      renderer.render(volumeRenderParamsRef.current, liveCameraRef.current, bgColorRef.current, { sliceStretch: sliceStretchRef.current, orthographic });
    }
  };
  const updateObliqueFromEndpoints = (moving: ImagePoint, opposite: ImagePoint) => {
    const start = clampPointToImage(moving, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET);
    const stop = clampPointToImage(opposite, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET);
    const centerCol = clampNumber(Math.round((start.col + stop.col) / 2), 0, numCols - 1);
    const centerRow = clampNumber(Math.round((start.row + stop.row) / 2), 0, numRows - 1);
    const rawAngle = (Math.atan2(start.row - stop.row, start.col - stop.col) * 180) / Math.PI;
    const nextAngle = ((rawAngle % 180) + 180) % 180;
    setObliqueAngle(nextAngle);
    setObliqueProfileLine(profileLinePayload(start, stop));
    setObliquePositionBounds(null);
    updateObliqueCenter(centerCol, centerRow, nextAngle, { start, stop });
    scheduleLiveFft([1], { segment: { start, stop } });
  };
  const updateObliqueFromNormalOffset = (offset: number) => {
    const dragBasis = obliquePositionDragRef.current;
    const angleDeg = dragBasis?.angleDeg ?? obliqueAngle;
    const baseStart = dragBasis?.start ?? obliqueSegment.start;
    const baseStop = dragBasis?.stop ?? obliqueSegment.stop;
    const currentOffset = dragBasis?.currentOffset ?? obliqueCenterOffset(numCols, numRows, angleDeg, baseStart, baseStop);
    const [minDelta, maxDelta] = dragBasis
      ? [dragBasis.minOffset - dragBasis.currentOffset, dragBasis.maxOffset - dragBasis.currentOffset]
      : obliqueSegmentOffsetBounds(numCols, numRows, angleDeg, baseStart, baseStop, OBLIQUE_PROFILE_EDGE_INSET);
    const normal = obliqueNormal(angleDeg);
    const nextOffset = clampNumber(offset, currentOffset + minDelta, currentOffset + maxDelta);
    const delta = nextOffset - currentOffset;
    const colDelta = normal.col * delta;
    const rowDelta = normal.row * delta;
    const start = clampPointToImage({ row: baseStart.row + rowDelta, col: baseStart.col + colDelta }, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET);
    const stop = clampPointToImage({ row: baseStop.row + rowDelta, col: baseStop.col + colDelta }, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET);
    obliquePlaybackStateRef.current = {
      ...obliquePlaybackStateRef.current,
      current: Math.round(nextOffset),
    };
    setLiveObliqueOffset(Math.round(nextOffset));
    setObliqueProfileLine(profileLinePayload(start, stop));
    updateObliqueCenter((start.col + stop.col) / 2, (start.row + stop.row) / 2, angleDeg, { start, stop });
    scheduleLiveFft([1], { segment: { start, stop } });
  };
  const updateObliqueFromLineDrag = (
    drag: {
      angleDeg: number;
      origin: ImagePoint;
      start: ImagePoint;
      stop: ImagePoint;
    },
    point: ImagePoint,
  ) => {
    const colDeltaRaw = point.col - drag.origin.col;
    const rowDeltaRaw = point.row - drag.origin.row;
    const { start, stop } = translateSegmentInsideImage(
      drag.start,
      drag.stop,
      colDeltaRaw,
      rowDeltaRaw,
      numCols,
      numRows,
      OBLIQUE_PROFILE_EDGE_INSET,
    );
    setLiveObliqueOffset(null);
    setObliquePositionBounds(null);
    setObliqueProfileLine(profileLinePayload(start, stop));
    updateObliqueCenter((start.col + stop.col) / 2, (start.row + stop.row) / 2, drag.angleDeg, { start, stop });
    scheduleLiveFft([1], { segment: { start, stop } });
  };
  // What the pointer is over on the Top panel's oblique line: the nearer
  // endpoint handle (screen-space radius around the drawn, edge-clamped handle)
  // or the line body (image-space distance). Null off the Top panel or image.
  const obliqueHitTest = (event: React.MouseEvent, axis: number): { target: "start" | "stop" | "line" | null; point: ImagePoint } | null => {
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
    const startCanvas = dataPointToCanvas(axis, start.col, start.row);
    const stopCanvas = dataPointToCanvas(axis, stop.col, stop.row);
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
    const lineDist = pointToSegmentDistance(point.col, point.row, start.col, start.row, stop.col, stop.row);
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
          origin: hit.point,
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
    // The document listener below moves the cut during a handle drag.
    if (axis === 0 && obliqueHandleDragRef.current) return;
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
    const point = imagePointFromEvent(event, axis);
    if (!point || !panelFloats || panelFloats.length === 0) return;
    const pixelCol = Math.floor(point.col);
    const pixelRow = Math.floor(point.row);
    const [sliceH, sliceW] = sliceDims[axis];
    if (pixelCol < 0 || pixelCol >= sliceW || pixelRow < 0 || pixelRow >= sliceH) {
      setCursorInfoThrottled(null);
      return;
    }
    // 3D voxel lookup. Top is one slice; oblique is a vertical plane through
    // the current (row, col) center, rotated about the slice axis.
    let value: number;
    if (axis === 0) {
      value = panelFloats[sliceIndex * numRows * numCols + pixelRow * numCols + pixelCol];
    } else {
      const denom = Math.max(1, sliceW - 1);
      const t = pixelCol / denom;
      const sampleCol = obliqueSegment.start.col + (obliqueSegment.stop.col - obliqueSegment.start.col) * t;
      const sampleRow = obliqueSegment.start.row + (obliqueSegment.stop.row - obliqueSegment.start.row) * t;
      value = sampleVolumeBilinear(panelFloats, numCols, numRows, numSlices, pixelRow, sampleCol, sampleRow);
    }
    setCursorInfoThrottled({
      row: pixelRow,
      col: pixelCol,
      value: Number.isFinite(value) ? value : Number.NaN,
      view: PANEL_NAMES[axis] ?? "Slice",
    });
  };

  // A handle drag moves the cut from this document listener only, so it keeps
  // going past the panel edge and each pointer move updates the cut once. The
  // listener is bound once; the ref hands it this render's geometry.
  const dragObliqueHandleRef = React.useRef<(event: MouseEvent) => void>(() => {});
  dragObliqueHandleRef.current = (event: MouseEvent) => {
    const drag = obliqueHandleDragRef.current;
    if (!drag) return;
    const point = imagePointFromEvent(event, 0);
    if (!point) return;
    if (drag.mode === "endpoint") {
      updateObliqueFromEndpoints(
        clampPointToImage(point, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET),
        drag.opposite,
      );
    } else {
      updateObliqueFromLineDrag(drag, point);
    }
  };
  React.useEffect(() => {
    const handleDocumentMove = (event: MouseEvent) => dragObliqueHandleRef.current(event);
    const handleDocumentUp = () => {
      obliqueHandleDragRef.current = null;
      obliquePositionDragRef.current = null;
      setObliqueHoverTarget(null);
      setLiveObliqueOffset(null);
    };
    document.addEventListener("mousemove", handleDocumentMove);
    document.addEventListener("mouseup", handleDocumentUp);
    return () => {
      document.removeEventListener("mousemove", handleDocumentMove);
      document.removeEventListener("mouseup", handleDocumentUp);
    };
  }, []);

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
                setRowIndex(pixelRow);
                setColIndex(pixelCol);
              } else {
                const denom = Math.max(1, sliceW - 1);
                const t = pixelCol / denom;
                const sampleCol = obliqueSegment.start.col + (obliqueSegment.stop.col - obliqueSegment.start.col) * t;
                const sampleRow = obliqueSegment.start.row + (obliqueSegment.stop.row - obliqueSegment.start.row) * t;
                const nextCol = Math.round(sampleCol);
                const nextRow = Math.round(sampleRow);
                if (nextCol >= 0 && nextCol < numCols && nextRow >= 0 && nextRow < numRows) {
                  setSliceIndex(pixelRow);
                  setColIndex(nextCol);
                  setRowIndex(nextRow);
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
    const resetZooms = AXES.map(() => DEFAULT_ZOOM);
    const resetFftZooms = AXES.map(() => DEFAULT_FFT_ZOOM);
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
        setSliceIndex(Math.max(0, Math.min(numSlices - 1, sliceIndex + delta)));
        return;
      }
      const state = obliquePlaybackStateRef.current;
      updateObliqueFromNormalOffset(clampNumber(state.current + delta, state.start, state.end));
    };
    const jumpTransportAxis = (axis: number, toEnd: boolean) => {
      event.preventDefault();
      if (axis === 0) {
        setSliceIndex(toEnd ? numSlices - 1 : 0);
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

  // A stationary click measures the d-spacing at the nearest FFT peak. The drag
  // ends before the async peak refinement, so a failed refinement cannot leave
  // the panel panning under a released button.
  const handleFftMouseUp = async (event: React.MouseEvent, axis: number) => {
    const click = fftClickStartRef.current;
    fftClickStartRef.current = null;
    if (fftZoomRafRef.current) { cancelAnimationFrame(fftZoomRafRef.current); fftZoomRafRef.current = 0; }
    commitLiveFftZoomsNow();
    setFftDragAxis(null);
    setFftDragStart(null);
    if (!click || click.axis !== axis || Math.hypot(event.clientX - click.x, event.clientY - click.y) >= 3) return;
    const canvas = fftCanvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const { w: canvasW, h: canvasH } = canvasSizes[axis];
    const view = liveFftZoomsRef.current[axis];
    const [sliceH, sliceW] = sliceDims[axis];
    const fftW = nextPow2(sliceW);
    const fftH = nextPow2(sliceH);
    const mouseX = (event.clientX - rect.left) * (canvasW / rect.width);
    const mouseY = (event.clientY - rect.top) * (canvasH / rect.height);
    const centerX = canvasW / 2, centerY = canvasH / 2;
    let imgCol = ((mouseX - centerX - view.panX) / view.zoom + centerX) / canvasW * fftW;
    let imgRow = ((mouseY - centerY - view.panY) / view.zoom + centerY) / canvasH * fftH;
    if (imgCol < 0 || imgCol >= fftW || imgRow < 0 || imgRow >= fftH) return;
    const cachedMag = fftMagCacheRefs.current[axis];
    if (cachedMag) {
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
    const dcCol = imgCol - fftW / 2;
    const dcRow = imgRow - fftH / 2;
    const distPx = Math.sqrt(dcCol * dcCol + dcRow * dcRow);
    if (distPx < 1) {
      setFftClickInfo(null);
      return;
    }
    let spatialFreq: number | null = null;
    let dSpacing: number | null = null;
    const { row: rowSpacing, col: colSpacing } = panelSampling(voxelSizeA, obliqueAngle)[axis];
    if (rowSpacing > 0 && colSpacing > 0) {
      ({ spatialFrequency: spatialFreq, dSpacing } = reciprocalCoordinatesFromShiftedOffset(dcRow, dcCol, fftH, fftW, rowSpacing, colSpacing));
    }
    setFftClickInfo({ axis, row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
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
  const sliceValues = [sliceIndex, rowIndex, colIndex];
  // Mirror of slice values for the playback loop to read between renders.
  // The loop's `sliceValuesRef.current[0] = value` writes are load-bearing
  // at high fps (>~20): React batches the three index setters so two ticks can fire
  // before the next render reassigns this ref to the new [sliceIndex,rowIndex,colIndex].
  // Without the mutation the second tick reads the stale value and computes the
  // same `next`, freezing playback.
  const sliceValuesRef = React.useRef(sliceValues);
  if (!playing) sliceValuesRef.current = sliceValues;
  const sliceMaxes = [numSlices - 1, numRows - 1, numCols - 1];
  // Live thumb mirror: updates per drag-frame so the thumb tracks, WITHOUT touching
  // the model traits (whose change re-runs the heavy render/layout/crosshair effects).
  // Those effects key on the three indices, so during a drag (traits unchanged) they don't
  // run - only the slider JSX re-renders + directPaintPlane paints the GPU image.
  const [liveSlider, setLiveSlider] = React.useState<number[]>([sliceIndex, rowIndex, colIndex]);
  const liveSliderRef = React.useRef<number[]>([sliceIndex, rowIndex, colIndex]);
  const pendingPaintRef = React.useRef<Map<number, number>>(new Map());
  const sliderPaintRafRef = React.useRef<number | null>(null);
  const pendingContrastRangeRef = React.useRef<[number, number] | null>(null);
  const contrastPaintRafRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    const next = [sliceIndex, rowIndex, colIndex];
    liveSliderRef.current = next;
    setLiveSlider(next);
  }, [sliceIndex, rowIndex, colIndex]);
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
  const previewTopPlane = (topSliceIndex: number) => {
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
    const slice = extractTopSlice(panelFloats, numCols, numRows, numSlices, topSliceIndex);
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
    liveSliceParamsRef.current = { sliceIndex: next[0], rowIndex: next[1], colIndex: next[2] };
    volumeRenderParamsRef.current = { ...volumeRenderParamsRef.current, ...liveSliceParamsRef.current };
    if (axis === 0) {
      scheduleLiveFft([0], { sliceIndex: next[0] });
    } else if (!obliqueSegment.explicit) {
      const [rawStart, rawStop] = obliqueLineEndpoints(numCols, numRows, next[2], next[1], obliqueAngle);
      const start = clampPointToImage(rawStart, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET);
      const stop = clampPointToImage(rawStop, numCols, numRows, OBLIQUE_PROFILE_EDGE_INSET);
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
    const [liveSliceIndex, liveRowIndex, liveColIndex] = liveSliderRef.current;
    if (sliceIndex !== liveSliceIndex) setSliceIndex(liveSliceIndex);
    if (rowIndex !== liveRowIndex) setRowIndex(liveRowIndex);
    if (colIndex !== liveColIndex) setColIndex(liveColIndex);
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
        setSliceIndex(start);
      } else {
        const start = obliquePlaybackStateRef.current.start;
        obliquePlaybackStateRef.current = { ...obliquePlaybackStateRef.current, current: start };
        updateObliqueFromNormalOffset(start);
      }
    }
    sliceValuesRef.current = next;
  };
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
    [setSliceIndex, setRowIndex, setColIndex][axis](values[1]);
  };
  const handleLoopSliderPointerDownCapture = (axis: number, event: React.PointerEvent<HTMLSpanElement>) => {
    if (event.button !== 0) return;
    const max = sliceMaxes[axis];
    trackSliderPress(event, (fraction) => Math.max(0, Math.min(max, Math.round(fraction * max))), (value, released) => {
      paintAndTrackRef.current?.(axis, value);
      if (released) [setSliceIndex, setRowIndex, setColIndex][axis](value);
    });
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

  // Few-slice layout: depth axis much smaller than lateral. Show the top panel
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
    if (Math.round(nextAngle) !== Math.round(obliqueAngle)) applyObliqueAngle(nextAngle);
  };
  const handleObliqueAngleSliderChange = (values: number[]) => {
    setObliqueAngleBounds([values[0], values[2]]);
    updateObliqueAngleWithinBounds(values[1], values[0], values[2]);
  };
  const handleObliqueAnglePointerDownCapture = (event: React.PointerEvent<HTMLSpanElement>) => {
    if (event.button !== 0) return;
    pausePlaybackForEdit();
    trackSliderPress(event, (fraction) => Math.round(clampNumber(
      obliqueAngleSliderMin + fraction * (obliqueAngleSliderMax - obliqueAngleSliderMin), obliqueAngleMinBound, obliqueAngleMaxBound,
    )), (value) => updateObliqueAngleWithinBounds(value));
  };
  const obliqueCurrentOffset = obliqueCenterOffset(numCols, numRows, obliqueAngle, obliqueSegment.start, obliqueSegment.stop);
  // Plane center in image coordinates. Position measures along the plane normal,
  // which turns with Angle, so its number moves on rotation even when the cut
  // does not; the center is stated in fixed image pixels instead.
  const obliqueCenterCol = (obliqueSegment.start.col + obliqueSegment.stop.col) / 2;
  const obliqueCenterRow = (obliqueSegment.start.row + obliqueSegment.stop.row) / 2;
  const [obliqueDeltaMin, obliqueDeltaMax] = obliqueSegmentOffsetBounds(
    numCols,
    numRows,
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
    const currentOffset = obliqueCenterOffset(numCols, numRows, obliqueAngle, obliqueSegment.start, obliqueSegment.stop);
    const [minDelta, maxDelta] = obliqueSegmentOffsetBounds(numCols, numRows, obliqueAngle, obliqueSegment.start, obliqueSegment.stop, OBLIQUE_PROFILE_EDGE_INSET);
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
    trackSliderPress(event, (fraction) => Math.round(clampNumber(
      obliquePositionSliderMin + fraction * (obliquePositionSliderMax - obliquePositionSliderMin), obliquePositionMinBound, obliquePositionMaxBound,
    )), (value, released) => {
      updateObliqueFromNormalOffset(value);
      if (released) endObliquePositionDrag();
    });
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
    Math.ceil(Math.max(numCols, numRows) / Math.max(1, numSlices - 1)),
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
          {...themedMenuProps}
          slotProps={{ ...themedMenuProps.slotProps, list: { "aria-label": "Export standalone HTML options" } }}
        >
          <MenuItem onClick={() => handleExportSelect("full")}>Exact float32 ({exactExportSize})</MenuItem>
          <MenuItem onClick={() => handleExportSelect("uint8")}>Quantized uint8 ({quantizedExportSize})</MenuItem>
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
              ["Shape", `${numSlices} x ${numRows} x ${numCols}`] as [string, React.ReactNode],
              ["Axes", Array.isArray(dimLabels) && dimLabels.length ? dimLabels.join(", ") : "slice, row, col"] as [string, React.ReactNode],
              ["Sampling", voxelSizeA.some((spacing) => spacing > 0)
                ? `${voxelSizeA.map((spacing) => formatNumber(spacing)).join(" x ")} Å/px`
                : "not set, lengths in px"] as [string, React.ReactNode],
              ["Display", `z stretch ${formatNumber(sliceStretch)}, ${orthographic ? "orthographic" : "perspective"}`] as [string, React.ReactNode],
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
                  aria-label={`3D volume rendering: ${displayTitle} (${numCols} by ${numRows} by ${numSlices} voxels). Drag to rotate, wheel to zoom.`}
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
              {/* The ray caster samples an 8-bit texture of the volume (r8unorm filters on every adapter). */}
              <Typography sx={{ ...controlLabel, color: themeColors.textMuted }}>
                3D view samples an 8-bit copy; the slice panels show the float32 values.
              </Typography>
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
              {/* displayH is the slice-stretched height on depth panels. */}
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
                    ? `XY slice ${liveSlider[0] + 1} of ${numSlices} along ${axisLabels[0]} axis${title ? `: ${title}` : ""} (${canvasW} by ${canvasH} px)`
                    : `Oblique vertical slice at ${obliqueAngle.toFixed(1)} degrees, position ${Math.round(obliqueCurrentOffset)}${title ? `: ${title}` : ""} (${canvasW} by ${canvasH} px)`}
                />
                <canvas
                  ref={(el) => { overlayRefs.current[axis] = el; }}
                  style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: displayH, pointerEvents: "none" }}
                  aria-hidden="true"
                />
                <canvas
                  ref={(el) => { uiRefs.current[axis] = el; }}
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
                        {`FFT ${axis === 0 ? "Top" : `oblique ${obliqueAngle.toFixed(1)}°`}${fftPath === "CPU" ? " (CPU fallback)" : ""}`}
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
                      aria-label={`FFT power spectrum of ${panelName} slice (reciprocal space, ${canvasW} by ${canvasH} px)`}
                    />
                    <canvas
                      ref={(el) => { fftOverlayRefs.current[axis] = el; }}
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
                        applyObliqueAngle shortens the segment and does move
                        the midpoint, so the center shifts there too. */}
                    <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 14 }}>
                      <Typography
                        sx={{ ...controlLabel, color: themeColors.textMuted, flexShrink: 0, minWidth: 42 }}
                        title="Plane center (row, col) in image px. Steady under rotation unless the cut is clamped at an image edge."
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
                    aria-label={`Loop range and current top slice (${liveSlider[axis] + 1} of ${sliceMaxes[axis] + 1}, loop ${loopStarts[axis] + 1} to ${effectiveLoopEnds[axis] + 1})`}
                    valueLabelFormat={(value) => `${value as number}`}
                  />
                ) : (
                  <Slider
                    value={liveSlider[axis]}
                    min={0}
                    max={sliceMaxes[axis]}
                    onChange={(_, value) => paintAndTrackRef.current!(0, value as number)}
                    onChangeCommitted={(_, value) => setSliceIndex(value as number)}
                    size="small"
                    sx={{ ...sliderStyles.small, flex: 1, minWidth: 40 }}
                    aria-label={`Top slice ${liveSlider[axis] + 1} of ${sliceMaxes[axis] + 1}`}
                    valueLabelDisplay="off"
                    valueLabelFormat={(value) => `${value as number}`}
                  />
                )}
                {/* Slices count from 1 here, as in the slider's accessible name; the slice_index trait stays 0-based. */}
                <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 28, textAlign: "right", flexShrink: 0 }}>
                  {liveSlider[axis] + 1}/{sliceMaxes[axis] + 1}
                </Typography>
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
              <LiveNumberSlider value={sliceStretch} min={1} max={50} step={0.5} onLiveChange={handleSliceStretchChange} onCommit={handleSliceStretchCommit} sx={{ ...sliderStyles.small, width: 80, mr: 1, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }} ariaLabel="Depth axis display stretch multiplier" />
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
                    ariaLabel={`Row shift per slice ${liveRowShift.toFixed(3)} px`}
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
                    ariaLabel={`Column shift per slice ${liveColShift.toFixed(3)} px`}
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
