/// <reference types="@webgpu/types" />
/**
 * Show3D - Interactive 3D stack viewer with playback controls.
 *
 * Features:
 * - Scroll to zoom, double-click to reset
 * - Adjustable ROI size via slider
 * - FPS slider control
 * - WebGPU-accelerated FFT
 * - Equal-sized FFT and histogram panels
 * - Automatic theme detection (light/dark mode)
 */

import * as React from "react";
import { sliderStyles } from "../controlStyles";
import { PlayPauseButton } from "../PlayPauseButton";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import IconButton from "@mui/material/IconButton";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import Badge from "@mui/material/Badge";
import TextField from "@mui/material/TextField";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import FastForwardIcon from "@mui/icons-material/FastForward";
import StopIcon from "@mui/icons-material/Stop";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import DragIndicatorIcon from "@mui/icons-material/DragIndicator";
import { useTheme } from "../theme";
import { useCanvasRepaintSignal } from "../canvasLifecycle";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawScaleBarInRegion, drawZoomIndicatorInRegion, drawColorbar, formatZoomLabel, roundToNiceValue, unitSymbol } from "../figure";
import {
  applyStandaloneWidgetViewState,
  downloadBlob,
  extractBytes,
  extractFloat32,
  formatNumber,
  preserveRestoredWidgetModelsOnSave,
  standaloneHtmlWithCurrentWidgetState,
  standaloneWidgetStaticHtmlFromDocument,
} from "../format";
import { useHideStaticFallback } from "../staticFallback";
import { findDataRange, applyLogScale, applyLogScaleInPlace, percentileClip, signedLog1p, sliderRange, computeStats } from "../display/stats";
import { MetadataSection } from "../widgetInfo";
import { FolderWatchBadge, useFolderWatchModelLive } from "../folderWatchStatus";
import { applyFrequencyFilterBrowser, frequencyFilterActive, getFrequencyFilterBackend, normalizeFrequencyFilterMode } from "../display/frequencyFilter";
import { encodeIndexedGif, quantizeRgbaForBrowserGif } from "./gif";
import { missingDisplayStackStatus } from "./displayStack";
import {
  MAX_PLAYBACK_FPS,
  clampPlaybackFps,
  frameKeyTarget,
  orderedFramePrewarmIndices,
  playbackIntervalMs,
  renderFramePlayback,
  renderFrameScaledPlayback,
  renderPackedPanelPlayback,
  writeLiveFrameControls,
} from "./playback";
import { type SubpixelShift, estimateSubpixelShift, finiteMedianSample, shiftFrameBilinear } from "./registration";
import { type ROIItem, ROI_COLORS, computeROIPixelStats, createROI, normalizeROI } from "./roi";
import {
  ANIMATION_QUALITY_OPTIONS,
  type AnimationQuality,
  DEFAULT_ANIMATION_EXPORT_FPS,
  EXPORT_SPATIAL_OPTIONS,
  type ExportPanelMode,
  type ExportSpatialPreset,
  GIF_EXPORT_PRESETS,
  type GifExportPreset,
  MIN_ANIMATION_OVERLAY_MARGIN_PX,
  MIN_ANIMATION_SCALE_BAR_THICKNESS_PX,
  MIN_ANIMATION_SCALE_FONT_PX,
  MIN_ANIMATION_TITLE_FONT_PX,
  animationOutputScale,
  buildAnimationFrameIndices,
  exportBlobType,
  exportPickerType,
  formatEstimatedAnimationWork,
  makeExportFilename,
  spatialOptionFor,
} from "./exportOptions";
import { formatLength, readableLength } from "./lengthLabel";
import { InfoTooltip } from "../shared/InfoTooltip";
import { RenderPathBadge } from "../shared/RenderPathBadge";
import { KeyboardShortcuts } from "../shared/KeyboardShortcuts";
import { useMobileViewport } from "../shared/useMobileViewport";
import { formatEstimatedHtmlSize, formatSavedBytes, isAbortLikeError } from "../shared/exportFormat";
import { pointToSegmentDistance } from "../shared/geometry";
import { WIDGET_TEXT_OR_VALUE_CONTROL_SELECTOR, shouldIgnoreWidgetShortcut } from "../shared/widgetShortcuts";
import { resolveDisplayBounds } from "../shared/displayRange";
import { type RichTitleSpan, renderMathExpression, renderRichTitle, richTitlePlainText } from "../shared/latexTitle";
import { type PanelAnnotationSpec, type PanelOverlaySpec, type OverlaySelection, type OverlayDragState, panelAnnotationSx, drawROI, drawPanelOverlays, panelOverlayHit, updateOverlayFromDrag, drawPanelOverlaySelection, normalizeHiddenPageSlots } from "../shared/panelOverlays";
import { Histogram } from "../shared/Histogram";
import { normalizeFilterMode, applyDisplayFilterBrowser, browserFilterSupported, filterKnobsActive } from "../display/filter";
import { COLORMAPS, COLORMAP_NAMES, applyColormap, renderToOffscreen, renderToOffscreenReuse, createGPUColormapEngine, GPUColormapEngine } from "../display/colormaps";
import { histogramBins } from "../display/cpuColormap";
import { DisplayFFT, getDisplayFFT, getGPUInfo, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2, applyHannWindow2D, reciprocalCoordinatesFromShiftedOffset } from "../display/fft";
import {
  cropMaskedRegionBrowser,
  findFFTPeakBrowser,
  sampleLineProfileBrowser,
  sampleLineProfileUint8Browser,
} from "../display/geometry";
import { dequantizeUint8 } from "../display/quantization";
import { computeFftQualityMetrics, formatFftQualityLabel, summarizeFftQualityMetrics, type FftQualityMetrics } from "../display/fftMetrics";
import {
  browserFilterCacheKey,
  normalizedAverageWindow,
  temporalAverageFrameIndices,
  requiresClientFrameTransform,
  shouldApplyClientDifference,
} from "./frameTransform";


const SHOW3D_STANDALONE_VIEW_STATE_KEYS = [
  "auto_contrast",
  "avg_window",
  "blink_fps",
  "bookmarked_frames",
  "boomerang",
  "cmap",
  "compare_background",
  "compare_mode",
  "compare_pair",
  "contrast_preset",
  "controls_collapsed",
  "denoise",
  "denoise_bin",
  "denoise_bins",
  "denoise_enabled",
  "denoise_modes",
  "denoise_scope",
  "denoise_sigma",
  "denoise_sigmas",
  "diff_cmap",
  "diff_mode",
  "fft_layout",
  "fft_overlay_position",
  "fft_overlay_size",
  "fft_overlay_zoom",
  "fft_window",
  "flip_horizontal",
  "flip_vertical",
  "fps",
  "frame_rotations",
  "frequency_filter",
  "frequency_filter_center",
  "frequency_filter_centers",
  "frequency_filter_cutoff",
  "frequency_filter_cutoffs",
  "frequency_filter_enabled",
  "frequency_filter_modes",
  "frequency_filter_scope",
  "frequency_filter_width",
  "frequency_filter_widths",
  "hidden_page_slots",
  "hidden_panels",
  "image_rotation",
  "image_vmax_pct",
  "image_vmin_pct",
  "link_contrast",
  "link_panels",
  "log_scale",
  "loop",
  "loop_end",
  "loop_start",
  "max_cols",
  "page_idx",
  "panel_annotations",
  "panel_cmaps",
  "inter_panel_gap_px",
  "panel_gap",
  "panel_order",
  "panel_overlays",
  "panel_title_spans",
  "percentile_high",
  "percentile_low",
  "playback_path",
  "playing",
  "profile_line",
  "profile_width",
  "roi_active",
  "roi_list",
  "roi_selected_idx",
  "rotation_scope",
  "scale_bar_visible",
  "selected_panels",
  "show_controls",
  "show_denoise",
  "show_fft",
  "show_frequency_filter",
  "show_kymograph",
  "show_panel_titles",
  "show_resize_handles",
  "show_stats",
  "show_title",
  "show_zoom_indicator",
  "slice_idx",
  "smooth",
  "starred",
  "subpixel_align_enabled",
  "subpixel_align_reference",
  "view_state",
  "vmax",
  "vmax_per_panel",
  "vmin",
  "vmin_per_panel",
] as const;
// ============================================================================
// Style tokens (inlined - matches Show2D/Show4DSTEM single-file convention)
// ============================================================================
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const UI_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const controlRow = {
  display: "flex",
  alignItems: "center",
  flexWrap: "wrap",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
  maxWidth: "100%",
  boxSizing: "border-box",
} as const;
const compactButton = {
  borderRadius: 0,
  fontSize: 10,
  fontFamily: "inherit",
  textTransform: "none" as const,
  letterSpacing: 0,
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

const PAGE_PLAY_FPS_OPTIONS = [1, 2, 3, 4] as const;
const CONTRAST_PRESETS = [
  { value: "custom", label: "Custom", low: 0, high: 100 },
  { value: "0.5-99.5", label: "0.5–99.5", low: 0.5, high: 99.5 },
  { value: "1-99", label: "1–99", low: 1, high: 99 },
  { value: "2-98", label: "2–98", low: 2, high: 98 },
  { value: "3-97", label: "3–97", low: 3, high: 97 },
  { value: "5-95", label: "5–95", low: 5, high: 95 },
  { value: "10-90", label: "10–90", low: 10, high: 90 },
] as const;
const OFFLINE_FRAME_CACHE_BYTES = 2 * 1024 * 1024 * 1024;
const OFFLINE_FRAME_CACHE_MIN_FRAMES = 2;
// The frame the browser shows before the embedded stack arrives.
const EMPTY_FRAME = new DataView(new ArrayBuffer(0));
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: UI_FONT },
  title: { fontWeight: "bold" as const },
};
type FftOverlayPosition = "top-left" | "top-right" | "bottom-left" | "bottom-right";
type ReorderPlacement = "before" | "after";
type ReorderDragVisual = {
  panel: number;
  label: string;
  imageUrl: string;
  width: number;
  height: number;
  x: number;
  y: number;
  offsetX: number;
  offsetY: number;
};

function renderPanelAnnotation(spec: PanelAnnotationSpec, fallback = ""): React.ReactNode {
  if (spec.math) return renderMathExpression(spec.math, "panel-annotation-math");
  return renderRichTitle(spec.spans, spec.text || fallback);
}

type ReorderDragStart = {
  x: number;
  y: number;
};
const REORDER_DRAG_THRESHOLD_PX = 8;

// ============================================================================
// Inlined utilities (matches Show2D/Show4DSTEM single-file convention)
// ============================================================================
const signedExpm1 = (x: number): number => x >= 0 ? Math.expm1(x) : -Math.expm1(-x);

type Show3DWritableFile = {
  write: (data: BlobPart) => Promise<void>;
  close: () => Promise<void>;
};

type Show3DFileHandle = {
  createWritable: () => Promise<Show3DWritableFile>;
};

type Show3DSavePickerOptions = {
  suggestedName?: string;
  types?: { description: string; accept: Record<string, string[]> }[];
};

type Show3DWindow = Window & typeof globalThis & {
  showSaveFilePicker?: (options?: Show3DSavePickerOptions) => Promise<Show3DFileHandle>;
};

type PanelStats = {
  panel: number;
  mean: number;
  min: number;
  max: number;
  std: number;
};

type CursorInfo = {
  row: number;
  col: number;
  value: number;
  panelIdx: number;
};

/**
 * Float32 view of frame `frameIdx` in a packed (frames, pixels) float32 stack,
 * or null when the frame runs past the buffer. It copies only when the byte
 * offset is not 4-byte aligned (a Float32Array view requires alignment) or
 * when the caller asks for an owned array.
 */
function float32FrameFromDataView(stack: DataView, frameIdx: number, pixelCount: number, copy: boolean): Float32Array | null {
  const byteStart = frameIdx * pixelCount * 4;
  const byteLength = pixelCount * 4;
  if (byteStart < 0 || byteStart + byteLength > stack.byteLength) return null;
  const byteOffset = stack.byteOffset + byteStart;
  let view: Float32Array;
  if (byteOffset % 4 === 0) {
    view = new Float32Array(stack.buffer, byteOffset, pixelCount);
  } else {
    const bytes = new Uint8Array(stack.buffer, byteOffset, byteLength);
    const aligned = new Uint8Array(byteLength);
    aligned.set(bytes);
    view = new Float32Array(aligned.buffer);
  }
  return copy ? new Float32Array(view) : view;
}

/**
 * Rec. 709 luminance, Y = 0.2126 R + 0.7152 G + 0.0722 B, of an interleaved RGB
 * frame. True-color stacks paint in color, but stats, ROI and FFT need one
 * value per pixel.
 */
function rgbFrameToLuminance(rgb: Float32Array, pixelCount: number): Float32Array {
  const luminance = new Float32Array(pixelCount);
  const pixelTotal = Math.min(pixelCount, Math.floor(rgb.length / 3));
  for (let k = 0; k < pixelTotal; k++) {
    luminance[k] = 0.2126 * rgb[3 * k] + 0.7152 * rgb[3 * k + 1] + 0.0722 * rgb[3 * k + 2];
  }
  return luminance;
}

const clampPct = (pct: number): number => Math.max(0, Math.min(100, pct));
const valueToPct = (value: number | null | undefined, min: number, max: number, fallback: number): number => {
  if (value == null || !Number.isFinite(value) || max <= min) return fallback;
  return clampPct(((value - min) / (max - min)) * 100);
};
const pctToValue = (pct: number, min: number, max: number): number => min + (max - min) * (clampPct(pct) / 100);
/**
 * Snap an FFT click to the brightest magnitude within `radius` of (row, col),
 * then refine it to the magnitude-weighted centroid of the 3x3 around that
 * peak. The min/max bounds keep a click near a tile edge from snapping into the
 * neighboring panel's FFT.
 */
function findFFTPeakInBounds(
  magnitude: Float32Array, width: number, height: number,
  col: number, row: number, radius: number,
  minCol: number, maxCol: number, minRow: number, maxRow: number,
): { row: number; col: number } {
  const colStart = Math.max(0, minCol, Math.floor(col) - radius);
  const rowStart = Math.max(0, minRow, Math.floor(row) - radius);
  const colEnd = Math.min(width - 1, maxCol, Math.floor(col) + radius);
  const rowEnd = Math.min(height - 1, maxRow, Math.floor(row) + radius);
  let bestCol = Math.round(col), bestRow = Math.round(row), bestValue = -Infinity;
  for (let candidateRow = rowStart; candidateRow <= rowEnd; candidateRow++) {
    for (let candidateCol = colStart; candidateCol <= colEnd; candidateCol++) {
      const value = magnitude[candidateRow * width + candidateCol];
      if (value > bestValue) { bestValue = value; bestCol = candidateCol; bestRow = candidateRow; }
    }
  }
  const windowColStart = Math.max(0, minCol, bestCol - 1), windowColEnd = Math.min(width - 1, maxCol, bestCol + 1);
  const windowRowStart = Math.max(0, minRow, bestRow - 1), windowRowEnd = Math.min(height - 1, maxRow, bestRow + 1);
  let weightSum = 0, weightedColSum = 0, weightedRowSum = 0;
  for (let candidateRow = windowRowStart; candidateRow <= windowRowEnd; candidateRow++) {
    for (let candidateCol = windowColStart; candidateCol <= windowColEnd; candidateCol++) {
      const weight = magnitude[candidateRow * width + candidateCol];
      weightSum += weight; weightedColSum += weight * candidateCol; weightedRowSum += weight * candidateRow;
    }
  }
  if (weightSum > 0) return { row: weightedRowSum / weightSum, col: weightedColSum / weightSum };
  return { row: bestRow, col: bestCol };
}

/**
 * Display range from the stack range (or the vmin/vmax traits when set), in
 * log space when logScale is on, narrowed by the histogram slider percentages.
 */
function resolveDisplayRange(
  dataMin: number, dataMax: number,
  traitVmin: number | null | undefined, traitVmax: number | null | undefined,
  logScale: boolean, vminPct: number, vmaxPct: number,
): { vmin: number; vmax: number } {
  const baseMin = logScale ? signedLog1p(traitVmin ?? dataMin) : (traitVmin ?? dataMin);
  const baseMax = logScale ? signedLog1p(traitVmax ?? dataMax) : (traitVmax ?? dataMax);
  return sliderRange(baseMin, baseMax, vminPct, vmaxPct);
}

/**
 * The precomputed auto-contrast range of frame `idx` (auto_vmins/auto_vmaxs or
 * the local cache), or null when missing or degenerate. Reusing it keeps a
 * scrub from running a percentile pass on every frame.
 */
function cachedAutoRange(
  vmins: number[] | null | undefined,
  vmaxs: number[] | null | undefined,
  idx: number,
): { vmin: number; vmax: number } | null {
  const vmin = vmins?.[idx];
  const vmax = vmaxs?.[idx];
  if (typeof vmin !== "number" || typeof vmax !== "number") return null;
  return Number.isFinite(vmin) && Number.isFinite(vmax) && vmax > vmin ? { vmin, vmax } : null;
}

/** cachedAutoRange in the display domain: through signedLog1p when log scale is on. */
function cachedAutoDisplayRange(
  vmins: number[] | null | undefined,
  vmaxs: number[] | null | undefined,
  idx: number,
  logScale: boolean,
): { vmin: number; vmax: number } | null {
  const range = cachedAutoRange(vmins, vmaxs, idx);
  if (!range) return null;
  if (!logScale) return range;
  return { vmin: signedLog1p(range.vmin), vmax: signedLog1p(range.vmax) };
}

function sameNumberArray(a: number[] | undefined, b: number[]): boolean {
  if (!Array.isArray(a) || a.length !== b.length) return false;
  return a.every((value, idx) => value === b[idx]);
}

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const container = {
  // Match the shared canvas scale-bar typography for a clean microscope-viewer UI.
  root: {
    p: 2,
    bgcolor: "transparent",
    color: "inherit",
    fontFamily: UI_FONT,
    overflow: "visible",
    "& .MuiTypography-root, & .MuiButton-root, & .MuiInputBase-root": { fontFamily: "inherit" },
  },
  imageBox: { bgcolor: "transparent", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

const DPR = window.devicePixelRatio || 1;
const RESIZE_HIT_AREA_PX = 10;
const ENABLE_GPU_CANVAS_DISPLAY = true;

/** 0xRRGGBB integer from "#rgb" or "#rrggbb"; the WebGPU grid shader takes the gap color packed. */
function packedRgbFromHex(color: string): number {
  const raw = (color.startsWith("#") ? color.slice(1) : color).trim();
  const expanded = raw.length === 3
    ? raw.split("").map(ch => ch + ch).join("")
    : raw.slice(0, 6);
  const parsed = Number.parseInt(expanded, 16);
  return Number.isFinite(parsed) ? parsed & 0xFFFFFF : 0;
}

function clonePanelOverlays(overlays: PanelOverlaySpec[][] | undefined): PanelOverlaySpec[][] {
  return (overlays || []).map((items) => (items || []).map((item) => ({ ...item })));
}

const FFT_SNAP_RADIUS = 5;

// ============================================================================
// Constants
// ============================================================================
const CANVAS_TARGET_SIZE = 600;
const MAX_INTERACTIVE_GRID_CANVAS_EDGE = 4096;
const MAX_INTERACTIVE_GRID_CANVAS_PIXELS = 8_388_608;
const MAX_PANEL_COLUMNS = 12;
const FFT_OVERLAY_MAX_SOURCE_SIZE = 512;
const FFT_PLAYBACK_UPDATE_INTERVAL_MS = 250;
const MIN_ZOOM = 0.5;
const MIN_IMAGE_ZOOM = 1;
const MAX_ZOOM = 30;

/** Top margin of the FFT inset: below the panel title on a multi-panel grid so the inset does not cover it. */
function fftOverlayTopInsetPad(
  insetPad: number,
  showPanelTitles: boolean | undefined,
  panelCount: number,
  panelTitleFontSize: number | undefined,
): number {
  if (showPanelTitles === false || panelCount <= 1) return insetPad;
  const titleClearance = 6 + Math.max(14, (panelTitleFontSize || 11) * 1.35);
  return Math.max(insetPad, titleClearance);
}

/** Subtract the azimuthal mean around the center from an fftshifted magnitude tile, in place. */
function suppressFftRadialBackgroundInPlace(data: Float32Array, width: number, height: number): void {
  if (width < 16 || height < 16 || data.length !== width * height) return;
  const centerCol = Math.floor(width / 2);
  const centerRow = Math.floor(height / 2);
  const maxRadius = Math.ceil(Math.hypot(Math.max(centerCol, width - centerCol), Math.max(centerRow, height - centerRow)));
  const sums = new Float64Array(maxRadius + 1);
  const counts = new Uint32Array(maxRadius + 1);

  for (let row = 0; row < height; row++) {
    const rowFromCenter = row - centerRow;
    const offset = row * width;
    for (let col = 0; col < width; col++) {
      const radius = Math.min(maxRadius, Math.floor(Math.hypot(col - centerCol, rowFromCenter)));
      sums[radius] += data[offset + col];
      counts[radius]++;
    }
  }

  for (let radius = 0; radius <= maxRadius; radius++) {
    if (counts[radius] > 0) sums[radius] /= counts[radius];
  }

  // Display-only whitening: remove the smooth radial pedestal so Bragg spots and
  // lattice peaks remain visible in small FFT overlays without changing the
  // underlying magnitude data used for measurements.
  for (let row = 0; row < height; row++) {
    const rowFromCenter = row - centerRow;
    const offset = row * width;
    for (let col = 0; col < width; col++) {
      const radius = Math.min(maxRadius, Math.floor(Math.hypot(col - centerCol, rowFromCenter)));
      data[offset + col] -= sums[radius];
    }
  }
}


// ============================================================================
// Main Component
// ============================================================================
function Show3D() {
  const isMobileViewport = useMobileViewport();
  const canvasRepaintSignal = useCanvasRepaintSignal();
  const model = useModel();
  const folderWatchLive = useFolderWatchModelLive(model);
  React.useLayoutEffect(() => applyStandaloneWidgetViewState(model), [model]);
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);

  // Theme detection (offline HTML exports force a light/white background)
  const [offlineForTheme] = useModelState<boolean>("_export_light");
  const { themeInfo, colors: baseColors } = useTheme(offlineForTheme);
  const themeColors = {
    ...baseColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#b08800",
  };
  const mobileControlRowSx = isMobileViewport
    ? ({ columnGap: "8px", rowGap: "4px", px: 0.75, py: 0.25 } as const)
    : ({} as const);

  // Theme-aware select style (matching Show4DSTEM)
  const themedSelect = {
    ...controlPanel.select,
    borderRadius: 0,
    fontFamily: "inherit",
    flexShrink: 0,  // never compress a dropdown below its width -> no truncated label
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiSelect-select": { py: 0.5, fontFamily: "inherit", textOverflow: "clip", overflow: "visible" },
    "& .MuiOutlinedInput-notchedOutline": { borderRadius: 0, borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { borderRadius: 0, bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}`, fontFamily: UI_FONT, "& .MuiMenuItem-root": { fontFamily: "inherit" } } },
  };
  const themedFastMenuProps = {
    ...themedMenuProps,
    keepMounted: true,
    transitionDuration: 0,
    MenuListProps: { dense: true },
  };

  // Model state (synced with Python)
  const [sliceIdx, setSliceIdx] = useModelState<number>("slice_idx");
  const [nSlices] = useModelState<number>("n_slices");
  const [folderWaiting] = useModelState<boolean>("folder_waiting");
  const [folderStatus] = useModelState<string>("folder_status");
  const [folderWatchState] = useModelState<string>("folder_watch_state");
  const [folderWatchDetail] = useModelState<string>("folder_watch_detail");
  const [labels] = useModelState<string[]>("labels");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  // True-color PNG/JPEG stacks: each frame is (H*W*3) float32 RGB in [0, 1].
  const [isRgb] = useModelState<boolean>("is_rgb");
  const [staticFallbackJpeg] = useModelState<string>("_static_fallback_jpeg");
  const [staticFallbackMime] = useModelState<string>("_static_fallback_mime");
  const rgbFrameDataRef = React.useRef<Float32Array | null>(null);
  // Defensive: traitlets.Bytes can identity-suppress trait events when content
  // and length are similar. frame_seq is incremented Python-side on every stack
  // replacement so JS effects always see a change. Use it in dep arrays alongside frameBytes.
  const [frameSeq] = useModelState<number>("frame_seq");
  // Show3D carries one embedded display stack: every frame the browser shows is
  // sliced from it. Native arrays remain in Python; the browser uploads this
  // float32 stack to WebGPU once when available.
  const [offlineStackTrait] = useModelState<DataView>("_offline_stack");
  const [offlineFloatStack] = useModelState<DataView>("_offline_float_stack");
  const [offlineMin] = useModelState<number>("_offline_min");
  const [offlineMax] = useModelState<number>("_offline_max");
  const [offlineMins] = useModelState<number[]>("_offline_mins");
  const [offlineMaxs] = useModelState<number[]>("_offline_maxs");
  const [nPanels] = useModelState<number>("n_panels");
  const [panelWidthPx] = useModelState<number>("panel_width_px");
  const [sharedPanelSource] = useModelState<boolean>("shared_panel_source");
  const [displayBin] = useModelState<number>("display_bin");
  const [sourceHeight] = useModelState<number>("source_height");
  const [nativeSourcePanelWidth] = useModelState<number>("source_panel_width");
  const offlineStack: DataView | null =
    offlineStackTrait && offlineStackTrait.byteLength > 0
      ? offlineStackTrait
      : null;
  // Reused scratch Float32Array sized to one frame so per-scrub dequant
  // doesn't re-allocate. Indexed by (RGB, width, height) since reshape resets it.
  const offlineScratch = React.useRef<Float32Array | null>(null);
  const offlineScratchKey = React.useRef<number>(-1);
  const offlineFrameCacheRef = React.useRef<Map<number, Float32Array>>(new Map());
  const offlineFramePrewarmSerialRef = React.useRef(0);
  // Local live index used by the frameBytes useMemo. During MUI Slider
  // drag, `setSliceIdx` (anywidget useModelState) goes through model.set +
  // save_changes and can batch under rapid pointer ticks. Keep slider/canvas
  // state local while dragging; commit the synced model trait on release or
  // after the scrub stream settles.
  const [liveSliceIdx, setLiveSliceIdx] = React.useState<number>(sliceIdx);
  React.useEffect(() => { setLiveSliceIdx(sliceIdx); }, [sliceIdx]);
  const sliceCommitTimerRef = React.useRef<number | null>(null);
  React.useEffect(() => () => {
    if (sliceCommitTimerRef.current !== null) {
      window.clearTimeout(sliceCommitTimerRef.current);
      sliceCommitTimerRef.current = null;
    }
  }, []);

  const offlineFrameCacheLimit = React.useMemo(() => {
    const frameCount = Math.max(1, Math.round(nSlices || 1));
    const pixelCount = Math.max(1, Math.round(width || 0) * Math.round(height || 0));
    if (pixelCount <= 0) return 0;
    if (offlineFloatStack && offlineFloatStack.byteLength > 0) return frameCount;
    const bytesPerFrame = Math.max(1, pixelCount * (isRgb ? 3 : 1) * 4);
    const budgetFrames = Math.max(1, Math.floor(OFFLINE_FRAME_CACHE_BYTES / bytesPerFrame));
    const minFrames = bytesPerFrame <= OFFLINE_FRAME_CACHE_BYTES / OFFLINE_FRAME_CACHE_MIN_FRAMES
      ? OFFLINE_FRAME_CACHE_MIN_FRAMES
      : 1;
    return Math.max(1, Math.min(frameCount, Math.max(minFrames, budgetFrames)));
  }, [offlineFloatStack, width, height, nSlices, isRgb]);
  React.useEffect(() => {
    offlineFrameCacheRef.current.clear();
    offlineFramePrewarmSerialRef.current++;
  }, [
    offlineStack,
    offlineFloatStack,
    offlineMin,
    offlineMax,
    offlineMins,
    offlineMaxs,
    width,
    height,
    nPanels,
    panelWidthPx,
    nSlices,
    isRgb,
    offlineFrameCacheLimit,
  ]);
  const putOfflineFrameCache = React.useCallback((idx: number, frame: Float32Array) => {
    if (offlineFrameCacheLimit <= 0) return;
    const cache = offlineFrameCacheRef.current;
    if (cache.has(idx)) cache.delete(idx);
    cache.set(idx, frame);
    while (cache.size > offlineFrameCacheLimit) {
      const oldest = cache.keys().next().value;
      if (oldest === undefined) break;
      cache.delete(oldest);
    }
  }, [offlineFrameCacheLimit]);
  const frameBytes = React.useMemo<DataView>(() => {
    // Gray: H*W floats. True-color RGB: H*W*3 floats packed channel-last.
    const channels = isRgb ? 3 : 1;
    const floatsPerFrame = channels * width * height;
    const pixelCount = width * height;
    if (offlineFloatStack && offlineFloatStack.byteLength > 0 && floatsPerFrame > 0) {
      const f32 = float32FrameFromDataView(offlineFloatStack, liveSliceIdx, floatsPerFrame, false);
      if (f32) return new DataView(f32.buffer, f32.byteOffset, f32.byteLength);
    }
    // Dequantize the explicitly requested uint8 standalone export.
    const dequantU8Frame = (u8: Uint8Array): DataView | null => {
      if (u8.byteLength < channels * pixelCount || width <= 0 || height <= 0) return null;
      const key = ((isRgb ? 1 : 0) << 30) | (width << 15) | height;
      if (offlineScratchKey.current !== key || offlineScratch.current === null) {
        offlineScratch.current = new Float32Array(floatsPerFrame);
        offlineScratchKey.current = key;
      }
      const f32 = offlineScratch.current;
      if (isRgb) {
        dequantizeUint8(u8, 0, 1, f32);
      } else {
        // Offline uint8 packs are already display-quantized per panel. Restore
        // physical units with a panel-tiled loop (not per-pixel panel index).
        const panelCount = Math.max(1, nPanels || 1);
        const panelRanges = panelCount > 1 && offlineMins?.length >= panelCount && offlineMaxs?.length >= panelCount;
        const panelW = Math.max(1, panelWidthPx || Math.floor(width / panelCount) || width);
        if (panelRanges) {
          for (let panel = 0; panel < panelCount; panel++) {
            const panelMin = offlineMins[panel] ?? offlineMin;
            const panelMax = offlineMaxs[panel] ?? offlineMax;
            const colStart = panel * panelW;
            const colEnd = Math.min(width, colStart + panelW);
            for (let row = 0; row < height; row++) {
              const start = row * width + colStart;
              const end = row * width + colEnd;
              dequantizeUint8(u8.subarray(start, end), panelMin, panelMax, f32.subarray(start, end));
            }
          }
        } else {
          dequantizeUint8(u8.subarray(0, pixelCount), offlineMin, offlineMax, f32);
        }
      }
      return new DataView(f32.buffer);
    };
    if (offlineStack && offlineStack.byteLength > 0 && width > 0 && height > 0) {
      // RGB uint8 pack is H*W*3 bytes per frame (display-ready 0–255 → /255).
      const bytesPerFrame = channels * pixelCount;
      const start = liveSliceIdx * bytesPerFrame;
      if (start + bytesPerFrame <= offlineStack.byteLength) {
        const u8 = new Uint8Array(offlineStack.buffer, offlineStack.byteOffset + start, bytesPerFrame);
        const view = dequantU8Frame(u8);
        if (view) return view;
      }
    }
    return EMPTY_FRAME;
  }, [offlineStack, offlineFloatStack, offlineMin, offlineMax, offlineMins, offlineMaxs, liveSliceIdx, width, height, nPanels, panelWidthPx, isRgb]);
  const getOfflineFrame = React.useCallback((idx: number): Float32Array | null => {
    // Cache per-frame Float32Array objects by frame index. The previous single
    // scratch buffer was unsafe because pointer-equality upload guards could skip
    // a texture refresh; per-index cached arrays keep stable identity without
    // mutating one shared backing store.
    if (width <= 0 || height <= 0) return null;
    const frameCount = Math.max(1, nSlices || 1);
    const normalized = ((Math.round(idx) % frameCount) + frameCount) % frameCount;
    const cached = offlineFrameCacheRef.current.get(normalized);
    if (cached) {
      offlineFrameCacheRef.current.delete(normalized);
      offlineFrameCacheRef.current.set(normalized, cached);
      return cached;
    }
    const channels = isRgb ? 3 : 1;
    const floatsPerFrame = channels * width * height;
    const pixelCount = width * height;
    if (offlineFloatStack && offlineFloatStack.byteLength > 0) {
      const frame = float32FrameFromDataView(offlineFloatStack, normalized, floatsPerFrame, false);
      if (frame) putOfflineFrameCache(normalized, frame);
      return frame;
    }
    const bytesPerFrame = channels * pixelCount;
    const dequantU8 = (u8: Uint8Array): Float32Array => {
      const f32 = new Float32Array(floatsPerFrame);
      if (isRgb) {
        return dequantizeUint8(u8, 0, 1, f32);
      }
      const panelCount = Math.max(1, nPanels || 1);
      const panelRanges = panelCount > 1 && offlineMins?.length >= panelCount && offlineMaxs?.length >= panelCount;
      const panelW = Math.max(1, panelWidthPx || Math.floor(width / panelCount) || width);
      if (panelRanges) {
        for (let panel = 0; panel < panelCount; panel++) {
          const panelMin = offlineMins[panel] ?? offlineMin;
          const panelMax = offlineMaxs[panel] ?? offlineMax;
          const colStart = panel * panelW;
          const colEnd = Math.min(width, colStart + panelW);
          for (let row = 0; row < height; row++) {
            const start = row * width + colStart;
            const end = row * width + colEnd;
            dequantizeUint8(u8.subarray(start, end), panelMin, panelMax, f32.subarray(start, end));
          }
        }
      } else {
        dequantizeUint8(u8.subarray(0, pixelCount), offlineMin, offlineMax, f32);
      }
      return f32;
    };
    if (!offlineStack || offlineStack.byteLength === 0) return null;
    const start = normalized * bytesPerFrame;
    if (start < 0 || start + bytesPerFrame > offlineStack.byteLength) return null;
    const u8 = new Uint8Array(offlineStack.buffer, offlineStack.byteOffset + start, bytesPerFrame);
    const f32 = dequantU8(u8);
    putOfflineFrameCache(normalized, f32);
    return f32;
  }, [
    width,
    height,
    nSlices,
    offlineFloatStack,
    offlineStack,
    offlineMins,
    offlineMaxs,
    offlineMin,
    offlineMax,
    nPanels,
    panelWidthPx,
    putOfflineFrameCache,
    isRgb,
  ]);

  React.useEffect(() => {
    if (width <= 0 || height <= 0 || nSlices <= 1 || offlineFrameCacheLimit <= 0) return;
    const serial = ++offlineFramePrewarmSerialRef.current;
    let cancelled = false;
    let timer: number | null = null;
    const order = orderedFramePrewarmIndices(liveSliceIdx, nSlices).slice(0, offlineFrameCacheLimit);
    const schedule = (delayMs = 0) => {
      timer = window.setTimeout(step, delayMs);
    };
    let cursor = 0;
    const step = () => {
      timer = null;
      if (cancelled || serial !== offlineFramePrewarmSerialRef.current) return;
      const frameBudgetStart = performance.now();
      while (cursor < order.length && performance.now() - frameBudgetStart < 8) {
        getOfflineFrame(order[cursor]);
        cursor++;
      }
      if (cursor < order.length) {
        schedule(0);
      }
    };
    schedule(0);
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [width, height, nSlices, liveSliceIdx, offlineFrameCacheLimit, getOfflineFrame]);


  const [title] = useModelState<string>("title");
  const [showTitle] = useModelState<boolean>("show_title");
  const [dimLabel] = useModelState<string>("dim_label");
  const [dimSampling] = useModelState<number>("dim_sampling");
  const [dimUnit] = useModelState<string>("dim_unit");
  const [panelTitles] = useModelState<string[]>("panel_titles");
  const [panelTitleSpans] = useModelState<RichTitleSpan[][]>("panel_title_spans");
  const [panelRealFrames] = useModelState<number[]>("panel_real_frames");
  const [starred, setStarred] = useModelState<number[]>("starred");
  const [hiddenPanels, setHiddenPanels] = useModelState<number[]>("hidden_panels");
  const [selectedPanels, setSelectedPanels] = useModelState<number[]>("selected_panels");
  const [hiddenPageSlotsTrait, setHiddenPageSlotsTrait] = useModelState<number[] | undefined>("hidden_page_slots");
  const [panelOrder, setPanelOrder] = useModelState<number[]>("panel_order");
  const [nPages] = useModelState<number>("n_pages");
  const [pageIdx, setPageIdx] = useModelState<number>("page_idx");
  const [panelsPerPage] = useModelState<number>("panels_per_page");
  const [pageLabels] = useModelState<string[]>("page_labels");
  const [pageStarred, setPageStarred] = useModelState<number[]>("page_starred");
  const [pagePlaying, setPagePlaying] = React.useState(false);
  const [pagePlayFps, setPagePlayFps] = React.useState<number>(2);
  const [pageSliderPreviewIdx, setPageSliderPreviewIdxState] = React.useState<number | null>(null);
  const pageSliderPreviewIdxRef = React.useRef<number | null>(null);
  const currentPageIdxRef = React.useRef(0);
  const pageCommitPendingRef = React.useRef<number | null>(null);
  const pageCommitRafRef = React.useRef<number | null>(null);
  const [reorderMode, setReorderMode] = React.useState(false);
  const [dragOverPanel, setDragOverPanel] = React.useState<number | null>(null);
  const [reorderPreviewOrder, setReorderPreviewOrder] = React.useState<number[] | null>(null);
  const [reorderDragVisual, setReorderDragVisual] = React.useState<ReorderDragVisual | null>(null);
  const draggedPanelRef = React.useRef<number | null>(null);
  const pointerReorderPanelRef = React.useRef<number | null>(null);
  const reorderPreviewOrderRef = React.useRef<number[] | null>(null);
  const reorderDragVisualRef = React.useRef<ReorderDragVisual | null>(null);
  const reorderGhostRef = React.useRef<HTMLDivElement>(null);
  const reorderGhostRafRef = React.useRef<number | null>(null);
  const lastSelectedPanelRef = React.useRef<number | null>(null);
  const reorderGhostPendingRef = React.useRef<{ x: number; y: number } | null>(null);
  const reorderDragStartRef = React.useRef<ReorderDragStart | null>(null);
  const reorderDragActivatedRef = React.useRef(false);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const gpuCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const canvasContainerRef = React.useRef<HTMLDivElement>(null);
  const totalPanelCount = Math.max(1, nPanels || 1);
  const isPaged = (nPages || 1) > 1 && (panelsPerPage || 0) > 0;
  const currentPageIdx = Math.max(0, Math.min((nPages || 1) - 1, Math.round(pageIdx || 0)));
  const displayPageIdx = pageSliderPreviewIdx === null
    ? currentPageIdx
    : Math.max(0, Math.min((nPages || 1) - 1, Math.round(pageSliderPreviewIdx || 0)));
  React.useEffect(() => {
    currentPageIdxRef.current = currentPageIdx;
  }, [currentPageIdx]);
  const clampPageIdx = React.useCallback((value: number) => (
    Math.max(0, Math.min((nPages || 1) - 1, Math.round(Number(value) || 0)))
  ), [nPages]);
  const setPageSliderPreviewIdx = React.useCallback((value: number | null) => {
    pageSliderPreviewIdxRef.current = value;
    setPageSliderPreviewIdxState(value);
  }, []);
  const commitPageIdx = React.useCallback((value: number, immediate = false) => {
    const next = clampPageIdx(value);
    pageCommitPendingRef.current = next;
    if (immediate) {
      if (pageCommitRafRef.current !== null) {
        window.cancelAnimationFrame(pageCommitRafRef.current);
        pageCommitRafRef.current = null;
      }
      pageCommitPendingRef.current = null;
      if (next !== currentPageIdxRef.current) setPageIdx(next);
      return;
    }
    if (pageCommitRafRef.current !== null) return;
    pageCommitRafRef.current = window.requestAnimationFrame(() => {
      pageCommitRafRef.current = null;
      const pending = pageCommitPendingRef.current;
      pageCommitPendingRef.current = null;
      if (pending !== null && pending !== currentPageIdxRef.current) setPageIdx(pending);
    });
  }, [clampPageIdx, setPageIdx]);
  const stopPagePlayback = React.useCallback(() => {
    setPagePlaying(value => value ? false : value);
  }, []);
  React.useEffect(() => {
    const preview = pageSliderPreviewIdxRef.current;
    if (preview !== null && preview === currentPageIdx) {
      setPageSliderPreviewIdx(null);
    }
  }, [currentPageIdx, setPageSliderPreviewIdx]);
  React.useEffect(() => () => {
    if (pageCommitRafRef.current !== null) {
      window.cancelAnimationFrame(pageCommitRafRef.current);
      pageCommitRafRef.current = null;
    }
  }, []);
  const pageControlIdx = clampPageIdx(pageSliderPreviewIdx ?? currentPageIdx);
  const pageControlLabel = pageLabels?.[pageControlIdx] || `Page ${pageControlIdx + 1}`;
  const pageControlStatus = `${pageControlLabel} ${pageControlIdx + 1}/${nPages || 1}`;
  React.useEffect(() => {
    if (!isPaged || (nPages || 1) <= 1) setPagePlaying(false);
  }, [isPaged, nPages]);
  React.useEffect(() => {
    if (!pagePlaying || !isPaged || (nPages || 1) <= 1) return;
    const timeout = window.setTimeout(() => {
      const next = (currentPageIdx + 1) % Math.max(1, nPages || 1);
      setPageSliderPreviewIdx(next);
      setPageIdx(next);
    }, 1000 / Math.max(1, pagePlayFps));
    return () => window.clearTimeout(timeout);
  }, [currentPageIdx, isPaged, nPages, pagePlayFps, pagePlaying, setPageIdx, setPageSliderPreviewIdx]);
  const activePageStart = isPaged ? displayPageIdx * Math.max(1, panelsPerPage || 1) : 0;
  const activePageEnd = isPaged ? Math.min(totalPanelCount, activePageStart + Math.max(1, panelsPerPage || 1)) : totalPanelCount;
  const activePageIndices = React.useMemo(
    () => Array.from({ length: Math.max(0, activePageEnd - activePageStart) }, (_, i) => activePageStart + i),
    [activePageStart, activePageEnd]
  );
  const activePanelCount = isPaged ? activePageIndices.length : totalPanelCount;
  const [hiddenPageSlots, setHiddenPageSlots] = React.useState<number[]>([]);
  const hiddenPageSlotsInitializedRef = React.useRef(false);
  React.useEffect(() => {
    if (!isPaged) {
      hiddenPageSlotsInitializedRef.current = false;
      setHiddenPageSlots(prev => prev.length === 0 ? prev : []);
      return;
    }
    if (Array.isArray(hiddenPageSlotsTrait)) {
      const slots = normalizeHiddenPageSlots(hiddenPageSlotsTrait, activePanelCount);
      hiddenPageSlotsInitializedRef.current = true;
      setHiddenPageSlots(prev => sameNumberArray(prev, slots) ? prev : slots);
      return;
    }
    if (hiddenPageSlotsInitializedRef.current) return;
    hiddenPageSlotsInitializedRef.current = true;
    const slots = normalizeHiddenPageSlots(
      (hiddenPanels || []).map((value) => Math.trunc(Number(value)) - activePageStart),
      activePanelCount,
    );
    setHiddenPageSlots(prev => sameNumberArray(prev, slots) ? prev : slots);
  }, [activePageStart, activePanelCount, hiddenPageSlotsTrait, hiddenPanels, isPaged]);
  const hiddenPanelSet = React.useMemo(() => {
    const clean = new Set<number>();
    if (isPaged) {
      for (const value of hiddenPageSlots || []) {
        const slot = Math.trunc(Number(value));
        const idx = activePageStart + slot;
        if (Number.isFinite(slot) && slot >= 0 && slot < activePanelCount && idx >= activePageStart && idx < activePageEnd) {
          clean.add(idx);
        }
      }
    } else {
      for (const value of hiddenPanels || []) {
        const idx = Math.trunc(Number(value));
        if (Number.isFinite(idx) && idx >= 0 && idx < totalPanelCount) clean.add(idx);
      }
    }
    const activeHiddenCount = (isPaged ? activePageIndices : Array.from({ length: totalPanelCount }, (_, panel) => panel))
      .filter((panel) => clean.has(panel)).length;
    if (activeHiddenCount >= Math.max(1, activePanelCount)) {
      const fallback = (isPaged ? activePageIndices : [totalPanelCount - 1])[Math.max(0, activePanelCount - 1)];
      clean.delete(fallback);
    }
    return clean;
  }, [activePageEnd, activePageIndices, activePageStart, activePanelCount, hiddenPageSlots, hiddenPanels, totalPanelCount, isPaged]);
  const naturalPanelOrder = React.useMemo(
    () => isPaged ? activePageIndices : Array.from({ length: totalPanelCount }, (_, panel) => panel),
    [activePageIndices, isPaged, totalPanelCount]
  );
  const orderedPanelIndices = React.useMemo(() => {
    if (isPaged) return naturalPanelOrder;
    const values = Array.isArray(panelOrder) ? panelOrder.map(value => Math.trunc(Number(value))) : [];
    const valid = (
      values.length === totalPanelCount &&
      values.every((value) => Number.isFinite(value) && value >= 0 && value < totalPanelCount) &&
      new Set(values).size === totalPanelCount
    );
    return valid ? values : naturalPanelOrder;
  }, [panelOrder, naturalPanelOrder, totalPanelCount, isPaged]);
  const previewOrderedPanelIndices = React.useMemo(() => {
    if (isPaged) return null;
    const values = Array.isArray(reorderPreviewOrder) ? reorderPreviewOrder.map(value => Math.trunc(Number(value))) : [];
    const valid = (
      values.length === totalPanelCount &&
      values.every((value) => Number.isFinite(value) && value >= 0 && value < totalPanelCount) &&
      new Set(values).size === totalPanelCount
    );
    return valid ? values : null;
  }, [reorderPreviewOrder, totalPanelCount, isPaged]);
  const displayOrderedPanelIndices = previewOrderedPanelIndices || orderedPanelIndices;
  const visiblePanelIndices = React.useMemo(
    () => displayOrderedPanelIndices.filter(panel => !hiddenPanelSet.has(panel)),
    [hiddenPanelSet, displayOrderedPanelIndices]
  );
  const visiblePanelCount = visiblePanelIndices.length;
  const panelMenuTotal = isPaged ? activePanelCount : totalPanelCount;
  const hasPanelChoices = panelMenuTotal > 1;
  const selectedPanelSet = React.useMemo(() => {
    const out = new Set<number>();
    for (const value of selectedPanels || []) {
      const panel = Math.trunc(Number(value));
      if (Number.isFinite(panel) && panel >= 0 && panel < totalPanelCount && !hiddenPanelSet.has(panel)) out.add(panel);
    }
    return out;
  }, [hiddenPanelSet, selectedPanels, totalPanelCount]);
  const selectedVisiblePanels = React.useMemo(
    () => visiblePanelIndices.filter((panel) => selectedPanelSet.has(panel)),
    [selectedPanelSet, visiblePanelIndices],
  );
  const selectedVisibleCount = selectedVisiblePanels.length;
  const panelLabel = React.useCallback((panel: number) => (
    (panelTitles && panelTitles[panel]) || `Panel ${panel + 1}`
  ), [panelTitles]);
  const panelTitleContent = React.useCallback((panel: number) => (
    renderRichTitle(panelTitleSpans?.[panel], panelLabel(panel))
  ), [panelLabel, panelTitleSpans]);
  const panelTitleText = React.useCallback((panel: number) => (
    richTitlePlainText(panelTitleSpans?.[panel], panelLabel(panel))
  ), [panelLabel, panelTitleSpans]);
  const setPanelHidden = React.useCallback((panel: number, hidden: boolean) => {
    if (panel < 0 || panel >= totalPanelCount) return;
    if (isPaged) {
      if (panel < activePageStart || panel >= activePageEnd) return;
      const slot = panel - activePageStart;
      const next = new Set<number>();
      for (const value of hiddenPageSlots || []) {
        const idx = Math.trunc(Number(value));
        if (Number.isFinite(idx) && idx >= 0 && idx < activePanelCount) next.add(idx);
      }
      if (hidden) {
        if (!next.has(slot) && activePanelCount - next.size <= 1) return;
        next.add(slot);
      } else {
        next.delete(slot);
      }
      const slots = normalizeHiddenPageSlots(Array.from(next), activePanelCount);
      setHiddenPageSlots(slots);
      setHiddenPageSlotsTrait(slots);
      return;
    }
    const next = new Set<number>();
    for (const value of hiddenPanels || []) {
      const idx = Math.trunc(Number(value));
      if (Number.isFinite(idx) && idx >= 0 && idx < totalPanelCount) next.add(idx);
    }
    if (hidden) {
      const activeVisible = (isPaged ? activePageIndices : Array.from({ length: totalPanelCount }, (_, idx) => idx))
        .filter((idx) => !next.has(idx)).length;
      if (!next.has(panel) && activeVisible <= 1) return;
      next.add(panel);
    } else {
      next.delete(panel);
    }
    setHiddenPanels(Array.from(next).sort((a, b) => a - b));
  }, [activePageEnd, activePageStart, activePanelCount, hiddenPageSlots, hiddenPanels, totalPanelCount, isPaged, activePageIndices, setHiddenPanels, setHiddenPageSlotsTrait]);
  const setPanelsHidden = React.useCallback((panels: number[], hidden: boolean) => {
    const panelSet = new Set(
      panels
        .map((panel) => Math.trunc(Number(panel)))
        .filter((panel) => Number.isFinite(panel) && panel >= 0 && panel < totalPanelCount),
    );
    if (panelSet.size === 0) return;
    if (isPaged) {
      const next = new Set<number>();
      for (const value of hiddenPageSlots || []) {
        const slot = Math.trunc(Number(value));
        if (Number.isFinite(slot) && slot >= 0 && slot < activePanelCount) next.add(slot);
      }
      for (const panel of panelSet) {
        if (panel < activePageStart || panel >= activePageEnd) continue;
        const slot = panel - activePageStart;
        if (hidden) next.add(slot);
        else next.delete(slot);
      }
      if (activePanelCount - next.size <= 0) return;
      const slots = normalizeHiddenPageSlots(Array.from(next), activePanelCount);
      setHiddenPageSlots(slots);
      setHiddenPageSlotsTrait(slots);
      return;
    }
    const next = new Set<number>();
    for (const value of hiddenPanels || []) {
      const idx = Math.trunc(Number(value));
      if (Number.isFinite(idx) && idx >= 0 && idx < totalPanelCount) next.add(idx);
    }
    for (const panel of panelSet) {
      if (hidden) next.add(panel);
      else next.delete(panel);
    }
    if (next.size >= totalPanelCount) return;
    setHiddenPanels(Array.from(next).sort((a, b) => a - b));
  }, [activePageEnd, activePageStart, activePanelCount, hiddenPageSlots, hiddenPanels, totalPanelCount, isPaged, setHiddenPanels, setHiddenPageSlotsTrait]);
  const handlePanelSelectionMouseDown = React.useCallback((event: React.MouseEvent, panel: number): boolean => {
    if (!hasPanelChoices || reorderMode || panel < 0) return false;
    const orderedVisible = displayOrderedPanelIndices.filter((idx) => visiblePanelIndices.includes(idx));
    const current = new Set(selectedPanelSet);
    let next: number[];
    if (event.shiftKey) {
      const anchor = lastSelectedPanelRef.current !== null && orderedVisible.includes(lastSelectedPanelRef.current)
        ? lastSelectedPanelRef.current
        : (selectedVisiblePanels[selectedVisiblePanels.length - 1] ?? orderedVisible[0] ?? panel);
      const anchorPos = orderedVisible.indexOf(anchor);
      const panelPos = orderedVisible.indexOf(panel);
      if (anchorPos >= 0 && panelPos >= 0) {
        const [rangeStart, rangeEnd] = anchorPos < panelPos ? [anchorPos, panelPos] : [panelPos, anchorPos];
        next = orderedVisible.slice(rangeStart, rangeEnd + 1);
      } else {
        next = [panel];
      }
      event.preventDefault();
      event.stopPropagation();
    } else if (event.metaKey || event.ctrlKey) {
      if (current.has(panel) && current.size > 1) current.delete(panel);
      else current.add(panel);
      next = orderedVisible.filter((idx) => current.has(idx));
      event.preventDefault();
      event.stopPropagation();
    } else {
      next = [panel];
    }
    lastSelectedPanelRef.current = panel;
    setSelectedPanels(next);
    return event.shiftKey || event.metaKey || event.ctrlKey;
  }, [displayOrderedPanelIndices, hasPanelChoices, reorderMode, selectedPanelSet, selectedVisiblePanels, setSelectedPanels, visiblePanelIndices]);
  React.useEffect(() => {
    if (!hasPanelChoices) {
      lastSelectedPanelRef.current = null;
      if ((selectedPanels || []).length > 0) setSelectedPanels([]);
      return;
    }
    const clean = visiblePanelIndices.filter((panel) => selectedPanelSet.has(panel));
    if (!sameNumberArray(selectedPanels, clean)) setSelectedPanels(clean);
    if (lastSelectedPanelRef.current !== null && !visiblePanelIndices.includes(lastSelectedPanelRef.current)) {
      lastSelectedPanelRef.current = clean[clean.length - 1] ?? null;
    }
  }, [hasPanelChoices, selectedPanelSet, selectedPanels, setSelectedPanels, visiblePanelIndices]);
  const applyPanelOrder = React.useCallback((order: number[]) => {
    const clean = order.filter((value) => Number.isInteger(value) && value >= 0 && value < totalPanelCount);
    if (clean.length !== totalPanelCount || new Set(clean).size !== totalPanelCount) return;
    const natural = clean.every((value, idx) => value === idx);
    setPanelOrder(natural ? [] : clean);
  }, [setPanelOrder, totalPanelCount]);
  const setReorderPreviewOrderValue = React.useCallback((order: number[] | null) => {
    reorderPreviewOrderRef.current = order;
    setReorderPreviewOrder(order);
  }, []);
  const setReorderDragVisualValue = React.useCallback((visual: ReorderDragVisual | null) => {
    reorderDragVisualRef.current = visual;
    setReorderDragVisual(visual);
  }, []);
  const captureReorderPanelImage = React.useCallback((panelRect: DOMRect, containerRect: DOMRect): string => {
    const container = canvasContainerRef.current;
    if (!container) return "";
    const canvases = Array.from(container.querySelectorAll("canvas")) as HTMLCanvasElement[];
    const source = canvases.find((canvas) => {
      const rect = canvas.getBoundingClientRect();
      const style = window.getComputedStyle(canvas);
      const opacity = Number(style.opacity || "1");
      return rect.width > 0 && rect.height > 0 && canvas.width > 0 && canvas.height > 0 &&
        style.display !== "none" && opacity > 0.5;
    }) || canvases.find((canvas) => canvas.width > 0 && canvas.height > 0);
    if (!source) return "";
    const scaleX = source.width / Math.max(1, containerRect.width);
    const scaleY = source.height / Math.max(1, containerRect.height);
    const sourceX = Math.max(0, Math.round((panelRect.left - containerRect.left) * scaleX));
    const sourceY = Math.max(0, Math.round((panelRect.top - containerRect.top) * scaleY));
    const sourceWidth = Math.max(1, Math.min(source.width - sourceX, Math.round(panelRect.width * scaleX)));
    const sourceHeight = Math.max(1, Math.min(source.height - sourceY, Math.round(panelRect.height * scaleY)));
    if (sourceWidth <= 0 || sourceHeight <= 0) return "";
    const scratch = document.createElement("canvas");
    scratch.width = sourceWidth;
    scratch.height = sourceHeight;
    const ctx = scratch.getContext("2d");
    if (!ctx) return "";
    try {
      ctx.drawImage(source, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, sourceWidth, sourceHeight);
      return scratch.toDataURL("image/png");
    } catch {
      return "";
    }
  }, []);
  const updateReorderGhostPosition = React.useCallback((clientX: number, clientY: number) => {
    const visual = reorderDragVisualRef.current;
    const container = canvasContainerRef.current;
    if (!visual || !container) return;
    const rect = container.getBoundingClientRect();
    const x = Math.max(0, Math.min(Math.max(0, rect.width - visual.width), clientX - rect.left - visual.offsetX));
    const y = Math.max(0, Math.min(Math.max(0, rect.height - visual.height), clientY - rect.top - visual.offsetY));
    reorderGhostPendingRef.current = { x, y };
    if (reorderGhostRafRef.current !== null) return;
    reorderGhostRafRef.current = window.requestAnimationFrame(() => {
      reorderGhostRafRef.current = null;
      const pending = reorderGhostPendingRef.current;
      const ghost = reorderGhostRef.current;
      if (!pending || !ghost) return;
      ghost.style.transform = `translate3d(${pending.x}px, ${pending.y}px, 0)`;
    });
  }, []);
  const beginReorderDragVisual = React.useCallback((event: React.PointerEvent, panel: number) => {
    const container = canvasContainerRef.current;
    if (!container) return;
    const panelRect = event.currentTarget.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const offsetX = Math.max(0, Math.min(panelRect.width, event.clientX - panelRect.left));
    const offsetY = Math.max(0, Math.min(panelRect.height, event.clientY - panelRect.top));
    const x = Math.max(0, Math.min(Math.max(0, containerRect.width - panelRect.width), event.clientX - containerRect.left - offsetX));
    const y = Math.max(0, Math.min(Math.max(0, containerRect.height - panelRect.height), event.clientY - containerRect.top - offsetY));
    setReorderDragVisualValue({
      panel,
      label: panelLabel(panel),
      imageUrl: captureReorderPanelImage(panelRect, containerRect),
      width: panelRect.width,
      height: panelRect.height,
      x,
      y,
      offsetX,
      offsetY,
    });
    reorderGhostPendingRef.current = { x, y };
    requestAnimationFrame(() => updateReorderGhostPosition(event.clientX, event.clientY));
  }, [captureReorderPanelImage, panelLabel, setReorderDragVisualValue, updateReorderGhostPosition]);
  const clearReorderDragVisual = React.useCallback(() => {
    if (reorderGhostRafRef.current !== null) {
      window.cancelAnimationFrame(reorderGhostRafRef.current);
      reorderGhostRafRef.current = null;
    }
    reorderGhostPendingRef.current = null;
    reorderDragStartRef.current = null;
    reorderDragActivatedRef.current = false;
    setReorderDragVisualValue(null);
  }, [setReorderDragVisualValue]);
  const reorderDragHasPassedThreshold = React.useCallback((clientX: number, clientY: number) => {
    const start = reorderDragStartRef.current;
    if (!start) return true;
    if (reorderDragActivatedRef.current) return true;
    const distance = Math.hypot(clientX - start.x, clientY - start.y);
    if (distance < REORDER_DRAG_THRESHOLD_PX) return false;
    reorderDragActivatedRef.current = true;
    return true;
  }, []);
  const buildPanelMovedOrder = React.useCallback((
    source: number,
    target: number,
    placement: ReorderPlacement,
    baseOrder?: number[] | null,
  ): number[] | null => {
    if (source === target) return null;
    const base = Array.isArray(baseOrder) && baseOrder.length === totalPanelCount
      ? baseOrder
      : orderedPanelIndices;
    const next = [...base];
    const from = next.indexOf(source);
    if (from < 0) return null;
    next.splice(from, 1);
    const targetIndex = next.indexOf(target);
    if (targetIndex < 0) return null;
    const insertAt = placement === "after" ? targetIndex + 1 : targetIndex;
    next.splice(insertAt, 0, source);
    return next;
  }, [orderedPanelIndices, totalPanelCount]);
  const panelReorderTargetFromPoint = React.useCallback((clientX: number, clientY: number): { panel: number; placement: ReorderPlacement } | null => {
    if (typeof document === "undefined") return null;
    const elements = document.elementsFromPoint(clientX, clientY);
    let targetEl: HTMLElement | null = null;
    for (const element of elements) {
      if (!(element instanceof HTMLElement)) continue;
      const candidate = element.closest("[data-show3d-reorder-panel]");
      if (candidate instanceof HTMLElement) {
        targetEl = candidate;
        break;
      }
    }
    const allTargets = Array.from(document.querySelectorAll<HTMLElement>("[data-show3d-reorder-panel]"))
      .map((element) => {
        const rect = element.getBoundingClientRect();
        const raw = element.dataset.show3dReorderPanel;
        const panel = raw == null ? Number.NaN : Math.trunc(Number(raw));
        return { element, rect, panel };
      })
      .filter((item) => Number.isFinite(item.panel) && item.rect.width > 0 && item.rect.height > 0);
    if (!targetEl && allTargets.length) {
      let best = allTargets[0];
      let bestDistance = Number.POSITIVE_INFINITY;
      for (const item of allTargets) {
        const dx = Math.max(item.rect.left - clientX, 0, clientX - item.rect.right);
        const dy = Math.max(item.rect.top - clientY, 0, clientY - item.rect.bottom);
        const distance = dx * dx + dy * dy;
        if (distance < bestDistance) {
          best = item;
          bestDistance = distance;
        }
      }
      targetEl = best.element;
    }
    if (!targetEl) return null;
    const raw = targetEl.dataset.show3dReorderPanel;
    const panel = raw == null ? Number.NaN : Math.trunc(Number(raw));
    if (!Number.isFinite(panel) || panel < 0 || panel >= totalPanelCount) return null;
    const rect = targetEl.getBoundingClientRect();
    const sameRowNeighbor = allTargets.some((item) => item.panel !== panel && Math.abs(item.rect.top - rect.top) < 8);
    const sameColumnNeighbor = allTargets.some((item) => item.panel !== panel && Math.abs(item.rect.left - rect.left) < 8);
    const useHorizontal = sameRowNeighbor || !sameColumnNeighbor;
    const placement: ReorderPlacement = useHorizontal
      ? (clientX >= rect.left + rect.width / 2 ? "after" : "before")
      : (clientY >= rect.top + rect.height / 2 ? "after" : "before");
    return { panel, placement };
  }, [totalPanelCount]);
  const previewPanelReorderFromPoint = React.useCallback((clientX: number, clientY: number) => {
    const source = pointerReorderPanelRef.current ?? draggedPanelRef.current;
    if (source === null) return;
    const target = panelReorderTargetFromPoint(clientX, clientY);
    if (!target) return;
    setDragOverPanel(target.panel);
    const base = reorderPreviewOrderRef.current || orderedPanelIndices;
    const next = buildPanelMovedOrder(source, target.panel, target.placement, base);
    if (!next) return;
    const current = reorderPreviewOrderRef.current || orderedPanelIndices;
    if (next.length === current.length && next.every((value, idx) => value === current[idx])) return;
    setReorderPreviewOrderValue(next);
  }, [buildPanelMovedOrder, orderedPanelIndices, panelReorderTargetFromPoint, setReorderPreviewOrderValue]);
  const commitPanelReorderPreview = React.useCallback(() => {
    const next = reorderPreviewOrderRef.current;
    if (next) applyPanelOrder(next);
    setReorderPreviewOrderValue(null);
    setDragOverPanel(null);
    draggedPanelRef.current = null;
    pointerReorderPanelRef.current = null;
    clearReorderDragVisual();
  }, [applyPanelOrder, clearReorderDragVisual, setReorderPreviewOrderValue]);
  const cancelPanelReorderPreview = React.useCallback(() => {
    setReorderPreviewOrderValue(null);
    setDragOverPanel(null);
    draggedPanelRef.current = null;
    pointerReorderPanelRef.current = null;
    clearReorderDragVisual();
  }, [clearReorderDragVisual, setReorderPreviewOrderValue]);
  const handlePanelDragStart = React.useCallback((event: React.DragEvent, panel: number) => {
    if (!reorderMode) return;
    draggedPanelRef.current = panel;
    setReorderPreviewOrderValue(orderedPanelIndices);
    setDragOverPanel(panel);
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", String(panel));
    const blankDragImage = document.createElement("canvas");
    blankDragImage.width = 1;
    blankDragImage.height = 1;
    event.dataTransfer.setDragImage(blankDragImage, 0, 0);
    event.stopPropagation();
  }, [orderedPanelIndices, reorderMode, setReorderPreviewOrderValue]);
  const handlePanelDragOver = React.useCallback((event: React.DragEvent, panel: number) => {
    if (!reorderMode) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    if (dragOverPanel !== panel) setDragOverPanel(panel);
    previewPanelReorderFromPoint(event.clientX, event.clientY);
    event.stopPropagation();
  }, [dragOverPanel, previewPanelReorderFromPoint, reorderMode]);
  const handlePanelDrop = React.useCallback((event: React.DragEvent) => {
    if (!reorderMode) return;
    event.preventDefault();
    const raw = event.dataTransfer.getData("text/plain");
    const source = raw.trim() !== "" && Number.isFinite(Number(raw))
      ? Math.trunc(Number(raw))
      : draggedPanelRef.current;
    if (source !== null && source !== undefined) {
      draggedPanelRef.current = source;
      previewPanelReorderFromPoint(event.clientX, event.clientY);
    }
    commitPanelReorderPreview();
    event.stopPropagation();
  }, [commitPanelReorderPreview, previewPanelReorderFromPoint, reorderMode]);
  const handlePanelDragEnd = React.useCallback(() => {
    cancelPanelReorderPreview();
  }, [cancelPanelReorderPreview]);
  const handlePanelReorderPointerDown = React.useCallback((event: React.PointerEvent, panel: number) => {
    if (!reorderMode) return;
    pointerReorderPanelRef.current = panel;
    draggedPanelRef.current = panel;
    reorderDragStartRef.current = { x: event.clientX, y: event.clientY };
    reorderDragActivatedRef.current = false;
    setReorderPreviewOrderValue(orderedPanelIndices);
    setDragOverPanel(panel);
    beginReorderDragVisual(event, panel);
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Some browser automation paths do not expose pointer capture.
    }
    event.preventDefault();
    event.stopPropagation();
  }, [beginReorderDragVisual, orderedPanelIndices, reorderMode, setReorderPreviewOrderValue]);
  const handlePanelReorderPointerEnter = React.useCallback((event: React.PointerEvent, panel: number) => {
    if (!reorderMode || pointerReorderPanelRef.current === null) return;
    if (dragOverPanel !== panel) setDragOverPanel(panel);
    event.stopPropagation();
  }, [dragOverPanel, reorderMode]);
  const handlePanelReorderPointerMove = React.useCallback((event: React.PointerEvent) => {
    if (!reorderMode || pointerReorderPanelRef.current === null) return;
    updateReorderGhostPosition(event.clientX, event.clientY);
    if (!reorderDragHasPassedThreshold(event.clientX, event.clientY)) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    previewPanelReorderFromPoint(event.clientX, event.clientY);
    event.preventDefault();
    event.stopPropagation();
  }, [previewPanelReorderFromPoint, reorderDragHasPassedThreshold, reorderMode, updateReorderGhostPosition]);
  const handlePanelReorderPointerUp = React.useCallback((event: React.PointerEvent) => {
    if (!reorderMode) return;
    updateReorderGhostPosition(event.clientX, event.clientY);
    if (reorderDragHasPassedThreshold(event.clientX, event.clientY)) {
      previewPanelReorderFromPoint(event.clientX, event.clientY);
      commitPanelReorderPreview();
    } else {
      cancelPanelReorderPreview();
    }
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // Ignore capture release failures from synthetic pointer streams.
    }
    event.preventDefault();
    event.stopPropagation();
  }, [cancelPanelReorderPreview, commitPanelReorderPreview, previewPanelReorderFromPoint, reorderDragHasPassedThreshold, reorderMode, updateReorderGhostPosition]);
  const resetPanelOrder = React.useCallback(() => {
    setPanelOrder([]);
    cancelPanelReorderPreview();
  }, [cancelPanelReorderPreview, setPanelOrder]);
  React.useEffect(() => {
    if (((nPanels || 1) <= 1 || isPaged) && reorderMode) setReorderMode(false);
  }, [nPanels, isPaged, reorderMode]);
  React.useEffect(() => {
    if (reorderMode) return;
    cancelPanelReorderPreview();
  }, [cancelPanelReorderPreview, reorderMode]);
  React.useEffect(() => () => {
    if (reorderGhostRafRef.current !== null) {
      window.cancelAnimationFrame(reorderGhostRafRef.current);
      reorderGhostRafRef.current = null;
    }
  }, []);
  const [maxCols, setMaxCols] = useModelState<number>("max_cols");
  const [linkPanels, setLinkPanels] = useModelState<boolean>("link_panels");
  const [showResizeHandles] = useModelState<boolean>("show_resize_handles");
  const allowResizeControls = showResizeHandles !== false;
  const [showZoomIndicator] = useModelState<boolean>("show_zoom_indicator");
  const [showPanelTitles] = useModelState<boolean>("show_panel_titles");
  const panelTitleFontSize = 11;
  const [legacyPanelGapTrait] = useModelState<number>("panel_gap");
  const [interPanelGapPxState] = useModelState<number>("inter_panel_gap_px");
  const [linkContrast, setLinkContrast] = useModelState<boolean>("link_contrast");
  const [cmap, setCmap] = useModelState<string>("cmap");
  // The model-backed setter can emit its React update after another synced
  // trait in the same control event. Direct WebGPU repaint paths must use the
  // palette selected by this event, not the palette from the previous render.
  const cmapLiveRef = React.useRef(cmap || "inferno");
  React.useEffect(() => {
    cmapLiveRef.current = cmap || "inferno";
  }, [cmap]);
  const [panelCmaps, setPanelCmaps] = useModelState<string[]>("panel_cmaps");
  const [panelAnnotations] = useModelState<PanelAnnotationSpec[][]>("panel_annotations");
  const [panelOverlays, setPanelOverlays] = useModelState<PanelOverlaySpec[][]>("panel_overlays");
  const panelGapPx = Math.max(0, Number.isFinite(interPanelGapPxState) ? interPanelGapPxState : (Number.isFinite(legacyPanelGapTrait) ? legacyPanelGapTrait : 0));
  const interPanelGapColor = String(themeColors.bg);
  const galleryOuterBorderPx = 0;
  const galleryOuterBorderColor = interPanelGapColor;
  const panelInnerBorderPx = 0;
  const panelInnerBorderColor = "#000000";
  const [flipRows, setFlipRows] = useModelState<boolean>("flip_vertical");
  const [flipCols, setFlipCols] = useModelState<boolean>("flip_horizontal");
  const [compareMode, setCompareMode] = useModelState<string>("compare_mode");
  const [comparePair, setComparePair] = useModelState<number[]>("compare_pair");
  const [blinkFps, setBlinkFps] = useModelState<number>("blink_fps");
  const [diffCmap, setDiffCmap] = useModelState<string>("diff_cmap");
  const [compareBackground, setCompareBackground] = useModelState<string>("compare_background");
  const [blinkPhase, setBlinkPhase] = React.useState(0);
  const normalizedPanelCmaps = React.useMemo(
    () => Array.isArray(panelCmaps) ? panelCmaps : [],
    [panelCmaps],
  );
  const panelCmapsLiveRef = React.useRef<string[]>(normalizedPanelCmaps);
  React.useEffect(() => {
    panelCmapsLiveRef.current = normalizedPanelCmaps;
  }, [normalizedPanelCmaps]);
  const panelCmapFor = React.useCallback((panelIdx: number) => {
    const value = normalizedPanelCmaps[panelIdx];
    return (value && COLORMAPS[value]) ? value : (cmap || "inferno");
  }, [normalizedPanelCmaps, cmap]);
  const hasMixedPanelCmaps = React.useMemo(() => {
    if (Math.max(1, nPanels || 1) <= 1) return false;
    if (normalizedPanelCmaps.length !== Math.max(1, nPanels || 1)) return false;
    const first = panelCmapFor(0);
    return normalizedPanelCmaps.some((_, idx) => panelCmapFor(idx) !== first);
  }, [normalizedPanelCmaps, nPanels, panelCmapFor]);
  const colorShared = normalizedPanelCmaps.length !== Math.max(1, nPanels || 1) || Math.max(1, nPanels || 1) <= 1;
  const setColorShared = React.useCallback((shared: boolean, panelIdx = 0) => {
    const panelCount = Math.max(1, nPanels || 1);
    if (shared || panelCount <= 1) {
      const value = panelCmapFor(panelIdx);
      cmapLiveRef.current = value;
      panelCmapsLiveRef.current = [];
      setCmap(value);
      setPanelCmaps([]);
      return;
    }
    const next = Array.from({ length: panelCount }, (_, idx) => panelCmapFor(idx));
    panelCmapsLiveRef.current = next;
    setPanelCmaps(next);
  }, [nPanels, panelCmapFor, setCmap, setPanelCmaps]);
  const setCmapForPanel = React.useCallback((panelIdx: number, value: string) => {
    const panelCount = Math.max(1, nPanels || 1);
    if (panelCount <= 1 || colorShared) {
      cmapLiveRef.current = value;
      setCmap(value);
      if (normalizedPanelCmaps.length > 0) setPanelCmaps([]);
      return;
    }
    const idx = Math.max(0, Math.min(panelCount - 1, Math.round(panelIdx)));
    const next = normalizedPanelCmaps.length === panelCount
      ? [...normalizedPanelCmaps]
      : Array.from({ length: panelCount }, () => cmap || "inferno");
    next[idx] = value;
    panelCmapsLiveRef.current = next;
    setPanelCmaps(next);
    if (idx === 0) setCmap(value);
  }, [cmap, colorShared, normalizedPanelCmaps, nPanels, setCmap, setPanelCmaps]);

  // Playback
  const [playing, setPlaying] = useModelState<boolean>("playing");
  const [reverse, setReverse] = useModelState<boolean>("reverse");
  const [boomerang, setBoomerang] = useModelState<boolean>("boomerang");
  const [fps, setFpsModel] = useModelState<number>("fps");
  const playbackFps = clampPlaybackFps(fps);
  const setPlaybackFps = React.useCallback((value: number) => {
    setFpsModel(clampPlaybackFps(value));
  }, [setFpsModel]);
  React.useEffect(() => {
    if (fps !== playbackFps) setFpsModel(playbackFps);
  }, [fps, playbackFps, setFpsModel]);
  const [loop, setLoop] = useModelState<boolean>("loop");
  const [loopStart, setLoopStart] = useModelState<number>("loop_start");
  const [loopEnd, setLoopEnd] = useModelState<number>("loop_end");
  const [bookmarkedFrames, setBookmarkedFrames] = useModelState<number[]>("bookmarked_frames");
  const [playbackPath, setPlaybackPath] = useModelState<number[]>("playback_path");

  // Boomerang direction ref (avoids stale closure in setInterval)
  const bounceDirRef = React.useRef<1 | -1>(1);

  // Stats
  const [showStats, setShowStats] = useModelState<boolean>("show_stats");
  // "More" overflow menu (mirrors Show2D): tucks Stats + Denoise off the crowded
  // top toolbar. Badge shows how many of its tools are active.
  const [moreMenuAnchor, setMoreMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [playbackStyleMenuAnchor, setPlaybackStyleMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [showRotationSettings, setShowRotationSettings] = React.useState(false);
  const [showControls] = useModelState<boolean>("show_controls");
  const [controlsCollapsed, setControlsCollapsed] = useModelState<boolean>("controls_collapsed");
  const controlsVisible = showControls && !controlsCollapsed;
  const panelChromeVisible = controlsVisible;
  const showResizeControls = allowResizeControls && panelChromeVisible;
  const toolControlsRef = React.useRef<HTMLDivElement>(null);
  const [toolControlsHeight, setToolControlsHeight] = React.useState(28);
  const resizeGripSx = React.useMemo(() => ({
    width: 16,
    height: 16,
    cursor: "nwse-resize",
    opacity: 0.6,
    background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
    touchAction: "none",
    zIndex: 5,
    "&:hover": { opacity: 1 },
  }), [themeColors.accent]);
  const [statsMean] = useModelState<number>("stats_mean");
  const [statsMin] = useModelState<number>("stats_min");
  const [statsMax] = useModelState<number>("stats_max");
  const [statsStd] = useModelState<number>("stats_std");

  React.useLayoutEffect(() => {
    if (!controlsVisible) {
      setToolControlsHeight(28);
      return;
    }
    const element = toolControlsRef.current;
    if (!element) return;
    const measure = () => {
      const next = Math.max(28, element.getBoundingClientRect().height);
      setToolControlsHeight((current) => (Math.abs(current - next) < 0.5 ? current : next));
    };
    measure();
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    observer?.observe(element);
    window.addEventListener("resize", measure);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [controlsVisible]);

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [percentileLow, setPercentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh, setPercentileHigh] = useModelState<number>("percentile_high");
  const [traitVmin] = useModelState<number | null>("vmin");
  const [traitVmax] = useModelState<number | null>("vmax");
  const [imageVminPct, setImageVminPct] = useModelState<number>("image_vmin_pct");
  const [imageVmaxPct, setImageVmaxPct] = useModelState<number>("image_vmax_pct");
  const [contrastPreset, setContrastPreset] = useModelState<string>("contrast_preset");
  const manualImageRangeBeforeAutoRef = React.useRef<{ min: number; max: number } | null>(null);
  const [vminPerPanel, setVminPerPanel] = useModelState<(number | null)[]>("vmin_per_panel");
  const [vmaxPerPanel, setVmaxPerPanel] = useModelState<(number | null)[]>("vmax_per_panel");
  const vminPerPanelLiveRef = React.useRef<(number | null)[]>(vminPerPanel);
  const vmaxPerPanelLiveRef = React.useRef<(number | null)[]>(vmaxPerPanel);
  React.useEffect(() => {
    vminPerPanelLiveRef.current = vminPerPanel;
  }, [vminPerPanel]);
  React.useEffect(() => {
    vmaxPerPanelLiveRef.current = vmaxPerPanel;
  }, [vmaxPerPanel]);
  const [dataMin] = useModelState<number>("data_min");
  const [dataMax] = useModelState<number>("data_max");
  const [autoVmins] = useModelState<number[]>("auto_vmins");
  const [autoVmaxs] = useModelState<number[]>("auto_vmaxs");
  const [autoVminsPerPanel] = useModelState<number[]>("auto_vmins_per_panel");
  const [autoVmaxsPerPanel] = useModelState<number[]>("auto_vmaxs_per_panel");
  const initialAutoPercentilesRef = React.useRef({
    low: percentileLow,
    high: percentileHigh,
  });
  React.useEffect(() => {
    if (compareMode !== "blink") {
      setBlinkPhase(0);
      return;
    }
    const intervalMs = 1000 / Math.max(0.25, Number(blinkFps || 2));
    const id = window.setInterval(() => setBlinkPhase((phase) => (phase + 1) % 2), intervalMs);
    return () => window.clearInterval(id);
  }, [blinkFps, compareMode]);
  // Scale bar
  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  // Display-only filter knobs for sparse map stacks (EDS, low dose). The
  // browser filters the displayed frame; raw data is never modified.
  const [displayFilter, setDisplayFilter] = useModelState<string>("denoise");
  const [displaySigma, setDisplaySigma] = useModelState<number>("denoise_sigma");
  const [spatialBin, setSpatialBin] = useModelState<number>("denoise_bin");
  const [displayFilters, setDisplayFilters] = useModelState<string[]>("denoise_modes");
  const [displaySigmas, setDisplaySigmas] = useModelState<number[]>("denoise_sigmas");
  const [spatialBins, setSpatialBins] = useModelState<number[]>("denoise_bins");
  const [denoiseScope, setDenoiseScope] = useModelState<string>("denoise_scope");
  const [displayFilterBanner] = useModelState<string>("denoise_banner");
  const [showDenoise, setShowDenoise] = useModelState<boolean>("show_denoise");
  // Master ON/OFF of the denoise EFFECT (off -> raw, config preserved & gated).
  const [denoiseEnabled, setDenoiseEnabled] = useModelState<boolean>("denoise_enabled");
  const [frequencyFilter, setFrequencyFilter] = useModelState<string>("frequency_filter");
  const [frequencyFilterEnabled, setFrequencyFilterEnabled] = useModelState<boolean>("frequency_filter_enabled");
  const [frequencyFilterCutoff, setFrequencyFilterCutoff] = useModelState<number>("frequency_filter_cutoff");
  const [frequencyFilterCenter, setFrequencyFilterCenter] = useModelState<number>("frequency_filter_center");
  const [frequencyFilterWidth, setFrequencyFilterWidth] = useModelState<number>("frequency_filter_width");
  const [frequencyFilterModes, setFrequencyFilterModes] = useModelState<string[]>("frequency_filter_modes");
  const [frequencyFilterCutoffs, setFrequencyFilterCutoffs] = useModelState<number[]>("frequency_filter_cutoffs");
  const [frequencyFilterCenters, setFrequencyFilterCenters] = useModelState<number[]>("frequency_filter_centers");
  const [frequencyFilterWidths, setFrequencyFilterWidths] = useModelState<number[]>("frequency_filter_widths");
  const [frequencyFilterScope, setFrequencyFilterScope] = useModelState<string>("frequency_filter_scope");
  const [showFrequencyFilter, setShowFrequencyFilter] = useModelState<boolean>("show_frequency_filter");
  const [subpixelAlignEnabled, setSubpixelAlignEnabled] = useModelState<boolean>("subpixel_align_enabled");
  const [subpixelAlignReference, setSubpixelAlignReference] = useModelState<number>("subpixel_align_reference");
  const [subpixelAlignStatus, setSubpixelAlignStatus] = React.useState("Off");
  const [subpixelAlignBusy, setSubpixelAlignBusy] = React.useState(false);
  const [subpixelAlignVersion, setSubpixelAlignVersion] = React.useState(0);
  const subpixelAlignShiftsRef = React.useRef<SubpixelShift[] | null>(null);
  const subpixelAlignCacheRef = React.useRef<Map<string, Float32Array>>(new Map());
  const subpixelAlignSerialRef = React.useRef(0);
  const [frequencyDraft, setFrequencyDraft] = React.useState<number | null>(null);
  const [frequencyRenderVersion, setFrequencyRenderVersion] = React.useState(0);
  const [frequencyFilterBackend, setFrequencyFilterBackend] = React.useState("off");
  const frequencyFilterCacheRef = React.useRef<Map<string, Float32Array>>(new Map());
  const frequencyFilterPendingRef = React.useRef<Set<string>>(new Set());
  const frequencyOptions = React.useMemo(() => {
    const mode = normalizeFrequencyFilterMode(frequencyFilter);
    return {
      mode,
      cutoff: frequencyDraft ?? frequencyFilterCutoff,
      center: mode === "bandpass" ? (frequencyDraft ?? frequencyFilterCenter) : frequencyFilterCenter,
      width: frequencyFilterWidth,
    };
  }, [frequencyFilter, frequencyDraft, frequencyFilterCutoff, frequencyFilterCenter, frequencyFilterWidth]);
  const scopedPanelForEdit = React.useMemo(() => {
    const fallback = visiblePanelIndices[0] ?? 0;
    const selected = selectedVisiblePanels[selectedVisiblePanels.length - 1] ?? fallback;
    return Math.max(0, Math.min(Math.max(0, (nPanels || 1) - 1), selected));
  }, [nPanels, selectedVisiblePanels, visiblePanelIndices]);
  const denoiseScopeAll = String(denoiseScope || "all") === "all" || (nPanels || 1) <= 1;
  const frequencyFilterScopeAll = String(frequencyFilterScope || "all") === "all" || (nPanels || 1) <= 1;
  const updateScopedArray = React.useCallback(<T,>(
    values: T[] | undefined,
    nextValue: T,
    fallback: T,
    scopeAll: boolean,
  ) => {
    const count = Math.max(1, nPanels || 1);
    const current = Array.from({ length: count }, (_, idx) => values?.[idx] ?? fallback);
    if (scopeAll) return current.map(() => nextValue);
    current[scopedPanelForEdit] = nextValue;
    return current;
  }, [nPanels, scopedPanelForEdit]);
  const denoiseKnobsForPanel = React.useCallback((panel: number) => {
    const idx = Math.max(0, Math.min(Math.max(0, (nPanels || 1) - 1), panel));
    const mode = denoiseScopeAll ? displayFilter : (displayFilters?.[idx] ?? displayFilter);
    return {
      mode: normalizeFilterMode(mode || "none"),
      sigma: denoiseScopeAll ? Number(displaySigma ?? 4) : Number(displaySigmas?.[idx] ?? displaySigma ?? 4),
      bin: denoiseScopeAll ? Number(spatialBin || 1) : Number(spatialBins?.[idx] ?? spatialBin ?? 1),
    };
  }, [denoiseScopeAll, displayFilter, displayFilters, displaySigma, displaySigmas, nPanels, spatialBin, spatialBins]);
  const frequencyKnobsForPanel = React.useCallback((panel: number) => {
    const idx = Math.max(0, Math.min(Math.max(0, (nPanels || 1) - 1), panel));
    const mode = normalizeFrequencyFilterMode(frequencyFilterScopeAll ? frequencyFilter : (frequencyFilterModes?.[idx] ?? frequencyFilter));
    return {
      mode,
      cutoff: frequencyFilterScopeAll ? Number(frequencyFilterCutoff ?? 0.15) : Number(frequencyFilterCutoffs?.[idx] ?? frequencyFilterCutoff ?? 0.15),
      center: frequencyFilterScopeAll ? Number(frequencyFilterCenter ?? 0.30) : Number(frequencyFilterCenters?.[idx] ?? frequencyFilterCenter ?? 0.30),
      width: frequencyFilterScopeAll ? Number(frequencyFilterWidth ?? 0.12) : Number(frequencyFilterWidths?.[idx] ?? frequencyFilterWidth ?? 0.12),
    };
  }, [frequencyFilter, frequencyFilterCenter, frequencyFilterCenters, frequencyFilterCutoff, frequencyFilterCutoffs, frequencyFilterModes, frequencyFilterScopeAll, frequencyFilterWidth, frequencyFilterWidths, nPanels]);
  const syncDenoisePanelKnob = React.useCallback((name: "mode" | "sigma" | "bin", value: string | number) => {
    if (name === "mode") setDisplayFilters(updateScopedArray(displayFilters, String(value), "none", denoiseScopeAll));
    else if (name === "sigma") setDisplaySigmas(updateScopedArray(displaySigmas, Number(value), 4, denoiseScopeAll));
    else setSpatialBins(updateScopedArray(spatialBins, Number(value), 1, denoiseScopeAll));
  }, [denoiseScopeAll, displayFilters, displaySigmas, setDisplayFilters, setDisplaySigmas, setSpatialBins, spatialBins, updateScopedArray]);
  const syncFrequencyPanelKnob = React.useCallback((name: "mode" | "cutoff" | "center" | "width", value: string | number) => {
    if (name === "mode") setFrequencyFilterModes(updateScopedArray(frequencyFilterModes, String(value), "none", frequencyFilterScopeAll));
    else if (name === "cutoff") setFrequencyFilterCutoffs(updateScopedArray(frequencyFilterCutoffs, Number(value), 0.15, frequencyFilterScopeAll));
    else if (name === "center") setFrequencyFilterCenters(updateScopedArray(frequencyFilterCenters, Number(value), 0.30, frequencyFilterScopeAll));
    else setFrequencyFilterWidths(updateScopedArray(frequencyFilterWidths, Number(value), 0.12, frequencyFilterScopeAll));
  }, [frequencyFilterScopeAll, frequencyFilterCenters, frequencyFilterCutoffs, frequencyFilterModes, frequencyFilterWidths, setFrequencyFilterCenters, setFrequencyFilterCutoffs, setFrequencyFilterModes, setFrequencyFilterWidths, updateScopedArray]);
  const frequencyFilterIsActive = !!frequencyFilterEnabled && !isRgb && (
    frequencyFilterScopeAll
      ? frequencyFilterActive(frequencyFilter)
      : Array.from({ length: Math.max(1, nPanels || 1) }, (_, panel) => frequencyKnobsForPanel(panel))
          .some((knobs) => frequencyFilterActive(knobs.mode))
  );
  const frequencyValueLabel = React.useCallback((value: number) => {
    const unit = String(pixelUnit || "").trim().toLowerCase();
    if (pixelSize > 0 && (unit === "nm" || unit.includes("nanometer"))) return `${(value / (2 * pixelSize)).toFixed(3)} nm⁻¹`;
    if (pixelSize > 0 && (unit === "a" || unit === "å" || unit.includes("angstrom"))) return `${(value * 10 / (2 * pixelSize)).toFixed(3)} nm⁻¹`;
    return `${value.toFixed(3)} Nyq`;
  }, [pixelSize, pixelUnit]);
  const setFrequencyMaster = (enabled: boolean) => {
    if (enabled && !frequencyFilterActive(frequencyFilter)) {
      setFrequencyFilter("lowpass");
      syncFrequencyPanelKnob("mode", "lowpass");
    }
    setFrequencyFilterEnabled(enabled);
    setShowFrequencyFilter(enabled); // reveal the settings row while filtering; hide it when off (mirrors Denoise)
  };
  // Local slider value during drag; the model only updates on release so
  // scrubbing sigma stays smooth on large stacks.
  const [sigmaDraft, setSigmaDraft] = React.useState<number | null>(null);
  const displayFilterOff = normalizeFilterMode(displayFilter || "none") === "none";
  // Python ships raw frames; the display filter (WGSL, or its CPU port without
  // WebGPU) applies gaussian/bin/anscombe here, so dragging sigma is live with
  // zero kernel round trips. True-color stacks are not filtered.
  const browserFilterActive = !isRgb && (denoiseEnabled ?? true);
  const denoiseResolved = { mode: normalizeFilterMode(displayFilter || "none"), bin: spatialBin || 1 };
  const denoiseSigmaLive = sigmaDraft ?? Number(displaySigma ?? 4);
  const browserFilterKnobsOn = browserFilterActive
    && (denoiseScopeAll
      ? filterKnobsActive(denoiseResolved.mode, denoiseResolved.bin) && browserFilterSupported(denoiseResolved.mode)
      : Array.from({ length: Math.max(1, nPanels || 1) }, (_, panel) => denoiseKnobsForPanel(panel))
          .some((knobs) => filterKnobsActive(knobs.mode, knobs.bin) && browserFilterSupported(knobs.mode)));
  // Filtered-frame cache keyed on frame_seq as well as the logical index and
  // view knobs: a stack replaced by set_image keeps the same frame indices, and
  // without frame_seq it would repaint filtered frames of the previous stack.
  const browserFilterCacheRef = React.useRef<Map<string, Float32Array>>(new Map());
  const browserFilterPendingRef = React.useRef<Set<string>>(new Set());
  const [browserFilterTick, setBrowserFilterTick] = React.useState(0);
  const applyPackedPanelTransform = React.useCallback(async (
    frame: Float32Array,
    transform: (panelFrame: Float32Array, panelWidth: number, panelHeight: number, panel: number) => Promise<Float32Array>,
  ): Promise<Float32Array> => {
    const panelCount = Math.max(1, nPanels || 1);
    if (panelCount <= 1 || sharedPanelSource || width % panelCount !== 0) {
      return transform(frame, width, height, 0);
    }
    const panelWidth = width / panelCount;
    const output = new Float32Array(frame.length);
    for (let panel = 0; panel < panelCount; panel++) {
      const panelFrame = new Float32Array(panelWidth * height);
      const srcColStart = panel * panelWidth;
      for (let row = 0; row < height; row++) {
        const srcOffset = row * width + srcColStart;
        const dstOffset = row * panelWidth;
        panelFrame.set(frame.subarray(srcOffset, srcOffset + panelWidth), dstOffset);
      }
      const filtered = await transform(panelFrame, panelWidth, height, panel);
      for (let row = 0; row < height; row++) {
        const srcOffset = row * panelWidth;
        const dstOffset = row * width + srcColStart;
        output.set(filtered.subarray(srcOffset, srcOffset + panelWidth), dstOffset);
      }
    }
    return output;
  }, [height, nPanels, sharedPanelSource, width]);
  // Live gate read inside memoized render ticks (avoids stale closures): when on,
  // denoise is treated as a client frame transform so every path routes through
  // displayFrameForIndex and skips the raw GPU-slot cache.
  const browserFilterOnRef = React.useRef(false);
  browserFilterOnRef.current = browserFilterKnobsOn;
  // Return a filtered copy of `frame` for DISPLAY only, keyed on the live knobs.
  // The GPU filter is async, so on a cache miss we return the raw frame (or
  // null when the caller cannot show raw pixels) and repaint once the filtered
  // result lands (setBrowserFilterTick).
  const scopedDenoiseKey = Array.from({ length: Math.max(1, nPanels || 1) }, (_, panel) => {
    const knobs = denoiseKnobsForPanel(panel);
    return `${panel}:${knobs.mode}:${Number(knobs.sigma).toFixed(2)}:${Math.round(knobs.bin)}`;
  }).join("|");
  const browserFilterCacheKeyForIndex = React.useCallback((idx: number) => {
    return browserFilterCacheKey({
      frameIndex: idx,
      frameSeq,
      mode: denoiseScopeAll ? denoiseResolved.mode : scopedDenoiseKey,
      sigma: denoiseScopeAll ? denoiseSigmaLive : 0,
      bin: denoiseScopeAll ? denoiseResolved.bin : 1,
      avgWindow: playRef.current.avgWindow,
      diffMode: playRef.current.diffMode,
      panels: (Math.max(1, nPanels || 1) > 1 && !sharedPanelSource) ? Math.max(1, nPanels || 1) : 1,
    });
  }, [denoiseResolved.mode, denoiseResolved.bin, denoiseScopeAll, denoiseSigmaLive, frameSeq, nPanels, scopedDenoiseKey, sharedPanelSource]);

  const browserFilterReadyForIndex = React.useCallback((idx: number) => {
    if (!browserFilterKnobsOn) return true;
    return browserFilterCacheRef.current.has(browserFilterCacheKeyForIndex(idx));
  }, [browserFilterCacheKeyForIndex, browserFilterKnobsOn]);

  const browserFilterFrame = React.useCallback((idx: number, frame: Float32Array | null, options: { allowRawOnMiss?: boolean } = {}): Float32Array | null => {
    if (!frame || !browserFilterKnobsOn) return frame;
    const allowRawOnMiss = options.allowRawOnMiss === true;
    const key = browserFilterCacheKeyForIndex(idx);
    const cache = browserFilterCacheRef.current;
    const hit = cache.get(key);
    if (hit) return hit;
    if (browserFilterPendingRef.current.has(key)) return allowRawOnMiss ? frame : null;
    browserFilterPendingRef.current.add(key);
    applyPackedPanelTransform(
      frame,
      (panelFrame, panelWidth, panelHeight, panel) => {
        const knobs = denoiseScopeAll ? { mode: denoiseResolved.mode, sigma: denoiseSigmaLive, bin: denoiseResolved.bin } : denoiseKnobsForPanel(panel);
        if (!filterKnobsActive(knobs.mode, knobs.bin)) return Promise.resolve(panelFrame);
        return applyDisplayFilterBrowser(panelFrame, panelWidth, panelHeight, knobs.mode, knobs.sigma, knobs.bin);
      },
    )
      .then((filtered) => {
        browserFilterPendingRef.current.delete(key);
        cache.set(key, filtered);
        if (cache.size > 48) cache.delete(cache.keys().next().value as string);
        setBrowserFilterTick((t) => t + 1);
      })
      .catch(() => { browserFilterPendingRef.current.delete(key); });
    return allowRawOnMiss ? frame : null;
  }, [applyPackedPanelTransform, browserFilterCacheKeyForIndex, browserFilterKnobsOn, denoiseKnobsForPanel, denoiseResolved.mode, denoiseResolved.bin, denoiseScopeAll, denoiseSigmaLive, height, width]);
  // The "Denoise" toggle is the master ON/OFF of the EFFECT: ON shows the
  // denoised view, OFF shows raw (nothing of the denoised view leaks through).
  // The config (mode/sigma/bin) is PRESERVED across the toggle; a clean widget
  // gets a visible gaussian (σ 4) the first time it is enabled.
  const toggleDenoise = () => {
    const next = !denoiseEnabled;
    setDenoiseEnabled(next);
    setShowDenoise(next); // editor follows: shown while denoising, hidden when raw
    if (next && displayFilterOff) {
      setDisplayFilter("gaussian");
      syncDenoisePanelKnob("mode", "gaussian");
    }
    // Turning OFF preserves the config; browserFilterActive gates the display.
  };
  const [imageRotation, setImageRotation] = useModelState<number>("image_rotation");
  const [rotationScope, setRotationScope] = useModelState<string>("rotation_scope");
  const [frameRotations, setFrameRotations] = useModelState<number[]>("frame_rotations");
  const normalizeRotation = React.useCallback((value: number) => {
    const quarterTurns = Math.round(Number(value) / 90);
    if (Number.isFinite(quarterTurns) && Math.abs(Number(value)) > 3) return ((quarterTurns % 4) + 4) % 4;
    return ((Math.round(Number(value)) % 4) + 4) % 4;
  }, []);

  // Customization
  const [canvasSizeTrait, setCanvasSizeTrait] = useModelState<number>("size");

  // ROI
  const [roiActive, setRoiActive] = useModelState<boolean>("roi_active");
  const [roiList, setRoiList] = useModelState<ROIItem[]>("roi_list");
  const [roiSelectedIdx, setRoiSelectedIdx] = useModelState<number>("roi_selected_idx");
  const [roiPlotData] = useModelState<DataView>("roi_plot_data");
  const [newRoiShape, setNewRoiShape] = React.useState<"circle" | "square" | "rectangle" | "annular">("square");

  // Diff mode
  const [diffMode, setDiffMode] = useModelState<string>("diff_mode");
  const [avgWindow, setAvgWindow] = useModelState<number>("avg_window");

  // FFT
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftLayout, setFftLayout] = useModelState<string>("fft_layout");
  const [fftOverlayPosition, setFftOverlayPosition] = useModelState<string>("fft_overlay_position");
  const [fftOverlaySize, setFftOverlaySize] = useModelState<number>("fft_overlay_size");
  const [fftOverlayZoomTrait, setFftOverlayZoomTrait] = useModelState<number>("fft_overlay_zoom");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const fftMetricsEnabled = true;
  const resolvedFftLayout = (["bottom", "right", "overlay"].includes(String(fftLayout)) ? String(fftLayout) : "bottom") as "bottom" | "right" | "overlay";
  const fftLayoutBottom = resolvedFftLayout === "bottom";
  const fftLayoutOverlay = resolvedFftLayout === "overlay";
  const resolvedFftOverlayPosition = (["top-left", "top-right", "bottom-left", "bottom-right"].includes(String(fftOverlayPosition)) ? String(fftOverlayPosition) : "top-left") as FftOverlayPosition;
  const resolvedFftOverlaySize = Math.max(0.2, Math.min(0.7, Number.isFinite(fftOverlaySize) ? fftOverlaySize : 0.35));
  const resolvedFftOverlayZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, Number.isFinite(fftOverlayZoomTrait) ? fftOverlayZoomTrait : 1));


  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportPayloadFilename] = useModelState<string>("export_filename");

  // Canvas refs
  const rootRef = React.useRef<HTMLDivElement>(null);
  const [rootLayoutWidth, setRootLayoutWidth] = React.useState(0);
  React.useLayoutEffect(() => {
    const element = rootRef.current;
    if (!element) return;
    const measure = () => {
      const next = Math.max(0, Math.floor(element.getBoundingClientRect().width));
      setRootLayoutWidth((current) => (Math.abs(current - next) < 2 ? current : next));
    };
    measure();
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    observer?.observe(element);
    window.addEventListener("resize", measure);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, []);
  const hasOfflineStack = !!offlineStack && offlineStack.byteLength > 0;
  const hasOfflineFloatStack = !!offlineFloatStack && offlineFloatStack.byteLength > 0;
  const canRenderLive = hasOfflineStack || hasOfflineFloatStack;
  const staticFallbackUrl = staticFallbackJpeg
    ? `data:${staticFallbackMime || "image/jpeg"};base64,${staticFallbackJpeg}`
    : "";
  const hasSavedStaticFallback = staticFallbackUrl.length > 0;
  useHideStaticFallback(
    model,
    rootRef,
    folderWaiting || canRenderLive || hasSavedStaticFallback,
  );
  const gpuCanvasCtxRef = React.useRef<GPUCanvasContext | null>(null);
  const gpuCanvasSizeRef = React.useRef<{ w: number; h: number } | null>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const uiRef = React.useRef<HTMLCanvasElement>(null);
  const canvasWheelHandlerRef = React.useRef<((event: WheelEvent) => void) | null>(null);
  const fftInsetNativeWheelHandlerRef = React.useRef<((event: WheelEvent) => boolean) | null>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftInsetLayerRef = React.useRef<HTMLCanvasElement>(null);

  const [exportMenuAnchor, setExportMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [panelMenuAnchor, setPanelMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [exportBusy, setExportBusy] = React.useState(false);
  const [exportPanelMode, setExportPanelMode] = React.useState<ExportPanelMode>("home");
  const [exportQuality, setExportQuality] = React.useState<AnimationQuality>("medium");
  const [exportFrameStart, setExportFrameStart] = React.useState(1);
  const [exportFrameEnd, setExportFrameEnd] = React.useState(Math.max(1, nSlices || 1));
  const [exportEveryN, setExportEveryN] = React.useState(1);
  const [exportMaxFrames, setExportMaxFrames] = React.useState(40);
  const [exportFps, setExportFps] = React.useState(DEFAULT_ANIMATION_EXPORT_FPS);
  const [exportSpatialPreset, setExportSpatialPreset] = React.useState<ExportSpatialPreset>("edge512");
  const [exportGifPreset, setExportGifPreset] = React.useState<GifExportPreset>("slides");
  const [localExportStatus, setLocalExportStatus] = React.useState("");
  const fftOverlayDragRef = React.useRef<{
    pointerId: number;
    startClientX: number;
    startClientY: number;
    startInsetX: number;
    startInsetY: number;
    panelLeft: number;
    panelTop: number;
    panelW: number;
    panelH: number;
    insetW: number;
    insetH: number;
    moved: boolean;
  } | null>(null);
  const [fftOverlayDragPreview, setFftOverlayDragPreview] = React.useState<{ x: number; y: number } | null>(null);
  const pendingExportRef = React.useRef<{
    id: string;
    filename: string;
    mode: string;
    downsample: number;
    handle: Show3DFileHandle | null;
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
    if (!localExportStatus || exportBusy) return;
    if (localExportStatus.startsWith("Preparing ") || localExportStatus.startsWith("Saving ")) return;
    const id = window.setTimeout(() => {
      setLocalExportStatus((current) => current === localExportStatus ? "" : current);
    }, 12000);
    return () => window.clearTimeout(id);
  }, [localExportStatus, exportBusy]);
  const voxelCount = Math.max(0, Math.floor(nSlices) * Math.floor(height) * Math.floor(width));
  const exactExportSize = formatEstimatedHtmlSize(voxelCount * 4);
  const quantizedExportSize = formatEstimatedHtmlSize(voxelCount);
  const quantizedExportSize2 = formatEstimatedHtmlSize(Math.ceil(voxelCount / 4));
  const quantizedExportSize4 = formatEstimatedHtmlSize(Math.ceil(voxelCount / 16));
  const quantizedExportSize8 = formatEstimatedHtmlSize(Math.ceil(voxelCount / 64));
  const selectedSpatialOption = spatialOptionFor(exportSpatialPreset);
  const animationPanelWidth = sharedPanelSource
    ? Math.max(1, width)
    : Math.max(1, panelWidthPx || Math.floor(width / Math.max(1, nPanels || 1)) || width);
  const animationPanelHeight = Math.max(1, height);
  const exportFrameIndices = React.useMemo(
    () => buildAnimationFrameIndices(nSlices, exportFrameStart, exportFrameEnd, exportEveryN, exportMaxFrames),
    [nSlices, exportFrameStart, exportFrameEnd, exportEveryN, exportMaxFrames],
  );
  const animationWorkEstimate = formatEstimatedAnimationWork(
    animationPanelWidth,
    animationPanelHeight,
    exportFrameIndices.length,
    visiblePanelCount,
    maxCols,
    panelGapPx,
    exportQuality,
    selectedSpatialOption.downsample,
    selectedSpatialOption.maxEdgePx,
  );
  const exportFpsValue = Math.max(1, Math.round(exportFps || DEFAULT_ANIMATION_EXPORT_FPS));
  const exportDurationSeconds = exportFrameIndices.length / exportFpsValue;
  const exportFrameSummary = `${exportFrameIndices.length}/${Math.max(1, nSlices || 1)} frames · ${exportDurationSeconds.toFixed(1)} s at ${exportFpsValue} fps`;
  const animationExportRequest = React.useMemo(() => ({
    fps: exportFpsValue,
    frame_start: exportFrameIndices[0] ?? 0,
    frame_stop: (exportFrameIndices[exportFrameIndices.length - 1] ?? 0) + 1,
    every_n: Math.max(1, Math.round(exportEveryN || 1)),
    max_frames: Math.max(0, Math.round(exportMaxFrames || 0)),
    downsample: selectedSpatialOption.downsample,
    max_edge_px: selectedSpatialOption.maxEdgePx || null,
    preset: exportGifPreset === "custom" ? "custom" : exportGifPreset,
    slides_preset: exportGifPreset === "slides",
    show_panel_titles: showPanelTitles !== false,
    show_scale_bar: Boolean(scaleBarVisible),
    show_zoom: showZoomIndicator === true,
  }), [
    exportFpsValue,
    exportFrameIndices,
    exportEveryN,
    exportMaxFrames,
    selectedSpatialOption.downsample,
    selectedSpatialOption.maxEdgePx,
    exportGifPreset,
    showPanelTitles,
    scaleBarVisible,
    showZoomIndicator,
  ]);
  const canDownloadCurrentHtml = !exportEnabled;
  const standaloneHtmlMode = hasOfflineFloatStack ? "exact" : "quantized";
  const standaloneHtmlLabel = standaloneHtmlMode === "quantized"
    ? `HTML encoded uint8 (${quantizedExportSize})`
    : `HTML exact float32 (${exactExportSize})`;
  const canExportStandaloneGif = !exportEnabled && width > 0 && height > 0 && nSlices > 0 && (
    hasOfflineStack || hasOfflineFloatStack
  );
  const standaloneGifUnavailableTitle = "GIF export needs embedded standalone image data.";
  const standaloneAnimationUsesEncodedUint8 = !exportEnabled && hasOfflineStack && !hasOfflineFloatStack;
  const standaloneAnimationSourceNote = exportEnabled
    ? "Source: live Python widget can export from the original stack."
    : hasOfflineFloatStack
      ? "Source: exact float32 embedded data."
      : hasOfflineStack
        ? "Source: encoded uint8 standalone data. For best movie fidelity, export/open HTML exact float32 from the live widget."
        : "";
  const standaloneAnimationQualityWarning = standaloneAnimationUsesEncodedUint8
    ? 'Warning: this standalone HTML stores encoded uint8 frames, not the original float32 stack. GIF adds a 256-color palette step and may change noisy or continuous colormaps. For publication-quality movies, export from the live widget or open an HTML exact float32 export with encoding="full".'
    : "";
  const handleExportMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setExportPanelMode("home");
    setExportMenuAnchor(event.currentTarget);
  };
  const handleExportMenuClose = () => {
    setExportMenuAnchor(null);
  };
  React.useEffect(() => {
    const total = Math.max(1, Math.floor(nSlices || 1));
    setExportFrameStart((current) => Math.max(1, Math.min(total, Math.round(current || 1))));
    setExportFrameEnd((current) => Math.max(1, Math.min(total, Math.round(current || total))));
  }, [nSlices]);
  const handleExportSelect = async (
    mode: string,
    quality = "medium",
    downsample = 1,
    requestOptions: Record<string, unknown> = {},
  ) => {
    setExportMenuAnchor(null);
    if (mode !== "exact" && mode !== "quantized" && mode !== "gif") return;
    const filename = makeExportFilename(title, nSlices, height, width, mode, quality, downsample);
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setExportBusy(true);
    setLocalExportStatus("Choose export location...");
    const picker = (window as Show3DWindow).showSaveFilePicker;
    let handle: Show3DFileHandle | null = null;
    if (picker) {
      try {
        handle = await picker({
          suggestedName: filename,
          types: [exportPickerType(mode)],
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
    pendingExportRef.current = { id, filename, mode, downsample, handle };
    setLocalExportStatus(`Preparing ${filename}...`);
    setExportRequest(JSON.stringify({ mode, quality, downsample, ...requestOptions, id, filename, download: true }));
  };
  const handleStandaloneHtmlDownload = () => {
    setExportMenuAnchor(null);
    const filename = makeExportFilename(title, nSlices, height, width, standaloneHtmlMode);
    try {
      const html = `<!doctype html>\n${standaloneHtmlWithCurrentWidgetState(
        model,
        standaloneWidgetStaticHtmlFromDocument(),
        SHOW3D_STANDALONE_VIEW_STATE_KEYS,
      )}`;
      const blob = new Blob([html], { type: "text/html;charset=utf-8" });
      downloadBlob(blob, filename);
      setLocalExportStatus(`Downloaded ${filename} to browser Downloads (${formatSavedBytes(blob.size)})`);
    } catch (err) {
      setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };
  const renderStandaloneAnimationCanvas = (
    frameIdx: number,
    quality: string,
    options: { downsample: number; maxEdgePx: number | null },
  ): { width: number; height: number; canvas: HTMLCanvasElement } | null => {
    const frame = getOfflineFrame(frameIdx);
    if (!frame) return null;
    const sourcePanelCount = Math.max(1, nPanels || 1);
    const panelW = sharedPanelSource ? Math.max(1, width) : Math.max(1, panelWidthPx || Math.floor(width / sourcePanelCount) || width);
    const panelH = Math.max(1, height);
    const panels = (visiblePanelIndices.length ? visiblePanelIndices : [0])
      .filter((panel) => panel >= 0 && panel < sourcePanelCount);
    const activePanels = panels.length ? panels : [0];
    const cols = panelColsForCount(activePanels.length);
    const rows = Math.max(1, Math.ceil(activePanels.length / cols));
    const scale = animationOutputScale(
      panelW,
      panelH,
      quality,
      options.downsample,
      options.maxEdgePx,
      activePanels.length,
      cols,
      panelGapPx,
    );
    const panelOutW = Math.max(1, Math.round(panelW * scale));
    const panelOutH = Math.max(1, Math.round(panelH * scale));
    const gap = activePanels.length > 1 ? Math.max(0, Math.round((panelGapPx) * scale)) : 0;
    const outer = Math.max(0, Math.round(galleryOuterBorderPx * scale));
    const innerBorder = Math.max(0, Math.round(panelInnerBorderPx * scale));
    const outW = cols * panelOutW + Math.max(0, cols - 1) * gap + 2 * outer;
    const outH = rows * panelOutH + Math.max(0, rows - 1) * gap + 2 * outer;
    const out = document.createElement("canvas");
    out.width = outW;
    out.height = outH;
    const outCtx = out.getContext("2d");
    if (!outCtx) return null;
    outCtx.imageSmoothingEnabled = smooth;
    outCtx.fillStyle = galleryOuterBorderPx > 0 ? galleryOuterBorderColor : interPanelGapColor;
    outCtx.fillRect(0, 0, outW, outH);
    if (gap > 0) {
      outCtx.fillStyle = interPanelGapColor;
      outCtx.fillRect(outer, outer, Math.max(0, outW - 2 * outer), Math.max(0, outH - 2 * outer));
    }

    const panelCanvas = document.createElement("canvas");
    panelCanvas.width = panelW;
    panelCanvas.height = panelH;
    const panelCtx = panelCanvas.getContext("2d");
    if (!panelCtx) return null;
    const panelImage = panelCtx.createImageData(panelW, panelH);
    const fallbackLut = COLORMAPS[cmap] || COLORMAPS.inferno;
    let sharedAutoRange: { vmin: number; vmax: number } | null = null;
    if (autoContrast && linkContrast) {
      sharedAutoRange =
        cachedAutoDisplayRange(autoVmins, autoVmaxs, frameIdx, logScale) ||
        cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, frameIdx, logScale);
      if (!sharedAutoRange) {
        const data = logScale ? applyLogScale(frame) : frame;
        sharedAutoRange = percentileClip(data, percentileLow, percentileHigh);
      }
    }

    for (let slot = 0; slot < activePanels.length; slot++) {
      const panel = activePanels[slot];
      const panelData = extractPanelSlice(frame, panel, logScale);
      if (!panelData) continue;
      const lut = COLORMAPS[panelCmapFor(panel)] || fallbackLut;
      let range: { vmin: number; vmax: number };
      if (autoContrast) {
        if (sharedAutoRange && linkContrast) {
          range = sharedAutoRange;
        } else {
          const clipped = percentileClip(panelData, percentileLow, percentileHigh);
          if (clipped.vmax > clipped.vmin) {
            range = clipped;
          } else {
            const fallback = findDataRange(panelData);
            range = { vmin: fallback.min, vmax: fallback.max };
          }
        }
      } else if (!linkContrast && activePanels.length > 1) {
        const stackBounds = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
        const panelDataRange = panelDataRanges[panel];
        const bounds = (perPanelHistogramEnabled && panelDataRange && panelDataRange.max > panelDataRange.min) ? panelDataRange : stackBounds;
        range = resolvePanelRange(panel, bounds);
      } else {
        range = resolveDisplayRange(
          dataMin,
          dataMax,
          traitVmin,
          traitVmax,
          logScale,
          imageVminPct,
          imageVmaxPct,
        );
      }
      renderFramePlayback(panelData, panelImage.data, lut, range.vmin, range.vmax, false);
      panelCtx.putImageData(panelImage, 0, 0);
      const col = slot % cols;
      const row = Math.floor(slot / cols);
      const x = outer + col * (panelOutW + gap);
      const y = outer + row * (panelOutH + gap);
      outCtx.save();
      outCtx.beginPath();
      outCtx.rect(x, y, panelOutW, panelOutH);
      outCtx.clip();
      outCtx.translate(x, y);
      if (flipCols || flipRows) {
        outCtx.translate(flipCols ? panelOutW : 0, flipRows ? panelOutH : 0);
        outCtx.scale(flipCols ? -1 : 1, flipRows ? -1 : 1);
      }
      if (imageRotation % 4 !== 0) {
        outCtx.translate(panelOutW / 2, panelOutH / 2);
        outCtx.rotate((imageRotation * Math.PI) / 2);
        outCtx.translate(-panelOutW / 2, -panelOutH / 2);
      }
      outCtx.drawImage(panelCanvas, 0, 0, panelW, panelH, 0, 0, panelOutW, panelOutH);
      outCtx.restore();
      if (innerBorder > 0) {
        outCtx.save();
        outCtx.strokeStyle = panelInnerBorderColor;
        outCtx.lineWidth = innerBorder;
        const inset = innerBorder / 2;
        outCtx.strokeRect(x + inset, y + inset, Math.max(0, panelOutW - innerBorder), Math.max(0, panelOutH - innerBorder));
        outCtx.restore();
      }
      if (showPanelTitles !== false) {
        const label = panelTitleText(panel);
        if (label) {
          outCtx.save();
          outCtx.font = `700 ${Math.max(MIN_ANIMATION_TITLE_FONT_PX, Math.round((panelTitleFontSize || 11) * scale))}px ${UI_FONT}`;
          outCtx.textAlign = "center";
          outCtx.textBaseline = "top";
          outCtx.shadowColor = "rgba(0,0,0,0.75)";
          outCtx.shadowBlur = 2;
          outCtx.shadowOffsetX = 1;
          outCtx.shadowOffsetY = 1;
          outCtx.fillStyle = "white";
          outCtx.fillText(label, x + panelOutW / 2, y + Math.max(3, Math.round(3 * scale)));
          outCtx.restore();
        }
      }
      if (scaleBarVisible && pixelSize > 0) {
        // Bar, font and margins scale with the export size but never drop below
        // the animation minimums so small GIF frames stay legible.
        drawScaleBarInRegion(outCtx, { x, y, width: panelOutW, height: panelOutH }, 1, Math.max(1e-6, scale), pixelSize, pixelUnit || "px", {
          showZoomIndicator: showZoomIndicator === true,
          maxBarFraction: 0.25,
          barThickness: Math.max(MIN_ANIMATION_SCALE_BAR_THICKNESS_PX, Math.round(5 * scale)),
          margin: Math.max(MIN_ANIMATION_OVERLAY_MARGIN_PX, Math.round(12 * scale)),
          labelGap: Math.max(2, Math.round(4 * scale)),
          font: `${Math.max(MIN_ANIMATION_SCALE_FONT_PX, Math.round(16 * scale))}px ${UI_FONT}`,
          textShadow: { kind: "blur", color: "rgba(0,0,0,0.75)" },
        });
      }
    }
    return { width: outW, height: outH, canvas: out };
  };
  const renderStandaloneGifFrame = (
    frameIdx: number,
    quality: string,
    options: { downsample: number; maxEdgePx: number | null },
  ): { width: number; height: number; indices: Uint8Array } | null => {
    const rendered = renderStandaloneAnimationCanvas(frameIdx, quality, options);
    if (!rendered) return null;
    const ctx = rendered.canvas.getContext("2d");
    if (!ctx) return null;
    const rgba = ctx.getImageData(0, 0, rendered.width, rendered.height).data;
    return { width: rendered.width, height: rendered.height, indices: quantizeRgbaForBrowserGif(rgba) };
  };
  const handleStandaloneGifDownload = async (
    quality = "medium",
    requestOptions: Record<string, unknown> = {},
  ) => {
    setExportMenuAnchor(null);
    if (!canExportStandaloneGif) {
      setLocalExportStatus(standaloneGifUnavailableTitle);
      return;
    }
    const filename = makeExportFilename(title, nSlices, height, width, "gif", quality);
    setExportBusy(true);
    setLocalExportStatus(`Preparing ${filename}...`);
    try {
      const frames: Uint8Array[] = [];
      let outW = 0;
      let outH = 0;
      const frameIndices = exportFrameIndices.length ? exportFrameIndices : [Math.max(0, Math.min(Math.max(1, nSlices || 1) - 1, sliceIdx || 0))];
      const spatialOption = {
        downsample: Number(requestOptions.downsample ?? selectedSpatialOption.downsample),
        maxEdgePx: requestOptions.max_edge_px == null ? null : Number(requestOptions.max_edge_px),
      };
      const total = frameIndices.length;
      for (let slot = 0; slot < total; slot++) {
        const frameIdx = frameIndices[slot];
        const rendered = renderStandaloneGifFrame(frameIdx, quality, spatialOption);
        if (!rendered) throw new Error(`frame ${frameIdx + 1}/${Math.max(1, nSlices || 1)} is not loaded`);
        outW = rendered.width;
        outH = rendered.height;
        frames.push(rendered.indices);
        if (slot === 0 || slot === total - 1 || (slot + 1) % 4 === 0) {
          setLocalExportStatus(`Encoding ${filename}... ${slot + 1}/${total}`);
          await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
        }
      }
      const delayCs = 100 / clampPlaybackFps(Number(requestOptions.fps ?? exportFps ?? playbackFps));
      const gif = encodeIndexedGif(outW, outH, frames, delayCs);
      const blob = new Blob([gif as BlobPart], { type: "image/gif" });
      downloadBlob(blob, filename);
      setLocalExportStatus(`Downloaded ${filename} to browser Downloads (${formatSavedBytes(blob.size)})`);
    } catch (err) {
      setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setExportBusy(false);
    }
  };
  const handleGifExportSelect = (quality: string, requestOptions = animationExportRequest) => {
    if (canExportStandaloneGif) {
      void handleStandaloneGifDownload(quality, requestOptions);
      return;
    }
    void handleExportSelect("gif", quality, 1, requestOptions);
  };
  const applyGifExportPreset = React.useCallback((preset: GifExportPreset) => {
    setExportPanelMode("gif");
    setExportGifPreset(preset);
    setExportFrameStart(1);
    setExportFrameEnd(Math.max(1, nSlices || 1));
    if (preset === "slides") {
      setExportQuality("medium");
      setExportEveryN(1);
      setExportMaxFrames(40);
      setExportFps(DEFAULT_ANIMATION_EXPORT_FPS);
      setExportSpatialPreset("edge512");
    } else if (preset === "compact") {
      setExportQuality("low");
      setExportEveryN(Math.max(1, Math.ceil(Math.max(1, nSlices || 1) / 24)));
      setExportMaxFrames(24);
      setExportFps(8);
      setExportSpatialPreset("down4");
    } else if (preset === "full") {
      setExportQuality("high");
      setExportEveryN(1);
      setExportMaxFrames(0);
      setExportFps(DEFAULT_ANIMATION_EXPORT_FPS);
      setExportSpatialPreset("full");
    }
  }, [nSlices]);
  const markGifPresetCustom = React.useCallback(() => {
    setExportGifPreset((preset) => preset === "custom" ? preset : "custom");
  }, []);
  const exportNumberFieldSx = {
    width: 72,
    "& .MuiInputBase-input": { py: 0.45, px: 0.75, fontSize: 11 },
  } as const;
  const exportPanelButtonSx = { ...compactButton, fontSize: 11, border: `1px solid ${themeColors.border}` } as const;
  const renderGifExportPanel = () => {
    const disabled = !(exportEnabled || canExportStandaloneGif);
    return (
      <Box sx={{ width: 360, maxWidth: "90vw", px: 1.25, py: 1, display: "flex", flexDirection: "column", gap: 0.9 }}>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
          <Button size="small" sx={compactButton} onClick={() => setExportPanelMode("home")}>Back</Button>
          <Box sx={{ display: "flex", alignItems: "center", flex: 1, minWidth: 0 }}>
            <Typography sx={{ ...typography.title, fontSize: 12 }}>GIF</Typography>
            <InfoTooltip maxWidth={360}
              theme={themeInfo.theme}
              icon="?"
              text={(
                <Box sx={{ fontSize: 11, lineHeight: 1.4 }}>
                  <b>GIF export lifecycle</b>
                  <br />1. Pick a preset or adjust frames, fps, quality, and size.
                  <br />2. Press <b>Export GIF</b>; the widget renders panel-only frames using the current contrast and overlays.
                  <br />3. In standalone HTML, the browser downloads the GIF to Downloads. In a live notebook, choose a save location when prompted.
                </Box>
              )}
            />
          </Box>
        </Box>
        <Typography sx={{ fontSize: 11, color: themeColors.textMuted, lineHeight: 1.35 }}>
          GIF exports the current panel movie without toolbar chrome. Use a preset, then export.
        </Typography>
        {standaloneAnimationSourceNote && (
          <Typography sx={{ fontSize: 11, color: themeColors.textMuted, lineHeight: 1.35 }}>
            {standaloneAnimationSourceNote}
          </Typography>
        )}
        {standaloneAnimationQualityWarning && (
          <Box
            data-show3d-encoded-source-animation-warning="true"
            sx={{
              border: `1px solid ${themeColors.accentYellow}`,
              bgcolor: themeInfo.theme === "dark" ? "rgba(255, 193, 7, 0.12)" : "rgba(255, 193, 7, 0.18)",
              color: themeColors.text,
              px: 0.8,
              py: 0.65,
              fontSize: 11,
              lineHeight: 1.35,
            }}
          >
            {standaloneAnimationQualityWarning}
          </Box>
        )}
        <Box sx={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 0.5 }}>
          {GIF_EXPORT_PRESETS.map((preset) => (
            <Button
              key={preset.value}
              size="small"
              sx={{
                ...compactButton,
                minWidth: 0,
                border: `1px solid ${exportGifPreset === preset.value ? themeColors.accent : themeColors.border}`,
                bgcolor: exportGifPreset === preset.value ? "rgba(25,118,210,0.12)" : "transparent",
              }}
              onClick={() => applyGifExportPreset(preset.value)}
            >
              {preset.label}
            </Button>
          ))}
        </Box>
        <Box sx={{ display: "grid", gridTemplateColumns: "auto 1fr auto 1fr", alignItems: "center", gap: 0.75 }}>
          <Typography sx={typography.label}>Frames</Typography>
          <TextField size="small" type="number" value={exportFrameStart} onChange={(event) => { markGifPresetCustom(); setExportFrameStart(Number(event.target.value)); }} inputProps={{ min: 1, max: Math.max(1, nSlices || 1), "aria-label": "Export first frame" }} sx={exportNumberFieldSx} />
          <Typography sx={typography.label}>to</Typography>
          <TextField size="small" type="number" value={exportFrameEnd} onChange={(event) => { markGifPresetCustom(); setExportFrameEnd(Number(event.target.value)); }} inputProps={{ min: 1, max: Math.max(1, nSlices || 1), "aria-label": "Export last frame" }} sx={exportNumberFieldSx} />
          <Typography sx={typography.label}>Every</Typography>
          <TextField size="small" type="number" value={exportEveryN} onChange={(event) => { markGifPresetCustom(); setExportEveryN(Math.max(1, Number(event.target.value))); }} inputProps={{ min: 1, max: Math.max(1, nSlices || 1), "aria-label": "Export every Nth frame" }} sx={exportNumberFieldSx} />
          <Typography sx={typography.label}>Max frames</Typography>
          <TextField size="small" type="number" value={exportMaxFrames} onChange={(event) => { markGifPresetCustom(); setExportMaxFrames(Math.max(0, Number(event.target.value))); }} inputProps={{ min: 0, max: Math.max(1, nSlices || 1), "aria-label": "Maximum exported frames; zero means all" }} sx={exportNumberFieldSx} />
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 0.75 }}>
          <Typography sx={typography.label}>fps</Typography>
          <TextField size="small" type="number" value={exportFps} onChange={(event) => { markGifPresetCustom(); setExportFps(Math.max(1, Number(event.target.value))); }} inputProps={{ min: 1, max: MAX_PLAYBACK_FPS, "aria-label": "Export animation frames per second" }} sx={exportNumberFieldSx} />
          <Typography sx={typography.label}>Quality</Typography>
          <Select size="small" value={exportQuality} onChange={(event) => { markGifPresetCustom(); setExportQuality(event.target.value as AnimationQuality); }} sx={{ ...themedSelect, minWidth: 86, fontSize: 11 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Animation quality" }}>
            {ANIMATION_QUALITY_OPTIONS.map((quality) => <MenuItem key={quality} value={quality}>{quality}</MenuItem>)}
          </Select>
          <Typography sx={typography.label}>Size</Typography>
          <Select size="small" value={exportSpatialPreset} onChange={(event) => { markGifPresetCustom(); setExportSpatialPreset(event.target.value as ExportSpatialPreset); }} sx={{ ...themedSelect, minWidth: 144, fontSize: 11 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Animation spatial size" }}>
            {EXPORT_SPATIAL_OPTIONS.map((option) => <MenuItem key={option.value} value={option.value}>{option.label}</MenuItem>)}
          </Select>
        </Box>
        <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
          {exportFrameSummary} · {animationWorkEstimate}
          {selectedSpatialOption.downsample > 1 ? ` · ${selectedSpatialOption.downsample}x downsample` : ""}
          {selectedSpatialOption.maxEdgePx ? ` · max edge ${selectedSpatialOption.maxEdgePx}px` : ""}
        </Typography>
        <Button
          size="small"
          sx={exportPanelButtonSx}
          disabled={disabled || exportBusy}
          title={disabled ? standaloneGifUnavailableTitle : undefined}
          onClick={() => handleGifExportSelect(exportQuality, animationExportRequest)}
        >
          Export GIF
        </Button>
      </Box>
    );
  };
  const renderHtmlExportPanel = () => (
    <Box sx={{ width: 340, maxWidth: "86vw", px: 1.25, py: 1, display: "flex", flexDirection: "column", gap: 0.8 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Button size="small" sx={compactButton} onClick={() => setExportPanelMode("home")}>Back</Button>
        <Typography sx={{ ...typography.title, fontSize: 12 }}>Interactive HTML</Typography>
      </Box>
      <Typography sx={{ fontSize: 11, color: themeColors.textMuted, lineHeight: 1.35 }}>
        HTML is the primary interactive sharing path. Exact keeps float32 data; encoded uint8 is smaller for visual reports.
      </Typography>
      {exportEnabled && <Button size="small" sx={exportPanelButtonSx} onClick={() => handleExportSelect("exact")}>HTML exact float32 ({exactExportSize})</Button>}
      {exportEnabled && <Button size="small" sx={exportPanelButtonSx} onClick={() => handleExportSelect("quantized")}>HTML encoded uint8 ({quantizedExportSize})</Button>}
      {exportEnabled && height >= 2 && width >= 2 && <Button size="small" sx={exportPanelButtonSx} onClick={() => handleExportSelect("quantized", "medium", 2)}>HTML encoded uint8, 2x downsample ({quantizedExportSize2})</Button>}
      {exportEnabled && height >= 4 && width >= 4 && <Button size="small" sx={exportPanelButtonSx} onClick={() => handleExportSelect("quantized", "medium", 4)}>HTML encoded uint8, 4x downsample ({quantizedExportSize4})</Button>}
      {exportEnabled && height >= 8 && width >= 8 && <Button size="small" sx={exportPanelButtonSx} onClick={() => handleExportSelect("quantized", "medium", 8)}>HTML encoded uint8, 8x downsample ({quantizedExportSize8})</Button>}
      {canDownloadCurrentHtml && standaloneHtmlMode === "quantized" && <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Exact float32 is not embedded in this standalone page; open the live widget for exact export.</Typography>}
      {canDownloadCurrentHtml && <Button size="small" sx={exportPanelButtonSx} onClick={handleStandaloneHtmlDownload}>{standaloneHtmlLabel}</Button>}
      {canDownloadCurrentHtml && standaloneHtmlMode !== "quantized" && <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Encoded uint8 export requires the Python backend to repack the current float32 stack.</Typography>}
    </Box>
  );
  const renderExportMenuContent = () => {
    if (exportPanelMode === "gif") return renderGifExportPanel();
    if (exportPanelMode === "html") return renderHtmlExportPanel();
    return (
      <Box sx={{ width: 320, maxWidth: "86vw", py: 0.5 }}>
        <MenuItem disabled={!(exportEnabled || canExportStandaloneGif)} title={!(exportEnabled || canExportStandaloneGif) ? standaloneGifUnavailableTitle : undefined} onClick={() => applyGifExportPreset("slides")}>
          GIF
        </MenuItem>
        <MenuItem disabled={!(exportEnabled || canDownloadCurrentHtml)} onClick={() => setExportPanelMode("html")}>
          Interactive HTML
        </MenuItem>
        <Box sx={{ px: 2, py: 0.75, borderTop: `1px solid ${themeColors.border}` }}>
          <Typography sx={{ fontSize: 11, color: themeColors.textMuted, lineHeight: 1.35 }}>
            Use GIF for slides and HTML for interactive review.
          </Typography>
        </Box>
      </Box>
    );
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
      const blob = new Blob([payload as BlobPart], { type: exportBlobType(pending.mode) });
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
        setLocalExportStatus(
          pending.handle
            ? `Saved ${filename} to selected location (${formatSavedBytes(bytes.byteLength)})`
            : `Downloaded ${filename} to browser Downloads (${formatSavedBytes(bytes.byteLength)})`,
        );
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

  // Local state
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false);
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const resizeAspectRef = React.useRef<number | null>(null);
  const roiItems = (roiList || []).map((roi, i) => normalizeROI(roi, i));
  const selectedRoi = roiSelectedIdx >= 0 && roiSelectedIdx < roiItems.length ? roiItems[roiSelectedIdx] : null;
  const [showRoiResizeHint, setShowRoiResizeHint] = React.useState(true);
  const [overlayEditMode, setOverlayEditMode] = React.useState(false);
  const [overlaySelection, setOverlaySelection] = React.useState<OverlaySelection | null>(null);
  const [isDraggingOverlay, setIsDraggingOverlay] = React.useState(false);
  const [isHoveringOverlay, setIsHoveringOverlay] = React.useState(false);
  const overlayDragRef = React.useRef<OverlayDragState | null>(null);
  const overlayBaselineRef = React.useRef<PanelOverlaySpec[][] | null>(null);
  const hasPanelOverlays = React.useMemo(() => (panelOverlays || []).some((items) => items && items.length > 0), [panelOverlays]);
  React.useEffect(() => {
    if (!overlayBaselineRef.current && hasPanelOverlays) {
      overlayBaselineRef.current = clonePanelOverlays(panelOverlays);
    }
  }, [hasPanelOverlays, panelOverlays]);
  React.useEffect(() => {
    if (!overlaySelection) return;
    const exists = Boolean(panelOverlays?.[overlaySelection.panel]?.[overlaySelection.overlay]);
    if (!exists) setOverlaySelection(null);
  }, [overlaySelection, panelOverlays]);

  const updatePanelOverlay = React.useCallback((panel: number, overlay: number, nextSpec: PanelOverlaySpec) => {
    const next = clonePanelOverlays(panelOverlays);
    while (next.length <= panel) next.push([]);
    if (!next[panel] || overlay < 0 || overlay >= next[panel].length) return;
    next[panel][overlay] = nextSpec;
    setPanelOverlays(next);
  }, [panelOverlays, setPanelOverlays]);

  const deleteSelectedOverlay = React.useCallback(() => {
    if (!overlaySelection) return;
    const next = clonePanelOverlays(panelOverlays);
    const items = next[overlaySelection.panel];
    if (!items || overlaySelection.overlay < 0 || overlaySelection.overlay >= items.length) return;
    items.splice(overlaySelection.overlay, 1);
    setPanelOverlays(next);
    setOverlaySelection(null);
  }, [overlaySelection, panelOverlays, setPanelOverlays]);

  const resetPanelOverlays = React.useCallback(() => {
    if (!overlayBaselineRef.current) return;
    setPanelOverlays(clonePanelOverlays(overlayBaselineRef.current));
    setOverlaySelection(null);
    overlayDragRef.current = null;
    setIsDraggingOverlay(false);
  }, [setPanelOverlays]);
  const pendingRoiAddRef = React.useRef<{ row: number; col: number } | null>(null);

  // Preview panel state (JS-only, shows ROI crop at full resolution - auto-shows when ROI selected)
  const [previewZoom, setPreviewZoom] = React.useState({ zoom: 1, panX: 0, panY: 0 });
  const previewCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const previewOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const previewContainerRef = React.useRef<HTMLDivElement>(null);
  const [isDraggingPreviewPan, setIsDraggingPreviewPan] = React.useState(false);
  const [previewPanStart, setPreviewPanStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);
  const [previewCropDims, setPreviewCropDims] = React.useState<{ w: number; h: number } | null>(null);
  const previewOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const [previewVersion, setPreviewVersion] = React.useState(0);

  const updateSelectedRoi = (updates: Partial<ROIItem>) => {
    if (roiSelectedIdx < 0 || !roiList) return;
    const newList = [...roiList];
    newList[roiSelectedIdx] = { ...newList[roiSelectedIdx], ...updates };
    setRoiList(newList);
  };
  // Per-panel zoom/pan: index 0 is also used as the shared linked state.
  // Each panel keeps its own state when unlinked.
  type PanelState = {
    zoom: number;
    panX: number;
    panY: number;
    imageVminPct: number;
    imageVmaxPct: number;
  };
  type TouchTransformState = {
    panelIdx: number;
    mode: "pan" | "pinch";
    startX: number;
    startY: number;
    startDistance: number;
    startMidX: number;
    startMidY: number;
    startState: PanelState;
  };
  type FftTouchTransformState = {
    mode: "pan" | "pinch";
    startX: number;
    startY: number;
    startDistance: number;
    startMidX: number;
    startMidY: number;
    startState: { zoom: number; panX: number; panY: number };
  };
  const initialState: PanelState = {
    zoom: 1,
    panX: 0,
    panY: 0,
    imageVminPct: 0,
    imageVmaxPct: 100,
  };
  type RenderRange = { vmin: number; vmax: number };
  type Show3DViewState = {
    linked_state?: Partial<PanelState>;
    panel_states?: Partial<PanelState>[];
  };
  const [viewState, setViewState] = useModelState<Show3DViewState>("view_state");
  const readNumber = (value: unknown, fallback: number): number => (
    typeof value === "number" && Number.isFinite(value) ? value : fallback
  );
  const normalizePanelState = (value: Partial<PanelState> | undefined, fallback: PanelState): PanelState => ({
    zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, readNumber(value?.zoom, fallback.zoom))),
    panX: readNumber(value?.panX, readNumber((value as { pan_x?: unknown } | undefined)?.pan_x, fallback.panX)),
    panY: readNumber(value?.panY, readNumber((value as { pan_y?: unknown } | undefined)?.pan_y, fallback.panY)),
    imageVminPct: readNumber(value?.imageVminPct, readNumber((value as { image_vmin_pct?: unknown } | undefined)?.image_vmin_pct, fallback.imageVminPct)),
    imageVmaxPct: readNumber(value?.imageVmaxPct, readNumber((value as { image_vmax_pct?: unknown } | undefined)?.image_vmax_pct, fallback.imageVmaxPct)),
  });
  const savedPanelStates = Array.isArray(viewState?.panel_states)
    ? viewState.panel_states.map(v => normalizePanelState(v, initialState))
    : [initialState];
  const [linkedState, setLinkedState] = React.useState<PanelState>(() => normalizePanelState(viewState?.linked_state, savedPanelStates[0] || initialState));
  const [panelStates, setPanelStates] = React.useState<PanelState[]>(() => savedPanelStates.length ? savedPanelStates : [initialState]);
  const linkedStateLiveRef = React.useRef<PanelState>(linkedState);
  const panelStatesLiveRef = React.useRef<PanelState[]>(panelStates);
  const transformRenderRafRef = React.useRef<number | null>(null);
  const transformStateCommitTimerRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    const panelCount = Math.max(1, nPanels || 1);
    setPanelStates(prev => {
      if (prev.length === panelCount) return prev;
      const next = Array.from({ length: panelCount }, (_, i) => prev[i] || { ...initialState });
      return next;
    });
  }, [nPanels]);
  // Seamless toggle: on link→unlink, copy linkedState into every panel; on
  // unlink→link, copy panel 0 into linkedState. Single effect so both axes
  // sync atomically.
  const prevLinkRef = React.useRef(linkPanels);
  React.useEffect(() => {
    if (prevLinkRef.current && !linkPanels) {
      // Linked → unlinked: distribute linkedState to all panels
      const linkedView = linkedState;
      setPanelStates(states => {
        const next = states.map(() => ({ ...linkedView }));
        setViewState({ linked_state: { ...linkedView }, panel_states: next.map(state => ({ ...state })) });
        return next;
      });
    } else if (!prevLinkRef.current && linkPanels) {
      // Unlinked → linked: adopt panel 0's state as the shared linked state
      const firstPanelView = panelStates[0] || initialState;
      setLinkedState({ ...firstPanelView });
      setViewState({ linked_state: { ...firstPanelView }, panel_states: panelStates.map(state => ({ ...state })) });
    }
    prevLinkRef.current = linkPanels;
  }, [linkPanels]);
  const stateFor = React.useCallback((panelIdx: number): PanelState => {
    const livePanels = panelStatesLiveRef.current;
    return linkPanels
      ? linkedStateLiveRef.current
      : (livePanels[panelIdx] || panelStates[panelIdx] || initialState);
  }, [linkPanels, panelStates]);
  const syncPlaybackPanelTransform = (panelIdx: number, nextZoom: number, nextPanX: number, nextPanY: number) => {
    const clampAxis = (pan: number, viewport: number, zoomValue: number) => {
      if (viewport <= 0) return 0;
      if (zoomValue <= 1) return viewport * (1 - zoomValue) / 2;
      return Math.max(viewport * (1 - zoomValue), Math.min(0, pan));
    };
    const panelCount = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(panelCount);
    const rows = Math.max(1, Math.ceil(panelCount / cols));
    const gap = panelCount > 1 ? (panelGapPx) : 0;
    const viewportW = (canvasW - gap * (cols - 1)) / cols;
    const viewportH = (canvasH - gap * (rows - 1)) / rows;
    const zoomValue = Math.max(MIN_IMAGE_ZOOM, Math.min(MAX_ZOOM, nextZoom));
    const panXValue = clampAxis(nextPanX, viewportW, zoomValue);
    const panYValue = clampAxis(nextPanY, viewportH, zoomValue);
    const c = playRef.current;
    if (c.linkPanels) {
      const nextLinked = { ...c.linkedState, zoom: zoomValue, panX: panXValue, panY: panYValue };
      c.linkedState = nextLinked;
      linkedStateLiveRef.current = nextLinked;
    } else {
      const next = c.panelStates.slice();
      const prev = next[panelIdx] || initialState;
      next[panelIdx] = { ...prev, zoom: zoomValue, panX: panXValue, panY: panYValue };
      c.panelStates = next;
      panelStatesLiveRef.current = next;
    }
  };
  // Back-compat aliases for the single-panel code paths (ROI, profile, etc.)
  // which still expect plain zoom/panX/panY. Use panel 0's state.
  const zoom = stateFor(0).zoom;
  const panX = stateFor(0).panX;
  const panY = stateFor(0).panY;
  const [isDraggingPan, setIsDraggingPan] = React.useState(false);
  const panDragRef = React.useRef<{ panelIdx: number, x: number, y: number, pX: number, pY: number } | null>(null);
  const [mainCanvasSize, setMainCanvasSize] = React.useState(CANVAS_TARGET_SIZE);
  // Raw scientific pixels for the current frame. Display-only transforms such
  // as denoise/frequency filtering must always start from this source.
  const sourceFrameDataRef = React.useRef<Float32Array | null>(null);
  // The frame the measurements read (stats, ROI stats, line profile, FFT, cursor value): the
  // data after the moving average, alignment, compare and difference the user chose, before the
  // view-only denoise and frequency filters. Those change what the image shows, not the data.
  const measuredFrameRef = React.useRef<Float32Array | null>(null);
  // The frame the image paints (the measured frame through the view-only filters); contrast
  // ranges, the histogram, lens and ROI preview follow what is on screen.
  const displayFrameRef = React.useRef<Float32Array | null>(null);
  const initialCanvasSizeRef = React.useRef<number>(canvasSizeTrait > 0 ? canvasSizeTrait : CANVAS_TARGET_SIZE);
  const defaultPanelCssSizeForCount = React.useCallback((count: number) => {
    const panelCount = Math.max(1, count || 1);
    if (canvasSizeTrait > 0) return canvasSizeTrait;
    if (panelCount <= 1) return CANVAS_TARGET_SIZE;
    const requestedCols = (maxCols && maxCols > 0)
      ? Math.min(maxCols, panelCount, MAX_PANEL_COLUMNS)
      : Math.min(panelCount, MAX_PANEL_COLUMNS);
    // Honor the scientist's column choice by shrinking square panels to fit
    // the current surface. Previously only galleries with 8+ panels did this,
    // so a 3-panel stack could display `Cols 2` while retaining max_cols=3/4,
    // then silently jump to three columns after an unrelated repaint.
    if (rootLayoutWidth > 0) {
      return Math.max(180, Math.min(500, Math.floor(rootLayoutWidth / requestedCols)));
    }
    return 500;
  }, [canvasSizeTrait, maxCols, rootLayoutWidth]);
  const panelColsForCount = React.useCallback((count: number) => {
    const panelCount = Math.max(1, count || 1);
    const requestedCols = (maxCols && maxCols > 0) ? Math.min(maxCols, panelCount, MAX_PANEL_COLUMNS) : Math.min(panelCount, MAX_PANEL_COLUMNS);
    if (panelCount <= 1) return 1;
    const preferredPanelWidth = defaultPanelCssSizeForCount(panelCount);
    const responsiveCols = rootLayoutWidth > 0
      ? Math.max(1, Math.min(panelCount, Math.floor(rootLayoutWidth / Math.max(1, preferredPanelWidth))))
      : requestedCols;
    return Math.max(1, Math.min(requestedCols, responsiveCols));
  }, [defaultPanelCssSizeForCount, maxCols, rootLayoutWidth]);
  const show3dColumnOptions = React.useMemo(() => {
    const visibleCount = Math.max(1, visiblePanelCount || 1);
    const values = new Set<number>([1, 2, 3, 4, 5, 6, 8, 10, 12]);
    return Array.from(values).filter((cols) => cols >= 1 && cols <= visibleCount).sort((a, b) => a - b);
  }, [visiblePanelCount]);
  const clampedMaxCols = panelColsForCount(visiblePanelCount || 1);

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<CursorInfo | null>(null);
  const [cursorReadoutVisible, setCursorReadoutVisible] = React.useState(false);
  const cursorReadoutVisibleRef = React.useRef(false);
  const cursorInfoPendingRef = React.useRef<CursorInfo | null>(null);
  const cursorInfoRafRef = React.useRef<number | null>(null);
  const [showRoiPlot, setShowRoiPlot] = React.useState(true);
  const roiPlotCanvasRef = React.useRef<HTMLCanvasElement>(null);

  // Lens (magnifier inset)
  const [showLens, setShowLens] = React.useState(false);
  const [lensPos, setLensPos] = React.useState<{ row: number; col: number } | null>(null);
  const [lensMag, setLensMag] = React.useState(4);
  const [lensDisplaySize, setLensDisplaySize] = React.useState(128);
  const [lensAnchor, setLensAnchor] = React.useState<{ x: number; y: number } | null>(null);
  const [isDraggingLens, setIsDraggingLens] = React.useState(false);
  const lensCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const lensDragStartRef = React.useRef<{ mx: number; my: number; ax: number; ay: number } | null>(null);
  const [isResizingLens, setIsResizingLens] = React.useState(false);
  const [isHoveringLensEdge, setIsHoveringLensEdge] = React.useState(false);
  const lensResizeStartRef = React.useRef<{ my: number; startSize: number } | null>(null);

  const scheduleCursorInfo = React.useCallback((next: CursorInfo | null) => {
    cursorInfoPendingRef.current = next;
    if (cursorReadoutVisibleRef.current !== Boolean(next)) {
      cursorReadoutVisibleRef.current = Boolean(next);
      setCursorReadoutVisible(Boolean(next));
    }
    if (cursorInfoRafRef.current != null) return;
    if (typeof window === "undefined" || typeof window.requestAnimationFrame !== "function") {
      setCursorInfo(next);
      return;
    }
    cursorInfoRafRef.current = window.requestAnimationFrame(() => {
      cursorInfoRafRef.current = null;
      const pending = cursorInfoPendingRef.current;
      if (!pending) {
        setCursorInfo(null);
        return;
      }
      setCursorInfo((prev) => (
        prev &&
        prev.row === pending.row &&
        prev.col === pending.col &&
        prev.panelIdx === pending.panelIdx &&
        prev.value === pending.value
          ? prev
          : pending
      ));
    });
  }, []);

  // Hover is inspection only. A committed color edit targets the selected
  // panel so moving from the canvas to the dropdown cannot silently retarget
  // the control or retain a stale hover from another panel.
  const colorTargetPanel = nPanels > 1
    ? Math.max(0, selectedVisiblePanels[0] ?? visiblePanelIndices[0] ?? 0)
    : 0;

  React.useEffect(() => () => {
    if (cursorInfoRafRef.current != null && typeof window !== "undefined") {
      window.cancelAnimationFrame(cursorInfoRafRef.current);
    }
  }, []);

  // Reusable rendering buffers (avoid per-frame allocation)
  const mainOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const mainImgDataRef = React.useRef<ImageData | null>(null);
  const mainOffscreenSourcePanelWidthRef = React.useRef<number | undefined>(undefined);
  const scaledPlaybackImgDataRef = React.useRef<{ width: number; height: number; imageData: ImageData } | null>(null);
  const scaledPlaybackMapRef = React.useRef<{
    srcW: number;
    srcH: number;
    outW: number;
    outH: number;
    xMap: Uint32Array;
    yMap: Uint32Array;
  } | null>(null);
  const logBufferRef = React.useRef<Float32Array | null>(null);

  // Seed from the model's slice_idx (not 0): on mount the not-playing branch
  // of the playback effect syncs this ref back onto slice_idx,
  // and a stale 0 would clobber a baked middle-slice start.
  const playbackIdxRef = React.useRef(Number.isFinite(sliceIdx) ? sliceIdx : 0);
  const playbackSliderRef = React.useRef<HTMLSpanElement>(null);
  const playbackLiveCountRef = React.useRef<HTMLElement>(null);
  const localAutoVminsRef = React.useRef<number[]>([]);
  const localAutoVmaxsRef = React.useRef<number[]>([]);
  const autoRangeComputeTokenRef = React.useRef(0);

  const [displaySliceIdx, setDisplaySliceIdx] = React.useState(sliceIdx);
  const [playbackUiSliceIdx, setPlaybackUiSliceIdx] = React.useState(sliceIdx);
  const [localStats, setLocalStats] = React.useState<{ mean: number; min: number; max: number; std: number } | null>(null);
  const [localPanelStats, setLocalPanelStats] = React.useState<PanelStats[] | null>(null);
  const setCompareActiveFromCurrentFrame = React.useCallback((enabled: boolean) => {
    if (!enabled) {
      setCompareMode("off");
      return;
    }
    const frameCount = Math.max(1, Math.round(nSlices || 1));
    const current = ((Math.round(playbackIdxRef.current || displaySliceIdx || sliceIdx || 0) % frameCount) + frameCount) % frameCount;
    const neighbor = current < frameCount - 1 ? current + 1 : Math.max(0, current - 1);
    setComparePair([current, neighbor]);
    setCompareMode("blink");
  }, [displaySliceIdx, nSlices, setCompareMode, setComparePair, sliceIdx]);
  const frameRotationFor = React.useCallback((frame: number) => {
    return ((Math.round(frameRotations?.[frame] ?? 0) % 4) + 4) % 4;
  }, [frameRotations]);
  const rotationActive = ((imageRotation % 4) + 4) % 4 !== 0
    || Boolean(frameRotations?.some(turns => ((turns % 4) + 4) % 4 !== 0));
  const clearRotations = React.useCallback(() => {
    setImageRotation(0);
    setFrameRotations(Array.from({ length: Math.max(1, nSlices || 1) }, () => 0));
    setShowRotationSettings(false);
  }, [nSlices, setFrameRotations, setImageRotation]);
  const setRotationForScope = React.useCallback((quarterTurns: number) => {
    const turns = normalizeRotation(quarterTurns);
    if ((rotationScope || "all") === "frame") {
      const idx = Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(displaySliceIdx || sliceIdx || 0)));
      const next = Array.from({ length: Math.max(1, nSlices || 1) }, (_, frame) => frameRotationFor(frame));
      next[idx] = turns;
      setFrameRotations(next);
      setImageRotation(turns);
      return;
    }
    setImageRotation(turns);
  }, [displaySliceIdx, frameRotationFor, nSlices, normalizeRotation, rotationScope, setFrameRotations, setImageRotation, sliceIdx]);
  React.useEffect(() => {
    if ((rotationScope || "all") !== "frame") return;
    const idx = Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(displaySliceIdx || sliceIdx || 0)));
    const turns = frameRotationFor(idx);
    if (((imageRotation % 4) + 4) % 4 !== turns) setImageRotation(turns);
  }, [displaySliceIdx, frameRotationFor, imageRotation, nSlices, rotationScope, setImageRotation, sliceIdx]);

  // WebGPU FFT state
  const gpuFFTRef = React.useRef<DisplayFFT | null>(null);
  const gpuFftInitPromiseRef = React.useRef<Promise<DisplayFFT> | null>(null);
  const [, setGpuReady] = React.useState(false);  // value unused; setter gates FFT-ready re-renders
  const [fftBackendInfo, setFftBackendInfo] = React.useState<{
    webgpu: "unknown" | "ready" | "unavailable";
    adapter: string;
    source: string;
    ms: number | null;
    panels: number | null;
    grid: string;
  }>({ webgpu: "unknown", adapter: "", source: "", ms: null, panels: null, grid: "" });
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const kymoOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  // WebGPU colormap engine (GPU-accelerated colormap for 4K frames)
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const gpuCmapReadyRef = React.useRef(false);
  const gpuFrameCacheUploadedRef = React.useRef<Set<number>>(new Set());
  const gpuUploadRef = React.useRef<{
    source: Float32Array | null;
    data: Float32Array | null;
    width: number;
    height: number;
    logScale: boolean;
  } | null>(null);
  const gpuRenderSerialRef = React.useRef(0);
  const gpuDisplayVisibleRef = React.useRef<boolean | null>(false);
  const [gpuDisplayVisible, setGpuDisplayVisibleState] = React.useState(false);
  const [gpuResidency, setGpuResidency] = React.useState<{
    // missing: the display stack never arrived; rgb: true color draws on the canvas by design.
    // Neither is a WebGPU fault, so only "fallback" says WebGPU is unavailable.
    stage: "waiting" | "uploading" | "ready" | "fallback" | "missing" | "rgb";
    ready: number;
    error: string;
  }>({ stage: "waiting", ready: 0, error: "" });
  const ensureFftGpu = React.useCallback(async (): Promise<DisplayFFT> => {
    if (gpuFFTRef.current) return gpuFFTRef.current;
    if (!gpuFftInitPromiseRef.current) {
      gpuFftInitPromiseRef.current = getDisplayFFT().then(fft => {
        gpuFFTRef.current = fft;
        setGpuReady(true);
        const webgpu = fft.path === "WebGPU";
        setFftBackendInfo(prev => ({ ...prev, webgpu: webgpu ? "ready" : "unavailable", adapter: webgpu ? getGPUInfo() : "" }));
        return fft;
      });
    }
    return gpuFftInitPromiseRef.current;
  }, []);

  const subpixelAlignSupported =
    !isRgb &&
    Math.max(1, nPanels || 1) === 1 &&
    width > 0 &&
    height > 0 &&
    nSlices > 1 &&
    ((!!offlineFloatStack && offlineFloatStack.byteLength >= nSlices * width * height * 4) ||
      (!!offlineStack && offlineStack.byteLength >= nSlices * width * height));

  const computeSubpixelAlignment = React.useCallback(async () => {
    const serial = ++subpixelAlignSerialRef.current;
    subpixelAlignCacheRef.current.clear();
    if (!subpixelAlignEnabled) {
      subpixelAlignShiftsRef.current = null;
      setSubpixelAlignStatus("Off");
      setSubpixelAlignVersion((value) => value + 1);
      return;
    }
    if (!subpixelAlignSupported) {
      subpixelAlignShiftsRef.current = null;
      setSubpixelAlignStatus("Needs a single-panel client-side stack");
      setSubpixelAlignVersion((value) => value + 1);
      return;
    }
    const frameCount = Math.max(1, nSlices || 1);
    const refIdx = Math.max(0, Math.min(frameCount - 1, Math.round(subpixelAlignReference || 0)));
    const reference = getOfflineFrame(refIdx);
    if (!reference || reference.length < width * height) {
      subpixelAlignShiftsRef.current = null;
      setSubpixelAlignStatus("Reference frame unavailable");
      setSubpixelAlignVersion((value) => value + 1);
      return;
    }
    setSubpixelAlignBusy(true);
    setSubpixelAlignStatus(`Aligning to frame ${refIdx + 1}…`);
    try {
      // Use the shared CPU FFT for this first production path. The WebGPU FFT
      // remains excellent for display FFTs, but registration needs stricter
      // row/column parity: a browser drive caught the GPU path reporting
      // near-zero row shifts on an intentionally drifted stack. Keep alignment
      // correct and visibly trustworthy, then promote a GPU path after parity
      // tests prove the same shifts.
      const gpu: DisplayFFT | null = null;
      const shifts: SubpixelShift[] = [];
      for (let idx = 0; idx < frameCount; idx++) {
        if (serial !== subpixelAlignSerialRef.current) return;
        if (idx === refIdx) {
          shifts.push({ row: 0, col: 0, quality: Infinity });
          continue;
        }
        const frame = getOfflineFrame(idx);
        if (!frame || frame.length < width * height) {
          shifts.push({ row: 0, col: 0, quality: 0 });
          continue;
        }
        shifts.push(await estimateSubpixelShift(reference, frame, width, height, gpu));
      }
      if (serial !== subpixelAlignSerialRef.current) return;
      subpixelAlignShiftsRef.current = shifts;
      subpixelAlignCacheRef.current.clear();
      const maxRow = shifts.reduce((value, shift) => Math.max(value, Math.abs(shift.row)), 0);
      const maxCol = shifts.reduce((value, shift) => Math.max(value, Math.abs(shift.col)), 0);
      const currentIdx = Math.max(0, Math.min(frameCount - 1, Math.round(liveSliceIdx || 0)));
      const currentShift = shifts[currentIdx] ?? { row: 0, col: 0, quality: 0 };
      const backend = gpu ? "WebGPU" : "CPU";
      setSubpixelAlignStatus(
        `Aligned to frame ${refIdx + 1} · current row ${currentShift.row.toFixed(1)} px, col ${currentShift.col.toFixed(1)} px · max ${maxRow.toFixed(1)}/${maxCol.toFixed(1)} px · ${backend}`,
      );
      setSubpixelAlignVersion((value) => value + 1);
    } catch (error) {
      if (serial !== subpixelAlignSerialRef.current) return;
      console.warn("[Show3D] sub-pixel alignment failed", error);
      subpixelAlignShiftsRef.current = null;
      setSubpixelAlignStatus("Alignment failed; showing raw frames");
      setSubpixelAlignVersion((value) => value + 1);
    } finally {
      if (serial === subpixelAlignSerialRef.current) setSubpixelAlignBusy(false);
    }
  }, [
    ensureFftGpu,
    getOfflineFrame,
    height,
    isRgb,
    liveSliceIdx,
    nPanels,
    nSlices,
    offlineFloatStack,
    offlineStack,
    subpixelAlignEnabled,
    subpixelAlignReference,
    subpixelAlignSupported,
    width,
  ]);

  React.useEffect(() => {
    subpixelAlignCacheRef.current.clear();
    subpixelAlignShiftsRef.current = null;
    setSubpixelAlignVersion((value) => value + 1);
    if (!subpixelAlignEnabled) {
      setSubpixelAlignStatus("Off");
      return;
    }
    if (!subpixelAlignSupported) {
      setSubpixelAlignStatus("Needs a single-panel client-side stack");
      return;
    }
    const refIdx = Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(subpixelAlignReference || 0)));
    setSubpixelAlignStatus(`Ready · press Align to use frame ${refIdx + 1}`);
  }, [nSlices, subpixelAlignEnabled, subpixelAlignReference, subpixelAlignSupported]);

  const subpixelAlignFrameForIndex = React.useCallback((idx: number, frame: Float32Array | null): Float32Array | null => {
    if (!frame || !subpixelAlignEnabled) return frame;
    const shifts = subpixelAlignShiftsRef.current;
    if (!shifts || shifts.length === 0) return frame;
    const frameCount = Math.max(1, nSlices || 1);
    const normalized = ((Math.round(idx) % frameCount) + frameCount) % frameCount;
    const shift = shifts[normalized];
    if (!shift) return frame;
    const key = [
      normalized,
      frameSeq,
      subpixelAlignVersion,
      playRef.current.avgWindow,
      playRef.current.diffMode,
      shift.row.toFixed(4),
      shift.col.toFixed(4),
    ].join(":");
    const cached = subpixelAlignCacheRef.current.get(key);
    if (cached) return cached;
    const shifted = shiftFrameBilinear(
      frame,
      width,
      height,
      shift.row,
      shift.col,
      finiteMedianSample(frame),
    );
    subpixelAlignCacheRef.current.set(key, shifted);
    if (subpixelAlignCacheRef.current.size > 48) {
      subpixelAlignCacheRef.current.delete(subpixelAlignCacheRef.current.keys().next().value as string);
    }
    return shifted;
  }, [frameSeq, height, nSlices, subpixelAlignEnabled, subpixelAlignVersion, width]);

  const setGpuDisplayVisible = React.useCallback((visible: boolean) => {
    gpuDisplayVisibleRef.current = visible;
    const gpuCanvas = gpuCanvasRef.current;
    const canvas = canvasRef.current;
    const gpuVisible = ENABLE_GPU_CANVAS_DISPLAY && visible;
    setGpuDisplayVisibleState(gpuVisible);
    if (gpuCanvas) gpuCanvas.style.opacity = gpuVisible ? "1" : "0";
    if (canvas) {
      canvas.style.opacity = gpuVisible ? "0" : "1";
      canvas.style.display = "block";
    }
  }, []);

  const ensureGpuDisplayContext = React.useCallback((
    engine: GPUColormapEngine,
    w: number,
    h: number,
  ): GPUCanvasContext | null => {
    const canvas = gpuCanvasRef.current;
    if (!canvas) return null;
    const widthPx = Math.max(1, Math.round(w));
    const heightPx = Math.max(1, Math.round(h));
    const size = gpuCanvasSizeRef.current;
    if (!gpuCanvasCtxRef.current || !size || size.w !== widthPx || size.h !== heightPx) {
      gpuCanvasCtxRef.current = engine.configureCanvas(canvas, widthPx, heightPx);
      gpuCanvasSizeRef.current = { w: widthPx, h: heightPx };
    }
    return gpuCanvasCtxRef.current;
  }, []);

  const retainedGpuSnapshotSerialRef = React.useRef(0);
  const retainGpuDisplayFrame = React.useCallback((
    durableFrame?: Promise<ImageBitmap | null>,
  ) => {
    const gpuCanvas = gpuCanvasRef.current;
    const canvas = canvasRef.current;
    const engine = gpuCmapRef.current;
    if (!gpuCanvas || !canvas || !engine || playing) return;
    const serial = ++retainedGpuSnapshotSerialRef.current;
    // A WebGPU canvas texture is presentation-only and may be discarded after
    // compositing. Snapshot it immediately after submission, then keep the
    // GPU-produced pixels on the durable 2D canvas while the view is idle.
    const bitmapPromise = durableFrame ?? engine.getDevice().queue.onSubmittedWorkDone().then(() => {
      if (serial !== retainedGpuSnapshotSerialRef.current || playing) return null;
      return createImageBitmap(gpuCanvas);
    });
    void bitmapPromise.then(bitmap => {
      if (!bitmap) return;
      try {
        if (serial !== retainedGpuSnapshotSerialRef.current || playing) return;
        const current = canvasRef.current;
        const ctx = current?.getContext("2d");
        if (!current || !ctx) return;
        ctx.clearRect(0, 0, current.width, current.height);
        ctx.drawImage(bitmap, 0, 0, current.width, current.height);
        setGpuDisplayVisible(false);
      } finally {
        bitmap.close();
      }
    }).catch(error => {
      console.warn("[Show3D] Could not retain the presented WebGPU frame", error);
    });
  }, [playing, setGpuDisplayVisible]);

  React.useEffect(() => {
    localAutoVminsRef.current = [];
    localAutoVmaxsRef.current = [];
    autoRangeComputeTokenRef.current++;
  }, [percentileLow, percentileHigh, nSlices, width, height]);

  const [gpuCmapReady, setGpuCmapReady] = React.useState(false);
  const [gpuCmapChecked, setGpuCmapChecked] = React.useState(false);
  React.useEffect(() => {
    let disposed = false;
    ensureFftGpu().then(fft => {
      if (disposed || !fft) return;
      gpuFFTRef.current = fft;
      setGpuReady(true);
    });
    createGPUColormapEngine().then(engine => {
      if (disposed) {
        engine?.destroy();
        return;
      }
      if (engine) {
        gpuCmapRef.current = engine;
        gpuCmapReadyRef.current = true;
        // State counterpart of the ref so downstream useEffects re-fire
        // when the GPU engine becomes available. Without this, the data
        // effect that fires at mount paints via the CPU fallback BEFORE
        // the engine is ready and never re-paints when it IS ready.
        setGpuCmapReady(true);
      }
      setGpuCmapChecked(true);
    }).catch(() => {
      if (!disposed) setGpuCmapChecked(true);
    });
    return () => {
      disposed = true;
      gpuCmapRef.current?.destroy();
      gpuCmapRef.current = null;
      gpuCmapReadyRef.current = false;
      setGpuCmapReady(false);
      gpuCanvasCtxRef.current = null;
      gpuCanvasSizeRef.current = null;
      gpuFrameCacheUploadedRef.current.clear();
    };
  }, []);

  // Sync displaySliceIdx with model when not playing
  React.useEffect(() => {
    if (!playing) {
      if (gpuResidency.stage !== "ready") setGpuDisplayVisible(false);
      playbackIdxRef.current = sliceIdx;
      setDisplaySliceIdx(sliceIdx);
      setPlaybackUiSliceIdx(sliceIdx);
    }
  }, [sliceIdx, playing, gpuResidency.stage, setGpuDisplayVisible]);

  // Histogram state for main image
  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  // GPU-computed 256-bin histogram. When non-null, the Histogram component
  // uses these bins directly and skips its CPU bin-scan fallback.
  const [imageHistogramBins, setImageHistogramBins] = React.useState<number[] | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [panelHistogramData, setPanelHistogramData] = React.useState<(Float32Array | null)[]>([]);
  const [panelHistogramBins, setPanelHistogramBins] = React.useState<(number[] | null)[]>([]);
  const [panelDataRanges, setPanelDataRanges] = React.useState<{ min: number; max: number }[]>([]);
  const imageHistogramPreviewPctRef = React.useRef<[number, number] | null>(null);
  const panelHistogramPreviewPctRef = React.useRef<Map<number, [number, number]>>(new Map());
  const histogramPreviewPaintRafRef = React.useRef<number | null>(null);
  // Packed panels are independent scientific domains. Always compute their
  // histograms and numerical ranges independently; `linkContrast` links only
  // the relative handle gesture and must never share one absolute range.
  const perPanelHistogramEnabled = (nPanels || 1) > 1 && !sharedPanelSource;

  const setPanelRangePercentages = (
    panel: number,
    minPct: number,
    maxPct: number,
    linked: boolean,
  ) => {
    const panelCount = Math.max(1, nPanels || 1);
    const stack = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    const liveStates = panelStatesLiveRef.current.length === panelCount
      ? panelStatesLiveRef.current
      : panelStates;
    const nextStates = Array.from({ length: panelCount }, (_, index) => {
      const state = liveStates[index] || initialState;
      return linked || index === panel
        ? { ...state, imageVminPct: minPct, imageVmaxPct: maxPct }
        : state;
    });
    const nextMins = Array.from({ length: panelCount }, (_, index) => vminPerPanelLiveRef.current[index] ?? null);
    const nextMaxs = Array.from({ length: panelCount }, (_, index) => vmaxPerPanelLiveRef.current[index] ?? null);
    for (let index = 0; index < panelCount; index++) {
      if (!linked && index !== panel) continue;
      const panelRange = panelDataRanges[index];
      const range = panelRange && panelRange.max > panelRange.min ? panelRange : stack;
      nextMins[index] = pctToValue(minPct, range.min, range.max);
      nextMaxs[index] = pctToValue(maxPct, range.min, range.max);
    }
    panelStatesLiveRef.current = nextStates;
    vminPerPanelLiveRef.current = nextMins;
    vmaxPerPanelLiveRef.current = nextMaxs;
    setPanelStates(nextStates);
    setVminPerPanel(nextMins);
    setVmaxPerPanel(nextMaxs);
  };
  const extractPanelSlice = React.useCallback((
    raw: Float32Array,
    panel: number,
    panelLogScale: boolean,
  ): Float32Array | null => {
    const panelCount = Math.max(1, nPanels || 1);
    if (height <= 0 || raw.length === 0) return null;
    const panelW = totalPanelCount > 1
      ? Math.max(1, panelWidthPx || Math.round(width / totalPanelCount))
      : Math.max(1, width);
    const fullW = raw.length === height * panelW ? panelW : width;
    const srcPanel = sharedPanelSource ? 0 : panel;
    const colStart = Math.min(Math.max(0, srcPanel * panelW), Math.max(0, fullW - panelW));
    if (raw.length < height * fullW || colStart + panelW > fullW || panel >= panelCount) return null;
    const out = new Float32Array(height * panelW);
    for (let row = 0; row < height; row++) {
      out.set(raw.subarray(row * fullW + colStart, row * fullW + colStart + panelW), row * panelW);
    }
    return panelLogScale ? applyLogScale(out) : out;
  }, [height, nPanels, panelWidthPx, sharedPanelSource, totalPanelCount, width]);

  const resolvePanelRange = (
    panel: number,
    range: { min: number; max: number },
    sharedAutoRange?: { vmin: number; vmax: number } | null,
  ): { vmin: number; vmax: number; logScale: boolean } => {
    const state = panelStatesLiveRef.current[panel] || panelStates[panel] || initialState;
    // Per-panel mode: always interpret slider pct in THIS panel's data
    // range. Stack-wide bounds (for mixed BF/DF counts vs SSB radians)
    // would decode SSB sliders to count-territory values → black image.
    const panelDataRange = panelDataRanges[panel];
    const effectiveRange = (panelDataRange && panelDataRange.max > panelDataRange.min)
      ? panelDataRange
      : range;
    const useStoredManual = !linkContrast && !playRef.current.autoContrast;
    if (useStoredManual) {
      const storedMin = vminPerPanelLiveRef.current[panel];
      const storedMax = vmaxPerPanelLiveRef.current[panel];
      if (storedMin != null || storedMax != null) {
        const lo = storedMin ?? effectiveRange.min;
        const hi = storedMax ?? effectiveRange.max;
        return { vmin: lo, vmax: Math.max(lo, hi), logScale };
      }
    }
    if (sharedAutoRange && !perPanelHistogramEnabled) {
      const stack = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
      const lowPct = valueToPct(sharedAutoRange.vmin, stack.min, stack.max, imageVminPct);
      const highPct = valueToPct(sharedAutoRange.vmax, stack.min, stack.max, imageVmaxPct);
      return { ...sliderRange(effectiveRange.min, effectiveRange.max, lowPct, highPct), logScale };
    }
    const lowPct = perPanelHistogramEnabled ? state.imageVminPct : imageVminPct;
    const highPct = perPanelHistogramEnabled ? state.imageVmaxPct : imageVmaxPct;
    const slider = sliderRange(effectiveRange.min, effectiveRange.max, lowPct, highPct);
    return { ...slider, logScale };
  };

  const autoPanelRangeFromData = (
    panelData: Float32Array | null,
    fallbackRange: { min: number; max: number },
    low: number,
    high: number,
  ): { vmin: number; vmax: number; logScale: boolean } | null => {
    if (!panelData || panelData.length === 0) return null;
    const dataRange = findDataRange(panelData);
    const range = dataRange.max > dataRange.min ? dataRange : fallbackRange;
    if (range.max <= range.min) return null;
    let clipped = percentileClip(panelData, low, high);
    const span = range.max - range.min;
    if (!Number.isFinite(clipped.vmin) || !Number.isFinite(clipped.vmax) || clipped.vmax <= clipped.vmin || clipped.vmax - clipped.vmin < span * 1e-4) {
      return { vmin: range.min, vmax: range.max, logScale };
    }
    return { vmin: clipped.vmin, vmax: Math.max(clipped.vmin, clipped.vmax), logScale };
  };

  const panelAutoClipPcts = (
    panel: number,
    state: PanelState,
    stackBounds: { min: number; max: number },
  ): (Pick<PanelState, "imageVminPct" | "imageVmaxPct"> & { vmin: number; vmax: number }) | null => {
    const panelRaw = panelHistogramData[panel];
    if (!panelRaw || panelRaw.length === 0) return null;
    // panelHistogramData is already in the active display domain; in log mode
    // refreshHistogram populated it from extractPanelSlice(..., logScale).
    const panelRange = panelDataRanges[panel];
    const range = (panelRange && panelRange.max > panelRange.min) ? panelRange : stackBounds;
    const span = range.max - range.min;
    if (span <= 0) return null;
    let clipped: { vmin: number; vmax: number } = percentileClip(panelRaw, percentileLow, percentileHigh);
    if (
      !Number.isFinite(clipped.vmin) ||
      !Number.isFinite(clipped.vmax) ||
      clipped.vmax <= clipped.vmin ||
      clipped.vmax - clipped.vmin < span * 1e-4
    ) {
      clipped = { vmin: range.min, vmax: range.max };
    }
    return {
      vmin: clipped.vmin,
      vmax: Math.max(clipped.vmin, clipped.vmax),
      imageVminPct: valueToPct(clipped.vmin, range.min, range.max, state.imageVminPct),
      imageVmaxPct: valueToPct(clipped.vmax, range.min, range.max, state.imageVmaxPct),
    };
  };

  const restorePanelManualClipPcts = () => {
    const stackBounds = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    if (stackBounds.max <= stackBounds.min) return;
    const panelCount = Math.max(1, nPanels || 1);
    const liveStates = panelStatesLiveRef.current.length === panelCount ? panelStatesLiveRef.current : panelStates;
    const nextStates = Array.from({ length: panelCount }, (_, i) => {
      const state = liveStates[i] || initialState;
      const storedMin = vminPerPanelLiveRef.current[i];
      const storedMax = vmaxPerPanelLiveRef.current[i];
      if (storedMin == null && storedMax == null) {
        return { ...state, imageVminPct: 0, imageVmaxPct: 100 };
      }
      const panelRange = panelDataRanges[i];
      const range = (panelRange && panelRange.max > panelRange.min) ? panelRange : stackBounds;
      if (range.max <= range.min) return { ...state, imageVminPct: 0, imageVmaxPct: 100 };
      const lo = storedMin ?? range.min;
      const hi = Math.max(lo, storedMax ?? range.max);
      return {
        ...state,
        imageVminPct: valueToPct(lo, range.min, range.max, state.imageVminPct),
        imageVmaxPct: valueToPct(hi, range.min, range.max, state.imageVmaxPct),
      };
    });
    panelStatesLiveRef.current = nextStates;
    setPanelStates(nextStates);
  };
  const freezeCurrentPanelContrastAsManual = (
    editedPanel: number | null = null,
    editedRangePct: { min: number; max: number } | null = null,
  ) => {
    const stackBounds = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    const panelCount = Math.max(1, nPanels || 1);
    const liveStates = panelStatesLiveRef.current.length === panelCount ? panelStatesLiveRef.current : panelStates;
    const nextStates = Array.from({ length: panelCount }, (_, i) => {
      const state = liveStates[i] || panelStates[i] || initialState;
      return i === editedPanel && editedRangePct
        ? { ...state, imageVminPct: editedRangePct.min, imageVmaxPct: editedRangePct.max }
        : { ...state };
    });
    const nextMins = Array.from({ length: panelCount }, (_, i) => vminPerPanelLiveRef.current[i] ?? null);
    const nextMaxs = Array.from({ length: panelCount }, (_, i) => vmaxPerPanelLiveRef.current[i] ?? null);
    for (let i = 0; i < panelCount; i++) {
      const panelRange = panelDataRanges[i];
      const range = (panelRange && panelRange.max > panelRange.min) ? panelRange : stackBounds;
      if (range.max <= range.min) continue;
      const state = nextStates[i] || initialState;
      nextMins[i] = pctToValue(state.imageVminPct, range.min, range.max);
      nextMaxs[i] = pctToValue(state.imageVmaxPct, range.min, range.max);
    }
    panelStatesLiveRef.current = nextStates;
    vminPerPanelLiveRef.current = nextMins;
    vmaxPerPanelLiveRef.current = nextMaxs;
    setPanelStates(nextStates);
    setVminPerPanel(nextMins);
    setVmaxPerPanel(nextMaxs);
  };
  const freezeCurrentSharedContrastAsManual = () => {
    const bounds = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    const span = bounds.max - bounds.min;
    if (span <= 0) return;
    const renderIdx = clampSlice(displaySliceIdx);
    const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, renderIdx, logScale)
      || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, renderIdx, logScale);
    const range = cached
      ?? (imageHistogramData && imageHistogramData.length > 0
        ? percentileClip(imageHistogramData, percentileLow, percentileHigh)
        : null);
    if (!range || range.vmax <= range.vmin) return;
    setImageVminPct(Math.max(0, Math.min(100, ((range.vmin - bounds.min) / span) * 100)));
    setImageVmaxPct(Math.max(0, Math.min(100, ((range.vmax - bounds.min) / span) * 100)));
  };

  const resolvePanelRenderRange = (
    panel: number,
    range: { min: number; max: number },
    sharedAutoRange: { vmin: number; vmax: number } | null,
    panelData: Float32Array | null,
    autoOn: boolean,
    low: number,
    high: number,
  ): { vmin: number; vmax: number; logScale: boolean } => {
    const initialPercentiles = initialAutoPercentilesRef.current;
    const stablePanelRange = stackAutoRangeApplies()
      && Math.abs(low - initialPercentiles.low) < 1e-6
      && Math.abs(high - initialPercentiles.high) < 1e-6
      ? cachedAutoDisplayRange(
          autoVminsPerPanel,
          autoVmaxsPerPanel,
          panel,
          logScale,
        )
      : null;
    if (autoOn && stablePanelRange) {
      return { ...stablePanelRange, logScale };
    }
    if (perPanelHistogramEnabled && autoOn) {
      const autoRange = autoPanelRangeFromData(panelData, range, low, high);
      if (autoRange) return autoRange;
    }
    return resolvePanelRange(panel, range, sharedAutoRange);
  };

  const handleAutoContrastChange = (on: boolean) => {
    if (on) {
      manualImageRangeBeforeAutoRef.current = { min: imageVminPct, max: imageVmaxPct };
    }
    setAutoContrast(on);
    if (perPanelHistogramEnabled) {
      if (on) {
        // Keep remembered manual per-panel clips. Auto rendering ignores them,
        // and toggling Auto back off should restore the user's manual window.
        // Per-panel snap fires automatically via the [autoContrast,
        // panelHistogramData, ...] useEffect below. Calling the legacy
        // stack-wide snap here would race-write 0/100 to every panel
        // before the effect overrode with the correct per-panel clip,
        // causing a 1-frame flash to washed contrast on every toggle.
      } else {
        // OFF restores manual contrast. If the user never set a manual range,
        // keep the visible Auto windows as the editable manual baseline.
        const hasManualPanelWindow =
          vminPerPanelLiveRef.current.some((value) => value != null) ||
          vmaxPerPanelLiveRef.current.some((value) => value != null);
        if (hasManualPanelWindow) restorePanelManualClipPcts();
        else freezeCurrentPanelContrastAsManual();
        manualImageRangeBeforeAutoRef.current = null;
      }
      return;
    }
    if (on && imageHistogramData) {
      // ON -> snap slider thumbs to actual percentile clip so slider shows what's rendered.
      const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
      const { vmin: pmin, vmax: pmax } = cached ?? percentileClip(imageHistogramData, percentileLow, percentileHigh);
      const { min: autoMin, max: autoMax } = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
      const span = autoMax - autoMin;
      if (span > 0) {
        setImageVminPct(Math.max(0, Math.min(100, ((pmin - autoMin) / span) * 100)));
        setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - autoMin) / span) * 100)));
      }
    } else {
      // OFF -> restore the user's manual window from before Auto was enabled.
      const restore = manualImageRangeBeforeAutoRef.current;
      if (restore) {
        setImageVminPct(restore.min);
        setImageVmaxPct(restore.max);
        manualImageRangeBeforeAutoRef.current = null;
      } else {
        freezeCurrentSharedContrastAsManual();
      }
    }
  };
  const applyContrastPreset = React.useCallback((preset: string) => {
    setContrastPreset(preset);
    if (preset === "custom") return;
    const match = preset.match(/^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$/);
    if (!match) return;
    const lo = Math.max(0, Math.min(99, Number(match[1])));
    const hi = Math.max(lo + 0.01, Math.min(100, Number(match[2])));
    setPercentileHigh(hi);
    setPercentileLow(lo);
    handleAutoContrastChange(true);
  }, [handleAutoContrastChange, setContrastPreset, setPercentileHigh, setPercentileLow]);

  // Histogram state for FFT
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftStats, setFftStats] = React.useState<{ mean: number; min: number; max: number; std: number }>({ mean: 0, min: 0, max: 0, std: 0 });
  const [fftQuality, setFftQuality] = React.useState<FftQualityMetrics | null>(null);
  const fftQualityKeyRef = React.useRef("");
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftLogScale, setFftLogScale] = React.useState(false);
  const [fftAuto, setFftAuto] = React.useState(true);  // Auto: mask DC + 99.9% clipping
  const [fftShowColorbar, setFftShowColorbar] = React.useState(false);
  const [fftOffscreenVersion, setFftOffscreenVersion] = React.useState(0);
  const [showColorbar, setShowColorbar] = React.useState(false);
  // True-color RGB figures: no colormap / intensity tools; pixelated (no smooth).
  React.useEffect(() => {
    if (isRgb) {
      setShowColorbar(false);
      if (autoContrast) setAutoContrast(false);
      if (logScale) setLogScale(false);
      if (diffMode && diffMode !== "off") setDiffMode("off");
      if (!displayFilterOff) setDisplayFilter("none");
      if (Number(spatialBin || 1) !== 1) setSpatialBin(1);
      if (showDenoise) setShowDenoise(false);
      if (smooth) setSmooth(false);
    }
  }, [isRgb]); // eslint-disable-line react-hooks/exhaustive-deps -- one-shot mode switch

  // Histogram state for kymograph (mirrors FFT contrast/colormap controls)
  const [kymoVminPct, setKymoVminPct] = React.useState(0);
  const [kymoVmaxPct, setKymoVmaxPct] = React.useState(100);
  const [kymoHistogramData, setKymoHistogramData] = React.useState<Float32Array | null>(null);
  const [kymoDataRange, setKymoDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [kymoStats, setKymoStats] = React.useState<{ mean: number; min: number; max: number; std: number }>({ mean: 0, min: 0, max: 0, std: 0 });
  const [kymoColormap, setKymoColormap] = React.useState("inferno");
  const [kymoLogScale, setKymoLogScale] = React.useState(false);
  const [kymoAuto, setKymoAuto] = React.useState(true);  // Auto: percentile-clip like the main image
  const [kymoShowColorbar, setKymoShowColorbar] = React.useState(false);

  const handleRootMouseDownCapture = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement | null;
    if (target && target.closest(WIDGET_TEXT_OR_VALUE_CONTROL_SELECTOR)) return;
    rootRef.current?.focus({ preventScroll: true });
  };


  // FFT d-spacing measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);

  // ROI FFT state: when ROI + FFT are both active, compute FFT of cropped ROI region
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);
  const fftCropDimsRef = React.useRef<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);
  const fftPanelGridRef = React.useRef<{ panelWidth: number; panelHeight: number; cols: number; rows: number; count: number } | null>(null);

  // FFT zoom/pan state
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const defaultFftViewState = React.useMemo(() => ({ zoom: 1, panX: 0, panY: 0 }), []);
  const [panelFftStates, setPanelFftStates] = React.useState<Map<number, { zoom: number; panX: number; panY: number }>>(new Map());
  const internalFftZoomSyncRef = React.useRef(false);
  const fftViewLiveRef = React.useRef({ zoom: 1, panX: 0, panY: 0 });
  const fftViewRafRef = React.useRef<number | null>(null);
  const fftViewReactSyncTimerRef = React.useRef<number | null>(null);
  const fftViewTraitSyncTimerRef = React.useRef<number | null>(null);
  const fftViewDirectRedrawRef = React.useRef<((view: { zoom: number; panX: number; panY: number }) => void) | null>(null);
  // Until the user zooms or pans, the FFT view is centred on its viewport: the view effect below
  // owns that centred pan, so every reader of the view sees the same numbers the draw uses.
  const fftUserAdjustedViewRef = React.useRef(false);

  const commitFftViewReactState = React.useCallback(() => {
    const live = fftViewLiveRef.current;
    setFftZoom(prev => Math.abs(prev - live.zoom) > 0.001 ? live.zoom : prev);
    setFftPanX(prev => Math.abs(prev - live.panX) > 0.5 ? live.panX : prev);
    setFftPanY(prev => Math.abs(prev - live.panY) > 0.5 ? live.panY : prev);
  }, []);

  const scheduleFftViewState = React.useCallback((next: { zoom: number; panX: number; panY: number }, syncTrait = false, directOnly = false) => {
    fftViewLiveRef.current = next;
    if (directOnly) {
      fftViewDirectRedrawRef.current?.(next);
      if (fftViewReactSyncTimerRef.current !== null) {
        window.clearTimeout(fftViewReactSyncTimerRef.current);
      }
      fftViewReactSyncTimerRef.current = window.setTimeout(() => {
        fftViewReactSyncTimerRef.current = null;
        commitFftViewReactState();
      }, 80);
    } else if (fftViewRafRef.current === null) {
      fftViewRafRef.current = window.requestAnimationFrame(() => {
        fftViewRafRef.current = null;
        commitFftViewReactState();
      });
    }
    if (syncTrait) {
      if (fftViewTraitSyncTimerRef.current !== null) {
        window.clearTimeout(fftViewTraitSyncTimerRef.current);
      }
      fftViewTraitSyncTimerRef.current = window.setTimeout(() => {
        fftViewTraitSyncTimerRef.current = null;
        internalFftZoomSyncRef.current = true;
        setFftOverlayZoomTrait(Number(fftViewLiveRef.current.zoom.toFixed(3)));
      }, 160);
    }
  }, [commitFftViewReactState, setFftOverlayZoomTrait]);

  const getFftViewForPanel = React.useCallback((panelIdx: number) => {
    return linkPanels
      ? fftViewLiveRef.current
      : (panelFftStates.get(panelIdx) || defaultFftViewState);
  }, [defaultFftViewState, linkPanels, panelFftStates]);

  const setFftViewForPanel = React.useCallback((panelIdx: number, next: { zoom: number; panX: number; panY: number }, syncTrait = false, directOnly = false) => {
    if (linkPanels) {
      scheduleFftViewState(next, syncTrait, directOnly);
      return;
    }
    setPanelFftStates(prev => {
      const map = new Map(prev);
      map.set(panelIdx, next);
      return map;
    });
  }, [linkPanels, scheduleFftViewState]);

  React.useEffect(() => {
    fftViewLiveRef.current = { zoom: fftZoom, panX: fftPanX, panY: fftPanY };
  }, [fftZoom, fftPanX, fftPanY]);

  React.useEffect(() => () => {
    if (fftViewRafRef.current !== null) {
      window.cancelAnimationFrame(fftViewRafRef.current);
      fftViewRafRef.current = null;
    }
    if (fftViewReactSyncTimerRef.current !== null) {
      window.clearTimeout(fftViewReactSyncTimerRef.current);
      fftViewReactSyncTimerRef.current = null;
    }
    if (fftViewTraitSyncTimerRef.current !== null) {
      window.clearTimeout(fftViewTraitSyncTimerRef.current);
      fftViewTraitSyncTimerRef.current = null;
    }
  }, []);

  React.useEffect(() => {
    if (internalFftZoomSyncRef.current) {
      internalFftZoomSyncRef.current = false;
      return;
    }
    const reset = { zoom: resolvedFftOverlayZoom, panX: 0, panY: 0 };
    fftViewLiveRef.current = reset;
    fftUserAdjustedViewRef.current = false;
    setFftZoom(reset.zoom);
    setFftPanX(reset.panX);
    setFftPanY(reset.panY);
  }, [resolvedFftOverlayZoom]);

  const previousFftLinkPanelsRef = React.useRef(linkPanels);
  React.useEffect(() => {
    const previous = previousFftLinkPanelsRef.current;
    if (previous && !linkPanels) {
      const shared = { zoom: fftZoom, panX: fftPanX, panY: fftPanY };
      setPanelFftStates(() => new Map(Array.from({ length: totalPanelCount }, (_, idx) => [idx, { ...shared }])));
    } else if (!previous && linkPanels) {
      const panel = visiblePanelIndices[0] ?? 0;
      const current = panelFftStates.get(panel) || defaultFftViewState;
      fftViewLiveRef.current = current;
      fftUserAdjustedViewRef.current = true;
      setFftZoom(current.zoom);
      setFftPanX(current.panX);
      setFftPanY(current.panY);
    }
    previousFftLinkPanelsRef.current = linkPanels;
  }, [defaultFftViewState, fftPanX, fftPanY, fftZoom, linkPanels, panelFftStates, totalPanelCount, visiblePanelIndices]);

  // The overlay inset and the docked FFT canvas are different viewports: a pan chosen in one
  // means nothing in the other, so moving between them re-centres the view.
  React.useEffect(() => {
    fftUserAdjustedViewRef.current = false;
  }, [fftLayoutOverlay]);
  const fftContainerRef = React.useRef<HTMLDivElement>(null);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileLine, setProfileLine] = useModelState<{row: number; col: number}[]>("profile_line");
  const [profileWidth, setProfileWidth] = useModelState<number>("profile_width");
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const [profilePanelIdx, setProfilePanelIdx] = React.useState(0);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profilePoints = profileLine || [];
  const singlePanelPageProfile = isPaged && Math.max(1, panelsPerPage || 0) === 1;
  React.useEffect(() => {
    if (!singlePanelPageProfile) return;
    setProfilePanelIdx((current) => current === activePageStart ? current : activePageStart);
  }, [activePageStart, singlePanelPageProfile]);
  // Kymograph (space-time) panel: static (nFrames, lineLen) image built by
  // sampling the profile line on every frame from the offline stack. Recompute
  // is cold-path (on line / width change only), not per render tick.
  const [showKymograph, setShowKymograph] = useModelState<boolean>("show_kymograph");
  const kymoCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const kymoOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const kymoDataRef = React.useRef<{ data: Float32Array; lineLen: number; nFrames: number } | null>(null);
  const [kymoVersion, setKymoVersion] = React.useState(0);
  // Kymograph zoom/pan state (mirrors FFT)
  const [kymoZoom, setKymoZoom] = React.useState(1);
  const [kymoPanX, setKymoPanX] = React.useState(0);
  const [kymoPanY, setKymoPanY] = React.useState(0);
  const kymoContainerRef = React.useRef<HTMLDivElement>(null);
  const kymoWheelHandlerRef = React.useRef<((event: WheelEvent) => void) | null>(null);
  // Click readout: cursor maps to (frame index, distance index) and looks up
  // intensity in the static kymograph image. Mirrors FFT d-spacing readout.
  const [kymoClickInfo, setKymoClickInfo] = React.useState<{
    timeVal: number; timeUnit: string; distVal: number; distUnit: string; intensity: number;
    col: number; row: number;
  } | null>(null);
  const kymoClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const [profileResizeStart, setProfileResizeStart] = React.useState<{ y: number; height: number } | null>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);

  // Sync sizes from Python and set initial minimum. In multi-panel mode the user
  // is comparing N images side-by-side; default per-panel sizing keeps each image
  // readable instead of crushed when the widget concatenates them into one wide
  // canvas (e.g. 4 panels at 500 px total → 125 px per panel = too small).
  React.useEffect(() => {
    // size is PER PANEL. For multi-panel, total canvas width = size * cols.
    // NEVER BIN rule: data is never averaged. CSS canvas scales the painted
    // image for display, source pixels stay intact. 500 px/panel default
    // gives 4 cols → 2000 px wide which fits a typical monitor; operator
    // drags the resize handle larger when they want pixel-1:1.
    const visibleCount = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(visibleCount);
    const perPanel = defaultPanelCssSizeForCount(visibleCount);
    const target = perPanel * cols;
    setMainCanvasSize(target);
    if (initialCanvasSizeRef.current === CANVAS_TARGET_SIZE) {
      initialCanvasSizeRef.current = target;
    }
  }, [defaultPanelCssSizeForCount, visiblePanelCount, panelColsForCount]);

  // Calculate display scale. In multi-panel mode `width` may be either the
  // concatenated source width or one shared source frame drawn into N slots.
  // `panel_width_px` keeps the per-panel source geometry explicit.
  const _nPanelsLocal = Math.max(1, visiblePanelCount || 1);
  const _colsLocal = panelColsForCount(_nPanelsLocal);
  const _rowsLocal = Math.ceil(_nPanelsLocal / _colsLocal);
  const fftAllowed = true;
  const effectiveShowFft = showFft && fftAllowed;
  const sourcePanelWidth = totalPanelCount > 1
    ? Math.max(1, panelWidthPx || Math.round(width / totalPanelCount))
    : Math.max(1, width);
  const sourcePanelHeight = Math.max(1, height);
  const isMultiPanelSource = totalPanelCount > 1;
  const requestedDisplayScale = isMultiPanelSource
    ? mainCanvasSize / Math.max(1, sourcePanelWidth * _colsLocal)
    : mainCanvasSize / Math.max(width, height);
  // For 90°/270° rotations, swap canvas dims so non-square images fit without clipping.
  const rotSwap = (imageRotation % 2) !== 0;
  const requestedCanvasW = isMultiPanelSource
    ? Math.round(sourcePanelWidth * requestedDisplayScale * _colsLocal)
    : Math.round((rotSwap ? height : width) * requestedDisplayScale);
  // Grid layout: when max_cols wraps panels into multiple rows, canvasH grows to fit `rows` rows.
  const _requestedCanvasHSingleRow = Math.round((rotSwap ? width : height) * requestedDisplayScale);
  const _gapForLayout = _nPanelsLocal > 1 ? (panelGapPx) : 0;
  const _requestedSlotWForLayout = (requestedCanvasW - _gapForLayout * (_colsLocal - 1)) / _colsLocal;
  const _requestedSlotHForLayout = _requestedSlotWForLayout * (sourcePanelHeight / sourcePanelWidth);
  const requestedCanvasH = isMultiPanelSource
    ? Math.round(_requestedSlotHForLayout * _rowsLocal + _gapForLayout * (_rowsLocal - 1))
    : _requestedCanvasHSingleRow;
  const gridCanvasCap = isMultiPanelSource
    ? Math.min(
        1,
        MAX_INTERACTIVE_GRID_CANVAS_EDGE / Math.max(1, requestedCanvasW, requestedCanvasH),
        Math.sqrt(MAX_INTERACTIVE_GRID_CANVAS_PIXELS / Math.max(1, requestedCanvasW * requestedCanvasH)),
      )
    : 1;
  const displayScale = requestedDisplayScale * gridCanvasCap;
  const canvasW = isMultiPanelSource
    ? Math.round(sourcePanelWidth * displayScale * _colsLocal)
    : Math.round((rotSwap ? height : width) * displayScale);
  const _canvasHSingleRow = Math.round((rotSwap ? width : height) * displayScale);
  const _slotWForLayout = (canvasW - _gapForLayout * (_colsLocal - 1)) / _colsLocal;
  const _slotHForLayout = _slotWForLayout * (sourcePanelHeight / sourcePanelWidth);
  const canvasH = isMultiPanelSource
    ? Math.round(_slotHForLayout * _rowsLocal + _gapForLayout * (_rowsLocal - 1))
    : _canvasHSingleRow;
  const mainPanelFrameWidth = canvasW + 2 * galleryOuterBorderPx;
  const mainPanelWidth = `min(100%, ${mainPanelFrameWidth}px)`;
  const mainPanelAspectRatio = `${Math.max(canvasW, 1)} / ${Math.max(canvasH, 1)}`;
  const effectiveLoopEnd = loopEnd < 0 ? nSlices - 1 : loopEnd;
  // ROI hidden while the kymograph is shown - both are line/region analysis on
  // the same side slot, and showing them together confuses which panel is which.
  const roiAllowed = totalPanelCount === 1 && !showKymograph;
  const effectiveRoiActive = roiAllowed && roiActive;

  type PanelGeometry = {
    panelIdx: number;
    slotX: number;
    slotY: number;
    slotW: number;
    slotH: number;
    scaleX: number;
    scaleY: number;
    state: PanelState;
  };
  const getPanelLayout = () => {
    const n = _nPanelsLocal;
    const cols = _colsLocal;
    const rows = _rowsLocal;
    const gap = n > 1 ? (panelGapPx) : 0;
    const slotW = (canvasW - gap * (cols - 1)) / cols;
    const slotH = (canvasH - gap * (rows - 1)) / rows;
    return { n, cols, rows, gap, slotW, slotH };
  };
  const getPanelGeometry = (panelIdx: number): PanelGeometry | null => {
    const { n, cols, rows, gap, slotW, slotH } = getPanelLayout();
    if (panelIdx < 0 || panelIdx >= totalPanelCount) return null;
    const slotIdx = visiblePanelIndices.indexOf(panelIdx);
    if (slotIdx < 0 || slotIdx >= n) return null;
    const col = slotIdx % cols;
    const row = Math.floor(slotIdx / cols);
    if (row >= rows) return null;
    return {
      panelIdx,
      slotX: col * (slotW + gap),
      slotY: row * (slotH + gap),
      slotW,
      slotH,
      scaleX: slotW / Math.max(1, sourcePanelWidth),
      scaleY: slotH / Math.max(1, sourcePanelHeight),
      state: stateFor(panelIdx),
    };
  };
  const getFftSlot = React.useCallback((slot: number, count: number, cols: number, rows: number) => {
    const gap = count > 1 ? (panelGapPx) : 0;
    const slotW = (canvasW - gap * (cols - 1)) / cols;
    const slotH = (canvasH - gap * (rows - 1)) / rows;
    const col = slot % cols;
    const row = Math.floor(slot / cols);
    return {
      x: col * (slotW + gap),
      y: row * (slotH + gap),
      w: slotW,
      h: slotH,
    };
  }, [canvasW, canvasH, panelGapPx]);
  const clearWithGridBackground = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = interPanelGapColor;
    ctx.fillRect(0, 0, w, h);
  };
  const strokePanelInnerBorder = (ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number) => {
    if (panelInnerBorderPx <= 0) return;
    ctx.save();
    ctx.strokeStyle = panelInnerBorderColor;
    ctx.lineWidth = panelInnerBorderPx;
    const inset = panelInnerBorderPx / 2;
    ctx.strokeRect(x + inset, y + inset, Math.max(0, w - panelInnerBorderPx), Math.max(0, h - panelInnerBorderPx));
    ctx.restore();
  };
  const drawFftOffscreen = React.useCallback((ctx: CanvasRenderingContext2D, offscreen: HTMLCanvasElement) => {
    ctx.clearRect(0, 0, canvasW, canvasH);
    const grid = fftPanelGridRef.current;
    if (grid) {
      ctx.fillStyle = interPanelGapColor;
      ctx.fillRect(0, 0, canvasW, canvasH);
      for (let slot = 0; slot < grid.count; slot++) {
        const srcCol = slot % grid.cols;
        const srcRow = Math.floor(slot / grid.cols);
        const srcX = srcCol * grid.panelWidth;
        const srcY = srcRow * grid.panelHeight;
        const dst = getFftSlot(slot, grid.count, grid.cols, grid.rows);
        const panel = visiblePanelIndices[slot] ?? slot;
        const view = linkPanels ? { zoom: fftZoom, panX: fftPanX, panY: fftPanY } : (panelFftStates.get(panel) || defaultFftViewState);
        ctx.imageSmoothingEnabled = smooth && (grid.panelWidth < dst.w || grid.panelHeight < dst.h);
        ctx.save();
        ctx.beginPath();
        ctx.rect(dst.x, dst.y, dst.w, dst.h);
        ctx.clip();
        ctx.translate(dst.x + view.panX, dst.y + view.panY);
        ctx.scale(view.zoom, view.zoom);
        ctx.drawImage(
          offscreen,
          srcX,
          srcY,
          grid.panelWidth,
          grid.panelHeight,
          0,
          0,
          dst.w,
          dst.h,
        );
        ctx.restore();
      }
    } else {
      ctx.save();
      ctx.translate(fftPanX, fftPanY);
      ctx.scale(fftZoom, fftZoom);
      ctx.imageSmoothingEnabled = smooth && (offscreen.width < canvasW || offscreen.height < canvasH);
      ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);
      ctx.restore();
    }
  }, [canvasW, canvasH, defaultFftViewState, fftPanX, fftPanY, fftZoom, getFftSlot, interPanelGapColor, linkPanels, panelFftStates, smooth, visiblePanelIndices]);
  const panelGlobalColOffset = (panelIdx: number) => (totalPanelCount > 1 && !sharedPanelSource) ? panelIdx * sourcePanelWidth : 0;
  const panelLocalCol = (globalCol: number, panelIdx: number) => globalCol - panelGlobalColOffset(panelIdx);
  const panelGlobalCol = (localCol: number, panelIdx: number) => localCol + panelGlobalColOffset(panelIdx);
  // A single-panel page reuses one spatial profile on every page. Keep the
  // trait coordinates page-local, then add the active page's packed-frame
  // offset only when sampling the concatenated source frame.
  const profileSampleColOffset = singlePanelPageProfile
    ? panelGlobalColOffset(activePageStart)
    : 0;
  const profileComputeGenerationRef = React.useRef(0);
  const sampleProfileForActivePage = React.useCallback((
    data: Float32Array,
    p0: { row: number; col: number },
    p1: { row: number; col: number },
    widthPx: number = profileWidth,
  ) => sampleLineProfileBrowser(
    data,
    width,
    height,
    p0.row,
    p0.col + profileSampleColOffset,
    p1.row,
    p1.col + profileSampleColOffset,
    widthPx,
  ), [height, profileSampleColOffset, profileWidth, width]);
  const updateProfileForActivePage = React.useCallback((
    data: Float32Array,
    p0: { row: number; col: number },
    p1: { row: number; col: number },
    widthPx: number = profileWidth,
  ) => {
    const generation = ++profileComputeGenerationRef.current;
    void sampleProfileForActivePage(data, p0, p1, widthPx).then((profile) => {
      if (generation === profileComputeGenerationRef.current) setProfileData(profile);
    }).catch((error) => {
      if (generation === profileComputeGenerationRef.current) {
        console.error("[Show3D] WebGPU line profile failed", error);
      }
    });
  }, [profileWidth, sampleProfileForActivePage]);
  const getImageHitRadius = (panelIdx: number) => {
    const geom = getPanelGeometry(panelIdx);
    if (!geom) return RESIZE_HIT_AREA_PX / Math.max(1e-6, displayScale * zoom);
    const scale = Math.max(1e-6, Math.min(geom.scaleX, geom.scaleY) * geom.state.zoom);
    return RESIZE_HIT_AREA_PX / scale;
  };
  const canvasPointFromEvent = (e: React.MouseEvent): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  // ROI FFT active: both ROI and FFT on, with a selected ROI
  const roiFftActive = effectiveShowFft && effectiveRoiActive && roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0);

  // Preview panel visible: auto-shows when ROI active with a selected ROI
  const previewVisible = effectiveRoiActive && roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0);
  const selectedRoiKey = (() => {
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return "";
    const roi = roiList[roiSelectedIdx];
    return `${roi.row},${roi.col},${roi.radius},${roi.radius_inner},${roi.width},${roi.height},${roi.shape}`;
  })();

  // Compute stats for ALL ROIs (memoized, recomputes on frame/ROI geometry change)
  const allRoiStats = React.useMemo(() => {
    const raw = measuredFrameRef.current;
    if (!effectiveRoiActive || !roiItems.length || !raw || !width || !height) return [];
    return roiItems.map(roi => computeROIPixelStats(raw, width, height, roi));
    // frameBytes triggers recompute on frame change; displaySliceIdx triggers recompute during playback
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveRoiActive, roiItems, width, height, frameBytes, displaySliceIdx]);

  // Initialize reusable offscreen canvas + ImageData (resized when dimensions change)
  React.useEffect(() => {
    if (width <= 0 || height <= 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    logBufferRef.current = new Float32Array(width * height);
    if (mainOffscreenRef.current && mainOffscreenSourcePanelWidthRef.current !== undefined && !displayFrameRef.current) {
      scaledPlaybackImgDataRef.current = null;
      scaledPlaybackMapRef.current = null;
      return;
    }
    mainOffscreenRef.current = canvas;
    mainOffscreenSourcePanelWidthRef.current = undefined;
    mainImgDataRef.current = canvas.getContext("2d")!.createImageData(width, height);
    scaledPlaybackImgDataRef.current = null;
    scaledPlaybackMapRef.current = null;
  }, [width, height]);

  // Prevent page scroll on secondary canvas containers. Main image wheel is
  // handled by a non-passive listener below so zoom works in notebook outputs.
  // Re-attach whenever a container can remount: the FFT panel leaves the tree
  // in the overlay layout, and the whole live view waits for a watched folder.
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const el2 = fftContainerRef.current;
    const el3 = previewContainerRef.current;
    el2?.addEventListener("wheel", preventDefault, { passive: false });
    el3?.addEventListener("wheel", preventDefault, { passive: false });
    return () => {
      el2?.removeEventListener("wheel", preventDefault);
      el3?.removeEventListener("wheel", preventDefault);
    };
  }, [effectiveShowFft, fftLayoutOverlay, previewVisible, folderWaiting, canRenderLive, hasSavedStaticFallback]);


  React.useEffect(() => {
    bounceDirRef.current = reverse ? -1 : 1;
  }, [reverse]);

  // All playback params as a single ref (avoids stale closures in rAF loop)
  const pathIdxRef = React.useRef(0);
  const playRef = React.useRef({
    fps: playbackFps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH, panelCols: _colsLocal,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    autoVmins, autoVmaxs,
    linkContrast,
    linkedState, linkPanels,
    panelStates, vminPerPanel, vmaxPerPanel,
    visiblePanelIndices,
    zoom, panX, panY, playbackPath,
    profileActive, profilePoints, profileWidth, profileColOffset: profileSampleColOffset,
    traitVmin, traitVmax, smooth, imageRotation, showStats,
    diffMode, avgWindow,
  });
  React.useEffect(() => {
    linkedStateLiveRef.current = linkedState;
  }, [linkedState]);
  React.useEffect(() => {
    panelStatesLiveRef.current = panelStates;
  }, [panelStates]);
  React.useEffect(() => {
    const liveLinkedState = linkedStateLiveRef.current;
    const livePanelStates = panelStatesLiveRef.current.length === Math.max(1, nPanels || 1)
      ? panelStatesLiveRef.current
      : panelStates;
    playRef.current = {
      fps: playbackFps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
      nSlices, width, height, displayScale, canvasW, canvasH, panelCols: _colsLocal,
      logScale, autoContrast, percentileLow, percentileHigh,
      dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
      autoVmins, autoVmaxs,
      linkContrast,
      linkedState: liveLinkedState, linkPanels,
      panelStates: livePanelStates, vminPerPanel, vmaxPerPanel,
      visiblePanelIndices,
      zoom, panX, panY, playbackPath,
      profileActive, profilePoints, profileWidth, profileColOffset: profileSampleColOffset,
      traitVmin, traitVmax, smooth, imageRotation, showStats,
      diffMode, avgWindow,
    };
  }, [playbackFps, reverse, boomerang, loop, loopStart, effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH, _colsLocal,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    autoVmins, autoVmaxs, linkContrast, linkedState, linkPanels, panelStates, vminPerPanel, vmaxPerPanel, visiblePanelIndices,
    zoom, panX, panY, playbackPath,
    profileActive, profilePoints, profileWidth, profileSampleColOffset,
    traitVmin, traitVmax, smooth, imageRotation, showStats, diffMode, avgWindow]);

  const changeLogScale = React.useCallback((nextLogScale: boolean) => {
    if (nextLogScale === logScale) return;

    // Manual per-panel limits are absolute values in the histogram's current
    // scale domain. Convert them together with the pixels. Reusing linear
    // limits after switching to log (for example 10,000 against log pixels
    // around 9) maps the entire canvas to the first LUT color and looks like an
    // empty blue/purple frame until the histogram is moved again.
    const convert = nextLogScale ? signedLog1p : signedExpm1;
    const convertValues = (values: (number | null)[]) => values.map((value) => (
      value == null || !Number.isFinite(value) ? value : convert(value)
    ));
    const nextMins = convertValues(vminPerPanelLiveRef.current);
    const nextMaxs = convertValues(vmaxPerPanelLiveRef.current);
    vminPerPanelLiveRef.current = nextMins;
    vmaxPerPanelLiveRef.current = nextMaxs;
    playRef.current.logScale = nextLogScale;
    playRef.current.vminPerPanel = nextMins;
    playRef.current.vmaxPerPanel = nextMaxs;
    setVminPerPanel(nextMins);
    setVmaxPerPanel(nextMaxs);
    setLogScale(nextLogScale);
  }, [logScale, setLogScale, setVmaxPerPanel, setVminPerPanel]);

  const updatePlaybackLiveControls = React.useCallback((idx: number) => {
    const c = playRef.current;
    writeLiveFrameControls(rootRef.current, playbackSliderRef.current, playbackLiveCountRef.current, idx, Math.max(1, c.nSlices || nSlices || 1), c.loop);
  }, [nSlices]);

  const viewportTransformActive = React.useCallback(() => {
    if (imageRotation % 4 !== 0 || flipRows || flipCols) return true;
    const panels = visiblePanelIndices.length
      ? visiblePanelIndices
      : Array.from({ length: Math.max(1, nPanels || 1) }, (_, idx) => idx);
    for (const panelIdx of panels) {
      const state = linkPanels
        ? linkedStateLiveRef.current
        : (panelStatesLiveRef.current[panelIdx] || stateFor(panelIdx));
      if (
        Math.abs((state.zoom || 1) - 1) > 1e-3 ||
        Math.abs(state.panX || 0) > 0.5 ||
        Math.abs(state.panY || 0) > 0.5
      ) {
        return true;
      }
    }
    return false;
  }, [flipCols, flipRows, imageRotation, linkPanels, nPanels, stateFor, visiblePanelIndices]);

  /**
   * Whether a GPU slot cannot show the current view, so the direct WebGPU draw must decline and the canvas
   * path paints. A resident slot holds the untransformed frame: with a moving average, difference, denoise,
   * frequency filter or alignment on it would show raw pixels (a scratch slot the GPU moving average filled is
   * already transformed), and a compare view is painted from two frames on the 2D canvas.
   */
  const gpuSlotMissesTheView = (residentSlotIdx: number | null): boolean => (
    String(compareMode || "off") !== "off" || (residentSlotIdx === null && frameTransformActive())
  );
  const frameTransformActive = () => requiresClientFrameTransform({
    diffMode: playRef.current.diffMode,
    avgWindow: playRef.current.avgWindow,
  }) || browserFilterOnRef.current || frequencyFilterIsActive || !!subpixelAlignEnabled;
  // Auto contrast keeps the stack-wide range through a moving average: a mean of
  // frames stays on the stack's scale and averaging must only reduce noise, as
  // on the WebGPU resident average. A frame difference or a browser, frequency
  // or alignment filter puts values on another scale, so those frames take
  // their own percentiles.
  const stackAutoRangeApplies = () => !(
    shouldApplyClientDifference(playRef.current.diffMode)
    || browserFilterOnRef.current || frequencyFilterIsActive || !!subpixelAlignEnabled
  );

  const rawFrameForIndex = (idx: number, currentIdx: number, currentFrame: Float32Array | null): Float32Array | null => {
    const frameCount = Math.max(1, nSlices || 1);
    const normalized = ((Math.round(idx) % frameCount) + frameCount) % frameCount;
    if (currentFrame && normalized === ((Math.round(currentIdx) % frameCount) + frameCount) % frameCount) return currentFrame;
    return getOfflineFrame(normalized);
  };

  // Mean of `avg_window` consecutive frames (temporal denoise). At the stack
  // ends the window SLIDES INWARD to stay full-width (frame 0, win 5 -> [0..4])
  // rather than shrinking - constant denoise strength, but the average is not
  // centered on `idx` near the ends. Even windows are front-biased.
  const averagedFrameForIndex = (idx: number, currentIdx: number, currentFrame: Float32Array | null): Float32Array | null => {
    const frameSize = width * height;
    const win = normalizedAverageWindow(playRef.current.avgWindow);
    if (win <= 1) return rawFrameForIndex(idx, currentIdx, currentFrame);
    const frameCount = Math.max(1, nSlices || 1);
    const indices = temporalAverageFrameIndices(idx, frameCount, win);
    const start = indices[0], end = indices[indices.length - 1];
    if (offlineFloatStack && offlineFloatStack.byteLength >= frameCount * frameSize * 4) {
      const out = new Float32Array(frameSize);
      let count = 0;
      for (let j = start; j <= end; j++) {
        const frame = float32FrameFromDataView(offlineFloatStack, j, frameSize, false);
        if (!frame || frame.length < frameSize) continue;
        for (let k = 0; k < frameSize; k++) out[k] += frame[k];
        count++;
      }
      if (count > 0) {
        const inv = 1 / count;
        for (let k = 0; k < frameSize; k++) out[k] *= inv;
        return out;
      }
    }
    if (offlineStack && offlineStack.byteLength >= frameCount * frameSize) {
      const out = new Float32Array(frameSize);
      let count = 0;
      for (let j = start; j <= end; j++) {
        const frame = getOfflineFrame(j);
        if (!frame || frame.length < frameSize) continue;
        for (let k = 0; k < frameSize; k++) out[k] += frame[k];
        count++;
      }
      if (count > 0) {
        const inv = 1 / count;
        for (let k = 0; k < frameSize; k++) out[k] *= inv;
        return out;
      }
    }
    const out = new Float32Array(frameSize);
    let count = 0;
    for (let j = start; j <= end; j++) {
      const frame = rawFrameForIndex(j, currentIdx, currentFrame);
      if (!frame || frame.length < frameSize) continue;
      for (let k = 0; k < frameSize; k++) out[k] += frame[k];
      count++;
    }
    if (count === 0) return rawFrameForIndex(idx, currentIdx, currentFrame);
    if (count > 1) {
      const inv = 1 / count;
      for (let k = 0; k < frameSize; k++) out[k] *= inv;
    }
    return out;
  };

  // Temporal mean of an RGB window: the color twin of averagedFrameForIndex.
  // Averages 3-channel frames straight from the offline color stack so avg
  // denoises true-color playback without collapsing to luminance.
  const averagedRgbFrameForIndex = (idx: number, fallback: Float32Array): Float32Array => {
    const win = normalizedAverageWindow(playRef.current.avgWindow);
    if (win <= 1) return fallback;
    const frameCount = Math.max(1, nSlices || 1);
    const center = Math.max(0, Math.min(frameCount - 1, Math.round(idx)));
    const half = Math.floor(win / 2);
    let start = center - half;
    let end = start + win - 1;
    if (start < 0) { end = Math.min(frameCount - 1, end - start); start = 0; }
    if (end >= frameCount) { start = Math.max(0, start - (end - frameCount + 1)); end = frameCount - 1; }
    const size = width * height * 3;
    const out = new Float32Array(size);
    let count = 0;
    for (let j = start; j <= end; j++) {
      const frame = j === center ? fallback : getOfflineFrame(j);
      if (!frame || frame.length < size) continue;
      for (let k = 0; k < size; k++) out[k] += frame[k];
      count++;
    }
    if (count === 0) return fallback;
    const inv = 1 / count;
    for (let k = 0; k < size; k++) out[k] *= inv;
    return out;
  };

  const frequencyFilterKeyForIndex = React.useCallback((idx: number) => {
    const mode = normalizeFrequencyFilterMode(frequencyFilter);
    const scopedFrequencyKey = frequencyFilterScopeAll
      ? ""
      : Array.from({ length: Math.max(1, nPanels || 1) }, (_, panel) => {
          const knobs = frequencyKnobsForPanel(panel);
          return `${panel}:${knobs.mode}:${Number(knobs.cutoff).toFixed(4)}:${Number(knobs.center).toFixed(4)}:${Number(knobs.width).toFixed(4)}`;
        }).join("|");
    const packedPanels = (Math.max(1, nPanels || 1) > 1 && !sharedPanelSource) ? Math.max(1, nPanels || 1) : 1;
    // Frequency filtering runs after display denoise/diff/avg. Its cache key
    // must include the upstream display transform; otherwise a σ/bin/mode
    // change can correctly update the denoise cache but the final painted
    // frequency-filtered frame stays frozen on the old upstream pixels.
    const denoiseKey = browserFilterKnobsOn
      ? `${denoiseResolved.mode}:${Number(denoiseSigmaLive ?? 0).toFixed(3)}:bin${denoiseResolved.bin}`
      : "raw";
    return [
      Math.round(idx),
      frameSeq,
      denoiseKey,
      `avg${playRef.current.avgWindow}`,
      `diff${playRef.current.diffMode}`,
      frequencyFilterScopeAll ? mode : scopedFrequencyKey,
      Number(frequencyOptions.cutoff ?? 0).toFixed(4),
      Number(frequencyOptions.center ?? 0).toFixed(4),
      Number(frequencyOptions.width ?? 0).toFixed(4),
      `panels${packedPanels}`,
    ].join(":");
  }, [browserFilterKnobsOn, denoiseResolved.mode, denoiseResolved.bin, denoiseSigmaLive, frameSeq, frequencyFilter, frequencyFilterScopeAll, frequencyKnobsForPanel, frequencyOptions, nPanels, sharedPanelSource]);

  const frequencyFilterFrameForDisplay = React.useCallback((idx: number, frame: Float32Array | null, options: { allowRawOnMiss?: boolean } = {}): Float32Array | null => {
    if (!frame || !frequencyFilterIsActive) return frame;
    const allowRawOnMiss = options.allowRawOnMiss === true;
    const key = frequencyFilterKeyForIndex(idx);
    const cache = frequencyFilterCacheRef.current;
    const hit = cache.get(key);
    if (hit) return hit;
    if (frequencyFilterPendingRef.current.has(key)) return allowRawOnMiss ? frame : null;
    frequencyFilterPendingRef.current.add(key);
    applyPackedPanelTransform(
      frame,
      (panelFrame, panelWidth, panelHeight, panel) => {
        const knobs = frequencyFilterScopeAll ? frequencyOptions : frequencyKnobsForPanel(panel);
        if (!frequencyFilterActive(knobs.mode)) return Promise.resolve(panelFrame);
        return applyFrequencyFilterBrowser(panelFrame, panelWidth, panelHeight, knobs);
      },
    )
      .then((filtered) => {
        frequencyFilterPendingRef.current.delete(key);
        cache.set(key, filtered);
        if (cache.size > 48) cache.delete(cache.keys().next().value as string);
        setFrequencyFilterBackend(getFrequencyFilterBackend());
        setFrequencyRenderVersion((value) => value + 1);
      })
      .catch((error) => {
        frequencyFilterPendingRef.current.delete(key);
        console.warn("[Show3D] frequency filter failed; showing unfiltered frame", error);
      });
    return allowRawOnMiss ? frame : null;
  }, [applyPackedPanelTransform, frequencyFilterIsActive, frequencyFilterKeyForIndex, frequencyFilterScopeAll, frequencyKnobsForPanel, frequencyOptions, height, width]);

  // One frame of a compare pair as analysed: the moving average, then the sub-pixel alignment. The compare
  // painter reads its frames here too; reading them back from the displayed frame, which already holds the
  // compare result, painted frame A where B - A belonged.
  const compareFrameFor = (frameIdx: number): Float32Array | null => {
    const clamped = clampSlice(frameIdx);
    return subpixelAlignFrameForIndex(clamped, averagedFrameForIndex(clamped, clamped, getOfflineFrame(clamped)));
  };
  const measuredFrameForIndex = (idx: number, currentFrame: Float32Array | null): Float32Array | null => {
    const activeCompareMode = String(compareMode || "off");
    let frame = subpixelAlignFrameForIndex(idx, averagedFrameForIndex(idx, idx, currentFrame));
    if (!isRgb && (nPanels || 1) === 1 && activeCompareMode !== "off") {
      const aIdx = clampSlice(comparePair?.[0] ?? 0);
      const bIdx = clampSlice(comparePair?.[1] ?? Math.min(1, nSlices - 1));
      const frameA = compareFrameFor(aIdx);
      const frameB = compareFrameFor(bIdx);
      if (activeCompareMode === "blink") {
        frame = (blinkPhase % 2 === 0 ? frameA : frameB) || frame;
      } else if (frameA && frameB) {
        const frameSize = width * height;
        const out = new Float32Array(frameSize);
        if (activeCompareMode === "difference") {
          for (let k = 0; k < frameSize; k++) out[k] = frameB[k] - frameA[k];
        } else if (activeCompareMode === "overlay") {
          for (let k = 0; k < frameSize; k++) out[k] = 0.5 * frameA[k] + 0.5 * frameB[k];
        }
        frame = out;
      }
    }
    const activeDiffMode = playRef.current.diffMode;
    let result: Float32Array | null = frame;
    if (frame && shouldApplyClientDifference(activeDiffMode)) {
      const refIdx = activeDiffMode === "first" ? 0 : Math.max(0, Math.round(idx) - 1);
      const ref = subpixelAlignFrameForIndex(refIdx, averagedFrameForIndex(refIdx, idx, currentFrame));
      if (ref) {
        const frameSize = width * height;
        const out = new Float32Array(frameSize);
        for (let k = 0; k < frameSize; k++) out[k] = frame[k] - ref[k];
        result = out;
      }
    }
    return result;
  };

  // Browser-side denoise (WGSL, LIVE sigma) and then the frequency filter on a measured frame.
  const viewFilteredFrame = (idx: number, measured: Float32Array | null, options: { allowRawOnMiss?: boolean } = {}): Float32Array | null => {
    const display = browserFilterFrame(idx, measured, options);
    if (frequencyFilterIsActive && browserFilterKnobsOn && !browserFilterReadyForIndex(idx)) {
      // Denoise/filter are layered display transforms. On a denoise cache miss,
      // displayFrameForIndex returns the pre-denoise fallback so the canvas can
      // stay responsive while WGSL finishes. Do not let the frequency filter
      // cache that fallback under the new sigma/bin key; otherwise a scientist
      // can drag sigma, see the label move, and keep looking at low-pass(raw).
      return display;
    }
    return frequencyFilterFrameForDisplay(idx, display, options);
  };
  const displayAndFrequencyFrameForIndex = (idx: number, currentFrame: Float32Array | null, options: { allowRawOnMiss?: boolean } = {}): Float32Array | null => (
    viewFilteredFrame(idx, measuredFrameForIndex(idx, currentFrame), options)
  );
  const refreshCurrentDisplayFrameForTransform = React.useCallback(() => {
    if (isRgb || width <= 0 || height <= 0 || nSlices <= 0) return;
    const idx = clampSlice(liveSliceIdx);
    const raw = getOfflineFrame(idx);
    if (!raw) return;
    const measured = measuredFrameForIndex(idx, raw);
    const display = viewFilteredFrame(idx, measured, { allowRawOnMiss: false });
    if (!display) return;
    measuredFrameRef.current = measured;
    displayFrameRef.current = display;
    rgbFrameDataRef.current = null;
    gpuUploadRef.current = null;
  }, [
    avgWindow,
    browserFilterTick,
    diffMode,
    displayFilter,
    frequencyFilterIsActive,
    frequencyRenderVersion,
    getOfflineFrame,
    height,
    isRgb,
    liveSliceIdx,
    nSlices,
    spatialBin,
    subpixelAlignEnabled,
    subpixelAlignVersion,
    width,
  ]);

  // A completed sub-pixel alignment changes the display transform, not the
  // underlying frame bytes. Refresh the currently visible frame before
  // the passive canvas paint effect runs; otherwise the More → Align button can
  // report "Aligned" while the canvas still holds the old unaligned buffer until
  // the user scrubs or toggles another display control.
  React.useLayoutEffect(() => {
    refreshCurrentDisplayFrameForTransform();
  }, [refreshCurrentDisplayFrameForTransform]);

  const warmPlaybackDisplayFrame = (idx: number, currentIdx: number, currentFrame: Float32Array | null) => {
    const raw = rawFrameForIndex(idx, currentIdx, currentFrame);
    if (!raw) return;
    void displayAndFrequencyFrameForIndex(idx, raw, { allowRawOnMiss: false });
  };

  const sharedDirectDisplayRange = (
    normalized: number,
    c: typeof playRef.current,
  ): RenderRange => {
    if (c.autoContrast) {
      const cached = cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, normalized, c.logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, normalized, c.logScale);
      if (cached) return cached;
    }
    return resolveDisplayRange(
      c.dataMin,
      c.dataMax,
      c.traitVmin,
      c.traitVmax,
      c.logScale,
      c.imageVminPct,
      c.imageVmaxPct,
    );
  };

  const directPanelRanges = (
    normalized: number,
    panels: number[],
    c: typeof playRef.current,
  ): RenderRange | RenderRange[] => {
    // A packed multi-panel source remains multi-panel even when the scientist
    // hides all but one panel. Basing this branch on `panels.length` decoded an
    // isolated phase panel against the global phase+BF+DF count range during
    // GPU scrubbing/playback, so the visible phase image became black.
    if (Math.max(1, nPanels || 1) > 1 && !sharedPanelSource) {
      const sharedAutoRange = c.autoContrast ? sharedDirectDisplayRange(normalized, c) : null;
      const stack = resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale);
      return panels.map((panel) => {
        const panelDataRange = panelDataRanges[panel];
        const bounds = (panelDataRange && panelDataRange.max > panelDataRange.min) ? panelDataRange : stack;
        return resolvePanelRenderRange(
          panel,
          bounds,
          sharedAutoRange,
          null,
          c.autoContrast,
          c.percentileLow,
          c.percentileHigh,
        );
      });
    }
    return sharedDirectDisplayRange(normalized, c);
  };

  const directPanelTransforms = (
    panels: number[],
    c: typeof playRef.current,
  ): { zoom: number; panX: number; panY: number }[] => panels.map((panel) => {
    const base = c.panelStates[panel] || initialState;
    return {
      zoom: c.linkPanels ? c.linkedState.zoom : base.zoom,
      panX: c.linkPanels ? c.linkedState.panX : base.panX,
      panY: c.linkPanels ? c.linkedState.panY : base.panY,
    };
  });

  const renderGpuPackedPanelTransformSlice = (
    idx: number,
    updateDisplayState = false,
    residentSlotIdx: number | null = null,
  ): boolean => {
    if (
      sharedPanelSource
      || isRgb
      || flipRows
      || flipCols
      || imageRotation % 4 !== 0
    ) {
      return false;
    }
    const panelSourceCount = Math.max(1, nPanels || 1);
    if (panelSourceCount <= 1 || width <= 0 || height <= 0 || canvasW <= 0 || canvasH <= 0) return false;
    const normalized = ((Math.round(idx) % Math.max(1, nSlices)) + Math.max(1, nSlices)) % Math.max(1, nSlices);
    const engine = gpuCmapRef.current;
    if (!engine || !gpuCmapReadyRef.current) return false;
    const c = playRef.current;
    if (c.imageRotation % 4 !== 0 || c.width <= 0 || c.height <= 0 || c.canvasW <= 0 || c.canvasH <= 0) return false;
    const panels = (c.visiblePanelIndices.length ? c.visiblePanelIndices : visiblePanelIndices)
      .filter((panel) => Number.isFinite(panel) && panel >= 0 && panel < panelSourceCount);
    if (panels.length === 0) return false;
    const sourcePanelWidth = Math.max(1, panelWidthPx || Math.round(c.width / panelSourceCount));
    if (sourcePanelWidth <= 0 || sourcePanelWidth > c.width) return false;
    const gpuCtx = ensureGpuDisplayContext(engine, c.canvasW, c.canvasH);
    if (!gpuCtx) return false;

    if (gpuSlotMissesTheView(residentSlotIdx)) return false;
    const rawCurrentFrame = getOfflineFrame(normalized);
    let slotIdx: number | null = residentSlotIdx;
    let renderLogScale = c.logScale;
    if (slotIdx !== null) {
      // A GPU transform such as temporal averaging already populated this
      // scratch slot. Do not upload or touch the raw resident frame.
    } else if (gpuFrameCacheUploadedRef.current.has(normalized)) {
      slotIdx = normalized;
    } else {
      const upload = gpuUploadRef.current;
      if (
        upload &&
        rawCurrentFrame &&
        upload.source === rawCurrentFrame &&
        upload.width === c.width &&
        upload.height === c.height
      ) {
        slotIdx = 0;
        renderLogScale = upload.logScale ? false : c.logScale;
      } else if (rawCurrentFrame && rawCurrentFrame.length >= c.width * c.height) {
        const rgbaCapacity = Math.max(1, Math.round(c.canvasW * c.canvasH));
        engine.uploadData(normalized, rawCurrentFrame, c.width, c.height, rgbaCapacity);
        gpuFrameCacheUploadedRef.current.add(normalized);
        slotIdx = normalized;
      }
    }
    if (slotIdx === null) return false;

    const liveCmap = cmapLiveRef.current || c.cmap;
    const lut = COLORMAPS[liveCmap] || COLORMAPS.inferno;
    engine.uploadLUT(liveCmap, lut);
    const panelCount = panels.length;
    // Playback continues in one long rAF effect while layout controls remain
    // live. Read columns from playRef with the other live canvas geometry;
    // panelColsForCount here would be the closure captured when Play started.
    const cols = Math.max(1, Math.min(panelCount, Math.round(c.panelCols || 1)));
    const rows = Math.ceil(panelCount / cols);
    const gap = panelCount > 1 ? panelGapPx : 0;
    const ranges = directPanelRanges(normalized, panels, c);
    const transforms = directPanelTransforms(panels, c);
    // Sharp panels enlarged by a non-integer factor in the default view take the Canvas2D draw, as
    // in the single-panel path, so WebGPU and CPU pick the same texel at texel edges.
    const slotW = (c.canvasW - gap * (cols - 1)) / cols;
    const slotH = (c.canvasH - gap * (rows - 1)) / rows;
    if (
      !c.smooth
      && (slotW > sourcePanelWidth || slotH > c.height)
      && (!Number.isInteger(slotW / sourcePanelWidth) || !Number.isInteger(slotH / c.height))
      && transforms.every((t) => t.zoom === 1 && t.panX === 0 && t.panY === 0)
    ) {
      return false;
    }
    const livePanelCmaps = panelCmapsLiveRef.current;
    const panelLuts = livePanelCmaps.length === panelSourceCount
      ? panels.map((panel) => {
          const name = livePanelCmaps[panel] && COLORMAPS[livePanelCmaps[panel]]
            ? livePanelCmaps[panel]
            : liveCmap;
          return { name, lut: COLORMAPS[name] || lut };
        })
      : undefined;
    const rendered = engine.renderCombinedPanelRegionsDirectToCanvas(
      slotIdx,
      ranges,
      renderLogScale,
      gpuCtx,
      {
        width: c.canvasW,
        height: c.canvasH,
        panelCount,
        cols,
        rows,
        gap,
        bgRgb: packedRgbFromHex(interPanelGapColor),
        sourcePanelWidth,
        transforms,
        sourcePanelIndices: panels,
        panelLuts,
        smooth: c.smooth,
      },
    );
    if (!rendered) return false;
    setGpuDisplayVisible(true);
    if (!playing) {
      retainGpuDisplayFrame(
        engine.renderCombinedPanelRegionsToImageBitmapAsync(
          slotIdx,
          ranges,
          renderLogScale,
          {
            width: c.canvasW,
            height: c.canvasH,
            panelCount,
            cols,
            rows,
            gap,
            bgRgb: packedRgbFromHex(interPanelGapColor),
            sourcePanelWidth,
            transforms,
            sourcePanelIndices: panels,
            panelLuts,
            smooth: c.smooth,
          },
        ),
      );
    }
    playbackIdxRef.current = normalized;
    if (updateDisplayState) setDisplaySliceIdx(normalized);
    return true;
  };

  const renderGpuCachedSliceDirect = (
    idx: number,
    updateDisplayState = true,
    residentSlotIdx: number | null = null,
  ): boolean => {
    const normalized = ((Math.round(idx) % Math.max(1, nSlices)) + Math.max(1, nSlices)) % Math.max(1, nSlices);
    if (residentSlotIdx === null && !gpuFrameCacheUploadedRef.current.has(normalized)) return false;
    if (gpuSlotMissesTheView(residentSlotIdx)) return false;
    const renderSlotIdx = residentSlotIdx ?? normalized;
    if (
      Math.max(1, nPanels || 1) > 1 &&
      renderGpuPackedPanelTransformSlice(normalized, updateDisplayState, residentSlotIdx)
    ) {
      return true;
    }
    const engine = gpuCmapRef.current;
    if (!engine || !gpuCmapReadyRef.current) return false;
    const c = playRef.current;
    if (c.imageRotation % 4 !== 0 || c.zoom !== 1 || c.panX !== 0 || c.panY !== 0) return false;
    const naturalVisibleOrder = visiblePanelIndices.length === Math.max(1, nPanels || 1)
      && visiblePanelIndices.every((panel, slot) => panel === slot);
    const panelViewsAreDefault = visiblePanelIndices.every(panel => {
      const state = c.panelStates[panel] || initialState;
      const view = c.linkPanels ? c.linkedState : state;
      return view.zoom === 1 && view.panX === 0 && view.panY === 0;
    });
    if (!naturalVisibleOrder || !panelViewsAreDefault) return false;
    // The grid fragment shader samples nearest. An enlarged view goes through the Canvas2D draw,
    // the same as the CPU path, when Smooth is on (bilinear) or the factor is not an integer: there
    // a pixel centre can land exactly on a texel edge, where the two samplers pick different texels.
    const enlarged = c.width < c.canvasW || c.height < c.canvasH;
    if (enlarged && (c.smooth || c.canvasW % c.width !== 0 || c.canvasH % c.height !== 0)) return false;
    const gpuCtx = ensureGpuDisplayContext(engine, c.canvasW, c.canvasH);
    if (!gpuCtx) return false;

    const liveCmap = cmapLiveRef.current || c.cmap;
    const lut = COLORMAPS[liveCmap] || COLORMAPS.inferno;
    engine.uploadLUT(liveCmap, lut);
    let vmin: number, vmax: number;
    if (c.autoContrast) {
      const cached = cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, normalized, c.logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, normalized, c.logScale);
      if (cached) {
        ({ vmin, vmax } = cached);
      } else {
        ({ vmin, vmax } = resolveDisplayRange(
          c.dataMin,
          c.dataMax,
          c.traitVmin,
          c.traitVmax,
          c.logScale,
          c.imageVminPct,
          c.imageVmaxPct,
        ));
      }
    } else {
      ({ vmin, vmax } = resolveDisplayRange(
        c.dataMin,
        c.dataMax,
        c.traitVmin,
        c.traitVmax,
        c.logScale,
        c.imageVminPct,
        c.imageVmaxPct,
      ));
    }

    if (hiddenPanelSet.size > 0) return false;
    const panelCount = Math.max(1, nPanels || 1);
    const cols = Math.max(1, Math.min(panelCount, Math.round(c.panelCols || 1)));
    const rows = Math.ceil(panelCount / cols);
    const gap = panelCount > 1 ? (panelGapPx) : 0;
    if (panelCount > 1 && !sharedPanelSource) {
      const panelW = Math.max(1, panelWidthPx || Math.round(c.width / panelCount));
      const regions = Array.from({ length: panelCount }, (_, panel) => ({
        x: panel * panelW, y: 0, width: panelW, height: c.height,
      }));
      const sharedAutoRange = c.autoContrast ? { vmin, vmax } : null;
      // The ranges come from the frame's own pixels (gpuSlotMissesTheView declined every transformed view
      // without a GPU-filled scratch slot), read by index: the displayed frame may belong to another index.
      const frameForRanges = getOfflineFrame(normalized);
      const ranges = Array.from({ length: panelCount }, (_, panel) => {
        const stack = resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale);
        const panelData = frameForRanges ? extractPanelSlice(frameForRanges, panel, c.logScale) : null;
        const panelDataRange = panelDataRanges[panel];
        const bounds = (panelData && panelData.length > 0)
          ? findDataRange(panelData)
          : ((perPanelHistogramEnabled && panelDataRange && panelDataRange.max > panelDataRange.min) ? panelDataRange : stack);
        return resolvePanelRenderRange(panel, bounds, sharedAutoRange, panelData, c.autoContrast, c.percentileLow, c.percentileHigh);
      });
      const logs = c.logScale;
      const bitmaps = engine.renderPerPanelGpuExplicit(renderSlotIdx, regions, ranges, logs);
      const offCtx = mainOffscreenRef.current?.getContext("2d");
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!bitmaps) return false;
      try {
        if (!offCtx || !ctx || !mainOffscreenRef.current) return false;
        offCtx.clearRect(0, 0, c.width, c.height);
        for (let panel = 0; panel < panelCount; panel++) {
          if (bitmaps[panel]) {
            offCtx.drawImage(bitmaps[panel], panel * panelW, 0);
          }
        }
      } finally {
        bitmaps.forEach(bitmap => bitmap?.close());
      }
      drawMain(ctx, mainOffscreenRef.current);
      setGpuDisplayVisible(false);
      playbackIdxRef.current = normalized;
      if (updateDisplayState) setDisplaySliceIdx(normalized);
      return true;
    }
    const sourcePanelWidthForGrid = sharedPanelSource
      ? Math.max(1, panelWidthPx || c.width)
      : Math.max(1, panelWidthPx || Math.round(c.width / panelCount));
    const gridOpts = {
      width: c.canvasW,
      height: c.canvasH,
      panelCount,
      cols,
      rows,
      gap,
      bgRgb: packedRgbFromHex(interPanelGapColor),
      sourcePanelWidth: sourcePanelWidthForGrid,
      sharedSource: !!sharedPanelSource,
    };
    const rendered = engine.renderSharedGridDirectToCanvas(
      renderSlotIdx,
      { vmin, vmax },
      c.logScale,
      gpuCtx,
      gridOpts,
    );
    if (!rendered) return false;
    setGpuDisplayVisible(true);
    if (!playing) {
      retainGpuDisplayFrame(
        engine.renderSharedGridToImageBitmapAsync(
          renderSlotIdx,
          { vmin, vmax },
          c.logScale,
          {
            width: c.canvasW,
            height: c.canvasH,
            panelCount,
            cols,
            rows,
            gap,
            bgRgb: packedRgbFromHex(interPanelGapColor),
            sourcePanelWidth: sourcePanelWidthForGrid,
            sharedSource: !!sharedPanelSource,
          },
        ),
      );
    }
    playbackIdxRef.current = normalized;
    if (updateDisplayState) setDisplaySliceIdx(normalized);
    return true;
  };

  const renderGpuTemporalAverageSliceDirect = (
    idx: number,
    updateDisplayState = false,
  ): boolean => {
    const c = playRef.current;
    const win = normalizedAverageWindow(c.avgWindow);
    if (
      isRgb || win <= 1 || c.diffMode !== "off" ||
      browserFilterOnRef.current || frequencyFilterIsActive || subpixelAlignEnabled ||
      c.imageRotation % 4 !== 0
    ) return false;
    const engine = gpuCmapRef.current;
    if (!engine || !gpuCmapReadyRef.current) return false;
    const normalized = ((Math.round(idx) % Math.max(1, nSlices)) + Math.max(1, nSlices)) % Math.max(1, nSlices);
    const sourceSlots = temporalAverageFrameIndices(normalized, nSlices, win);
    if (!sourceSlots.every(sourceIdx => gpuFrameCacheUploadedRef.current.has(sourceIdx))) return false;
    const scratchSlot = Math.max(1, nSlices);
    if (!engine.averageResidentSlotsInto(scratchSlot, sourceSlots)) return false;
    if (!renderGpuCachedSliceDirect(normalized, updateDisplayState, scratchSlot)) return false;
    return true;
  };

  React.useEffect(() => {
    let cancelled = false;
    const frameCount = Math.max(1, nSlices || 1);
    const frameSize = Math.max(1, width * height * (isRgb ? 3 : 1));
    const hasEmbeddedDisplayStack = Boolean(
      (offlineFloatStack && offlineFloatStack.byteLength >= frameCount * frameSize * 4)
      || (offlineStack && offlineStack.byteLength >= frameCount * frameSize)
    );

    if (!hasEmbeddedDisplayStack) {
      setGpuResidency(nSlices > 0
        ? { stage: "missing", ready: 0, error: missingDisplayStackStatus(frameCount * frameSize * 4, Math.max(1, displayBin || 1)) }
        : { stage: "waiting", ready: 0, error: "" });
      return;
    }
    if (isRgb) {
      setGpuResidency({ stage: "rgb", ready: 0, error: "" });
      return;
    }
    if (!gpuCmapChecked) {
      setGpuResidency({ stage: "waiting", ready: 0, error: "" });
      return;
    }
    const engine = gpuCmapRef.current;
    const adapterInfo = getGPUInfo();
    if (!engine || !gpuCmapReadyRef.current || /swiftshader|software/i.test(adapterInfo)) {
      // say why, so the reader knows what to change: an insecure page hides navigator.gpu entirely
      const reason = typeof window !== "undefined" && !window.isSecureContext
        ? "not a secure context: open this page over https, localhost, or as a local file"
        : /swiftshader|software/i.test(adapterInfo) ? `software renderer (${adapterInfo})` : "no WebGPU adapter";
      setGpuResidency({ stage: "fallback", ready: 0, error: reason });
      return;
    }

    const current = ((Math.round(sliceIdx) % frameCount) + frameCount) % frameCount;
    const order = [current, ...Array.from({ length: frameCount }, (_, idx) => idx).filter(idx => idx !== current)];
    setGpuResidency({ stage: "uploading", ready: 0, error: "" });
    gpuFrameCacheUploadedRef.current.clear();

    void (async () => {
      try {
        for (let position = 0; position < order.length; position++) {
          if (cancelled) return;
          const idx = order[position];
          const frame = getOfflineFrame(idx);
          if (!frame || frame.length < frameSize) {
            throw new Error(`display frame ${idx + 1} is unavailable`);
          }
          engine.uploadData(idx, frame, width, height, 1, true);
          gpuFrameCacheUploadedRef.current.add(idx);
          const ready = position + 1;
          setGpuResidency({ stage: "uploading", ready, error: "" });
          if (position === 0) renderGpuCachedSliceDirect(current, false);
          if (position % 4 === 3) await new Promise<void>(resolve => window.setTimeout(resolve, 0));
        }
        await engine.waitForSubmittedWork();
        if (cancelled) return;
        setGpuResidency({ stage: "ready", ready: frameCount, error: "" });
        renderGpuCachedSliceDirect(current, false);
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : String(error);
        setGpuResidency({ stage: "fallback", ready: 0, error: message });
        gpuFrameCacheUploadedRef.current.clear();
      }
    })();

    return () => {
      cancelled = true;
      for (let idx = 0; idx < frameCount; idx++) engine.releaseSlot(idx);
      gpuFrameCacheUploadedRef.current.clear();
    };
  // The embedded stack identity and geometry define GPU residency. Display
  // controls repaint resident slots and must never trigger another upload.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offlineFloatStack, offlineStack, width, height, nSlices, isRgb, displayBin, gpuCmapChecked, gpuCmapReady]);


  // Slider gestures temporarily own the displayed frame without changing Play.
  const sliderScrubbingRef = React.useRef(false);
  const sliderGestureCleanupRef = React.useRef<(() => void) | null>(null);
  React.useEffect(() => () => sliderGestureCleanupRef.current?.(), []);
  const playbackHistogramCounterRef = React.useRef(0);
  const refreshHistogramRef = React.useRef<((idxArg?: number) => void | Promise<void>) | null>(null);

  // Playback logic - rAF-driven, zero React re-renders in hot path
  React.useEffect(() => {
    if (!playing) {
      // Playback stopped - sync final position to Python
      if (playbackIdxRef.current !== sliceIdx) {
        setLiveSliceIdx(playbackIdxRef.current);
        setSliceIdx(playbackIdxRef.current);
      }
      if (!playRef.current.showStats) setLocalStats(null);
      return;
    }

    // === PLAYBACK START ===
    // Snap slice_idx into [loop_start, loop_end] before first tick, otherwise
    // playback walked outside the loop range on the first frame.
    {
      const c0 = playRef.current;
      const rs0 = c0.loop ? Math.max(0, Math.min(c0.loopStart, c0.nSlices - 1)) : 0;
      const re0 = c0.loop ? Math.max(rs0, Math.min(c0.loopEnd, c0.nSlices - 1)) : c0.nSlices - 1;
      const liveStart = Number.isFinite(playbackIdxRef.current)
        ? playbackIdxRef.current
        : (Number.isFinite(displaySliceIdx) ? displaySliceIdx : sliceIdx);
      playbackIdxRef.current = Math.max(rs0, Math.min(re0, Math.round(liveStart)));
    }
    const pathLen = playRef.current.playbackPath?.length ?? 0;
    pathIdxRef.current = pathLen > 0 ? (playRef.current.reverse ? pathLen : -1) : 0;
    bounceDirRef.current = playRef.current.reverse ? -1 : 1;
    let lastFrameTime = 0;
    let lastUIUpdate = 0;
    let animId = 0;
    let tick: (now: number) => void = () => {};
    const scheduleTick = () => {
      animId = requestAnimationFrame(tick);
    };

    tick = (_now: number) => {
      if (sliderScrubbingRef.current) {
        lastFrameTime = 0;
        scheduleTick();
        return;
      }
      const tickNow = performance.now();
      const c = playRef.current;
      const effectiveFps = clampPlaybackFps(c.fps);
      const intervalMs = playbackIntervalMs(effectiveFps);
      const uiUpdateIntervalMs = effectiveFps >= 60 ? 250 : 100;

      // First tick paints immediately; otherwise every playback start drops
      // one frame before the cadence timer is even allowed to run.
      if (lastFrameTime === 0) {
        lastFrameTime = tickNow - intervalMs;
        lastUIUpdate = tickNow;
      }

      const elapsed = tickNow - lastFrameTime;
      // Frame-pacing tolerance: at 60 fps intervalMs (16.67) equals the vsync
      // period, so a rAF tick arriving a hair early (elapsed 16.6 < 16.67) would
      // be dropped and cost a whole vsync -> steady 17/33 ms alternation = 30 fps.
      // Allow a tick that is within tolerance of the deadline through, and
      // phase-correct lastFrameTime by the deadline (not tickNow) so drift does
      // not accumulate. Restores 60 fps on the GPU-cached multi-panel path.
      const framePacingToleranceMs = Math.min(6, intervalMs * 0.2);
      if (elapsed + framePacingToleranceMs < intervalMs) {
        scheduleTick();
        return;
      }
      lastFrameTime = tickNow - Math.max(0, elapsed - intervalMs);

      let next: number;
      if (c.playbackPath && c.playbackPath.length > 0) {
        const pp = c.playbackPath;
        let pi = pathIdxRef.current;
        if (c.boomerang) {
          // Loop remains the master repeat control. When a scientist turns
          // Loop off, Bounce should shape motion only until the path endpoint;
          // it must not keep ping-ponging forever.
          // Visit endpoints once (matches grid-mode boomerang). Earlier code
          // jumped to pp.length-2 / 1 on overshoot, skipping endpoints.
          pi += bounceDirRef.current;
          if (pi >= pp.length) {
            if (!c.loop) { setPlaying(false); return; }
            bounceDirRef.current = -1;
            pi = pp.length - 1;
          } else if (pi < 0) {
            if (!c.loop) { setPlaying(false); return; }
            bounceDirRef.current = 1;
            pi = 0;
          }
        } else {
          pi += (c.reverse ? -1 : 1);
          if (pi >= pp.length) { if (!c.loop) { setPlaying(false); return; } pi = 0; }
          if (pi < 0) { if (!c.loop) { setPlaying(false); return; } pi = pp.length - 1; }
        }
        pi = Math.max(0, Math.min(pp.length - 1, pi));
        pathIdxRef.current = pi;
        next = pp[pi];
      } else {
        const rangeStart = c.loop ? Math.max(0, Math.min(c.loopStart, c.nSlices - 1)) : 0;
        const rangeEnd = c.loop ? Math.max(rangeStart, Math.min(c.loopEnd, c.nSlices - 1)) : c.nSlices - 1;
        const prev = Number.isFinite(playbackIdxRef.current)
          ? Math.round(playbackIdxRef.current)
          : Math.max(rangeStart, Math.min(rangeEnd, Math.round(displaySliceIdx || 0)));

        if (c.boomerang) {
          next = prev + bounceDirRef.current;
          if (next > rangeEnd) {
            if (!c.loop) { setPlaying(false); return; }
            bounceDirRef.current = -1;
            next = prev - 1 >= rangeStart ? prev - 1 : prev;
          } else if (next < rangeStart) {
            if (!c.loop) { setPlaying(false); return; }
            bounceDirRef.current = 1;
            next = prev + 1 <= rangeEnd ? prev + 1 : prev;
          }
        } else {
          next = prev + (c.reverse ? -1 : 1);
          if (c.reverse) {
            if (next < rangeStart) { if (!c.loop) { setPlaying(false); return; } next = rangeEnd; }
          } else {
            if (next > rangeEnd) { if (!c.loop) { setPlaying(false); return; } next = rangeStart; }
          }
        }
      }

      // The embedded stack owns playback. Select its resident GPU slot when
      // available; otherwise use the on-demand canvas fallback below.
      const transformActive = requiresClientFrameTransform({
        diffMode: c.diffMode,
        avgWindow: c.avgWindow,
      }) || browserFilterOnRef.current || frequencyFilterIsActive;
      let frame: Float32Array | null = null;
      // The GPU-cache fast paths (renderGpuCachedSliceDirect and the resident
      // average) only handle imageRotation%4===0 and return false on a 90/270
      // rotation, which froze playback (renderedFrames + canvas stuck, playing
      // true). When rotated, skip the GPU-cache path so the frame is fetched and
      // drawMain applies the rotation. Verified bug 2026-05-29.
      const rotationAllowsGpuCache = (c.imageRotation % 4) === 0;
      let gpuResidentTransformReady = false;
      if (rotationAllowsGpuCache && normalizedAverageWindow(c.avgWindow) > 1) {
        try {
          gpuResidentTransformReady = renderGpuTemporalAverageSliceDirect(next, false);
        } catch {
          // a failed GPU draw falls back to the embedded frame below
        }
      }
      const gpuCachedSlotReady = !transformActive
        && rotationAllowsGpuCache
        && gpuFrameCacheUploadedRef.current.has(next);
      // Cache presence alone is not render readiness: hidden/reordered panels
      // and non-default transforms can make the direct path decline. Probe the
      // actual renderer; on a miss we still acquire a CPU frame instead
      // of advancing the slider over a frozen canvas.
      let gpuCachedFrameReady = false;
      if (gpuCachedSlotReady) {
        try {
          gpuCachedFrameReady = renderGpuCachedSliceDirect(next, false);
        } catch {
          // a failed GPU draw falls back to the embedded frame below
        }
      }
      frame = getOfflineFrame(next);
      if (!frame && !gpuCachedFrameReady && !gpuResidentTransformReady) {
        // The embedded frame is not available yet; retry on the next tick.
        scheduleTick();
        return;
      }

      const sourceFrame = frame;
      if (frame && transformActive && !gpuResidentTransformReady && !isRgb) {
        const filteredFrame = displayAndFrequencyFrameForIndex(next, frame, { allowRawOnMiss: false });
        if (!filteredFrame) {
          warmPlaybackDisplayFrame(next, playbackIdxRef.current, frame);
          scheduleTick();
          return;
        }
        frame = filteredFrame;
      }
      playbackIdxRef.current = next;
      updatePlaybackLiveControls(next);
      if (frame && isRgb && frame.length >= c.width * c.height * 3) {
        rgbFrameDataRef.current = frame;
        sourceFrameDataRef.current = sourceFrame;
        displayFrameRef.current = rgbFrameToLuminance(frame, c.width * c.height);
      } else if (frame) {
        sourceFrameDataRef.current = sourceFrame;
        displayFrameRef.current = frame;
      }
      const directRender = (
        !isRgb &&
        !!frame &&
        !!gpuCmapRef.current &&
        gpuCmapReadyRef.current
      );
      // Static paint is driven by liveSliceIdx. When WebGPU is ready we render
      // frames directly in the rAF hot path and throttle React state updates
      // below so large 2k/4k stacks do not double-paint.
      const gpuDirectRender = gpuCachedFrameReady || gpuResidentTransformReady;
      if (!directRender && !gpuDirectRender) setLiveSliceIdx(next);
      // Without a direct WebGPU paint, hand the frame to the React static paint
      // pipeline (proven smooth on slider drag) and skip the rAF direct paint
      // entirely. The two paths fought on Mac/retina (Linux didn't expose it),
      // producing the "play is flaky while drag is smooth" symptom verified
      // 2026-05-24 on sample_device_trial.html.
      if (!directRender) {
        setGpuDisplayVisible(false);
        if (tickNow - lastUIUpdate > uiUpdateIntervalMs) {
          lastUIUpdate = tickNow;
          setDisplaySliceIdx(next);
          setPlaybackUiSliceIdx(next);
          playbackHistogramCounterRef.current = (playbackHistogramCounterRef.current + 1) % 2;
          if (playbackHistogramCounterRef.current === 0) {
            void refreshHistogramRef.current?.(next);
          }
        }
        scheduleTick();
        return;
      }
      if (gpuCachedFrameReady || gpuResidentTransformReady) {
        if (tickNow - lastUIUpdate > uiUpdateIntervalMs) {
          lastUIUpdate = tickNow;
          setDisplaySliceIdx(next);
          setPlaybackUiSliceIdx(next);
          playbackHistogramCounterRef.current = (playbackHistogramCounterRef.current + 1) % 2;
          if (playbackHistogramCounterRef.current === 0) void refreshHistogramRef.current?.(next);
        }
        scheduleTick();
        return;
      }

      // Render frame. The 4k playback hot path must stay off the JS CPU:
      // one 4096^2 colormap loop alone is ~37 ms, before auto-contrast/canvas.
      const liveCmap = cmapLiveRef.current || c.cmap;
      const lut = COLORMAPS[liveCmap] || COLORMAPS.inferno;
      if (mainOffscreenRef.current && mainImgDataRef.current) {
        let vmin: number, vmax: number;
        let cpuData: Float32Array | null = frame;
        let cpuDataAlreadyLogged = false;
        if (c.autoContrast) {
          const cached = stackAutoRangeApplies() ? (
            cachedAutoDisplayRange(c.autoVmins, c.autoVmaxs, next, c.logScale)
            || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, next, c.logScale)
          ) : null;
          if (cached) {
            ({ vmin, vmax } = cached);
          } else if (frame && c.logScale && logBufferRef.current) {
            applyLogScaleInPlace(frame, logBufferRef.current);
            ({ vmin, vmax } = percentileClip(logBufferRef.current, c.percentileLow, c.percentileHigh));
            cpuData = logBufferRef.current;
            cpuDataAlreadyLogged = true;
          } else if (frame) {
            ({ vmin, vmax } = percentileClip(frame, c.percentileLow, c.percentileHigh));
          } else {
            ({ vmin, vmax } = resolveDisplayRange(
              c.dataMin,
              c.dataMax,
              c.traitVmin,
              c.traitVmax,
              c.logScale,
              c.imageVminPct,
              c.imageVmaxPct,
            ));
          }
        } else {
          ({ vmin, vmax } = resolveDisplayRange(
            c.dataMin,
            c.dataMax,
            c.traitVmin,
            c.traitVmax,
            c.logScale,
            c.imageVminPct,
            c.imageVmaxPct,
          ));
        }

        let rendered = false;
        let drewDisplayDirect = false;
        const dw = Math.max(1, Math.round(c.width * c.displayScale));
        const dh = Math.max(1, Math.round(c.height * c.displayScale));
        const panelCountForGrid = Math.max(1, nPanels || 1);
        // Playback stays on the stable 2D display canvas: direct WebGPU canvas
        // presentation can briefly win the opacity handoff before the next image
        // is visible, producing a black flash when playback starts. The resident
        // WebGPU path above still uses the direct GPU canvas.
        const canScaledDirect =
          nPanels === 1 &&
          c.imageRotation % 4 === 0 &&
          c.zoom === 1 &&
          c.panX === 0 &&
          c.panY === 0 &&
          dw <= c.canvasW &&
          dh <= c.canvasH;
        const renderPackedPanels2D = (): boolean => {
          if (panelCountForGrid <= 1 || sharedPanelSource || !frame) return false;
          const offscreen = mainOffscreenRef.current;
          const canvas = canvasRef.current;
          const offCtx = offscreen?.getContext("2d");
          const ctx = canvas?.getContext("2d");
          if (!offscreen || !offCtx || !ctx) return false;
          const panelW = Math.max(1, Math.floor(c.width / panelCountForGrid));
          const sourceW = frame.length === c.height * panelW ? panelW : c.width;
          if (sourceW <= 0 || frame.length < c.height * sourceW) return false;
          const panelImg = offCtx.createImageData(panelW, c.height);
          const sharedAutoRange = c.autoContrast ? { vmin, vmax } : null;
          const livePanelCmaps = panelCmapsLiveRef.current;
          offCtx.clearRect(0, 0, offscreen.width, offscreen.height);
          for (const panel of c.visiblePanelIndices) {
            if (panel < 0 || panel >= panelCountForGrid) continue;
            const srcPanel = Math.min(Math.max(0, panel), panelCountForGrid - 1);
            const x0 = Math.min(Math.max(0, srcPanel * panelW), Math.max(0, sourceW - panelW));
            const panelDataRange = panelDataRanges[panel];
            // Do not allocate/copy the panel during playback. The detailed
            // per-panel percentile window refreshes on idle/static paints; the
            // playback hot path reuses the remembered panel range (or the
            // stack range) so large real-data reports stay responsive.
            const panelRange = (panelDataRange && panelDataRange.max > panelDataRange.min)
              ? panelDataRange
              : resolveDisplayBounds(c.dataMin, c.dataMax, c.traitVmin, c.traitVmax, c.logScale);
            const range = resolvePanelRenderRange(panel, panelRange, sharedAutoRange, null, c.autoContrast, c.percentileLow, c.percentileHigh);
            const panelCmap = livePanelCmaps.length === panelCountForGrid
              ? livePanelCmaps[panel]
              : liveCmap;
            const panelLut = COLORMAPS[panelCmap] || lut;
            renderPackedPanelPlayback(frame, sourceW, x0, panelW, c.height, panelImg.data, panelLut, range.vmin, range.vmax, c.logScale);
            offCtx.putImageData(panelImg, panel * panelW, 0);
          }
          drawMain(ctx, offscreen);
          setGpuDisplayVisible(false);
          return true;
        };
        if (!rendered && renderPackedPanels2D()) {
          rendered = true;
          drewDisplayDirect = true;
        }
        const engine = gpuCmapRef.current;
        const preferGpuScaledPlayback = !!engine && gpuCmapReadyRef.current;
        if (frame && canScaledDirect && !c.smooth && !preferGpuScaledPlayback) {
          const canvas = canvasRef.current;
          const ctx = canvas?.getContext("2d");
          if (ctx) {
            let cached = scaledPlaybackImgDataRef.current;
            if (!cached || cached.width !== dw || cached.height !== dh) {
              cached = { width: dw, height: dh, imageData: ctx.createImageData(dw, dh) };
              scaledPlaybackImgDataRef.current = cached;
            }
            let map = scaledPlaybackMapRef.current;
            if (!map || map.srcW !== c.width || map.srcH !== c.height || map.outW !== dw || map.outH !== dh) {
              const xMap = new Uint32Array(dw);
              const yMap = new Uint32Array(dh);
              for (let x = 0; x < dw; x++) {
                xMap[x] = Math.min(c.width - 1, Math.floor(((x + 0.5) * c.width) / dw));
              }
              for (let y = 0; y < dh; y++) {
                yMap[y] = Math.min(c.height - 1, Math.floor(((y + 0.5) * c.height) / dh)) * c.width;
              }
              map = { srcW: c.width, srcH: c.height, outW: dw, outH: dh, xMap, yMap };
              scaledPlaybackMapRef.current = map;
            }
            renderFrameScaledPlayback(frame, cached.imageData.data, map.xMap, map.yMap, dw, dh, lut, vmin, vmax, c.logScale);
            ctx.imageSmoothingEnabled = false;
            ctx.clearRect(0, 0, c.canvasW, c.canvasH);
            ctx.putImageData(cached.imageData, 0, 0);
            setGpuDisplayVisible(false);
            rendered = true;
            drewDisplayDirect = true;
          }
        }
        if (!rendered && engine && gpuCmapReadyRef.current) {
          try {
            engine.uploadLUT(liveCmap, lut);
            const hasGpuSlot = gpuFrameCacheUploadedRef.current.has(next);
            const canGpuFrameCache = hasGpuSlot;
            const slotIdx = canGpuFrameCache ? next : 0;
            const gpuRgbaCapacityHint = canScaledDirect ? dw * dh : undefined;
            if (canGpuFrameCache) {
              if (!gpuFrameCacheUploadedRef.current.has(slotIdx)) {
                if (frame) {
                  engine.uploadData(slotIdx, frame, c.width, c.height, gpuRgbaCapacityHint);
                  gpuFrameCacheUploadedRef.current.add(slotIdx);
                }
              }
            } else if (frame) {
              engine.uploadData(0, frame, c.width, c.height, gpuRgbaCapacityHint);
            }
            if (canScaledDirect) {
              const bitmap = rendered
                ? null
                : engine.renderSlotScaledToImageBitmap(slotIdx, { vmin, vmax }, c.logScale, dw, dh);
              const canvas = canvasRef.current;
              const ctx = canvas?.getContext("2d");
              if (bitmap) {
                try {
                  if (ctx) {
                    ctx.imageSmoothingEnabled = c.smooth;
                    ctx.clearRect(0, 0, c.canvasW, c.canvasH);
                    ctx.drawImage(bitmap, 0, 0, dw, dh);
                    setGpuDisplayVisible(false);
                    rendered = true;
                    drewDisplayDirect = true;
                  }
                } finally {
                  bitmap.close();
                }
              }
            }
            if (!rendered && frame) {
              const bitmaps = engine.renderSlotsToImageBitmap([slotIdx], [{ vmin, vmax }], c.logScale);
              if (bitmaps && bitmaps[0]) {
                try {
                  const offCtx = mainOffscreenRef.current.getContext("2d");
                  if (offCtx) {
                    offCtx.drawImage(bitmaps[0], 0, 0);
                    rendered = true;
                  }
                } finally {
                  bitmaps[0].close();
                }
              }
            }
          } catch (err) {
            rendered = false;
            drewDisplayDirect = false;
          }
        }
        if (!rendered) {
          if (!frame && !cpuData) {
            scheduleTick();
            return;
          }
          if (cpuDataAlreadyLogged && cpuData) {
            renderToOffscreenReuse(cpuData, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
          } else if (frame) {
            renderFramePlayback(frame, mainImgDataRef.current.data, lut, vmin, vmax, c.logScale);
            mainOffscreenRef.current.getContext("2d")!.putImageData(mainImgDataRef.current, 0, 0);
          }
        }

        // Draw to display canvas. Apply image_rotation so playback matches the
        // static render path (drawMain); otherwise rotated stacks
        // silently lose their rotation when the user hits Play.
        const canvas = canvasRef.current;
        if (canvas && !drewDisplayDirect) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            if ((nPanels || 1) > 1) {
              drawMain(ctx, mainOffscreenRef.current);
            } else {
              ctx.imageSmoothingEnabled = c.smooth;
              ctx.clearRect(0, 0, c.canvasW, c.canvasH);
              ctx.save();
              ctx.translate(c.panX, c.panY);
              ctx.scale(c.zoom, c.zoom);
              const dw = c.width * c.displayScale, dh = c.height * c.displayScale;
              if (c.imageRotation % 4 !== 0) {
                const cx = c.canvasW / 2 / c.zoom, cy = c.canvasH / 2 / c.zoom;
                ctx.translate(cx, cy);
                ctx.rotate((c.imageRotation * Math.PI) / 2);
                ctx.translate(-dw / 2, -dh / 2);
                ctx.drawImage(mainOffscreenRef.current, 0, 0, dw, dh);
              } else {
                ctx.drawImage(mainOffscreenRef.current, 0, 0, dw, dh);
              }
              ctx.restore();
            }
          }
        }
      }

      // Throttled UI updates for the slider. At the 60 fps cap, keep React
      // comfortably out of the frame loop; the canvas still renders every rAF.
      // liveSliceIdx is per-tick for static paint and throttled for direct
      // WebGPU paint to avoid a competing React render path. Each liveSliceIdx
      // change reruns the frame effect, which alone derives the stats and the
      // profile from the measured frame (moving average included, also when the
      // GPU drew the average).
      if (tickNow - lastUIUpdate > uiUpdateIntervalMs) {
        lastUIUpdate = tickNow;
        if (directRender) setLiveSliceIdx(next);
        setDisplaySliceIdx(next);
        setPlaybackUiSliceIdx(next);
        // Histogram refresh during playback. The non-playback effect path is keyed on
        // frameBytes/frameSeq, which DON'T change during rAF playback, so we drive
        // histogram updates directly here on every second throttled UI update.
        playbackHistogramCounterRef.current = (playbackHistogramCounterRef.current + 1) % 2;
        if (playbackHistogramCounterRef.current === 0) {
          // Refresh the visible histogram from the current resident frame.
          // Independent packed panels use one GPU submission for all visible
          // regions, so playback never allocates panel-sized Float32Array slabs.
          void refreshHistogramRef.current?.(next);
        }
      }
      if (!isRgb && transformActive) {
        const warmDirection = c.reverse ? -1 : 1;
        warmPlaybackDisplayFrame(next + warmDirection, next, frame);
        warmPlaybackDisplayFrame(next + warmDirection * 2, next, frame);
      }

      scheduleTick();
    };

    scheduleTick();
    return () => {
      cancelAnimationFrame(animId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  // The current frame's measured and display copies and its stats. The one place stats are derived, so it is
  // keyed on everything the measured frame depends on: the frame, moving average, difference, compare and alignment.
  React.useEffect(() => {
    // RGB frames ship as H*W*3 float32; gray remains H*W.
    const expectedFloats = isRgb ? width * height * 3 : width * height;
    const parsed = extractFloat32(frameBytes, expectedFloats);
    if (!parsed || parsed.length === 0) return;
    if (isRgb) {
      // Keep color plane for paint; expose Rec. 709 luminance for stats/FFT.
      rgbFrameDataRef.current = parsed;
      sourceFrameDataRef.current = parsed;
      displayFrameRef.current = rgbFrameToLuminance(parsed, width * height);
      measuredFrameRef.current = displayFrameRef.current;
    } else {
      rgbFrameDataRef.current = null;
      sourceFrameDataRef.current = parsed;
      measuredFrameRef.current = measuredFrameForIndex(liveSliceIdx, parsed) ?? parsed;
      displayFrameRef.current = viewFilteredFrame(liveSliceIdx, measuredFrameRef.current) ?? parsed;
    }
    const measured = measuredFrameRef.current;
    gpuUploadRef.current = null;
    if (!showStats) {
      setLocalStats(null);
      setLocalPanelStats(null);
      return;
    }
    // Recompute stats JS-side only while visible. On 4k frames this is a full
    // 16M-float scan, so keep the default hidden state paint-limited.
    const panelCount = Math.max(1, nPanels || 1);
    const total = computeStats(measured);
    setLocalStats(total);
    if (panelCount > 1 && height > 0 && width > 0 && width % panelCount === 0) {
      const panelWidth = width / panelCount;
      const panels: PanelStats[] = [];
      for (const panel of visiblePanelIndices) {
        const slab = new Float32Array(height * panelWidth);
        for (let row = 0; row < height; row++) {
          const sourceOffset = row * width + panel * panelWidth;
          slab.set(measured.subarray(sourceOffset, sourceOffset + panelWidth), row * panelWidth);
        }
        panels.push({ panel, ...computeStats(slab) });
      }
      setLocalPanelStats(panels);
    } else {
      setLocalPanelStats(null);
    }
  }, [frameBytes, frameSeq, nPanels, visiblePanelIndices, width, height, showStats, diffMode, avgWindow, liveSliceIdx, isRgb, frequencyFilterIsActive, frequencyOptions, browserFilterTick, subpixelAlignEnabled, subpixelAlignVersion, compareMode, comparePair, blinkPhase]);

  // Histogram bins are computed on the GPU via `engine.computeHistogramWithRange`
  // when the colormap engine is ready. CPU fallback (computeHistogramFromBytes
  // inside the Histogram component) still runs if WebGPU isn't available.
  // Debounce: 100 ms past the last scrub frame so drag doesn't fire bin scans
  // on every tick. Playback uses the established 2-tick (5 Hz) throttle.
  const histogramTimerRef = React.useRef<number | null>(null);
  const histogramRefreshInFlightRef = React.useRef(false);
  const histogramRefreshPendingIdxRef = React.useRef<number | null>(null);
  const histogramRefreshSerialRef = React.useRef(0);
  const refreshHistogram = React.useCallback(async (idxArg?: number) => {
    if (isRgb) return;
    const renderIdx = clampSlice(idxArg ?? displaySliceIdx);
    if (histogramRefreshInFlightRef.current) {
      histogramRefreshPendingIdxRef.current = renderIdx;
      return;
    }
    histogramRefreshInFlightRef.current = true;
    const serial = ++histogramRefreshSerialRef.current;
    try {
      // Bin the requested frame of the embedded stack directly so the
      // histogram tracks the playing frame (the stack has no
      // native-resolution slots).
      if (!perPanelHistogramEnabled) {
        const offFrame = getOfflineFrame(renderIdx);
        if (offFrame && offFrame.length) {
          const engine = gpuCmapRef.current;
          let bins: number[] | null = null;
          if (
            engine &&
            gpuCmapReadyRef.current &&
            gpuFrameCacheUploadedRef.current.has(renderIdx) &&
            dataMax > dataMin
          ) {
            try {
              bins = await engine.computeHistogramWithRange(renderIdx, dataMin, dataMax, logScale);
            } catch {
              bins = null;  // Histogram component CPU-bins from imageHistogramData below
            }
          }
          setImageDataRange(resolveDisplayBounds(dataMin, dataMax, null, null, logScale));
          setImageHistogramBins(bins ?? histogramBins(offFrame, dataMin, dataMax, logScale));
          setImageHistogramData(null);
          return;
        }
      }
      if (!perPanelHistogramEnabled) {
        const engine = gpuCmapRef.current;
        if (
          engine &&
          gpuCmapReadyRef.current &&
          gpuFrameCacheUploadedRef.current.has(renderIdx) &&
          dataMax > dataMin
        ) {
          let bins: number[] | null = null;
          try {
            bins = await engine.computeHistogramWithRange(renderIdx, dataMin, dataMax, logScale);
          } catch {
            bins = null;
          }
          if (serial === histogramRefreshSerialRef.current && bins) {
            setImageDataRange(resolveDisplayBounds(dataMin, dataMax, null, null, logScale));
            setImageHistogramBins(bins);
            setImageHistogramData(null);
            setPanelHistogramBins([]);
            return;
          }
        }
      }

      const raw = displayFrameRef.current;
      if (!raw || raw.length === 0) return;
      if (perPanelHistogramEnabled) {
        const panelCount = Math.max(1, nPanels || 1);
        const engine = gpuCmapRef.current;
        if (
          engine &&
          gpuCmapReadyRef.current &&
          gpuFrameCacheUploadedRef.current.has(renderIdx) &&
          !frameTransformActive()
        ) {
          const panelW = totalPanelCount > 1
            ? Math.max(1, panelWidthPx || Math.round(width / totalPanelCount))
            : Math.max(1, width);
          const regions = visiblePanelIndices.map(panel => ({
            x: sharedPanelSource ? 0 : panel * panelW,
            y: 0,
            width: panelW,
            height,
          }));
          try {
            const histograms = await engine.computeHistogramRegions(
              renderIdx,
              regions,
              logScale,
            );
            if (
              serial === histogramRefreshSerialRef.current &&
              histograms.length === visiblePanelIndices.length
            ) {
              const nextBins: (number[] | null)[] = Array.from({ length: panelCount }, () => null);
              const nextRanges: { min: number; max: number }[] = Array.from(
                { length: panelCount },
                (_, panel) => panelDataRanges[panel]
                  ?? resolveDisplayBounds(dataMin, dataMax, null, null, logScale),
              );
              visiblePanelIndices.forEach((panel, k) => {
                nextBins[panel] = histograms[k].bins;
                nextRanges[panel] = histograms[k].range;
              });
              setPanelHistogramData(Array.from({ length: panelCount }, () => null));
              setPanelHistogramBins(nextBins);
              setPanelDataRanges(nextRanges);
              setImageHistogramBins(null);
              return;
            }
          } catch {
            // Preserve the existing CPU fallback for browsers without a usable
            // region compute path. Hardware WebGPU stays on the resident path.
          }
        }
        const nextData: (Float32Array | null)[] = Array.from({ length: panelCount }, () => null);
        const nextRanges: { min: number; max: number }[] = Array.from(
          { length: panelCount },
          () => resolveDisplayBounds(dataMin, dataMax, null, null, logScale),
        );
        const nextBins: (number[] | null)[] = Array.from({ length: panelCount }, () => null);
        for (const panel of visiblePanelIndices) {
          const panelData = extractPanelSlice(raw, panel, logScale);
          nextData[panel] = panelData;
          nextRanges[panel] = panelData && panelData.length > 0
            ? findDataRange(panelData)
            : resolveDisplayBounds(dataMin, dataMax, null, null, logScale);
          if (panelData && panelData.length > 0) nextBins[panel] = histogramBins(panelData, nextRanges[panel].min, nextRanges[panel].max);
        }
        setPanelHistogramData(nextData);
        setPanelHistogramBins(nextBins);
        setPanelDataRanges(nextRanges);
        setImageHistogramBins(null);
        return;
      }
      const data = logScale ? applyLogScale(raw) : raw;
      setImageDataRange(resolveDisplayBounds(dataMin, dataMax, null, null, logScale));
      // GPU bins: the colormap engine has the frame data uploaded to slot 0
      // already (via the render effect). Reuse that slot's buffer for a
      // 256-bin compute pass; fall back to CPU bins in the Histogram component
      // when the engine isn't ready or returns null.
      const engine = gpuCmapRef.current;
      let bins: number[] | null = null;
      if (engine && gpuCmapReadyRef.current && dataMax > dataMin) {
        try {
          // Use the requested frame's slot, not a hardcoded 0 (which is whatever
          // the data effect last uploaded, not the playing frame).
          const slot = gpuFrameCacheUploadedRef.current.has(renderIdx) ? renderIdx : 0;
          bins = await engine.computeHistogramWithRange(slot, dataMin, dataMax, logScale);
        } catch {
          bins = null;  // fall through to CPU path
        }
      }
      setImageHistogramBins(bins ?? histogramBins(raw, dataMin, dataMax, logScale));
      setImageHistogramData(data);
    } finally {
      histogramRefreshInFlightRef.current = false;
      const pending = histogramRefreshPendingIdxRef.current;
      histogramRefreshPendingIdxRef.current = null;
      if (pending !== null && pending !== renderIdx) {
        window.setTimeout(() => { void refreshHistogram(pending); }, 0);
      }
    }
  }, [logScale, dataMin, dataMax, perPanelHistogramEnabled, nPanels, nSlices, visiblePanelIndices, extractPanelSlice, displaySliceIdx, isRgb, height, panelDataRanges, panelWidthPx, sharedPanelSource, totalPanelCount, width, diffMode, avgWindow, frequencyFilterIsActive, subpixelAlignEnabled]);
  refreshHistogramRef.current = refreshHistogram;
  React.useEffect(() => {
    if (playing) {
      return;
    }
    playbackHistogramCounterRef.current = 0;
    if (histogramTimerRef.current !== null) {
      window.clearTimeout(histogramTimerRef.current);
    }
    histogramTimerRef.current = window.setTimeout(() => {
      refreshHistogram(displaySliceIdx);
      histogramTimerRef.current = null;
    }, 32);
  }, [frameBytes, frameSeq, playing, displaySliceIdx, refreshHistogram, gpuResidency.stage]);

  // Auto-snap thumbs to percentile-clip values while Auto is on. Fires once at mount
  // (so the slider visually reflects the percentile-clipped contrast the canvas paints
  // when auto_contrast=True), and re-fires when logScale flips (linear vs log percentile
  // give different clip values, so the thumbs must follow). The lastLogScaleRef tracks
  // the previous logScale value so we only re-snap on transitions, not on every render.
  const initialAutoSnappedRef = React.useRef(false);
  const lastLogScaleRef = React.useRef(logScale);
  const lastAutoContrastRef = React.useRef(autoContrast);
  React.useEffect(() => {
    const logScaleChanged = lastLogScaleRef.current !== logScale;
    // Detect Auto toggled false -> true (user re-engages Auto).
    // Re-snap thumbs to auto range whenever Auto turns back on.
    const autoToggledOn = !lastAutoContrastRef.current && autoContrast;
    lastLogScaleRef.current = logScale;
    lastAutoContrastRef.current = autoContrast;
    if (perPanelHistogramEnabled) return;
    if (!autoContrast || !imageHistogramData || imageHistogramData.length === 0) return;
    // Skip initial snap if user already moved thumbs (e.g. loaded from saved state).
    if (!initialAutoSnappedRef.current && (imageVminPct !== 0 || imageVmaxPct !== 100)) {
      initialAutoSnappedRef.current = true;
      return;
    }
    // After first snap, re-snap only on logScale OR Auto-toggle-on transitions.
    if (initialAutoSnappedRef.current && !logScaleChanged && !autoToggledOn) return;
    const { min: autoMin, max: autoMax } = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    const span = autoMax - autoMin;
    if (span <= 0) return;
    const cached = stackAutoRangeApplies() ? (
      cachedAutoDisplayRange(autoVmins, autoVmaxs, sliceIdx, logScale)
      || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, sliceIdx, logScale)
    ) : null;
    const { vmin: pmin, vmax: pmax } = cached ?? percentileClip(imageHistogramData, percentileLow, percentileHigh);
    setImageVminPct(Math.max(0, Math.min(100, ((pmin - autoMin) / span) * 100)));
    setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - autoMin) / span) * 100)));
    initialAutoSnappedRef.current = true;
  }, [autoContrast, imageHistogramData, dataMin, dataMax, traitVmin, traitVmax, autoVmins, autoVmaxs, sliceIdx, percentileLow, percentileHigh, logScale, imageVminPct, imageVmaxPct, perPanelHistogramEnabled]);

  // useEffect (not useLayoutEffect) so the per-panel auto-snap runs AFTER
  // the data effect populates panelHistogramData for the new frame.
  // useLayoutEffect fires BEFORE useEffects → displayFrameRef would be
  // stale and the snap would bail at mount.
  React.useEffect(() => {
    if (!perPanelHistogramEnabled || !autoContrast || panelHistogramData.length === 0) return;
    const stackBounds = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
    if (stackBounds.max <= stackBounds.min) return;
    setPanelStates(prev => {
      const out = prev.map((state, i) => {
        // PER-PANEL auto: percentile-clip THIS panel's own data, then map
        // pct in THIS panel's data range. Mixed-unit stacks (BF/DF counts
        // vs SSB radians) span many orders of magnitude; using stack
        // range squashes tight panels to pct ≈ 0.
        const clip = panelAutoClipPcts(i, state, stackBounds);
        return clip ? { ...state, imageVminPct: clip.imageVminPct, imageVmaxPct: clip.imageVmaxPct } : state;
      });
      return out;
    });
  }, [perPanelHistogramEnabled, autoContrast, panelHistogramData, panelDataRanges, dataMin, dataMax, traitVmin, traitVmax, logScale, percentileLow, percentileHigh]);

  React.useEffect(() => {
    if (!perPanelHistogramEnabled || autoContrast || panelDataRanges.length === 0) return;
    setPanelStates(prev => {
      let changed = false;
      const out = prev.map((state, i) => {
        const storedMin = vminPerPanel[i];
        const storedMax = vmaxPerPanel[i];
        if (storedMin == null && storedMax == null) return state;
        const panelRange = panelDataRanges[i];
        const range = (panelRange && panelRange.max > panelRange.min)
          ? panelRange
          : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
        if (range.max <= range.min) return state;
        const lo = storedMin ?? range.min;
        const hi = Math.max(lo, storedMax ?? range.max);
        const nextMinPct = valueToPct(lo, range.min, range.max, state.imageVminPct);
        const nextMaxPct = valueToPct(hi, range.min, range.max, state.imageVmaxPct);
        if (Math.abs(nextMinPct - state.imageVminPct) < 0.01 && Math.abs(nextMaxPct - state.imageVmaxPct) < 0.01) return state;
        changed = true;
        return { ...state, imageVminPct: nextMinPct, imageVmaxPct: nextMaxPct };
      });
      return changed ? out : prev;
    });
  }, [perPanelHistogramEnabled, autoContrast, panelDataRanges, vminPerPanel, vmaxPerPanel, dataMin, dataMax, traitVmin, traitVmax, logScale]);

  React.useEffect(() => {
    if (!effectiveRoiActive || roiItems.length === 0 || !showRoiResizeHint) return;
    const timer = window.setTimeout(() => setShowRoiResizeHint(false), 6000);
    return () => window.clearTimeout(timer);
  }, [effectiveRoiActive, roiItems.length, showRoiResizeHint]);

  // Data effect: normalize + colormap → reusable offscreen canvas, then draw
  React.useEffect(() => {
    // Invalidate any rAF/mapAsync work from the previous render before every
    // early ownership return (notably the transition into playback).
    const renderSerial = ++gpuRenderSerialRef.current;
    const confirmStaticCanvasPresent = () => {
      if (playing) return;
      const present = () => {
        if (renderSerial !== gpuRenderSerialRef.current) return;
        const canvas = canvasRef.current;
        const offscreen = mainOffscreenRef.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !offscreen || !ctx) return;
        setGpuDisplayVisible(false);
        drawMain(ctx, offscreen, {
          sourcePanelWidth: mainOffscreenSourcePanelWidthRef.current,
        });
      };
      requestAnimationFrame(() => requestAnimationFrame(present));
      window.setTimeout(present, 180);
    };
    const sourceFrameData = sourceFrameDataRef.current ?? displayFrameRef.current;
    if (!sourceFrameData || sourceFrameData.length === 0) return;
    const renderIdx = liveSliceIdx;
    if (!isRgb) measuredFrameRef.current = measuredFrameForIndex(renderIdx, sourceFrameData);
    const transformedFrame = !isRgb
      ? viewFilteredFrame(renderIdx, measuredFrameRef.current, { allowRawOnMiss: false })
      : sourceFrameData;
    if (!isRgb && (browserFilterKnobsOn || frequencyFilterIsActive) && !transformedFrame) return;
    const frameData = transformedFrame ?? sourceFrameData;
    displayFrameRef.current = frameData;
    // Difference and overlay paint their own two-frame map; the colormap pass below would cover it whenever
    // this effect reran after the compare effect.
    if (!isRgb && (compareMode === "difference" || compareMode === "overlay")) {
      paintCompare();
      return;
    }
    if (!mainOffscreenRef.current || !mainImgDataRef.current) return;
    if (
      gpuDisplayVisibleRef.current === true &&
      imageRotation % 4 === 0 &&
      !playing &&
      !isRgb &&
      compareMode === "off"
    ) {
      try {
        if (renderCurrentPanelTransformDirect()) {
          return;
        }
      } catch (err) {
        console.warn("[Show3D] WebGPU transform refresh failed during static paint; using retained 2D canvas", err);
      }
    }
    // True-color RGB: paint on the GPU (paintRgbFrame), applying the moving
    // average across color frames when avg > 1 so an avg change re-denoises the
    // static frame, not just live playback.
    if (isRgb && rgbFrameDataRef.current && rgbFrameDataRef.current.length >= width * height * 3) {
      const rgb = normalizedAverageWindow(avgWindow) > 1
        ? averagedRgbFrameForIndex(liveSliceIdx, rgbFrameDataRef.current)
        : rgbFrameDataRef.current;
      paintRgbFrame(rgb);
      return;
    }
    const gpuPlaybackOwnsCanvas = (
      playing &&
      !frequencyFilterIsActive &&
      !!gpuCmapRef.current &&
      gpuCmapReadyRef.current
    );
    if (gpuPlaybackOwnsCanvas) return;
    // Apply log scale using reusable buffer
    const processed = logScale && logBufferRef.current
      ? applyLogScaleInPlace(frameData, logBufferRef.current)
      : frameData;

    const panelCount = Math.max(1, nPanels || 1);
    const perPanelContrast = panelCount > 1 && !sharedPanelSource && width % panelCount === 0 && height > 0;

    // Compute vmin/vmax (per-panel branch uses GPU multi-slot below)
    let vmin: number, vmax: number;
    const hasTraitRange = traitVmin != null || traitVmax != null;
    if (hasTraitRange) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (autoContrast) {
      const cached = stackAutoRangeApplies() ? (
        cachedAutoDisplayRange(autoVmins, autoVmaxs, renderIdx, logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, renderIdx, logScale)
      ) : null;
      if (cached) {
        ({ vmin, vmax } = cached);
      } else {
        ({ vmin, vmax } = percentileClip(processed, percentileLow, percentileHigh));
      }
    } else {
      // Use the global data range (loaded once at widget mount) rather than
      // re-scanning the frame on every scrub. findDataRange does an O(N) min/max
      // pass which is ~8 ms at 4k - avoidable when the stack-wide bounds already
      // bracket the per-frame range.
      const lo = logScale ? signedLog1p(dataMin) : dataMin;
      const hi = logScale ? signedLog1p(dataMax) : dataMax;
      ({ vmin, vmax } = sliderRange(lo, hi, imageVminPct, imageVmaxPct));
    }

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const mixedPanelCmaps = hasMixedPanelCmaps && panelCount > 1 && !sharedPanelSource && width % panelCount === 0 && height > 0;
    const renderPackedPanelsCpu = (
      offscreen: HTMLCanvasElement | OffscreenCanvas,
      offCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
      sharedAutoRange: { vmin: number; vmax: number } | null,
    ) => {
      offCtx.clearRect(0, 0, offscreen.width, offscreen.height);
      const panelW = Math.max(1, Math.floor(width / panelCount));
      const panelImg = offCtx.createImageData(panelW, height);
      for (const p of visiblePanelIndices) {
        if (p < 0 || p >= panelCount) continue;
        const panelData = extractPanelSlice(frameData, p, logScale);
        if (!panelData) continue;
        const panelDataRange = panelDataRanges[p];
        const panelRange = panelData.length > 0
          ? findDataRange(panelData)
          : ((perPanelHistogramEnabled && panelDataRange && panelDataRange.max > panelDataRange.min)
              ? panelDataRange
              : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale));
        const range = perPanelContrast
          ? resolvePanelRenderRange(p, panelRange, sharedAutoRange, panelData, autoContrast, percentileLow, percentileHigh)
          : { vmin, vmax };
        const panelLut = COLORMAPS[panelCmapFor(p)] || lut;
        applyColormap(panelData, panelImg.data, panelLut, range.vmin, range.vmax);
        offCtx.putImageData(panelImg, p * panelW, 0);
      }
    };

    const canvas = canvasRef.current;
    const offscreen = mainOffscreenRef.current;
    const imgData = mainImgDataRef.current;
    const ctx = canvas?.getContext("2d");
    const offCtx = offscreen?.getContext("2d");
    if (!canvas || !offscreen || !imgData || !ctx || !offCtx) return;
    if (perPanelContrast || mixedPanelCmaps) {
      const sharedAutoRange = autoContrast ? { vmin, vmax } : null;
      renderPackedPanelsCpu(offscreen, offCtx, sharedAutoRange);
    } else {
      renderToOffscreenReuse(processed, lut, vmin, vmax, offscreen, imgData);
    }
    drawMain(ctx, offscreen);
    confirmStaticCanvasPresent();
  }, [frameBytes, frameSeq, width, height, cmap, panelCmapFor, hasMixedPanelCmaps, displayScale, canvasW, canvasH, imageVminPct, imageVmaxPct, logScale, autoContrast, percentileLow, percentileHigh, traitVmin, traitVmax, dataMin, dataMax, autoVmins, autoVmaxs, smooth, imageRotation, nPanels, sharedPanelSource, visiblePanelIndices, perPanelHistogramEnabled, linkContrast, panelStates, panelDataRanges, vminPerPanel, vmaxPerPanel, liveSliceIdx, diffMode, avgWindow, playing, gpuCmapReady, canvasRepaintSignal, isRgb, browserFilterTick, denoiseSigmaLive, displayFilter, spatialBin, browserFilterKnobsOn, frequencyRenderVersion, frequencyFilterIsActive, subpixelAlignEnabled, subpixelAlignVersion, compareMode, comparePair, blinkPhase, getOfflineFrame, nSlices]);

  // Per-panel render: each slot gets its own zoom/pan transform. The
  // panelGapPx gutter between slots keeps the grid fill from clearWithGridBackground.
  const drawMain = (
    ctx: CanvasRenderingContext2D,
    offscreen: HTMLCanvasElement | OffscreenCanvas,
    options: { preserveGpuDisplay?: boolean; sourcePanelWidth?: number } = {},
  ) => {
    const drawSliceIdx = liveSliceIdx;
    const keepDirectGpuVisible =
      gpuCmapReadyRef.current &&
      gpuFrameCacheUploadedRef.current.has(displaySliceIdx) &&
      imageRotation % 4 === 0 &&
      hiddenPanelSet.size === 0 &&
      (
        linkedState.zoom === 1 &&
        linkedState.panX === 0 &&
        linkedState.panY === 0
      );
    const preserveActiveGpuTransform =
      gpuDisplayVisibleRef.current === true &&
      imageRotation % 4 === 0 &&
      !frameTransformActive() &&
      viewportTransformActive();
    if (!keepDirectGpuVisible && !options.preserveGpuDisplay && !preserveActiveGpuTransform) {
      setGpuDisplayVisible(false);
    } else if (preserveActiveGpuTransform) {
    }
    ctx.imageSmoothingEnabled = smooth;
    // Clear entire canvas to the configured inter-panel layer. Slot-level bg
    // fill happens inside the per-panel loop.
    clearWithGridBackground(ctx, canvasW, canvasH);
    const visibleCount = Math.max(1, visiblePanelCount || 1);
    const sourcePanelCount = Math.max(1, nPanels || 1);
    const cols = panelColsForCount(visibleCount);
    const rows = Math.ceil(visibleCount / cols);
    const srcPanelW = options.sourcePanelWidth
      ? Math.max(1, options.sourcePanelWidth)
      : sharedPanelSource
      ? offscreen.width
      : Math.max(1, panelWidthPx || offscreen.width / sourcePanelCount);
    const srcH = offscreen.height;
    const gap = visibleCount > 1 ? (panelGapPx) : 0;
    const outPanelW = (canvasW - gap * (cols - 1)) / cols;
    const outPanelH = (canvasH - gap * (rows - 1)) / rows;
    for (let slot = 0; slot < visibleCount; slot++) {
      const i = visiblePanelIndices[slot] ?? slot;
      const panelState = stateFor(i);
      const col = slot % cols;
      const row = Math.floor(slot / cols);
      const slotX = col * (outPanelW + gap);
      const slotY = row * (outPanelH + gap);
      // Slot background under the image; empty cells in a partial last row
      // keep the grid fill.
      ctx.fillStyle = themeColors.bg;
      ctx.fillRect(slotX, slotY, outPanelW, outPanelH);
      // End-of-stack: when the current frame exceeds this panel's real frame
      // count, blur the (repeated last) frame so the operator sees they are
      // scrubbing past real data.
      const realN = panelRealFrames && panelRealFrames[i];
      const pastEnd = !!(realN && drawSliceIdx >= realN);
      ctx.save();
      ctx.beginPath();
      ctx.rect(slotX, slotY, outPanelW, outPanelH);
      ctx.clip();
      ctx.translate(slotX + panelState.panX, slotY + panelState.panY);
      ctx.scale(panelState.zoom, panelState.zoom);
      const w = outPanelW, h = outPanelH;
      if (flipCols || flipRows) {
        ctx.translate(flipCols ? w : 0, flipRows ? h : 0);
        ctx.scale(flipCols ? -1 : 1, flipRows ? -1 : 1);
      }
      if (imageRotation % 4 !== 0) {
        const cx = w / 2 / panelState.zoom, cy = h / 2 / panelState.zoom;
        ctx.translate(cx, cy);
        ctx.rotate((imageRotation * Math.PI) / 2);
        ctx.translate(-w / 2, -h / 2);
      }
      if (pastEnd) ctx.filter = "blur(4px)";
      const srcX = sharedPanelSource ? 0 : i * srcPanelW;
      ctx.drawImage(offscreen as CanvasImageSource, srcX, 0, srcPanelW, srcH, 0, 0, w, h);
      ctx.restore();
      strokePanelInnerBorder(ctx, slotX, slotY, outPanelW, outPanelH);
    }
  };


  const paintHistogramPreviewGpu = React.useCallback(() => {
    if (isRgb || canvasW <= 0 || canvasH <= 0) return;
    const frameCount = Math.max(1, Math.round(nSlices || 1));
    const drawIdx = ((Math.round(playbackIdxRef.current || liveSliceIdx || 0) % frameCount) + frameCount) % frameCount;
    const rendered = gpuFrameCacheUploadedRef.current.has(drawIdx) && (
      renderGpuTemporalAverageSliceDirect(drawIdx, false)
      || renderGpuCachedSliceDirect(drawIdx, false)
    );
    if (rendered) {
      updatePlaybackLiveControls(drawIdx);
      return;
    }
  }, [
    canvasH,
    canvasW,
    isRgb,
    liveSliceIdx,
    nSlices,
    renderGpuCachedSliceDirect,
    renderGpuTemporalAverageSliceDirect,
    updatePlaybackLiveControls,
  ]);

  const scheduleHistogramPreviewPaint = React.useCallback(() => {
    if (histogramPreviewPaintRafRef.current !== null) return;
    histogramPreviewPaintRafRef.current = window.requestAnimationFrame(() => {
      histogramPreviewPaintRafRef.current = null;
      paintHistogramPreviewGpu();
    });
  }, [paintHistogramPreviewGpu]);

  React.useEffect(() => () => {
    if (histogramPreviewPaintRafRef.current !== null) {
      window.cancelAnimationFrame(histogramPreviewPaintRafRef.current);
      histogramPreviewPaintRafRef.current = null;
    }
  }, []);

  React.useEffect(() => {
    const preview = imageHistogramPreviewPctRef.current;
    if (
      preview &&
      Math.abs(preview[0] - imageVminPct) < 0.01 &&
      Math.abs(preview[1] - imageVmaxPct) < 0.01
    ) {
      imageHistogramPreviewPctRef.current = null;
    }
  }, [imageVminPct, imageVmaxPct]);

  React.useEffect(() => {
    const previews = panelHistogramPreviewPctRef.current;
    if (previews.size === 0) return;
    for (const [panel, preview] of Array.from(previews.entries())) {
      const state = panelStates[panel];
      if (
        state &&
        Math.abs(preview[0] - state.imageVminPct) < 0.01 &&
        Math.abs(preview[1] - state.imageVmaxPct) < 0.01
      ) {
        previews.delete(panel);
      }
    }
  }, [panelStates]);





  // Paints a compare view onto the canvas: blink shows one frame of the pair, difference a signed
  // magenta/green map of B - A, overlay A in magenta and B in green.
  const paintCompare = () => {
    if (compareMode === "off" || isRgb || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    setGpuDisplayVisible(false);  // the WebGPU canvas would cover the 2D compare view
    const frameCount = Math.max(1, nSlices || 1);
    const pair = Array.isArray(comparePair) && comparePair.length === 2 ? comparePair : [0, 1];
    const aIdx = Math.max(0, Math.min(frameCount - 1, Math.round(pair[0] ?? 0)));
    const bIdx = Math.max(0, Math.min(frameCount - 1, Math.round(pair[1] ?? Math.min(1, frameCount - 1))));
    const activeIdx = compareMode === "blink" && blinkPhase ? bIdx : aIdx;
    const frameA = compareFrameFor(aIdx);
    const frameB = compareFrameFor(bIdx);
    const active = activeIdx === aIdx ? frameA : frameB;
    if (!frameA || !frameB || !active) return;
    const panelCount = Math.max(1, nPanels || 1);
    const panelW = sharedPanelSource ? width : Math.max(1, panelWidthPx || Math.floor(width / panelCount) || width);
    const out = document.createElement("canvas");
    out.width = width;
    out.height = height;
    const outCtx = out.getContext("2d");
    if (!outCtx) return;
    if (compareBackground === "dark") {
      ctx.save();
      ctx.fillStyle = "#050505";
      ctx.fillRect(0, 0, canvasW, canvasH);
      ctx.restore();
    }
    const paintNormal = (frame: Float32Array) => {
      const data = logScale ? applyLogScale(frame) : frame;
      const range = percentileClip(data, percentileLow, percentileHigh);
      const img = outCtx.createImageData(width, height);
      const lut = COLORMAPS[panelCmapFor(visiblePanelIndices[0] ?? 0)] || COLORMAPS.plasma;
      renderFramePlayback(data, img.data, lut, range.vmin, range.vmax, false);
      outCtx.putImageData(img, 0, 0);
      drawMain(ctx, out, { sourcePanelWidth: sharedPanelSource ? undefined : panelW });
    };
    if (compareMode === "blink") {
      paintNormal(active);
      return;
    }
    const pixels = outCtx.createImageData(width, height);
    const rgba = pixels.data;
    if (compareMode === "overlay") {
      const aRange = percentileClip(frameA, percentileLow, percentileHigh);
      const bRange = percentileClip(frameB, percentileLow, percentileHigh);
      const aSpan = Math.max(1e-12, aRange.vmax - aRange.vmin);
      const bSpan = Math.max(1e-12, bRange.vmax - bRange.vmin);
      for (let i = 0; i < width * height; i++) {
        const valueA = Math.max(0, Math.min(1, (frameA[i] - aRange.vmin) / aSpan));
        const valueB = Math.max(0, Math.min(1, (frameB[i] - bRange.vmin) / bSpan));
        rgba[4 * i] = Math.round(255 * valueA);
        rgba[4 * i + 1] = Math.round(255 * valueB);
        rgba[4 * i + 2] = Math.round(255 * valueA);
        rgba[4 * i + 3] = 255;
      }
    } else {
      let maxAbsDifference = 0;
      for (let i = 0; i < width * height; i++) maxAbsDifference = Math.max(maxAbsDifference, Math.abs(frameB[i] - frameA[i]));
      const scale = maxAbsDifference > 0 ? 1 / maxAbsDifference : 1;
      const magentaPositive = String(diffCmap || "magenta-green").toLowerCase() === "magenta-green";
      for (let i = 0; i < width * height; i++) {
        const difference = Math.max(-1, Math.min(1, (frameB[i] - frameA[i]) * scale));
        const intensity = Math.round(255 * Math.abs(difference));
        const positive = difference >= 0;
        const magenta = positive === magentaPositive;
        rgba[4 * i] = magenta ? intensity : 0;
        rgba[4 * i + 1] = magenta ? 0 : intensity;
        rgba[4 * i + 2] = magenta ? intensity : 0;
        rgba[4 * i + 3] = 255;
      }
    }
    outCtx.putImageData(pixels, 0, 0);
    drawMain(ctx, out, { sourcePanelWidth: sharedPanelSource ? undefined : panelW });
  };
  React.useEffect(() => {
    paintCompare();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compareMode, comparePair, blinkPhase, blinkFps, compareBackground, diffCmap, isRgb, canvasW, canvasH, width, height, nSlices, nPanels, panelWidthPx, sharedPanelSource, displaySliceIdx, frameBytes, frameSeq, avgWindow, subpixelAlignEnabled, subpixelAlignVersion, cmap, panelCmaps, percentileLow, percentileHigh, logScale, visiblePanelIndices, canvasRepaintSignal]);

  const ensureFullSizeMainOffscreen = React.useCallback((): boolean => {
    if (width <= 0 || height <= 0) return false;
    const current = mainOffscreenRef.current;
    if (
      current &&
      current.width === width &&
      current.height === height &&
      mainImgDataRef.current &&
      mainOffscreenSourcePanelWidthRef.current === undefined
    ) {
      return true;
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    mainOffscreenRef.current = canvas;
    mainOffscreenSourcePanelWidthRef.current = undefined;
    mainImgDataRef.current = canvas.getContext("2d")!.createImageData(width, height);
    return true;
  }, [width, height]);

  const paintRgbFrame = (rgb: Float32Array): boolean => {
    if (!ensureFullSizeMainOffscreen() || !mainOffscreenRef.current) return false;
    // WebGPU passthrough: pack the RGB channels on the GPU and blit, keeping the
    // per-pixel loop off the UI thread. Falls back to the CPU loop when the
    // engine is unavailable or the frame exceeds the storage-buffer limit.
    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current) {
      const bitmap = engine.renderRgbToImageBitmap(rgb, width, height);
      if (bitmap) {
        try {
          const octx = mainOffscreenRef.current.getContext("2d");
          if (octx) octx.drawImage(bitmap, 0, 0);
        } finally {
          bitmap.close();
        }
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (ctx) drawMain(ctx, mainOffscreenRef.current);
        return true;
      }
    }
    if (!mainImgDataRef.current) return false;
    const px = mainImgDataRef.current.data;
    const pixelTotal = Math.min(width * height, Math.floor(rgb.length / 3));
    for (let k = 0; k < pixelTotal; k++) {
      px[4 * k] = Math.max(0, Math.min(255, Math.round(rgb[3 * k] * 255)));
      px[4 * k + 1] = Math.max(0, Math.min(255, Math.round(rgb[3 * k + 1] * 255)));
      px[4 * k + 2] = Math.max(0, Math.min(255, Math.round(rgb[3 * k + 2] * 255)));
      px[4 * k + 3] = 255;
    }
    mainOffscreenRef.current.getContext("2d")!.putImageData(mainImgDataRef.current, 0, 0);
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (ctx) drawMain(ctx, mainOffscreenRef.current);
    return true;
  };


  const commitLivePanelTransforms = () => {
    if (transformStateCommitTimerRef.current !== null) {
      window.clearTimeout(transformStateCommitTimerRef.current);
      transformStateCommitTimerRef.current = null;
    }
    const nextLinked = linkedStateLiveRef.current;
    const nextPanels = panelStatesLiveRef.current;
    setViewState({ linked_state: { ...nextLinked }, panel_states: nextPanels.map(v => ({ ...v })) });
    setLinkedState(prev => (
      prev.zoom === nextLinked.zoom &&
      prev.panX === nextLinked.panX &&
      prev.panY === nextLinked.panY
        ? prev
        : { ...prev, zoom: nextLinked.zoom, panX: nextLinked.panX, panY: nextLinked.panY }
    ));
    setPanelStates(prev => {
      const panelCount = Math.max(prev.length, nextPanels.length);
      let changed = prev.length !== panelCount;
      const merged = Array.from({ length: panelCount }, (_, i) => {
        const base = prev[i] || initialState;
        const live = nextPanels[i] || base;
        if (
          base.zoom !== live.zoom ||
          base.panX !== live.panX ||
          base.panY !== live.panY ||
          base.imageVminPct !== live.imageVminPct ||
          base.imageVmaxPct !== live.imageVmaxPct
        ) {
          changed = true;
        }
        return { ...base, ...live };
      });
      return changed ? merged : prev;
    });
    if (!playing) renderCurrentPanelTransformDirect();
  };

  const scheduleTransformStateCommit = (delayMs = 120) => {
    if (transformStateCommitTimerRef.current !== null) {
      window.clearTimeout(transformStateCommitTimerRef.current);
    }
    transformStateCommitTimerRef.current = window.setTimeout(commitLivePanelTransforms, delayMs);
  };

  const renderCurrentPanelTransformDirect = (): boolean => {
    // Interactive zoom/pan owns the visible canvas. Invalidate any pending
    // static GPU->2D blit scheduled by the data effect so it cannot hide the
    // WebGPU canvas after this transform frame presents.
    gpuRenderSerialRef.current++;
    const frameCount = Math.max(1, nSlices || 1);
    const idx = ((Math.round(playbackIdxRef.current) % frameCount) + frameCount) % frameCount;
    if (renderGpuPackedPanelTransformSlice(idx, false)) return true;
    const offscreen = mainOffscreenRef.current;
    const ctx = canvasRef.current?.getContext("2d");
    if (!offscreen || !ctx) return false;
    gpuDisplayVisibleRef.current = false;
    setGpuDisplayVisible(false);
    drawMain(ctx, offscreen);
    return true;
  };

  // Display controls repaint the current resident GPU slot immediately. They
  // never resend or re-upload the embedded stack.
  React.useEffect(() => {
    if (gpuResidency.stage !== "ready" || !gpuCmapReadyRef.current) {
      return;
    }
    const frameCount = Math.max(1, nSlices || 1);
    const idx = ((Math.round(playbackIdxRef.current) % frameCount) + frameCount) % frameCount;
    if (!gpuFrameCacheUploadedRef.current.has(idx)) {
      return;
    }
    const rendered = renderGpuTemporalAverageSliceDirect(idx, false)
      || renderGpuCachedSliceDirect(idx, false);
    if (!rendered) {
      return;
    }
    updatePlaybackLiveControls(idx);
  // renderGpuCachedSliceDirect intentionally reads the live refs updated by
  // these controls; listing the style inputs here is the repaint contract.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    gpuResidency.stage,
    gpuResidency.ready,
    gpuCmapReady,
    cmap,
    panelCmaps,
    logScale,
    smooth,
    imageVminPct,
    imageVmaxPct,
    autoContrast,
    percentileLow,
    percentileHigh,
    traitVmin,
    traitVmax,
    linkContrast,
    panelStates,
    panelDataRanges,
    vminPerPanel,
    vmaxPerPanel,
  ]);


  const scheduleTransformRender = (): boolean => {
    if (transformRenderRafRef.current !== null) return true;
    transformRenderRafRef.current = window.requestAnimationFrame(() => {
      transformRenderRafRef.current = null;
      renderCurrentPanelTransformDirect();
    });
    return true;
  };

  React.useEffect(() => () => {
    if (transformRenderRafRef.current !== null) {
      window.cancelAnimationFrame(transformRenderRafRef.current);
      transformRenderRafRef.current = null;
    }
    if (transformStateCommitTimerRef.current !== null) {
      window.clearTimeout(transformStateCommitTimerRef.current);
      transformStateCommitTimerRef.current = null;
    }
  }, []);

  React.useLayoutEffect(() => {
    if (!mainOffscreenRef.current || !canvasRef.current) return;
    const gpuPlaybackOwnsCanvas = (
      playing &&
      !!gpuCmapRef.current &&
      gpuCmapReadyRef.current
    );
    if (gpuPlaybackOwnsCanvas) return;
    const viewTransformActive = viewportTransformActive();
    const preserveGpuDisplay = gpuDisplayVisibleRef.current === true && imageRotation % 4 === 0 && !viewTransformActive;
    if (preserveGpuDisplay) {
      try {
        if (renderCurrentPanelTransformDirect()) return;
      } catch (err) {
        console.warn("[Show3D] WebGPU transform repaint failed; using retained 2D canvas", err);
      }
    }
    const ctx = canvasRef.current.getContext("2d");
    if (ctx) drawMain(ctx, mainOffscreenRef.current, {
      preserveGpuDisplay,
      sourcePanelWidth: mainOffscreenSourcePanelWidthRef.current,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [smooth, canvasW, canvasH, nPanels, visiblePanelIndices, maxCols, imageRotation, flipRows, flipCols, panelStates, linkedState, linkPanels, themeColors.bg, interPanelGapColor, panelInnerBorderColor, panelInnerBorderPx, panelRealFrames, panelTitles, showPanelTitles, panelGapPx, panelTitleFontSize, panelWidthPx, sharedPanelSource, sliceIdx, displaySliceIdx, liveSliceIdx, playing, nSlices, canvasRepaintSignal, viewportTransformActive]);

  // A presented WebGPU texture is not a durable cache. Re-present the current
  // cached frame when it owns the live display; otherwise re-blit the retained
  // 2D offscreen. Neither path changes the frame index or playback state.
  React.useEffect(() => {
    if (canvasRepaintSignal === 0) return;
    let restoredGpu = false;
    const gpuPlaybackOwnsCanvas = (
      playing
      && !!gpuCmapRef.current
      && gpuCmapReadyRef.current
    );
    // The playback loop owns and refreshes the direct GPU canvas. Hiding it
    // here exposes a stale 2D offscreen until the user presses Play again.
    if (gpuPlaybackOwnsCanvas) {
      return;
    }
    if (gpuDisplayVisibleRef.current && imageRotation % 4 === 0) {
      try {
        restoredGpu = renderCurrentPanelTransformDirect();
      } catch (err) {
        console.warn("[Show3D] Foreground WebGPU re-present failed; using the retained 2D frame", err);
      }
    }
    if (!restoredGpu) {
      setGpuDisplayVisible(false);
      const ctx = canvasRef.current?.getContext("2d");
      if (ctx && mainOffscreenRef.current) drawMain(ctx, mainOffscreenRef.current);
    }
    // The foreground signal is the intentional invalidation boundary. The
    // render helpers read the latest state through refs/playRef.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canvasRepaintSignal]);

  // Render overlay (ROI only) - HiDPI aware
  React.useEffect(() => {
    if (!overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);
    // Match the main image's rotation so ROIs / profile sit on the right pixels.
    // Image draw applies `translate(panX,panY) → scale(zoom) → rotate(around cx)`,
    // so the rotation pivot in screen pixels is (canvasW/2+panX, canvasH/2+panY).
    // Overlay must use the SAME screen-space pivot - earlier bug used (canvasW/2,
    // canvasH/2) without pan offset, drifting ROIs when user panned + rotated.
    if (imageRotation % 4 !== 0) {
      const cx = canvasW / 2 + panX;
      const cy = canvasH / 2 + panY;
      ctx.translate(cx, cy);
      ctx.rotate((imageRotation * Math.PI) / 2);
      ctx.translate(-cx, -cy);
    }
    for (const panel of visiblePanelIndices) {
      const overlaySpecs = panelOverlays?.[panel] || [];
      if (!overlaySpecs.length) continue;
      const geom = getPanelGeometry(panel);
      if (!geom) continue;
      ctx.save();
      ctx.beginPath();
      ctx.rect(geom.slotX, geom.slotY, geom.slotW, geom.slotH);
      ctx.clip();
      const toScreenX = (col: number) => geom.slotX + geom.state.panX + col * geom.scaleX * geom.state.zoom;
      const toScreenY = (row: number) => geom.slotY + geom.state.panY + row * geom.scaleY * geom.state.zoom;
      drawPanelOverlays(ctx, overlaySpecs, toScreenX, toScreenY, sourcePanelWidth, sourcePanelHeight);
      if (overlaySelection?.panel === panel) {
        drawPanelOverlaySelection(ctx, overlaySpecs[overlaySelection.overlay], toScreenX, toScreenY, sourcePanelWidth, sourcePanelHeight);
      }
      ctx.restore();
    }

    if (effectiveRoiActive && roiItems.length > 0) {
      const highlightedRois = roiItems.filter(roiItem => roiItem.highlight);
      if (highlightedRois.length > 0) {
        ctx.save();
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(0, 0, canvasW, canvasH);
        ctx.globalCompositeOperation = "destination-out";
        for (const roi of highlightedRois) {
          const screenX = roi.col * displayScale * zoom + panX;
          const screenY = roi.row * displayScale * zoom + panY;
          const screenRadius = roi.radius * displayScale * zoom;
          const shape = roi.shape || "circle";
          ctx.fillStyle = "rgba(0,0,0,1)";
          if (shape === "circle") {
            ctx.beginPath(); ctx.arc(screenX, screenY, screenRadius, 0, Math.PI * 2); ctx.fill();
          } else if (shape === "square") {
            ctx.fillRect(screenX - screenRadius, screenY - screenRadius, screenRadius * 2, screenRadius * 2);
          } else if (shape === "rectangle") {
            const screenWidth = roi.width * displayScale * zoom;
            const screenHeight = roi.height * displayScale * zoom;
            ctx.fillRect(screenX - screenWidth / 2, screenY - screenHeight / 2, screenWidth, screenHeight);
          } else if (shape === "annular") {
            ctx.beginPath(); ctx.arc(screenX, screenY, screenRadius, 0, Math.PI * 2); ctx.fill();
            ctx.globalCompositeOperation = "source-over";
            ctx.fillStyle = "rgba(0,0,0,0.6)";
            const screenRadiusInner = roi.radius_inner * displayScale * zoom;
            ctx.beginPath(); ctx.arc(screenX, screenY, screenRadiusInner, 0, Math.PI * 2); ctx.fill();
            ctx.globalCompositeOperation = "destination-out";
          }
        }
        ctx.restore();
      }

      for (let roiIdx = 0; roiIdx < roiItems.length; roiIdx++) {
        const roi = roiItems[roiIdx];
        const isSelected = roiIdx === roiSelectedIdx;
        const screenX = roi.col * displayScale * zoom + panX;
        const screenY = roi.row * displayScale * zoom + panY;
        const screenRadius = roi.radius * displayScale * zoom;
        const screenWidth = roi.width * displayScale * zoom;
        const screenHeight = roi.height * displayScale * zoom;
        const screenRadiusInner = roi.radius_inner * displayScale * zoom;
        const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
        ctx.lineWidth = roi.line_width || 2;
        const color = roi.color || ROI_COLORS[roiIdx % ROI_COLORS.length];
        drawROI(ctx, screenX, screenY, shape, screenRadius, screenWidth, screenHeight, color, color, isSelected && isDraggingROI, screenRadiusInner);
        if (isSelected) {
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1;
          if (shape === "circle" || shape === "annular") {
            ctx.beginPath(); ctx.arc(screenX, screenY, screenRadius + 3, 0, Math.PI * 2); ctx.stroke();
          } else if (shape === "square") {
            ctx.strokeRect(screenX - screenRadius - 3, screenY - screenRadius - 3, (screenRadius + 3) * 2, (screenRadius + 3) * 2);
          } else if (shape === "rectangle") {
            ctx.strokeRect(screenX - screenWidth / 2 - 3, screenY - screenHeight / 2 - 3, screenWidth + 6, screenHeight + 6);
          }
          ctx.setLineDash([]);
        }
      }
    }

    // Line profile overlay. Use the same slot, clip, zoom, pan, and rotation
    // transform as drawMain so profiles stay attached to their panel.
    if (profileActive && profilePoints.length > 0) {
      const ownerPanel = singlePanelPageProfile
        ? activePageStart
        : Math.max(0, Math.min(totalPanelCount - 1, profilePanelIdx));
      const geom = getPanelGeometry(ownerPanel);
      if (geom) {
        const profileLocalCol = (col: number) => singlePanelPageProfile
          ? col
          : panelLocalCol(col, ownerPanel);
        const toPanelX = (col: number) => profileLocalCol(col) * geom.scaleX;
        const toPanelY = (row: number) => row * geom.scaleY;
        const inverseZoom = 1 / Math.max(1, geom.state.zoom);
        const markerR = 8 * inverseZoom;
        const profileColor = "#00e5ff";
        const profileHalo = "rgba(0, 0, 0, 0.88)";
        const drawEndpoint = (x: number, y: number, label: string) => {
          ctx.fillStyle = profileHalo;
          ctx.beginPath();
          ctx.arc(x, y, markerR + 2 * inverseZoom, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = profileColor;
          ctx.beginPath();
          ctx.arc(x, y, markerR, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#001018";
          ctx.font = `700 ${10 * inverseZoom}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(label, x, y + 0.5 * inverseZoom);
        };
        ctx.save();
        ctx.beginPath();
        ctx.rect(geom.slotX, geom.slotY, geom.slotW, geom.slotH);
        ctx.clip();
        ctx.translate(geom.slotX + geom.state.panX, geom.slotY + geom.state.panY);
        ctx.scale(geom.state.zoom, geom.state.zoom);
        if (imageRotation % 4 !== 0) {
          const cx = geom.slotW / 2 / geom.state.zoom;
          const cy = geom.slotH / 2 / geom.state.zoom;
          ctx.translate(cx, cy);
          ctx.rotate((imageRotation * Math.PI) / 2);
          ctx.translate(-geom.slotW / 2, -geom.slotH / 2);
        }

        const ax = toPanelX(profilePoints[0].col);
        const ay = toPanelY(profilePoints[0].row);

        if (profilePoints.length === 2) {
          const bx = toPanelX(profilePoints[1].col);
          const by = toPanelY(profilePoints[1].row);

          // Draw band when profile width > 1
          if (profileWidth > 1) {
            const dc = profilePoints[1].col - profilePoints[0].col;
            const dr = profilePoints[1].row - profilePoints[0].row;
            const lineLen = Math.sqrt(dc * dc + dr * dr);
            if (lineLen > 0) {
              const halfW = (profileWidth - 1) / 2;
              const perpR = -dc / lineLen * halfW;
              const perpC = dr / lineLen * halfW;
              ctx.fillStyle = "rgba(0, 229, 255, 0.22)";
              ctx.strokeStyle = "rgba(0, 0, 0, 0.72)";
              ctx.lineWidth = 2 * inverseZoom;
              ctx.beginPath();
              ctx.moveTo(toPanelX(profilePoints[0].col + perpC), toPanelY(profilePoints[0].row + perpR));
              ctx.lineTo(toPanelX(profilePoints[1].col + perpC), toPanelY(profilePoints[1].row + perpR));
              ctx.lineTo(toPanelX(profilePoints[1].col - perpC), toPanelY(profilePoints[1].row - perpR));
              ctx.lineTo(toPanelX(profilePoints[0].col - perpC), toPanelY(profilePoints[0].row - perpR));
              ctx.closePath();
              ctx.fill();
              ctx.stroke();
            }
          }

          ctx.strokeStyle = profileHalo;
          ctx.lineWidth = 6 * inverseZoom;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();

          ctx.strokeStyle = profileColor;
          ctx.lineWidth = 2.5 * inverseZoom;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();

          drawEndpoint(ax, ay, "1");
          drawEndpoint(bx, by, "2");
        } else {
          drawEndpoint(ax, ay, "1");
        }
        ctx.restore();
      }
    }
  }, [activePageStart, effectiveRoiActive, roiItems, roiSelectedIdx, isDraggingROI, canvasW, canvasH, displayScale, zoom, panX, panY, themeColors, profileActive, profilePoints, profileWidth, profilePanelIdx, nPanels, panelTitles, imageRotation, width, height, panelStates, linkedState, linkPanels, panelGapPx, sourcePanelWidth, sourcePanelHeight, sharedPanelSource, singlePanelPageProfile, totalPanelCount, canvasRepaintSignal, panelOverlays, overlaySelection, visiblePanelIndices]);

  // Lens inset rendering
  React.useEffect(() => {
    const lensCanvas = lensCanvasRef.current;
    if (lensCanvas) {
      const lensCtx = lensCanvas.getContext("2d");
      if (lensCtx) lensCtx.clearRect(0, 0, lensCanvas.width, lensCanvas.height);
    }
    if (!showLens || !lensPos || !displayFrameRef.current) return;
    if ((nPanels || 1) > 1) return;  // Lens disabled in multi-panel mode
    if (!lensCanvas) return;
    const ctx = lensCanvas.getContext("2d");
    if (!ctx) return;

    const raw = displayFrameRef.current;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const processed = logScale ? applyLogScale(raw) : raw;
    let vmin: number, vmax: number;
    if (traitVmin != null || traitVmax != null) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (autoContrast) {
      const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
        || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
      ({ vmin, vmax } = cached ?? percentileClip(processed, percentileLow, percentileHigh));
    } else if (imageDataRange.min !== imageDataRange.max) {
      ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
    } else {
      const dataRange = findDataRange(processed);
      vmin = dataRange.min; vmax = dataRange.max;
    }

    const regionSize = Math.max(4, Math.round(lensDisplaySize / lensMag));
    const lensSize = lensDisplaySize;
    const margin = 12;
    const half = Math.floor(regionSize / 2);
    const rowStart = lensPos.row - half;
    const colStart = lensPos.col - half;

    const regionCanvas = document.createElement("canvas");
    regionCanvas.width = regionSize;
    regionCanvas.height = regionSize;
    const regionCtx = regionCanvas.getContext("2d");
    if (!regionCtx) return;
    const imgData = regionCtx.createImageData(regionSize, regionSize);
    const range = vmax - vmin || 1;
    for (let rowOffset = 0; rowOffset < regionSize; rowOffset++) {
      for (let colOffset = 0; colOffset < regionSize; colOffset++) {
        const sourceRow = rowStart + rowOffset;
        const sourceCol = colStart + colOffset;
        const idx = (rowOffset * regionSize + colOffset) * 4;
        if (sourceRow < 0 || sourceRow >= height || sourceCol < 0 || sourceCol >= width) {
          imgData.data[idx] = 0; imgData.data[idx + 1] = 0; imgData.data[idx + 2] = 0; imgData.data[idx + 3] = 255;
        } else {
          const val = processed[sourceRow * width + sourceCol];
          const normalized = Math.max(0, Math.min(1, (val - vmin) / range));
          const lutIndex = Math.round(normalized * 255);
          imgData.data[idx] = lut[lutIndex * 3]; imgData.data[idx + 1] = lut[lutIndex * 3 + 1]; imgData.data[idx + 2] = lut[lutIndex * 3 + 2]; imgData.data[idx + 3] = 255;
        }
      }
    }
    regionCtx.putImageData(imgData, 0, 0);

    ctx.save();
    ctx.scale(DPR, DPR);
    // Clamp anchor + default position to canvas bounds. Without clamp a small canvas
    // (e.g. multi-panel 100 px tall) puts the inset off-screen (-60 px) because
    // default lensY = canvasH - lensSize - margin - 20 goes negative.
    const cssH = canvasH;
    const cssW = canvasW;
    const rawLensX = lensAnchor ? lensAnchor.x : margin;
    const rawLensY = lensAnchor ? lensAnchor.y : cssH - lensSize - margin - 20;
    const lensX = Math.max(0, Math.min(cssW - lensSize, rawLensX));
    const lensY = Math.max(0, Math.min(cssH - lensSize, rawLensY));
    ctx.imageSmoothingEnabled = smooth;
    ctx.drawImage(regionCanvas, lensX, lensY, lensSize, lensSize);
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(lensX, lensY, lensSize, lensSize);
    const cx = lensX + lensSize / 2;
    const cy = lensY + lensSize / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "10px monospace";
    ctx.fillText(`${lensMag}×`, lensX + 4, lensY + lensSize - 4);
    ctx.restore();
  }, [showLens, lensPos, cmap, logScale, autoContrast, imageDataRange, imageVminPct, imageVmaxPct, dataMin, dataMax, traitVmin, traitVmax, width, height, canvasW, canvasH, themeColors, lensMag, lensDisplaySize, lensAnchor, percentileLow, percentileHigh, frameBytes, sliceIdx, displaySliceIdx, nPanels, canvasRepaintSignal]);

  // ROI sparkline plot
  React.useEffect(() => {
    const canvas = roiPlotCanvasRef.current;
    if (!canvas || !showRoiPlot || !effectiveRoiActive) return;
    const plotW = canvasW;
    const plotH = 76;
    canvas.width = Math.round(plotW * DPR);
    canvas.height = Math.round(plotH * DPR);
    canvas.style.width = `${plotW}px`;
    canvas.style.height = `${plotH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, plotW, plotH);

    if (!roiPlotData || roiPlotData.byteLength < 4) return;
    const values = extractFloat32(roiPlotData);
    if (!values || values.length === 0) return;
    let min = values[0], max = values[0];
    for (let i = 1; i < values.length; i++) {
      if (values[i] < min) min = values[i];
      if (values[i] > max) max = values[i];
    }
    const range = max - min || 1;
    const padY = 14;
    const drawH = plotH - padY * 2;

    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const denom = Math.max(1, values.length - 1);
    for (let i = 0; i < values.length; i++) {
      const x = (i / denom) * plotW;
      const y = padY + drawH - ((values[i] - min) / range) * drawH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const activeIdx = displaySliceIdx;
    const markerIdx = Math.max(0, Math.min(values.length - 1, activeIdx));
    const markerX = (markerIdx / denom) * plotW;
    ctx.strokeStyle = themeColors.textMuted;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(markerX, padY);
    ctx.lineTo(markerX, padY + drawH);
    ctx.stroke();
    ctx.setLineDash([]);

    if (values.length > 0) {
      const cy = padY + drawH - ((values[markerIdx] - min) / range) * drawH;
      ctx.fillStyle = themeColors.accent;
      ctx.beginPath();
      ctx.arc(markerX, cy, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = themeColors.textMuted;
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText(formatNumber(max), 2, padY - 2);
    ctx.fillText(formatNumber(min), 2, padY + drawH + 10);
  }, [roiPlotData, effectiveRoiActive, showRoiPlot, canvasW, themeColors, sliceIdx, displaySliceIdx, playing, canvasRepaintSignal]);

  // Keep sampled profile data current, but do not reopen the profile UI after
  // the user has turned it off. The line stays cached so toggling Profile back
  // on restores the latest sampled data.
  React.useEffect(() => {
    if (profilePoints.length === 2 && measuredFrameRef.current) {
      updateProfileForActivePage(measuredFrameRef.current, profilePoints[0], profilePoints[1]);
    } else {
      setProfileData(null);
    }
  }, [frameBytes, profilePoints, profileWidth, updateProfileForActivePage]);

  // Render profile sparkline
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasW;
    const cssH = profileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, cssW, cssH);

    if (!profileData || profileData.length < 2) {
      ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = isDark ? "#555" : "#999";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        profilePoints.length === 1
          ? "Choose point 2 on the image"
          : "Click point 1, then point 2, to draw a profile",
        cssW / 2,
        cssH / 2,
      );
      return;
    }

    const padLeft = 40;
    const padRight = 8;
    const padTop = 6;
    const padBottom = 18;
    const plotW = cssW - padLeft - padRight;
    const plotH = cssH - padTop - padBottom;

    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < profileData.length; i++) {
      if (profileData[i] < gMin) gMin = profileData[i];
      if (profileData[i] > gMax) gMax = profileData[i];
    }
    const range = gMax - gMin || 1;

    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < profileData.length; i++) {
      const x = padLeft + (i / (profileData.length - 1)) * plotW;
      const y = padTop + plotH - ((profileData[i] - gMin) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // X-axis: calibrated distance
    let totalDist = profileData.length - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const deltaCol = profilePoints[1].col - profilePoints[0].col;
      const deltaRow = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(deltaCol * deltaCol + deltaRow * deltaRow);
      if (pixelSize > 0) {
        ({ value: totalDist, unit: xUnit } = readableLength(distPx * pixelSize, pixelUnit));
      } else {
        totalDist = distPx;
      }
    }

    const tickY = padTop + plotH;
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    const idealTicks = Math.max(2, Math.floor(plotW / 70));
    const tickStep = roundToNiceValue(totalDist / idealTicks);
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textBaseline = "top";
    const ticks: number[] = [];
    for (let v = 0; v <= totalDist + tickStep * 0.01; v += tickStep) {
      if (v > totalDist * 1.001) break;
      ticks.push(v);
    }
    for (let i = 0; i < ticks.length; i++) {
      const v = ticks[i];
      const frac = totalDist > 0 ? v / totalDist : 0;
      const x = padLeft + frac * plotW;
      ctx.beginPath(); ctx.moveTo(x, tickY); ctx.lineTo(x, tickY + 3); ctx.stroke();
      ctx.textAlign = frac < 0.05 ? "left" : frac > 0.95 ? "right" : "center";
      const valStr = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${valStr} ${xUnit}` : valStr, x, tickY + 4);
    }

    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), padLeft - 3, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), padLeft - 3, padTop + plotH);

    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Save base rendering + layout for hover overlay
    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [profileActive, profileData, profilePoints, pixelSize, pixelUnit, canvasW, themeInfo.theme, themeColors.accent, profileHeight, canvasRepaintSignal]);

  // Profile hover handler - draws crosshair + value readout
  const handleProfileMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    const layout = profileLayoutRef.current;
    if (!canvas || !base || !layout) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit } = layout;
    const range = gMax - gMin || 1;

    ctx.putImageData(base, 0, 0);
    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.strokeStyle = themeInfo.theme === "dark" ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on profile line + value
    if (profileData && profileData.length >= 2) {
      const dataIdx = Math.min(profileData.length - 1, Math.max(0, Math.round(frac * (profileData.length - 1))));
      const val = profileData[dataIdx];
      const y = padTop + plotH - ((val - gMin) / range) * plotH;
      ctx.fillStyle = themeColors.accent;
      ctx.beginPath();
      ctx.arc(cssX, y, 3, 0, Math.PI * 2);
      ctx.fill();

      // Value readout label
      const dist = frac * totalDist;
      const label = `${formatNumber(val)}  @  ${dist.toFixed(1)} ${xUnit}`;
      const isDark = themeInfo.theme === "dark";
      ctx.font = "bold 9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      const textW = ctx.measureText(label).width;
      const labelX = Math.min(cssX + 6, padLeft + plotW - textW - 2);
      const labelY = padTop + 2;
      ctx.fillStyle = isDark ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.8)";
      ctx.fillRect(labelX - 2, labelY - 1, textW + 4, 11);
      ctx.fillStyle = isDark ? "#fff" : "#000";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(label, labelX, labelY);
    }

    ctx.restore();
  };

  const handleProfileMouseLeave = () => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  };

  // Profile height resize
  React.useEffect(() => {
    if (!isResizingProfile) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!profileResizeStart) return;
      const delta = e.clientY - profileResizeStart.y;
      setProfileHeight(Math.max(40, Math.min(300, profileResizeStart.height + delta)));
    };
    const handleMouseUp = () => {
      setIsResizingProfile(false);
      setProfileResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingProfile, profileResizeStart]);

  // Render HiDPI scale bar + zoom indicator + colorbar
  React.useEffect(() => {
    if (!uiRef.current) return;
    const ctx = uiRef.current.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, uiRef.current.width, uiRef.current.height);
    const showImageZoomIndicator = showZoomIndicator === true && panelChromeVisible;
    if (scaleBarVisible || showImageZoomIndicator) {
      const unit = pixelSize > 0 ? pixelUnit : "px";
      const pxSize = pixelSize > 0 ? pixelSize : 1;
      // Each slot draws its own bar from its own zoom so panels at different
      // zoom levels show their own length.
      const visibleCount = Math.max(1, visiblePanelCount || 1);
      const cols = panelColsForCount(visibleCount);
      const rows = Math.ceil(visibleCount / cols);
      const gap = visibleCount > 1 ? (panelGapPx) : 0;
      const cssW = uiRef.current.width / DPR;
      const cssH = uiRef.current.height / DPR;
      const slotW = (cssW - gap * (cols - 1)) / cols;
      const slotH = (cssH - gap * (rows - 1)) / rows;
      ctx.save();
      ctx.scale(DPR, DPR);
      for (let slot = 0; slot < visibleCount; slot++) {
        const i = visiblePanelIndices[slot] ?? slot;
        const panelState = stateFor(i);
        const col = slot % cols;
        const row = Math.floor(slot / cols);
        const slotX = col * (slotW + gap);
        const slotY = row * (slotH + gap);
        if (panelInnerBorderPx > 0) {
          ctx.save();
          ctx.shadowColor = "transparent";
          ctx.shadowBlur = 0;
          ctx.shadowOffsetX = 0;
          ctx.shadowOffsetY = 0;
          ctx.strokeStyle = panelInnerBorderColor;
          ctx.lineWidth = panelInnerBorderPx;
          const inset = panelInnerBorderPx / 2;
          ctx.strokeRect(slotX + inset, slotY + inset, Math.max(0, slotW - panelInnerBorderPx), Math.max(0, slotH - panelInnerBorderPx));
          ctx.restore();
        }
        // Each slot is its own scale bar region: panels at different zoom levels
        // show their own length bar, capped at 25% of the slot so it never overflows.
        const slotRegion = { x: slotX, y: slotY, width: slotW, height: slotH };
        if (scaleBarVisible) {
          drawScaleBarInRegion(ctx, slotRegion, panelState.zoom, panelState.zoom * (slotW / sourcePanelWidth), pxSize, unit, { showZoomIndicator: showImageZoomIndicator, maxBarFraction: 0.25 });
        } else if (showImageZoomIndicator) {
          drawZoomIndicatorInRegion(ctx, slotRegion, panelState.zoom);
        }
      }
      ctx.restore();
    }
    if (showColorbar) {
      const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
      // Colorbar must match what's painted on the image, not the raw data range.
      // When autoContrast is on, the image uses percentileClip(low, high) of the
      // current frame - show that range. Otherwise use slider range over data.
      let vmin: number, vmax: number;
      if (traitVmin != null || traitVmax != null) {
        ({ vmin, vmax } = resolveDisplayRange(
          dataMin,
          dataMax,
          traitVmin,
          traitVmax,
          logScale,
          imageVminPct,
          imageVmaxPct,
        ));
      } else if (autoContrast && imageHistogramData && imageHistogramData.length > 0) {
        const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
          || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
        ({ vmin, vmax } = cached ?? percentileClip(imageHistogramData, percentileLow, percentileHigh));
      } else {
        ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
      }
      ctx.save();
      ctx.scale(DPR, DPR);
      const visibleCount = Math.max(1, visiblePanelCount || 1);
      const cols = panelColsForCount(visibleCount);
      const rows = Math.ceil(visibleCount / cols);
      const gap = visibleCount > 1 ? (panelGapPx) : 0;
      const cssW = uiRef.current.width / DPR;
      const cssH = uiRef.current.height / DPR;
      const slotW = (cssW - gap * (cols - 1)) / cols;
      const slotH = (cssH - gap * (rows - 1)) / rows;
      const perPanelColorbar = visibleCount > 1 && !linkContrast && !sharedPanelSource;
      const currentFrame = displayFrameRef.current;
      const sharedAutoRange = autoContrast ? { vmin, vmax } : null;
      for (let slot = 0; slot < visibleCount; slot++) {
        const panel = visiblePanelIndices[slot] ?? slot;
        let panelVmin = vmin;
        let panelVmax = vmax;
        if (perPanelColorbar) {
          const panelData = currentFrame ? extractPanelSlice(currentFrame, panel, logScale) : null;
          const panelDataRange = panelDataRanges[panel];
          const panelRange = panelData && panelData.length > 0
            ? findDataRange(panelData)
            : ((perPanelHistogramEnabled && panelDataRange && panelDataRange.max > panelDataRange.min)
                ? panelDataRange
                : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale));
          const resolved = resolvePanelRenderRange(panel, panelRange, sharedAutoRange, panelData, autoContrast, percentileLow, percentileHigh);
          panelVmin = resolved.vmin;
          panelVmax = resolved.vmax;
        }
        const col = slot % cols;
        const row = Math.floor(slot / cols);
        const slotX = col * (slotW + gap);
        const slotY = row * (slotH + gap);
        ctx.save();
        ctx.beginPath();
        ctx.rect(slotX, slotY, slotW, slotH);
        ctx.clip();
        ctx.translate(slotX, slotY);
        drawColorbar(ctx, slotW, slotH, lut, panelVmin, panelVmax, logScale);
        ctx.restore();
      }
      ctx.restore();
    }
  }, [pixelSize, pixelUnit, scaleBarVisible, width, sourcePanelWidth, canvasW, canvasH, displayScale, zoom, nPanels, visiblePanelCount, visiblePanelIndices, maxCols, panelStates, linkedState, linkPanels, panelGapPx, showZoomIndicator, panelChromeVisible, showColorbar, cmap, imageDataRange, imageVminPct, imageVmaxPct, logScale, autoContrast, imageHistogramData, autoVmins, autoVmaxs, displaySliceIdx, percentileLow, percentileHigh, dataMin, dataMax, traitVmin, traitVmax, linkContrast, sharedPanelSource, panelDataRanges, vminPerPanel, vmaxPerPanel, canvasRepaintSignal]);

  // Compute FFT magnitude (expensive, async - only re-run on data/GPU changes)
  // Supports ROI-scoped FFT: when ROI is active with a selected ROI, compute
  // FFT of the cropped region instead of the full frame.
  type FftMagnitudeCacheEntry = {
    mag: Float32Array;
    cropDims: { cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null;
    grid: { panelWidth: number; panelHeight: number; cols: number; rows: number; count: number } | null;
    source: string;
    panels: number;
    gridLabel: string | null;
    sizeLabel: string;
  };
  const fftMagnitudeCacheBaseMaxBytes = 256 * 1024 * 1024;
  const fftMagRef = React.useRef<Float32Array | null>(null);
  const fftMagnitudeCacheRef = React.useRef<Map<string, FftMagnitudeCacheEntry>>(new Map());
  const fftActiveCacheKeyRef = React.useRef<string | null>(null);
  const fftDataGenerationRef = React.useRef(0);
  const fftPlaybackComputeInFlightRef = React.useRef(false);
  const fftPlaybackLastComputeAtRef = React.useRef(0);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);

  React.useEffect(() => {
    fftDataGenerationRef.current += 1;
    fftMagnitudeCacheRef.current.clear();
    fftActiveCacheKeyRef.current = null;
    fftMagRef.current = null;
    fftMagCacheRef.current = null;
    fftPanelGridRef.current = null;
    fftCropDimsRef.current = null;
    fftOffscreenRef.current = null;
    fftQualityKeyRef.current = "";
    setFftCropDims(null);
    setFftHistogramData(null);
    setFftQuality(null);
    setFftOffscreenVersion(v => v + 1);
    setFftBackendInfo(prev => ({ ...prev, source: "", ms: null, panels: null, grid: "" }));
  }, [frameSeq, width, height, nSlices, nPanels, sourcePanelWidth, sharedPanelSource]);

  React.useEffect(() => {
    if (!effectiveShowFft) return;
    // FFT is useful context, but it must not own the playback budget. During
    // playback, recompute from the frame that was actually drawn at a bounded
    // cadence; outside playback, update immediately for the settled view.
    const playbackFft = Boolean(playing);
    if (playbackFft) {
      const now = performance.now();
      if (fftPlaybackComputeInFlightRef.current) {
        return;
      }
      if (now - fftPlaybackLastComputeAtRef.current < FFT_PLAYBACK_UPDATE_INTERVAL_MS) {
        return;
      }
      fftPlaybackComputeInFlightRef.current = true;
      fftPlaybackLastComputeAtRef.current = now;
    }
    const fftGeneration = fftDataGenerationRef.current;
    let cancelled = false;
    const doCompute = async () => {
      const fftStartMs = performance.now();
      const fftFrameIdx = clampSlice(playbackFft ? playbackIdxRef.current : liveSliceIdx);
      const currentIdx = playbackFft ? playbackIdxRef.current : liveSliceIdx;
      const panelCount = Math.max(1, nPanels || 1);
      const fftDataHeight = height;
      const fftDataWidth = width;
      const data = rawFrameForIndex(fftFrameIdx, currentIdx, measuredFrameRef.current);
      if (!data) return;
      const multiPanelFft = panelCount > 1 && !roiFftActive;
      const selectedRoi = roiFftActive && roiList && roiSelectedIdx >= 0 && roiSelectedIdx < roiList.length
        ? roiList[roiSelectedIdx]
        : null;
      const roiKey = selectedRoi
        ? JSON.stringify({
          idx: roiSelectedIdx,
          row: Math.round(Number(selectedRoi.row ?? 0) * 100) / 100,
          col: Math.round(Number(selectedRoi.col ?? 0) * 100) / 100,
          radius: Math.round(Number(selectedRoi.radius ?? 0) * 100) / 100,
          radius_inner: Math.round(Number(selectedRoi.radius_inner ?? 0) * 100) / 100,
          width: Math.round(Number(selectedRoi.width ?? 0) * 100) / 100,
          height: Math.round(Number(selectedRoi.height ?? 0) * 100) / 100,
          shape: selectedRoi.shape,
        })
        : "none";
      const fftGridCols = multiPanelFft ? panelColsForCount(Math.max(1, visiblePanelIndices.length || 1)) : 1;
      // frame_seq changes only when set_image replaces the embedded stack.
      const fftCacheKey = [
        `frame=${fftFrameIdx}`,
        `data=${frameSeq || 0}`,
        `dims=${fftDataWidth}x${fftDataHeight}`,
        `panels=${panelCount}`,
        `visible=${visiblePanelIndices.join(",")}`,
        `cols=${fftGridCols}`,
        `sourceW=${sourcePanelWidth}`,
        `overlay=${fftLayoutOverlay ? 1 : 0}`,
        `overlayCap=${fftLayoutOverlay ? FFT_OVERLAY_MAX_SOURCE_SIZE : 0}`,
        `shared=${sharedPanelSource ? 1 : 0}`,
        `roi=${roiFftActive ? roiKey : "none"}`,
        `window=${fftWindow ? 1 : 0}`,
        `transform=${diffMode}:${Math.max(1, Math.round(avgWindow || 1))}`,
      ].join("|");
      const cache = fftMagnitudeCacheRef.current;
      const cached = cache.get(fftCacheKey);
      if (cached) {
        if (cancelled || fftGeneration !== fftDataGenerationRef.current) return;
        cache.delete(fftCacheKey);
        cache.set(fftCacheKey, cached);
        if (fftActiveCacheKeyRef.current === fftCacheKey) {
          return;
        }
        fftActiveCacheKeyRef.current = fftCacheKey;
        fftMagRef.current = cached.mag;
        fftMagCacheRef.current = cached.mag;
        fftPanelGridRef.current = cached.grid;
        fftCropDimsRef.current = cached.cropDims;
        setFftCropDims(cached.cropDims);
        setFftMagVersion(v => v + 1);
        setFftBackendInfo(prev => ({
          ...prev,
          source: `${cached.source}-cache`,
          ms: 0,
          panels: cached.panels,
          grid: cached.gridLabel || "",
        }));
        return;
      }
      const rememberFft = (entry: FftMagnitudeCacheEntry) => {
        cache.set(fftCacheKey, entry);
        const maxEntries = Math.max(2, Math.min(24, nSlices || 12));
        const maxBytes = Math.max(fftMagnitudeCacheBaseMaxBytes, Math.min(1024 * 1024 * 1024, entry.mag.byteLength * 3));
        let totalBytes = Array.from(cache.values()).reduce((total, item) => total + item.mag.byteLength, 0);
        while (cache.size > maxEntries || totalBytes > maxBytes) {
          const oldest = cache.keys().next().value;
          if (oldest === undefined) break;
          const oldestEntry = cache.get(oldest);
          cache.delete(oldest);
          totalBytes -= oldestEntry?.mag.byteLength ?? 0;
        }
      };

      if (multiPanelFft) {
        const panelW = sharedPanelSource
          ? Math.max(1, sourcePanelWidth)
          : Math.max(1, Math.floor(fftDataWidth / panelCount));
        const panelH = fftDataHeight;
        const overlayScale = fftLayoutOverlay
          ? Math.max(1, Math.ceil(Math.max(panelW, panelH) / FFT_OVERLAY_MAX_SOURCE_SIZE))
          : 1;
        const fftSourceW = Math.max(1, Math.ceil(panelW / overlayScale));
        const fftSourceH = Math.max(1, Math.ceil(panelH / overlayScale));
        const fftW = nextPow2(fftSourceW);
        const fftH = nextPow2(fftSourceH);
        const panels: { real: Float32Array; imag: Float32Array }[] = [];
        const fullW = data.length === fftDataHeight * panelW ? panelW : fftDataWidth;
        for (const panel of visiblePanelIndices) {
          const srcPanel = sharedPanelSource ? 0 : panel;
          const colStart = Math.min(Math.max(0, srcPanel * panelW), Math.max(0, fullW - panelW));
          if (data.length < fftDataHeight * fullW || colStart + panelW > fullW) continue;
          const source = new Float32Array(fftSourceW * fftSourceH);
          if (overlayScale > 1) {
            for (let row = 0; row < fftSourceH; row++) {
              const srcRow = Math.min(panelH - 1, row * overlayScale);
              const srcOffset = srcRow * fullW + colStart;
              const dstOffset = row * fftSourceW;
              for (let col = 0; col < fftSourceW; col++) {
                source[dstOffset + col] = data[srcOffset + Math.min(panelW - 1, col * overlayScale)];
              }
            }
          } else {
            for (let row = 0; row < panelH; row++) {
              source.set(data.subarray(row * fullW + colStart, row * fullW + colStart + panelW), row * fftSourceW);
            }
          }
          // Window the real source extent, then pad. Applying the taper to the
          // already-padded grid changes the intended Hann profile.
          if (fftWindow) applyHannWindow2D(source, fftSourceW, fftSourceH);
          const real = new Float32Array(fftW * fftH);
          for (let row = 0; row < fftSourceH; row++) {
            real.set(source.subarray(row * fftSourceW, (row + 1) * fftSourceW), row * fftW);
          }
          panels.push({ real, imag: new Float32Array(real.length) });
        }
        if (panels.length === 0) return;

        let results: { real: Float32Array; imag: Float32Array }[];
        const fftGpu = await ensureFftGpu();
        if (cancelled || fftGeneration !== fftDataGenerationRef.current) return;
        const fftSource = fftGpu.path === "WebGPU" ? "webgpu-batch" : "cpu-batch";
        if (panels.length > 1) {
          results = await fftGpu.fft2DBatch(
            panels.map(({ real, imag }) => ({ real, imag })),
            fftW,
            fftH,
          );
        } else {
          results = [await fftGpu.fft2D(panels[0].real, panels[0].imag, fftW, fftH, false)];
        }
        if (cancelled || fftGeneration !== fftDataGenerationRef.current) return;

        const cols = panelColsForCount(panels.length);
        const rows = Math.ceil(panels.length / cols);
        const gridW = cols * fftW;
        const gridH = rows * fftH;
        const gridMag = new Float32Array(gridW * gridH);
        for (let panel = 0; panel < results.length; panel++) {
          const { real, imag } = results[panel];
          fftshift(real, fftW, fftH);
          fftshift(imag, fftW, fftH);
          const mag = computeMagnitude(real, imag);
          const col = panel % cols;
          const row = Math.floor(panel / cols);
          const dstCol = col * fftW;
          const dstRow = row * fftH;
          for (let fftRow = 0; fftRow < fftH; fftRow++) {
            gridMag.set(mag.subarray(fftRow * fftW, fftRow * fftW + fftW), (dstRow + fftRow) * gridW + dstCol);
          }
        }

        fftMagRef.current = gridMag;
        fftActiveCacheKeyRef.current = fftCacheKey;
        fftMagCacheRef.current = gridMag;
        const gridInfo = { panelWidth: fftW, panelHeight: fftH, cols, rows, count: panels.length };
        const cropDims = { cropWidth: fftSourceW, cropHeight: fftSourceH, fftWidth: gridW, fftHeight: gridH };
        fftPanelGridRef.current = gridInfo;
        fftCropDimsRef.current = cropDims;
        setFftCropDims(cropDims);
        rememberFft({
          mag: gridMag,
          cropDims,
          grid: gridInfo,
          source: fftSource,
          panels: panels.length,
          gridLabel: `${gridW}x${gridH}`,
          sizeLabel: overlayScale > 1 ? `${fftW}x${fftH} overlay/${overlayScale}x` : `${fftW}x${fftH}`,
        });
        setFftMagVersion(v => v + 1);
        const elapsedMs = Number((performance.now() - fftStartMs).toFixed(2));
        setFftBackendInfo(prev => ({
          ...prev,
          source: fftSource,
          ms: elapsedMs,
          panels: panels.length,
          grid: `${gridW}x${gridH}`,
        }));
        return;
      }

      fftPanelGridRef.current = null;
      fftCropDimsRef.current = null;
      let fftW = fftDataWidth;
      let fftH = fftDataHeight;
      let inputData = data;

      // ROI crop: extract bounding box and optionally zero-mask outside radius
      let origCropW = 0, origCropH = 0;
      if (roiFftActive && roiList && roiSelectedIdx >= 0 && roiSelectedIdx < roiList.length) {
        const roi = roiList[roiSelectedIdx];
        const crop = await cropMaskedRegionBrowser(data, fftDataWidth, fftDataHeight, roi);
        if (crop) {
          origCropW = crop.cropW;
          origCropH = crop.cropH;
          // Apply Hann window to crop at native dimensions BEFORE zero-padding
          if (fftWindow) applyHannWindow2D(crop.cropped, crop.cropW, crop.cropH);
          // Pad to next power-of-2 so fft2d doesn't truncate frequency data
          const padW = nextPow2(crop.cropW);
          const padH = nextPow2(crop.cropH);
          const padded = new Float32Array(padW * padH);
          for (let row = 0; row < crop.cropH; row++) {
            for (let col = 0; col < crop.cropW; col++) {
              padded[row * padW + col] = crop.cropped[row * crop.cropW + col];
            }
          }
          inputData = padded;
          fftW = padW;
          fftH = padH;
        }
      }

      // Pre-pad non-power-of-2 full images so fft2d doesn't truncate frequency data
      if (origCropW === 0) {
        if (fftWindow) {
          inputData = data.slice();
          applyHannWindow2D(inputData, fftDataWidth, fftDataHeight);
        }
        const padW = nextPow2(fftW);
        const padH = nextPow2(fftH);
        if (padW !== fftW || padH !== fftH) {
          const padded = new Float32Array(padW * padH);
          for (let row = 0; row < fftH; row++) {
            for (let col = 0; col < fftW; col++) {
              padded[row * padW + col] = inputData[row * fftW + col];
            }
          }
          inputData = padded;
          fftW = padW;
          fftH = padH;
        }
      }

      let real: Float32Array, imag: Float32Array;

      const fftGpu = await ensureFftGpu();
      if (cancelled || fftGeneration !== fftDataGenerationRef.current) return;
      const fftSource = fftGpu.path === "WebGPU" ? "webgpu" : "cpu";
      const result = await fftGpu.fft2D(
        inputData.slice(),
        new Float32Array(inputData.length),
        fftW,
        fftH,
        false,
      );
      real = result.real;
      imag = result.imag;

      if (cancelled || fftGeneration !== fftDataGenerationRef.current) return;
      fftshift(real, fftW, fftH);
      fftshift(imag, fftW, fftH);

      fftMagRef.current = computeMagnitude(real, imag);
      fftActiveCacheKeyRef.current = fftCacheKey;
      fftMagCacheRef.current = fftMagRef.current;
      // Track FFT dimensions when they differ from image dimensions (ROI crop or non-pow2 padding)
      let cropDims: { cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null = null;
      if (origCropW > 0) {
        cropDims = { cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH };
      } else if (fftW !== fftDataWidth || fftH !== fftDataHeight) {
        cropDims = { cropWidth: fftDataWidth, cropHeight: fftDataHeight, fftWidth: fftW, fftHeight: fftH };
      }
      fftCropDimsRef.current = cropDims;
      setFftCropDims(cropDims);
      rememberFft({
        mag: fftMagRef.current,
        cropDims,
        grid: null,
        source: fftSource,
        panels: 1,
        gridLabel: `${fftW}x${fftH}`,
        sizeLabel: `${fftW}x${fftH}`,
      });
      setFftMagVersion(v => v + 1);
      const elapsedMs = Number((performance.now() - fftStartMs).toFixed(2));
      setFftBackendInfo(prev => ({
        ...prev,
        source: fftSource,
        ms: elapsedMs,
        panels: 1,
        grid: `${fftW}x${fftH}`,
      }));
    };

    void doCompute().finally(() => {
      if (playbackFft) fftPlaybackComputeInFlightRef.current = false;
    });

    return () => {
      if (!playbackFft) cancelled = true;
    };
  }, [effectiveShowFft, playing, frameBytes, frameSeq, liveSliceIdx, width, height, roiFftActive, roiList, roiSelectedIdx, fftWindow, nPanels, nSlices, visiblePanelIndices, sourcePanelWidth, sharedPanelSource, maxCols, panelColsForCount, fftLayoutOverlay, extractPanelSlice, ensureFftGpu, diffMode, avgWindow]);

  // Clear FFT measurement when ROI FFT state changes
  React.useEffect(() => { setFftClickInfo(null); }, [roiFftActive, roiSelectedIdx]);

  // Process FFT magnitude → histogram + colormap rendering (cheap, sync)
  React.useEffect(() => {
    const mag = fftMagRef.current;
    if (!effectiveShowFft || !mag) return;

    // Use ref-backed dimensions so the magnitude and its layout metadata remain
    // consistent in the same render tick; React state may lag by one effect.
    const cropDimsForRender = fftCropDimsRef.current;
    const fftW = cropDimsForRender?.fftWidth ?? width;
    const fftH = cropDimsForRender?.fftHeight ?? height;
    const grid = fftPanelGridRef.current;
    if (fftMetricsEnabled) {
      const qualityKey = `${fftMagVersion}:${fftW}x${fftH}:${pixelSize || 0}:${pixelUnit || ""}:${grid ? `${grid.panelWidth}x${grid.panelHeight}x${grid.cols}x${grid.count}` : "single"}`;
      if (fftQualityKeyRef.current !== qualityKey) {
        fftQualityKeyRef.current = qualityKey;
        let nextQuality: FftQualityMetrics | null;
        if (grid) {
          const panelMetrics: Array<FftQualityMetrics | null> = [];
          for (let panel = 0; panel < grid.count; panel++) {
            panelMetrics.push(computeFftQualityMetrics(mag, fftW, fftH, {
              sampling: pixelSize,
              unit: pixelUnit,
              region: {
                x: (panel % grid.cols) * grid.panelWidth,
                y: Math.floor(panel / grid.cols) * grid.panelHeight,
                width: grid.panelWidth,
                height: grid.panelHeight,
              },
            }));
          }
          nextQuality = summarizeFftQualityMetrics(panelMetrics);
        } else {
          nextQuality = computeFftQualityMetrics(mag, fftW, fftH, { sampling: pixelSize, unit: pixelUnit });
        }
        setFftQuality(nextQuality);
      }
    } else if (fftQualityKeyRef.current) {
      fftQualityKeyRef.current = "";
      setFftQuality(null);
    }

    let displayMin: number, displayMax: number;
    let displayData: Float32Array;
    if (fftAuto && grid) {
      // Multi-panel FFTs can differ by orders of magnitude (BF/DF vs SSB).
      // Auto mode should reveal each panel, so normalize every FFT tile before
      // composing the shared canvas. Manual mode below intentionally stays global.
      displayData = new Float32Array(mag.length);
      const panelDisplay = new Float32Array(grid.panelWidth * grid.panelHeight);
      for (let panel = 0; panel < grid.count; panel++) {
        const tileCol = panel % grid.cols;
        const tileRow = Math.floor(panel / grid.cols);
        const tileColStart = tileCol * grid.panelWidth;
        const tileRowStart = tileRow * grid.panelHeight;
        for (let row = 0; row < grid.panelHeight; row++) {
          const srcOffset = (tileRowStart + row) * fftW + tileColStart;
          const dstOffset = row * grid.panelWidth;
          for (let col = 0; col < grid.panelWidth; col++) {
            // FFT magnitudes are extremely heavy-tailed; even in "Lin" UI mode,
            // auto contrast should reveal Bragg/fringe peaks instead of letting
            // the DC/low-frequency pedestal flatten the tile.
            panelDisplay[dstOffset + col] = Math.log1p(Math.max(0, mag[srcOffset + col]));
          }
        }
        const centerCol = Math.floor(grid.panelWidth / 2);
        const centerRow = Math.floor(grid.panelHeight / 2);
        const dcRadius = Math.max(2, Math.round(Math.min(grid.panelWidth, grid.panelHeight) * 0.01));
        const ringRadius = dcRadius + 2;
        let ringSum = 0;
        let ringCount = 0;
        for (let row = Math.max(0, centerRow - ringRadius); row <= Math.min(grid.panelHeight - 1, centerRow + ringRadius); row++) {
          for (let col = Math.max(0, centerCol - ringRadius); col <= Math.min(grid.panelWidth - 1, centerCol + ringRadius); col++) {
            const dist = Math.hypot(col - centerCol, row - centerRow);
            if (dist > dcRadius && dist <= ringRadius) {
              ringSum += panelDisplay[row * grid.panelWidth + col];
              ringCount++;
            }
          }
        }
        const dcFill = ringCount > 0 ? ringSum / ringCount : 0;
        for (let row = Math.max(0, centerRow - dcRadius); row <= Math.min(grid.panelHeight - 1, centerRow + dcRadius); row++) {
          for (let col = Math.max(0, centerCol - dcRadius); col <= Math.min(grid.panelWidth - 1, centerCol + dcRadius); col++) {
            panelDisplay[row * grid.panelWidth + col] = dcFill;
          }
        }
        suppressFftRadialBackgroundInPlace(panelDisplay, grid.panelWidth, grid.panelHeight);

        const range = findDataRange(panelDisplay);
        const clipped = percentileClip(panelDisplay, 5, 99.99);
        const pMin = clipped.vmin < clipped.vmax ? clipped.vmin : range.min;
        const pMax = clipped.vmax > pMin ? clipped.vmax : range.max;
        const denom = pMax > pMin ? pMax - pMin : 1;
        for (let row = 0; row < grid.panelHeight; row++) {
          const dstOffset = (tileRowStart + row) * fftW + tileColStart;
          const srcOffset = row * grid.panelWidth;
          for (let col = 0; col < grid.panelWidth; col++) {
            const normalized = (panelDisplay[srcOffset + col] - pMin) / denom;
            displayData[dstOffset + col] = Math.max(0, Math.min(1, normalized));
          }
        }
      }
      displayMin = 0;
      displayMax = 1;
    } else {
      if (fftAuto) {
        ({ min: displayMin, max: displayMax } = autoEnhanceFFT(mag, fftW, fftH));
      } else {
        ({ min: displayMin, max: displayMax } = findDataRange(mag));
      }
      displayData = fftLogScale ? applyLogScale(mag) : mag;
      if (fftLogScale) {
        displayMin = Math.log1p(displayMin);
        displayMax = Math.log1p(displayMax);
      }
    }

    setFftHistogramData(displayData);
    setFftDataRange({ min: displayMin, max: displayMax });
    setFftStats(computeStats(displayData));

    const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(displayData, fftW, fftH, lut, vmin, vmax);
    if (!offscreen) return;

    fftOffscreenRef.current = offscreen;
    setFftOffscreenVersion(v => v + 1);

    if (fftCanvasRef.current) {
      const ctx = fftCanvasRef.current.getContext("2d");
      if (ctx) {
        drawFftOffscreen(ctx, offscreen);
      }
    }
  }, [effectiveShowFft, fftMagVersion, fftLogScale, fftAuto, fftVminPct, fftVmaxPct, fftColormap, width, height, canvasW, canvasH, fftCropDims, drawFftOffscreen, pixelSize, pixelUnit, fftMetricsEnabled, canvasRepaintSignal]);

  // Redraw cached FFT with zoom/pan/resize before paint. Changing a canvas
  // width/height attribute clears its bitmap, so a normal effect can expose a
  // one-frame blank flash during resize drags.
  React.useLayoutEffect(() => {
    if (!effectiveShowFft || !fftCanvasRef.current || !fftOffscreenRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    drawFftOffscreen(ctx, fftOffscreenRef.current);
  }, [effectiveShowFft, fftOffscreenVersion, fftZoom, fftPanX, fftPanY, canvasW, canvasH, drawFftOffscreen, canvasRepaintSignal]);

  const drawFftInsetLayer = React.useCallback((
    view: { zoom: number; panX: number; panY: number } = fftViewLiveRef.current,
  ) => {
    const canvas = fftInsetLayerRef.current;
    if (!canvas || !effectiveShowFft || !fftLayoutOverlay || !fftOffscreenRef.current) return;
    const offscreen = fftOffscreenRef.current;
    const grid = fftPanelGridRef.current;
    const count = grid ? grid.count : 1;
    const visibleCount = Math.max(1, visiblePanelCount || 1);
    const cols = panelColsForCount(visibleCount);
    const rows = Math.ceil(visibleCount / cols);
    const gap = visibleCount > 1 ? (panelGapPx) : 0;
    const panelW = (canvasW - gap * (cols - 1)) / cols;
    const panelH = (canvasH - gap * (rows - 1)) / rows;
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const srcW = grid ? grid.panelWidth : fftW;
    const srcH = grid ? grid.panelHeight : fftH;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = smooth && (srcW < panelW || srcH < panelH);
    visiblePanelIndices.forEach((_panel, slot) => {
      if (slot >= count) return;
      const panelLeft = (slot % cols) * (panelW + gap);
      const panelTop = Math.floor(slot / cols) * (panelH + gap);
      const insetPad = Math.min(8, Math.max(3, panelW * 0.025));
      const insetMaxW = Math.max(24, panelW - insetPad * 2);
      const insetMaxH = Math.max(20, panelH - insetPad * 2);
      const insetBase = Math.min(insetMaxW, insetMaxH);
      const insetW = Math.max(24, Math.min(insetMaxW, insetBase * resolvedFftOverlaySize));
      const insetH = Math.max(20, Math.min(insetMaxH, insetBase * resolvedFftOverlaySize));
      const topInsetPad = fftOverlayTopInsetPad(insetPad, showPanelTitles, nPanels || 1, panelTitleFontSize);
      const insetX = resolvedFftOverlayPosition.endsWith("right")
        ? panelLeft + panelW - insetW - insetPad
        : panelLeft + insetPad;
      const insetY = resolvedFftOverlayPosition.startsWith("bottom")
        ? panelTop + panelH - insetH - insetPad
        : panelTop + topInsetPad;
      const dstX = fftOverlayDragPreview ? panelLeft + fftOverlayDragPreview.x : insetX;
      const dstY = fftOverlayDragPreview ? panelTop + fftOverlayDragPreview.y : insetY;
      const srcX = grid ? (slot % grid.cols) * grid.panelWidth : 0;
      const srcY = grid ? Math.floor(slot / grid.cols) * grid.panelHeight : 0;
      const insetPanX = !fftUserAdjustedViewRef.current && view.zoom > 1
        ? insetW * (1 - view.zoom) / 2
        : view.panX;
      const insetPanY = !fftUserAdjustedViewRef.current && view.zoom > 1
        ? insetH * (1 - view.zoom) / 2
        : view.panY;
      ctx.save();
      ctx.fillStyle = "#000";
      ctx.fillRect(dstX, dstY, insetW, insetH);
      ctx.beginPath();
      ctx.rect(dstX, dstY, insetW, insetH);
      ctx.clip();
      ctx.translate(dstX + insetPanX, dstY + insetPanY);
      ctx.scale(view.zoom, view.zoom);
      ctx.drawImage(offscreen, srcX, srcY, srcW, srcH, 0, 0, insetW, insetH);
      ctx.restore();
      ctx.strokeStyle = "rgba(255,255,255,0.48)";
      ctx.lineWidth = 1;
      ctx.strokeRect(dstX + 0.5, dstY + 0.5, Math.max(0, insetW - 1), Math.max(0, insetH - 1));
    });
  }, [effectiveShowFft, fftLayoutOverlay, fftCropDims, width, height, visiblePanelCount, visiblePanelIndices, panelColsForCount, panelGapPx, canvasW, canvasH, resolvedFftOverlaySize, resolvedFftOverlayPosition, fftOverlayDragPreview, showPanelTitles, panelTitleFontSize, nPanels, smooth]);

  React.useEffect(() => {
    fftViewDirectRedrawRef.current = () => {
      if (!effectiveShowFft || !fftLayoutOverlay || !fftOffscreenRef.current) return;
      if (fftViewRafRef.current !== null) return;
      fftViewRafRef.current = window.requestAnimationFrame(() => {
        fftViewRafRef.current = null;
        drawFftInsetLayer(fftViewLiveRef.current);
      });
    };
    return () => {
      fftViewDirectRedrawRef.current = null;
    };
  }, [drawFftInsetLayer, effectiveShowFft, fftLayoutOverlay]);

  React.useLayoutEffect(() => {
    if (!effectiveShowFft || !fftLayoutOverlay || !fftOffscreenRef.current) return;
    drawFftInsetLayer();
  }, [effectiveShowFft, fftLayoutOverlay, fftOffscreenVersion, fftZoom, fftPanX, fftPanY, fftCropDims, width, height, drawFftInsetLayer, canvasRepaintSignal]);

  // === Kymograph (space-time) ===
  // A sub-feature of the line profile ("the profile feature created a 2D
  // image ... distance along the line ... time axis"). Requires the profile tool
  // ON with a drawn line and the embedded display stack available.
  const kymoExactStackReady = !!offlineFloatStack && offlineFloatStack.byteLength > 0;
  const kymoQuantizedStackReady = !!offlineStack && offlineStack.byteLength > 0;
  const kymoOfflineStackReady = kymoExactStackReady || kymoQuantizedStackReady;
  const kymographAvailable = ((nPanels || 1) === 1 || singlePanelPageProfile)
    && kymoOfflineStackReady
    && width > 0 && height > 0 && nSlices > 1;
  const canKymograph = kymographAvailable && profileActive && profilePoints.length === 2;
  const kymoReady = canKymograph && showKymograph;

  // Compute the (nFrames, lineLen) image: sample the profile line on every
  // frame. Cold path - fires on line / width / stack change, never per tick.
  React.useEffect(() => {
    if (!kymoReady) { kymoDataRef.current = null; return; }
    const p0 = profilePoints[0], p1 = profilePoints[1];
    const pixelCount = width * height;
    const panelIdx = singlePanelPageProfile
      ? activePageStart
      : Math.max(0, Math.min(totalPanelCount - 1, profilePanelIdx));
    const colOffset = singlePanelPageProfile ? panelGlobalColOffset(panelIdx) : 0;
    const row0 = p0.row, col0 = p0.col + colOffset;
    const row1 = p1.row, col1 = p1.col + colOffset;
    let cancelled = false;

    const publish = (kymo: Float32Array, lineLen: number) => {
      if (cancelled) return;
      kymoDataRef.current = { data: kymo, lineLen, nFrames: nSlices };
      setKymoVersion(v => v + 1);
    };

    if (kymoExactStackReady && offlineFloatStack) {
      const sampleFrame = (frameIdx: number): Promise<Float32Array> => {
        const frame = float32FrameFromDataView(offlineFloatStack, frameIdx, pixelCount, false);
        return frame
          ? sampleLineProfileBrowser(frame, width, height, row0, col0, row1, col1, profileWidth)
          : Promise.resolve(new Float32Array(0));
      };
      void (async () => {
        const first = await sampleFrame(0);
        const lineLen = first.length;
        if (lineLen < 2) {
          if (!cancelled) kymoDataRef.current = null;
          return;
        }
        const kymo = new Float32Array(nSlices * lineLen);
        kymo.set(first.subarray(0, lineLen), 0);
        for (let frameIdx = 1; frameIdx < nSlices; frameIdx++) {
          if (cancelled) return;
          const profile = await sampleFrame(frameIdx);
          kymo.set(profile.subarray(0, lineLen), frameIdx * lineLen);
        }
        publish(kymo, lineLen);
      })().catch(error => {
        if (!cancelled) console.error("[Show3D] WebGPU kymograph profile failed", error);
      });
      return () => { cancelled = true; };
    }

    if (kymoQuantizedStackReady && offlineStack) {
      // The qgpu kernel reads uint8+range directly and dequantizes only the
      // bilinear corners used by the profile.
      const u8 = new Uint8Array(offlineStack.buffer, offlineStack.byteOffset, offlineStack.byteLength);
      const sampleFrame = (frameIdx: number) => sampleLineProfileUint8Browser(
        u8.subarray(frameIdx * pixelCount, (frameIdx + 1) * pixelCount),
        offlineMin, offlineMax, width, height, row0, col0, row1, col1, profileWidth,
      );
      void (async () => {
        const first = await sampleFrame(0);
        const lineLen = first.length;
        if (lineLen < 2) { if (!cancelled) kymoDataRef.current = null; return; }
        const kymo = new Float32Array(nSlices * lineLen);
        kymo.set(first, 0);
        for (let frame = 1; frame < nSlices; frame++) {
          if (cancelled) return;
          kymo.set(await sampleFrame(frame), frame * lineLen);
        }
        publish(kymo, lineLen);
      })().catch(error => {
        if (!cancelled) console.error("[Show3D] Quantized WebGPU kymograph profile failed", error);
      });
      return () => { cancelled = true; };
    }

    kymoDataRef.current = null;
    return undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kymoReady, kymoExactStackReady, kymoQuantizedStackReady, offlineStack, offlineFloatStack, offlineMin, offlineMax, width, height, nSlices,
      profileWidth, profilePoints[0]?.row, profilePoints[0]?.col,
      profilePoints[1]?.row, profilePoints[1]?.col, profilePanelIdx, activePageStart,
      singlePanelPageProfile, totalPanelCount]);

  // Process kymograph data → histogram + colormap rendering (cheap, sync).
  // Mirrors the FFT pipeline: range → log scale → histogram/stats → slider
  // range → LUT → offscreen → draw with zoom/pan. Cold path, image is tiny.
  React.useEffect(() => {
    const kymo = kymoDataRef.current;
    if (!kymoReady || !kymo) return;
    const { data, lineLen, nFrames } = kymo;

    let displayMin: number, displayMax: number;
    if (kymoAuto) {
      ({ vmin: displayMin, vmax: displayMax } = percentileClip(data, percentileLow, percentileHigh));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(data));
    }

    const displayData = kymoLogScale ? applyLogScale(data) : data;
    if (kymoLogScale) {
      displayMin = Math.log1p(displayMin);
      displayMax = Math.log1p(displayMax);
    }

    setKymoHistogramData(displayData);
    setKymoDataRange({ min: displayMin, max: displayMax });
    setKymoStats(computeStats(displayData));

    const { vmin, vmax } = sliderRange(displayMin, displayMax, kymoVminPct, kymoVmaxPct);
    const lut = COLORMAPS[kymoColormap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(displayData, lineLen, nFrames, lut, vmin, vmax);
    if (!offscreen) return;

    kymoOffscreenRef.current = offscreen;

    if (kymoCanvasRef.current) {
      const ctx = kymoCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.imageSmoothingEnabled = smooth && (lineLen < canvasW || nFrames < canvasH);
        ctx.clearRect(0, 0, canvasW, canvasH);
        ctx.save();
        ctx.translate(kymoPanX, kymoPanY);
        ctx.scale(kymoZoom, kymoZoom);
        ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);
        ctx.restore();
      }
    }
  }, [kymoReady, kymoVersion, kymoLogScale, kymoAuto, kymoVminPct, kymoVmaxPct, kymoColormap,
      percentileLow, percentileHigh, canvasW, canvasH, canvasRepaintSignal, smooth]);

  // Redraw cached kymograph with zoom/pan (cheap - no recomputation)
  React.useEffect(() => {
    if (!kymoReady || !kymoCanvasRef.current || !kymoOffscreenRef.current) return;
    const canvas = kymoCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const offW = kymoOffscreenRef.current.width;
    const offH = kymoOffscreenRef.current.height;
    ctx.imageSmoothingEnabled = smooth && (offW < canvasW || offH < canvasH);
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(kymoPanX, kymoPanY);
    ctx.scale(kymoZoom, kymoZoom);
    ctx.drawImage(kymoOffscreenRef.current, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [kymoReady, kymoZoom, kymoPanX, kymoPanY, canvasW, canvasH, canvasRepaintSignal, smooth]);

  // Render kymograph overlay (playhead + axis scale bars + colorbar + click
  // crosshair). Mirrors the FFT overlay structure; the playhead is the only
  // part that tracks the current frame. Never recomputes the image.
  React.useEffect(() => {
    const overlay = kymoOverlayRef.current;
    const kymo = kymoDataRef.current;
    if (!overlay || !kymoReady || !kymo) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    overlay.width = Math.round(canvasW * DPR);
    overlay.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Playhead row marker - tracks the current frame in zoomed/panned space.
    const y = kymoPanY + kymoZoom * (((liveSliceIdx + 0.5) / kymo.nFrames) * canvasH);
    ctx.save();
    ctx.scale(DPR, DPR);
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvasW, y);
    ctx.stroke();
    ctx.restore();

    // Distance scale bar along the bottom edge (distance axis, pixelUnit).
    if (pixelSize > 0) {
      drawScaleBarHiDPI(overlay, DPR, kymoZoom, pixelSize, pixelUnit || "px", kymo.lineLen);
    }

    // Time scale bar along the left edge (time axis, dimUnit). Vertical bar +
    // label so the operator can read the temporal extent of the kymograph.
    if (dimSampling > 0 && dimUnit) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const targetBarPx = 60;
      const barThickness = 5;
      const margin = 12;
      const scaleY = canvasH / kymo.nFrames;
      const effectiveZoom = kymoZoom * scaleY;
      const targetPhysical = (targetBarPx / effectiveZoom) * dimSampling;
      const nicePhysical = roundToNiceValue(targetPhysical);
      const barPx = (nicePhysical / dimSampling) * effectiveZoom;
      const barX = margin;
      const barY = margin;
      ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
      ctx.shadowBlur = 2;
      ctx.shadowOffsetX = 1;
      ctx.shadowOffsetY = 1;
      ctx.fillStyle = "white";
      ctx.fillRect(barX, barY, barThickness, barPx);
      ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const label = nicePhysical >= 1 ? `${nicePhysical} ${dimUnit}` : `${nicePhysical.toPrecision(2)} ${dimUnit}`;
      ctx.fillText(label, barX + barThickness + 4, barY + barPx / 2);
      ctx.restore();
    }

    // Colorbar when enabled (mirror FFT colorbar draw).
    if (kymoShowColorbar && kymoDataRange.min !== kymoDataRange.max) {
      const { vmin, vmax } = sliderRange(kymoDataRange.min, kymoDataRange.max, kymoVminPct, kymoVmaxPct);
      const lut = COLORMAPS[kymoColormap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      drawColorbar(ctx, overlay.width / DPR, overlay.height / DPR, lut, vmin, vmax, kymoLogScale);
      ctx.restore();
    }

    // Click crosshair marker - mirror FFT marker, coordinates in zoomed space.
    if (kymoClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const screenX = kymoPanX + kymoZoom * (kymoClickInfo.col / kymo.lineLen * canvasW);
      const screenY = kymoPanY + kymoZoom * (kymoClickInfo.row / kymo.nFrames * canvasH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
      ctx.shadowBlur = 2;
      ctx.lineWidth = 1.5;
      const armLength = 8;
      ctx.beginPath();
      ctx.moveTo(screenX - armLength, screenY); ctx.lineTo(screenX - 3, screenY);
      ctx.moveTo(screenX + 3, screenY); ctx.lineTo(screenX + armLength, screenY);
      ctx.moveTo(screenX, screenY - armLength); ctx.lineTo(screenX, screenY - 3);
      ctx.moveTo(screenX, screenY + 3); ctx.lineTo(screenX, screenY + armLength);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }, [kymoReady, kymoVersion, liveSliceIdx, canvasW, canvasH, themeColors.accent, kymoZoom, kymoPanX, kymoPanY,
      pixelSize, pixelUnit, dimSampling, dimUnit, kymoShowColorbar, kymoDataRange, kymoVminPct, kymoVmaxPct,
      kymoColormap, kymoLogScale, kymoClickInfo, canvasRepaintSignal]);

  // Render FFT overlay (reciprocal-space scale bar + colorbar)
  React.useEffect(() => {
    const overlay = fftOverlayRef.current;
    if (!overlay || !effectiveShowFft) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    overlay.width = Math.round(canvasW * DPR);
    overlay.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Use crop dimensions for reciprocal-space calculations
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;

    // Reciprocal-space scale bar (pixelSize is in pixelUnit)
    if (pixelSize > 0) {
      const panelGrid = fftPanelGridRef.current;
      const reciprocalWidth = panelGrid ? panelGrid.panelWidth : fftW;
      const fftPixelSize = 1 / (reciprocalWidth * pixelSize);
      drawFFTScaleBarHiDPI(overlay, DPR, fftZoom, fftPixelSize, fftW, `${unitSymbol(pixelUnit || "px")}⁻¹`, false);
    }

    if (fftShowColorbar && fftDataRange.min !== fftDataRange.max) {
      const { vmin, vmax } = sliderRange(fftDataRange.min, fftDataRange.max, fftVminPct, fftVmaxPct);
      const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      const cssW = overlay.width / DPR;
      const cssH = overlay.height / DPR;
      drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, fftLogScale);
      ctx.restore();
    }

    // D-spacing crosshair marker - use crop dims for coordinate mapping
    if (fftClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      let screenX = fftPanX + fftZoom * (fftClickInfo.col / fftW * canvasW);
      let screenY = fftPanY + fftZoom * (fftClickInfo.row / fftH * canvasH);
      let centerX = fftPanX + fftZoom * (canvasW / 2);
      let centerY = fftPanY + fftZoom * (canvasH / 2);
      let radiusX = fftZoom * (fftClickInfo.distPx / Math.max(1, fftW)) * canvasW;
      let radiusY = fftZoom * (fftClickInfo.distPx / Math.max(1, fftH)) * canvasH;
      let clipRect: { x: number; y: number; w: number; h: number } | null = null;
      const panelGrid = fftPanelGridRef.current;
      if (panelGrid) {
        const slot = Math.max(0, Math.min(panelGrid.count - 1, Math.floor(fftClickInfo.row / panelGrid.panelHeight) * panelGrid.cols + Math.floor(fftClickInfo.col / panelGrid.panelWidth)));
        const dst = getFftSlot(slot, panelGrid.count, panelGrid.cols, panelGrid.rows);
        const localCol = fftClickInfo.col - (slot % panelGrid.cols) * panelGrid.panelWidth;
        const localRow = fftClickInfo.row - Math.floor(slot / panelGrid.cols) * panelGrid.panelHeight;
        screenX = dst.x + fftPanX + fftZoom * ((localCol / panelGrid.panelWidth) * dst.w);
        screenY = dst.y + fftPanY + fftZoom * ((localRow / panelGrid.panelHeight) * dst.h);
        centerX = dst.x + fftPanX + fftZoom * (dst.w / 2);
        centerY = dst.y + fftPanY + fftZoom * (dst.h / 2);
        radiusX = fftZoom * (fftClickInfo.distPx / Math.max(1, panelGrid.panelWidth)) * dst.w;
        radiusY = fftZoom * (fftClickInfo.distPx / Math.max(1, panelGrid.panelHeight)) * dst.h;
        clipRect = dst;
      }
      ctx.lineCap = "round";
      ctx.shadowBlur = 0;
      const armLength = 8;
      const drawRing = () => {
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
        ctx.stroke();
      };
      const drawMarker = () => {
        ctx.beginPath();
        ctx.moveTo(screenX - armLength, screenY); ctx.lineTo(screenX - 3, screenY);
        ctx.moveTo(screenX + 3, screenY); ctx.lineTo(screenX + armLength, screenY);
        ctx.moveTo(screenX, screenY - armLength); ctx.lineTo(screenX, screenY - 3);
        ctx.moveTo(screenX, screenY + 3); ctx.lineTo(screenX, screenY + armLength);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
        ctx.stroke();
      };
      if (clipRect) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(clipRect.x, clipRect.y, clipRect.w, clipRect.h);
        ctx.clip();
      }
      ctx.strokeStyle = "rgba(0, 0, 0, 0.78)";
      ctx.lineWidth = 4;
      drawRing();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.64)";
      ctx.lineWidth = 1.25;
      drawRing();
      if (clipRect) ctx.restore();
      ctx.strokeStyle = "rgba(0, 0, 0, 0.92)";
      ctx.lineWidth = 4;
      drawMarker();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.96)";
      ctx.lineWidth = 1.5;
      drawMarker();
      const label = fftClickInfo.dSpacing != null
        ? `d = ${formatLength(fftClickInfo.dSpacing, pixelUnit)}`
        : `dist = ${fftClickInfo.distPx.toFixed(1)} px`;
      ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const padX = 5;
      const labelW = Math.ceil(ctx.measureText(label).width + padX * 2);
      const labelH = 18;
      const cssW = overlay.width / DPR;
      const cssH = overlay.height / DPR;
      const labelX = Math.max(2, Math.min(cssW - labelW - 2, screenX + 10));
      const labelY = Math.max(labelH / 2 + 2, Math.min(cssH - labelH / 2 - 2, screenY - 10));
      ctx.fillStyle = "rgba(0, 0, 0, 0.74)";
      ctx.strokeStyle = "rgba(255, 255, 255, 0.82)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(labelX, labelY - labelH / 2, labelW, labelH, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "white";
      ctx.fillText(label, labelX + padX, labelY);
      ctx.restore();
    }
  }, [effectiveShowFft, fftZoom, fftPanX, fftPanY, canvasW, canvasH, pixelSize, pixelUnit, width, height, fftDataRange, fftVminPct, fftVmaxPct, fftColormap, fftLogScale, fftShowColorbar, fftClickInfo, fftCropDims, getFftSlot, canvasRepaintSignal]);

  // -------------------------------------------------------------------------
  // Preview panel - cache colormapped offscreen (only recomputes when ROI
  // geometry, data, or display settings change - NOT on zoom/pan)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    let cancelled = false;
    if (!previewVisible || !displayFrameRef.current) {
      previewOffscreenRef.current = null;
      return;
    }

    const raw = displayFrameRef.current;
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return;

    const roi = roiList[roiSelectedIdx];
    void cropMaskedRegionBrowser(raw, width, height, roi).then(crop => {
      if (cancelled) return;
      if (!crop) {
        previewOffscreenRef.current = null;
        setPreviewCropDims(null);
        setPreviewVersion(v => v + 1);
        return;
      }

      setPreviewCropDims({ w: crop.cropW, h: crop.cropH });

      const processed = logScale ? applyLogScale(crop.cropped) : crop.cropped;
      const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

      let vmin: number, vmax: number;
      const panelCount = Math.max(1, nPanels || 1);
      const hasTraitRange = traitVmin != null || traitVmax != null;
      const perPanelContrast = panelCount > 1 && !sharedPanelSource && width % panelCount === 0 && height > 0;
      if (hasTraitRange) {
        ({ vmin, vmax } = resolveDisplayRange(
          dataMin,
          dataMax,
          traitVmin,
          traitVmax,
          logScale,
          imageVminPct,
          imageVmaxPct,
        ));
      } else if (autoContrast) {
        const cached = cachedAutoDisplayRange(autoVmins, autoVmaxs, displaySliceIdx, logScale)
          || cachedAutoDisplayRange(localAutoVminsRef.current, localAutoVmaxsRef.current, displaySliceIdx, logScale);
        const mainProcessed = logScale ? applyLogScale(raw) : raw;
        ({ vmin, vmax } = cached ?? percentileClip(mainProcessed, percentileLow, percentileHigh));
      } else if (perPanelContrast) {
        const panelW = width / panelCount;
        const panel = Math.max(0, Math.min(panelCount - 1, Math.floor((Number(roi.col) || 0) / panelW)));
        const panelData = extractPanelSlice(raw, panel, logScale);
        const panelDataRange = panelDataRanges[panel];
        const panelRange = (perPanelHistogramEnabled && panelDataRange && panelDataRange.max > panelDataRange.min)
          ? panelDataRange
          : (panelData && panelData.length > 0
              ? findDataRange(panelData)
              : resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale));
        const resolved = resolvePanelRange(panel, panelRange, null);
        vmin = resolved.vmin;
        vmax = resolved.vmax;
      } else {
        const lo = logScale ? signedLog1p(dataMin) : dataMin;
        const hi = logScale ? signedLog1p(dataMax) : dataMax;
        ({ vmin, vmax } = sliderRange(lo, hi, imageVminPct, imageVmaxPct));
      }

      const offscreen = renderToOffscreen(processed, crop.cropW, crop.cropH, lut, vmin, vmax);
      previewOffscreenRef.current = offscreen;
      setPreviewVersion(v => v + 1);
    }).catch(error => {
      if (!cancelled) console.error("[Show3D] WebGPU ROI preview crop failed", error);
    });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewVisible, selectedRoiKey, cmap, logScale, autoContrast, imageVminPct, imageVmaxPct, dataMin, dataMax, traitVmin, traitVmax, percentileLow, percentileHigh, width, height, frameBytes, displaySliceIdx, autoVmins, autoVmaxs, nPanels, linkContrast, sharedPanelSource, panelStates, vminPerPanel, vmaxPerPanel, canvasRepaintSignal]);

  // -------------------------------------------------------------------------
  // Preview panel - compute aspect-ratio-aware canvas dimensions
  // -------------------------------------------------------------------------
  const previewCanvasDims = (() => {
    if (!previewCropDims) return { w: canvasW, h: canvasH };
    const { w: cropW, h: cropH } = previewCropDims;
    const aspect = cropW / cropH;
    if (aspect >= 1) {
      return { w: canvasW, h: Math.max(20, Math.round(canvasW / aspect)) };
    } else {
      return { w: Math.max(20, Math.round(canvasH * aspect)), h: canvasH };
    }
  })();

  // -------------------------------------------------------------------------
  // Preview panel - draw cached offscreen with zoom/pan (fast, no recompute)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = previewCanvasRef.current;
    if (!canvas || !previewVisible) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const previewWidth = previewCanvasDims.w;
    const previewHeight = previewCanvasDims.h;
    const offscreen = previewOffscreenRef.current;
    if (!offscreen || !previewCropDims) {
      ctx.clearRect(0, 0, previewWidth, previewHeight);
      return;
    }

    ctx.imageSmoothingEnabled = smooth;
    ctx.clearRect(0, 0, previewWidth, previewHeight);

    const { zoom: previewScale, panX: previewPanX, panY: previewPanY } = previewZoom;
    if (previewScale !== 1 || previewPanX !== 0 || previewPanY !== 0) {
      ctx.save();
      const cx = previewWidth / 2;
      const cy = previewHeight / 2;
      ctx.translate(cx + previewPanX, cy + previewPanY);
      ctx.scale(previewScale, previewScale);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, previewCropDims.w, previewCropDims.h, 0, 0, previewWidth, previewHeight);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, previewCropDims.w, previewCropDims.h, 0, 0, previewWidth, previewHeight);
    }
  }, [previewVisible, previewVersion, previewZoom, previewCanvasDims, previewCropDims, canvasRepaintSignal]);

  // Preview overlay - scale bar + zoom indicator
  React.useEffect(() => {
    const overlay = previewOverlayRef.current;
    if (!overlay || !previewVisible) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (previewCropDims && pixelSize > 0) {
      drawScaleBarHiDPI(overlay, DPR, previewZoom.zoom, pixelSize, pixelUnit, previewCropDims.w);
    }
  }, [previewVisible, previewZoom, previewCropDims, previewCanvasDims, pixelSize, pixelUnit, canvasRepaintSignal]);

  // Mouse handlers
  const panelIdxFromXY = (cssX: number, cssY: number): number => {
    const { n, cols, rows, gap, slotW, slotH } = getPanelLayout();
    if (n === 1) {
      return cssX >= 0 && cssX <= canvasW && cssY >= 0 && cssY <= canvasH
        ? (visiblePanelIndices[0] ?? 0)
        : -1;
    }
    const col = Math.floor(cssX / Math.max(1, slotW + gap));
    const row = Math.floor(cssY / Math.max(1, slotH + gap));
    if (col < 0 || col >= cols || row < 0 || row >= rows) return -1;
    const localX = cssX - col * (slotW + gap);
    const localY = cssY - row * (slotH + gap);
    if (localX < 0 || localX > slotW || localY < 0 || localY > slotH) return -1;
    const idx = row * cols + col;
    // Empty grid cells past N panels (partial last row) are not panels.
    return idx >= n ? -1 : (visiblePanelIndices[idx] ?? -1);
  };
  const panelIdxFromEvent = (e: React.MouseEvent): number => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    const cssX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const cssY = (e.clientY - rect.top) * (canvas.height / rect.height);
    return panelIdxFromXY(cssX, cssY);
  };
  const canvasPointFromClient = (clientX: number, clientY: number): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  };
  const panelIdxFromClient = (clientX: number, clientY: number): number => {
    const pt = canvasPointFromClient(clientX, clientY);
    return pt ? panelIdxFromXY(pt.x, pt.y) : -1;
  };
  const beginPan = (e: React.MouseEvent) => {
    const idx = panelIdxFromEvent(e);
    if (idx < 0) return;
    const live = playRef.current;
    const base = live.panelStates[idx] || stateFor(idx);
    const view = {
      ...base,
      zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
      panX: live.linkPanels ? live.linkedState.panX : base.panX,
      panY: live.linkPanels ? live.linkedState.panY : base.panY,
    };
    panDragRef.current = {
      panelIdx: idx,
      x: e.clientX,
      y: e.clientY,
      pX: view.panX,
      pY: view.panY,
    };
    setIsDraggingPan(true);
  };
  const applyCanvasWheelZoom = (clientX: number, clientY: number, deltaY: number): boolean => {
    const canvas = canvasRef.current;
    if (!canvas) return false;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    const mouseX = (clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (clientY - rect.top) * (canvas.height / rect.height);
    const panelIdx = panelIdxFromXY(mouseX, mouseY);
    if (panelIdx < 0) return false;
    const live = playRef.current;
    const base = live.panelStates[panelIdx] || stateFor(panelIdx);
    const cur = {
      ...base,
      zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
      panX: live.linkPanels ? live.linkedState.panX : base.panX,
      panY: live.linkPanels ? live.linkedState.panY : base.panY,
    };
    const zoomFactor = Math.max(0.75, Math.min(1.35, Math.exp(-deltaY * 0.002)));
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, cur.zoom * zoomFactor));
    const zoomRatio = newZoom / cur.zoom;
    // Mouse position relative to this panel's slot (so zoom anchors to cursor within slot).
    const geom = getPanelGeometry(panelIdx);
    if (!geom) return false;
    const localX = mouseX - geom.slotX;
    const localY = mouseY - geom.slotY;
    const newPanX = localX - (localX - cur.panX) * zoomRatio;
    const newPanY = localY - (localY - cur.panY) * zoomRatio;
    syncPlaybackPanelTransform(panelIdx, newZoom, newPanX, newPanY);
    if (scheduleTransformRender()) {
      scheduleTransformStateCommit();
    } else {
      commitLivePanelTransforms();
    }
    return true;
  };

  canvasWheelHandlerRef.current = (event: WheelEvent) => {
    if (fftInsetNativeWheelHandlerRef.current?.(event)) return;
    event.preventDefault();
    event.stopPropagation();
    if (reorderMode) return;
    applyCanvasWheelZoom(event.clientX, event.clientY, event.deltaY);
  };

  React.useEffect(() => {
    const el = canvasContainerRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => canvasWheelHandlerRef.current?.(event);
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [canvasW, canvasH]);

  // The FFT overlay inset's one wheel path: a non-passive capture listener on
  // the widget root, so the zoom never scrolls the notebook (React's onWheel is
  // passive) and runs before the image canvas below would zoom instead.
  React.useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const onFftInsetWheelCapture = (event: WheelEvent) => {
      fftInsetNativeWheelHandlerRef.current?.(event);
    };
    root.addEventListener("wheel", onFftInsetWheelCapture, { capture: true, passive: false });
    return () => root.removeEventListener("wheel", onFftInsetWheelCapture, { capture: true });
  }, []);

  const handleDoubleClick = () => {
    const resetPanels = Array.from({ length: Math.max(1, nPanels || 1) }, (_, i) => ({
      ...(playRef.current.panelStates[i] || initialState),
      zoom: 1,
      panX: 0,
      panY: 0,
    }));
    const resetLinked = { ...playRef.current.linkedState, zoom: 1, panX: 0, panY: 0 };
    playRef.current.linkedState = resetLinked;
    playRef.current.panelStates = resetPanels;
    linkedStateLiveRef.current = resetLinked;
    panelStatesLiveRef.current = resetPanels;
    setLinkedState(s => ({ ...s, zoom: 1, panX: 0, panY: 0 }));
    setPanelStates(arr => arr.map(s => ({ ...s, zoom: 1, panX: 0, panY: 0 })));
    setViewState({ linked_state: { ...resetLinked }, panel_states: resetPanels.map(v => ({ ...v })) });
    scheduleTransformRender();
  };

  const addROIAt = (row: number, col: number, shape: "circle" | "square" | "rectangle" | "annular" = newRoiShape) => {
    const clampedRow = Math.max(0, Math.min(height - 1, Math.round(row)));
    const clampedCol = Math.max(0, Math.min(width - 1, Math.round(col)));
    const next = [...roiItems, createROI(clampedRow, clampedCol, shape, roiItems.length, width, height)];
    setRoiList(next);
    setRoiSelectedIdx(next.length - 1);
    setShowRoiResizeHint(true);
  };

  const deleteSelectedROI = () => {
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return;
    const next = roiList.filter((_, i) => i !== roiSelectedIdx);
    setRoiList(next);
    setRoiSelectedIdx(next.length > 0 ? Math.min(roiSelectedIdx, next.length - 1) : -1);
  };

  const duplicateSelectedROI = () => {
    if (!selectedRoi) return;
    const duplicated: ROIItem = {
      ...selectedRoi,
      row: Math.max(0, Math.min(height - 1, Math.round(selectedRoi.row + 3))),
      col: Math.max(0, Math.min(width - 1, Math.round(selectedRoi.col + 3))),
      shape: selectedRoi.shape,
      radius: selectedRoi.radius,
      radius_inner: selectedRoi.radius_inner,
      width: selectedRoi.width,
      height: selectedRoi.height,
      color: ROI_COLORS[roiItems.length % ROI_COLORS.length],
      line_width: selectedRoi.line_width,
      highlight: false,
    };
    const next = [...roiItems, duplicated];
    setRoiList(next);
    setRoiSelectedIdx(next.length - 1);
  };


  const handleCopy = async () => {
    if (!canvasRef.current) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvasRef.current!.toBlob(resolve, "image/png"));
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch (err) {
      console.warn("Show3D copy failed", err);
    }
  };


  const clickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const touchTransformRef = React.useRef<TouchTransformState | null>(null);
  const fftTouchTransformRef = React.useRef<FftTouchTransformState | null>(null);
  const fftInsetTouchTransformRef = React.useRef<FftTouchTransformState | null>(null);
  const kymoTouchTransformRef = React.useRef<FftTouchTransformState | null>(null);
  const lastTapRef = React.useRef<{ time: number; panelIdx: number } | null>(null);
  const lastFftTapRef = React.useRef<{ time: number } | null>(null);
  const lastFftInsetTapRef = React.useRef<{ time: number } | null>(null);
  const lastKymoTapRef = React.useRef<{ time: number } | null>(null);
  const [draggingProfileEndpoint, setDraggingProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingProfileLine, setIsDraggingProfileLine] = React.useState(false);
  const [hoveredProfileEndpoint, setHoveredProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringProfileLine, setIsHoveringProfileLine] = React.useState(false);
  const profileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);

  const screenToImg = (e: React.MouseEvent): { imgCol: number; imgRow: number; panelIdx: number; panelCol: number } => {
    const pt = canvasPointFromEvent(e);
    if (!pt) return { imgCol: 0, imgRow: 0, panelIdx: -1, panelCol: 0 };
    const panelIdx = panelIdxFromXY(pt.x, pt.y);
    const geom = getPanelGeometry(panelIdx);
    if (!geom) return { imgCol: 0, imgRow: 0, panelIdx: -1, panelCol: 0 };
    // Undo slot offset, pan, zoom, then panel source scaling.
    let localCol = (pt.x - geom.slotX - geom.state.panX) / (geom.scaleX * geom.state.zoom);
    let row = (pt.y - geom.slotY - geom.state.panY) / (geom.scaleY * geom.state.zoom);
    // Undo image_rotation in panel-local source coordinates.
    const quarterTurns = (((imageRotation % 4) + 4) % 4) | 0;
    if (quarterTurns !== 0) {
      const rotSwap = (quarterTurns % 2) !== 0;
      const visW = rotSwap ? sourcePanelHeight : sourcePanelWidth;
      const visH = rotSwap ? sourcePanelWidth : sourcePanelHeight;
      const centeredCol = localCol - visW / 2;
      const centeredRow = row - visH / 2;
      let unrotatedCol: number, unrotatedRow: number;
      if (quarterTurns === 1) { unrotatedCol = centeredRow; unrotatedRow = -centeredCol; }
      else if (quarterTurns === 2) { unrotatedCol = -centeredCol; unrotatedRow = -centeredRow; }
      else { unrotatedCol = -centeredRow; unrotatedRow = centeredCol; }
      localCol = unrotatedCol + sourcePanelWidth / 2;
      row = unrotatedRow + sourcePanelHeight / 2;
    }
    return { imgCol: panelGlobalCol(localCol, panelIdx), imgRow: row, panelIdx, panelCol: localCol };
  };
  const profileCoordinateWidth = singlePanelPageProfile ? sourcePanelWidth : width;
  const screenToProfileImg = (e: React.MouseEvent): { imgCol: number; imgRow: number; panelIdx: number; panelCol: number } => {
    const point = screenToImg(e);
    return singlePanelPageProfile && point.panelIdx >= 0
      ? { ...point, imgCol: point.panelCol }
      : point;
  };

  const hitTestROI = (imgCol: number, imgRow: number): number => {
    if (!effectiveRoiActive || roiItems.length === 0) return -1;
    for (let roiIdx = roiItems.length - 1; roiIdx >= 0; roiIdx--) {
      const roi = roiItems[roiIdx];
      const shape = roi.shape || "circle";
      if (shape === "circle" || shape === "annular") {
        if (Math.sqrt((imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2) <= roi.radius) return roiIdx;
      } else if (shape === "square") {
        if (Math.abs(imgCol - roi.col) <= roi.radius && Math.abs(imgRow - roi.row) <= roi.radius) return roiIdx;
      } else if (shape === "rectangle") {
        if (Math.abs(imgCol - roi.col) <= roi.width / 2 && Math.abs(imgRow - roi.row) <= roi.height / 2) return roiIdx;
      }
    }
    return -1;
  };

  const getHitArea = () => RESIZE_HIT_AREA_PX / (displayScale * zoom);

  const isNearEdge = (imgCol: number, imgRow: number, roi: ROIItem): boolean => {
    const hitArea = getHitArea();
    const shape = roi.shape || "circle";
    if (shape === "circle" || shape === "annular") {
      const dist = Math.sqrt((imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2);
      return Math.abs(dist - roi.radius) < hitArea;
    }
    if (shape === "square") {
      const colDistance = Math.abs(imgCol - roi.col);
      const rowDistance = Math.abs(imgRow - roi.row);
      const radius = roi.radius;
      return (colDistance <= radius + hitArea && rowDistance <= radius + hitArea) && (Math.abs(colDistance - radius) < hitArea || Math.abs(rowDistance - radius) < hitArea);
    }
    if (shape === "rectangle") {
      const colDistance = Math.abs(imgCol - roi.col);
      const rowDistance = Math.abs(imgRow - roi.row);
      const halfWidth = roi.width / 2;
      const halfHeight = roi.height / 2;
      return (colDistance <= halfWidth + hitArea && rowDistance <= halfHeight + hitArea) && (Math.abs(colDistance - halfWidth) < hitArea || Math.abs(rowDistance - halfHeight) < hitArea);
    }
    return false;
  };

  const isNearResizeHandle = (imgCol: number, imgRow: number): boolean => {
    if (!effectiveRoiActive || !selectedRoi) return false;
    return isNearEdge(imgCol, imgRow, selectedRoi);
  };

  const isNearAnyEdge = (imgCol: number, imgRow: number): boolean => {
    if (!effectiveRoiActive || roiItems.length === 0) return false;
    return roiItems.some(roi => isNearEdge(imgCol, imgRow, roi));
  };

  const isNearResizeHandleInner = (imgCol: number, imgRow: number): boolean => {
    if (!effectiveRoiActive || !selectedRoi || selectedRoi.shape !== "annular") return false;
    const hitArea = getHitArea();
    const dist = Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
    return Math.abs(dist - selectedRoi.radius_inner) < hitArea;
  };

  const updateROI = (e: React.MouseEvent) => {
    if (!selectedRoi) return;
    const { imgCol, imgRow } = screenToImg(e);
    updateSelectedRoi({
      col: Math.max(0, Math.min(width - 1, Math.floor(imgCol))),
      row: Math.max(0, Math.min(height - 1, Math.floor(imgRow))),
    });
  };

  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    // Ignore clicks in empty grid cells (partial last row when N isn't a
    // multiple of max_cols). Otherwise the click attributes to the last
    // real panel and zoom/pan jumps unexpectedly.
    const panelForSelection = panelIdxFromEvent(e);
    if (panelForSelection < 0) return;
    if (handlePanelSelectionMouseDown(e, panelForSelection)) return;
    clickStartRef.current = { x: e.clientX, y: e.clientY };
    pendingRoiAddRef.current = null;
    // Check if clicking on lens inset for drag or resize
    if (showLens) {
      const rect = canvasContainerRef.current?.getBoundingClientRect();
      if (rect) {
        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        if (cssX >= lx && cssX <= lx + lensDisplaySize && cssY >= ly && cssY <= ly + lensDisplaySize) {
          const edgeHit = 8;
          const nearEdge = cssX - lx < edgeHit || lx + lensDisplaySize - cssX < edgeHit ||
                           cssY - ly < edgeHit || ly + lensDisplaySize - cssY < edgeHit;
          if (nearEdge) {
            setIsResizingLens(true);
            lensResizeStartRef.current = { my: e.clientY, startSize: lensDisplaySize };
          } else {
            setIsDraggingLens(true);
            lensDragStartRef.current = { mx: e.clientX, my: e.clientY, ax: lx, ay: ly };
          }
          return;
        }
      }
    }
    if (overlayEditMode) {
      const { imgRow, panelIdx, panelCol } = screenToImg(e);
      if (panelIdx >= 0) {
        const hitRadius = getImageHitRadius(panelIdx);
        const hit = panelOverlayHit(panelOverlays?.[panelIdx], imgRow, panelCol, sourcePanelWidth, sourcePanelHeight, hitRadius);
        if (hit) {
          const original = panelOverlays?.[panelIdx]?.[hit.overlay];
          if (!original) return;
          setOverlaySelection({ panel: panelIdx, overlay: hit.overlay });
          overlayDragRef.current = {
            mode: hit.mode,
            panel: panelIdx,
            overlay: hit.overlay,
            handle: hit.handle,
            startRow: imgRow,
            startCol: panelCol,
            original,
          };
          setIsDraggingOverlay(true);
          setIsDraggingPan(false);
          panDragRef.current = null;
          e.preventDefault();
          return;
        }
      }
      setOverlaySelection(null);
    }
    if (profileActive) {
      const { imgCol, imgRow, panelIdx } = screenToProfileImg(e);
      if (profilePoints.length === 2) {
        if (panelIdx !== profilePanelIdx) {
          beginPan(e);
          return;
        }
        const p0 = profilePoints[0];
        const p1 = profilePoints[1];
        const hitRadius = getImageHitRadius(profilePanelIdx);
        const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
        const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingPan(false);
          panDragRef.current = null;
          return;
        }
        if (pointToSegmentDistance(imgCol, imgRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
          setIsDraggingProfileLine(true);
          profileDragStartRef.current = {
            row: imgRow,
            col: imgCol,
            p0: { row: p0.row, col: p0.col },
            p1: { row: p1.row, col: p1.col },
          };
          setIsDraggingPan(false);
          panDragRef.current = null;
          return;
        }
      }
      beginPan(e);
      return;
    }
    if (effectiveRoiActive) {
      const { imgCol, imgRow } = screenToImg(e);
      if (isNearResizeHandleInner(imgCol, imgRow)) {
        setIsDraggingResizeInner(true);
        return;
      }
      if (isNearResizeHandle(imgCol, imgRow)) {
        e.preventDefault();
        resizeAspectRef.current = selectedRoi && (selectedRoi.shape === "rectangle") && selectedRoi.width > 0 && selectedRoi.height > 0 ? selectedRoi.width / selectedRoi.height : null;
        setIsDraggingResize(true);
        return;
      }
      if (roiItems.length > 0) {
        for (let roiIdx = roiItems.length - 1; roiIdx >= 0; roiIdx--) {
          const roi = roiItems[roiIdx];
          if (isNearEdge(imgCol, imgRow, roi)) {
            e.preventDefault();
            resizeAspectRef.current = roi && (roi.shape === "rectangle") && roi.width > 0 && roi.height > 0 ? roi.width / roi.height : null;
            setRoiSelectedIdx(roiIdx);
            setIsDraggingResize(true);
            return;
          }
        }
      }
      const hitIdx = hitTestROI(imgCol, imgRow);
      if (hitIdx >= 0) {
        setRoiSelectedIdx(hitIdx);
        setIsDraggingROI(true);
        return;
      }
      setRoiSelectedIdx(-1);
      pendingRoiAddRef.current = {
        row: Math.max(0, Math.min(height - 1, Math.round(imgRow))),
        col: Math.max(0, Math.min(width - 1, Math.round(imgCol))),
      };
      return;
    }
    beginPan(e);
  };

  type TouchPoint = { clientX: number; clientY: number };
  const touchDistance = (first: TouchPoint, second: TouchPoint) => Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY);
  const touchMidpoint = (first: TouchPoint, second: TouchPoint) => ({ x: (first.clientX + second.clientX) / 2, y: (first.clientY + second.clientY) / 2 });

  const handleCanvasTouchStart = (e: React.TouchEvent) => {
    if (profileActive || effectiveRoiActive) return;
    if (e.touches.length === 1) {
      const touch = e.touches[0];
      const panelIdx = panelIdxFromClient(touch.clientX, touch.clientY);
      if (panelIdx < 0) return;
      const now = Date.now();
      const lastTap = lastTapRef.current;
      if (lastTap && lastTap.panelIdx === panelIdx && now - lastTap.time < 320) {
        e.preventDefault();
        handleDoubleClick();
        lastTapRef.current = null;
        touchTransformRef.current = null;
        return;
      }
      lastTapRef.current = { time: now, panelIdx };
      if (showLens) return;
      const live = playRef.current;
      const base = live.panelStates[panelIdx] || stateFor(panelIdx);
      touchTransformRef.current = {
        panelIdx,
        mode: "pan",
        startX: touch.clientX,
        startY: touch.clientY,
        startDistance: 0,
        startMidX: touch.clientX,
        startMidY: touch.clientY,
        startState: {
          ...base,
          zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
          panX: live.linkPanels ? live.linkedState.panX : base.panX,
          panY: live.linkPanels ? live.linkedState.panY : base.panY,
        },
      };
      e.preventDefault();
      return;
    }
    if (e.touches.length >= 2) {
      const firstTouch = e.touches[0];
      const secondTouch = e.touches[1];
      const mid = touchMidpoint(firstTouch, secondTouch);
      const panelIdx = panelIdxFromClient(mid.x, mid.y);
      if (panelIdx < 0) return;
      const live = playRef.current;
      const base = live.panelStates[panelIdx] || stateFor(panelIdx);
      touchTransformRef.current = {
        panelIdx,
        mode: "pinch",
        startX: mid.x,
        startY: mid.y,
        startDistance: Math.max(1, touchDistance(firstTouch, secondTouch)),
        startMidX: mid.x,
        startMidY: mid.y,
        startState: {
          ...base,
          zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
          panX: live.linkPanels ? live.linkedState.panX : base.panX,
          panY: live.linkPanels ? live.linkedState.panY : base.panY,
        },
      };
      e.preventDefault();
    }
  };

  const handleCanvasTouchMove = (e: React.TouchEvent) => {
    const start = touchTransformRef.current;
    if (!start) return;
    const canvas = canvasRef.current;
    const geom = getPanelGeometry(start.panelIdx);
    if (!canvas || !geom) return;
    e.preventDefault();
    const base = start.startState;
    if (start.mode === "pinch" && e.touches.length >= 2) {
      const firstTouch = e.touches[0];
      const secondTouch = e.touches[1];
      const mid = touchMidpoint(firstTouch, secondTouch);
      const startPoint = canvasPointFromClient(start.startMidX, start.startMidY);
      const currentPoint = canvasPointFromClient(mid.x, mid.y);
      if (!startPoint || !currentPoint) return;
      const startLocalX = startPoint.x - geom.slotX;
      const startLocalY = startPoint.y - geom.slotY;
      const currentLocalX = currentPoint.x - geom.slotX;
      const currentLocalY = currentPoint.y - geom.slotY;
      const imageX = (startLocalX - base.panX) / base.zoom;
      const imageY = (startLocalY - base.panY) / base.zoom;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base.zoom * (touchDistance(firstTouch, secondTouch) / start.startDistance)));
      syncPlaybackPanelTransform(start.panelIdx, newZoom, currentLocalX - imageX * newZoom, currentLocalY - imageY * newZoom);
    } else if (start.mode === "pan" && e.touches.length === 1) {
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / Math.max(1, rect.width);
      const scaleY = canvas.height / Math.max(1, rect.height);
      syncPlaybackPanelTransform(
        start.panelIdx,
        base.zoom,
        base.panX + (touch.clientX - start.startX) * scaleX,
        base.panY + (touch.clientY - start.startY) * scaleY,
      );
    }
    if (scheduleTransformRender()) scheduleTransformStateCommit();
    else commitLivePanelTransforms();
  };

  const handleCanvasTouchEnd = (e: React.TouchEvent) => {
    if (e.touches.length > 0 || !touchTransformRef.current) return;
    commitLivePanelTransforms();
    touchTransformRef.current = null;
  };

  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    if (overlayDragRef.current) {
      const drag = overlayDragRef.current;
      const { imgRow, panelIdx, panelCol } = screenToImg(e);
      if (panelIdx !== drag.panel) return;
      updatePanelOverlay(
        drag.panel,
        drag.overlay,
        updateOverlayFromDrag(drag.original, drag.mode, drag.startRow, drag.startCol, imgRow, panelCol, sourcePanelWidth, sourcePanelHeight, drag.handle),
      );
      e.preventDefault();
      return;
    }
    // Fast path: during pan drag, skip all cursor/hover/lens work - just update pan
    const panDrag = panDragRef.current;
    if (panDrag) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - panDrag.x) * scaleX;
      const dy = (e.clientY - panDrag.y) * scaleY;
      const newPanX = panDrag.pX + dx;
      const newPanY = panDrag.pY + dy;
      const live = playRef.current;
      const base = live.panelStates[panDrag.panelIdx] || stateFor(panDrag.panelIdx);
      const current = {
        ...base,
        zoom: live.linkPanels ? live.linkedState.zoom : base.zoom,
        panX: live.linkPanels ? live.linkedState.panX : base.panX,
        panY: live.linkPanels ? live.linkedState.panY : base.panY,
      };
      syncPlaybackPanelTransform(panDrag.panelIdx, current.zoom, newPanX, newPanY);
      if (!scheduleTransformRender()) commitLivePanelTransforms();
      return;
    }

    // Cursor readout: convert screen position to image pixel coordinates.
    // Skip when hovering an empty grid cell (partial last row when nPanels
    // isn't a multiple of max_cols) so dead space doesn't flash row/col
    // numbers from a phantom panel.
    const canvas = canvasRef.current;
    const hoverPanelIdx = panelIdxFromEvent(e);
    if (hoverPanelIdx < 0) {
      scheduleCursorInfo(null);
      if (showLens) setLensPos(null);
    } else if (canvas && measuredFrameRef.current) {
      const { imgRow, imgCol, panelIdx, panelCol } = screenToImg(e);
      const pixelDataCol = Math.floor(imgCol);
      const pixelPanelCol = Math.floor(panelCol);
      const pixelRow = Math.floor(imgRow);
      if (
        pixelDataCol >= 0 && pixelDataCol < width &&
        pixelPanelCol >= 0 && pixelPanelCol < sourcePanelWidth &&
        pixelRow >= 0 && pixelRow < height
      ) {
        const rawData = measuredFrameRef.current;
        scheduleCursorInfo({
          row: pixelRow,
          col: pixelPanelCol,
          value: rawData[pixelRow * width + pixelDataCol],
          panelIdx,
        });
        if (showLens) setLensPos({ row: pixelRow, col: pixelDataCol });
      } else {
        scheduleCursorInfo(null);
        if (showLens) setLensPos(null);
      }
    }

    // Lens edge hover detection
    if (showLens) {
      const rect2 = canvasContainerRef.current?.getBoundingClientRect();
      if (rect2) {
        const cssX2 = e.clientX - rect2.left;
        const cssY2 = e.clientY - rect2.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        const inside = cssX2 >= lx && cssX2 <= lx + lensDisplaySize && cssY2 >= ly && cssY2 <= ly + lensDisplaySize;
        const edgeHit = 8;
        const nearEdge = inside && (cssX2 - lx < edgeHit || lx + lensDisplaySize - cssX2 < edgeHit ||
                                     cssY2 - ly < edgeHit || ly + lensDisplaySize - cssY2 < edgeHit);
        setIsHoveringLensEdge(nearEdge);
      }
    } else {
      setIsHoveringLensEdge(false);
    }
    if (overlayEditMode && !isDraggingPan) {
      const { imgRow, panelIdx, panelCol } = screenToImg(e);
      const hit = panelIdx >= 0
        ? panelOverlayHit(panelOverlays?.[panelIdx], imgRow, panelCol, sourcePanelWidth, sourcePanelHeight, getImageHitRadius(panelIdx))
        : null;
      setIsHoveringOverlay(Boolean(hit));
      return;
    } else if (isHoveringOverlay) {
      setIsHoveringOverlay(false);
    }

    // Lens drag
    if (isDraggingLens && lensDragStartRef.current) {
      const dx = e.clientX - lensDragStartRef.current.mx;
      const dy = e.clientY - lensDragStartRef.current.my;
      setLensAnchor({ x: lensDragStartRef.current.ax + dx, y: lensDragStartRef.current.ay + dy });
      return;
    }

    // Lens resize drag
    if (isResizingLens && lensResizeStartRef.current) {
      const dy = e.clientY - lensResizeStartRef.current.my;
      setLensDisplaySize(Math.max(64, Math.min(256, lensResizeStartRef.current.startSize + dy)));
      return;
    }

    if (profileActive && profilePoints.length === 2) {
      const { imgCol, imgRow, panelIdx } = screenToProfileImg(e);
      const p0 = profilePoints[0];
      const p1 = profilePoints[1];
      const hitRadius = getImageHitRadius(profilePanelIdx);
      const sameProfilePanel = panelIdx === profilePanelIdx;
      const d0 = sameProfilePanel ? Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2) : Infinity;
      const d1 = sameProfilePanel ? Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2) : Infinity;
      if (draggingProfileEndpoint !== null) {
        if (!measuredFrameRef.current || panelIdx !== profilePanelIdx) return;
        const clampedRow = Math.max(0, Math.min(height - 1, imgRow));
        const clampedCol = Math.max(0, Math.min(profileCoordinateWidth - 1, imgCol));
        const next = [
          draggingProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
          draggingProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
        ];
        setProfileLine(next);
        updateProfileForActivePage(measuredFrameRef.current, next[0], next[1]);
        return;
      }
      if (isDraggingProfileLine && profileDragStartRef.current) {
        if (!measuredFrameRef.current || panelIdx !== profilePanelIdx) return;
        const drag = profileDragStartRef.current;
        let deltaRow = imgRow - drag.row;
        let deltaCol = imgCol - drag.col;
        const minRow = Math.min(drag.p0.row, drag.p1.row);
        const maxRow = Math.max(drag.p0.row, drag.p1.row);
        const minCol = Math.min(drag.p0.col, drag.p1.col);
        const maxCol = Math.max(drag.p0.col, drag.p1.col);
        deltaRow = Math.max(deltaRow, -minRow);
        deltaRow = Math.min(deltaRow, (height - 1) - maxRow);
        deltaCol = Math.max(deltaCol, -minCol);
        deltaCol = Math.min(deltaCol, (profileCoordinateWidth - 1) - maxCol);
        const next = [
          { row: drag.p0.row + deltaRow, col: drag.p0.col + deltaCol },
          { row: drag.p1.row + deltaRow, col: drag.p1.col + deltaCol },
        ];
        setProfileLine(next);
        updateProfileForActivePage(measuredFrameRef.current, next[0], next[1]);
        return;
      }
      const nextHoveredEndpoint: 0 | 1 | null = d0 <= hitRadius ? 0 : d1 <= hitRadius ? 1 : null;
      const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(imgCol, imgRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius;
      setHoveredProfileEndpoint(nextHoveredEndpoint);
      setIsHoveringProfileLine(nextHoverLine);
    } else {
      if (hoveredProfileEndpoint !== null) setHoveredProfileEndpoint(null);
      if (isHoveringProfileLine) setIsHoveringProfileLine(false);
    }

    // Resize handle dragging
    if (isDraggingResizeInner && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const newR = Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
      updateSelectedRoi({ radius_inner: Math.max(1, Math.min(selectedRoi.radius - 1, Math.round(newR))) });
      setShowRoiResizeHint(false);
      return;
    }
    if (isDraggingResize && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const shape = selectedRoi.shape || "circle";
      if (shape === "rectangle") {
        let newW = Math.max(2, Math.round(Math.abs(ic - selectedRoi.col) * 2));
        let newH = Math.max(2, Math.round(Math.abs(ir - selectedRoi.row) * 2));
        if (e.shiftKey && resizeAspectRef.current != null) {
          const aspect = resizeAspectRef.current;
          if (newW / newH > aspect) newH = Math.max(2, Math.round(newW / aspect));
          else newW = Math.max(2, Math.round(newH * aspect));
        }
        updateSelectedRoi({ width: newW, height: newH });
      } else {
        const newR = shape === "square"
          ? Math.max(Math.abs(ic - selectedRoi.col), Math.abs(ir - selectedRoi.row))
          : Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
        const minR = shape === "annular" ? selectedRoi.radius_inner + 1 : 1;
        updateSelectedRoi({ radius: Math.max(minR, Math.round(newR)) });
      }
      setShowRoiResizeHint(false);
      return;
    }

    // Hover state for resize handles
    if (effectiveRoiActive && !isDraggingROI && !isDraggingPan) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const hoveringInner = isNearResizeHandleInner(ic, ir);
      const hoveringOuter = isNearAnyEdge(ic, ir);
      setIsHoveringResizeInner(hoveringInner);
      setIsHoveringResize(hoveringOuter);
      if (hoveringInner || hoveringOuter) setShowRoiResizeHint(false);
    }

    if (isDraggingROI) {
      updateROI(e);
    }
  };

  const handleCanvasMouseUp = (e: React.MouseEvent) => {
    if (overlayDragRef.current) {
      overlayDragRef.current = null;
      setIsDraggingOverlay(false);
      return;
    }
    if (draggingProfileEndpoint !== null || isDraggingProfileLine) {
      setDraggingProfileEndpoint(null);
      setIsDraggingProfileLine(false);
      profileDragStartRef.current = null;
      clickStartRef.current = null;
      pendingRoiAddRef.current = null;
      setIsDraggingROI(false);
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      setIsDraggingLens(false);
      lensDragStartRef.current = null;
      setIsResizingLens(false);
      lensResizeStartRef.current = null;
      setIsDraggingPan(false);
      panDragRef.current = null;
      setHoveredProfileEndpoint(null);
      setIsHoveringProfileLine(false);
      return;
    }

    // Profile click capture
    if (profileActive && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        if (measuredFrameRef.current) {
          const { imgCol, imgRow, panelIdx } = screenToProfileImg(e);
          if (panelIdx >= 0 && imgCol >= 0 && imgCol < profileCoordinateWidth && imgRow >= 0 && imgRow < height) {
            const pt = { row: imgRow, col: imgCol };
            if (profilePoints.length === 0 || profilePoints.length === 2 || panelIdx !== profilePanelIdx) {
              setProfilePanelIdx(panelIdx);
              setProfileLine([pt]);
              setProfileData(null);
            } else {
              const p0 = profilePoints[0];
              setProfileLine([p0, pt]);
              updateProfileForActivePage(measuredFrameRef.current, p0, pt);
            }
          }
        }
      }
    }

    // ROI click-to-add (empty-area click)
    if (effectiveRoiActive && pendingRoiAddRef.current && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        addROIAt(pendingRoiAddRef.current.row, pendingRoiAddRef.current.col);
      }
    }
    clickStartRef.current = null;
    pendingRoiAddRef.current = null;
    if (panDragRef.current) commitLivePanelTransforms();
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingLens(false);
    lensDragStartRef.current = null;
    setIsResizingLens(false);
    lensResizeStartRef.current = null;
    setIsDraggingPan(false);
    panDragRef.current = null;
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    profileDragStartRef.current = null;
  };

  const handleCanvasMouseLeave = () => {
    scheduleCursorInfo(null);
    // Lens persists at last position when cursor exits main canvas. Wiping on every
    // leave kills the inset whenever the user touches a slider, FFT panel, or any
    // sibling control - surprising "lens vanished" footgun. User explicitly turns
    // lens off via the Lens switch.
    pendingRoiAddRef.current = null;
    overlayDragRef.current = null;
    setIsDraggingOverlay(false);
    setIsHoveringOverlay(false);
    if (panDragRef.current) commitLivePanelTransforms();
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingLens(false);
    lensDragStartRef.current = null;
    setIsResizingLens(false);
    lensResizeStartRef.current = null;
    setIsHoveringLensEdge(false);
    setIsHoveringResize(false);
    setIsHoveringResizeInner(false);
    setIsDraggingPan(false);
    panDragRef.current = null;
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    profileDragStartRef.current = null;
  };

  // FFT mouse handlers
  const [isFftDragging, setIsFftDragging] = React.useState(false);
  const [fftPanStart, setFftPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number, panelIdx: number | null, viewportW: number, viewportH: number } | null>(null);

  const clampFftPan = React.useCallback((panX: number, panY: number, zoom: number, viewportW: number, viewportH: number) => {
    const clampAxis = (pan: number, viewport: number) => {
      if (zoom <= 1 || viewport <= 0) return 0;
      return Math.max(viewport * (1 - zoom), Math.min(0, pan));
    };
    return {
      panX: clampAxis(panX, viewportW),
      panY: clampAxis(panY, viewportH),
    };
  }, []);

  const zoomFftAtPoint = React.useCallback((anchorX: number, anchorY: number, deltaY: number, viewportW?: number, viewportH?: number, panelIdx: number | null = null) => {
    const currentBase = panelIdx != null && !linkPanels
      ? getFftViewForPanel(panelIdx)
      : fftViewLiveRef.current;
    const current = !fftUserAdjustedViewRef.current && currentBase.zoom > 1 && viewportW != null && viewportH != null
      ? {
        zoom: currentBase.zoom,
        panX: viewportW * (1 - currentBase.zoom) / 2,
        panY: viewportH * (1 - currentBase.zoom) / 2,
      }
      : currentBase;
    fftUserAdjustedViewRef.current = true;
    const zoomFactor = Math.max(0.75, Math.min(1.35, Math.exp(-deltaY * 0.002)));
    const minZoom = fftLayoutOverlay ? 1 : MIN_ZOOM;
    const newZoom = Math.max(minZoom, Math.min(MAX_ZOOM, current.zoom * zoomFactor));
    const zoomRatio = newZoom / Math.max(1e-6, current.zoom);
    const nextPanX = anchorX - (anchorX - current.panX) * zoomRatio;
    const nextPanY = anchorY - (anchorY - current.panY) * zoomRatio;
    const clamped = viewportW != null && viewportH != null
      ? clampFftPan(nextPanX, nextPanY, newZoom, viewportW, viewportH)
      : { panX: nextPanX, panY: nextPanY };
    const next = { zoom: newZoom, panX: clamped.panX, panY: clamped.panY };
    if (panelIdx != null && !linkPanels) {
      setFftViewForPanel(panelIdx, next);
    } else {
      scheduleFftViewState(next, true, fftLayoutOverlay);
    }
  }, [clampFftPan, fftLayoutOverlay, getFftViewForPanel, linkPanels, scheduleFftViewState, setFftViewForPanel]);

  fftInsetNativeWheelHandlerRef.current = (event: WheelEvent) => {
    const target = event.target;
    let inset: Element | null = target instanceof Element
      ? target.closest('[data-show3d-fft-inset="true"]')
      : null;
    if (!(inset instanceof HTMLElement)) {
      inset = document.elementsFromPoint(event.clientX, event.clientY)
        .find(el => el instanceof Element && el.closest('[data-show3d-fft-inset="true"]'))
        ?.closest('[data-show3d-fft-inset="true"]') ?? null;
    }
    if (!(inset instanceof HTMLElement)) {
      const root = rootRef.current;
      const hit = root
        ? Array.from(root.querySelectorAll<HTMLElement>('[data-show3d-fft-inset="true"]')).find(el => {
          const rect = el.getBoundingClientRect();
          return event.clientX >= rect.left && event.clientX <= rect.right
            && event.clientY >= rect.top && event.clientY <= rect.bottom;
        })
        : null;
      inset = hit ?? null;
    }
    if (!(inset instanceof HTMLElement)) return false;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    const rect = inset.getBoundingClientRect();
    zoomFftAtPoint(event.clientX - rect.left, event.clientY - rect.top, event.deltaY, rect.width, rect.height);
    return true;
  };

  const handleFftWheel = (e: React.WheelEvent) => {
    e.stopPropagation();
    const canvas = fftCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const panelGrid = fftPanelGridRef.current;
    if (panelGrid) {
      for (let slot = 0; slot < panelGrid.count; slot++) {
        const dst = getFftSlot(slot, panelGrid.count, panelGrid.cols, panelGrid.rows);
        if (mouseX < dst.x || mouseX >= dst.x + dst.w || mouseY < dst.y || mouseY >= dst.y + dst.h) continue;
        const localX = mouseX - dst.x;
        const localY = mouseY - dst.y;
        const panel = visiblePanelIndices[slot] ?? slot;
        zoomFftAtPoint(localX, localY, e.deltaY, dst.w, dst.h, panel);
        return;
      }
    }
    zoomFftAtPoint(mouseX, mouseY, e.deltaY, canvas.width, canvas.height);
  };

  const handleFftInsetTouchStart = (e: React.TouchEvent<HTMLElement>) => {
    const now = Date.now();
    if (e.touches.length === 1) {
      const lastTap = lastFftInsetTapRef.current;
      if (lastTap && now - lastTap.time < 320) {
        e.preventDefault();
        e.stopPropagation();
        handleFftReset();
        lastFftInsetTapRef.current = null;
        fftInsetTouchTransformRef.current = null;
        return;
      }
      lastFftInsetTapRef.current = { time: now };
      return;
    }
    if (e.touches.length < 2) return;
    e.preventDefault();
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const firstTouch = e.touches[0];
    const secondTouch = e.touches[1];
    const mid = touchMidpoint(firstTouch, secondTouch);
    const live = fftViewLiveRef.current;
    const base = !fftUserAdjustedViewRef.current && live.zoom > 1
      ? {
        zoom: live.zoom,
        panX: rect.width * (1 - live.zoom) / 2,
        panY: rect.height * (1 - live.zoom) / 2,
      }
      : live;
    fftInsetTouchTransformRef.current = {
      mode: "pinch",
      startX: mid.x,
      startY: mid.y,
      startDistance: Math.max(1, touchDistance(firstTouch, secondTouch)),
      startMidX: mid.x,
      startMidY: mid.y,
      startState: base,
    };
  };

  const handleFftInsetTouchMove = (e: React.TouchEvent<HTMLElement>) => {
    const start = fftInsetTouchTransformRef.current;
    if (!start || e.touches.length < 2) return;
    e.preventDefault();
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const firstTouch = e.touches[0];
    const secondTouch = e.touches[1];
    const mid = touchMidpoint(firstTouch, secondTouch);
    const startX = start.startMidX - rect.left;
    const startY = start.startMidY - rect.top;
    const currentX = mid.x - rect.left;
    const currentY = mid.y - rect.top;
    const base = start.startState;
    const imageX = (startX - base.panX) / Math.max(1e-6, base.zoom);
    const imageY = (startY - base.panY) / Math.max(1e-6, base.zoom);
    const newZoom = Math.max(1, Math.min(MAX_ZOOM, base.zoom * (touchDistance(firstTouch, secondTouch) / start.startDistance)));
    const clamped = clampFftPan(
      currentX - imageX * newZoom,
      currentY - imageY * newZoom,
      newZoom,
      rect.width,
      rect.height,
    );
    fftUserAdjustedViewRef.current = true;
    scheduleFftViewState({ zoom: newZoom, panX: clamped.panX, panY: clamped.panY }, true, true);
  };

  const handleFftInsetTouchEnd = (e: React.TouchEvent<HTMLElement>) => {
    if (e.touches.length < 2) fftInsetTouchTransformRef.current = null;
  };

  const handleFftInsetPointerDown = (
    e: React.PointerEvent<HTMLElement>,
    panelLeft: number,
    panelTop: number,
    panelW: number,
    panelH: number,
    insetX: number,
    insetY: number,
    insetW: number,
    insetH: number,
  ) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    fftOverlayDragRef.current = {
      pointerId: e.pointerId,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startInsetX: insetX,
      startInsetY: insetY,
      panelLeft,
      panelTop,
      panelW,
      panelH,
      insetW,
      insetH,
      moved: false,
    };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const handleFftInsetPointerMove = (e: React.PointerEvent<HTMLElement>) => {
    const drag = fftOverlayDragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    if (Math.hypot(e.clientX - drag.startClientX, e.clientY - drag.startClientY) > 4) {
      drag.moved = true;
    }
    if (drag.moved) {
      const nextX = Math.max(drag.panelLeft, Math.min(drag.panelLeft + drag.panelW - drag.insetW, drag.startInsetX + e.clientX - drag.startClientX));
      const nextY = Math.max(drag.panelTop, Math.min(drag.panelTop + drag.panelH - drag.insetH, drag.startInsetY + e.clientY - drag.startClientY));
      setFftOverlayDragPreview({ x: nextX - drag.panelLeft, y: nextY - drag.panelTop });
    }
  };

  const handleFftInsetPointerUp = (e: React.PointerEvent<HTMLElement>) => {
    const drag = fftOverlayDragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    fftOverlayDragRef.current = null;
    setFftOverlayDragPreview(null);
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    if (!drag.moved) return;
    const centerX = drag.startInsetX + e.clientX - drag.startClientX - drag.panelLeft + drag.insetW / 2;
    const centerY = drag.startInsetY + e.clientY - drag.startClientY - drag.panelTop + drag.insetH / 2;
    const vertical = centerY < drag.panelH / 2 ? "top" : "bottom";
    const horizontal = centerX < drag.panelW / 2 ? "left" : "right";
    setFftOverlayPosition(`${vertical}-${horizontal}`);
  };

  const handleFftInsetMouseDown = (
    e: React.MouseEvent<HTMLElement>,
    panelLeft: number,
    panelTop: number,
    panelW: number,
    panelH: number,
    insetX: number,
    insetY: number,
    insetW: number,
    insetH: number,
  ) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const drag = {
      pointerId: -1,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startInsetX: insetX,
      startInsetY: insetY,
      panelLeft,
      panelTop,
      panelW,
      panelH,
      insetW,
      insetH,
      moved: false,
    };
    fftOverlayDragRef.current = drag;
    const onMove = (ev: MouseEvent) => {
      ev.preventDefault();
      if (Math.hypot(ev.clientX - drag.startClientX, ev.clientY - drag.startClientY) > 4) {
        drag.moved = true;
      }
      if (drag.moved) {
        const nextX = Math.max(drag.panelLeft, Math.min(drag.panelLeft + drag.panelW - drag.insetW, drag.startInsetX + ev.clientX - drag.startClientX));
        const nextY = Math.max(drag.panelTop, Math.min(drag.panelTop + drag.panelH - drag.insetH, drag.startInsetY + ev.clientY - drag.startClientY));
        setFftOverlayDragPreview({ x: nextX - drag.panelLeft, y: nextY - drag.panelTop });
      }
    };
    const onUp = (ev: MouseEvent) => {
      window.removeEventListener("mousemove", onMove, true);
      window.removeEventListener("mouseup", onUp, true);
      if (fftOverlayDragRef.current === drag) fftOverlayDragRef.current = null;
      setFftOverlayDragPreview(null);
      if (!drag.moved) return;
      const centerX = drag.startInsetX + ev.clientX - drag.startClientX - drag.panelLeft + drag.insetW / 2;
      const centerY = drag.startInsetY + ev.clientY - drag.startClientY - drag.panelTop + drag.insetH / 2;
      const vertical = centerY < drag.panelH / 2 ? "top" : "bottom";
      const horizontal = centerX < drag.panelW / 2 ? "left" : "right";
      setFftOverlayPosition(`${vertical}-${horizontal}`);
    };
    window.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);
  };

  const handleFftInsetPanMouseDown = (e: React.MouseEvent<HTMLElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const target = e.currentTarget;
    const rect = target.getBoundingClientRect();
    const viewportW = Math.max(1, rect.width);
    const viewportH = Math.max(1, rect.height);
    const startX = e.clientX;
    const startY = e.clientY;
    const current = fftViewLiveRef.current;
    const startView = !fftUserAdjustedViewRef.current && current.zoom > 1
      ? {
        zoom: current.zoom,
        panX: viewportW * (1 - current.zoom) / 2,
        panY: viewportH * (1 - current.zoom) / 2,
      }
      : current;
    if (!fftUserAdjustedViewRef.current) {
      scheduleFftViewState(startView, false, fftLayoutOverlay);
    }
    fftUserAdjustedViewRef.current = true;
    const onMove = (ev: MouseEvent) => {
      ev.preventDefault();
      const clamped = clampFftPan(
        startView.panX + (ev.clientX - startX),
        startView.panY + (ev.clientY - startY),
        startView.zoom,
        viewportW,
        viewportH,
      );
      scheduleFftViewState({ zoom: startView.zoom, panX: clamped.panX, panY: clamped.panY }, false, fftLayoutOverlay);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove, true);
      window.removeEventListener("mouseup", onUp, true);
    };
    window.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);
  };

  React.useEffect(() => {
    if (!effectiveShowFft) return;
    const overlayCanvas = fftLayoutOverlay ? fftInsetLayerRef.current : null;
    const fftCanvas = fftCanvasRef.current;
    const panelGrid = fftPanelGridRef.current;
    const viewport = overlayCanvas
      ? (() => {
        const visibleCount = Math.max(1, visiblePanelCount || 1);
        const cols = panelColsForCount(visibleCount);
        const rows = Math.ceil(visibleCount / cols);
        const gap = visibleCount > 1 ? (panelGapPx) : 0;
        const panelW = (canvasW - gap * (cols - 1)) / cols;
        const panelH = (canvasH - gap * (rows - 1)) / rows;
        const insetPad = Math.min(8, Math.max(3, panelW * 0.025));
        const insetMaxW = Math.max(24, panelW - insetPad * 2);
        const insetMaxH = Math.max(20, panelH - insetPad * 2);
        const insetBase = Math.min(insetMaxW, insetMaxH);
        return {
          w: Math.max(24, Math.min(insetMaxW, insetBase * resolvedFftOverlaySize)),
          h: Math.max(20, Math.min(insetMaxH, insetBase * resolvedFftOverlaySize)),
        };
      })()
      : fftCanvas
        ? panelGrid
          ? getFftSlot(0, panelGrid.count, panelGrid.cols, panelGrid.rows)
          : { w: fftCanvas.width, h: fftCanvas.height }
        : null;
    if (!viewport) return;
    // The draw translates by the pan, then scales about the viewport's top-left corner, so the
    // viewport centre (w/2) stays put when pan = w/2 - zoom * w/2 = w * (1 - zoom) / 2.
    const current = fftViewLiveRef.current;
    const wanted = fftUserAdjustedViewRef.current
      ? { panX: current.panX, panY: current.panY }
      : { panX: viewport.w * (1 - current.zoom) / 2, panY: viewport.h * (1 - current.zoom) / 2 };
    const clamped = clampFftPan(wanted.panX, wanted.panY, current.zoom, viewport.w, viewport.h);
    if (Math.abs(clamped.panX - current.panX) > 0.5 || Math.abs(clamped.panY - current.panY) > 0.5) {
      // Committed in this effect, not on the next frame: a deferred commit would race the effect
      // that copies React's view state into fftViewLiveRef and lose the new pan.
      fftViewLiveRef.current = { zoom: current.zoom, panX: clamped.panX, panY: clamped.panY };
      commitFftViewReactState();
    }
  }, [clampFftPan, commitFftViewReactState, effectiveShowFft, fftLayoutOverlay, fftZoom, fftPanX, fftPanY, canvasW, canvasH, resolvedFftOverlaySize, visiblePanelCount, panelColsForCount, panelGapPx, fftOffscreenVersion]);

  // Convert FFT canvas mouse position to FFT image pixel coordinates
  const fftScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    const panelGrid = fftPanelGridRef.current;
    if (panelGrid) {
      for (let slot = 0; slot < panelGrid.count; slot++) {
        const dst = getFftSlot(slot, panelGrid.count, panelGrid.cols, panelGrid.rows);
        if (mouseX < dst.x || mouseX >= dst.x + dst.w || mouseY < dst.y || mouseY >= dst.y + dst.h) continue;
        const panel = visiblePanelIndices[slot] ?? slot;
        const view = linkPanels ? { zoom: fftZoom, panX: fftPanX, panY: fftPanY } : getFftViewForPanel(panel);
        const localX = (mouseX - dst.x - view.panX) / view.zoom;
        const localY = (mouseY - dst.y - view.panY) / view.zoom;
        if (localX < 0 || localX >= dst.w || localY < 0 || localY >= dst.h) return null;
        const srcCol = slot % panelGrid.cols;
        const srcRow = Math.floor(slot / panelGrid.cols);
        const tileX = (localX / Math.max(1, dst.w)) * panelGrid.panelWidth;
        const tileY = (localY / Math.max(1, dst.h)) * panelGrid.panelHeight;
        return {
          col: srcCol * panelGrid.panelWidth + Math.max(0, Math.min(panelGrid.panelWidth - 1, tileX)),
          row: srcRow * panelGrid.panelHeight + Math.max(0, Math.min(panelGrid.panelHeight - 1, tileY)),
        };
      }
      return null;
    }
    const localX = (mouseX - fftPanX) / fftZoom;
    const localY = (mouseY - fftPanY) / fftZoom;
    const imgCol = localX / canvasW * fftW;
    const imgRow = localY / canvasH * fftH;
    if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
      return { col: imgCol, row: imgRow };
    }
    return null;
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsFftDragging(true);
    const canvas = fftCanvasRef.current;
    if (!canvas) {
      setFftPanStart({ x: e.clientX, y: e.clientY, pX: fftPanX, pY: fftPanY, panelIdx: null, viewportW: canvasW, viewportH: canvasH });
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / Math.max(1, rect.width));
    const mouseY = (e.clientY - rect.top) * (canvas.height / Math.max(1, rect.height));
    const panelGrid = fftPanelGridRef.current;
    if (panelGrid) {
      for (let slot = 0; slot < panelGrid.count; slot++) {
        const dst = getFftSlot(slot, panelGrid.count, panelGrid.cols, panelGrid.rows);
        if (mouseX < dst.x || mouseX >= dst.x + dst.w || mouseY < dst.y || mouseY >= dst.y + dst.h) continue;
        const panel = visiblePanelIndices[slot] ?? slot;
        const view = getFftViewForPanel(panel);
        setFftPanStart({ x: e.clientX, y: e.clientY, pX: view.panX, pY: view.panY, panelIdx: panel, viewportW: dst.w, viewportH: dst.h });
        return;
      }
    }
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: fftPanX, pY: fftPanY, panelIdx: null, viewportW: canvas.width, viewportH: canvas.height });
  };

  const handleFftMouseMove = (e: React.MouseEvent) => {
    if (isFftDragging && fftPanStart) {
      const canvas = fftCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - fftPanStart.x) * scaleX;
      const dy = (e.clientY - fftPanStart.y) * scaleY;
      const view = fftPanStart.panelIdx != null && !linkPanels
        ? getFftViewForPanel(fftPanStart.panelIdx)
        : { zoom: fftZoom, panX: fftPanX, panY: fftPanY };
      const clamped = clampFftPan(fftPanStart.pX + dx, fftPanStart.pY + dy, view.zoom, fftPanStart.viewportW, fftPanStart.viewportH);
      if (fftPanStart.panelIdx != null && !linkPanels) {
        setFftViewForPanel(fftPanStart.panelIdx, { zoom: view.zoom, panX: clamped.panX, panY: clamped.panY });
      } else {
        fftUserAdjustedViewRef.current = true;
        setFftPanX(clamped.panX);
        setFftPanY(clamped.panY);
      }
    }
  };

  const handleFftMouseUp = async (e: React.MouseEvent) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const pos = fftScreenToImg(e);
        if (pos) {
          // Use crop dimensions when ROI FFT is active
          const fftW = fftCropDims?.fftWidth ?? width;
          const fftH = fftCropDims?.fftHeight ?? height;
          const panelGrid = fftPanelGridRef.current;
          let imgCol = pos.col;
          let imgRow = pos.row;
          if (fftMagCacheRef.current) {
            const bounds = panelGrid ? (() => {
              const panelCol = Math.max(0, Math.min(panelGrid.cols - 1, Math.floor(imgCol / panelGrid.panelWidth)));
              const panelRow = Math.max(0, Math.min(panelGrid.rows - 1, Math.floor(imgRow / panelGrid.panelHeight)));
              return {
                minCol: panelCol * panelGrid.panelWidth,
                maxCol: Math.min(fftW - 1, (panelCol + 1) * panelGrid.panelWidth - 1),
                minRow: panelRow * panelGrid.panelHeight,
                maxRow: Math.min(fftH - 1, (panelRow + 1) * panelGrid.panelHeight - 1),
              };
            })() : null;
            const snapped = bounds
              ? findFFTPeakInBounds(fftMagCacheRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS, bounds.minCol, bounds.maxCol, bounds.minRow, bounds.maxRow)
              : await findFFTPeakBrowser(fftMagCacheRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
            imgCol = snapped.col;
            imgRow = snapped.row;
          }
          const local = panelGrid ? (() => {
            const panelCol = Math.max(0, Math.min(panelGrid.cols - 1, Math.floor(imgCol / panelGrid.panelWidth)));
            const panelRow = Math.max(0, Math.min(panelGrid.rows - 1, Math.floor(imgRow / panelGrid.panelHeight)));
            return {
              col: imgCol - panelCol * panelGrid.panelWidth,
              row: imgRow - panelRow * panelGrid.panelHeight,
              width: panelGrid.panelWidth,
              height: panelGrid.panelHeight,
            };
          })() : { col: imgCol, row: imgRow, width: fftW, height: fftH };
          const halfW = Math.floor(local.width / 2);
          const halfH = Math.floor(local.height / 2);
          const dcol = local.col - halfW;
          const drow = local.row - halfH;
          const distPx = Math.sqrt(dcol * dcol + drow * drow);
          if (distPx < 1) {
            setFftClickInfo(null);
          } else {
            let spatialFreq: number | null = null;
            let dSpacing: number | null = null;
            if (pixelSize > 0) {
              const paddedW = nextPow2(local.width);
              const paddedH = nextPow2(local.height);
              ({ spatialFrequency: spatialFreq, dSpacing } = reciprocalCoordinatesFromShiftedOffset(
                Math.round(local.row) - halfH,
                Math.round(local.col) - halfW,
                paddedH,
                paddedW,
                pixelSize,
                pixelSize,
              ));
            }
            setFftClickInfo({ row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
          }
        }
      }
      fftClickStartRef.current = null;
    }
    setIsFftDragging(false);
    setFftPanStart(null);
  };

  const handleFftTouchStart = (e: React.TouchEvent) => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return;
    const now = Date.now();
    const base = fftViewLiveRef.current;
    if (e.touches.length === 1) {
      const lastTap = lastFftTapRef.current;
      if (lastTap && now - lastTap.time < 320) {
        e.preventDefault();
        handleFftReset();
        lastFftTapRef.current = null;
        fftTouchTransformRef.current = null;
        return;
      }
      lastFftTapRef.current = { time: now };
      const touch = e.touches[0];
      fftTouchTransformRef.current = {
        mode: "pan",
        startX: touch.clientX,
        startY: touch.clientY,
        startDistance: 0,
        startMidX: touch.clientX,
        startMidY: touch.clientY,
        startState: base,
      };
      e.preventDefault();
      return;
    }
    if (e.touches.length >= 2) {
      const firstTouch = e.touches[0];
      const secondTouch = e.touches[1];
      const mid = touchMidpoint(firstTouch, secondTouch);
      fftTouchTransformRef.current = {
        mode: "pinch",
        startX: mid.x,
        startY: mid.y,
        startDistance: Math.max(1, touchDistance(firstTouch, secondTouch)),
        startMidX: mid.x,
        startMidY: mid.y,
        startState: base,
      };
      e.preventDefault();
    }
  };

  const handleFftTouchMove = (e: React.TouchEvent) => {
    const start = fftTouchTransformRef.current;
    const canvas = fftCanvasRef.current;
    if (!start || !canvas) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const toCanvas = (clientX: number, clientY: number) => ({
      x: (clientX - rect.left) * (canvas.width / Math.max(1, rect.width)),
      y: (clientY - rect.top) * (canvas.height / Math.max(1, rect.height)),
    });
    const base = start.startState;
    if (start.mode === "pinch" && e.touches.length >= 2) {
      const firstTouch = e.touches[0];
      const secondTouch = e.touches[1];
      const mid = touchMidpoint(firstTouch, secondTouch);
      const startCanvas = toCanvas(start.startMidX, start.startMidY);
      const currentCanvas = toCanvas(mid.x, mid.y);
      const imageX = (startCanvas.x - base.panX) / base.zoom;
      const imageY = (startCanvas.y - base.panY) / base.zoom;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base.zoom * (touchDistance(firstTouch, secondTouch) / start.startDistance)));
      fftUserAdjustedViewRef.current = true;
      scheduleFftViewState({
        zoom: newZoom,
        panX: currentCanvas.x - imageX * newZoom,
        panY: currentCanvas.y - imageY * newZoom,
      }, true);
      return;
    }
    if (start.mode === "pan" && e.touches.length === 1) {
      const touch = e.touches[0];
      const scaleX = canvas.width / Math.max(1, rect.width);
      const scaleY = canvas.height / Math.max(1, rect.height);
      fftUserAdjustedViewRef.current = true;
      scheduleFftViewState({
        zoom: base.zoom,
        panX: base.panX + (touch.clientX - start.startX) * scaleX,
        panY: base.panY + (touch.clientY - start.startY) * scaleY,
      });
    }
  };

  const handleFftTouchEnd = (e: React.TouchEvent) => {
    if (e.touches.length > 0 || !fftTouchTransformRef.current) return;
    fftTouchTransformRef.current = null;
  };

  const handleFftReset = () => {
    const reset = { zoom: 1, panX: 0, panY: 0 };
    fftViewLiveRef.current = reset;
    fftUserAdjustedViewRef.current = false;
    setFftZoom(reset.zoom);
    internalFftZoomSyncRef.current = true;
    setFftOverlayZoomTrait(1);
    setFftPanX(reset.panX);
    setFftPanY(reset.panY);
    setPanelFftStates(new Map());
    setFftClickInfo(null);
  };

  // Kymograph mouse handlers (mirror FFT: wheel-zoom + pan-drag). Click readout
  // replaces the FFT d-spacing measurement (domain adaptation).
  const [isKymoDragging, setIsKymoDragging] = React.useState(false);
  const [kymoPanStart, setKymoPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  // Kymograph wheel zoom must not scroll the notebook. React registers onWheel
  // as a passive listener, where preventDefault is ignored, so the kymograph
  // takes a native non-passive listener instead.
  kymoWheelHandlerRef.current = (e: WheelEvent) => {
    e.preventDefault();
    const canvas = kymoCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, kymoZoom * zoomFactor));
    const zoomRatio = newZoom / kymoZoom;
    setKymoZoom(newZoom);
    setKymoPanX(mouseX - (mouseX - kymoPanX) * zoomRatio);
    setKymoPanY(mouseY - (mouseY - kymoPanY) * zoomRatio);
  };

  React.useEffect(() => {
    const el = kymoContainerRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => kymoWheelHandlerRef.current?.(event);
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [kymoReady]);

  // Convert kymograph canvas mouse position to (frame index, distance index).
  const kymoScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = kymoCanvasRef.current;
    const kymo = kymoDataRef.current;
    if (!canvas || !kymo) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;
    // The click already passed the canvas hit-test, so map into the image and
    // clamp - edge/last-row clicks must still yield a readout (a strict
    // `< nFrames` check silently dropped clicks on the bottom row).
    const imgCol = Math.max(0, Math.min(kymo.lineLen - 1, ((mouseX - kymoPanX) / kymoZoom) / canvasW * kymo.lineLen));
    const imgRow = Math.max(0, Math.min(kymo.nFrames - 1, ((mouseY - kymoPanY) / kymoZoom) / canvasH * kymo.nFrames));
    return { col: imgCol, row: imgRow };
  };

  const handleKymoMouseDown = (e: React.MouseEvent) => {
    kymoClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsKymoDragging(true);
    setKymoPanStart({ x: e.clientX, y: e.clientY, pX: kymoPanX, pY: kymoPanY });
  };

  const handleKymoMouseMove = (e: React.MouseEvent) => {
    if (isKymoDragging && kymoPanStart) {
      const canvas = kymoCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - kymoPanStart.x) * scaleX;
      const dy = (e.clientY - kymoPanStart.y) * scaleY;
      setKymoPanX(kymoPanStart.pX + dx);
      setKymoPanY(kymoPanStart.pY + dy);
    }
  };

  const handleKymoMouseUp = (e: React.MouseEvent) => {
    // Click detection for intensity readout at (time, distance).
    if (kymoClickStartRef.current) {
      const dx = e.clientX - kymoClickStartRef.current.x;
      const dy = e.clientY - kymoClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const pos = kymoScreenToImg(e);
        const kymo = kymoDataRef.current;
        if (pos && kymo) {
          const frame = Math.max(0, Math.min(kymo.nFrames - 1, Math.round(pos.row)));
          const dist = Math.max(0, Math.min(kymo.lineLen - 1, Math.round(pos.col)));
          const intensity = kymo.data[frame * kymo.lineLen + dist];
          const timeVal = dimSampling > 0 && dimUnit ? frame * dimSampling : frame;
          const timeUnit = dimSampling > 0 && dimUnit ? unitSymbol(dimUnit) : "frame";
          const distVal = pixelSize > 0 ? dist * pixelSize : dist;
          const distUnit = pixelSize > 0 ? unitSymbol(pixelUnit || "px") : "px";
          setKymoClickInfo({ timeVal, timeUnit, distVal, distUnit, intensity, col: dist, row: frame });
        } else {
          setKymoClickInfo(null);
        }
      }
      kymoClickStartRef.current = null;
    }
    setIsKymoDragging(false);
    setKymoPanStart(null);
  };

  const handleKymoTouchStart = (e: React.TouchEvent) => {
    const canvas = kymoCanvasRef.current;
    if (!canvas) return;
    const now = Date.now();
    const base = { zoom: kymoZoom, panX: kymoPanX, panY: kymoPanY };
    if (e.touches.length === 1) {
      const lastTap = lastKymoTapRef.current;
      if (lastTap && now - lastTap.time < 320) {
        e.preventDefault();
        handleKymoReset();
        lastKymoTapRef.current = null;
        kymoTouchTransformRef.current = null;
        return;
      }
      lastKymoTapRef.current = { time: now };
      const touch = e.touches[0];
      kymoTouchTransformRef.current = {
        mode: "pan",
        startX: touch.clientX,
        startY: touch.clientY,
        startDistance: 0,
        startMidX: touch.clientX,
        startMidY: touch.clientY,
        startState: base,
      };
      e.preventDefault();
      return;
    }
    if (e.touches.length >= 2) {
      const firstTouch = e.touches[0];
      const secondTouch = e.touches[1];
      const mid = touchMidpoint(firstTouch, secondTouch);
      kymoTouchTransformRef.current = {
        mode: "pinch",
        startX: mid.x,
        startY: mid.y,
        startDistance: Math.max(1, touchDistance(firstTouch, secondTouch)),
        startMidX: mid.x,
        startMidY: mid.y,
        startState: base,
      };
      e.preventDefault();
    }
  };

  const handleKymoTouchMove = (e: React.TouchEvent) => {
    const start = kymoTouchTransformRef.current;
    const canvas = kymoCanvasRef.current;
    if (!start || !canvas) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const toCanvas = (clientX: number, clientY: number) => ({
      x: (clientX - rect.left) * (canvas.width / Math.max(1, rect.width)),
      y: (clientY - rect.top) * (canvas.height / Math.max(1, rect.height)),
    });
    const base = start.startState;
    if (start.mode === "pinch" && e.touches.length >= 2) {
      const firstTouch = e.touches[0];
      const secondTouch = e.touches[1];
      const mid = touchMidpoint(firstTouch, secondTouch);
      const startCanvas = toCanvas(start.startMidX, start.startMidY);
      const currentCanvas = toCanvas(mid.x, mid.y);
      const imageX = (startCanvas.x - base.panX) / base.zoom;
      const imageY = (startCanvas.y - base.panY) / base.zoom;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base.zoom * (touchDistance(firstTouch, secondTouch) / start.startDistance)));
      setKymoZoom(newZoom);
      setKymoPanX(currentCanvas.x - imageX * newZoom);
      setKymoPanY(currentCanvas.y - imageY * newZoom);
      return;
    }
    if (start.mode === "pan" && e.touches.length === 1) {
      const touch = e.touches[0];
      const scaleX = canvas.width / Math.max(1, rect.width);
      const scaleY = canvas.height / Math.max(1, rect.height);
      setKymoPanX(base.panX + (touch.clientX - start.startX) * scaleX);
      setKymoPanY(base.panY + (touch.clientY - start.startY) * scaleY);
    }
  };

  const handleKymoTouchEnd = (e: React.TouchEvent) => {
    if (e.touches.length > 0 || !kymoTouchTransformRef.current) return;
    kymoTouchTransformRef.current = null;
  };

  const handleKymoReset = () => {
    setKymoZoom(1);
    setKymoPanX(0);
    setKymoPanY(0);
    setKymoClickInfo(null);
  };

  const kymoNeedsReset = kymoZoom !== 1 || kymoPanX !== 0 || kymoPanY !== 0;

  // Preview panel zoom/pan handlers
  const handlePreviewWheel = (e: React.WheelEvent) => {
    const canvas = previewCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const previewWidth = previewCanvasDims.w;
    const previewHeight = previewCanvasDims.h;
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = previewWidth / 2;
    const cy = previewHeight / 2;
    const mouseImageX = (mouseCanvasX - cx - previewZoom.panX) / previewZoom.zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - previewZoom.panY) / previewZoom.zoom + cy;
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, previewZoom.zoom * zoomFactor));
    const newPanX = mouseCanvasX - (mouseImageX - cx) * newZoom - cx;
    const newPanY = mouseCanvasY - (mouseImageY - cy) * newZoom - cy;
    setPreviewZoom({ zoom: newZoom, panX: newPanX, panY: newPanY });
  };

  const handlePreviewMouseDown = (e: React.MouseEvent) => {
    setIsDraggingPreviewPan(true);
    setPreviewPanStart({ x: e.clientX, y: e.clientY, pX: previewZoom.panX, pY: previewZoom.panY });
  };

  const handlePreviewMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingPreviewPan || !previewPanStart) return;
    const canvas = previewCanvasRef.current;
    const rect = canvas?.getBoundingClientRect();
    const scaleX = canvas && rect ? canvas.width / Math.max(1, rect.width) : 1;
    const scaleY = canvas && rect ? canvas.height / Math.max(1, rect.height) : 1;
    const dx = (e.clientX - previewPanStart.x) * scaleX;
    const dy = (e.clientY - previewPanStart.y) * scaleY;
    setPreviewZoom(prev => ({ ...prev, panX: previewPanStart.pX + dx, panY: previewPanStart.pY + dy }));
  };

  const handlePreviewMouseUp = () => {
    setIsDraggingPreviewPan(false);
    setPreviewPanStart(null);
  };

  const handlePreviewDoubleClick = () => {
    setPreviewZoom({ zoom: 1, panX: 0, panY: 0 });
  };

  // Resize handlers
  const handleMainResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    const rect = canvasContainerRef.current?.getBoundingClientRect();
    const startSize = rect && rect.width > 0 ? rect.width : mainCanvasSize;
    const startX = e.clientX;
    const startY = e.clientY;
    const visiblePanels = Math.max(1, visiblePanelCount || 1);
    let rafId = 0;
    let latestSize = startSize;
    const handleMouseMove = (e: MouseEvent) => {
      const delta = Math.max(e.clientX - startX, e.clientY - startY);
      const nextSize = startSize + delta;
      // Absolute minimum: 200 px per panel column. Lets reader shrink BELOW
      // the initial `size=` value (preset / kwarg) when their screen is small,
      // without collapsing the canvas to an unreadable sliver.
      const colsLocal = panelColsForCount(visiblePanels);
      const minSize = 200 * colsLocal;
      latestSize = Math.max(minSize, nextSize);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setMainCanvasSize(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      cancelAnimationFrame(rafId);
      setMainCanvasSize(latestSize);
      const colsLocal = panelColsForCount(visiblePanels);
      setCanvasSizeTrait(Math.round(latestSize / colsLocal));
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const clampSlice = (idx: number) => Math.max(0, Math.min(nSlices - 1, Math.round(idx)));
  const frameLabelForIndex = React.useCallback((idx: number): string => {
    const label = labels?.[idx];
    if (label == null) return "";
    const text = String(label).trim();
    if (!text || text === String(idx) || text === String(idx + 1)) return "";
    return text;
  }, [labels]);
  const panelFrameLabelForIndex = React.useCallback((_panel: number, idx: number): string => frameLabelForIndex(idx), [frameLabelForIndex]);
  const formatFrameValueLabel = React.useCallback((idx: number) => {
    const rounded = clampSlice(idx);
    const label = frameLabelForIndex(rounded);
    return label ? `${rounded + 1}: ${label}` : `${rounded + 1}`;
  }, [frameLabelForIndex, nSlices]);
  const visibleSliceIdx = clampSlice(playing ? playbackUiSliceIdx : liveSliceIdx);
  React.useLayoutEffect(() => {
    updatePlaybackLiveControls(visibleSliceIdx);
  }, [updatePlaybackLiveControls, visibleSliceIdx]);
  const normalizedBookmarkedFrames = React.useMemo(() => {
    const seen = new Set<number>();
    for (const raw of bookmarkedFrames || []) {
      const value = Math.round(Number(raw));
      if (Number.isFinite(value) && value >= 0 && value < nSlices) seen.add(value);
    }
    return Array.from(seen).sort((a, b) => a - b);
  }, [bookmarkedFrames, nSlices]);
  const bookmarkedFrameMarks = React.useMemo(
    () => normalizedBookmarkedFrames.map((value) => ({ value })),
    [normalizedBookmarkedFrames]
  );
  const currentFrameBookmarked = normalizedBookmarkedFrames.includes(visibleSliceIdx);
  const toggleCurrentFrameBookmark = React.useCallback(() => {
    const frame = visibleSliceIdx;
    const next = new Set(normalizedBookmarkedFrames);
    if (next.has(frame)) next.delete(frame);
    else next.add(frame);
    setBookmarkedFrames(Array.from(next).sort((a, b) => a - b));
  }, [normalizedBookmarkedFrames, setBookmarkedFrames, visibleSliceIdx]);
  const currentPlaybackIndex = () => (
    Number.isFinite(playbackIdxRef.current)
      ? playbackIdxRef.current
      : (Number.isFinite(displaySliceIdx) ? displaySliceIdx : sliceIdx)
  );
  const playFromCurrentFrame = (direction: 1 | -1 | null = null) => {
    if (sliceCommitTimerRef.current !== null) {
      window.clearTimeout(sliceCommitTimerRef.current);
      sliceCommitTimerRef.current = null;
    }
    const nextReverse = direction === null ? reverse : direction < 0;
    const rangeStart = loop ? Math.max(0, Math.min(loopStart, nSlices - 1)) : 0;
    const rangeEnd = loop ? Math.max(rangeStart, Math.min(effectiveLoopEnd, nSlices - 1)) : nSlices - 1;
    let start = Math.max(rangeStart, Math.min(rangeEnd, Math.round(currentPlaybackIndex())));
    if (!loop) {
      if (!nextReverse && start >= rangeEnd) start = rangeStart;
      if (nextReverse && start <= rangeStart) start = rangeEnd;
    }
    playbackIdxRef.current = start;
    setDisplaySliceIdx(start);
    setPlaybackUiSliceIdx(start);
    setLiveSliceIdx(start);
    setSliceIdx(start);
    if (direction !== null) setReverse(nextReverse);
    setPlaying(true);
  };
  const pausePlayback = () => {
    if (sliceCommitTimerRef.current !== null) {
      window.clearTimeout(sliceCommitTimerRef.current);
      sliceCommitTimerRef.current = null;
    }
    const current = clampSlice(currentPlaybackIndex());
    playbackIdxRef.current = current;
    setDisplaySliceIdx(current);
    setPlaybackUiSliceIdx(current);
    setLiveSliceIdx(current);
    setSliceIdx(current);
    setPlaying(false);
  };
  const stopPlayback = () => {
    if (sliceCommitTimerRef.current !== null) {
      window.clearTimeout(sliceCommitTimerRef.current);
      sliceCommitTimerRef.current = null;
    }
    const home = loop ? Math.max(0, Math.min(loopStart, nSlices - 1)) : 0;
    playbackIdxRef.current = home;
    setDisplaySliceIdx(home);
    setPlaybackUiSliceIdx(home);
    setLiveSliceIdx(home);
    setSliceIdx(home);
    setPlaying(false);
  };
  const playbackPathLength = Array.isArray(playbackPath) ? playbackPath.length : 0;
  const playbackStyleSummary = playbackPathLength > 0 ? `Path ${playbackPathLength}` : "Linear";
  const clampFrameIndex = React.useCallback(
    (value: number) => Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(value))),
    [nSlices],
  );
  const makePlaybackStylePath = React.useCallback((style: "power-in" | "power-out" | "ease-in-out") => {
    const start = loop ? Math.max(0, Math.min(loopStart, nSlices - 1)) : 0;
    const loopEndCandidate = loop ? Math.min(effectiveLoopEnd, nSlices - 1) : Math.max(0, nSlices - 1);
    // A one-frame loop is meaningful for manual review but not for choosing a
    // temporal curve. If the range is still hydrating or collapsed, style
    // buttons fall back to the full stack instead of producing "Path 1".
    const end = loopEndCandidate > start ? loopEndCandidate : Math.max(start, nSlices - 1);
    const span = Math.max(0, end - start);
    if (span <= 0) return [start];
    const steps = Math.max(span + 1, Math.min(96, Math.round((span + 1) * 1.75)));
    const path: number[] = [];
    for (let i = 0; i < steps; i++) {
      const t = steps <= 1 ? 1 : i / (steps - 1);
      let eased = t;
      if (style === "power-in") eased = t * t;
      else if (style === "power-out") eased = 1 - ((1 - t) * (1 - t));
      else eased = 0.5 - 0.5 * Math.cos(Math.PI * t);
      path.push(clampFrameIndex(start + eased * span));
    }
    if (path[0] !== start) path.unshift(start);
    if (path[path.length - 1] !== end) path.push(end);
    return path;
  }, [clampFrameIndex, effectiveLoopEnd, loop, loopStart, nSlices]);
  const applyPlaybackStylePreset = React.useCallback((style: "linear" | "power-in" | "power-out" | "ease-in-out") => {
    if (style === "linear") {
      setPlaybackPath([]);
    } else {
      setPlaybackPath(makePlaybackStylePath(style));
    }
    setPlaying(false);
    setPlaybackStyleMenuAnchor(null);
  }, [makePlaybackStylePath, setPlaybackPath, setPlaying]);
  const playbackStyleActive = React.useMemo<"linear" | "power-in" | "power-out" | "ease-in-out" | null>(() => {
    if (!playbackPathLength) return "linear";
    const samePath = (candidate: number[]) => (
      candidate.length === playbackPathLength
      && candidate.every((value, idx) => value === playbackPath[idx])
    );
    for (const style of ["power-in", "power-out", "ease-in-out"] as const) {
      if (samePath(makePlaybackStylePath(style))) return style;
    }
    return null;
  }, [makePlaybackStylePath, playbackPath, playbackPathLength]);
  const playbackStyleButtonSx = React.useCallback((style: "linear" | "power-in" | "power-out" | "ease-in-out") => {
    const active = playbackStyleActive === style;
    return {
      ...compactButton,
      justifyContent: "flex-start",
      color: active ? themeColors.accent : themeColors.textMuted,
      border: `1px solid ${active ? themeColors.accent : "transparent"}`,
      bgcolor: active ? themeColors.controlBg : "transparent",
      "&:hover": {
        color: active ? themeColors.accent : themeColors.text,
        borderColor: active ? themeColors.accent : themeColors.border,
        bgcolor: themeColors.controlBg,
      },
    };
  }, [playbackStyleActive, themeColors.accent, themeColors.border, themeColors.controlBg, themeColors.text, themeColors.textMuted]);

  // Keyboard
  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (shouldIgnoreWidgetShortcut(e.target, e.key)) return;

    let handled = false;
    const frameTarget = frameKeyTarget(e.key, clampSlice(currentPlaybackIndex()), nSlices, loop ? [loopStart, effectiveLoopEnd] : null);
    if (frameTarget !== null) {
      scrubToSlice(frameTarget);
      handled = true;
    }
    switch (e.key) {
        case " ":
          if (playing) pausePlayback();
          else playFromCurrentFrame();
          handled = true;
          break;
        case "r":
        case "R":
          handleDoubleClick();
          handled = true;
          break;
        case "c":
        case "C":
          if (cursorInfo && cursorReadoutVisible) {
            navigator.clipboard.writeText(`(${cursorInfo.row}, ${cursorInfo.col}, ${cursorInfo.value})`);
            handled = true;
          }
          break;
        case "h":
        case "H": {
          if (hasPanelChoices && selectedVisiblePanels.length > 0) {
            const hideable = selectedVisiblePanels.filter((panel) => visiblePanelIndices.includes(panel));
            if (hideable.length > 0 && visiblePanelCount - hideable.length >= 1) {
              setPanelsHidden(hideable, true);
              handled = true;
            }
          }
          break;
        }
        case "Delete":
        case "Backspace":
          if (overlayEditMode && overlaySelection) {
            deleteSelectedOverlay();
            handled = true;
          } else if (effectiveRoiActive && roiSelectedIdx >= 0) {
            deleteSelectedROI();
            handled = true;
          }
          break;
        case "d":
        case "D":
          if (effectiveRoiActive && roiSelectedIdx >= 0 && (e.metaKey || e.ctrlKey || e.shiftKey)) {
            duplicateSelectedROI();
            handled = true;
          }
          break;
        case "Escape":
          rootRef.current?.blur();
          handled = true;
          break;
      }
    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;
  const scheduleScrubModelCommit = (idx: number, delayMs = 350) => {
    const next = clampSlice(idx);
    if (sliceCommitTimerRef.current !== null) {
      window.clearTimeout(sliceCommitTimerRef.current);
    }
    sliceCommitTimerRef.current = window.setTimeout(() => {
      sliceCommitTimerRef.current = null;
      setLiveSliceIdx(next);
      setDisplaySliceIdx(next);
      setPlaybackUiSliceIdx(next);
      setSliceIdx(next);
    }, delayMs);
  };
  const scrubToSlice = (idx: number, scheduleIdleCommit = true) => {
    const next = clampSlice(idx);
    // A seek changes position, not playback intent. Pointer gestures hold the
    // frame loop until release; keyboard seeks keep the loop running.
    playbackIdxRef.current = next;
    const commitLater = scheduleIdleCommit && !playing;
    const transformActive = frameTransformActive();
    // Keep the high-frequency scrub path browser-local. React state and the
    // Jupyter trait commit once the input settles; the image and slider
    // thumb move immediately from their WebGPU/DOM caches on every sample.
    if (
      renderGpuTemporalAverageSliceDirect(next, false)
      || (!transformActive && gpuFrameCacheUploadedRef.current.has(next) && renderGpuCachedSliceDirect(next, false))
    ) {
      updatePlaybackLiveControls(next);
      if (commitLater) scheduleScrubModelCommit(next);
      return;
    }
    setPlaybackUiSliceIdx(next);
    setLiveSliceIdx(next);
    if (transformActive || !renderGpuCachedSliceDirect(next)) setDisplaySliceIdx(next);
    if (commitLater) scheduleScrubModelCommit(next);
  };
  const commitSlice = (idx: number) => {
    const next = clampSlice(idx);
    playbackIdxRef.current = next;
    if (sliceCommitTimerRef.current !== null) {
      window.clearTimeout(sliceCommitTimerRef.current);
      sliceCommitTimerRef.current = null;
    }
    setLiveSliceIdx(next);
    setDisplaySliceIdx(next);
    setPlaybackUiSliceIdx(next);
    setSliceIdx(next);
  };
  const handleLoopSliderPointerDownCapture = (e: React.PointerEvent<HTMLSpanElement>) => {
    if (e.button !== 0) return;
    sliderGestureCleanupRef.current?.();
    sliderScrubbingRef.current = true;
    if (sliceCommitTimerRef.current !== null) {
      window.clearTimeout(sliceCommitTimerRef.current);
      sliceCommitTimerRef.current = null;
    }
    const controller = new AbortController();
    let finish = (_event: Event) => {};
    let cancelPaint = () => {};
    const cleanup = () => {
      cancelPaint();
      controller.abort();
      sliderScrubbingRef.current = false;
      sliderGestureCleanupRef.current = null;
    };
    sliderGestureCleanupRef.current = cleanup;
    const endGesture = (event: Event) => {
      if (event instanceof PointerEvent && event.pointerId !== e.pointerId) return;
      try { finish(event); } finally { cleanup(); }
    };
    const options = { capture: true, signal: controller.signal };
    window.addEventListener("pointerup", endGesture, options);
    window.addEventListener("pointercancel", endGesture, options);
    window.addEventListener("blur", endGesture, options);
    const target = e.target as HTMLElement;
    const thumb = target.closest(".MuiSlider-thumb") as HTMLElement | null;
    // Loop sliders have start/current/end thumbs. Leave start/end to MUI so
    // range editing still works, but own the current-frame thumb and track
    // because anywidget trait commits can batch under rapid pointer drags.
    if (loop && thumb && thumb.getAttribute("data-index") !== "1") return;
    const rect = e.currentTarget.getBoundingClientRect();
    const lo = loop ? Math.max(0, Math.min(loopStart, nSlices - 1)) : 0;
    const hi = loop ? Math.max(lo, Math.min(effectiveLoopEnd, nSlices - 1)) : nSlices - 1;
    const sliceFromClientX = (clientX: number) => {
      const pct = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
      return Math.max(lo, Math.min(hi, clampSlice(pct * Math.max(0, nSlices - 1))));
    };
    let pendingClientX = e.clientX;
    let scrubRaf = 0;
    let lastPainted = -1;
    const paintCurrent = () => {
      scrubRaf = 0;
      const next = sliceFromClientX(pendingClientX);
      if (next === lastPainted) return;
      lastPainted = next;
      scrubToSlice(next, false);
    };
    const scheduleCurrent = (clientX: number) => {
      pendingClientX = clientX;
      if (scrubRaf) return;
      scrubRaf = window.requestAnimationFrame(paintCurrent);
    };
    const commitCurrent = (clientX: number) => {
      pendingClientX = clientX;
      if (scrubRaf) {
        window.cancelAnimationFrame(scrubRaf);
        scrubRaf = 0;
      }
      const next = sliceFromClientX(clientX);
      if (next !== lastPainted) scrubToSlice(next, false);
      commitSlice(next);
    };
    e.preventDefault();
    e.stopPropagation();
    e.nativeEvent.stopImmediatePropagation();
    paintCurrent();
    const onMove = (ev: PointerEvent) => {
      if (ev.pointerId !== e.pointerId) return;
      ev.preventDefault();
      scheduleCurrent(ev.clientX);
    };
    cancelPaint = () => { if (scrubRaf) window.cancelAnimationFrame(scrubRaf); };
    finish = (event: Event) => {
      // Cancellation or leaving the window commits the last real position;
      // neither can leave a dangling listener holding playback indefinitely.
      const clientX = event.type === "pointerup" && event instanceof PointerEvent
        ? event.clientX : pendingClientX;
      commitCurrent(clientX);
    };
    window.addEventListener("pointermove", onMove, options);
  };
  const overlayCanvasVisible = effectiveRoiActive || profileActive || (panelOverlays || []).some((items) => items && items.length > 0);
  const lensCanvasVisible = showLens && lensPos !== null;
  const keyboardShortcutItems: [string, string][] = [
    ["Space", "Play / Pause"],
    ["← / →", `Prev / Next ${dimLabel.toLowerCase()}`],
    ["Home / End", `First / Last ${dimLabel.toLowerCase()}`],
    ["R", "Reset zoom"],
    ["C", "Copy cursor coords"],
    ...(hasPanelChoices ? [["Shift-click", "Select panel range"], ["Ctrl/⌘-click", "Toggle panel selection"], ["H", "Hide selected panels"]] as [string, string][] : []),
    ...(roiAllowed ? [["Del", "Delete selected ROI"], ["Ctrl/⌘+D", "Duplicate selected ROI"]] as [string, string][] : []),
    ["Esc", "Release keyboard focus"],
    ["Scroll", "Zoom"],
    ["Dbl-click", "Reset view"],
  ];
  const webgpuStatusLabel =
    fftBackendInfo.webgpu === "ready" ? "available"
      : fftBackendInfo.webgpu === "unavailable" ? "unavailable"
        : "checking";
  const fftSourceRaw = fftBackendInfo.source || "";
  const fftSourceCached = fftSourceRaw.endsWith("-cache");
  const fftSourceBase = fftSourceCached ? fftSourceRaw.slice(0, -6) : fftSourceRaw;
  const fftSourceLabel =
    fftSourceCached ? "Cached"
      : fftSourceBase.startsWith("webgpu") ? "WebGPU"
        : fftSourceBase ? "CPU"
        : "not run yet";
  const fftSourceDetail = fftSourceBase;
  const resolvedDisplayBin = Math.max(1, displayBin || 1);
  const displayStackBytes = offlineFloatStack?.byteLength || offlineStack?.byteLength || 0;
  const displayPayloadLabel = offlineFloatStack?.byteLength
    ? "float32"
    : offlineStack?.byteLength
      ? "uint8 (decoded once for WebGPU)"
      : "float32";
  const displayBinLabel = resolvedDisplayBin === 1 ? "native" : `${resolvedDisplayBin}× mean-binned`;
  const gpuStatusText = gpuResidency.stage === "uploading"
    ? `Uploading ${displayBinLabel} ${displayPayloadLabel} display to WebGPU · ${gpuResidency.ready}/${Math.max(1, nSlices)} frames`
    : gpuResidency.stage === "ready"
      ? `${displayBinLabel} ${displayPayloadLabel} display · WebGPU resident · ${gpuResidency.ready}/${Math.max(1, nSlices)} frames`
      : gpuResidency.stage === "fallback"
        ? `WebGPU unavailable · using CPU/canvas${gpuResidency.error ? `: ${gpuResidency.error}` : ""}`
        : gpuResidency.stage === "missing"
          ? gpuResidency.error
          : gpuResidency.stage === "rgb"
            ? `${displayBinLabel} true-color display · drawn on the CPU canvas`
            : "Loading one display-resolution stack into the browser";
  const sourceBytes = Math.max(1, nSlices) * (sharedPanelSource ? 1 : Math.max(1, nPanels)) * Math.max(1, sourceHeight || height) * Math.max(1, nativeSourcePanelWidth || panelWidthPx || width) * 4;
  const gpuDetailText = `Original: ${Math.max(1, nSlices)} frames × ${Math.max(1, nPanels)} panels × ${Math.max(1, sourceHeight || height)}×${Math.max(1, nativeSourcePanelWidth || panelWidthPx || width)} · float32 · ${formatSavedBytes(sourceBytes || 0)}. ${resolvedDisplayBin === 1 ? "Native display requested" : `${resolvedDisplayBin}×${resolvedDisplayBin} mean bin`} → ${Math.max(1, height)}×${Math.max(1, panelWidthPx || Math.round(width / Math.max(1, nPanels)))} per panel · ${displayStackBytes > 0 ? formatSavedBytes(displayStackBytes) : "not received"}.`;
  const gpuStatusTitle = "One display-resolution stack is embedded in the widget and uploaded once to WebGPU when available; native source arrays are not duplicated in the browser.";
  const frequencyRingValue = normalizeFrequencyFilterMode(frequencyFilter) === "bandpass"
    ? (frequencyDraft ?? frequencyFilterCenter)
    : (frequencyDraft ?? frequencyFilterCutoff);
  const show3dFrequencyRing = frequencyFilterIsActive ? (
    <Box
      className="quantem-frequency-filter-ring"
      data-frequency-filter={normalizeFrequencyFilterMode(frequencyFilter)}
      aria-label={`Draggable ${normalizeFrequencyFilterMode(frequencyFilter)} frequency ring at ${frequencyValueLabel(frequencyRingValue)}`}
      title="Drag the ring to choose a frequency from the FFT"
      onMouseDown={(event: React.MouseEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.stopPropagation();
        const parent = event.currentTarget.parentElement;
        if (!parent) return;
        const rect = parent.getBoundingClientRect();
        const valueAt = (clientX: number, clientY: number) => Math.max(0, Math.min(1, Math.hypot(clientX - (rect.left + rect.width / 2), clientY - (rect.top + rect.height / 2)) / (Math.min(rect.width, rect.height) / 2)));
        const onMove = (moveEvent: MouseEvent) => setFrequencyDraft(valueAt(moveEvent.clientX, moveEvent.clientY));
        const onUp = (upEvent: MouseEvent) => {
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
          const value = valueAt(upEvent.clientX, upEvent.clientY);
          if (normalizeFrequencyFilterMode(frequencyFilter) === "bandpass") setFrequencyFilterCenter(value);
          else setFrequencyFilterCutoff(value);
          setFrequencyDraft(null);
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
      }}
      onPointerDown={(event: React.PointerEvent<HTMLDivElement>) => {
        event.stopPropagation();
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event: React.PointerEvent<HTMLDivElement>) => {
        if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
        const parent = event.currentTarget.parentElement;
        if (!parent) return;
        const rect = parent.getBoundingClientRect();
        setFrequencyDraft(Math.max(0, Math.min(1, Math.hypot(event.clientX - (rect.left + rect.width / 2), event.clientY - (rect.top + rect.height / 2)) / (Math.min(rect.width, rect.height) / 2))));
      }}
      onPointerUp={(event: React.PointerEvent<HTMLDivElement>) => {
        event.stopPropagation();
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        if (normalizeFrequencyFilterMode(frequencyFilter) === "bandpass") setFrequencyFilterCenter(frequencyRingValue);
        else setFrequencyFilterCutoff(frequencyRingValue);
        setFrequencyDraft(null);
      }}
      sx={{ position: "absolute", left: "50%", top: "50%", width: `${frequencyRingValue * 100}%`, height: `${frequencyRingValue * 100}%`, transform: "translate(-50%, -50%)", borderRadius: "50%", border: "2px solid rgba(0,229,255,0.95)", bgcolor: normalizeFrequencyFilterMode(frequencyFilter) === "highpass" ? "rgba(0,0,0,0.55)" : "transparent", boxShadow: normalizeFrequencyFilterMode(frequencyFilter) === "lowpass" ? "0 0 0 1px rgba(0,0,0,0.75), 0 0 0 9999px rgba(0,0,0,0.55)" : "0 0 0 1px rgba(0,0,0,0.75)", cursor: "crosshair", touchAction: "none", zIndex: 6 }}
    >
      {normalizeFrequencyFilterMode(frequencyFilter) === "bandpass" && [
        Math.max(0, frequencyFilterCenter - frequencyFilterWidth / 2),
        Math.min(1, frequencyFilterCenter + frequencyFilterWidth / 2),
      ].map((radius, index) => <Box key={index} sx={{ position: "absolute", left: "50%", top: "50%", width: `${radius / Math.max(0.001, frequencyRingValue) * 100}%`, height: `${radius / Math.max(0.001, frequencyRingValue) * 100}%`, transform: "translate(-50%, -50%)", borderRadius: "50%", border: "1px dashed rgba(255,255,255,0.95)", bgcolor: index === 0 ? "rgba(0,0,0,0.55)" : "transparent", boxShadow: index === 1 ? "0 0 0 9999px rgba(0,0,0,0.55)" : "none", pointerEvents: "none" }} />)}
      <Box sx={{ position: "absolute", left: "50%", top: -24, transform: "translateX(-50%)", px: 0.75, py: 0.25, borderRadius: 0.75, bgcolor: "rgba(0,0,0,0.78)", color: "rgba(200,250,255,0.98)", fontSize: 9, lineHeight: 1.2, fontWeight: 700, whiteSpace: "nowrap", pointerEvents: "none", textShadow: "0 1px 1px #000" }}>
        {normalizeFrequencyFilterMode(frequencyFilter) === "lowpass" ? "Inside kept" : normalizeFrequencyFilterMode(frequencyFilter) === "highpass" ? "Outside kept" : "Band kept"}
      </Box>
    </Box>
  ) : null;
  return (
    <Box
      ref={rootRef}
      className="show3d-root"
      data-show3d-canvas-repaint-signal={canvasRepaintSignal}
      data-frequency-filter-backend={frequencyFilterIsActive ? frequencyFilterBackend : "off"}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
      sx={{ ...container.root, width: "100%", maxWidth: "100%", boxSizing: "border-box", position: "relative", bgcolor: themeColors.bg, color: themeColors.text, outline: "none", "&:focus::after": { content: '""', position: "absolute", inset: 0, pointerEvents: "none", zIndex: 20, boxShadow: "inset 0 0 0 2px #0af" }, "& canvas": { display: "block" }, "@media (max-width: 700px)": { p: 0, ".jp-OutputArea-output &, .jp-OutputArea-child &": { width: "calc(100vw - 96px)", maxWidth: "calc(100vw - 96px)" } } }}
    >
      <FolderWatchBadge
        state={folderWatchState}
        detail={folderWatchDetail}
        live={folderWatchLive}
      />
      {gpuStatusText && (
        <Box
          role="status"
          aria-live="polite"
          data-show3d-gpu-residency={gpuResidency.stage}
          title={gpuStatusTitle}
          sx={{
            width: "100%",
            px: 1.5,
            py: 0.75,
            mb: 1,
            boxSizing: "border-box",
            borderRadius: 1,
            bgcolor: themeColors.controlBg,
            border: `1px solid ${gpuResidency.stage === "ready" ? themeColors.accent : themeColors.border}`,
            // CPU canvas is a supported display path, so every stage uses the normal status text color.
            color: themeColors.text,
          }}
        >
          <Typography sx={{ fontSize: 12, fontWeight: 600 }}>{gpuStatusText}</Typography>
          <Typography sx={{ fontSize: 11, mt: 0.25, color: themeColors.textMuted }}>{gpuDetailText}</Typography>
        </Box>
      )}
      {folderWaiting && (
        <Box
          role="region"
          aria-label="Show3D folder waiting view"
          data-show3d-folder-waiting="true"
          sx={{
            width: "100%",
            minHeight: 120,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            px: 2,
            py: 3,
            boxSizing: "border-box",
            border: `1px dashed ${themeColors.border}`,
            borderRadius: 1,
            color: themeColors.textMuted,
          }}
        >
          <Typography sx={{ fontSize: 12, textAlign: "center" }}>
            {folderStatus || "Waiting for the first stable frame"}
          </Typography>
        </Box>
      )}
      {!folderWaiting && !canRenderLive && hasSavedStaticFallback && (
        <Box sx={{ width: "100%", maxWidth: mainPanelWidth, boxSizing: "border-box" }}>
          <Box
            component="img"
            src={staticFallbackUrl}
            alt={`${title || "Show3D"} saved preview`}
            sx={{
              display: "block",
              width: "100%",
              maxWidth: mainPanelWidth,
              height: "auto",
              border: `1px solid ${themeColors.border}`,
              boxSizing: "border-box",
            }}
          />
        </Box>
      )}
      {!folderWaiting && (canRenderLive || !hasSavedStaticFallback) && (
      <>
      <Stack
        direction="row"
        spacing={`${SPACING.SM}px`}
        alignItems="flex-start"
        sx={{
          flexWrap: effectiveShowFft && fftLayoutBottom ? "wrap" : "nowrap",
          width: "100%",
          maxWidth: "100%",
          minWidth: 0,
          boxSizing: "border-box",
          "@media (max-width: 900px)": {
            flexDirection: "column",
            alignItems: "stretch",
            flexWrap: "nowrap",
            "& > :not(style) + :not(style)": {
              marginLeft: "0 !important",
              marginTop: `${SPACING.SM}px`,
            },
          },
        }}
      >
        <Box sx={{ width: mainPanelWidth, maxWidth: "100%", flexShrink: effectiveShowFft && fftLayoutBottom ? 0 : 1, boxSizing: "border-box" }}>
          {/* Title row */}
          {showTitle && <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block", height: 16, lineHeight: "16px", overflow: "hidden" }}>
            {title || "Image"}
            <RenderPathBadge colors={themeColors} />
            {diffMode !== "off" && (
              <Typography component="span" sx={{ fontSize: 9, fontWeight: "bold", color: "#fff", bgcolor: "#e65100", px: 0.5, py: 0.125, ml: 0.5, verticalAlign: "middle" }}>
                {diffMode === "previous" ? "\u0394-PREV" : "\u0394-FIRST"}
              </Typography>
            )}
	            {showControls && <InfoTooltip maxWidth={360} text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <MetadataSection rows={[
                ["Shape", `${nSlices} x ${height} x ${width}`],
                ["Panels", nPanels > 1 ? `${nPanels} panels` : "single panel"],
                ["Frame axis", `${dimLabel || "Frame"}${dimSampling ? `, ${formatNumber(dimSampling)} ${dimUnit || ""}` : ""}`],
                ["Sampling", pixelSize > 0 ? `${formatNumber(pixelSize)} ${unitSymbol(pixelUnit || "px")}/px` : ""],
                ["Source", "embedded stack"],
              ]} />
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum (Fourier transform) alongside image.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                FFT d-spacing uses the provided real-space sampling: Δk = 1 / (N × pixel_size), |g| = √(kx² + ky²), d = 1 / |g|. Current pixel_size: {pixelSize > 0 ? `${formatNumber(pixelSize)} ${unitSymbol(pixelUnit || "px")}/px` : "not set, so only pixel distances are shown"}.
              </Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Backend</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                WebGPU: {webgpuStatusLabel}{fftBackendInfo.adapter ? ` (${fftBackendInfo.adapter})` : ""}.
              </Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                FFT compute: {fftSourceLabel}{fftSourceDetail && fftSourceDetail !== fftSourceLabel ? ` (${fftSourceDetail})` : ""}
                {fftBackendInfo.ms != null ? `, ${fftBackendInfo.ms.toFixed(1)} ms` : ""}.
              </Typography>
              {fftBackendInfo.panels != null && (
                <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>
                  FFT panels: {fftBackendInfo.panels}{fftBackendInfo.grid ? `, grid ${fftBackendInfo.grid}` : ""}.
                </Typography>
              )}
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on image to draw a line intensity profile.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Lens: Magnifier inset that follows the cursor.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Scale: Linear or logarithmic intensity mapping.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Stack-wide percentile contrast for Show3D image panels. FFT Auto masks DC + clips to 99.9th.</Typography>
              {roiAllowed && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>ROI: Click empty image to add at cursor, click ROI to select, drag to move, hover edge to resize. Del removes selected; Ctrl/⌘+D duplicates.</Typography>}
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Cols / Panels: Change the panel grid or hide panels without changing the source stack.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Pinning: Click a panel to select or pin it for keyboard actions, per-panel zoom, ROI edits, and deletion shortcuts.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Pan: With Pan enabled, drag the image to move the zoomed view. With Link Zoom on, pan and zoom move together across panels.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop: Loop playback. Drag end markers on slider for loop range.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Bounce: Ping-pong playback - alternates forward and reverse.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Speed: Lower fps, increase avg only when needed, shorten the loop range, hide panels, or turn off FFT/Profile/Stats to reduce heavy-stack playback work.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT layout: Use side, bottom, or overlay mode. Overlay FFTs can be resized; wheel and drag over the overlay inspect FFT detail independently.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Export / Copy: Export HTML or GIF panel-only animations, or copy the current panel view from the toolbar.</Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
              <KeyboardShortcuts items={keyboardShortcutItems} />
	            </Box>} theme={themeInfo.theme} />}
	            {showControls && (
	              <Button
	                size="small"
	                sx={{
	                  ...compactButton,
	                  ml: 0.75,
	                  py: 0,
	                  px: 0.5,
	                  minHeight: 16,
	                  lineHeight: "16px",
	                  verticalAlign: "baseline",
	                }}
	                onClick={() => setControlsCollapsed(!controlsCollapsed)}
	                aria-label={controlsCollapsed ? "Show controls" : "Hide controls"}
	                aria-pressed={!controlsCollapsed}
	                title={controlsCollapsed ? "Show controls" : "Hide controls"}
	              >
	                Controls
	              </Button>
	            )}
	            {showControls && controlsCollapsed && (exportEnabled || canDownloadCurrentHtml) && (
	              <>
	                <Button
	                  size="small"
	                  sx={{
	                    ...compactButton,
	                    ml: 0.5,
	                    py: 0,
	                    px: 0.5,
	                    minHeight: 16,
	                    lineHeight: "16px",
	                    verticalAlign: "baseline",
	                  }}
                  disabled={exportBusy || (!exportEnabled && !canDownloadCurrentHtml && !canExportStandaloneGif)}
	                  onClick={handleExportMenuOpen}
	                  aria-label="Export widget or animation"
	                  aria-controls={exportMenuAnchor ? "show3d-export-menu-collapsed" : undefined}
	                  aria-expanded={exportMenuAnchor ? "true" : undefined}
	                  aria-haspopup="menu"
	                  title={localExportStatus || exportStatus || (exportEnabled ? "Export HTML or GIF with a save dialog" : "Export standalone HTML or GIF")}
	                >
	                  {exportBusy ? "Exporting" : "Export"}
	                </Button>
	                <Menu
	                  id="show3d-export-menu-collapsed"
	                  anchorEl={exportMenuAnchor}
	                  open={Boolean(exportMenuAnchor)}
	                  onClose={handleExportMenuClose}
	                  MenuListProps={{ "aria-label": "Export options" }}
	                  {...themedMenuProps}
	                >
	                  {renderExportMenuContent()}
	                </Menu>
	              </>
	            )}
	          </Typography>}
	          {/* Page navigation sits above the analysis toolbar so a long page
	              label never competes with Profile / Stats / FFT controls. */}
	          {controlsVisible && isPaged && (
	            <Box
	              data-show3d-page-controls="true"
	              aria-label="Page navigation"
	              sx={{
	                display: "flex",
	                alignItems: "center",
	                flexWrap: "wrap",
	                columnGap: "8px",
	                rowGap: "3px",
	                mb: "3px",
	                minHeight: 26,
	                pb: "3px",
	                borderBottom: `1px solid ${themeColors.border}`,
	              }}
	            >
	              <Box sx={{ display: "flex", alignItems: "baseline", gap: "5px", flex: "1 1 240px", minWidth: 0 }}>
	                <Typography sx={{ ...typography.label, fontSize: 10, flexShrink: 0 }}>Page</Typography>
	                <Typography
	                  data-show3d-page-status="true"
	                  title={pageControlStatus}
	                  sx={{
	                    ...typography.label,
	                    fontSize: 10,
	                    lineHeight: 1.25,
	                    color: themeColors.accent,
	                    minWidth: 0,
	                    whiteSpace: "normal",
	                    overflowWrap: "anywhere",
	                    fontVariantNumeric: "tabular-nums",
	                  }}
	                >
	                  {pageControlStatus}
	                </Typography>
	              </Box>
	              <Box sx={{ display: "flex", alignItems: "center", gap: "4px", flex: "0 1 auto", minWidth: 0 }}>
	                <Slider
	                  value={pageControlIdx}
	                  min={0}
	                  max={Math.max(0, (nPages || 1) - 1)}
	                  step={1}
	                  onPointerDownCapture={() => {
	                    stopPagePlayback();
	                    setPageSliderPreviewIdx(currentPageIdx);
	                  }}
	                  onKeyDown={() => stopPagePlayback()}
	                  onChange={(_, value) => {
	                    const raw = Array.isArray(value) ? value[0] : value;
	                    const next = clampPageIdx(Number(raw));
	                    setPageSliderPreviewIdx(next);
	                    commitPageIdx(next);
	                  }}
	                  onChangeCommitted={(_, value) => {
	                    const raw = Array.isArray(value) ? value[0] : value;
	                    const next = clampPageIdx(Number(raw));
	                    stopPagePlayback();
	                    setPageSliderPreviewIdx(next);
	                    commitPageIdx(next, true);
	                  }}
	                  size="small"
	                  sx={{ ...sliderStyles.small, width: 150, flex: "0 1 150px", minWidth: 92, color: themeColors.accent }}
	                  aria-label="Page"
	                />
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
	                  onChange={(e) => setPagePlayFps(Number(e.target.value) || 2)}
	                  size="small"
	                  sx={{ ...themedSelect, minWidth: 48, fontSize: 10 }}
	                  MenuProps={themedMenuProps}
	                  inputProps={{ "aria-label": "Page playback frames per second" }}
	                  title="Page playback speed"
	                >
	                  {PAGE_PLAY_FPS_OPTIONS.map((fps) => (
	                    <MenuItem key={fps} value={String(fps)}>{fps} fps</MenuItem>
	                  ))}
	                </Select>
	                <IconButton
	                  size="small"
	                  onClick={() => {
	                    const next = Array.from({ length: Math.max(1, nPages || 1) }, (_, idx) => pageStarred?.[idx] ? 1 : 0);
	                    next[pageControlIdx] = next[pageControlIdx] ? 0 : 1;
	                    setPageStarred(next);
	                  }}
	                  title={(pageStarred?.[pageControlIdx] ? "Unstar " : "Star ") + pageControlLabel}
	                  aria-label={(pageStarred?.[pageControlIdx] ? "Unstar " : "Star ") + pageControlLabel}
	                  sx={{
	                    width: 24,
	                    height: 24,
	                    p: 0,
	                    color: pageStarred?.[pageControlIdx] ? "#ffc107" : themeColors.textMuted,
	                    "&:hover": { color: pageStarred?.[pageControlIdx] ? "#ffc107" : themeColors.text },
	                  }}
	                >
	                  {pageStarred?.[pageControlIdx] ? "★" : "☆"}
	                </IconButton>
	              </Box>
	            </Box>
	          )}
	          {/* Analysis and display controls row. */}
	          {controlsVisible && (
	          <Box ref={toolControlsRef} data-show3d-tool-controls="true" sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: "4px", mb: `${SPACING.XS}px`, minHeight: 28 }}>
            {visiblePanelCount > 1 && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Cols</Typography>
                <Select
                  value={String(clampedMaxCols)}
                  onChange={(e) => {
                    const next = Math.max(1, Math.min(Number(e.target.value) || 1, visiblePanelCount || 1, MAX_PANEL_COLUMNS));
                    setMaxCols(next);
                  }}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 48, fontSize: 10 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Show3D panel columns" }}
                  title="Maximum panel columns; the viewer reduces columns when the window is too narrow"
                >
                  {show3dColumnOptions.map((cols) => (
                    <MenuItem key={cols} value={String(cols)}>{cols}</MenuItem>
                  ))}
                </Select>
              </>
            )}
            {/* Kymograph toggle: HIDDEN until a profile line exists (canKymograph),
                not shown-but-disabled, because the kymograph is built from that
                line. Turning it on takes the side slot from FFT. */}
            {canKymograph && <>
              <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Kymo</Typography>
              <Switch checked={showKymograph} onChange={(e) => { const on = e.target.checked; setShowKymograph(on); if (on) setShowFft(false); }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle kymograph space-time panel" } }} />
            </>}
            {/* Profile and ROI are mutually exclusive line/region tools. Turning
                one on turns the other off. Kymograph rides on Profile. */}
            <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Profile</Typography>
            <Switch checked={profileActive} onChange={(e) => {
              const on = e.target.checked;
              setProfileActive(on);
              if (on) {
                setRoiActive(false); setRoiSelectedIdx(-1);
              } else {
                // Toggle OFF hides overlay + kymograph but keeps the line + data
                // so re-enable restores instantly. Use Clear to actively wipe.
                setShowKymograph(false);
                setHoveredProfileEndpoint(null); setIsHoveringProfileLine(false);
              }
            }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle line intensity profile tool" } }} />
            {profileActive && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "4px" }}>W</Typography>
                <Slider value={profileWidth} min={1} max={15} step={1} onChange={(_, v) => setProfileWidth(v as number)} size="small" valueLabelDisplay="auto" sx={{ width: 60, ml: "2px" }} aria-label={`Profile width ${profileWidth} px`} />
              </>
            )}
            {(nPanels || 1) === 1 && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Lens</Typography>
                <Switch
                  checked={showLens}
                  onChange={() => {
                    if (!showLens) { setShowLens(true); setLensPos({ row: Math.floor(height / 2), col: Math.floor(width / 2) }); }
                    else { setShowLens(false); setLensPos(null); }
                  }}
                  size="small"
                  sx={switchStyles.small}
                  slotProps={{ input: { "aria-label": "Toggle magnifier lens" } }}
                />
              </>
            )}
            {/* ROI hidden while kymograph is shown (roiAllowed already encodes
                single-panel && !showKymograph). */}
            {roiAllowed && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>ROI</Typography>
                <Switch checked={roiActive} onChange={(e) => {
                  const on = e.target.checked;
                  if (on) {
                    setRoiActive(true); setShowRoiResizeHint(true);
                    setProfileActive(false); setProfileLine([]); setProfileData(null); setHoveredProfileEndpoint(null); setIsHoveringProfileLine(false);
                  } else {
                    setRoiActive(false); setRoiSelectedIdx(-1); pendingRoiAddRef.current = null;
                  }
                }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle ROI selection tool" } }} />
              </>
            )}
            {/* "More" overflow: Stats + Denoise + Filter live here (mirrors Show2D) to
                keep the top toolbar calm. */}
            <Badge
              badgeContent={(showStats ? 1 : 0) + (overlayEditMode ? 1 : 0) + (denoiseEnabled ? 1 : 0) + (!isRgb && frequencyFilterIsActive ? 1 : 0) + (subpixelAlignEnabled ? 1 : 0) + (!isRgb && hasPanelChoices && !colorShared ? 1 : 0) + (flipRows ? 1 : 0) + (flipCols ? 1 : 0) + (compareMode !== "off" ? 1 : 0) + (rotationActive ? 1 : 0)}
              invisible={!showStats && !overlayEditMode && !showDenoise && !(!isRgb && frequencyFilterIsActive) && !subpixelAlignEnabled && !(!isRgb && hasPanelChoices && !colorShared) && !flipRows && !flipCols && compareMode === "off" && !rotationActive}
              sx={{ "& .MuiBadge-badge": { bgcolor: themeColors.accent, color: "#fff", fontSize: 9, fontWeight: 600, minWidth: 14, height: 14, px: 0.25 } }}
            >
              <Button
                size="small"
                sx={{ minWidth: 0, px: 0.75, fontSize: 10, textTransform: "none", color: (showStats || overlayEditMode || showDenoise || (!isRgb && frequencyFilterIsActive) || subpixelAlignEnabled || (!isRgb && hasPanelChoices && !colorShared) || flipRows || flipCols || compareMode !== "off" || rotationActive) ? themeColors.accent : themeColors.text }}
                onClick={(e) => setMoreMenuAnchor(e.currentTarget)}
                aria-label="More tools"
                aria-haspopup="menu"
                title="More tools: Stats, Denoise, Filter, Sub-pixel alignment, Color, Flip, Rotate, Compare"
              >
                More
              </Button>
            </Badge>
            <Menu
              anchorEl={moreMenuAnchor}
              open={Boolean(moreMenuAnchor)}
              onClose={() => setMoreMenuAnchor(null)}
              MenuListProps={{ "aria-label": "More tools" }}
              {...themedMenuProps}
            >
              <Box sx={{ px: 1.5, pt: 0.75, pb: 0.35, minWidth: 260 }}>
                <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Readout</Typography>
              </Box>
              <MenuItem dense onClick={() => setShowStats(!showStats)} sx={{ fontSize: 12, gap: 1, color: showStats ? themeColors.accent : themeColors.text }}>
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Mean / min / max / std readout under the image.">Stats</Typography>
                <Switch checked={showStats} onClick={(e) => e.stopPropagation()} onChange={(e) => setShowStats(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle statistics readout" } }} />
              </MenuItem>
              {hasPanelOverlays && (
                <MenuItem dense onClick={() => setOverlayEditMode(!overlayEditMode)} sx={{ fontSize: 12, gap: 1, color: overlayEditMode ? themeColors.accent : themeColors.text }}>
                  <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Edit API-defined circles and rectangles: click to select, drag to move, drag an edge to resize.">Overlay Edit</Typography>
                  <Switch
                    checked={overlayEditMode}
                    onClick={(event) => event.stopPropagation()}
                    onChange={(event) => setOverlayEditMode(event.target.checked)}
                    size="small"
                    sx={switchStyles.small}
                    slotProps={{ input: { "aria-label": "Toggle overlay editing" } }}
                  />
                </MenuItem>
              )}
              {hasPanelOverlays && overlayBaselineRef.current && (
                <MenuItem dense onClick={resetPanelOverlays} sx={{ fontSize: 12, color: overlaySelection ? themeColors.accent : themeColors.text }}>
                  Reset Overlays
                </MenuItem>
              )}
              <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${themeColors.border}`, opacity: 0.9 }} />
              <Box sx={{ px: 1.5, pt: 0.35, pb: 0.35 }}>
                <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Processing</Typography>
              </Box>
              <MenuItem dense onClick={toggleDenoise} sx={{ fontSize: 12, gap: 1, color: denoiseEnabled ? themeColors.accent : themeColors.text }}>
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only denoise: ON shows the denoised view, OFF shows raw (config preserved). Raw data and stats keep original counts.">Denoise</Typography>
                <Switch checked={denoiseEnabled ?? false} onClick={(e) => e.stopPropagation()} onChange={toggleDenoise} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle denoise on/off" } }} />
              </MenuItem>
              {!isRgb && (
                <MenuItem dense onClick={() => setFrequencyMaster(!frequencyFilterEnabled)} sx={{ fontSize: 12, gap: 1, color: frequencyFilterIsActive ? themeColors.accent : themeColors.text }}>
                  <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Off by default. Turn on to remove a background or isolate a periodicity; raw counts remain unchanged.">Filter</Typography>
                  <Switch checked={frequencyFilterEnabled ?? false} onClick={(e) => e.stopPropagation()} onChange={() => setFrequencyMaster(!frequencyFilterEnabled)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle frequency filter effect" } }} />
                </MenuItem>
              )}
              {!isRgb && (
                <MenuItem
                  dense
                  onClick={() => {
                    const next = !subpixelAlignEnabled;
                    setSubpixelAlignEnabled(next);
                    if (next && !subpixelAlignSupported) {
                      setSubpixelAlignStatus("Needs a single-panel client-side stack");
                    }
                  }}
                  sx={{ fontSize: 12, gap: 1, color: subpixelAlignEnabled ? themeColors.accent : themeColors.text }}
                >
                  <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only sub-pixel frame alignment. First scope: single-panel client-side stacks; raw data stays unchanged.">Sub-pixel align</Typography>
                  <Switch
                    checked={subpixelAlignEnabled ?? false}
                    onClick={(e) => e.stopPropagation()}
                    onChange={() => setSubpixelAlignEnabled(!subpixelAlignEnabled)}
                    size="small"
                    sx={switchStyles.small}
                    slotProps={{ input: { "aria-label": "Toggle sub-pixel alignment" } }}
                  />
                </MenuItem>
              )}
              {(subpixelAlignEnabled || subpixelAlignStatus !== "Off") && !isRgb && (
                <Box
                  onClick={(e) => e.stopPropagation()}
                  sx={{
                    px: 1.5,
                    py: 0.75,
                    minWidth: 260,
                    display: "grid",
                    gridTemplateColumns: "1fr auto",
                    gap: 0.75,
                    alignItems: "center",
                  }}
                >
                  <TextField
                    label="Reference frame"
                    type="number"
                    size="small"
                    value={Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(subpixelAlignReference || 0)))}
                    onChange={(e) => setSubpixelAlignReference(Number(e.target.value))}
                    inputProps={{ min: 0, max: Math.max(0, nSlices - 1), "aria-label": "Sub-pixel alignment reference frame" }}
                    sx={{
                      "& .MuiInputBase-input": { fontSize: 11, py: 0.5 },
                      "& .MuiInputLabel-root": { fontSize: 11 },
                    }}
                  />
                  <Button
                    size="small"
                    sx={compactButton}
                    disabled={!subpixelAlignEnabled || subpixelAlignBusy || !subpixelAlignSupported}
                    onClick={() => void computeSubpixelAlignment()}
                    title="Compute alignment now and repaint the current frame"
                  >
                    {subpixelAlignBusy ? "Aligning" : subpixelAlignShiftsRef.current ? "Re-align" : "Align"}
                  </Button>
                  <Typography sx={{ gridColumn: "1 / -1", fontSize: 10, color: subpixelAlignSupported || !subpixelAlignEnabled ? themeColors.textMuted : themeColors.accentYellow }}>
                    {subpixelAlignStatus}
                  </Typography>
                </Box>
              )}
              <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${themeColors.border}`, opacity: 0.9 }} />
              <Box sx={{ px: 1.5, pt: 0.35, pb: 0.35 }}>
                <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Orientation</Typography>
              </Box>
              <MenuItem dense onClick={() => setFlipRows(!flipRows)} sx={{ fontSize: 12, gap: 1, color: flipRows ? themeColors.accent : themeColors.text }}>
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only vertical flip for orientation checks; raw data and coordinates are unchanged.">Flip Rows</Typography>
                <Switch checked={flipRows} onClick={(e) => e.stopPropagation()} onChange={(e) => setFlipRows(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle vertical row flip" } }} />
              </MenuItem>
              <MenuItem dense onClick={() => setFlipCols(!flipCols)} sx={{ fontSize: 12, gap: 1, color: flipCols ? themeColors.accent : themeColors.text }}>
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only horizontal flip for handedness checks; raw data and coordinates are unchanged.">Flip Cols</Typography>
                <Switch checked={flipCols} onClick={(e) => e.stopPropagation()} onChange={(e) => setFlipCols(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle horizontal column flip" } }} />
              </MenuItem>
              <MenuItem
                dense
                onClick={() => {
                  if (rotationActive) clearRotations();
                  else setShowRotationSettings(!showRotationSettings);
                }}
                sx={{ fontSize: 12, gap: 1, color: (rotationActive || showRotationSettings) ? themeColors.accent : themeColors.text }}
              >
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only orientation review. Turn on to choose angle and scope.">Rotate</Typography>
                <Switch
                  checked={rotationActive || showRotationSettings}
                  onClick={(event) => event.stopPropagation()}
                  onChange={(event) => {
                    if (event.target.checked) setShowRotationSettings(true);
                    else clearRotations();
                  }}
                  size="small"
                  sx={switchStyles.small}
                  slotProps={{ input: { "aria-label": "Toggle rotation settings" } }}
                />
              </MenuItem>
              {(rotationActive || showRotationSettings) && (
                <Box
                  onClick={(event) => event.stopPropagation()}
                  sx={{
                    px: 1.5,
                    pb: 1,
                    minWidth: 260,
                    display: "grid",
                    gridTemplateColumns: "auto 1fr",
                    gap: 0.75,
                    alignItems: "center",
                  }}
                >
                  <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Angle</Typography>
                  <Select
                    value={String(((imageRotation % 4) + 4) % 4 * 90)}
                    onChange={(event) => setRotationForScope(Number(event.target.value) / 90)}
                    size="small"
                    sx={{ ...themedSelect, minWidth: 92 }}
                    MenuProps={themedMenuProps}
                    inputProps={{ "aria-label": "Display rotation" }}
                    title="Display-only rotation; raw data coordinates stay unchanged"
                  >
                    <MenuItem value="0">0°</MenuItem>
                    <MenuItem value="90">90°</MenuItem>
                    <MenuItem value="180">180°</MenuItem>
                    <MenuItem value="270">270°</MenuItem>
                  </Select>
                  <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Scope</Typography>
                  <Select
                    value={rotationScope || "all"}
                    onChange={(event) => setRotationScope(String(event.target.value))}
                    size="small"
                    sx={{ ...themedSelect, minWidth: 92 }}
                    MenuProps={themedMenuProps}
                    inputProps={{ "aria-label": "Rotation scope" }}
                  >
                    <MenuItem value="all">All</MenuItem>
                    <MenuItem value="frame">Frame</MenuItem>
                  </Select>
                  <Typography sx={{ gridColumn: "1 / -1", fontSize: 10, color: themeColors.textMuted }}>
                    {(rotationScope || "all") === "frame"
                      ? `${dimLabel || "Frame"} ${Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(displaySliceIdx || sliceIdx || 0)))} only`
                      : "Applies to the whole stack"}
                  </Typography>
                </Box>
              )}
              {!isRgb && hasPanelChoices && (
                <>
                  <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${themeColors.border}`, opacity: 0.9 }} />
                  <Box sx={{ px: 1.5, pt: 0.35, pb: 0.35 }}>
                    <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Color</Typography>
                  </Box>
                  <MenuItem
                    dense
                    onClick={() => setColorShared(
                      colorShared ? false : true,
                      colorTargetPanel,
                    )}
                    sx={{ fontSize: 12, gap: 1, color: !colorShared ? themeColors.accent : themeColors.text }}
                  >
                    <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Shared keeps one colormap for every panel. Turn off to let the Color dropdown edit only the selected panel.">Color shared</Typography>
                    <Switch
                      checked={colorShared}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => setColorShared(
                        e.target.checked,
                        colorTargetPanel,
                      )}
                      size="small"
                      sx={switchStyles.small}
                      slotProps={{ input: { "aria-label": "Toggle shared panel colormap" } }}
                    />
                  </MenuItem>
                </>
              )}
              <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${themeColors.border}`, opacity: 0.9 }} />
              <Box sx={{ px: 1.5, pt: 0.35, pb: 0.35 }}>
                <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Compare</Typography>
              </Box>
              <MenuItem
                dense
                onClick={() => setCompareActiveFromCurrentFrame(compareMode === "off")}
                sx={{ fontSize: 12, gap: 1, color: compareMode !== "off" ? themeColors.accent : themeColors.text }}
              >
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Blink, difference, or overlay two frames for change detection.">Compare</Typography>
                <Switch
                  checked={compareMode !== "off"}
                  onClick={(event) => event.stopPropagation()}
                  onChange={(event) => setCompareActiveFromCurrentFrame(event.target.checked)}
                  size="small"
                  sx={switchStyles.small}
                  slotProps={{ input: { "aria-label": "Toggle compare settings" } }}
                />
              </MenuItem>
              {compareMode !== "off" && (
                <Box
                  onClick={(e) => e.stopPropagation()}
                  sx={{
                    px: 1.5,
                    pb: 1,
                    minWidth: 260,
                    display: "grid",
                    gridTemplateColumns: "auto 1fr",
                    gap: 0.75,
                    alignItems: "center",
                  }}
                >
                  <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Mode</Typography>
                  <Select
                    value={compareMode || "blink"}
                    onChange={(e) => setCompareMode(String(e.target.value))}
                    size="small"
                    sx={{ ...themedSelect, minWidth: 120 }}
                    MenuProps={themedMenuProps}
                    inputProps={{ "aria-label": "Compare mode" }}
                  >
                    <MenuItem value="blink">Blink</MenuItem>
                    <MenuItem value="difference">Difference</MenuItem>
                    <MenuItem value="overlay">Overlay</MenuItem>
                  </Select>
                <Typography sx={{ fontSize: 12, color: themeColors.text }}>A</Typography>
                <TextField
                  type="number"
                  size="small"
                  value={Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(comparePair?.[0] ?? 0)))}
                  onChange={(e) => setComparePair([Number(e.target.value) || 0, comparePair?.[1] ?? 1])}
                  inputProps={{ min: 0, max: Math.max(0, nSlices - 1), "aria-label": "Compare frame A" }}
                  sx={{ input: { color: themeColors.text, fontSize: 12, py: 0.5 }, "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border } }}
                />
                <Typography sx={{ fontSize: 12, color: themeColors.text }}>B</Typography>
                <TextField
                  type="number"
                  size="small"
                  value={Math.max(0, Math.min(Math.max(0, nSlices - 1), Math.round(comparePair?.[1] ?? 1)))}
                  onChange={(e) => setComparePair([comparePair?.[0] ?? 0, Number(e.target.value) || 0])}
                  inputProps={{ min: 0, max: Math.max(0, nSlices - 1), "aria-label": "Compare frame B" }}
                  sx={{ input: { color: themeColors.text, fontSize: 12, py: 0.5 }, "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border } }}
                />
                <Typography sx={{ fontSize: 12, color: themeColors.text }}>Speed</Typography>
                <Select
                  value={String(blinkFps)}
                  onChange={(e) => setBlinkFps(Number(e.target.value) || 2)}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 92 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Blink speed" }}
                >
                  <MenuItem value="0.5">0.5x</MenuItem>
                  <MenuItem value="1">1x</MenuItem>
                  <MenuItem value="2">2x</MenuItem>
                  <MenuItem value="4">4x</MenuItem>
                </Select>
                <Typography sx={{ fontSize: 12, color: themeColors.text }}>Background</Typography>
                <Select
                  value={compareBackground || "dark"}
                  onChange={(e) => setCompareBackground(String(e.target.value))}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 92 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Compare background" }}
                >
                  <MenuItem value="dark">Dark</MenuItem>
                  <MenuItem value="light">Light</MenuItem>
                </Select>
                <Typography sx={{ fontSize: 12, color: themeColors.text }}>Diff</Typography>
                <Select
                  value={diffCmap || "magenta-green"}
                  onChange={(e) => setDiffCmap(String(e.target.value))}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 120 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Difference colormap" }}
                >
                  <MenuItem value="magenta-green">Magenta/Green</MenuItem>
                  <MenuItem value="red-blue">Red/Blue</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                </Box>
              )}
              {!isRgb && (
                <>
                <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${themeColors.border}`, opacity: 0.9 }} />
                <Box
                  onClick={(event) => event.stopPropagation()}
                  sx={{
                    px: 1.5,
                    pt: 0.35,
                    pb: 1,
                    minWidth: 260,
                    display: "grid",
                    gridTemplateColumns: "auto 1fr",
                    gap: 0.75,
                    alignItems: "center",
                  }}
                >
                  <Typography sx={{ gridColumn: "1 / -1", fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: themeColors.textMuted, textTransform: "uppercase" }}>Contrast</Typography>
                  <Typography sx={{ fontSize: 12, color: themeColors.text }} title="Choose the percentile contrast range. Histogram stays visible below the image.">Range</Typography>
                  <Select
                    size="small"
                    value={contrastPreset || "custom"}
                    onChange={(e) => applyContrastPreset(String(e.target.value))}
                    sx={{ ...themedSelect, minWidth: 110 }}
                    MenuProps={themedMenuProps}
                    inputProps={{ "aria-label": "Contrast percentile range" }}
                  >
                    {CONTRAST_PRESETS.map((preset) => (
                      <MenuItem key={preset.value} value={preset.value}>{preset.label}</MenuItem>
                    ))}
                  </Select>
                </Box>
                </>
              )}
            </Menu>
            {hasPanelChoices && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Link</Typography>
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Zoom</Typography>
                <Switch checked={linkPanels} onChange={(e) => setLinkPanels(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Link zoom and pan across panels" } }} />
                <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Contrast</Typography>
                <Switch checked={linkContrast} onChange={(e) => setLinkContrast(e.target.checked)} size="small" sx={switchStyles.small} title="Link relative contrast adjustment; numerical ranges remain panel-local" slotProps={{ input: { "aria-label": "Link contrast across panels" } }} />
              </>
            )}
            {fftAllowed && (
              <Box aria-hidden="true" sx={{ width: "1px", height: 20, flex: "0 0 1px", alignSelf: "center", mx: "4px", bgcolor: themeColors.border, opacity: 0.8 }} />
            )}
            {/* FFT can be shown below, beside, or as an inset over the image grid. */}
            {fftAllowed && <>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT</Typography>
              <Switch checked={showFft} onChange={(e) => { const on = e.target.checked; setShowFft(on); if (on) setShowKymograph(false); }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle FFT power spectrum panel" } }} />
              {showFft && (
                <Select
                  value={resolvedFftLayout}
                  onChange={(e) => setFftLayout(String(e.target.value))}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 78, fontSize: 10, ml: "2px" }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "FFT panel layout" }}
                >
                  <MenuItem value="bottom">Bottom</MenuItem>
                  <MenuItem value="right">Right</MenuItem>
                  <MenuItem value="overlay">Overlay</MenuItem>
                </Select>
              )}
              {showFft && fftLayoutOverlay && (
                <>
                  <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Size</Typography>
                  <Select
                    value={String(Math.round(resolvedFftOverlaySize * 100))}
                    onChange={(e) => setFftOverlaySize(Number(e.target.value) / 100)}
                    size="small"
                    sx={{ ...themedSelect, minWidth: 52, fontSize: 10, ml: "2px" }}
                    MenuProps={themedMenuProps}
                    inputProps={{ "aria-label": "FFT overlay size" }}
                  >
                    <MenuItem value="25">25%</MenuItem>
                    <MenuItem value="35">35%</MenuItem>
                    <MenuItem value="50">50%</MenuItem>
                    <MenuItem value="65">65%</MenuItem>
                  </Select>
                </>
              )}
            </>}
            <Box sx={{ flex: 1 }} />
            <Box sx={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <Button size="small" sx={compactButton} onClick={handleCopy} aria-label="Copy current frame to clipboard as PNG">Copy</Button>
              {hasPanelChoices && (
                <>
                  {!isPaged && (
                    <Button
                      size="small"
                      sx={{
                        ...compactButton,
                        color: reorderMode ? themeColors.accent : themeColors.text,
                        "& .MuiButton-startIcon": { mr: 0.4 },
                      }}
                      startIcon={<DragIndicatorIcon sx={{ fontSize: 14 }} />}
                      onClick={() => setReorderMode((value) => !value)}
                      aria-pressed={reorderMode ? "true" : "false"}
                      aria-label={reorderMode ? "Finish reordering panels" : "Reorder panels"}
                      title={reorderMode ? "Finish reordering panels" : "Reorder panels"}
                    >
                      Reorder
                    </Button>
                  )}
                  <Button
                    size="small"
                    sx={{ ...compactButton, "& .MuiButton-startIcon": { mr: 0.4 } }}
                    startIcon={<VisibilityIcon sx={{ fontSize: 14 }} />}
                    onClick={(event) => setPanelMenuAnchor(event.currentTarget)}
                    aria-label="Choose visible panels"
                    aria-controls={panelMenuAnchor ? "show3d-panels-menu" : undefined}
                    aria-expanded={panelMenuAnchor ? "true" : undefined}
                    aria-haspopup="menu"
                  >
                    {visiblePanelCount === panelMenuTotal ? "Panels" : `Panels ${visiblePanelCount}/${panelMenuTotal}`}
                  </Button>
                  {selectedVisibleCount > 1 && selectedVisibleCount < visiblePanelCount && (
                    <Button
                      size="small"
                      sx={compactButton}
                      onClick={() => setPanelsHidden(selectedVisiblePanels, true)}
                      aria-label={`Hide ${selectedVisibleCount} selected panels`}
                      title={`Hide ${selectedVisibleCount} selected panels`}
                    >
                      Hide {selectedVisibleCount}
                    </Button>
                  )}
                  <Menu
                    id="show3d-panels-menu"
                    anchorEl={panelMenuAnchor}
                    open={Boolean(panelMenuAnchor)}
                    onClose={() => setPanelMenuAnchor(null)}
                    MenuListProps={{ "aria-label": "Panel visibility options" }}
                    {...themedMenuProps}
                  >
                    {orderedPanelIndices.map((panel) => {
                      const hidden = hiddenPanelSet.has(panel);
                      const disabled = !hidden && visiblePanelCount <= 1;
                      return (
                        <MenuItem
                          key={`panel-menu-${panel}`}
                          dense
                          disabled={disabled}
                          onClick={() => setPanelHidden(panel, !hidden)}
                          title={disabled ? "At least one panel must remain visible" : undefined}
                        >
                          {hidden
                            ? <VisibilityOffIcon sx={{ fontSize: 16, mr: 1, color: themeColors.textMuted }} />
                            : <VisibilityIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />}
                          <Typography sx={{ fontSize: 11, color: disabled ? themeColors.textMuted : themeColors.text }}>
                            {panelTitleContent(panel)}
                          </Typography>
                        </MenuItem>
                      );
                    })}
                    <MenuItem
                      dense
                      disabled={hiddenPanelSet.size === 0}
                      onClick={() => {
                        if (isPaged) {
                          setHiddenPageSlots([]);
                          setHiddenPageSlotsTrait([]);
                        }
                        setHiddenPanels([]);
                        setPanelMenuAnchor(null);
                      }}
                    >
                      <VisibilityIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />
                      <Typography sx={{ fontSize: 11 }}>Show all panels</Typography>
                    </MenuItem>
                    <MenuItem
                      dense
                      disabled={selectedVisibleCount <= 1 || selectedVisibleCount >= visiblePanelCount}
                      onClick={() => setPanelsHidden(selectedVisiblePanels, true)}
                      title={selectedVisibleCount >= visiblePanelCount ? "At least one panel must remain visible" : undefined}
                    >
                      <VisibilityOffIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />
                      <Typography sx={{ fontSize: 11 }}>Hide selected ({selectedVisibleCount})</Typography>
                    </MenuItem>
                    <MenuItem
                      dense
                      disabled={selectedVisibleCount <= 1}
                      onClick={() => setSelectedPanels([])}
                    >
                      <VisibilityIcon sx={{ fontSize: 16, mr: 1, color: themeColors.textMuted }} />
                      <Typography sx={{ fontSize: 11 }}>Clear selection</Typography>
                    </MenuItem>
                    {!isPaged && (
                      <MenuItem
                        dense
                        disabled={(panelOrder || []).length === 0}
                        onClick={resetPanelOrder}
                      >
                        <DragIndicatorIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />
                        <Typography sx={{ fontSize: 11 }}>Reset order</Typography>
                      </MenuItem>
                    )}
                  </Menu>
                </>
              )}
              {(exportEnabled || canDownloadCurrentHtml) && (
                <>
                  <Button
                    size="small"
                    sx={compactButton}
                  disabled={exportBusy || (!exportEnabled && !canDownloadCurrentHtml && !canExportStandaloneGif)}
                    onClick={handleExportMenuOpen}
                    aria-label="Export widget or animation"
                    aria-controls={exportMenuAnchor ? "show3d-export-menu" : undefined}
                    aria-expanded={exportMenuAnchor ? "true" : undefined}
                    aria-haspopup="menu"
                    title={localExportStatus || exportStatus || (exportEnabled ? "Export HTML or GIF with a save dialog" : "Export standalone HTML or GIF")}
                  >
                    {exportBusy ? "Exporting" : "Export"}
                  </Button>
                  <Menu
                    id="show3d-export-menu"
                    anchorEl={exportMenuAnchor}
                    open={Boolean(exportMenuAnchor)}
                    onClose={handleExportMenuClose}
                    MenuListProps={{ "aria-label": "Export options" }}
                    {...themedMenuProps}
                  >
                    {renderExportMenuContent()}
                  </Menu>
                </>
              )}
              {(exportEnabled || canDownloadCurrentHtml || canExportStandaloneGif) && (localExportStatus || exportStatus) && (
                <Typography
                  sx={{
                    ...typography.label,
                    fontSize: 10,
                    maxWidth: 260,
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
              <Button size="small" sx={compactButton} disabled={!needsReset} onClick={handleDoubleClick} aria-label="Reset zoom and pan">Reset</Button>
	          </Box>
	          </Box>
	          )}
          <Box
            ref={canvasContainerRef}
            sx={{
              ...container.imageBox,
              bgcolor: compareMode !== "off" ? (compareBackground === "light" ? "#f7f7f7" : "#050505") : container.imageBox.bgcolor,
              width: "100%",
              maxWidth: canvasW,
              boxSizing: "content-box",
              border: galleryOuterBorderPx > 0 ? `${galleryOuterBorderPx}px solid ${galleryOuterBorderColor}` : "none",
              aspectRatio: mainPanelAspectRatio,
              height: "auto",
              overscrollBehavior: "contain",
              touchAction: "none",
              ...(reorderMode ? {
                "@keyframes show3d-reorder-jiggle": {
                  "0%": { rotate: "-0.45deg" },
                  "100%": { rotate: "0.45deg" },
                },
              } : {}),
              cursor: reorderMode
                ? "grab"
                : overlayEditMode
                ? (isDraggingOverlay ? "grabbing" : isHoveringOverlay ? "nwse-resize" : "crosshair")
                : isHoveringLensEdge
                ? "nwse-resize"
                : (isHoveringResize || isDraggingResize || isHoveringResizeInner || isDraggingResizeInner)
                  ? "nwse-resize"
                  : (draggingProfileEndpoint !== null || isDraggingProfileLine)
                    ? "grabbing"
                    : (profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine))
                      ? "grab"
                      : (effectiveRoiActive || profileActive)
                        ? "crosshair"
                        : "grab",
            }}
            onMouseDown={reorderMode ? undefined : handleCanvasMouseDown}
            onMouseMove={reorderMode ? undefined : handleCanvasMouseMove}
            onMouseUp={reorderMode ? undefined : handleCanvasMouseUp}
            onMouseLeave={reorderMode ? undefined : handleCanvasMouseLeave}
            onDoubleClick={reorderMode ? undefined : handleDoubleClick}
          >
            <canvas
              data-quantem-scientific-output="show3d-image"
              ref={canvasRef}
              width={canvasW}
              height={canvasH}
              onTouchStart={reorderMode ? undefined : handleCanvasTouchStart}
              onTouchMove={reorderMode ? undefined : handleCanvasTouchMove}
              onTouchEnd={reorderMode ? undefined : handleCanvasTouchEnd}
              onTouchCancel={reorderMode ? undefined : handleCanvasTouchEnd}
              style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", opacity: gpuDisplayVisible ? 0 : 1, display: "block", touchAction: "none" }}
              role="img"
              aria-label={`Slice image ${visibleSliceIdx + 1} of ${nSlices}${title ? `: ${title}` : ""} (${width} by ${height} pixels). Use arrow keys to scrub frames.`}
            />
            <canvas ref={gpuCanvasRef} width={canvasW} height={canvasH} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", pointerEvents: "none", opacity: gpuDisplayVisible ? 1 : 0 }} aria-hidden="true" />
            <canvas ref={overlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", display: overlayCanvasVisible ? "block" : "none" }} aria-hidden="true" />
            <canvas ref={uiRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
            <canvas ref={lensCanvasRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", display: lensCanvasVisible ? "block" : "none" }} aria-hidden="true" />
            {effectiveShowFft && fftLayoutOverlay && (
              <canvas
                ref={fftInsetLayerRef}
                width={canvasW}
                height={canvasH}
                style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", pointerEvents: "none", zIndex: 7 }}
                aria-hidden="true"
              />
            )}
            {(nPanels || 1) > 1 && visiblePanelIndices.map((panel, slot) => {
              if (!selectedPanelSet.has(panel)) return null;
              const visibleCount = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(visibleCount);
              const rows = Math.ceil(visibleCount / cols);
              const gap = visibleCount > 1 ? (panelGapPx) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              return (
                <Box
                  key={`panel-selection-${panel}`}
                  data-show3d-panel-selection={panel}
                  title={`Selected ${panelLabel(panel)}`}
                  sx={{
                    position: "absolute",
                    left: `${(panelLeft / Math.max(1, canvasW)) * 100}%`,
                    top: `${(panelTop / Math.max(1, canvasH)) * 100}%`,
                    width: `${(panelW / Math.max(1, canvasW)) * 100}%`,
                    height: `${(panelH / Math.max(1, canvasH)) * 100}%`,
                    boxSizing: "border-box",
                    boxShadow: `inset 0 0 0 3px ${themeColors.accent}`,
                    pointerEvents: "none",
                    zIndex: 9,
                  }}
                />
              );
            })}
            {visiblePanelIndices.flatMap((panel, slot) => {
              const annotations = panelAnnotations?.[panel] || [];
              if (!annotations.length) return [];
              const visibleCount = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(visibleCount);
              const rows = Math.ceil(visibleCount / cols);
              const gap = visibleCount > 1 ? (panelGapPx) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              return annotations.map((annotation, annotationIdx) => (
                <Box
                  key={`panel-annotation-${panel}-${annotationIdx}`}
                  className={annotation.class_name}
                  data-show3d-panel-annotation={panel}
                  data-show3d-panel-annotation-index={annotationIdx}
                  data-show3d-panel-annotation-position={annotation.position || "top-left"}
                  data-show3d-panel-annotation-variant={annotation.variant || "badge"}
                  title={annotation.text}
                  sx={{
                    position: "absolute",
                    left: `${(panelLeft / Math.max(1, canvasW)) * 100}%`,
                    top: `${(panelTop / Math.max(1, canvasH)) * 100}%`,
                    width: `${(panelW / Math.max(1, canvasW)) * 100}%`,
                    height: `${(panelH / Math.max(1, canvasH)) * 100}%`,
                    pointerEvents: "none",
                    zIndex: 10,
                  }}
                >
                  <Box
                    component="span"
                    sx={panelAnnotationSx(annotation, 10)}
                  >
                    {renderPanelAnnotation(annotation)}
                  </Box>
                </Box>
              ));
            })}
            {showPanelTitles !== false && (nPanels || 1) > 1 && visiblePanelIndices.map((panel, slot) => {
              const titleText = panelTitleText(panel);
              if (!titleText) return null;
              const visibleCount = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(visibleCount);
              const rows = Math.ceil(visibleCount / cols);
              const gap = visibleCount > 1 ? (panelGapPx) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const shownIdx = visibleSliceIdx;
              const realN = panelRealFrames?.[panel];
              const shown = realN ? Math.min(shownIdx + 1, realN) : shownIdx + 1;
              const total = realN || nSlices;
              const frameLabel = panelFrameLabelForIndex(panel, shownIdx);
              return (
                <Box
                  key={`panel-title-${panel}`}
                  data-show3d-panel-title={panel}
                  sx={{
                    position: "absolute",
                    top: `${((panelTop + 6) / Math.max(1, canvasH)) * 100}%`,
                    left: `${(panelLeft / Math.max(1, canvasW)) * 100}%`,
                    width: `${(panelW / Math.max(1, canvasW)) * 100}%`,
                    px: 1,
                    boxSizing: "border-box",
                    color: "rgba(255, 255, 255, 0.95)",
                    fontFamily: UI_FONT,
                    fontSize: Math.max(8, panelTitleFontSize || 11),
                    fontWeight: 700,
                    lineHeight: 1.2,
                    textAlign: "center",
                    textShadow: "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px rgba(0,0,0,0.75)",
                    pointerEvents: "none",
                    userSelect: "none",
                    zIndex: 2,
                    whiteSpace: "normal",
                    overflow: "visible",
                    textOverflow: "clip",
                    overflowWrap: "anywhere",
                  }}
                >

                  {panelTitleContent(panel)}{frameLabel ? ` · ${frameLabel}` : ""}{" "}
                  <span data-show3d-panel-frame-count="true" data-real-frame-count={total}>
                    {shown}/{total}
                  </span>

                </Box>
              );
            })}
            {/* Per-panel "best frame" stars. One gold ★ button top-right of
                each panel. Click toggles the star on the currently displayed
                slice for THAT panel. Programmatic API: widget.star_panel(i). */}
	            {panelChromeVisible && hasPanelChoices && visiblePanelIndices.map((i, slot) => {
              const visibleCount = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(visibleCount);
              const gap = visibleCount > 1 ? (panelGapPx) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (Math.ceil(visibleCount / cols) - 1)) / Math.ceil(visibleCount / cols);
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const starredFrame = starred?.[i] ?? -1;
              const isStarredHere = starredFrame === visibleSliceIdx;
              const starElsewhere = starredFrame >= 0 && !isStarredHere;
              const tooltip = isStarredHere
                ? `★ Starred. Click to unstar frame ${visibleSliceIdx + 1}.`
                : starElsewhere
                  ? `Star is on frame ${starredFrame + 1}. Click to move it to frame ${visibleSliceIdx + 1}.`
                  : `Click to mark frame ${visibleSliceIdx + 1} as best for ${panelLabel(i)}.`;
              const hideVisible = cursorInfo?.panelIdx === i;
              return (
                <React.Fragment key={`panel-actions-${i}`}>
                  <IconButton
                    className="show3d-panel-hide-button"
                    size="small"
                    disabled={visiblePanelCount <= 1}
                    onMouseDown={(event) => event.stopPropagation()}
                    onClick={(event) => {
                      event.stopPropagation();
                      setPanelHidden(i, true);
                    }}
                    aria-label={visiblePanelCount <= 1 ? "Cannot hide the last visible panel" : `Hide ${panelLabel(i)}`}
                    title={visiblePanelCount <= 1 ? "Cannot hide the last visible panel" : `Hide ${panelLabel(i)}`}
                    sx={{
                      position: "absolute",
                      top: `${((panelTop + 6) / Math.max(1, canvasH)) * 100}%`,
                      left: `${((panelLeft + 6) / Math.max(1, canvasW)) * 100}%`,
                      width: 20,
                      height: 20,
                      p: 0,
                      opacity: hideVisible ? 1 : 0,
                      transform: hideVisible ? "translateY(0)" : "translateY(-3px)",
                      transition: "opacity 120ms ease, transform 120ms ease, background-color 120ms ease, color 120ms ease",
                      color: visiblePanelCount <= 1 ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.78)",
                      bgcolor: "rgba(0,0,0,0.22)",
                      pointerEvents: hideVisible ? "auto" : "none",
                      textShadow: "0 0 3px rgba(0,0,0,0.8)",
                      zIndex: 3,
                      "&:hover, &:focus-visible": {
                        opacity: 1,
                        bgcolor: "rgba(0,0,0,0.42)",
                        color: "rgba(255,255,255,0.95)",
                      },
                    }}
                  >
                    <VisibilityOffIcon sx={{ fontSize: 15 }} />
                  </IconButton>
                  <button
                    onMouseDown={(event) => event.stopPropagation()}
                    onClick={() => {
                      const cur = Array.from({ length: totalPanelCount }, (_, k) => starred?.[k] ?? -1);
                      cur[i] = isStarredHere ? -1 : visibleSliceIdx;
                      setStarred(cur);
                    }}
                    title={tooltip}
                    aria-label={tooltip}
                    style={{
                      position: "absolute",
                      top: `${((panelTop + 6) / Math.max(1, canvasH)) * 100}%`,
                      left: `calc(${((panelLeft + panelW) / Math.max(1, canvasW)) * 100}% - 26px)`,
                      width: 20, height: 20,
                      padding: 0,
                      border: "none",
                      background: "transparent",
                      cursor: "pointer",
                      fontSize: 18,
                      lineHeight: "20px",
                      textAlign: "center",
                      color: isStarredHere
                        ? "#ffc107"  // bright gold: star IS on this frame
                        : starElsewhere
                          ? "rgba(255, 193, 7, 0.45)"  // faded gold: star elsewhere on this panel
                          : "rgba(255,255,255,0.5)",   // grey: no star on this panel
                      textShadow: "0 0 3px rgba(0,0,0,0.8)",
                      pointerEvents: "auto",
                      userSelect: "none",
                    }}
                  >
                    {isStarredHere ? "★" : "☆"}
                  </button>
                </React.Fragment>
              );
            })}
            {panelChromeVisible && reorderMode && (nPanels || 1) > 1 && visiblePanelIndices.map((panel, slot) => {
              const visibleCount = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(visibleCount);
              const rows = Math.ceil(visibleCount / cols);
              const gap = visibleCount > 1 ? (panelGapPx) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const panelLeft = (slot % cols) * (panelW + gap);
              const panelTop = Math.floor(slot / cols) * (panelH + gap);
              const active = dragOverPanel === panel;
              const draggingThisPanel = reorderDragVisual?.panel === panel;
              return (
                <Box
                  key={`panel-reorder-${panel}`}
                  draggable={reorderMode}
                  role="button"
                  data-show3d-reorder-panel={panel}
                  aria-label={`Move ${panelLabel(panel)}`}
                  title={`Drag to reorder ${panelLabel(panel)}`}
                  onDragStart={(event) => handlePanelDragStart(event, panel)}
                  onDragOver={(event) => handlePanelDragOver(event, panel)}
                  onDrop={handlePanelDrop}
                  onDragEnd={handlePanelDragEnd}
                  onPointerDown={(event) => handlePanelReorderPointerDown(event, panel)}
                  onPointerEnter={(event) => handlePanelReorderPointerEnter(event, panel)}
                  onPointerMove={handlePanelReorderPointerMove}
                  onPointerUp={handlePanelReorderPointerUp}
                  onPointerCancel={cancelPanelReorderPreview}
                  sx={{
                    position: "absolute",
                    top: `${(panelTop / Math.max(1, canvasH)) * 100}%`,
                    left: `${(panelLeft / Math.max(1, canvasW)) * 100}%`,
                    width: `${(panelW / Math.max(1, canvasW)) * 100}%`,
                    height: `${(panelH / Math.max(1, canvasH)) * 100}%`,
                    boxSizing: "border-box",
                    border: `2px solid ${active ? themeColors.accent : "rgba(255,255,255,0.48)"}`,
                    bgcolor: draggingThisPanel ? "rgba(0,0,0,0.28)" : active ? "rgba(79, 195, 247, 0.16)" : "rgba(0,0,0,0.04)",
                    outline: active ? `1px solid ${themeColors.accent}` : "none",
                    opacity: draggingThisPanel ? 0.38 : 1,
                    transform: active ? "translateY(-3px) scale(1.006)" : "translateY(0) scale(1)",
                    transition: "transform 110ms ease, opacity 110ms ease, background-color 110ms ease, border-color 110ms ease, box-shadow 110ms ease",
                    animation: "show3d-reorder-jiggle 220ms ease-in-out infinite alternate",
                    boxShadow: active ? `0 0 0 2px ${themeColors.accent}, 0 8px 18px rgba(0,0,0,0.20)` : "none",
                    cursor: draggedPanelRef.current === panel ? "grabbing" : "grab",
                    pointerEvents: "auto",
                    zIndex: 8,
                  }}
                >
                  <Box
                    sx={{
                      position: "absolute",
                      bottom: 6,
                      left: "50%",
                      transform: "translateX(-50%)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      width: 30,
                      height: 22,
                      borderRadius: 1,
                      bgcolor: "rgba(0,0,0,0.38)",
                      color: "rgba(255,255,255,0.92)",
                      pointerEvents: "none",
                    }}
                  >
                    <DragIndicatorIcon sx={{ fontSize: 18 }} />
                  </Box>
                </Box>
              );
            })}
            {panelChromeVisible && reorderMode && reorderDragVisual && (
              <Box
                ref={reorderGhostRef}
                data-show3d-reorder-ghost={reorderDragVisual.panel}
                aria-hidden="true"
                sx={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: `${reorderDragVisual.width}px`,
                  height: `${reorderDragVisual.height}px`,
                  transform: `translate3d(${reorderDragVisual.x}px, ${reorderDragVisual.y}px, 0)`,
                  boxSizing: "border-box",
                  overflow: "hidden",
                  border: `2px solid ${themeColors.accent}`,
                  bgcolor: reorderDragVisual.imageUrl ? "rgba(0,0,0,0.04)" : "rgba(25,25,25,0.68)",
                  boxShadow: `0 10px 24px rgba(0,0,0,0.32), 0 0 0 1px ${themeColors.accent}`,
                  opacity: 0.9,
                  pointerEvents: "none",
                  zIndex: 12,
                  willChange: "transform",
                }}
              >
                {reorderDragVisual.imageUrl && (
                  <Box
                    sx={{
                      position: "absolute",
                      inset: 0,
                      backgroundImage: `url(${reorderDragVisual.imageUrl})`,
                      backgroundSize: "100% 100%",
                      backgroundPosition: "center",
                      imageRendering: smooth ? "auto" : "pixelated",
                    }}
                  />
                )}
                <Box
                  sx={{
                    position: "absolute",
                    top: 6,
                    left: 8,
                    right: 8,
                    px: 0.75,
                    py: 0.25,
                    borderRadius: 0.75,
                    bgcolor: "rgba(0,0,0,0.48)",
                    color: "rgba(255,255,255,0.96)",
                    fontFamily: UI_FONT,
                    fontSize: Math.max(8, panelTitleFontSize || 11),
                    fontWeight: 700,
                    lineHeight: 1.2,
                    textAlign: "center",
                    textShadow: "0 1px 2px rgba(0,0,0,0.9)",
                    whiteSpace: "normal",
                    overflow: "visible",
                    textOverflow: "clip",
                    overflowWrap: "anywhere",
                  }}
                >
                  {reorderDragVisual.label}
                </Box>
                <Box
                  sx={{
                    position: "absolute",
                    bottom: 8,
                    left: "50%",
                    transform: "translateX(-50%)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: 34,
                    height: 24,
                    borderRadius: 1,
                    bgcolor: "rgba(0,0,0,0.48)",
                    color: "rgba(255,255,255,0.95)",
                  }}
                >
                  <DragIndicatorIcon sx={{ fontSize: 19 }} />
                </Box>
              </Box>
            )}
            {/* Cursor readout overlay */}
	            {panelChromeVisible && cursorInfo && (() => {
              const visibleCount = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(visibleCount);
              const rows = Math.ceil(visibleCount / cols);
              const gap = visibleCount > 1 ? (panelGapPx) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              const slot = visiblePanelIndices.indexOf(cursorInfo.panelIdx);
              if (slot < 0) return null;
              const col = slot % cols;
              const row = Math.floor(slot / cols);
              const panelLeft = col * (panelW + gap);
              const panelTop = row * (panelH + gap);
              return (
                <Box className="show3d-cursor-readout" sx={{
                  position: "absolute",
                  top: `${((panelTop + 3) / Math.max(1, canvasH)) * 100}%`,
                  right: `calc(${((canvasW - (panelLeft + panelW)) / Math.max(1, canvasW)) * 100}% + 3px)`,
                  bgcolor: "rgba(0,0,0,0.35)",
                  px: 0.5,
                  py: 0.15,
                  opacity: cursorReadoutVisible ? 1 : 0,
                  transform: cursorReadoutVisible ? "translateY(0)" : "translateY(-2px)",
                  transition: "opacity 90ms ease, transform 90ms ease",
                  willChange: "opacity, transform",
                  pointerEvents: "none",
                  minWidth: 78,
                  maxWidth: `calc(${(panelW / Math.max(1, canvasW)) * 100}% - 6px)`,
                  textAlign: "right",
                }}>
                  <Typography sx={{ fontSize: 9, fontFamily: "monospace", fontVariantNumeric: "tabular-nums", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2, overflow: "hidden", textOverflow: "ellipsis" }}>
                    ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                  </Typography>
                </Box>
              );
            })()}
	            {panelChromeVisible && effectiveRoiActive && roiItems.length > 0 && showRoiResizeHint && (
              <Box sx={{ position: "absolute", left: 6, top: 6, px: 0.6, py: 0.25, bgcolor: "rgba(0,0,0,0.45)", pointerEvents: "none" }}>
                <Typography sx={{ fontSize: 9, color: "rgba(255,255,255,0.8)", lineHeight: 1.1 }}>
                  Hover ROI edge to resize
                </Typography>
              </Box>
            )}
            {/* Per-panel resize corner. Empty cells (partial last row) get
                no handle. Each handle scales the whole multi-panel canvas
                (linked behavior). User trait `show_resize_handles` toggles
                visibility. */}
            {showResizeControls && (() => {
              const visibleCount = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(visibleCount);
              const rows = Math.ceil(visibleCount / cols);
              const gap = visibleCount > 1 ? (panelGapPx) : 0;
              const outPanelW = (canvasW - gap * (cols - 1)) / cols;
              const outPanelH = (canvasH - gap * (rows - 1)) / rows;
              return visiblePanelIndices.map((panel, slot) => {
                const col = slot % cols;
                const row = Math.floor(slot / cols);
                const slotX = col * (outPanelW + gap);
                const slotY = row * (outPanelH + gap);
                return (
                  <Box
                      key={`resize-${panel}`}
                      onMouseDown={handleMainResizeStart}
                      title="Resize panels"
                      sx={{
                        position: "absolute",
                        left: `calc(${((slotX + outPanelW) / Math.max(1, canvasW)) * 100}% - 16px)`,
                        top: `calc(${((slotY + outPanelH) / Math.max(1, canvasH)) * 100}% - 16px)`,
                        ...resizeGripSx,
                      }}
                    />
                );
              });
            })()}
            {effectiveShowFft && fftLayoutOverlay && (() => {
              const visibleCount = Math.max(1, visiblePanelCount || 1);
              const cols = panelColsForCount(visibleCount);
              const rows = Math.ceil(visibleCount / cols);
              const gap = visibleCount > 1 ? (panelGapPx) : 0;
              const panelW = (canvasW - gap * (cols - 1)) / cols;
              const panelH = (canvasH - gap * (rows - 1)) / rows;
              return visiblePanelIndices.map((panel, slot) => {
                const panelLeft = (slot % cols) * (panelW + gap);
                const panelTop = Math.floor(slot / cols) * (panelH + gap);
                const insetPad = Math.min(8, Math.max(3, panelW * 0.025));
                const insetMaxW = Math.max(24, panelW - insetPad * 2);
                const insetMaxH = Math.max(20, panelH - insetPad * 2);
                const insetBase = Math.min(insetMaxW, insetMaxH);
                const insetW = Math.max(24, Math.min(insetMaxW, insetBase * resolvedFftOverlaySize));
                const insetH = Math.max(20, Math.min(insetMaxH, insetBase * resolvedFftOverlaySize));
                const topInsetPad = fftOverlayTopInsetPad(insetPad, showPanelTitles, nPanels || 1, panelTitleFontSize);
                const insetX = resolvedFftOverlayPosition.endsWith("right")
                  ? panelLeft + panelW - insetW - insetPad
                  : panelLeft + insetPad;
                const insetY = resolvedFftOverlayPosition.startsWith("bottom")
                  ? panelTop + panelH - insetH - insetPad
                  : panelTop + topInsetPad;
                const previewInsetX = fftOverlayDragPreview ? panelLeft + fftOverlayDragPreview.x : insetX;
                const previewInsetY = fftOverlayDragPreview ? panelTop + fftOverlayDragPreview.y : insetY;
                return (
                  <Box
                    key={`fft-overlay-inset-${panel}`}
                    data-show3d-fft-inset="true"
                    title="Drag to move FFT overlay; Shift-drag to pan FFT detail"
                    onMouseDown={(e) => {
                      if (e.shiftKey) {
                        handleFftInsetPanMouseDown(e);
                      } else {
                        handleFftInsetMouseDown(e, panelLeft, panelTop, panelW, panelH, insetX, insetY, insetW, insetH);
                      }
                    }}
                    onDoubleClick={(e) => { e.preventDefault(); e.stopPropagation(); handleFftReset(); }}
                    onTouchStart={handleFftInsetTouchStart}
                    onTouchMove={handleFftInsetTouchMove}
                    onTouchEnd={handleFftInsetTouchEnd}
                    onTouchCancel={handleFftInsetTouchEnd}
                    role="img"
                    aria-label={`FFT power spectrum overlay for ${panelLabel(panel)}`}
                    sx={{
                      position: "absolute",
                      left: `${(previewInsetX / Math.max(1, canvasW)) * 100}%`,
                      top: `${(previewInsetY / Math.max(1, canvasH)) * 100}%`,
                      width: `${(insetW / Math.max(1, canvasW)) * 100}%`,
                      height: `${(insetH / Math.max(1, canvasH)) * 100}%`,
                      bgcolor: "transparent",
                      border: "1px solid transparent",
                      zIndex: 8,
                      overflow: "hidden",
                      pointerEvents: "auto",
                      cursor: "move",
                      touchAction: "none",
                    }}
                  >
                    <Box
                      data-show3d-fft-move-handle="true"
                      aria-label="Move FFT overlay; snaps to nearest corner"
                      onPointerDown={(e) => handleFftInsetPointerDown(e, panelLeft, panelTop, panelW, panelH, insetX, insetY, insetW, insetH)}
                      onPointerMove={handleFftInsetPointerMove}
                      onPointerUp={handleFftInsetPointerUp}
                      onPointerCancel={(e) => {
                        if (fftOverlayDragRef.current?.pointerId === e.pointerId) {
                          fftOverlayDragRef.current = null;
                          setFftOverlayDragPreview(null);
                        }
                      }}
                      onMouseDown={(e) => handleFftInsetMouseDown(e, panelLeft, panelTop, panelW, panelH, insetX, insetY, insetW, insetH)}
                      sx={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        right: 0,
                        height: Math.min(16, Math.max(10, insetH * 0.18)),
                        zIndex: 2,
                        cursor: "move",
                        background: "linear-gradient(180deg, rgba(0,0,0,0.38), rgba(0,0,0,0))",
                        opacity: 0.65,
                        touchAction: "none",
                        "&:hover": { opacity: 1 },
                      }}
                    />
                    {showZoomIndicator === true && panelChromeVisible && (
                      (() => {
                        const fftView = linkPanels ? { zoom: fftZoom, panX: fftPanX, panY: fftPanY } : getFftViewForPanel(panel);
                        const zoomLabel = formatZoomLabel(fftView.zoom);
                        return (
                      <Box
                        className="quantem-fft-zoom-label"
                        data-show3d-fft-zoom-indicator={panel}
                        data-fft-zoom={zoomLabel}
                        aria-label={`FFT zoom for ${panelLabel(panel)}: ${zoomLabel}`}
                        sx={{
                          position: "absolute",
                          left: Math.min(12, Math.max(5, insetW * 0.08)),
                          bottom: Math.min(7, Math.max(4, insetH * 0.06)),
                          color: "white",
                          fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                          fontSize: Math.max(9, Math.min(14, insetW * 0.1)),
                          fontWeight: 400,
                          fontVariantNumeric: "tabular-nums",
                          lineHeight: 1,
                          textShadow: "1px 1px 2px rgba(0,0,0,0.85)",
                          pointerEvents: "none",
                          userSelect: "none",
                          zIndex: 3,
                        }}
                      >
                        {zoomLabel}
                      </Box>
                        );
                      })()
                    )}
                    {slot === 0 && fftMetricsEnabled && fftQuality && (
                      <Box
                        className="quantem-fft-quality-label"
                        aria-label={`FFT quality: ${formatFftQualityLabel(fftQuality)}`}
                        sx={{
                          position: "absolute",
                          top: 4,
                          left: 5,
                          right: 5,
                          color: "rgba(255,255,255,0.96)",
                          fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                          fontSize: 10,
                          fontWeight: 700,
                          lineHeight: 1.15,
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          textShadow: "1px 1px 0 rgba(0,0,0,0.9), 0 0 3px rgba(0,0,0,0.85)",
                          pointerEvents: "none",
                          userSelect: "none",
                          zIndex: 3,
                        }}
                      >
                        {formatFftQualityLabel(fftQuality)}
                      </Box>
                    )}
                  </Box>
                );
              });
            })()}
          </Box>
          {/* Statistics bar - right below the image. Multi-panel = one row per panel. */}
          {showStats && (
            (localPanelStats && (nPanels || 1) > 1) ? (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", flexDirection: "column", gap: 0.25, width: "100%", maxWidth: canvasW, boxSizing: "border-box", fontFamily: "ui-monospace, monospace" }}>
                {localPanelStats.map((st) => (
                  <Box key={st.panel} sx={{ display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap", maxWidth: "100%" }}>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted, minWidth: 80 }}>{panelTitleContent(st.panel)}</Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.mean)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.min)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.max)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.std)}</Box></Typography>
                  </Box>
                ))}
              </Box>
            ) : (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap", width: "100%", maxWidth: canvasW, boxSizing: "border-box" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.mean : statsMean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.min : statsMin)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.max : statsMax)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.std : statsStd)}</Box></Typography>
              </Box>
            )
          )}
          {/* Line profile sparkline */}
          {profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                onMouseMove={handleProfileMouseMove}
                onMouseLeave={handleProfileMouseLeave}
                style={{ width: "100%", height: profileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
                role="img"
                aria-label="Line intensity profile along the drawn line"
              />
              {showResizeControls && (
                <div
                  onMouseDown={(e) => { e.preventDefault(); setIsResizingProfile(true); setProfileResizeStart({ y: e.clientY, height: profileHeight }); }}
                  style={{ width: "100%", height: 4, cursor: "ns-resize", borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, background: `linear-gradient(to bottom, ${themeColors.border}, transparent)` }}
                />
              )}
            </Box>
          )}
          {/* ROI sparkline plot */}
          {effectiveRoiActive && showRoiPlot && roiPlotData && roiPlotData.byteLength >= 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, boxSizing: "border-box" }}>
              <canvas
                ref={roiPlotCanvasRef}
                style={{ width: "100%", height: 76, display: "block", border: `1px solid ${themeColors.border}` }}
                role="img"
                aria-label="ROI mean intensity over frames"
              />
            </Box>
          )}
          {/* Image controls stay content-sized so multi-panel stacks do not
              create a large empty gutter between display and playback rows. */}
	          {controlsVisible && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", columnGap: `${SPACING.SM}px`, rowGap: `${SPACING.XS}px`, alignItems: "flex-start", justifyContent: "flex-start", width: "fit-content", maxWidth: "100%", boxSizing: "border-box", flexWrap: "wrap" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: isMobileViewport ? "1 1 100%" : "0 0 auto", width: isMobileViewport ? "100%" : "auto", maxWidth: "100%", minWidth: 0, justifyContent: "center" }}>
                {/* True-color figure stacks: hide colormap / intensity / Smooth;
                    paper figures are already final pixels. */}
                {!isRgb && (<>
                {/* Row 1: Scale + Auto + Colorbar */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale</Typography>
                  <Select value={logScale ? "log" : "linear"} onChange={(e) => changeLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Intensity scale (linear or logarithmic)" }}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }} title={perPanelHistogramEnabled ? "Stack-wide auto contrast. Turn off for independent panel clips." : "Automatic percentile-based contrast."}>
                    {perPanelHistogramEnabled ? "Auto stack" : "Auto"}
                  </Typography>
                  <Switch checked={autoContrast} onChange={(e) => handleAutoContrastChange(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": perPanelHistogramEnabled ? "Toggle stack-wide automatic contrast" : "Toggle automatic percentile-based contrast" } }} />
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar</Typography>
                  <Switch checked={showColorbar} onChange={(e) => setShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle colorbar overlay" } }} />
                </Box>
                {/* Row 2: Color + Smooth + Diff */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color</Typography>
                  <Select
                    size="small"
                    value={panelCmapFor(colorTargetPanel)}
                    onChange={(e) => setCmapForPanel(
                      colorTargetPanel,
                      e.target.value,
                    )}
                    MenuProps={themedFastMenuProps}
                    sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }}
                    inputProps={{ "aria-label": nPanels > 1 ? (colorShared ? "Shared colormap for all panels" : "Selected panel colormap") : "Image colormap" }}
                  >
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name} dense>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Smooth</Typography>
                  <Switch checked={smooth} onChange={(e) => setSmooth(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle bilinear smoothing" } }} />
                  {!showDenoise && displayFilterBanner && (
                    /* House rule: an active reduction is never invisible,
                       even with the denoise controls row hidden. */
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent }} title={displayFilterBanner}>
                      {displayFilterBanner.split(" (")[0]}
                    </Typography>
                  )}
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Diff</Typography>
                  <Select value={diffMode} onChange={(e) => setDiffMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Difference mode (off, previous frame, first frame)" }}>
                    <MenuItem value="off">Off</MenuItem>
                    <MenuItem value="previous">Prev</MenuItem>
                    <MenuItem value="first">First</MenuItem>
                  </Select>
                </Box>
                {/* Row 3 (toggle-gated): display-only denoise for sparse map stacks (EDS, low dose) */}
                {showDenoise && (
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  {nPanels > 1 && (
                    <>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }} title="Link denoise settings across all panels. Off edits only the selected panel.">Link Denoise</Typography>
                      <Switch checked={denoiseScopeAll} onChange={() => setDenoiseScope(denoiseScopeAll ? "panel" : "all")} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle linked denoise settings across panels" } }} />
                    </>
                  )}
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }} title="Poisson (Anscombe): count-respecting smoothing for sparse EDS/counting data - recommended with Bin 2, sigma 6-10. Gaussian: simple smooth for decent-dose images. None: raw counts (use for anything quantitative).">Denoise</Typography>
                  <Select size="small" value={denoiseKnobsForPanel(scopedPanelForEdit).mode} onChange={(e) => { const value = String(e.target.value); setDisplayFilter(value); syncDenoisePanelKnob("mode", value); if (normalizeFilterMode(value) !== "none" || (denoiseKnobsForPanel(scopedPanelForEdit).bin || 1) > 1) setDenoiseEnabled(true); }} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 88, fontSize: 10 }} inputProps={{ "aria-label": denoiseScopeAll ? "Display-only denoise method for all panels" : "Display-only denoise method for selected panel" }}>
                    {[["none", "None"], ["gaussian", "Gaussian"], ["anscombe", "Poisson (Anscombe)"]].map(([mode, label]) => (
                      <MenuItem key={mode} value={mode}>{label}</MenuItem>
                    ))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted, minWidth: 40, display: "inline-block" }}>σ {(sigmaDraft ?? denoiseKnobsForPanel(scopedPanelForEdit).sigma).toFixed(1)}</Typography>
                  <Slider
                    value={sigmaDraft ?? denoiseKnobsForPanel(scopedPanelForEdit).sigma}
                    min={0} max={20} step={0.5}
                    onChange={(_, v) => { if (displayFilterOff) { setDisplayFilter("gaussian"); syncDenoisePanelKnob("mode", "gaussian"); } setSigmaDraft(v as number); }}
                    onChangeCommitted={(_, v) => { setDisplaySigma(v as number); syncDenoisePanelKnob("sigma", v as number); setSigmaDraft(null); if (displayFilterOff) { setDisplayFilter("gaussian"); syncDenoisePanelKnob("mode", "gaussian"); } setDenoiseEnabled(true); }}
                    size="small" sx={{ ...sliderStyles.small, width: 60 }}
                    aria-label="Display filter sigma in pixels"
                  />
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }} title="Display-side 2x bin passes for SNR, combined with the denoise method. 1 is lossless.">Bin</Typography>
                  <Select size="small" value={String(denoiseKnobsForPanel(scopedPanelForEdit).bin || 1)} onChange={(e) => { const b = parseInt(e.target.value, 10); setSpatialBin(b); syncDenoisePanelKnob("bin", b); if (b > 1 || normalizeFilterMode(denoiseKnobsForPanel(scopedPanelForEdit).mode) !== "none") setDenoiseEnabled(true); }} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 40, fontSize: 10 }} inputProps={{ "aria-label": denoiseScopeAll ? "Display spatial bin factor for all panels" : "Display spatial bin factor for selected panel" }}>
                    {[1, 2, 4].map((b) => (<MenuItem key={b} value={String(b)}>{b}</MenuItem>))}
                  </Select>
                </Box>
                )}
                {showFrequencyFilter && (
                <Box sx={{ ...controlRow, ...mobileControlRowSx, width: "100%", maxWidth: "100%", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  {nPanels > 1 && (
                    <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, flexWrap: "nowrap" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }} title="Link frequency filter settings across all panels. Off edits only the selected panel.">Link Filter</Typography>
                      <Switch checked={frequencyFilterScopeAll} onChange={() => setFrequencyFilterScope(frequencyFilterScopeAll ? "panel" : "all")} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle linked frequency filter settings across panels" } }} />
                    </Box>
                  )}
                  <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, flexWrap: "nowrap" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }} title="Low-pass removes fine detail; High-pass removes slow background; Band-pass isolates a periodicity.">Filter</Typography>
                    <Select size="small" value={frequencyKnobsForPanel(scopedPanelForEdit).mode} onChange={(event) => { const mode = String(event.target.value); setFrequencyFilter(mode); syncFrequencyPanelKnob("mode", mode); if (mode !== "none") setFrequencyFilterEnabled(true); }} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 84, fontSize: 10 }} inputProps={{ "aria-label": frequencyFilterScopeAll ? "Frequency filter mode for all panels" : "Frequency filter mode for selected panel" }}>
                      <MenuItem value="none">None</MenuItem>
                      <MenuItem value="lowpass">Low-pass</MenuItem>
                      <MenuItem value="highpass">High-pass</MenuItem>
                      <MenuItem value="bandpass">Band-pass</MenuItem>
                    </Select>
                  </Box>
                  {frequencyKnobsForPanel(scopedPanelForEdit).mode === "bandpass" ? (<>
                    <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, flexWrap: "nowrap" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted, minWidth: 84, display: "inline-block" }}>Center {frequencyValueLabel(frequencyDraft ?? frequencyKnobsForPanel(scopedPanelForEdit).center)}</Typography>
                      <Slider value={frequencyDraft ?? frequencyKnobsForPanel(scopedPanelForEdit).center} min={0} max={1} step={0.005} onChange={(_, value) => setFrequencyDraft(value as number)} onChangeCommitted={(_, value) => { setFrequencyFilterCenter(value as number); syncFrequencyPanelKnob("center", value as number); setFrequencyDraft(null); }} size="small" sx={{ ...sliderStyles.small, width: 72 }} aria-label="Band-pass center as fraction of Nyquist" />
                    </Box>
                    <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, flexWrap: "nowrap" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted, minWidth: 80, display: "inline-block" }}>Width {frequencyValueLabel(frequencyKnobsForPanel(scopedPanelForEdit).width)}</Typography>
                      <Slider value={frequencyKnobsForPanel(scopedPanelForEdit).width} min={0.01} max={1} step={0.005} onChange={(_, value) => { setFrequencyFilterWidth(value as number); syncFrequencyPanelKnob("width", value as number); }} size="small" sx={{ ...sliderStyles.small, width: 72 }} aria-label="Band-pass width as fraction of Nyquist" />
                    </Box>
                  </>) : (<>
                    <Box sx={{ display: "inline-flex", alignItems: "center", gap: `${SPACING.XS}px`, flexWrap: "nowrap" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted, minWidth: 84, display: "inline-block" }}>Cutoff {frequencyValueLabel(frequencyDraft ?? frequencyKnobsForPanel(scopedPanelForEdit).cutoff)}</Typography>
                      <Slider value={frequencyDraft ?? frequencyKnobsForPanel(scopedPanelForEdit).cutoff} min={0} max={1} step={0.005} disabled={!frequencyFilterActive(frequencyKnobsForPanel(scopedPanelForEdit).mode)} onChange={(_, value) => setFrequencyDraft(value as number)} onChangeCommitted={(_, value) => { setFrequencyFilterCutoff(value as number); syncFrequencyPanelKnob("cutoff", value as number); setFrequencyDraft(null); }} size="small" sx={{ ...sliderStyles.small, width: 72 }} aria-label="Frequency cutoff as fraction of Nyquist" />
                    </Box>
                  </>)}
                </Box>
                )}
                </>)}
              </Box>
              {/* Playback: 2 rows side-by-side with Display + Histogram. */}
              {(() => { const activeIdx = visibleSliceIdx; return (
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: `${SPACING.XS}px`, flex: "0 1 auto", minWidth: 0, maxWidth: "100%", justifyContent: "center" }}>
                  <Box sx={{ ...controlRow, ...mobileControlRowSx, width: "fit-content", maxWidth: "100%", flexWrap: "nowrap", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, boxSizing: "border-box" }}>
                    <Stack direction="row" spacing={0} sx={{ flexShrink: 0, mr: 0.5 }}>
                      <IconButton size="small" onClick={() => playFromCurrentFrame(-1)} sx={{ color: reverse && playing ? themeColors.accent : themeColors.textMuted, p: 0.25 }} aria-label="Play in reverse" title="Play reverse">
                        <FastRewindIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                      <PlayPauseButton playing={playing} color={themeColors.accent}
                        onToggle={() => { if (playing) pausePlayback(); else playFromCurrentFrame(); }} />
                      <IconButton size="small" onClick={() => playFromCurrentFrame(1)} sx={{ color: !reverse && playing ? themeColors.accent : themeColors.textMuted, p: 0.25 }} aria-label="Play forward" title="Play forward">
                        <FastForwardIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                      <IconButton size="small" onClick={stopPlayback} sx={{ color: themeColors.textMuted, p: 0.25 }} aria-label="Stop and rewind to start" title="Stop">
                        <StopIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Stack>
                    {loop ? (
                      <Slider ref={playbackSliderRef} value={[loopStart, activeIdx, effectiveLoopEnd]} onPointerDownCapture={handleLoopSliderPointerDownCapture} onChange={(_, v) => { const vals = v as number[]; if (vals[0] !== loopStart) setLoopStart(vals[0]); scrubToSlice(vals[1]); if (vals[2] !== effectiveLoopEnd) setLoopEnd(vals[2]); }} onChangeCommitted={(_, v) => { const vals = v as number[]; if (vals[0] !== loopStart) setLoopStart(vals[0]); commitSlice(vals[1]); if (vals[2] !== effectiveLoopEnd) setLoopEnd(vals[2]); }} disableSwap min={0} max={nSlices - 1} size="small" valueLabelDisplay="auto" valueLabelFormat={(v) => formatFrameValueLabel(v)} marks={bookmarkedFrameMarks} aria-label={`Loop range and current ${dimLabel.toLowerCase()} (frame ${activeIdx + 1} of ${nSlices}, loop ${loopStart + 1} to ${effectiveLoopEnd + 1})`} sx={{ ...sliderStyles.small, width: 150, flex: "0 1 150px", minWidth: 90, "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: themeColors.textMuted }, "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 }, "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: themeColors.textMuted }, "& .MuiSlider-mark": { bgcolor: "#ffc107", width: 5, height: 5, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px", maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }} />
                    ) : (
                      <Slider ref={playbackSliderRef} value={activeIdx} onPointerDownCapture={handleLoopSliderPointerDownCapture} onChange={(_, v) => scrubToSlice(v as number)} onChangeCommitted={(_, v) => commitSlice(v as number)} min={0} max={nSlices - 1} size="small" valueLabelDisplay="auto" valueLabelFormat={(v) => formatFrameValueLabel(v)} marks={bookmarkedFrameMarks} aria-label={`Current ${dimLabel.toLowerCase()} (${activeIdx + 1} of ${nSlices})`} sx={{ ...sliderStyles.small, width: 150, flex: "0 1 150px", minWidth: 90, "& .MuiSlider-mark": { bgcolor: "#ffc107", width: 5, height: 5, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" }, "& .MuiSlider-valueLabel": { maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }} />
                    )}
                    <span
                      ref={playbackLiveCountRef}
                      data-show3d-playback-count="true"
                      style={{
                        fontSize: 10,
                        fontFamily: UI_FONT,
                        color: themeColors.textMuted,
                        minWidth: `${String(nSlices).length * 2 + 1}ch`,
                        fontVariantNumeric: "tabular-nums",
                        textAlign: "right",
                        flexShrink: 0,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {`${activeIdx + 1}/${nSlices}`}
                    </span>
                    <IconButton size="small" onClick={toggleCurrentFrameBookmark} aria-pressed={currentFrameBookmarked} aria-label={`${currentFrameBookmarked ? "Unstar" : "Star"} frame ${activeIdx + 1}`} title={`${currentFrameBookmarked ? "Unstar" : "Star"} frame ${activeIdx + 1}`} sx={{ color: currentFrameBookmarked ? "#ffc107" : themeColors.textMuted, p: 0.25, width: 22, height: 22, flexShrink: 0, "&:hover": { color: currentFrameBookmarked ? "#ffc107" : themeColors.text } }}>
                      <Box component="span" sx={{ fontSize: 18, lineHeight: "18px" }}>{currentFrameBookmarked ? "★" : "☆"}</Box>
                    </IconButton>
                    <Badge
                      badgeContent={playbackPathLength > 0 ? 1 : 0}
                      invisible={playbackPathLength === 0}
                      sx={{ "& .MuiBadge-badge": { bgcolor: themeColors.accent, color: "#fff", fontSize: 9, fontWeight: 600, minWidth: 12, height: 12, px: 0.25 } }}
                    >
                      <Button
                        size="small"
                        sx={{ minWidth: 0, px: 0.6, py: 0.1, fontSize: 10, lineHeight: 1.2, textTransform: "none", color: playbackPathLength > 0 ? themeColors.accent : themeColors.textMuted, flexShrink: 0 }}
                        onClick={(e) => setPlaybackStyleMenuAnchor(e.currentTarget)}
                        aria-label="More playback style options"
                        aria-haspopup="menu"
                        title={`Playback style: ${playbackStyleSummary}`}
                      >
                        More
                      </Button>
                    </Badge>
                    <Menu
                      anchorEl={playbackStyleMenuAnchor}
                      open={Boolean(playbackStyleMenuAnchor)}
                      onClose={() => setPlaybackStyleMenuAnchor(null)}
                      MenuListProps={{ "aria-label": "Playback style options" }}
                      {...themedMenuProps}
                    >
                      <Box sx={{ px: 1.5, py: 0.75, minWidth: 240 }}>
                        <Typography sx={{ fontSize: 11, fontWeight: 700, color: themeColors.text, mb: 0.5 }}>Play Style</Typography>
                        <Typography sx={{ fontSize: 10, color: themeColors.textMuted, mb: 0.75 }} title="Changes only playback_path. Loop, Bounce, fps, and range stay user-controlled in the playback row.">
                          {playbackStyleSummary} · uses current range
                        </Typography>
                        <Box sx={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 0.5 }}>
                          <Button size="small" sx={playbackStyleButtonSx("linear")} aria-pressed={playbackStyleActive === "linear"} onClick={() => applyPlaybackStylePreset("linear")} title="Use the current loop range at constant frame spacing.">Linear</Button>
                          <Button size="small" sx={playbackStyleButtonSx("power-in")} aria-pressed={playbackStyleActive === "power-in"} onClick={() => applyPlaybackStylePreset("power-in")} title="Start slowly, then accelerate through the current range.">Power In</Button>
                          <Button size="small" sx={playbackStyleButtonSx("power-out")} aria-pressed={playbackStyleActive === "power-out"} onClick={() => applyPlaybackStylePreset("power-out")} title="Move quickly at first, then settle near the end of the current range.">Power Out</Button>
                          <Button size="small" sx={playbackStyleButtonSx("ease-in-out")} aria-pressed={playbackStyleActive === "ease-in-out"} onClick={() => applyPlaybackStylePreset("ease-in-out")} title="Smoothly accelerate, then decelerate through the current range.">Ease In/Out</Button>
                        </Box>
                      </Box>
                    </Menu>
                  </Box>
                  <Box sx={{ ...controlRow, ...mobileControlRowSx, width: "fit-content", maxWidth: "100%", flexWrap: "wrap", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, boxSizing: "border-box" }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>fps</Typography>
                      <Slider value={playbackFps} min={1} max={MAX_PLAYBACK_FPS} step={1} onChange={(_, v) => setPlaybackFps(v as number)} size="small" sx={{ ...sliderStyles.small, width: isMobileViewport ? 40 : 44, mx: isMobileViewport ? "3px" : 0, flexShrink: 0 }} aria-label="Playback frames per second" valueLabelDisplay="auto" />
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, minWidth: isMobileViewport ? 16 : 20, flexShrink: 0 }}>{Math.round(playbackFps)}</Typography>
                    </Box>
                    <Box
                      title="Moving average window"
                      sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}
                    >
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>avg</Typography>
                      <Slider
                        value={avgWindow}
                        min={1}
                        max={15}
                        step={1}
                        onChange={(_, v) => setAvgWindow(v as number)}
                        size="small"
                        sx={{ ...sliderStyles.small, width: isMobileViewport ? 40 : 44, mx: isMobileViewport ? "3px" : 0, flexShrink: 0 }}
                        aria-label="Moving average window"
                        valueLabelDisplay="auto"
                      />
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, minWidth: 16, flexShrink: 0 }}>{Math.round(avgWindow || 1)}</Typography>
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>Loop</Typography>
                      <Switch size="small" checked={loop} onChange={() => setLoop(!loop)} sx={{ ...switchStyles.small, flexShrink: 0 }} slotProps={{ input: { "aria-label": "Toggle loop playback" } }} />
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", gap: isMobileViewport ? "4px" : `${SPACING.SM}px`, flexShrink: 0 }}>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: isMobileViewport ? 10 : typography.label.fontSize, flexShrink: 0 }}>Bounce</Typography>
                      <Switch size="small" checked={boomerang} onChange={() => setBoomerang(!boomerang)} sx={{ ...switchStyles.small, flexShrink: 0 }} slotProps={{ input: { "aria-label": "Toggle bounce playback" } }} />
                    </Box>
                  </Box>
                </Box>
              ); })()}
              {/* Intensity histogram + clip sliders are gray-only (colormap window). */}
              {!isRgb && (() => {
                // Global stack range from Python (data_min/data_max trait), not per-frame.
                // Log mode: log1p the range so bins line up with the log-scaled frame data.
                const { min: histMin, max: histMax } = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
                if (perPanelHistogramEnabled) {
                  const visibleCount = Math.max(1, visiblePanelCount || 1);
                  const cols = panelColsForCount(visibleCount);
                  // Match Show2D shell exactly (width=110, height=58, gap=15px)
                  // so the per-panel histogram strip is visually consistent
                  // across widgets.
                  const panelHistWidth = 110;
                  const panelHistGap = 15;
                  const panelHistMaxWidth = cols * panelHistWidth + Math.max(0, cols - 1) * panelHistGap;
                  return (
                    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start", gap: 0.5, opacity: 1, pointerEvents: "auto", maxWidth: "100%" }}>
                      <Box sx={{ display: "grid", gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${panelHistWidth}px), ${panelHistWidth}px))`, gap: `${panelHistGap}px`, width: "100%", maxWidth: panelHistMaxWidth, justifyContent: "start" }}>
                      {visiblePanelIndices.map((panel) => {
                        const state = panelStates[panel] || initialState;
                        // Per-panel histogram uses THIS panel's data range
                        // (not stack-wide histMin/histMax). Tight-range
                        // modalities (SSB phase) get a sensible slider
                        // space instead of being squashed by DF counts.
                        const panelDataRange = panelDataRanges[panel];
                        const panelRange = (panelDataRange && panelDataRange.max > panelDataRange.min) ? panelDataRange : { min: histMin, max: histMax };
                        const vminPct = state.imageVminPct;
                        const vmaxPct = state.imageVmaxPct;
                        return (
                          <Histogram
                            key={`panel-hist-${panel}`}
                            bins={panelHistogramBins[panel] ?? null}
                            vminPct={vminPct}
                            vmaxPct={vmaxPct}
                            onRangeChange={(min, max) => {
                              panelHistogramPreviewPctRef.current.set(panel, [min, max]);
                              const commitPanelRange = () => {
                                setPanelRangePercentages(panel, min, max, linkContrast);
                                if (autoContrast) {
                                  manualImageRangeBeforeAutoRef.current = null;
                                  setAutoContrast(false);
                                }
                                const live = playRef.current;
                                live.autoContrast = false;
                                live.panelStates = panelStatesLiveRef.current;
                                live.vminPerPanel = vminPerPanelLiveRef.current;
                                live.vmaxPerPanel = vmaxPerPanelLiveRef.current;
                                scheduleHistogramPreviewPaint();
                              };
                              commitPanelRange();
                            }}
                            width={110}
                            height={58}
                            valueDecimals={2}
                            labelFontFamily={UI_FONT}
                            theme={themeInfo.theme === "dark" ? "dark" : "light"}
                            dataMin={panelRange.min}
                            dataMax={panelRange.max}
                          />
                        );
                      })}
                      </Box>
                    </Box>
                  );
                }
                return (
                <Box sx={{
                  // Match Show2D histogram shell exactly so visual stays consistent
                  // across widgets. alignItems: flex-end (not stretch) prevents the
                  // inner Slider thumbs from overflowing onto the canvas, which was
                  // the source of the "2.8 tooltip overlaps bars" overlap bug.
                  display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start", gap: 0.5,
                }}>
                  <Histogram
                    bins={imageHistogramBins}
                    vminPct={imageVminPct}
                    vmaxPct={imageVmaxPct}
                    onRangeChange={(min, max) => {
                      imageHistogramPreviewPctRef.current = [min, max];
                      const commitSharedRange = () => {
                        const live = playRef.current;
                        live.imageVminPct = min;
                        live.imageVmaxPct = max;
                        live.autoContrast = false;
                        setImageVminPct(min);
                        setImageVmaxPct(max);
                        if (autoContrast) {
                          manualImageRangeBeforeAutoRef.current = null;
                          setAutoContrast(false);
                        }
                        scheduleHistogramPreviewPaint();
                      };
                      commitSharedRange();
                    }}
                    width={110}
                    height={58}
                    valueDecimals={2}
                    labelFontFamily={UI_FONT}
                    theme={themeInfo.theme === "dark" ? "dark" : "light"}
                    dataMin={histMin}
                    dataMax={histMax}
                  />
                </Box>
                );
              })()}
            </Box>
          )}
          {/* Lens settings row (when Lens is active) */}
          {showLens && (
            <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: "fit-content" }}>
              <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Lens {lensMag}×</Typography>
                <Slider value={lensMag} min={2} max={8} step={1} onChange={(_, v) => setLensMag(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} aria-label="Lens magnification" valueLabelDisplay="auto" />
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>{lensDisplaySize}px</Typography>
                <Slider value={lensDisplaySize} min={64} max={256} step={16} onChange={(_, v) => setLensDisplaySize(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} aria-label="Lens display size in pixels" valueLabelDisplay="auto" />
              </Box>
            </Box>
          )}
          {/* ROI settings row (when ROI is active) */}
          {effectiveRoiActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: "fit-content" }}>
              <Box sx={{ border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, px: 1, py: 0.5, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>
                {/* ROI: shape + add/duplicate + plot + dim */}
                <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>ROI</Typography>
                  <Select
                    size="small"
                    value={newRoiShape}
                    onChange={(e) => setNewRoiShape(e.target.value as "circle" | "square" | "rectangle" | "annular")}
                    MenuProps={themedMenuProps}
                    sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                    inputProps={{ "aria-label": "New ROI shape" }}
                  >
                    {(["square", "rectangle", "circle", "annular"] as const).map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                  </Select>
                  <Button size="small" sx={compactButton} onClick={() => addROIAt(height / 2, width / 2)} aria-label="Add ROI at image center">Add</Button>
                  <Button size="small" sx={compactButton} disabled={!selectedRoi} onClick={duplicateSelectedROI} aria-label="Duplicate selected ROI">Dup</Button>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Plot</Typography>
                  <Switch checked={showRoiPlot} onChange={(e) => setShowRoiPlot(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle ROI intensity plot" } }} />
                  <Box sx={{ flex: 1 }} />
                  <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 24, color: "#ef5350" }} disabled={!roiItems.length} onClick={() => { setRoiList([]); setRoiSelectedIdx(-1); }} aria-label="Clear all ROIs">Clear</Button>
                </Box>

                {/* Selected ROI details */}
                {selectedRoi && (
                  <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: `${SPACING.SM}px`, borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: selectedRoi.color }}>#{roiSelectedIdx + 1}/{roiItems.length}</Typography>
                    <Select
                      size="small"
                      value={selectedRoi.shape || "circle"}
                      onChange={(e) => updateSelectedRoi({ shape: String(e.target.value) })}
                      MenuProps={themedMenuProps}
                      sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                      inputProps={{ "aria-label": "Selected ROI shape" }}
                    >
                      {(["square", "rectangle", "circle", "annular"] as const).map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                    </Select>
                    {selectedRoi.shape === "rectangle" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>W</Typography>
                        <Slider value={selectedRoi.width} min={5} max={width} onChange={(_, v) => updateSelectedRoi({ width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="ROI width" valueLabelDisplay="auto" />
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>H</Typography>
                        <Slider value={selectedRoi.height} min={5} max={height} onChange={(_, v) => updateSelectedRoi({ height: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="ROI height" valueLabelDisplay="auto" />
                      </>
                    )}
                    {selectedRoi.shape === "annular" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Inner</Typography>
                        <Slider value={selectedRoi.radius_inner} min={1} max={Math.max(2, selectedRoi.radius - 1)} onChange={(_, v) => updateSelectedRoi({ radius_inner: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="Annular ROI inner radius" valueLabelDisplay="auto" />
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Outer</Typography>
                        <Slider value={selectedRoi.radius} min={selectedRoi.radius_inner + 1} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="Annular ROI outer radius" valueLabelDisplay="auto" />
                      </>
                    )}
                    {selectedRoi.shape !== "rectangle" && selectedRoi.shape !== "annular" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Size</Typography>
                        <Slider value={selectedRoi.radius} min={5} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 50 }} aria-label="ROI radius" valueLabelDisplay="auto" />
                      </>
                    )}
                    <Box sx={{ display: "flex", gap: "2px" }}>
                      {ROI_COLORS.map(c => (
                        <Box key={c} onClick={() => updateSelectedRoi({ color: c })} sx={{ width: 12, height: 12, bgcolor: c, cursor: "pointer", border: c === selectedRoi.color ? `2px solid ${themeColors.text}` : "1px solid transparent", "&:hover": { opacity: 0.8 } }} />
                      ))}
                    </Box>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Border</Typography>
                    <Slider value={selectedRoi.line_width} min={1} max={6} step={1} onChange={(_, v) => updateSelectedRoi({ line_width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 30 }} aria-label="ROI border line width" valueLabelDisplay="auto" />
                    <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 20, color: "#ef5350" }} onClick={deleteSelectedROI} aria-label="Delete selected ROI">&times;</Button>
                  </Box>
                )}

                {/* ROI list */}
                {roiItems.length > 0 && (
                  <Box sx={{ display: "flex", flexDirection: "column", borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                    {roiItems.map((roi, i) => {
                      const c = roi.color || ROI_COLORS[i % ROI_COLORS.length];
                      const isSelected = i === roiSelectedIdx;
                      const shapeLabel = roi.shape === "rectangle" ? `${roi.width}×${roi.height}` : roi.shape === "annular" ? `r${roi.radius_inner}-${roi.radius}` : `r${roi.radius}`;
                      return (
                        <Box key={i} onClick={() => setRoiSelectedIdx(i)} sx={{ display: "flex", alignItems: "center", gap: "3px", lineHeight: 1.6, cursor: "pointer", "&:hover .roi-delete": { opacity: 1 } }}>
                          <Box sx={{ width: 8, height: 8, borderRadius: roi.shape === "square" || roi.shape === "rectangle" ? 0 : "50%", bgcolor: c, border: isSelected ? "2px solid #fff" : "1px solid transparent", flexShrink: 0 }} />
                          <Typography component="span" sx={{ fontSize: 10, color: isSelected ? themeColors.text : themeColors.textMuted, fontWeight: isSelected ? "bold" : "normal" }}>
                            <Box component="span" sx={{ color: c }}>{i + 1}</Box>{" "}
                            {roi.shape} ({Math.round(roi.row)}, {Math.round(roi.col)}) {shapeLabel}
                          </Typography>
                          <Box
                            onClick={(e) => { e.stopPropagation(); const newList = roiItems.map((r, j) => ({ ...r, highlight: j === i ? !r.highlight : false })); setRoiList(newList); }}
                            sx={{ cursor: "pointer", fontSize: 10, color: roi.highlight ? themeColors.accentGreen : themeColors.textMuted, lineHeight: 1, opacity: roi.highlight ? 1 : 0.5, "&:hover": { opacity: 1 } }}
                            title="Focus (dim outside)"
                          >{roi.highlight ? "\u25C9" : "\u25CB"}</Box>
                          <Box
                            className="roi-delete"
                            onClick={(e) => { e.stopPropagation(); const newList = roiItems.filter((_, j) => j !== i); setRoiList(newList); setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1); }}
                            sx={{ opacity: 0, cursor: "pointer", fontSize: 10, color: themeColors.textMuted, ml: 0.5, lineHeight: 1, "&:hover": { color: "#f44336" } }}
                          >&times;</Box>
                        </Box>
                      );
                    })}
                  </Box>
                )}
              </Box>
            </Box>
          )}
        </Box>

        {/* Preview Panel - ROI crop at full resolution with aspect ratio */}
        {previewVisible && (
          <Box sx={{ width: "100%", maxWidth: canvasW, boxSizing: "border-box" }}>
            {/* Spacer - matches main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {/* Header row - matches main panel controls row height */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 28, height: "auto", flexWrap: "wrap", gap: `${SPACING.XS}px` }}>
              <Typography sx={{ ...typography.label, color: themeColors.accentGreen }}>
                Preview{previewCropDims ? ` (${previewCropDims.w}\u00d7${previewCropDims.h})` : ""}
              </Typography>
              <Button size="small" sx={compactButton} disabled={previewZoom.zoom === 1 && previewZoom.panX === 0 && previewZoom.panY === 0} onClick={handlePreviewDoubleClick} aria-label="Reset preview zoom and pan">Reset</Button>
            </Stack>
            <Box
              ref={previewContainerRef}
              sx={{
                position: "relative",
                bgcolor: "#000",
                border: `1px solid ${themeColors.border}`,
                cursor: "grab",
                width: "100%",
                maxWidth: previewCanvasDims.w,
                aspectRatio: `${Math.max(previewCanvasDims.w, 1)} / ${Math.max(previewCanvasDims.h, 1)}`,
                height: "auto",
              }}
              onWheel={handlePreviewWheel}
              onDoubleClick={handlePreviewDoubleClick}
              onMouseDown={handlePreviewMouseDown}
              onMouseMove={handlePreviewMouseMove}
              onMouseUp={handlePreviewMouseUp}
              onMouseLeave={handlePreviewMouseUp}
            >
              <canvas ref={previewCanvasRef} width={previewCanvasDims.w} height={previewCanvasDims.h} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: "pixelated" }} role="img" aria-label={`ROI preview crop${previewCropDims ? ` (${previewCropDims.w} by ${previewCropDims.h} pixels)` : ""}`} />
              <canvas ref={previewOverlayRef} width={Math.round(previewCanvasDims.w * DPR)} height={Math.round(previewCanvasDims.h * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
              {showResizeControls && (
                <Box onMouseDown={handleMainResizeStart} title="Resize image" sx={{ position: "absolute", bottom: 0, right: 0, ...resizeGripSx }} />
              )}
            </Box>
            {/* All-ROI Stats - one row per ROI, same style as main stats bar */}
            {showStats && allRoiStats.length > 0 && (
              <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: 0.5, width: "100%", maxWidth: previewCanvasDims.w, boxSizing: "border-box" }}>
                {allRoiStats.map((stats, i) => {
                  if (!stats) return null;
                  const color = roiItems[i]?.color || ROI_COLORS[i % ROI_COLORS.length];
                  const isSelected = i === roiSelectedIdx;
                  return (
                    <Box key={i} sx={{ px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap", border: isSelected ? `1px solid ${color}` : `1px solid transparent` }}>
                      <Box sx={{ width: 8, height: 8, bgcolor: color, borderRadius: "50%", flexShrink: 0 }} />
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color }}>{formatNumber(stats.mean)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color }}>{formatNumber(stats.min)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color }}>{formatNumber(stats.max)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color }}>{formatNumber(stats.std)}</Box></Typography>
                    </Box>
                  );
                })}
              </Box>
            )}
          </Box>
        )}

        {/* FFT Panel - same size as main image. Bottom stacks below; Right uses the side slot. */}
        {effectiveShowFft && !fftLayoutOverlay && (
          <Box sx={{
            width: "100%",
            maxWidth: fftLayoutBottom ? "100%" : canvasW,
            flex: fftLayoutBottom ? "1 0 100%" : `0 1 min(100%, ${canvasW}px)`,
            minWidth: fftLayoutBottom ? "100%" : undefined,
            ml: fftLayoutBottom ? "0 !important" : undefined,
            mt: fftLayoutBottom ? "0 !important" : undefined,
            boxSizing: "border-box",
          }}>
            {/* Spacer - matches main panel title row height for canvas alignment */}
            {!fftLayoutBottom && <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />}
            {!fftLayoutBottom && controlsVisible && isPaged && (
              <Box
                aria-hidden="true"
                sx={{
                  minHeight: 28,
                  mb: "3px",
                  pb: "3px",
                  borderBottom: `1px solid ${themeColors.border}`,
                }}
              />
            )}
            {/* Controls row - mirrors the measured main toolbar height, including wraps. */}
            {(!fftLayoutBottom || (roiFftActive && fftCropDims)) && (
              <Stack
                direction="row"
                justifyContent="space-between"
                alignItems="center"
                data-show3d-fft-tool-spacer="true"
                sx={{
                  mb: `${SPACING.XS}px`,
                  minHeight: 28,
                  height: !fftLayoutBottom && controlsVisible ? toolControlsHeight : "auto",
                  flexWrap: "wrap",
                  gap: `${SPACING.XS}px`,
                }}
              >
                {roiFftActive && fftCropDims ? (
                  <Typography sx={{ ...typography.label, color: themeColors.accentGreen }}>
                    ROI FFT ({fftCropDims.cropWidth}&times;{fftCropDims.cropHeight})
                  </Typography>
                ) : <Box />}
              </Stack>
            )}
            {/* FFT Canvas - same size as main image */}
            <Box
              ref={fftContainerRef}
              sx={{
                ...container.imageBox,
                width: "100%",
                maxWidth: canvasW,
                aspectRatio: mainPanelAspectRatio,
                height: "auto",
                cursor: "grab",
                touchAction: "none",
              }}
              onMouseDown={handleFftMouseDown}
              onMouseMove={handleFftMouseMove}
              onMouseUp={handleFftMouseUp}
              onMouseLeave={() => { fftClickStartRef.current = null; setIsFftDragging(false); setFftPanStart(null); }}
              onWheel={handleFftWheel}
              onDoubleClick={handleFftReset}
              onTouchStart={handleFftTouchStart}
              onTouchMove={handleFftTouchMove}
              onTouchEnd={handleFftTouchEnd}
              onTouchCancel={handleFftTouchEnd}
            >
              <canvas data-quantem-scientific-output="show3d-fft" ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: smooth ? "auto" : "pixelated", touchAction: "none" }} role="img" aria-label={roiFftActive && fftCropDims ? `FFT power spectrum of ROI crop (${fftCropDims.cropWidth} by ${fftCropDims.cropHeight} pixels)` : "FFT power spectrum of current frame"} />
              <canvas ref={fftOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
              {show3dFrequencyRing}
              {showZoomIndicator === true && panelChromeVisible && (() => {
                const visibleCount = Math.max(1, visiblePanelCount || 1);
                const cols = panelColsForCount(visibleCount);
                const rows = Math.ceil(visibleCount / cols);
                const gap = visibleCount > 1 ? (panelGapPx) : 0;
                const outPanelW = (canvasW - gap * (cols - 1)) / cols;
                const outPanelH = (canvasH - gap * (rows - 1)) / rows;
                return visiblePanelIndices.map((panel, slot) => {
                  const col = slot % cols;
                  const row = Math.floor(slot / cols);
                  const slotX = col * (outPanelW + gap);
                  const slotY = row * (outPanelH + gap);
                  const fftView = linkPanels ? { zoom: fftZoom, panX: fftPanX, panY: fftPanY } : getFftViewForPanel(panel);
                  const zoomLabel = formatZoomLabel(fftView.zoom);
                  return (
                    <Box
                      key={`fft-zoom-${panel}`}
                      className="quantem-fft-zoom-label"
                      data-show3d-fft-zoom-indicator={panel}
                      data-fft-zoom={zoomLabel}
                      aria-label={`FFT zoom for ${panelLabel(panel)}: ${zoomLabel}`}
                      sx={{
                        position: "absolute",
                        left: `calc(${(slotX / Math.max(1, canvasW)) * 100}% + 12px)`,
                        top: `calc(${((slotY + outPanelH) / Math.max(1, canvasH)) * 100}% - 23px)`,
                        maxWidth: `calc(${(outPanelW / Math.max(1, canvasW)) * 100}% - 24px)`,
                        color: "white",
                        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                        fontSize: 16,
                        fontWeight: 400,
                        fontVariantNumeric: "tabular-nums",
                        lineHeight: 1,
                        textShadow: "1px 1px 2px rgba(0,0,0,0.85)",
                        pointerEvents: "none",
                        userSelect: "none",
                        zIndex: 4,
                      }}
                    >
                      {zoomLabel}
                    </Box>
                  );
                });
              })()}
              {fftMetricsEnabled && fftQuality && (
                <Box
                  className="quantem-fft-quality-label"
                  aria-label={`FFT quality: ${formatFftQualityLabel(fftQuality)}`}
                  sx={{
                    position: "absolute",
                    top: 8,
                    left: 8,
                    maxWidth: "calc(100% - 16px)",
                    color: "rgba(255,255,255,0.96)",
                    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    fontSize: 11,
                    fontWeight: 700,
                    lineHeight: 1.2,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    textShadow: "1px 1px 0 rgba(0,0,0,0.9), 0 0 3px rgba(0,0,0,0.85)",
                    pointerEvents: "none",
                    userSelect: "none",
                    zIndex: 4,
                  }}
                >
                  {formatFftQualityLabel(fftQuality)}
                </Box>
              )}
              {showResizeControls && (() => {
                const visibleCount = Math.max(1, visiblePanelCount || 1);
                const cols = panelColsForCount(visibleCount);
                const rows = Math.ceil(visibleCount / cols);
                const gap = visibleCount > 1 ? (panelGapPx) : 0;
                const outPanelW = (canvasW - gap * (cols - 1)) / cols;
                const outPanelH = (canvasH - gap * (rows - 1)) / rows;
                return visiblePanelIndices.map((panel, slot) => {
                  const col = slot % cols;
                  const row = Math.floor(slot / cols);
                  const slotX = col * (outPanelW + gap);
                  const slotY = row * (outPanelH + gap);
                  return (
                    <Box
                      key={`fft-resize-${panel}`}
                      onMouseDown={handleMainResizeStart}
                      title="Resize FFT panels"
                      sx={{
                        position: "absolute",
                        left: `calc(${((slotX + outPanelW) / Math.max(1, canvasW)) * 100}% - 16px)`,
                        top: `calc(${((slotY + outPanelH) / Math.max(1, canvasH)) * 100}% - 16px)`,
                        ...resizeGripSx,
                      }}
                    />
                  );
                });
              })()}
            </Box>
            {/* FFT Statistics bar */}
            {showStats && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, flexWrap: "wrap" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.mean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.min)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.max)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.std)}</Box></Typography>
              </Box>
            )}
            {fftClickInfo && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, border: `1px solid ${themeColors.border}`, display: "flex", gap: 1.25, alignItems: "center", flexWrap: "wrap", width: "fit-content", maxWidth: canvasW, boxSizing: "border-box" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted, fontWeight: 600 }}>FFT mark</Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                  {fftClickInfo.dSpacing != null ? (
                    <>d = <Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{formatLength(fftClickInfo.dSpacing, pixelUnit)}</Box>{" | |g| = "}<Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.spatialFreq!.toFixed(4)} {unitSymbol(pixelUnit)}⁻¹</Box></>
                  ) : (
                    <>dist = <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                  )}
                </Typography>
              </Box>
            )}
            {/* FFT Controls - two rows with histogram on right (like Show4DSTEM) */}
	            {controlsVisible && <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", maxWidth: canvasW, boxSizing: "border-box", flexWrap: "wrap" }}>
              {/* Left: two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Auto */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale</Typography>
                  <Select value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT intensity scale (linear or logarithmic)" }}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto</Typography>
                  <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle automatic FFT contrast" } }} />
                  {roiFftActive && fftCropDims && (
                    <>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Win</Typography>
                      <Switch checked={fftWindow} onChange={(e) => setFftWindow(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle Hann windowing before FFT" } }} />
                    </>
                  )}
                </Box>
                {/* Row 2: Color + Colorbar */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT colormap" }}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar</Typography>
                  <Switch checked={fftShowColorbar} onChange={(e) => setFftShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle FFT colorbar overlay" } }} />
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                <Histogram
                  data={fftHistogramData}
                  pinBinsToRange
                  vminPct={fftVminPct}
                  vmaxPct={fftVmaxPct}
                  onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }}
                  width={110}
                  height={58}
                  valueDecimals={2}
                  labelFontFamily={UI_FONT}
                  theme={themeInfo.theme}
                  dataMin={fftDataRange.min}
                  dataMax={fftDataRange.max}
                />
              </Box>
            </Box>}
          </Box>
        )}

        {/* Kymograph Panel - static space-time image (X = distance along line,
            Y = frame/time). Shares the side slot with FFT (mutually exclusive).
            Mirrors the FFT panel's adjustability (contrast, zoom/pan, colormap). */}
        {kymoReady && (
          <Box sx={{ width: "100%", maxWidth: canvasW, boxSizing: "border-box" }}>
            {/* Spacer - matches main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {controlsVisible && isPaged && (
              <Box
                aria-hidden="true"
                sx={{
                  minHeight: 28,
                  mb: "3px",
                  pb: "3px",
                  borderBottom: `1px solid ${themeColors.border}`,
                }}
              />
            )}
            {/* Controls row - title on left, Reset on right */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 28, height: "auto", flexWrap: "wrap", gap: `${SPACING.XS}px` }}>
              <Typography sx={{ ...typography.label, color: themeColors.accentGreen }}>
                Kymograph{singlePanelPageProfile ? ` · ${pageControlLabel}` : ""} ({kymoDataRef.current?.nFrames ?? nSlices} {dimUnit ? unitSymbol(dimUnit) : "frames"} &times; {kymoDataRef.current?.lineLen ?? 0} px)
              </Typography>
              <Button size="small" sx={compactButton} disabled={!kymoNeedsReset} onClick={handleKymoReset} aria-label="Reset kymograph zoom and pan">Reset</Button>
            </Stack>
            {/* Kymograph canvas - same size as main image */}
            <Box
              ref={kymoContainerRef}
              sx={{
                ...container.imageBox,
                width: "100%",
                maxWidth: canvasW,
                aspectRatio: mainPanelAspectRatio,
                height: "auto",
                cursor: "grab",
                position: "relative",
                touchAction: "none",
              }}
              onMouseDown={handleKymoMouseDown}
              onMouseMove={handleKymoMouseMove}
              onMouseUp={handleKymoMouseUp}
              onMouseLeave={() => { kymoClickStartRef.current = null; setIsKymoDragging(false); setKymoPanStart(null); }}
              onDoubleClick={handleKymoReset}
              onTouchStart={handleKymoTouchStart}
              onTouchMove={handleKymoTouchMove}
              onTouchEnd={handleKymoTouchEnd}
              onTouchCancel={handleKymoTouchEnd}
            >
              <canvas ref={kymoCanvasRef} width={canvasW} height={canvasH} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", imageRendering: "pixelated", touchAction: "none" }} role="img" aria-label={`Kymograph${singlePanelPageProfile ? ` for ${pageControlLabel}` : ""}: distance along profile line versus frame index`} />
              <canvas ref={kymoOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
            </Box>
            {/* Axis labels - kymograph-specific footer */}
            <Box sx={{ display: "flex", justifyContent: "space-between", mt: 0.5, px: 0.5 }}>
              <Typography sx={{ fontSize: 9, color: themeColors.textMuted }}>
                {dimUnit ? `time (${unitSymbol(dimUnit)})${dimSampling && dimSampling !== 1 ? `, ${(dimSampling).toFixed(2)}/frame` : ""} ↓` : "frame ↓"}
              </Typography>
              <Typography sx={{ fontSize: 9, color: themeColors.textMuted }}>distance along line →</Typography>
            </Box>
            {/* Kymograph Statistics bar */}
            {showStats && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, flexWrap: "wrap" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.mean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.min)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.max)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoStats.std)}</Box></Typography>
                {kymoClickInfo && (
                  <>
                    <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                      t = <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoClickInfo.timeVal)} {kymoClickInfo.timeUnit}</Box>{" | d = "}<Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(kymoClickInfo.distVal)} {kymoClickInfo.distUnit}</Box>{" | I = "}<Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{formatNumber(kymoClickInfo.intensity)}</Box>
                    </Typography>
                  </>
                )}
              </Box>
            )}
            {/* Kymograph Controls - two rows with histogram on right (mirror FFT) */}
	            {controlsVisible && <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", maxWidth: canvasW, boxSizing: "border-box", flexWrap: "wrap" }}>
              {/* Left: two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Auto */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale</Typography>
                  <Select value={kymoLogScale ? "log" : "linear"} onChange={(e) => setKymoLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Kymograph intensity scale (linear or logarithmic)" }}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto</Typography>
                  <Switch checked={kymoAuto} onChange={(e) => setKymoAuto(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle automatic kymograph contrast" } }} />
                </Box>
                {/* Row 2: Color + Colorbar */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color</Typography>
                  <Select value={kymoColormap} onChange={(e) => setKymoColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Kymograph colormap" }}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar</Typography>
                  <Switch checked={kymoShowColorbar} onChange={(e) => setKymoShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Toggle kymograph colorbar overlay" } }} />
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                <Histogram
                  data={kymoHistogramData}
                  pinBinsToRange
                  vminPct={kymoVminPct}
                  vmaxPct={kymoVmaxPct}
                  onRangeChange={(min, max) => { setKymoVminPct(min); setKymoVmaxPct(max); }}
                  width={110}
                  height={58}
                  valueDecimals={2}
                  labelFontFamily={UI_FONT}
                  theme={themeInfo.theme}
                  dataMin={kymoDataRange.min}
                  dataMax={kymoDataRange.max}
                />
              </Box>
            </Box>}
          </Box>
        )}
      </Stack>
      </>
      )}

    </Box>
  );
}

// anywidget v0.9+ deprecates `export render` in favor of `export default { render }`.
const render = createRender(Show3D);
export default { render };
