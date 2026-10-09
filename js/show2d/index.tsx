/**
 * Show2D - Static 2D image viewer with gallery support.
 * 
 * Features:
 * - Single image or gallery mode with configurable columns
 * - Scroll to zoom, double-click to reset
 * - WebGPU-accelerated FFT with default 2x zoom
 * - Equal-sized FFT and histogram panels
 * - Click to select image in gallery mode
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
import IconButton from "@mui/material/IconButton";
import Badge from "@mui/material/Badge";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import DragIndicatorIcon from "@mui/icons-material/DragIndicator";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { useTheme } from "../theme";
import { useCanvasRepaintSignal } from "../canvasLifecycle";
import { type ScaleBarOptions, drawColorbar, drawScaleBarInRegion, formatZoomLabel, roundToNiceValue, scaleBarGeometry, unitSymbol } from "../figure";
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
import { findDataRange, applyLogScale, signedLog1p, sliderRange, computeStats } from "../display/stats";
import { dequantizeUint8 } from "../display/quantization";
import { MetadataSection } from "../widgetInfo";
import { FolderWatchBadge, useFolderWatchModelLive } from "../folderWatchStatus";
import { getDisplayFFT, DisplayFFT, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2, applyHannWindow2D, reciprocalCoordinatesFromShiftedOffset } from "../display/fft";
import { computeFftQualityMetrics, formatFftQualityLabel, type FftQualityMetrics } from "../display/fftMetrics";
import {
  cropMaskedRegionBrowser,
  findFFTPeakBrowser,
  sampleLineProfileBrowser,
} from "../display/geometry";
import { COLORMAPS, COLORMAP_NAMES, GPUColormapEngine, renderToOffscreen } from "../display/colormaps";
import { colormapSampledPixels, createColormapEngine, createCPUColormapEngine, type DisplayColormapEngine } from "../display/cpuColormap";
import { onGPULost } from "../display/device";
import { applyDisplayFilterBrowser, browserFilterSupported, filterKnobsActive, normalizeFilterMode, resolvePanelDenoiseKnobs } from "../display/filter";
import { applyFrequencyFilterBrowser, frequencyFilterActive, getFrequencyFilterBackend, normalizeFrequencyFilterMode } from "../display/frequencyFilter";
import { InfoTooltip } from "../shared/InfoTooltip";
import { RenderPathBadge } from "../shared/RenderPathBadge";
import { KeyboardShortcuts } from "../shared/KeyboardShortcuts";
import { useMobileViewport } from "../shared/useMobileViewport";
import { exportTitleSlug, formatEstimatedHtmlSize, formatSavedBytes, isAbortLikeError } from "../shared/exportFormat";
import { pointToSegmentDistance } from "../shared/geometry";
import { shouldIgnoreWidgetShortcut } from "../shared/widgetShortcuts";
import { type RichTitleSpan, renderMathExpression, renderRichTitle, richTitlePlainText } from "../shared/latexTitle";
import { type PanelAnnotationSpec, type PanelOverlaySpec, type OverlaySelection, type OverlayDragState, styleNumber, styleString, annotationAnchorTransform, panelAnnotationSx, drawROI, drawPanelOverlays, panelOverlayHit, updateOverlayFromDrag, drawPanelOverlaySelection, normalizeHiddenPageSlots } from "../shared/panelOverlays";
import { Histogram } from "../shared/Histogram";
import {
  GALLERY_FFT_CACHE_MAX_BYTES,
  GALLERY_FFT_CACHE_MAX_ENTRIES,
  clampPanelPlaybackFps,
  makeGalleryFftCacheKey,
  panelPlaybackIntervalMs,
  readGalleryFftCache,
  rememberGalleryFftCache,
  resolveVisibleDiffPlan,
  type GalleryFftCacheEntry,
} from "./localStack";
import {
  itemPageIndices,
  pageShortcutTarget,
  usesGalleryLayout,
} from "./itemPages";
import {
  cancelContrastPreview,
  contrastPreviewIndices,
  enqueueContrastPreview,
  linkedAutoContrastPercentRange,
  seedAutoContrastMirror,
  transitionContrastLinkState,
  type ContrastPreviewQueue,
} from "./contrastPreview";
import { applyPanelTransform, sampledSourceIndex, type PanelPlacement } from "./panelView";
import {
  drawInsetPlot,
  insetHoverAt,
  insetPlotGeometry,
  SYSTEM_FONT,
  type InsetDragState,
  type InsetHoverInfo,
  type InsetPlotSpec,
} from "./insetPlots";
import {
  escapeXmlAttr,
  escapeXmlText,
  svgColor,
  svgColorbarElements,
  svgInsetPlotElement,
  svgPanelAnnotationElement,
  svgPanelOverlayElement,
  svgTextFromRichSpans,
  wrapSvgTextLines,
} from "./svgExport";
import { formatLength } from "../show3d/lengthLabel";

const SHOW2D_STANDALONE_VIEW_STATE_KEYS = [
  "auto_contrast",
  "cmap",
  "col_markers",
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
  "diff_mode",
  "diff_reference",
  "fft_metrics",
  "fft_window",
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
  "gallery_outer_border_color",
  "gallery_outer_border_px",
  "hidden_page_slots",
  "hidden_panels",
  "image_flips_horizontal",
  "image_flips_vertical",
  "image_rotations",
  "initial_zoom",
  "inter_panel_gap_color",
  "inter_panel_gap_px",
  "inset_plots",
  "link_contrast",
  "link_pan",
  "link_zoom",
  "log_scale",
  "marker_colors",
  "marker_style",
  "ncols",
  "page_idx",
  "panel_annotations",
  "panel_cmaps",
  "panel_frame_indices",
  "panel_inner_border_color",
  "panel_inner_border_px",
  "panel_order",
  "panel_overlays",
  "panel_playback_fps",
  "panel_title_font_size",
  "panel_title_spans",
  "panel_title_style",
  "profile_line",
  "roi_active",
  "roi_list",
  "roi_selected_idx",
  "rotation_scope",
  "row_markers",
  "scale_bar_label",
  "scale_bar_length",
  "scale_bar_panels",
  "scale_bar_position",
  "scale_bar_style",
  "scale_bar_visible",
  "selected_idx",
  "selected_panels",
  "show_controls",
  "show_denoise",
  "show_fft",
  "show_frequency_filter",
  "show_inset_plots",
  "show_panel_titles",
  "show_stats",
  "show_title",
  "show_zoom_indicator",
  "smooth",
  "starred",
  "view_box",
  "vmax",
  "vmaxs",
  "vmin",
  "vmins",
  "zoom_col",
  "zoom_row",
] as const;


// One font stack for canvas text, SVG export and panel chrome, so an export
// renders with the same face as the screen.
type PanelTitleStyle = Record<string, unknown>;
type ScaleBarStyle = Record<string, unknown>;
type MarkerMap = Record<string, string>;
type AnnotationSelection = { panel: number; annotation: number };
type AnnotationDragState = {
  panel: number;
  annotation: number;
  startClientX: number;
  startClientY: number;
  panelWidth: number;
  panelHeight: number;
  original: PanelAnnotationSpec;
};
type Show2DSvgExport = {
  svg: string;
  width: number;
  height: number;
  filename: string;
  scale: number;
};
type Show2DSvgPreview = Show2DSvgExport & {
  url: string;
  size: number;
};

/**
 * One colormapped bitmap per panel, each in its own colormap and at its
 * offscreen's size, through the engine's scaled pass. That pass reads packed
 * uint8 slots with the uint8 shader and resamples to the offscreen (packed
 * folder frames are drawn at canvas size), so it serves uint8 folder previews
 * and galleries with mixed panel colormaps; the settled paint and the live
 * contrast preview both use it, so a histogram drag shows what its release paints.
 */
function renderScaledPanelBitmaps(
  engine: DisplayColormapEngine,
  panels: number[],
  ranges: { vmin: number; vmax: number }[],
  logScale: boolean,
  cmapNames: string[],
  offscreens: (HTMLCanvasElement | null | undefined)[],
): Promise<(ImageBitmap | null)[]> {
  return Promise.all(panels.map((panel, k) => {
    const offscreen = offscreens[panel];
    const name = cmapNames[panel];
    return offscreen
      ? engine.renderSlotScaledToImageBitmapAsync(panel, ranges[k] || { vmin: 0, vmax: 1 }, logScale, offscreen.width, offscreen.height, name, COLORMAPS[name] || COLORMAPS.inferno)
      : Promise.resolve(null);
  }));
}

function panelTitleChromeSx(
  style: PanelTitleStyle | undefined,
  defaults: Record<string, unknown> = {},
): Record<string, unknown> {
  const titleStyle = style || {};
  const borderWidth = Math.max(0, styleNumber(titleStyle.border_width, 0));
  const background = styleString(titleStyle.bg);
  const align = styleString(titleStyle.align, String(defaults.textAlign || "center"));
  const maxWidth = titleStyle.max_width;
  const mode = typeof maxWidth === "string" ? maxWidth : "";
  const sx: Record<string, unknown> = {
    ...defaults,
    color: styleString(titleStyle.fg, String(defaults.color || "rgba(255,255,255,0.95)")),
    bgcolor: background || defaults.bgcolor,
    border: borderWidth > 0 ? `${borderWidth}px solid ${styleString(titleStyle.border_color, "rgba(255,255,255,0.35)")}` : defaults.border,
    borderRadius: titleStyle.radius != null ? `${Math.max(0, styleNumber(titleStyle.radius, 0))}px` : defaults.borderRadius,
    px: titleStyle.pad_x != null ? `${Math.max(0, styleNumber(titleStyle.pad_x, 0))}px` : defaults.px,
    py: titleStyle.pad_y != null ? `${Math.max(0, styleNumber(titleStyle.pad_y, 0))}px` : defaults.py,
    fontWeight: titleStyle.font_weight != null ? titleStyle.font_weight : defaults.fontWeight,
    fontFamily: styleString(titleStyle.font_family, String(defaults.fontFamily || SYSTEM_FONT)),
    opacity: titleStyle.opacity != null ? Math.max(0, Math.min(1, styleNumber(titleStyle.opacity, 1))) : defaults.opacity,
    textAlign: align,
    WebkitTextStroke: styleNumber(titleStyle.outline_width, 0) > 0 ? `${styleNumber(titleStyle.outline_width, 0)}px ${styleString(titleStyle.outline_color, "rgba(0,0,0,0.85)")}` : undefined,
    paintOrder: styleNumber(titleStyle.outline_width, 0) > 0 ? "stroke fill" : undefined,
    maxWidth: mode && mode !== "panel" && mode !== "hug" ? mode : defaults.maxWidth,
    width: mode === "panel" ? (defaults.width || "calc(100% - 56px)") : defaults.width,
    boxSizing: "border-box",
  };
  if (mode === "hug") {
    sx.left = defaults.left != null && defaults.width != null
      ? `calc(${String(defaults.left)} + (${String(defaults.width)}) / 2)`
      : "50%";
    sx.right = "auto";
    sx.width = "fit-content";
    sx.maxWidth = "calc(100% - 16px)";
    sx.transform = defaults.transform
      ? `${String(defaults.transform)} translateX(-50%)`
      : "translateX(-50%)";
  }
  if (Number.isFinite(Number(titleStyle.x)) || Number.isFinite(Number(titleStyle.y))) {
    const offset = Array.isArray(titleStyle.offset) ? titleStyle.offset.map(Number) : [0, 0];
    const left = Number.isFinite(Number(titleStyle.x)) ? Number(titleStyle.x) * 100 : 50;
    const top = Number.isFinite(Number(titleStyle.y)) ? Number(titleStyle.y) * 100 : 0;
    sx.left = `calc(${left}% + ${Number(offset[0] || 0)}px)`;
    sx.top = `calc(${top}% + ${Number(offset[1] || 0)}px)`;
    sx.right = "auto";
    sx.bottom = "auto";
    sx.width = mode === "panel" ? sx.width : "fit-content";
    sx.maxWidth = sx.maxWidth || "calc(100% - 16px)";
    sx.transform = annotationAnchorTransform(styleString(titleStyle.anchor, "top-center"));
  }
  return sx;
}


function renderPanelAnnotation(spec: PanelAnnotationSpec, fallback = ""): React.ReactNode {
  if (spec.math) return renderMathExpression(spec.math, "panel-annotation-math");
  return renderRichTitle(spec.spans, spec.text || fallback);
}

function annotationAnchorFractions(anchor: string | undefined): [number, number] {
  const value = anchor || "top-left";
  const x = value.endsWith("right") ? 1 : value.endsWith("center") || value === "center" ? 0.5 : 0;
  const y = value.startsWith("bottom") ? 1 : value.startsWith("center") || value === "center" ? 0.5 : 0;
  return [x, y];
}

function draggableAnnotationSpec(
  spec: PanelAnnotationSpec,
  element: HTMLElement,
  container: HTMLElement,
): PanelAnnotationSpec {
  if (Array.isArray(spec.box) && spec.box.length === 4) return { ...spec };
  if (Number.isFinite(spec.x) && Number.isFinite(spec.y)) return { ...spec };
  const containerRect = container.getBoundingClientRect();
  const elementRect = element.getBoundingClientRect();
  const anchor = spec.anchor || spec.position || "top-left";
  const [anchorFracX, anchorFracY] = annotationAnchorFractions(anchor);
  const panelWidth = Math.max(1, containerRect.width);
  const panelHeight = Math.max(1, containerRect.height);
  return {
    ...spec,
    anchor,
    x: Math.max(0, Math.min(1, (elementRect.left - containerRect.left + elementRect.width * anchorFracX) / panelWidth)),
    y: Math.max(0, Math.min(1, (elementRect.top - containerRect.top + elementRect.height * anchorFracY) / panelHeight)),
  };
}

function updateAnnotationFromDrag(drag: AnnotationDragState, clientX: number, clientY: number): PanelAnnotationSpec {
  const dx = (clientX - drag.startClientX) / Math.max(1, drag.panelWidth);
  const dy = (clientY - drag.startClientY) / Math.max(1, drag.panelHeight);
  const next = { ...drag.original };
  if (Array.isArray(next.box) && next.box.length === 4) {
    const [left, top, boxW, boxH] = next.box;
    next.box = [
      Math.max(0, Math.min(1 - Math.max(0, boxW), left + dx)),
      Math.max(0, Math.min(1 - Math.max(0, boxH), top + dy)),
      boxW,
      boxH,
    ];
    return next;
  }
  next.x = Math.max(0, Math.min(1, styleNumber(next.x, 0) + dx));
  next.y = Math.max(0, Math.min(1, styleNumber(next.y, 0) + dy));
  return next;
}


const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};
// Page galleries are normally used as a quick visual sweep, rather than as a
// slow slide show. Keep 2 fps available for careful inspection, but make the
// first-play experience responsive on cached local panels.
const PAGE_PLAY_FPS_OPTIONS = [1, 2, 4, 8, 12] as const;
const CONTRAST_PRESETS = [
  { value: "manual", label: "Manual", low: 0, high: 100 },
  { value: "0.5-99.5", label: "0.5–99.5", low: 0.5, high: 99.5 },
  { value: "1-99", label: "1–99", low: 1, high: 99 },
  { value: "2-98", label: "2–98", low: 2, high: 98 },
  { value: "3-97", label: "3–97", low: 3, high: 97 },
  { value: "5-95", label: "5–95", low: 5, high: 95 },
  { value: "10-90", label: "10–90", low: 10, high: 90 },
] as const;
const IDENTITY_PALETTE = ["#2e7d32", "#c62828", "#d81b60", "#1565c0", "#f9a825", "#6a1b9a"] as const;

/**
 * The entries of a synced index list that fall in [0, count), as a Set. Hidden
 * panel lists round and hidden page-slot lists truncate, matching how each list
 * is written, so `snap` picks which; out-of-range or non-finite entries are
 * dropped rather than hiding a panel that does not exist.
 */
function validIndexSet(values: number[] | undefined, count: number, snap: (value: number) => number): Set<number> {
  const indices = new Set<number>();
  for (const value of values || []) {
    const index = snap(Number(value));
    if (Number.isFinite(index) && index >= 0 && index < count) indices.add(index);
  }
  return indices;
}

/**
 * A per-panel knob list after an edit of `panel`. Scope "all" writes every entry;
 * panel scope keeps every other panel's value and fills any missing entry with
 * the NEUTRAL value (never the new one), so a stale or short list cannot
 * broadcast one panel's edit to every panel.
 */
function scopedKnobValues<T>(current: T[] | undefined | null, value: T, neutral: T, count: number, panel: number, allPanels: boolean): T[] {
  if (allPanels) return new Array<T>(count).fill(value);
  const values = Array.from({ length: count }, (_, i) => (current && i < current.length) ? current[i] : neutral);
  values[panel] = value;
  return values;
}

function sameNumberArray(a: number[] | undefined, b: number[]): boolean {
  if (!Array.isArray(a) || a.length !== b.length) return false;
  return a.every((value, idx) => value === b[idx]);
}

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 20;
const DPR = window.devicePixelRatio || 1;

type Show2DWritableFile = {
  write: (data: BlobPart) => Promise<void>;
  close: () => Promise<void>;
};

type Show2DFileHandle = {
  createWritable: () => Promise<Show2DWritableFile>;
};

type Show2DSavePickerOptions = {
  suggestedName?: string;
  types?: { description: string; accept: Record<string, string[]> }[];
};

type Show2DWindow = Window & typeof globalThis & {
  showSaveFilePicker?: (options?: Show2DSavePickerOptions) => Promise<Show2DFileHandle>;
};

function makeHtmlExportFilename(title: string, nImages: number, height: number, width: number, mode: string): string {
  const slug = exportTitleSlug(title, "show2d");
  const shape = nImages > 1 ? `${nImages}x${height}x${width}` : `${height}x${width}`;
  const suffix = mode === "quantized" ? "quantized" : mode === "current" ? "current" : "exact";
  return `${slug}_${shape}_${suffix}.html`;
}

function makeSvgExportFilename(title: string, nImages: number, height: number, width: number, scale: number): string {
  const slug = exportTitleSlug(title, "show2d");
  const shape = nImages > 1 ? `${nImages}x${height}x${width}` : `${height}x${width}`;
  return `${slug}_${shape}_svg_${scale}x.svg`;
}

function scaledCanvasPngDataUrl(canvas: HTMLCanvasElement, scale: number, smooth: boolean): string {
  const exportScale = Math.max(1, Math.min(8, Number.isFinite(scale) ? scale : 2));
  const out = document.createElement("canvas");
  out.width = Math.max(1, Math.round(canvas.width * exportScale));
  out.height = Math.max(1, Math.round(canvas.height * exportScale));
  const ctx = out.getContext("2d");
  if (!ctx) return canvas.toDataURL("image/png");
  ctx.imageSmoothingEnabled = smooth;
  ctx.drawImage(canvas, 0, 0, out.width, out.height);
  return out.toDataURL("image/png");
}

/** Map the scale_bar_* traits onto the shared scale bar options. The label is
 *  lifted with a hard 1 px offset copy (not a blurred canvas shadow) so the SVG
 *  export, which has no canvas shadow, matches the on-screen bar. */
function show2dScaleBarOptions(
  position: string,
  requestedPhysical: number | null | undefined,
  requestedLabel: string | null | undefined,
  style: ScaleBarStyle | null | undefined,
  showZoomIndicator: boolean,
): ScaleBarOptions {
  const shadowColor = styleString(style?.shadow_color, "rgba(0,0,0,0.85)");
  return {
    position: position === "bottom-left" ? "bottom-left" : "bottom-right",
    showZoomIndicator,
    physicalLength: requestedPhysical,
    label: requestedLabel,
    offset: scaleBarOffset(style),
    barThickness: Math.max(0.5, styleNumber(style?.bar_height, 5)),
    labelGap: styleNumber(style?.label_gap, 4),
    font: scaleBarCanvasFont(style),
    color: styleString(style?.color, "#fff"),
    textShadow: { kind: "offset", color: shadowColor },
    barShadowColor: style?.shadow_color == null ? null : shadowColor,
    outline: { color: styleString(style?.outline_color, "rgba(0,0,0,0.85)"), width: Math.max(0, styleNumber(style?.outline_width, 0)) },
    // The zoom label sits on the default bar's baseline even when bar_height changes.
    zoomIndicatorGap: 7,
  };
}

function scaleBarOffset(style?: ScaleBarStyle | null): [number, number] {
  const raw = style?.offset;
  if (!Array.isArray(raw) || raw.length < 2) return [0, 0];
  const x = Number(raw[0]);
  const y = Number(raw[1]);
  return [Number.isFinite(x) ? x : 0, Number.isFinite(y) ? y : 0];
}

function scaleBarFontFamily(style?: ScaleBarStyle | null): string {
  return styleString(style?.font_family, SYSTEM_FONT);
}

function scaleBarFontSize(style?: ScaleBarStyle | null): number {
  return Math.max(1, styleNumber(style?.font_size, 16));
}

function scaleBarCanvasFont(style?: ScaleBarStyle | null): string {
  const weight = style?.font_weight;
  const weightText = weight !== undefined && weight !== null && String(weight).trim() ? `${String(weight).trim()} ` : "";
  return `${weightText}${scaleBarFontSize(style)}px ${scaleBarFontFamily(style)}`;
}

function scaleBarSvgFontAttrs(style?: ScaleBarStyle | null): string {
  const weight = style?.font_weight;
  const weightText = weight !== undefined && weight !== null && String(weight).trim()
    ? ` font-weight="${escapeXmlAttr(String(weight).trim())}"`
    : "";
  return `font-family="${escapeXmlAttr(scaleBarFontFamily(style))}" font-size="${scaleBarFontSize(style)}"${weightText}`;
}

const FFT_SNAP_RADIUS = 5;

type ZoomState = { zoom: number; panX: number; panY: number };
type ZoomAnchor = { zoom: number; rowFrac: number; colFrac: number };

/**
 * Zoom and pan link independently: a linked part comes from `linkedSource`, an
 * unlinked part from `ownSource`. Reading a panel's view, applying an edit to the
 * shared view, and keeping a panel's own view under an edit are all this one rule,
 * so the image and FFT galleries cannot drift apart in how linking behaves.
 */
function combineViews(linkedSource: ZoomState, ownSource: ZoomState, linkZoom: boolean, linkPan: boolean): ZoomState {
  return {
    zoom: linkZoom ? linkedSource.zoom : ownSource.zoom,
    panX: linkPan ? linkedSource.panX : ownSource.panX,
    panY: linkPan ? linkedSource.panY : ownSource.panY,
  };
}

// One fetched detail window per panel: raw floats plus the colormapped canvas
// drawn over the binned preview. row0/col0 are FULL-resolution pixel
// coordinates; each tile pixel covers `bin` full-res pixels.
type DetailTile = {
  row0: number;
  col0: number;
  rows: number;
  cols: number;
  bin: number;
  floats: Float32Array;
  canvas: HTMLCanvasElement | null;
};
// Cap on one detail reply (float32 bytes) so a zoom refetch stays sub-second
// on a slow kernel->browser channel. Mirrors Show2D._DETAIL_BUDGET_BYTES.
const DETAIL_BUDGET_BYTES = 8 * 1024 * 1024;

/**
 * Whether a fetched detail tile still covers the requested full-resolution
 * window at the same or a finer bin. A tile that does not would paint stale or
 * blurrier pixels over the preview, so it is pruned and never drawn.
 */
function detailTileCoversWindow(
  tile: DetailTile,
  detailWindow: { fullRow0: number; fullRow1: number; fullCol0: number; fullCol1: number; bin: number } | null,
): boolean {
  if (!detailWindow) return false;
  const eps = 1e-3;
  const tileRow1 = tile.row0 + tile.rows * tile.bin;
  const tileCol1 = tile.col0 + tile.cols * tile.bin;
  return tile.row0 <= detailWindow.fullRow0 + eps
    && tileRow1 >= detailWindow.fullRow1 - eps
    && tile.col0 <= detailWindow.fullCol0 + eps
    && tileCol1 >= detailWindow.fullCol1 - eps
    && tile.bin <= detailWindow.bin;
}

const MAX_PANEL_COLUMNS = 12;

type TouchZoomState = {
  idx: number;
  mode: "pan" | "pinch";
  startX: number;
  startY: number;
  startDistance: number;
  startMidX: number;
  startMidY: number;
  startState: ZoomState;
};

const SINGLE_IMAGE_TARGET = 500;
const GALLERY_IMAGE_TARGET = 300;
const DEFAULT_FFT_ZOOM = 2;
const GALLERY_FFT_OVERVIEW_MAX_DIM = 2048;
// A paged gallery may revisit many more panels than are visible at once. Keep
// its overview FFTs small enough that a complete pass can stay in the bounded
// LRU, rather than evicting the first page while the user is playing through
// later pages. ROI FFTs remain native-resolution below.
const PAGED_GALLERY_FFT_OVERVIEW_MAX_DIM = 1024;
const PROFILE_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];
type ROIItem = { row: number; col: number; shape: string; radius: number; radius_inner: number; width: number; height: number; color: string; line_width: number; highlight: boolean };
const ROI_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];
const RESIZE_HIT_AREA_PX = 10;

function clonePanelOverlays(overlays: PanelOverlaySpec[][] | undefined): PanelOverlaySpec[][] {
  return (overlays || []).map((items) => (items || []).map((item) => ({ ...item })));
}

function clonePanelAnnotations(annotations: PanelAnnotationSpec[][] | undefined): PanelAnnotationSpec[][] {
  return (annotations || []).map((items) => (items || []).map((item) => ({ ...item })));
}

function displayValue(value: number, logScale: boolean): number {
  return logScale ? signedLog1p(value) : value;
}

function displayRange(min: number, max: number, logScale: boolean): { min: number; max: number } {
  return { min: displayValue(min, logScale), max: displayValue(max, logScale) };
}

function mergeDataRanges(ranges: { min: number; max: number }[]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const range of ranges) {
    if (!Number.isFinite(range.min) || !Number.isFinite(range.max)) continue;
    if (range.min < min) min = range.min;
    if (range.max > max) max = range.max;
  }
  if (min === Infinity || max === -Infinity) return { min: 0, max: 1 };
  return { min, max };
}

function mergeHistogramBins(histograms: number[][]): number[] {
  const bins = new Array(256).fill(0);
  for (const histogram of histograms) {
    for (let i = 0; i < Math.min(256, histogram.length); i++) bins[i] += histogram[i];
  }
  const maxCount = Math.max(...bins);
  if (maxCount > 0) for (let i = 0; i < bins.length; i++) bins[i] /= maxCount;
  return bins;
}

function meanDownsample2D(data: Float32Array, width: number, height: number, factor: number): { data: Float32Array; width: number; height: number } {
  if (factor <= 1) return { data, width, height };
  const outW = Math.max(1, Math.ceil(width / factor));
  const outH = Math.max(1, Math.ceil(height / factor));
  const out = new Float32Array(outW * outH);
  for (let outRow = 0; outRow < outH; outRow++) {
    const row0 = outRow * factor;
    const row1 = Math.min(height, row0 + factor);
    for (let outCol = 0; outCol < outW; outCol++) {
      const col0 = outCol * factor;
      const col1 = Math.min(width, col0 + factor);
      let sum = 0;
      let count = 0;
      for (let row = row0; row < row1; row++) {
        const rowOffset = row * width;
        for (let col = col0; col < col1; col++) {
          sum += data[rowOffset + col];
          count++;
        }
      }
      out[outRow * outW + outCol] = count > 0 ? sum / count : 0;
    }
  }
  return { data: out, width: outW, height: outH };
}

/**
 * Top-left CSS position of the lens inset: where the user dragged it, else
 * bottom-left with room for the scale bar strip. Paint and both hit tests use
 * this one rule so a click always lands on the inset that was drawn.
 */
function lensInsetOrigin(anchor: { x: number; y: number } | null, canvasHeight: number, lensSize: number): { x: number; y: number } {
  const margin = 12;
  return {
    x: anchor ? anchor.x : margin,
    y: anchor ? anchor.y : canvasHeight - lensSize - margin - 20,
  };
}

/** Where a CSS point falls on the lens inset: its 8 px resize edge, its interior, or outside it. */
function lensInsetHit(cssX: number, cssY: number, lensX: number, lensY: number, lensSize: number): "edge" | "interior" | null {
  const inside = cssX >= lensX && cssX <= lensX + lensSize && cssY >= lensY && cssY <= lensY + lensSize;
  if (!inside) return null;
  const edgeHit = 8;
  const nearEdge = cssX - lensX < edgeHit || lensX + lensSize - cssX < edgeHit || cssY - lensY < edgeHit || lensY + lensSize - cssY < edgeHit;
  return nearEdge ? "edge" : "interior";
}

/**
 * Copy a row-major (height, width) image into the top-left corner of a zeroed
 * (paddedHeight, paddedWidth) buffer. The display FFT needs power-of-two sides,
 * and zero fill adds no signal, so the spectrum is only sampled more finely.
 */
function zeroPadTopLeft(source: Float32Array, width: number, height: number, paddedWidth: number, paddedHeight: number): Float32Array {
  const padded = new Float32Array(paddedWidth * paddedHeight);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      padded[row * paddedWidth + col] = source[row * width + col];
    }
  }
  return padded;
}

/**
 * The (lo, hi) range one panel's offline uint8 bytes were quantized with: the
 * per-panel lists when present, else the legacy global scalars. Dequantizing
 * each panel with its own range keeps differently scaled panels exact.
 */
function offlineQuantRange(panel: number, mins: number[] | undefined, maxs: number[] | undefined, min: number, max: number): [number, number] {
  return [
    (mins && mins.length > panel) ? mins[panel] : min,
    (maxs && maxs.length > panel) ? maxs[panel] : max,
  ];
}

/** True when every panel's offline range is exactly [0, 255], so the bytes are the raw values and need no dequantization. */
function offlineRangesAreIdentity(count: number, mins: number[] | undefined, maxs: number[] | undefined, min: number, max: number): boolean {
  return Array.from({ length: count }).every((_, panel) => {
    const [lo, hi] = offlineQuantRange(panel, mins, maxs, min, max);
    return lo === 0 && hi === 255;
  });
}

// UI constants shared in style with Show4DSTEM.
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
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
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  textTransform: "none" as const,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};
const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
};
const sliderStyles = {
  small: { py: 0, "& .MuiSlider-thumb": { width: 10, height: 10 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 } },
};
const imagePanelRadius = 0;

function Show2D() {
  const isMobileViewport = useMobileViewport();
  const canvasRepaintSignal = useCanvasRepaintSignal();
  const model = useModel();
  const folderWatchLive = useFolderWatchModelLive(model);
  React.useLayoutEffect(() => applyStandaloneWidgetViewState(model), [model]);
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);

  const staticFallbackRootRef = React.useRef<HTMLDivElement | null>(null);

  // Theme (offline HTML exports force a light/white background)
  const [offlineForTheme] = useModelState<boolean>("_export_light");
  const { themeInfo, colors: baseThemeColors } = useTheme(offlineForTheme);
  const themeColors = {
    ...baseThemeColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
  };
  const mobileControlRowSx = isMobileViewport
    ? ({ columnGap: "8px", rowGap: "4px", px: 0.75, py: 0.25 } as const)
    : ({} as const);
  const controlPairSx = {
    display: "inline-flex",
    alignItems: "center",
    gap: isMobileViewport ? "4px" : `${SPACING.XS}px`,
    flexShrink: 0,
  } as const;
  // Primary compact control labels read at full strength; textMuted is
  // reserved for status text (export/page status), never for a label that
  // names a live control.
  const compactLabelSx = { ...typography.label, fontSize: 10, color: themeColors.text } as const;
  // Bordered wrapper for a scoped sub-group (e.g. the Link toggles) so the
  // governing word stays with its switches when the row wraps on narrow
  // viewports.
  const controlSubGroupSx = {
    display: "inline-flex",
    alignItems: "center",
    flexWrap: "wrap" as const,
    gap: isMobileViewport ? "4px" : `${SPACING.XS}px`,
    border: `1px solid ${themeColors.border}`,
    borderRadius: "3px",
    px: 0.5,
    py: 0.125,
  } as const;

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
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } },
  };
  const themedTopMenuProps = {
    PaperProps: themedMenuProps.PaperProps,
    sx: { zIndex: 9999 },
  };

  const [nImages] = useModelState<number>("n_images");
  const [folderWaiting] = useModelState<boolean>("folder_waiting");
  const [folderStatus] = useModelState<string>("folder_status");
  const [folderWatchState] = useModelState<string>("folder_watch_state");
  const [folderWatchDetail] = useModelState<string>("folder_watch_detail");
  const [nPages] = useModelState<number>("n_pages");
  const [pageIdx, setPageIdx] = useModelState<number>("page_idx");
  const [panelsPerPage] = useModelState<number>("panels_per_page");
  const [pageKind] = useModelState<"comparison" | "items">("page_kind");
  const [pageLabels] = useModelState<string[]>("page_labels");
  const [pageStarred, setPageStarred] = useModelState<number[]>("page_starred");
  const isPaged = (nPages || 1) > 1 && (panelsPerPage || 0) > 0;
  const isItemPaged = isPaged && pageKind === "items";
  const currentPageIdx = Math.max(0, Math.min((nPages || 1) - 1, Math.round(pageIdx || 0)));
  const [pagePlaying, setPagePlaying] = React.useState(false);
  const [pagePlayFps, setPagePlayFps] = React.useState<number>(8);
  const [pageSliderPreviewIdx, setPageSliderPreviewIdxState] = React.useState<number | null>(null);
  const pageSliderPreviewIdxRef = React.useRef<number | null>(null);
  const currentPageIdxRef = React.useRef(0);
  const pageCommitPendingRef = React.useRef<number | null>(null);
  const pageCommitRafRef = React.useRef<number | null>(null);
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
    if (preview !== null && preview === currentPageIdx) setPageSliderPreviewIdx(null);
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
  const activePageStart = isPaged ? pageControlIdx * Math.max(1, panelsPerPage || 1) : 0;
  const activePageEnd = isPaged ? Math.min(nImages, activePageStart + Math.max(1, panelsPerPage || 1)) : nImages;
  const activePageIndices = React.useMemo(
    () => Array.from({ length: Math.max(0, activePageEnd - activePageStart) }, (_, i) => activePageStart + i),
    [activePageStart, activePageEnd]
  );
  const activePanelCount = isPaged ? activePageIndices.length : Math.max(1, nImages || 1);
  // A one-item final page still uses the gallery branch so every canvas,
  // handler, and frame lookup stays keyed to that page's absolute panel index.
  const isGallery = usesGalleryLayout(nImages, isPaged);
  React.useEffect(() => {
    if (!isPaged || (nPages || 1) <= 1) setPagePlaying(false);
  }, [isPaged, nPages]);
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [panelFrameCounts] = useModelState<number[]>("panel_frame_counts");
  const [panelFrameIndices, setPanelFrameIndices] = useModelState<number[]>("panel_frame_indices");
  const [panelPlaybackFpsTrait] = useModelState<number>("panel_playback_fps");
  const panelPlaybackFps = clampPanelPlaybackFps(panelPlaybackFpsTrait);
  const [panelStackOffsets] = useModelState<number[]>("panel_stack_offsets");
  const [panelStackBytes] = useModelState<DataView>("panel_stack_bytes");
  const [panelStackMins] = useModelState<number[]>("_panel_stack_mins");
  const [panelStackMaxs] = useModelState<number[]>("_panel_stack_maxs");
  const [staticFallbackJpeg] = useModelState<string>("_static_fallback_jpeg");
  const [staticFallbackMime] = useModelState<string>("_static_fallback_mime");
  const hasLiveFrameBytes = !!frameBytes && frameBytes.byteLength > 0;
  const frameSourceKey = `single-buffer:${frameBytes?.byteLength ?? 0}`;
  const staticFallbackUrl = staticFallbackJpeg
    ? `data:${staticFallbackMime || "image/jpeg"};base64,${staticFallbackJpeg}`
    : "";
  const hasSavedStaticFallback = staticFallbackUrl.length > 0;
  // Per-panel RGB flags: RGB panels carry display-ready (H, W, 3) float pixels
  // that bypass the colormap LUT + contrast pipeline and paint directly.
  const [isRgbFlags] = useModelState<boolean[]>("is_rgb");
  const isRgbPanel = React.useCallback((i: number) => !!(isRgbFlags && isRgbFlags[i]), [isRgbFlags]);
  const [labels] = useModelState<string[]>("labels");
  const [panelTitleSpans] = useModelState<RichTitleSpan[][]>("panel_title_spans");
  const [starred, setStarred] = useModelState<number[]>("starred");
  const [hiddenPanels, setHiddenPanels] = useModelState<number[]>("hidden_panels");
  const [hiddenPageSlotsTrait, setHiddenPageSlotsTrait] = useModelState<number[] | undefined>("hidden_page_slots");
  const [panelOrder, setPanelOrder] = useModelState<number[]>("panel_order");
  const activeItemPageIndices = React.useMemo(
    () => itemPageIndices(nImages, pageControlIdx, panelsPerPage, panelOrder),
    [nImages, pageControlIdx, panelsPerPage, panelOrder],
  );
  const activePagePanelIndices = isItemPaged ? activeItemPageIndices : activePageIndices;
  const [showPanelTitles] = useModelState<boolean>("show_panel_titles");
  const [panelTitleFontSize] = useModelState<number>("panel_title_font_size");
  const [panelTitleStyle] = useModelState<PanelTitleStyle>("panel_title_style");
  const [interPanelGapPxState] = useModelState<number>("inter_panel_gap_px");
  const [interPanelGapColorState] = useModelState<string>("inter_panel_gap_color");
  const [galleryOuterBorderPxState] = useModelState<number>("gallery_outer_border_px");
  const [galleryOuterBorderColorState] = useModelState<string>("gallery_outer_border_color");
  const [panelInnerBorderPxState] = useModelState<number>("panel_inner_border_px");
  const [panelInnerBorderColorState] = useModelState<string>("panel_inner_border_color");
  const [title] = useModelState<string>("title");
  const [showTitle] = useModelState<boolean>("show_title");
  const [displayBinFactor] = useModelState<number>("_display_bin_factor");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [panelCmaps, setPanelCmaps] = useModelState<string[]>("panel_cmaps");
  const [panelCmapsMemory, setPanelCmapsMemory] = useModelState<string[]>("panel_cmaps_memory");
  const [ncols, setNcols] = useModelState<number>("ncols");
  const panelCmapFor = React.useCallback((idx: number) => {
    const value = panelCmaps && idx >= 0 && idx < panelCmaps.length ? panelCmaps[idx] : "";
    return value || cmap || "inferno";
  }, [panelCmaps, cmap]);
  const colorShared = !(panelCmaps && panelCmaps.length === nImages && nImages > 1);

  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [traitVmin] = useModelState<number | null>("vmin");
  const [traitVmax] = useModelState<number | null>("vmax");
  const [traitVmins] = useModelState<(number | null)[] | null>("vmins");
  const [traitVmaxs] = useModelState<(number | null)[] | null>("vmaxs");
  const [zoomRowTrait, setZoomRowTrait] = useModelState<number | null>("zoom_row");
  const [zoomColTrait, setZoomColTrait] = useModelState<number | null>("zoom_col");
  const [, setViewBoxTrait] = useModelState<number[]>("view_box");
  const [diffMode, setDiffMode] = useModelState<boolean>("diff_mode");
  const [diffReference] = useModelState<number>("diff_reference");

  const [canvasSizeTrait, setCanvasSizeTrait] = useModelState<number>("size");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const imageRenderingStyle = smooth ? "auto" : "pixelated";
  // Display-only filter knobs for sparse maps (EDS, low dose). Python owns
  // the math and repacks frame_bytes on change; raw data is never modified.
  const [displayFilter, setDisplayFilter] = useModelState<string>("denoise");
  const [displaySigma, setDisplaySigma] = useModelState<number>("denoise_sigma");
  const [spatialBin, setSpatialBin] = useModelState<number>("denoise_bin");
  const [displayFilterBanner] = useModelState<string>("denoise_banner");
  // Local slider value during drag; the model (and the Python refilter) only
  // updates on release so scrubbing sigma stays smooth on large galleries.
  const [sigmaDraft, setSigmaDraft] = React.useState<number | null>(null);
  const [sigmaFilterDraft, setSigmaFilterDraft] = React.useState<number | null>(null);
  const sigmaFilterDraftRafRef = React.useRef<number | null>(null);
  const sigmaFilterDraftPendingRef = React.useRef<number | null>(null);
  const setSigmaDraftDuringDrag = React.useCallback((value: number) => {
    setSigmaDraft(value);
    sigmaFilterDraftPendingRef.current = value;
    if (sigmaFilterDraftRafRef.current !== null) return;
    sigmaFilterDraftRafRef.current = window.requestAnimationFrame(() => {
      sigmaFilterDraftRafRef.current = null;
      setSigmaFilterDraft(sigmaFilterDraftPendingRef.current);
    });
  }, []);
  React.useEffect(() => () => {
    if (sigmaFilterDraftRafRef.current !== null) {
      window.cancelAnimationFrame(sigmaFilterDraftRafRef.current);
      sigmaFilterDraftRafRef.current = null;
    }
  }, []);
  // Canonical method for the UI menu.
  const denoiseBaseMode = normalizeFilterMode(displayFilter || "none");
  // Denoise controls row visibility (the editor; secondary to the on/off).
  const [showDenoise, setShowDenoise] = useModelState<boolean>("show_denoise");
  // Master ON/OFF of the denoise EFFECT. Off shows the RAW view (nothing of the
  // denoised view "underneath" leaks through); the per-panel config is PRESERVED
  // (just gated), so toggling back on restores exactly what was there. A clean
  // widget with no config gets a visible gaussian (σ 4) the first time it's
  // enabled so the toggle always does something.
  const [denoiseEnabled, setDenoiseEnabled] = useModelState<boolean>("denoise_enabled");
  const toggleDenoise = () => {
    const next = !denoiseEnabled;
    setDenoiseEnabled(next);
    setShowDenoise(next); // reveal the editor while denoising; hide it when raw
    if (next) {
      const hasConfig = (displayFilters && displayFilters.some((mode) => normalizeFilterMode(mode || "none") !== "none"))
        || (spatialBins && spatialBins.some((bin) => (bin || 1) > 1))
        || denoiseBaseMode !== "none";
      if (!hasConfig) setDisplayFilter("gaussian");
    }
    // Turning OFF preserves the config; the render gate (denoiseEnabled) hides it.
  };
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
  const frequencyFilterScopeAll = frequencyFilterScope !== "panel";
  const [showFrequencyFilter, setShowFrequencyFilter] = useModelState<boolean>("show_frequency_filter");
  const [frequencyDraft, setFrequencyDraft] = React.useState<number | null>(null);
  const [frequencyFilterBackend, setFrequencyFilterBackend] = React.useState("off");
  const [denoiseScope, setDenoiseScope] = useModelState<string>("denoise_scope");
  const denoiseScopeAll = denoiseScope !== "panel";
  // Per-panel resolved knobs (the packing source of truth in Python).
  const [displayFilters, setDisplayFilters] = useModelState<string[]>("denoise_modes");
  const [displaySigmas, setDisplaySigmas] = useModelState<number[]>("denoise_sigmas");
  const [spatialBins, setSpatialBins] = useModelState<number[]>("denoise_bins");
  // Frames ship raw; the display filter in ../display/filter.ts (WGSL, or its
  // CPU port without WebGPU) applies gaussian/bin/anscombe client-side. The
  // master denoise switch gates it: off shows raw frames and keeps the config.
  const browserFilterActive = denoiseEnabled ?? true;

  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelSizes] = useModelState<number[]>("pixel_sizes");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [scaleBarPosition] = useModelState<string>("scale_bar_position");
  const [scaleBarPanels] = useModelState<number[]>("scale_bar_panels");
  const [scaleBarLength] = useModelState<number | null>("scale_bar_length");
  const [scaleBarLabel] = useModelState<string>("scale_bar_label");
  const [scaleBarStyle] = useModelState<ScaleBarStyle>("scale_bar_style");
  const [showZoomIndicator] = useModelState<boolean>("show_zoom_indicator");

  const [showControls] = useModelState<boolean>("show_controls");
  const [controlsCollapsed, setControlsCollapsed] = useModelState<boolean>("controls_collapsed");
  const controlsVisible = showControls && !controlsCollapsed;
  const panelChromeVisible = controlsVisible;
  const showResizeControls = panelChromeVisible;
  const resizeGripSx = React.useMemo(() => ({
    position: "absolute",
    bottom: 0,
    right: 0,
    width: 16,
    height: 16,
    cursor: "nwse-resize",
    opacity: 0.6,
    pointerEvents: "auto",
    background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`,
    touchAction: "none",
    zIndex: 5,
    "&:hover": { opacity: 1 },
  }), [themeColors.accent]);
  const [showStats] = useModelState<boolean>("show_stats");
  const [statsMean] = useModelState<number[]>("stats_mean");
  const [statsMin] = useModelState<number[]>("stats_min");
  const [statsMax] = useModelState<number[]>("stats_max");
  const [statsStd] = useModelState<number[]>("stats_std");
  const [localPanelFrameStats, setLocalPanelFrameStats] = React.useState<Map<number, { mean: number; min: number; max: number; std: number }>>(new Map());

  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const [fftMetricsTrait] = useModelState<boolean>("fft_metrics");
  const fftMetricsEnabled = fftMetricsTrait !== false;

  const [selectedIdx, setSelectedIdx] = useModelState<number>("selected_idx");
  const [markerColors] = useModelState<string[]>("marker_colors");
  const [markerStyle] = useModelState<string>("marker_style");
  const [rowMarkers] = useModelState<MarkerMap>("row_markers");
  const [colMarkers] = useModelState<MarkerMap>("col_markers");
  const [selectedPanels, setSelectedPanels] = useModelState<number[]>("selected_panels");
  const [insetPlots, setInsetPlots] = useModelState<InsetPlotSpec[]>("inset_plots");
  const [showInsetPlots, setShowInsetPlots] = useModelState<boolean>("show_inset_plots");
  const [panelAnnotations, setPanelAnnotations] = useModelState<PanelAnnotationSpec[][]>("panel_annotations");
  const [panelOverlays, setPanelOverlays] = useModelState<PanelOverlaySpec[][]>("panel_overlays");

  const [contrastPreset, setContrastPreset] = useModelState<string>("contrast_preset");
  const [imageFlipsHorizontal, setImageFlipsHorizontal] = useModelState<boolean[]>("image_flips_horizontal");
  const [imageFlipsVertical, setImageFlipsVertical] = useModelState<boolean[]>("image_flips_vertical");
  const [imageRotations, setImageRotations] = useModelState<number[]>("image_rotations");
  const [rotationScope, setRotationScope] = useModelState<string>("rotation_scope");
  const panelMarkerColor = React.useCallback((panel: number) => {
    const value = markerColors?.[panel];
    return value || IDENTITY_PALETTE[panel % IDENTITY_PALETTE.length];
  }, [markerColors]);
  const hasPanelMarkers = React.useMemo(
    () => Array.isArray(markerColors) && markerColors.some(Boolean),
    [markerColors],
  );
  const hasInsetPlots = React.useMemo(
    () => Array.isArray(insetPlots) && insetPlots.some(spec => Array.isArray(spec?.y) && spec.y.length >= 2),
    [insetPlots],
  );
  const insetPlotSpecFor = React.useCallback((panel: number): InsetPlotSpec | undefined => {
    return insetDragDraftRef.current.get(panel) || insetPlots?.[panel];
  }, [insetPlots]);
  const markerAround = (markerStyle || "left") === "around";
  const lastSelectedPanelRef = React.useRef<number | null>(null);
  const normalizeRotation = React.useCallback((value: number) => {
    const turns = Math.round(Number(value) / 90);
    if (Number.isFinite(turns) && Math.abs(Number(value)) > 3) return ((turns % 4) + 4) % 4;
    return ((Math.round(Number(value)) % 4) + 4) % 4;
  }, []);
  const rotationForPanel = React.useCallback((panel: number) => {
    return ((Math.round(imageRotations?.[panel] ?? 0) % 4) + 4) % 4;
  }, [imageRotations]);
  const rotationGlyph = React.useCallback((quarterTurns: number) => {
    const turns = ((Math.round(quarterTurns) % 4) + 4) % 4;
    if (turns === 1) return "↺90°";
    if (turns === 2) return "↻180°";
    if (turns === 3) return "↻90°";
    return "";
  }, []);
  const setRotationForPanel = React.useCallback((panel: number, quarterTurns: number) => {
    const panelCount = Math.max(1, nImages || 1);
    const next = Array.from({ length: panelCount }, (_, idx) => rotationForPanel(idx));
    next[Math.max(0, Math.min(panelCount - 1, panel))] = normalizeRotation(quarterTurns);
    setImageRotations(next);
  }, [nImages, normalizeRotation, rotationForPanel, setImageRotations]);
  const setRotationForScope = React.useCallback((quarterTurns: number) => {
    const panelCount = Math.max(1, nImages || 1);
    const turns = normalizeRotation(quarterTurns);
    if ((rotationScope || "all") === "panel") {
      setRotationForPanel(selectedIdx, turns);
    } else {
      setImageRotations(Array.from({ length: panelCount }, () => turns));
    }
  }, [nImages, normalizeRotation, rotationScope, selectedIdx, setImageRotations, setRotationForPanel]);
  const togglePanelFlip = React.useCallback((panel: number, axis: "h" | "v") => {
    const panelCount = Math.max(1, nImages || 1);
    const source = axis === "h" ? imageFlipsHorizontal : imageFlipsVertical;
    const next = Array.from({ length: panelCount }, (_, idx) => Boolean(source?.[idx]));
    const idx = Math.max(0, Math.min(panelCount - 1, panel));
    next[idx] = !next[idx];
    if (axis === "h") setImageFlipsHorizontal(next);
    else setImageFlipsVertical(next);
  }, [imageFlipsHorizontal, imageFlipsVertical, nImages, setImageFlipsHorizontal, setImageFlipsVertical]);
  const setColorShared = React.useCallback((shared: boolean) => {
    if (shared) {
      if (panelCmaps && panelCmaps.length === nImages) {
        setPanelCmapsMemory(panelCmaps.slice());
      }
      setCmap(panelCmapFor(selectedIdx));
      setPanelCmaps([]);
      return;
    }
    const restored = panelCmapsMemory && panelCmapsMemory.length === nImages
      ? panelCmapsMemory.slice()
      : Array.from({ length: nImages }, (_, i) => panelCmapFor(i));
    setPanelCmaps(restored);
    setPanelCmapsMemory(restored);
  }, [nImages, panelCmapFor, panelCmaps, panelCmapsMemory, selectedIdx, setCmap, setPanelCmaps, setPanelCmapsMemory]);
  const selectedCmap = colorShared ? (cmap || "inferno") : panelCmapFor(selectedIdx);
  const setSelectedCmap = React.useCallback((value: string) => {
    const batchPanels = Array.from(new Set((selectedPanels || [])
      .map((panel) => Math.round(Number(panel)))
      .filter((panel) => Number.isFinite(panel) && panel >= 0 && panel < nImages)));
    if (isGallery && batchPanels.length > 1) {
      const next = panelCmaps && panelCmaps.length === nImages
        ? panelCmaps.slice()
        : Array.from({ length: nImages }, (_, i) => panelCmapFor(i));
      for (const panel of batchPanels) next[panel] = value;
      setPanelCmaps(next);
      setPanelCmapsMemory(next);
      if (!cmap) setCmap(value);
      return;
    }
    if (isGallery && !colorShared) {
      const next = panelCmaps && panelCmaps.length === nImages
        ? panelCmaps.slice()
        : Array.from({ length: nImages }, (_, i) => (i === selectedIdx ? value : cmap));
      next[selectedIdx] = value;
      setPanelCmaps(next);
      setPanelCmapsMemory(next);
      if (!cmap) setCmap(value);
    } else {
      setCmap(value);
      setPanelCmaps([]);
    }
  }, [colorShared, isGallery, panelCmaps, nImages, selectedIdx, selectedPanels, cmap, panelCmapFor, setPanelCmaps, setPanelCmapsMemory, setCmap]);
  // In panel scope the scalar traits are the editor for the selected panel,
  // while the arrays remain the render/source of truth for every panel. Keep
  // the editor pointed at the newly selected panel without continuously
  // mirroring array updates back into an in-progress slider edit.
  const denoiseEditorPanelRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    if (!isGallery || denoiseScopeAll || nImages <= 0) {
      denoiseEditorPanelRef.current = null;
      return;
    }
    const idx = Math.min(Math.max(0, selectedIdx || 0), nImages - 1);
    if (denoiseEditorPanelRef.current === idx) return;
    denoiseEditorPanelRef.current = idx;
    const { mode: nextMode, sigma: nextSigma, bin: nextBin } = resolvePanelDenoiseKnobs(
      idx, displayFilters, displaySigmas, spatialBins,
      { mode: "none", sigma: 4, bin: 1 },
    );
    if (displayFilter !== nextMode) setDisplayFilter(nextMode);
    if (Number(displaySigma ?? 4) !== nextSigma) setDisplaySigma(nextSigma);
    if (Number(spatialBin || 1) !== nextBin) setSpatialBin(nextBin);
    setSigmaDraft(null);
    setSigmaFilterDraft(null);
  }, [isGallery, denoiseScopeAll, nImages, selectedIdx, displayFilters,
      displaySigmas, spatialBins, displayFilter, displaySigma, spatialBin,
      setDisplayFilter, setDisplaySigma, setSpatialBin]);
  const hasLocalPanelStacks = React.useMemo(
    () => Array.from({ length: nImages }, (_, i) => panelFrameCounts?.[i] || 1).some(count => count > 1),
    [nImages, panelFrameCounts]
  );
  const normalizedPanelFrameIndices = React.useMemo(
    () => Array.from({ length: nImages }, (_, panel) => {
      const count = Math.max(1, panelFrameCounts?.[panel] || 1);
      return Math.max(0, Math.min(count - 1, panelFrameIndices?.[panel] || 0));
    }),
    [nImages, panelFrameCounts, panelFrameIndices]
  );
  const [panelFramePreviewIndices, setPanelFramePreviewIndices] = React.useState<number[]>([]);
  const [playingPanelFrames, setPlayingPanelFrames] = React.useState<Set<number>>(new Set());
  const pendingPanelFrameIndicesRef = React.useRef<number[] | null>(null);
  const panelFrameCommitRafRef = React.useRef(0);
  React.useEffect(() => {
    setPanelFramePreviewIndices(normalizedPanelFrameIndices);
  }, [normalizedPanelFrameIndices]);
  React.useEffect(() => {
    setPlayingPanelFrames(previous => {
      const next = new Set(
        Array.from(previous).filter(panel => (panelFrameCounts?.[panel] || 1) > 1)
      );
      return next.size === previous.size ? previous : next;
    });
  }, [panelFrameCounts]);
  const commitPanelFrameIndex = React.useCallback((panel: number, frame: number, immediate = false) => {
    const count = Math.max(1, panelFrameCounts?.[panel] || 1);
    const nextFrame = Math.max(0, Math.min(count - 1, Math.round(frame)));
    const next = [...(pendingPanelFrameIndicesRef.current || normalizedPanelFrameIndices)];
    while (next.length < nImages) next.push(0);
    next[panel] = nextFrame;
    pendingPanelFrameIndicesRef.current = next;
    const flush = () => {
      panelFrameCommitRafRef.current = 0;
      const pending = pendingPanelFrameIndicesRef.current;
      pendingPanelFrameIndicesRef.current = null;
      if (pending) setPanelFrameIndices(pending);
    };
    if (immediate) {
      if (panelFrameCommitRafRef.current) window.cancelAnimationFrame(panelFrameCommitRafRef.current);
      flush();
    } else if (!panelFrameCommitRafRef.current) {
      panelFrameCommitRafRef.current = window.requestAnimationFrame(flush);
    }
  }, [nImages, normalizedPanelFrameIndices, panelFrameCounts, setPanelFrameIndices]);
  const setPanelFrameIndex = React.useCallback((panel: number, frame: number, immediate = false) => {
    const count = Math.max(1, panelFrameCounts?.[panel] || 1);
    const nextFrame = Math.max(0, Math.min(count - 1, Math.round(frame)));
    setPanelFramePreviewIndices(previous => {
      const next = previous.length === nImages ? [...previous] : [...normalizedPanelFrameIndices];
      next[panel] = nextFrame;
      return next;
    });
    commitPanelFrameIndex(panel, nextFrame, immediate);
  }, [commitPanelFrameIndex, nImages, normalizedPanelFrameIndices, panelFrameCounts]);
  const stopPanelPlayback = React.useCallback((panel: number) => {
    setPlayingPanelFrames(previous => {
      if (!previous.has(panel)) return previous;
      const next = new Set(previous);
      next.delete(panel);
      return next;
    });
  }, []);
  const togglePanelPlayback = React.useCallback((panel: number) => {
    setPlayingPanelFrames(previous => {
      const next = new Set(previous);
      if (next.has(panel)) next.delete(panel);
      else next.add(panel);
      return next;
    });
  }, []);
  React.useEffect(() => {
    if (playingPanelFrames.size === 0) return;
    const timeout = window.setTimeout(() => {
      const next = [...normalizedPanelFrameIndices];
      playingPanelFrames.forEach(panel => {
        const count = Math.max(1, panelFrameCounts?.[panel] || 1);
        if (count > 1) next[panel] = (next[panel] + 1) % count;
      });
      setPanelFramePreviewIndices(next);
      setPanelFrameIndices(next);
    }, panelPlaybackIntervalMs(panelPlaybackFps));
    return () => window.clearTimeout(timeout);
  }, [normalizedPanelFrameIndices, panelFrameCounts, panelPlaybackFps, playingPanelFrames, setPanelFrameIndices]);
  React.useEffect(() => () => {
    if (panelFrameCommitRafRef.current) window.cancelAnimationFrame(panelFrameCommitRafRef.current);
  }, []);

  const [roiActive, setRoiActive] = useModelState<boolean>("roi_active");
  const [roiList, setRoiList] = useModelState<ROIItem[]>("roi_list");
  const [roiSelectedIdx, setRoiSelectedIdx] = useModelState<number>("roi_selected_idx");
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false);
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const resizeAspectRef = React.useRef<number | null>(null);
  const [newRoiShape, setNewRoiShape] = React.useState<"circle" | "square" | "rectangle" | "annular">("square");
  const [overlayEditMode, setOverlayEditMode] = React.useState(false);
  const [overlaySelection, setOverlaySelection] = React.useState<OverlaySelection | null>(null);
  const [annotationSelection, setAnnotationSelection] = React.useState<AnnotationSelection | null>(null);
  const [isDraggingOverlay, setIsDraggingOverlay] = React.useState(false);
  const [isDraggingAnnotation, setIsDraggingAnnotation] = React.useState(false);
  const [isHoveringOverlay, setIsHoveringOverlay] = React.useState(false);
  const overlayDragRef = React.useRef<OverlayDragState | null>(null);
  const annotationDragRef = React.useRef<AnnotationDragState | null>(null);
  const overlayBaselineRef = React.useRef<PanelOverlaySpec[][] | null>(null);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);
  const [panelMenuAnchor, setPanelMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [moreMenuAnchor, setMoreMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [showRotationSettings, setShowRotationSettings] = React.useState(false);
  const [reorderMode, setReorderMode] = React.useState(false);
  const [dragOverPanel, setDragOverPanel] = React.useState<number | null>(null);
  const draggedPanelRef = React.useRef<number | null>(null);
  const pointerReorderPanelRef = React.useRef<number | null>(null);
  // Maps-style detail streaming: when the preview is binned
  // (_display_bin_factor > 1), zooming past the preview's resolution requests
  // a full-res crop of the visible window from Python instead of ever
  // shipping the whole image over the wire.
  const [, setDetailRequest] = useModelState<string>("_detail_request");
  const [detailMeta] = useModelState<string>("_detail_meta");
  const [detailBytes] = useModelState<DataView>("_detail_bytes");
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [offline] = useModelState<boolean>("offline");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportPayloadFilename] = useModelState<string>("export_filename");
  const [exportBusy, setExportBusy] = React.useState(false);
  const [localExportStatus, setLocalExportStatus] = React.useState("");
  const svgPreviewUrlRef = React.useRef<string | null>(null);
  const svgPreviewImageRef = React.useRef<HTMLImageElement | null>(null);
  const svgPreviewSnapRef = React.useRef({ x: 0, y: 0 });
  const [svgPreview, setSvgPreview] = React.useState<Show2DSvgPreview | null>(null);
  const [svgPreviewSnap, setSvgPreviewSnap] = React.useState({ x: 0, y: 0 });
  const pendingHtmlExportRef = React.useRef<{
    id: string;
    filename: string;
    mode: string;
    handle: Show2DFileHandle | null;
  } | null>(null);
  const selectedRoi = roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0) ? roiList[roiSelectedIdx] : null;
  const hasPanelOverlays = React.useMemo(() => (panelOverlays || []).some((items) => items && items.length > 0), [panelOverlays]);
  const hasPanelAnnotations = React.useMemo(() => (panelAnnotations || []).some((items) => items && items.length > 0), [panelAnnotations]);
  const hasEditablePanelDecorations = hasPanelOverlays || hasPanelAnnotations;
  const scaleBarPanelSet = React.useMemo(() => new Set((scaleBarPanels || []).map((value) => Number(value)).filter((value) => Number.isFinite(value))), [scaleBarPanels]);
  const panelHasScaleBar = React.useCallback((panel: number) => scaleBarVisible && (scaleBarPanelSet.size === 0 || scaleBarPanelSet.has(panel)), [scaleBarPanelSet, scaleBarVisible]);
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
  React.useEffect(() => {
    if (!annotationSelection) return;
    const exists = Boolean(panelAnnotations?.[annotationSelection.panel]?.[annotationSelection.annotation]);
    if (!exists) setAnnotationSelection(null);
  }, [annotationSelection, panelAnnotations]);

  const updatePanelOverlay = React.useCallback((panel: number, overlay: number, nextSpec: PanelOverlaySpec) => {
    const next = clonePanelOverlays(panelOverlays);
    while (next.length <= panel) next.push([]);
    if (!next[panel] || overlay < 0 || overlay >= next[panel].length) return;
    next[panel][overlay] = nextSpec;
    setPanelOverlays(next);
  }, [panelOverlays, setPanelOverlays]);

  const updatePanelAnnotation = React.useCallback((panel: number, annotation: number, nextSpec: PanelAnnotationSpec) => {
    const next = clonePanelAnnotations(panelAnnotations);
    while (next.length <= panel) next.push([]);
    if (!next[panel] || annotation < 0 || annotation >= next[panel].length) return;
    next[panel][annotation] = nextSpec;
    setPanelAnnotations(next);
  }, [panelAnnotations, setPanelAnnotations]);

  const beginPanelAnnotationDrag = React.useCallback((event: React.MouseEvent<HTMLElement>, panel: number, annotation: number) => {
    if (!overlayEditMode) return;
    const original = panelAnnotations?.[panel]?.[annotation];
    const container = imageContainerRefs.current[panel];
    if (!original || !container) return;
    const draggableOriginal = draggableAnnotationSpec(original, event.currentTarget, container);
    const rect = container.getBoundingClientRect();
    annotationDragRef.current = {
      panel,
      annotation,
      startClientX: event.clientX,
      startClientY: event.clientY,
      panelWidth: Math.max(1, rect.width),
      panelHeight: Math.max(1, rect.height),
      original: draggableOriginal,
    };
    setAnnotationSelection({ panel, annotation });
    setOverlaySelection(null);
    setSelectedIdx(panel);
    setIsDraggingAnnotation(true);
    setIsDraggingPan(false);
    setPanStart(null);
    setPanningIdx(null);
    const handleDocumentMove = (moveEvent: MouseEvent) => {
      const drag = annotationDragRef.current;
      if (!drag) return;
      updatePanelAnnotation(drag.panel, drag.annotation, updateAnnotationFromDrag(drag, moveEvent.clientX, moveEvent.clientY));
      moveEvent.preventDefault();
    };
    const handleDocumentUp = () => {
      document.removeEventListener("mousemove", handleDocumentMove);
      document.removeEventListener("mouseup", handleDocumentUp);
      annotationDragRef.current = null;
      setIsDraggingAnnotation(false);
    };
    document.addEventListener("mousemove", handleDocumentMove);
    document.addEventListener("mouseup", handleDocumentUp);
    event.preventDefault();
    event.stopPropagation();
  }, [overlayEditMode, panelAnnotations, setSelectedIdx, updatePanelAnnotation]);

  const deleteSelectedOverlay = React.useCallback(() => {
    if (!overlaySelection) return;
    const next = clonePanelOverlays(panelOverlays);
    const items = next[overlaySelection.panel];
    if (!items || overlaySelection.overlay < 0 || overlaySelection.overlay >= items.length) return;
    items.splice(overlaySelection.overlay, 1);
    setPanelOverlays(next);
    setOverlaySelection(null);
  }, [overlaySelection, panelOverlays, setPanelOverlays]);

  const deleteSelectedAnnotation = React.useCallback(() => {
    if (!annotationSelection) return;
    const next = clonePanelAnnotations(panelAnnotations);
    const items = next[annotationSelection.panel];
    if (!items || annotationSelection.annotation < 0 || annotationSelection.annotation >= items.length) return;
    items.splice(annotationSelection.annotation, 1);
    setPanelAnnotations(next);
    setAnnotationSelection(null);
  }, [annotationSelection, panelAnnotations, setPanelAnnotations]);

  const resetPanelOverlays = React.useCallback(() => {
    if (!overlayBaselineRef.current) return;
    setPanelOverlays(clonePanelOverlays(overlayBaselineRef.current));
    setOverlaySelection(null);
    overlayDragRef.current = null;
    setIsDraggingOverlay(false);
  }, [setPanelOverlays]);

  const handleOverlayEditMenuToggle = React.useCallback((event?: React.SyntheticEvent) => {
    event?.preventDefault();
    event?.stopPropagation();
    setOverlayEditMode((value) => !value);
    setMoreMenuAnchor(null);
    window.setTimeout(() => setMoreMenuAnchor(null), 0);
  }, []);

  const effectiveShowFft = showFft;
  const galleryColumnOptions = React.useMemo(() => {
    const maxCols = Math.max(1, Math.min(isPaged ? activePanelCount : nImages, MAX_PANEL_COLUMNS));
    return Array.from({ length: maxCols }, (_, i) => i + 1);
  }, [activePanelCount, isPaged, nImages]);
  React.useEffect(() => {
    if (!exportStatus) return;
    const preparing = exportStatus.startsWith("Preparing ") || exportStatus.startsWith("Exporting ");
    if (preparing) {
      setExportBusy(true);
    } else if (!pendingHtmlExportRef.current) {
      setExportBusy(false);
    }
  }, [exportStatus]);
  const htmlPixelCount = Math.max(0, Math.floor(nImages) * Math.floor(height) * Math.floor(width));
  const exactHtmlSize = formatEstimatedHtmlSize(htmlPixelCount * 4);
  const quantizedHtmlSize = formatEstimatedHtmlSize(htmlPixelCount);
  const canDownloadCurrentHtml = !exportEnabled && offlineForTheme;
  const standaloneHtmlMode = offline ? "quantized" : "exact";
  const standaloneHtmlLabel = offline
    ? `HTML quantized uint8 (${quantizedHtmlSize})`
    : `HTML exact float32 (${exactHtmlSize})`;
  const unavailableStandaloneHtmlLabel = offline
    ? "HTML exact float32 (not embedded)"
    : "HTML quantized uint8 (requires backend)";

  const handleStandaloneHtmlDownload = () => {
    setExportAnchor(null);
    const filename = makeHtmlExportFilename(title, nImages, height, width, standaloneHtmlMode);
    try {
      const html = `<!doctype html>\n${standaloneHtmlWithCurrentWidgetState(
        model,
        standaloneWidgetStaticHtmlFromDocument(),
        SHOW2D_STANDALONE_VIEW_STATE_KEYS,
      )}`;
      const blob = new Blob([html], { type: "text/html;charset=utf-8" });
      downloadBlob(blob, filename);
      setLocalExportStatus(`Saved ${filename} (${formatSavedBytes(blob.size)})`);
    } catch (err) {
      setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleHtmlExportSelect = async (mode: string) => {
    setExportAnchor(null);
    if (mode !== "exact" && mode !== "quantized") return;
    const filename = makeHtmlExportFilename(title, nImages, height, width, mode);
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setExportBusy(true);
    setLocalExportStatus("Choose export location...");
    const picker = (window as Show2DWindow).showSaveFilePicker;
    let handle: Show2DFileHandle | null = null;
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
    pendingHtmlExportRef.current = { id, filename, mode, handle };
    setLocalExportStatus(`Preparing ${filename}...`);
    setExportRequest(JSON.stringify({ mode, id, filename, download: true }));
  };

  React.useEffect(() => {
    const pending = pendingHtmlExportRef.current;
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
        pendingHtmlExportRef.current = null;
        setExportBusy(false);
        setLocalExportStatus(`Saved ${filename} (${formatSavedBytes(bytes.byteLength)})`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      } catch (err) {
        if (canceled) return;
        pendingHtmlExportRef.current = null;
        setExportBusy(false);
        setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      }
    };
    void save();
    return () => { canceled = true; };
  }, [exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest]);

  const updateSelectedRoi = (updates: Partial<ROIItem>) => {
    if (roiSelectedIdx < 0 || !roiList) return;
    const newList = [...roiList];
    newList[roiSelectedIdx] = { ...newList[roiSelectedIdx], ...updates };
    setRoiList(newList);
  };

  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const imageContainerRefs = React.useRef<(HTMLDivElement | null)[]>([]);
  const fftContainerRefs = React.useRef<(HTMLDivElement | null)[]>([]);
  const singleFftContainerRef = React.useRef<HTMLDivElement>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const [canvasReady, setCanvasReady] = React.useState(0);  // Trigger re-render when refs attached
  const canRenderLive = hasLiveFrameBytes || canvasReady > 0;
  useHideStaticFallback(
    model,
    staticFallbackRootRef,
    folderWaiting || canRenderLive || hasSavedStaticFallback,
  );

  // Zoom/Pan state - per-image when not linked, shared when linked
  const [initialZoom, setInitialZoom] = useModelState<number>("initial_zoom");
  const [linkPan, setLinkPan] = useModelState<boolean>("link_pan");
  // Note: pan derived from zoom_row/zoom_col is applied via a useEffect AFTER canvasW/canvasH
  // are computed (see "Initial pan from zoom_row/zoom_col" effect below).
  const initialZoomState: ZoomState = React.useMemo(
    () => ({ zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, initialZoom || 1)), panX: 0, panY: 0 }),
    [initialZoom]
  );
  const resetZoomStateRef = React.useRef<ZoomState | null>(null);
  const canvasResizeViewAnchorRef = React.useRef<{
    linked: ZoomAnchor;
    per: Map<number, ZoomAnchor>;
    reset: ZoomAnchor | null;
  } | null>(null);
  const [zoomStates, setZoomStates] = React.useState<Map<number, ZoomState>>(new Map());
  const [linkedZoomState, setLinkedZoomState] = React.useState<ZoomState>(initialZoomState);
  const [linkedZoom, setLinkedZoom] = useModelState<boolean>("link_zoom");
  // Wheel and trackpad events can arrive faster than the display refreshes.
  // Keep an immediate mirror for correct cursor-anchored math, then commit at
  // most once per animation frame. This keeps a large gallery responsive
  // without dropping any accumulated zoom steps.
  const zoomStateMirrorRef = React.useRef<{ linked: ZoomState; per: Map<number, ZoomState> }>({
    linked: initialZoomState,
    per: new Map(),
  });
  const pendingWheelZoomRef = React.useRef<{ idx: number; state: ZoomState } | null>(null);
  const wheelZoomCommitRafRef = React.useRef(0);
  const activeViewInteractionPanelRef = React.useRef<number | null>(null);
  const settleViewPaintTimerRef = React.useRef(0);
  const [settledViewPaintVersion, setSettledViewPaintVersion] = React.useState(0);
  const [isDraggingPan, setIsDraggingPan] = React.useState(false);
  const [panStart, setPanStart] = React.useState<{ x: number, y: number, panX: number, panY: number } | null>(null);

  // Maps-style detail state. One tile per panel covering the last-fetched
  // visible window; the binned preview stays the fallback wherever the tile
  // doesn't cover (pan outside the window shows preview until the refetch).
  const detailTilesRef = React.useRef<Map<number, DetailTile>>(new Map());
  // Last requested window key per panel: suppresses duplicate requests while
  // one is in flight or already satisfied.
  const detailSentKeysRef = React.useRef<Map<number, string>>(new Map());
  // Monotonic request id: replies carrying a superseded id are dropped.
  const detailRequestIdRef = React.useRef(0);
  const detailViewSignatureRef = React.useRef("");
  const [detailVersion, setDetailVersion] = React.useState(0);
  const [detailPaintVersion, setDetailPaintVersion] = React.useState(0);
  const [detailStreamStatus, setDetailStreamStatus] = React.useState<"preview" | "streaming" | "ready">("preview");
  // Per-panel display-space (vmin, vmax) actually used for the preview
  // colormap. Detail tiles must map through the exact same window or the
  // tile would visibly "pop" against the preview around it.
  const panelRangesRef = React.useRef<{ vmin: number; vmax: number }[]>([]);
  const clearDetailTile = React.useCallback((panel: number) => {
    const removed = detailTilesRef.current.delete(panel);
    detailSentKeysRef.current.delete(panel);
    return removed;
  }, []);

  // Helper to get zoom state for an image. zoom and pan link independently:
  //   zoom from linkedZoomState if linkedZoom else per-image
  //   pan  from linkedZoomState if linkPan  else per-image
  const getZoomState = React.useCallback((idx: number): ZoomState => {
    const per = zoomStates.get(idx) || initialZoomState;
    return combineViews(linkedZoomState, per, linkedZoom, linkPan);
  }, [linkedZoom, linkPan, linkedZoomState, zoomStates, initialZoomState]);

  React.useEffect(() => {
    zoomStateMirrorRef.current = { linked: linkedZoomState, per: new Map(zoomStates) };
  }, [linkedZoomState, zoomStates]);

  const getImmediateZoomState = React.useCallback((idx: number): ZoomState => {
    const mirror = zoomStateMirrorRef.current;
    const per = mirror.per.get(idx) || initialZoomState;
    return combineViews(mirror.linked, per, linkedZoom, linkPan);
  }, [initialZoomState, linkedZoom, linkPan]);

  // Helper to set zoom state for an image. zoom and pan honored independently:
  //   zoom: writes to linkedZoomState if linkedZoom, else per-image
  //   pan:  writes to linkedZoomState if linkPan, else per-image
  const setZoomState = React.useCallback((idx: number, state: ZoomState) => {
    const mirror = zoomStateMirrorRef.current;
    if (linkedZoom || linkPan) mirror.linked = combineViews(state, mirror.linked, linkedZoom, linkPan);
    if (!linkedZoom || !linkPan) mirror.per.set(idx, combineViews(mirror.per.get(idx) || initialZoomState, state, linkedZoom, linkPan));
    if (linkedZoom || linkPan) setLinkedZoomState(prev => combineViews(state, prev, linkedZoom, linkPan));
    if (!linkedZoom || !linkPan) {
      setZoomStates(prev => {
        const next = new Map(prev);
        next.set(idx, combineViews(next.get(idx) || initialZoomState, state, linkedZoom, linkPan));
        return next;
      });
    }
  }, [linkedZoom, linkPan, initialZoomState]);

  const scheduleWheelZoomState = React.useCallback((idx: number, state: ZoomState) => {
    // Update the mirror immediately so several wheel events in one frame use
    // the result of the previous event instead of stale React state.
    const mirror = zoomStateMirrorRef.current;
    if (linkedZoom || linkPan) mirror.linked = combineViews(state, mirror.linked, linkedZoom, linkPan);
    if (!linkedZoom || !linkPan) mirror.per.set(idx, combineViews(mirror.per.get(idx) || initialZoomState, state, linkedZoom, linkPan));
    pendingWheelZoomRef.current = { idx, state };
    if (wheelZoomCommitRafRef.current) return;
    wheelZoomCommitRafRef.current = requestAnimationFrame(() => {
      wheelZoomCommitRafRef.current = 0;
      const pending = pendingWheelZoomRef.current;
      pendingWheelZoomRef.current = null;
      if (pending) setZoomState(pending.idx, pending.state);
    });
  }, [initialZoomState, linkPan, linkedZoom, setZoomState]);
  React.useEffect(() => () => {
    cancelAnimationFrame(wheelZoomCommitRafRef.current);
  }, []);

  // During a non-linked gallery zoom, the user's gesture changes one panel.
  // Repainting every nearby panel for each wheel event makes large report
  // galleries feel sticky.  Paint the active panel during the gesture, then
  // repaint the viewport once it settles so any unrelated display update is
  // still reconciled without sacrificing interaction latency.
  const beginViewInteraction = React.useCallback((idx: number) => {
    activeViewInteractionPanelRef.current = idx;
    window.clearTimeout(settleViewPaintTimerRef.current);
    settleViewPaintTimerRef.current = window.setTimeout(() => {
      activeViewInteractionPanelRef.current = null;
      setSettledViewPaintVersion((version) => version + 1);
    }, 140);
  }, []);
  React.useEffect(() => () => window.clearTimeout(settleViewPaintTimerRef.current), []);

  // FFT zoom/pan state (single mode)
  const [fftZoom, setFftZoom] = React.useState(DEFAULT_FFT_ZOOM);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const [isDraggingFftPan, setIsDraggingFftPan] = React.useState(false);
  const [fftPanStart, setFftPanStart] = React.useState<{ x: number, y: number, panX: number, panY: number } | null>(null);

  // Histogram state: per-image contrast ranges (gallery) or one range (single image)
  const [linkedContrast, setLinkedContrast] = useModelState<boolean>("link_contrast");
  const [linkedContrastState, setLinkedContrastState] = React.useState<{ vminPct: number; vmaxPct: number }>({ vminPct: 0, vmaxPct: 100 });
  const [contrastStates, setContrastStates] = React.useState<Map<number, { vminPct: number; vmaxPct: number }>>(new Map());
  // Ref mirror for fast slider path (bypass React effect batching)
  const contrastRef = React.useRef<{ linked: { vminPct: number; vmaxPct: number }; perImage: Map<number, { vminPct: number; vmaxPct: number }> }>({ linked: { vminPct: 0, vmaxPct: 100 }, perImage: new Map() });
  const visibleImageIndicesRef = React.useRef<number[]>([]);
  const sliderFrameQueueRef = React.useRef<ContrastPreviewQueue>({ frame: 0, inFlight: false, pendingPanel: null, epoch: 0 });
  const contrastPreviewGenerationRef = React.useRef(0);
  React.useEffect(() => () => {
    contrastPreviewGenerationRef.current += 1;
    cancelContrastPreview(
      sliderFrameQueueRef.current,
      (frame) => window.cancelAnimationFrame(frame),
    );
  }, []);
  const getContrastState = React.useCallback((idx: number) => {
    if (linkedContrast) return linkedContrastState;
    return contrastStates.get(idx) || { vminPct: 0, vmaxPct: 100 };
  }, [linkedContrast, linkedContrastState, contrastStates]);
  const previousLinkedContrastRef = React.useRef(linkedContrast);
  React.useLayoutEffect(() => {
    if (previousLinkedContrastRef.current === linkedContrast) return;
    const next = transitionContrastLinkState(
      contrastRef.current,
      linkedContrast,
      selectedIdx,
      nImages,
    );
    contrastRef.current = next;
    if (linkedContrast) {
      setLinkedContrastState(next.linked);
    } else {
      setContrastStates(next.perImage);
    }
    previousLinkedContrastRef.current = linkedContrast;
  }, [linkedContrast, nImages, selectedIdx]);
  const setContrastState = React.useCallback((idx: number, state: { vminPct: number; vmaxPct: number }, commit = true) => {
    // Update ref immediately (for fast rAF render)
    if (linkedContrast) {
      contrastRef.current.linked = state;
      if (commit) setLinkedContrastState(state);
    } else {
      contrastRef.current.perImage.set(idx, state);
      if (commit) setContrastStates(prev => new Map(prev).set(idx, state));
    }
    // Fast path: direct GPU render via rAF, bypassing React effect batching
    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current && engine.slotCount >= nImages) {
      enqueueContrastPreview(
        sliderFrameQueueRef.current,
        idx,
        (callback) => window.requestAnimationFrame(callback),
        async (previewIdx) => {
        const previewGeneration = contrastPreviewGenerationRef.current;
        const cachedRanges = dataRangesRef.current;
        if (cachedRanges.length === 0) return;
        // Independent contrast is a panel-local interaction. Repainting the
        // whole gallery here can momentarily apply stale/default ranges to
        // untouched panels before React's settled render restores them. RGB
        // overlays stay on their true-color path in both linked modes.
        const previewIndices = contrastPreviewIndices(
          linkedContrast,
          visibleImageIndicesRef.current,
          previewIdx,
          nImages,
          isRgbFlags,
        );
        const previewLogScale = logScaleRef.current ?? false;
        const hasAbsoluteRange = traitVmin != null && traitVmax != null;
        const baseRanges: { min: number; max: number }[] = [];
        let hasAnyPerImageRange = false;
        for (let i = 0; i < nImages; i++) {
          const panelVmin = traitVmins && traitVmins[i] != null ? traitVmins[i] : null;
          const panelVmax = traitVmaxs && traitVmaxs[i] != null ? traitVmaxs[i] : null;
          if (panelVmin != null && panelVmax != null) {
            hasAnyPerImageRange = true;
            baseRanges.push(displayRange(panelVmin, panelVmax, previewLogScale));
            continue;
          }
          if (hasAbsoluteRange) {
            baseRanges.push(displayRange(traitVmin!, traitVmax!, previewLogScale));
            continue;
          }
          let range = cachedRanges[i];
          if (!range || range.min === range.max) {
            const raw = rawDataRef.current?.[i];
            if (raw) {
              const rawRange = findDataRange(raw);
              range = displayRange(rawRange.min, rawRange.max, previewLogScale);
            }
          }
          baseRanges.push(range || { min: 0, max: 1 });
        }
        const linkedRange = linkedContrast && isGallery && !hasAbsoluteRange && !hasAnyPerImageRange
          ? mergeDataRanges(baseRanges)
          : null;
        const ranges: { vmin: number; vmax: number }[] = [];
        for (let i = 0; i < nImages; i++) {
          const contrast = linkedContrast ? contrastRef.current.linked : (contrastRef.current.perImage.get(i) || { vminPct: 0, vmaxPct: 100 });
          const range = linkedRange || baseRanges[i] || { min: 0, max: 1 };
          if (contrast.vminPct > 0 || contrast.vmaxPct < 100) {
            ranges.push(sliderRange(range.min, range.max, contrast.vminPct, contrast.vmaxPct));
          } else {
            ranges.push({ vmin: range.min, vmax: range.max });
          }
        }
        panelRangesRef.current = ranges;  // keep detail tiles on the live contrast window
        const bitmapRanges = previewIndices.map(i => ranges[i] || { vmin: 0, vmax: 1 });
        // Paint as the settled render does: every panel in its own colormap,
        // and packed uint8 frames through the scaled pass at canvas size.
        const cmapNames = Array.from({ length: nImages }, (_, i) => panelCmapForRef.current(i));
        if (paintSampledPreviewRef.current(previewIndices, bitmapRanges, previewLogScale, cmapNames)) return;
        const bitmaps = uint8FolderPreviewModeRef.current || cmapNames.some(name => name !== cmapNames[0])
          ? await renderScaledPanelBitmaps(engine, previewIndices, bitmapRanges, previewLogScale, cmapNames, mainOffscreensRef.current)
          : await engine.renderSlotsToImageBitmapAsync(previewIndices, bitmapRanges, previewLogScale, cmapNames[0], COLORMAPS[cmapNames[0]] || COLORMAPS.inferno);
        if (previewGeneration !== contrastPreviewGenerationRef.current) {
          bitmaps?.forEach((bitmap) => bitmap?.close());
          return;
        }
        if (bitmaps && bitmaps[0]) {
          try {
            for (let k = 0; k < bitmaps.length; k++) {
              const bitmap = bitmaps[k];
              if (!bitmap) continue;
              const panel = previewIndices[k];
              const offscreen = mainOffscreensRef.current[panel];
              if (offscreen) offscreen.getContext("2d")?.drawImage(bitmap, 0, 0);
            }
          } finally {
            bitmaps.forEach(bitmap => bitmap?.close());
          }
          paintPreviewPanelsRef.current(previewIndices);
        }
        },
      );
    }
  }, [linkedContrast, nImages, isGallery, traitVmin, traitVmax, traitVmins, traitVmaxs, isRgbFlags]);
  const applyContrastPreset = React.useCallback((preset: string) => {
    setContrastPreset(preset);
    if (preset === "manual" || preset === "custom") {
      setAutoContrast(false);
      return;
    }
    const match = preset.match(/^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$/);
    if (!match) return;
    const lo = Math.max(0, Math.min(100, Number(match[1])));
    const hi = Math.max(lo + 0.01, Math.min(100, Number(match[2])));
    setAutoContrast(false);
    if (linkedContrast) {
      setLinkedContrastState({ vminPct: lo, vmaxPct: hi });
      contrastRef.current.linked = { vminPct: lo, vmaxPct: hi };
      return;
    }
    Array.from({ length: nImages }, (_, i) => setContrastState(i, { vminPct: lo, vmaxPct: hi }, true));
  }, [linkedContrast, nImages, setAutoContrast, setContrastPreset, setContrastState]);
  // A histogram drag leaves Auto and the named presets. The preview commits to
  // React state only on the tick that leaves Auto; the preset is written only
  // when it changes, since every model write is a message to the kernel.
  const leaveAutoContrast = () => {
    const leavingAuto = autoContrast;
    if (leavingAuto) setAutoContrast(false);
    if (contrastPreset !== "manual") setContrastPreset("manual");
    return leavingAuto;
  };
  const histogramRangeHandlers = (panel: number) => ({
    onRangeChange: (min: number, max: number) => { leaveAutoContrast(); setContrastState(panel, { vminPct: min, vmaxPct: max }); },
    onRangePreview: (min: number, max: number) => setContrastState(panel, { vminPct: min, vmaxPct: max }, leaveAutoContrast()),
    onRangeCommit: (min: number, max: number) => { leaveAutoContrast(); setContrastState(panel, { vminPct: min, vmaxPct: max }, true); },
  });
  // Convenience accessors for active image
  const activeContrastIdx = nImages > 1 ? selectedIdx : 0;
  const imageVminPct = getContrastState(activeContrastIdx).vminPct;
  const imageVmaxPct = getContrastState(activeContrastIdx).vmaxPct;

  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  const [imageHistogramBins, setImageHistogramBins] = React.useState<number[] | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  // autoContrast cache + version forward-declared here so the histogram thumbs
  // can read the populated cache. Effect that populates lives later in file.
  const autoContrastCacheRef = React.useRef<{ vmin: number; vmax: number }[]>([]);
  const [autoContrastVersion, setAutoContrastVersion] = React.useState(0);
  // Moves when the uploaded frames' data ranges arrive (histogram and colormap wait for them).
  const [rangeVersion, setRangeVersion] = React.useState(0);

  // FFT display state (single mode)
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftScaleMode, setFftScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const [fftAuto, setFftAuto] = React.useState(true);
  // Sharp FFT pixels by default, like the image; Smooth opts into bilinear enlargement.
  const [fftSmooth, setFftSmooth] = React.useState(false);
  const effectiveFftLinkedZoom = linkedZoom;
  const effectiveFftLinkPan = linkPan;
  const effectiveFftLinkedContrast = linkedContrast;
  // Per-image FFT contrast (used when global linked contrast is off)
  const [fftContrastStates, setFftContrastStates] = React.useState<Map<number, { vminPct: number; vmaxPct: number }>>(new Map());
  const fftContrastFor = React.useCallback((idx: number) => {
    if (effectiveFftLinkedContrast) return { vminPct: fftVminPct, vmaxPct: fftVmaxPct };
    return fftContrastStates.get(idx) || { vminPct: 0, vmaxPct: 100 };
  }, [effectiveFftLinkedContrast, fftVminPct, fftVmaxPct, fftContrastStates]);
  const setFftContrastFor = React.useCallback((idx: number, val: { vminPct: number; vmaxPct: number }) => {
    if (effectiveFftLinkedContrast) {
      setFftVminPct(val.vminPct);
      setFftVmaxPct(val.vmaxPct);
    } else {
      setFftContrastStates(prev => new Map(prev).set(idx, val));
    }
  }, [effectiveFftLinkedContrast]);
  const [fftStats, setFftStats] = React.useState<number[] | null>(null);
  const [fftQuality, setFftQuality] = React.useState<FftQualityMetrics | null>(null);
  const [galleryFftQuality, setGalleryFftQuality] = React.useState<Array<FftQualityMetrics | null>>([]);
  const fftQualityKeyRef = React.useRef("");
  const [fftShowColorbar, setFftShowColorbar] = React.useState(false);

  // FFT loading state, shown as a pulsing overlay while FFT computes
  const [fftComputing, setFftComputing] = React.useState(false);
  const [fftProgress, setFftProgress] = React.useState("");

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ idx: number; row: number; col: number; value: number; rgb?: [number, number, number] | null; valueSource?: "preview" | "detail" | "native" } | null>(null);
  const [insetHoverInfo, setInsetHoverInfo] = React.useState<InsetHoverInfo | null>(null);
  const insetHoverKeyRef = React.useRef<string>("");
  const insetDragStateRef = React.useRef<InsetDragState | null>(null);
  const insetDragDraftRef = React.useRef<Map<number, InsetPlotSpec>>(new Map());
  const insetDragRafRef = React.useRef<number | null>(null);
  const [insetDragVersion, setInsetDragVersion] = React.useState(0);
  const scheduleInsetDragPaint = React.useCallback(() => {
    if (insetDragRafRef.current !== null) return;
    insetDragRafRef.current = window.requestAnimationFrame(() => {
      insetDragRafRef.current = null;
      setInsetDragVersion(version => version + 1);
    });
  }, []);
  React.useEffect(() => () => {
    if (insetDragRafRef.current !== null) window.cancelAnimationFrame(insetDragRafRef.current);
  }, []);

  // Colorbar state (single image mode only)
  const [showColorbar, setShowColorbar] = React.useState(false);

  // Inset magnifier state
  const [showLens, setShowLens] = React.useState(false);
  const [lensPos, setLensPos] = React.useState<{ row: number; col: number } | null>(null);
  const [lensMag, setLensMag] = React.useState(4);       // magnification 2×–8×
  const [lensDisplaySize, setLensDisplaySize] = React.useState(128); // CSS px 64–256
  const [lensAnchor, setLensAnchor] = React.useState<{ x: number; y: number } | null>(null); // custom position (CSS px from top-left of canvas)
  const [isDraggingLens, setIsDraggingLens] = React.useState(false);
  const [isResizingLens, setIsResizingLens] = React.useState(false);
  const [isHoveringLensEdge, setIsHoveringLensEdge] = React.useState(false);
  const lensDragStartRef = React.useRef<{ mouseX: number; mouseY: number; anchorX: number; anchorY: number } | null>(null);
  const lensResizeStartRef = React.useRef<{ mouseY: number; startSize: number } | null>(null);
  const lensCanvasRef = React.useRef<HTMLCanvasElement | null>(null);

  // FFT d-spacing measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileLine, setProfilePoints] = useModelState<{ row: number; col: number }[]>("profile_line");
  const [profileDataAll, setProfileDataAll] = React.useState<(Float32Array | null)[]>([]);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; globalMin: number; globalMax: number; totalDist: number; xUnit: string } | null>(null);
  const profilePoints = profileLine || [];

  // Distance measurement state (JS-only, not persisted)
  const [measureActive, setMeasureActive] = React.useState(false);
  const [measurePoints, setMeasurePoints] = React.useState<{row: number; col: number}[]>([]);

  // FFT zoom/pan state (gallery mode: per-image or linked)
  const [galleryFftStates, setGalleryFftStates] = React.useState<Map<number, ZoomState>>(new Map());
  const [linkedFftZoomState, setLinkedFftZoomState] = React.useState<ZoomState>({ zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 });
  const [fftPanningIdx, setFftPanningIdx] = React.useState<number | null>(null);
  const getGalleryFftState = React.useCallback((idx: number) => {
    const per = galleryFftStates.get(idx) || { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 };
    return combineViews(linkedFftZoomState, per, effectiveFftLinkedZoom, effectiveFftLinkPan);
  }, [effectiveFftLinkedZoom, effectiveFftLinkPan, linkedFftZoomState, galleryFftStates]);
  const setGalleryFftState = React.useCallback((idx: number, state: ZoomState) => {
    if (effectiveFftLinkedZoom || effectiveFftLinkPan) {
      setLinkedFftZoomState(prev => combineViews(state, prev, effectiveFftLinkedZoom, effectiveFftLinkPan));
    }
    if (!effectiveFftLinkedZoom || !effectiveFftLinkPan) {
      setGalleryFftStates(prev => {
        const current = prev.get(idx) || { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 };
        const next = new Map(prev);
        next.set(idx, combineViews(current, state, effectiveFftLinkedZoom, effectiveFftLinkPan));
        return next;
      });
    }
  }, [effectiveFftLinkedZoom, effectiveFftLinkPan]);
  const previousEffectiveFftLinkRef = React.useRef({ zoom: effectiveFftLinkedZoom, pan: effectiveFftLinkPan, contrast: effectiveFftLinkedContrast });
  React.useEffect(() => {
    const previous = previousEffectiveFftLinkRef.current;
    const zoomJustLinked = !previous.zoom && effectiveFftLinkedZoom;
    const panJustLinked = !previous.pan && effectiveFftLinkPan;
    const zoomJustUnlinked = previous.zoom && !effectiveFftLinkedZoom;
    const panJustUnlinked = previous.pan && !effectiveFftLinkPan;
    const contrastJustLinked = !previous.contrast && effectiveFftLinkedContrast;
    const contrastJustUnlinked = previous.contrast && !effectiveFftLinkedContrast;
    if (zoomJustLinked || panJustLinked) {
      const current = galleryFftStates.get(selectedIdx) || { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 };
      setLinkedFftZoomState(prev => ({
        zoom: zoomJustLinked ? current.zoom : prev.zoom,
        panX: panJustLinked ? current.panX : prev.panX,
        panY: panJustLinked ? current.panY : prev.panY,
      }));
    }
    if (zoomJustUnlinked || panJustUnlinked) {
      const shared = linkedFftZoomState;
      setGalleryFftStates(prev => {
        const next = new Map(prev);
        for (let idx = 0; idx < nImages; idx++) {
          const current = next.get(idx) || { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 };
          next.set(idx, {
            zoom: zoomJustUnlinked ? shared.zoom : current.zoom,
            panX: panJustUnlinked ? shared.panX : current.panX,
            panY: panJustUnlinked ? shared.panY : current.panY,
          });
        }
        return next;
      });
    }
    if (contrastJustLinked) {
      const current = fftContrastStates.get(selectedIdx) || { vminPct: fftVminPct, vmaxPct: fftVmaxPct };
      setFftVminPct(current.vminPct);
      setFftVmaxPct(current.vmaxPct);
    }
    if (contrastJustUnlinked) {
      const shared = { vminPct: fftVminPct, vmaxPct: fftVmaxPct };
      setFftContrastStates(prev => {
        const next = new Map(prev);
        for (let idx = 0; idx < nImages; idx++) next.set(idx, shared);
        return next;
      });
    }
    previousEffectiveFftLinkRef.current = { zoom: effectiveFftLinkedZoom, pan: effectiveFftLinkPan, contrast: effectiveFftLinkedContrast };
  }, [effectiveFftLinkedZoom, effectiveFftLinkPan, effectiveFftLinkedContrast, galleryFftStates, linkedFftZoomState, fftContrastStates, fftVminPct, fftVmaxPct, nImages, selectedIdx]);

  // Resizable state (gallery starts smaller)
  const [canvasSize, setCanvasSize] = React.useState(nImages > 1 ? GALLERY_IMAGE_TARGET : SINGLE_IMAGE_TARGET);

  // Sync initial sizes from traits
  React.useEffect(() => {
    if (canvasSizeTrait > 0) setCanvasSize(canvasSizeTrait);
  }, [canvasSizeTrait]);

  const canvasResizeCleanupRef = React.useRef<(() => void) | null>(null);

  // Profile height resize
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const [profileResizeStart, setProfileResizeStart] = React.useState<{ y: number; height: number } | null>(null);

  // WebGPU FFT
  const gpuFFTRef = React.useRef<DisplayFFT | null>(null);
  const rawDataRef = React.useRef<Float32Array[] | null>(null);
  const lastAppliedPanelFrameIndicesRef = React.useRef<number[]>([]);
  const filterInputSourceRef = React.useRef<{
    allFloats: Float32Array | null;
    allPanelStackFloats: Float32Array | null;
    frameSourceKey: string;
    width: number;
    height: number;
    nImages: number;
  } | null>(null);
  const appliedPanelViewSignaturesRef = React.useRef<string[]>([]);
  // Interleaved (r, g, b) floats for RGB panels; null for grayscale panels.
  // rawDataRef holds the Rec. 709 luminance of RGB panels so every grayscale
  // consumer (FFT, histogram, profile, diff, stats) works unchanged.
  const rgbDataRef = React.useRef<(Float32Array | null)[]>([]);
  const diffCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const diffFftCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const diffFftMagRef = React.useRef<Float32Array | null>(null);
  const diffFftDimsRef = React.useRef<{ width: number; height: number } | null>(null);
  const [diffFftMagVersion, setDiffFftMagVersion] = React.useState(0);

  // Widget-owned colormap engine (WebGPU, or the Canvas2D fallback). It lives
  // in refs so engine start-up alone re-runs no effect; the reconcile effect
  // keyed on gpuCmapReadyVersion uploads the current frames once it is ready.
  const gpuCmapRef = React.useRef<DisplayColormapEngine | null>(null);
  const gpuCmapReadyRef = React.useRef(false);

  // Cached offscreen canvases for main image rendering (avoids per-zoom/pan recompute)
  const mainOffscreensRef = React.useRef<HTMLCanvasElement[]>([]);
  const mainImgDatasRef = React.useRef<ImageData[]>([]);
  const colorbarVminRef = React.useRef(0);
  const colorbarVmaxRef = React.useRef(1);
  const [offscreenVersion, setOffscreenVersion] = React.useState(0);
  const mainCmapGenerationRef = React.useRef(0);

  // Truthful first-render signal: flipped ONCE after the first colormap pass has
  // actually painted.  Python side observes `_js_rendered` and prints the real
  // end-to-end wall clock.  Two rAFs ensure the browser has composited before we
  // fire, so the printed time reflects "user can see the widget," not "data arrived."
  const [, setJsRendered] = useModelState<boolean>("_js_rendered");
  const firstRenderFiredRef = React.useRef(false);
  React.useEffect(() => {
    if (firstRenderFiredRef.current) return;
    if (offscreenVersion === 0) return;
    firstRenderFiredRef.current = true;
    requestAnimationFrame(() => requestAnimationFrame(() => setJsRendered(true)));
  }, [offscreenVersion, setJsRendered]);

  // Inline FFT refs for gallery mode
  const fftCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const fftOffscreensRef = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const fftMagCacheGalleryRef = React.useRef<(Float32Array | null)[]>([]);
  const galleryFftMagnitudeLruRef = React.useRef<Map<string, GalleryFftCacheEntry>>(new Map());
  const galleryFftActiveKeysRef = React.useRef<(string | null)[]>([]);
  const galleryFftTargetKeysRef = React.useRef<(string | null)[]>([]);
  // Each panel owns its source epoch. A view-only filter edit in panel A must
  // not evict panel B's FFT or make it flash through an unnecessary recompute.
  const galleryFftPanelEpochsRef = React.useRef<number[]>([]);
  const galleryFftComputeSerialRef = React.useRef(0);
  const galleryFftDimsRef = React.useRef<{ w: number; h: number } | null>(null);
  const [galleryFftMagVersion, setGalleryFftMagVersion] = React.useState(0);
  // Page playback must wait for a cached FFT to be painted, not merely for its
  // magnitude to exist. Otherwise a rapid page change can show quality text
  // above an unpainted black FFT canvas.
  const [galleryFftOffscreenVersion, setGalleryFftOffscreenVersion] = React.useState(0);

  React.useEffect(() => {
    if (!pagePlaying || !isPaged || (nPages || 1) <= 1) return;
    // The magnitude cache can already contain this page while its colorized
    // canvas is still being uploaded. Hold here until the pixels exist so
    // scientist-facing page playback never flashes a black FFT.
    const fftPageReady = !effectiveShowFft || activePageIndices.every(
      idx => !!fftOffscreensRef.current[idx],
    );
    if (!fftPageReady) return;
    const timeout = window.setTimeout(() => {
      const next = (currentPageIdx + 1) % Math.max(1, nPages || 1);
      setPageSliderPreviewIdx(next);
      setPageIdx(next);
    }, 1000 / Math.max(1, pagePlayFps));
    return () => window.clearTimeout(timeout);
  }, [
    activePageIndices,
    currentPageIdx,
    effectiveShowFft,
    galleryFftOffscreenVersion,
    isPaged,
    nPages,
    pagePlayFps,
    pagePlaying,
    setPageIdx,
    setPageSliderPreviewIdx,
  ]);

  const galleryFftPipelineRef = React.useRef<({
    displayData: Float32Array;
    displayMin: number;
    displayMax: number;
    sourceKey: string;
    scaleMode: string;
    fftAuto: boolean;
    uploadedKey: string;
    paintedKey?: string;
  } | null)[]>([]);
  const galleryFftColorGenRef = React.useRef(0);
  // Each gallery FFT canvas is redrawn only when its own pixels, view or size change:
  // a paint stamp per offscreen and the key each canvas was last drawn with.
  const galleryFftPaintStampRef = React.useRef({ next: 0, stamps: new WeakMap<HTMLCanvasElement, number>() });
  const galleryFftDrawnKeysRef = React.useRef(new WeakMap<HTMLCanvasElement, string>());

  // Cached FFT magnitude for single image mode (avoids recomputing on zoom/pan)
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);
  // Generation counter for FFT: coalesces rapid ROI drag events to ≤1 FFT/frame
  const fftGenRef = React.useRef(0);

  // Cached FFT offscreen canvas for single mode (avoids reprocessing on zoom/pan)
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  // Caches transformed magnitude + range + stats so contrast slider drag
  // doesn't re-run log/power/findDataRange/autoEnhance on every tick.
  const fftPipelineRef = React.useRef<{
    magnitude: Float32Array;
    displayMin: number;
    displayMax: number;
    magVersion: number;
    scaleMode: string;
    fftAuto: boolean;
  } | null>(null);
  const singleFftColorGenRef = React.useRef(0);
  const [fftOffscreenVersion, setFftOffscreenVersion] = React.useState(0);

  // ROI FFT state: when ROI + FFT are both active, compute FFT of cropped ROI region
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);

  const totalPanelCount = Math.max(1, nImages || 1);
  const [hiddenPageSlots, setHiddenPageSlots] = React.useState<number[]>([]);
  const hiddenPageSlotsInitializedRef = React.useRef(false);
  React.useEffect(() => {
    if (!isPaged || isItemPaged) {
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
  }, [activePageStart, activePanelCount, hiddenPageSlotsTrait, hiddenPanels, isItemPaged, isPaged]);
  const hiddenPanelSet = React.useMemo(() => {
    const out = new Set<number>();
    if (isPaged && !isItemPaged) {
      for (const value of hiddenPageSlots || []) {
        const slot = Math.trunc(Number(value));
        const idx = activePageStart + slot;
        if (Number.isFinite(slot) && slot >= 0 && slot < activePanelCount && idx >= activePageStart && idx < activePageEnd) {
          out.add(idx);
        }
      }
    } else {
      for (const value of hiddenPanels || []) {
        const idx = Math.round(Number(value));
        if (Number.isFinite(idx) && idx >= 0 && idx < totalPanelCount) out.add(idx);
      }
    }
    const currentPanels = isPaged
      ? activePagePanelIndices
      : Array.from({ length: totalPanelCount }, (_, i) => i);
    const activeHiddenCount = currentPanels
      .filter((idx) => out.has(idx)).length;
    if (!isItemPaged && activeHiddenCount >= Math.max(1, activePanelCount)) {
      out.delete(currentPanels[Math.max(0, activePanelCount - 1)]);
    }
    return out;
  }, [activePageEnd, activePagePanelIndices, activePageStart, activePanelCount, hiddenPageSlots, hiddenPanels, totalPanelCount, isItemPaged, isPaged]);
  React.useEffect(() => {
    setPlayingPanelFrames(previous => {
      const next = new Set(Array.from(previous).filter(panel => !hiddenPanelSet.has(panel)));
      return next.size === previous.size ? previous : next;
    });
  }, [hiddenPanelSet]);
  const naturalPanelOrder = React.useMemo(
    () => isPaged ? activePagePanelIndices : Array.from({ length: totalPanelCount }, (_, i) => i),
    [activePagePanelIndices, isPaged, totalPanelCount]
  );
  const orderedImageIndices = React.useMemo(() => {
    if (isPaged) return naturalPanelOrder;
    const values = Array.isArray(panelOrder) ? panelOrder.map(value => Math.round(Number(value))) : [];
    const valid = (
      values.length === totalPanelCount &&
      values.every((value) => Number.isFinite(value) && value >= 0 && value < totalPanelCount) &&
      new Set(values).size === totalPanelCount
    );
    return valid ? values : naturalPanelOrder;
  }, [panelOrder, naturalPanelOrder, totalPanelCount, isPaged]);
  const visibleImageIndices = React.useMemo(
    () => orderedImageIndices.filter(i => !hiddenPanelSet.has(i)),
    [hiddenPanelSet, orderedImageIndices]
  );
  visibleImageIndicesRef.current = visibleImageIndices;
  // A large report gallery can have one hundred panels while a fullscreen
  // viewport only shows a handful.  Keep data and panel state for every
  // panel, but limit hot canvas repaint work (zoom/pan and overlays) to the
  // viewport.  Without this, one wheel notch
  // repaints every offscreen canvas in a full-detector report.
  const [viewportPanelIndices, setViewportPanelIndices] = React.useState<number[]>([]);
  React.useEffect(() => {
    if (!isGallery || visibleImageIndices.length <= 12) {
      setViewportPanelIndices([]);
      return;
    }
    if (typeof IntersectionObserver === "undefined") {
      setViewportPanelIndices(visibleImageIndices);
      return;
    }
    const visible = new Set<number>();
    const nodes = visibleImageIndices
      .map((idx) => ({ idx, node: imageContainerRefs.current[idx] }))
      .filter((item): item is { idx: number; node: HTMLDivElement } => !!item.node);
    if (nodes.length === 0) return;
    const commit = () => {
      const next = visibleImageIndices.filter((idx) => visible.has(idx));
      setViewportPanelIndices((previous) => (
        previous.length === next.length && previous.every((idx, pos) => idx === next[pos])
          ? previous
          : next
      ));
    };
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const item = nodes.find((candidate) => candidate.node === entry.target);
        if (!item) continue;
        if (entry.isIntersecting) visible.add(item.idx);
        else visible.delete(item.idx);
      }
      commit();
    }, { root: null, rootMargin: "0px", threshold: 0.01 });
    nodes.forEach(({ node }) => observer.observe(node));
    return () => observer.disconnect();
  }, [isGallery, visibleImageIndices]);
  const viewportPaintImageIndices = React.useMemo(() => {
    if (!isGallery || visibleImageIndices.length <= 12 || viewportPanelIndices.length === 0) {
      return visibleImageIndices;
    }
    const inViewport = new Set(viewportPanelIndices);
    return visibleImageIndices.filter((idx) => inViewport.has(idx));
  }, [isGallery, visibleImageIndices, viewportPanelIndices]);
  const selectedPanelSet = React.useMemo(() => {
    const out = new Set<number>();
    for (const value of selectedPanels || []) {
      const panel = Math.round(Number(value));
      if (Number.isFinite(panel) && panel >= 0 && panel < totalPanelCount && !hiddenPanelSet.has(panel)) out.add(panel);
    }
    return out;
  }, [hiddenPanelSet, selectedPanels, totalPanelCount]);
  const selectedVisiblePanels = React.useMemo(
    () => visibleImageIndices.filter((panel) => selectedPanelSet.has(panel)),
    [selectedPanelSet, visibleImageIndices],
  );
  const selectedVisibleCount = selectedVisiblePanels.length;
  const visibleDiffPlan = React.useMemo(
    () => resolveVisibleDiffPlan(visibleImageIndices, isRgbFlags, diffReference),
    [diffReference, isRgbFlags, visibleImageIndices],
  );
  const visibleGrayscaleIndices = visibleDiffPlan.visibleGrayscale;
  // When the configured reference is hidden, compare from the first visible
  // grayscale panel. This makes "hide to two, then Diff" deterministic.
  const effectiveDiffReference = visibleDiffPlan.reference;
  const diffOtherIndices = visibleDiffPlan.others;
  const showDiffPanel = diffMode && visibleGrayscaleIndices.length >= 2 && !isPaged;
  const diffPanelCount = showDiffPanel ? diffOtherIndices.length : 0;
  const visibleImageCount = Math.max(1, visibleImageIndices.length);
  const panelMenuTotal = isPaged ? activePanelCount : totalPanelCount;
  const allCurrentPanelsVisible = visibleImageCount === panelMenuTotal;
  const panelLabel = React.useCallback((idx: number) => labels?.[idx] || `Image ${idx + 1}`, [labels]);
  const panelTitleContent = React.useCallback((idx: number) => (
    renderRichTitle(panelTitleSpans?.[idx], panelLabel(idx))
  ), [panelLabel, panelTitleSpans]);
  const panelTitleText = React.useCallback((idx: number) => (
    richTitlePlainText(panelTitleSpans?.[idx], panelLabel(idx))
  ), [panelLabel, panelTitleSpans]);
  const pixelSizeForPanel = React.useCallback((idx: number) => {
    const perPanel = pixelSizes?.[idx];
    return perPanel && perPanel > 0 ? perPanel : pixelSize;
  }, [pixelSize, pixelSizes]);
  const setPanelHidden = React.useCallback((panel: number, hidden: boolean) => {
    if (isPaged && !isItemPaged) {
      if (panel < activePageStart || panel >= activePageEnd) return;
      const slot = panel - activePageStart;
      const next = validIndexSet(hiddenPageSlots, activePanelCount, Math.trunc);
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
    const next = validIndexSet(hiddenPanels, totalPanelCount, Math.round);
    if (hidden) next.add(panel);
    else next.delete(panel);
    if (next.size >= totalPanelCount) return;
    if (
      isItemPaged
      && activePagePanelIndices.every(value => next.has(value))
    ) return;
    setHiddenPanels(Array.from(next).sort((a, b) => a - b));
  }, [activePageEnd, activePagePanelIndices, activePageStart, activePanelCount, hiddenPageSlots, hiddenPanels, totalPanelCount, isItemPaged, isPaged, setHiddenPanels, setHiddenPageSlotsTrait]);
  const setPanelsHidden = React.useCallback((panels: number[], hidden: boolean) => {
    const panelSet = new Set(
      panels
        .map((panel) => Math.round(Number(panel)))
        .filter((panel) => Number.isFinite(panel) && panel >= 0 && panel < totalPanelCount),
    );
    if (panelSet.size === 0) return;
    if (isPaged && !isItemPaged) {
      const next = validIndexSet(hiddenPageSlots, activePanelCount, Math.trunc);
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
    const next = validIndexSet(hiddenPanels, totalPanelCount, Math.round);
    for (const panel of panelSet) {
      if (hidden) next.add(panel);
      else next.delete(panel);
    }
    if (next.size >= totalPanelCount) return;
    if (isItemPaged && activePagePanelIndices.every(value => next.has(value))) return;
    setHiddenPanels(Array.from(next).sort((a, b) => a - b));
  }, [activePageEnd, activePagePanelIndices, activePageStart, activePanelCount, hiddenPageSlots, hiddenPanels, totalPanelCount, isItemPaged, isPaged, setHiddenPanels, setHiddenPageSlotsTrait]);
  const handlePanelSelectionMouseDown = React.useCallback((event: React.MouseEvent, panel: number): boolean => {
    if (!isGallery || reorderMode) return false;
    const orderedVisible = orderedImageIndices.filter((idx) => visibleImageIndices.includes(idx));
    const current = new Set(selectedPanelSet);
    let next: number[];
    if (event.shiftKey) {
      const anchor = lastSelectedPanelRef.current !== null && orderedVisible.includes(lastSelectedPanelRef.current)
        ? lastSelectedPanelRef.current
        : (selectedVisiblePanels[selectedVisiblePanels.length - 1] ?? selectedIdx);
      const anchorPos = orderedVisible.indexOf(anchor);
      const panelPos = orderedVisible.indexOf(panel);
      if (anchorPos >= 0 && panelPos >= 0) {
        const [start, end] = anchorPos < panelPos ? [anchorPos, panelPos] : [panelPos, anchorPos];
        next = orderedVisible.slice(start, end + 1);
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
      if (
        selectedIdx === panel &&
        selectedVisiblePanels.length === 1 &&
        selectedVisiblePanels[0] === panel
      ) {
        lastSelectedPanelRef.current = panel;
        return false;
      }
    }
    lastSelectedPanelRef.current = panel;
    setSelectedIdx(panel);
    setSelectedPanels(next);
    return event.shiftKey || event.metaKey || event.ctrlKey;
  }, [isGallery, orderedImageIndices, reorderMode, selectedIdx, selectedPanelSet, selectedVisiblePanels, setSelectedIdx, setSelectedPanels, visibleImageIndices]);
  const togglePanelStar = React.useCallback((panel: number) => {
    const next = Array.from({ length: totalPanelCount }, (_, idx) => starred?.[idx] ? 1 : 0);
    next[panel] = next[panel] ? 0 : 1;
    setStarred(next);
  }, [starred, totalPanelCount, setStarred]);
  const applyPanelOrder = React.useCallback((order: number[]) => {
    const clean = order.filter((value) => Number.isInteger(value) && value >= 0 && value < totalPanelCount);
    if (clean.length !== totalPanelCount || new Set(clean).size !== totalPanelCount) return;
    const natural = clean.every((value, idx) => value === idx);
    setPanelOrder(natural ? [] : clean);
  }, [setPanelOrder, totalPanelCount]);
  const movePanelBefore = React.useCallback((source: number, target: number) => {
    if (source === target) return;
    const next = [...orderedImageIndices];
    const from = next.indexOf(source);
    if (from < 0) return;
    next.splice(from, 1);
    const to = next.indexOf(target);
    if (to < 0) return;
    next.splice(to, 0, source);
    applyPanelOrder(next);
  }, [applyPanelOrder, orderedImageIndices]);
  const handlePanelDragStart = React.useCallback((event: React.DragEvent, panel: number) => {
    if (!reorderMode) return;
    draggedPanelRef.current = panel;
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", String(panel));
    event.stopPropagation();
  }, [reorderMode]);
  const handlePanelDragOver = React.useCallback((event: React.DragEvent, panel: number) => {
    if (!reorderMode) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    if (dragOverPanel !== panel) setDragOverPanel(panel);
    event.stopPropagation();
  }, [dragOverPanel, reorderMode]);
  const handlePanelDrop = React.useCallback((event: React.DragEvent, panel: number) => {
    if (!reorderMode) return;
    event.preventDefault();
    const raw = event.dataTransfer.getData("text/plain");
    const source = raw.trim() !== "" && Number.isFinite(Number(raw))
      ? Math.round(Number(raw))
      : draggedPanelRef.current;
    if (source !== null && source !== undefined) movePanelBefore(source, panel);
    setDragOverPanel(null);
    draggedPanelRef.current = null;
    event.stopPropagation();
  }, [movePanelBefore, reorderMode]);
  const handlePanelDragEnd = React.useCallback(() => {
    setDragOverPanel(null);
    draggedPanelRef.current = null;
  }, []);
  const handlePanelReorderPointerDown = React.useCallback((event: React.PointerEvent, panel: number) => {
    if (!reorderMode) return;
    pointerReorderPanelRef.current = panel;
    draggedPanelRef.current = panel;
    setDragOverPanel(panel);
    event.preventDefault();
    event.stopPropagation();
  }, [reorderMode]);
  const handlePanelReorderPointerEnter = React.useCallback((event: React.PointerEvent, panel: number) => {
    if (!reorderMode || pointerReorderPanelRef.current === null) return;
    if (dragOverPanel !== panel) setDragOverPanel(panel);
    event.stopPropagation();
  }, [dragOverPanel, reorderMode]);
  const handlePanelReorderPointerUp = React.useCallback((event: React.PointerEvent, panel: number) => {
    if (!reorderMode) return;
    const source = pointerReorderPanelRef.current;
    if (source !== null) movePanelBefore(source, panel);
    pointerReorderPanelRef.current = null;
    draggedPanelRef.current = null;
    setDragOverPanel(null);
    event.preventDefault();
    event.stopPropagation();
  }, [movePanelBefore, reorderMode]);
  const resetPanelOrder = React.useCallback(() => {
    setPanelOrder([]);
    setDragOverPanel(null);
    draggedPanelRef.current = null;
    pointerReorderPanelRef.current = null;
  }, [setPanelOrder]);
  React.useEffect(() => {
    if (!isGallery && reorderMode) setReorderMode(false);
  }, [isGallery, reorderMode]);
  React.useEffect(() => {
    if (reorderMode) return;
    pointerReorderPanelRef.current = null;
    draggedPanelRef.current = null;
    setDragOverPanel(null);
  }, [reorderMode]);
  React.useEffect(() => {
    if (!isGallery) return;
    if (!hiddenPanelSet.has(selectedIdx)) return;
    setSelectedIdx(visibleImageIndices[0] ?? 0);
  }, [hiddenPanelSet, isGallery, selectedIdx, setSelectedIdx, visibleImageIndices]);
  React.useEffect(() => {
    if (!isPaged) return;
    if (visibleImageIndices.includes(selectedIdx)) return;
    setSelectedIdx(visibleImageIndices[0] ?? activePageStart);
  }, [activePageStart, isPaged, selectedIdx, setSelectedIdx, visibleImageIndices]);
  React.useEffect(() => {
    if (!isGallery) {
      if ((selectedPanels || []).length > 0) setSelectedPanels([]);
      return;
    }
    const clean = Array.from(new Set((selectedPanels || [])
      .map((value) => Math.round(Number(value)))
      .filter((panel) => Number.isFinite(panel) && visibleImageIndices.includes(panel))));
    if (clean.length === 0 && visibleImageIndices.includes(selectedIdx)) clean.push(selectedIdx);
    if (!sameNumberArray(selectedPanels, clean)) setSelectedPanels(clean);
  }, [isGallery, selectedIdx, selectedPanels, setSelectedPanels, visibleImageIndices]);
  const clampedNcols = Math.max(1, Math.min(ncols || 1, visibleImageCount, MAX_PANEL_COLUMNS));
  const effectiveNcols = clampedNcols + diffPanelCount;
  const displayScale = canvasSize / Math.max(width, height);
  const canvasW = Math.round(width * displayScale);
  const canvasH = Math.round(height * displayScale);
  const galleryGapPx = Math.max(0, Number.isFinite(interPanelGapPxState) ? interPanelGapPxState : 0);
  const galleryGapColorResolved = String(interPanelGapColorState || "");
  const galleryOuterBorderPx = isGallery ? Math.max(0, Number.isFinite(galleryOuterBorderPxState) ? galleryOuterBorderPxState : 0) : 0;
  const galleryOuterBorderColor = String(galleryOuterBorderColorState || galleryGapColorResolved || "");
  const panelInnerBorderPx = Math.max(0, Number.isFinite(panelInnerBorderPxState) ? panelInnerBorderPxState : 1);
  const panelInnerBorderColor = String(panelInnerBorderColorState || themeColors.border);
  const histogramWidthPx = 110;
  const histogramGapPx = 15;
  const galleryGridWidth = isGallery ? effectiveNcols * canvasW + (effectiveNcols - 1) * galleryGapPx + 2 * galleryOuterBorderPx : canvasW;
  // Wrap against the actual notebook/container width. maxWidth below still
  // enforces the requested column count, while auto-fit avoids viewport-only
  // breakpoints that overflow narrow sidebars in a wide browser window.
  const galleryGridColumns = `repeat(auto-fit, minmax(min(100%, ${canvasW}px), ${canvasW}px))`;
  const histogramGridMaxWidth = effectiveNcols * histogramWidthPx + (effectiveNcols - 1) * histogramGapPx;
  const histogramGridColumns = `repeat(auto-fit, minmax(min(100%, ${histogramWidthPx}px), ${histogramWidthPx}px))`;
  const zoomStateToAnchor = React.useCallback((state: ZoomState): ZoomAnchor => ({
    zoom: state.zoom,
    rowFrac: canvasH > 0 && state.zoom > 0 ? 0.5 - state.panY / (state.zoom * canvasH) : 0.5,
    colFrac: canvasW > 0 && state.zoom > 0 ? 0.5 - state.panX / (state.zoom * canvasW) : 0.5,
  }), [canvasW, canvasH]);
  const zoomAnchorToState = React.useCallback((anchor: ZoomAnchor, nextCanvasW: number, nextCanvasH: number): ZoomState => ({
    zoom: anchor.zoom,
    panX: anchor.zoom * nextCanvasW * (0.5 - anchor.colFrac),
    panY: anchor.zoom * nextCanvasH * (0.5 - anchor.rowFrac),
  }), []);
  const applyResizeViewAnchor = React.useCallback((nextSize: number) => {
    const anchor = canvasResizeViewAnchorRef.current;
    if (!anchor || width <= 0 || height <= 0) return;
    const nextScale = nextSize / Math.max(width, height);
    const nextCanvasW = Math.round(width * nextScale);
    const nextCanvasH = Math.round(height * nextScale);
    if (nextCanvasW <= 0 || nextCanvasH <= 0) return;
    setLinkedZoomState(zoomAnchorToState(anchor.linked, nextCanvasW, nextCanvasH));
    setZoomStates(prevStates => {
      if (anchor.per.size === 0 && prevStates.size === 0) return prevStates;
      const next = new Map<number, ZoomState>();
      const keys = new Set<number>([...Array.from(prevStates.keys()), ...Array.from(anchor.per.keys())]);
      keys.forEach(idx => {
        const panelAnchor = anchor.per.get(idx) || zoomStateToAnchor(prevStates.get(idx) || initialZoomState);
        next.set(idx, zoomAnchorToState(panelAnchor, nextCanvasW, nextCanvasH));
      });
      return next;
    });
    if (anchor.reset) resetZoomStateRef.current = zoomAnchorToState(anchor.reset, nextCanvasW, nextCanvasH);
  }, [width, height, zoomAnchorToState, zoomStateToAnchor, initialZoomState]);
  const responsivePanelSx = {
    position: "relative",
    bgcolor: "#000",
    borderRadius: imagePanelRadius,
    boxSizing: "border-box",
    overflow: "hidden",
    width: "100%",
    maxWidth: "100%",
    aspectRatio: `${Math.max(canvasW, 1)} / ${Math.max(canvasH, 1)}`,
    touchAction: "none",
  };
  const responsiveCanvasStyle: React.CSSProperties = {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    imageRendering: imageRenderingStyle,
    display: "block",
  };
  const responsiveOverlayStyle: React.CSSProperties = {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    pointerEvents: "none",
  };
  const responsivePanelWidthSx = {
    width: "100%",
    maxWidth: canvasW,
    boxSizing: "border-box",
  };
  const viewBoxTimerRef = React.useRef(0);
  const persistZoomState = React.useCallback((state: ZoomState) => {
    if (canvasW <= 0 || canvasH <= 0 || width <= 0 || height <= 0 || state.zoom <= 0) return;
    const row = height * (0.5 - state.panY / (state.zoom * canvasH));
    const col = width * (0.5 - state.panX / (state.zoom * canvasW));
    const nextZoom = state.zoom;
    const nextRow = Math.max(0, Math.min(height - 1, row));
    const nextCol = Math.max(0, Math.min(width - 1, col));
    // Persisting traits is for notebook/Python state and current_view capture.
    // The visible canvas has already updated through local zoom state, so keep
    // trait writes out of the high-frequency wheel/drag path.
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const row0 = Math.max(0, ((0 - cy - state.panY) / state.zoom + cy) / displayScale);
    const row1 = Math.min(height, ((canvasH - cy - state.panY) / state.zoom + cy) / displayScale);
    const col0 = Math.max(0, ((0 - cx - state.panX) / state.zoom + cx) / displayScale);
    const col1 = Math.min(width, ((canvasW - cx - state.panX) / state.zoom + cx) / displayScale);
    window.clearTimeout(viewBoxTimerRef.current);
    viewBoxTimerRef.current = window.setTimeout(() => {
      setInitialZoom(nextZoom);
      setZoomRowTrait(nextRow);
      setZoomColTrait(nextCol);
      setViewBoxTrait([row0, row1, col0, col1]);
    }, 120);
  }, [canvasW, canvasH, width, height, displayScale, setInitialZoom, setZoomRowTrait, setZoomColTrait, setViewBoxTrait]);

  // Initial pan from zoom_row/zoom_col, applied once the canvas has valid dims.
  // panX/panY computed so target image (zoomRow, zoomCol) lands at canvas center after transform:
  //   ctx.translate(cx+panX, cy+panY) ⋅ scale(zoom) ⋅ translate(-cx,-cy)
  //   target screen = cx + panX + zoom * (target_canvas - cx) = cx
  //   ⟹ panX = zoom * (cx - target_canvas) = zoom * canvasW * (0.5 - col/width)
  const initialPanAppliedRef = React.useRef(false);
  React.useEffect(() => {
    if (initialPanAppliedRef.current) return;
    if (zoomRowTrait == null && zoomColTrait == null) return;
    if (canvasW <= 0 || canvasH <= 0 || width <= 0 || height <= 0) return;
    const startZoom = initialZoomState.zoom;
    const panX = zoomColTrait != null ? startZoom * canvasW * (0.5 - zoomColTrait / width) : 0;
    const panY = zoomRowTrait != null ? startZoom * canvasH * (0.5 - zoomRowTrait / height) : 0;
    setLinkedZoomState({ zoom: startZoom, panX, panY });
    setZoomStates(prev => {
      const next = new Map(prev);
      for (let i = 0; i < nImages; i++) next.set(i, { zoom: startZoom, panX, panY });
      return next;
    });
    if (resetZoomStateRef.current === null) resetZoomStateRef.current = { zoom: startZoom, panX, panY };
    initialPanAppliedRef.current = true;
  }, [zoomRowTrait, zoomColTrait, canvasW, canvasH, width, height, nImages, initialZoomState.zoom]);
  React.useEffect(() => {
    if (resetZoomStateRef.current !== null) return;
    if (zoomRowTrait != null || zoomColTrait != null) return;
    if (canvasW <= 0 || canvasH <= 0 || width <= 0 || height <= 0) return;
    resetZoomStateRef.current = initialZoomState;
  }, [zoomRowTrait, zoomColTrait, canvasW, canvasH, width, height, initialZoomState]);
  const previousCanvasDimsRef = React.useRef<{ w: number; h: number } | null>(null);
  React.useEffect(() => {
    if (canvasW <= 0 || canvasH <= 0) return;
    const prev = previousCanvasDimsRef.current;
    previousCanvasDimsRef.current = { w: canvasW, h: canvasH };
    if (canvasResizeViewAnchorRef.current) return;
    if (!prev || prev.w <= 0 || prev.h <= 0 || (prev.w === canvasW && prev.h === canvasH)) return;
    const scaleX = canvasW / prev.w;
    const scaleY = canvasH / prev.h;
    const scaleState = (state: ZoomState): ZoomState => ({
      ...state,
      panX: state.panX * scaleX,
      panY: state.panY * scaleY,
    });
    setLinkedZoomState(prevState => scaleState(prevState));
    setZoomStates(prevStates => {
      if (prevStates.size === 0) return prevStates;
      const next = new Map<number, ZoomState>();
      prevStates.forEach((state, idx) => next.set(idx, scaleState(state)));
      return next;
    });
    if (resetZoomStateRef.current) resetZoomStateRef.current = scaleState(resetZoomStateRef.current);
  }, [canvasW, canvasH]);
  // Per-panel float offsets into frame_bytes: grayscale panels are W*H floats,
  // RGB panels are 3*W*H interleaved floats. Last entry = total float count.
  const panelFloatOffsets = React.useMemo(() => {
    const offsets: number[] = [];
    let acc = 0;
    for (let i = 0; i < nImages; i++) {
      offsets.push(acc);
      acc += (isRgbFlags && isRgbFlags[i] ? 3 : 1) * width * height;
    }
    offsets.push(acc);
    return offsets;
  }, [nImages, width, height, isRgbFlags]);
  const profileCanvasWidth = galleryGridWidth;
  const groupMarkerOverlays = React.useMemo(() => {
    if (!isGallery || visibleImageIndices.length === 0 || canvasW <= 0 || canvasH <= 0) return [];
    const cols = Math.max(1, effectiveNcols);
    const gap = Math.max(0, galleryGapPx);
    const build = (markers: MarkerMap | undefined, axis: "row" | "col") => Object.entries(markers || {})
      .map(([rawKey, color]) => {
        const target = Number(rawKey);
        if (!Number.isFinite(target) || target < 0 || !color) return null;
        const slots = visibleImageIndices
          .map((_, slot) => slot)
          .filter((slot) => (axis === "row" ? Math.floor(slot / cols) : slot % cols) === target);
        if (slots.length === 0) return null;
        const rowVals = slots.map((slot) => Math.floor(slot / cols));
        const colVals = slots.map((slot) => slot % cols);
        const row0 = Math.min(...rowVals);
        const row1 = Math.max(...rowVals);
        const col0 = Math.min(...colVals);
        const col1 = Math.max(...colVals);
        return {
          key: `${axis}-${rawKey}`,
          axis,
          color: String(color),
          left: galleryOuterBorderPx + col0 * (canvasW + gap),
          top: galleryOuterBorderPx + row0 * (canvasH + gap),
          width: (col1 - col0 + 1) * canvasW + Math.max(0, col1 - col0) * gap,
          height: (row1 - row0 + 1) * canvasH + Math.max(0, row1 - row0) * gap,
        };
      })
      .filter(Boolean) as Array<{ key: string; axis: "row" | "col"; color: string; left: number; top: number; width: number; height: number }>;
    return [...build(rowMarkers, "row"), ...build(colMarkers, "col")];
  }, [canvasH, canvasW, colMarkers, effectiveNcols, galleryOuterBorderPx, galleryGapPx, isGallery, rowMarkers, visibleImageIndices]);

  // ROI FFT active: both ROI and FFT on, with a selected ROI
  const roiFftActive = effectiveShowFft && roiActive && roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0);

  // Stable key for ROI geometry: only changes when the selected ROI's geometry changes,
  // not when other ROIs move or roiList gets a new reference from unrelated edits.
  // Shared by both ROI FFT and preview panel to avoid redundant recomputes.
  const selectedRoiKey = React.useMemo(() => {
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return "";
    const roi = roiList[roiSelectedIdx];
    return `${roi.row},${roi.col},${roi.radius},${roi.radius_inner},${roi.width},${roi.height},${roi.shape}`;
  }, [roiList, roiSelectedIdx]);
  const roiFftKey = roiFftActive ? selectedRoiKey : "";

  // Extract raw float32 bytes and parse into Float32Arrays
  const [offlineMin] = useModelState<number>("_offline_min");
  const [offlineMax] = useModelState<number>("_offline_max");
  const [offlineMins] = useModelState<number[]>("_offline_mins");
  const [offlineMaxs] = useModelState<number[]>("_offline_maxs");
  const expectedFrameValueCount = panelFloatOffsets[panelFloatOffsets.length - 1];
  const uint8FolderIdentityEncoding = React.useMemo(() => {
    if (!offline) return false;
    if ((isRgbFlags || []).some(Boolean)) return false;
    if (hasLocalPanelStacks) return false;
    return offlineRangesAreIdentity(Math.max(1, nImages || 1), offlineMins, offlineMaxs, offlineMin, offlineMax);
  }, [hasLocalPanelStacks, isRgbFlags, nImages, offline, offlineMin, offlineMax, offlineMins, offlineMaxs]);
  const uint8FolderFrameBytes = React.useMemo(() => {
    if (!uint8FolderIdentityEncoding || !frameBytes || frameBytes.byteLength === 0) return null;
    if (frameBytes.byteLength !== expectedFrameValueCount) return null;
    return new Uint8Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength);
  }, [frameBytes, expectedFrameValueCount, uint8FolderIdentityEncoding]);
  const uint8FolderPreviewMode = !!uint8FolderFrameBytes;
  const decodedFramesRef = React.useRef<Float32Array | null>(null);
  const allFloats = React.useMemo(() => {
    const expectedLength = expectedFrameValueCount;
    if (uint8FolderFrameBytes) {
      decodedFramesRef.current = null;
      return uint8FolderFrameBytes as unknown as Float32Array;
    }
    if (!frameBytes || frameBytes.byteLength === 0) {
      const cached = decodedFramesRef.current;
      return cached !== null && cached.length === expectedLength ? cached : new Float32Array(expectedLength);
    }
    let decoded: Float32Array;
    if (offline) {
      // Offline mode: bytes are uint8-quantized PER IMAGE. Dequantize each panel
      // with its own (lo, hi) so a gallery of differently-scaled panels stays
      // exact - a single global scale combs the narrow panels' histograms.
      const quantized = new Uint8Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength);
      const perImage = width * height;
      const rangeCount = (offlineMins && offlineMins.length > 0) ? offlineMins.length : 1;
      if (offlineRangesAreIdentity(rangeCount, offlineMins, offlineMaxs, offlineMin, offlineMax)) {
        decoded = new Float32Array(quantized);
        decodedFramesRef.current = decoded;
        return decoded;
      }
      const dequantized = new Float32Array(quantized.length);
      for (let panel = 0; panel < rangeCount; panel++) {
        const [lo, hi] = offlineQuantRange(panel, offlineMins, offlineMaxs, offlineMin, offlineMax);
        const base = panel * perImage;
        const end = Math.min(quantized.length, base + perImage);
        if (end > base) dequantizeUint8(quantized.subarray(base, end), lo, hi, dequantized.subarray(base, end));
      }
      decoded = dequantized;
    } else {
      decoded = extractFloat32(frameBytes, expectedLength) ?? new Float32Array(expectedLength);
    }
    decodedFramesRef.current = decoded;
    return decoded;
  }, [frameBytes, expectedFrameValueCount, offline, offlineMin, offlineMax, offlineMins, offlineMaxs, nImages, width, height, panelFloatOffsets, uint8FolderFrameBytes]);

  const panelStackFloatCount = React.useMemo(() => {
    const perImage = width * height;
    return Array.from({ length: nImages }, (_, panel) => {
      const count = panelFrameCounts?.[panel] || 1;
      return count > 1 ? count * perImage : 0;
    }).reduce((sum, value) => sum + value, 0);
  }, [height, nImages, panelFrameCounts, width]);
  const decodedPanelStacksRef = React.useRef<Float32Array | null>(null);
  const allPanelStackFloats = React.useMemo(() => {
    if (panelStackFloatCount <= 0) return new Float32Array(0);
    if (!panelStackBytes || panelStackBytes.byteLength === 0) {
      const cached = decodedPanelStacksRef.current;
      return cached !== null && cached.length === panelStackFloatCount
        ? cached
        : new Float32Array(panelStackFloatCount);
    }
    let decoded: Float32Array;
    if (offline) {
      const quantized = new Uint8Array(
        panelStackBytes.buffer,
        panelStackBytes.byteOffset,
        panelStackBytes.byteLength
      );
      const dequantized = new Float32Array(panelStackFloatCount);
      const perImage = width * height;
      for (let panel = 0; panel < nImages; panel++) {
        const count = panelFrameCounts?.[panel] || 1;
        const offset = panelStackOffsets?.[panel] ?? -1;
        if (count <= 1 || offset < 0) continue;
        const lo = panelStackMins?.[panel] ?? 0;
        const hi = panelStackMaxs?.[panel] ?? 1;
        const length = count * perImage;
        const end = Math.min(quantized.length, offset + length);
        if (end > offset) dequantizeUint8(quantized.subarray(offset, end), lo, hi, dequantized.subarray(offset, end));
      }
      decoded = dequantized;
    } else {
      decoded = extractFloat32(panelStackBytes, panelStackFloatCount)
        ?? new Float32Array(panelStackFloatCount);
    }
    decodedPanelStacksRef.current = decoded;
    return decoded;
  }, [
    panelStackBytes,
    panelStackFloatCount,
    offline,
    width,
    height,
    nImages,
    panelFrameCounts,
    panelStackOffsets,
    panelStackMins,
    panelStackMaxs,
  ]);

  const [dataVersion, setDataVersion] = React.useState(0);
  React.useEffect(() => {
    // A completed bitmap belongs to the data and display context that started
    // it. Link, color, scale, or range changes must not let an older GPU paint
    // flash over the newly selected context.
    contrastPreviewGenerationRef.current += 1;
    cancelContrastPreview(
      sliderFrameQueueRef.current,
      (frame) => window.cancelAnimationFrame(frame),
    );
  }, [cmap, dataVersion, linkedContrast, logScale, nImages, traitVmin, traitVmax, traitVmins, traitVmaxs]);
  const [gpuCmapVersion, setGpuCmapVersion] = React.useState(0);
  const [gpuCmapReadyVersion, setGpuCmapReadyVersion] = React.useState(0);

  // Initialize WebGPU FFT + a widget-owned colormap engine on mount. Colormap
  // slots are numbered from zero, so sharing the singleton across two Show2D
  // instances lets one widget overwrite the other's scientific pixels.
  // The FFT engine sits in a ref and is picked up on the next natural re-run;
  // the colormap engine bumps gpuCmapReadyVersion so the reconcile effect
  // uploads the current frames.
  React.useEffect(() => {
    let disposed = false;
    getDisplayFFT().then(fft => {
      if (disposed) return;
      if (fft) {
        gpuFFTRef.current = fft;
      }
    }).catch(error => console.error("[Show2D] display FFT failed to start", error));
    createColormapEngine().then(engine => {
      if (disposed) {
        engine?.destroy();
        return;
      }
      if (engine) {
        gpuCmapRef.current = engine;
        gpuCmapReadyRef.current = true;
        setGpuCmapReadyVersion(version => version + 1);
      }
    }).catch(error => console.error("[Show2D] colormap engine failed to start", error));
    // A lost device (GPU process crash, or its watchdog resetting a hung queue)
    // never completes another paint or readback. Move the slots to the Canvas2D
    // engine; the reconcile effect re-uploads the current frames and repaints.
    const stopWatchingLoss = onGPULost(() => {
      if (disposed || !(gpuCmapRef.current instanceof GPUColormapEngine)) return;
      const lost = gpuCmapRef.current;
      gpuCmapRef.current = createCPUColormapEngine();
      lost.destroy();
      gpuFFTRef.current = null;
      setGpuCmapReadyVersion(version => version + 1);
    });
    return () => {
      disposed = true;
      stopWatchingLoss();
      gpuCmapReadyRef.current = false;
      const engine = gpuCmapRef.current;
      gpuCmapRef.current = null;
      engine?.destroy();
    };
  }, []);

  // Keep inline FFT ref arrays in sync with nImages
  React.useEffect(() => {
    fftCanvasRefs.current = fftCanvasRefs.current.slice(0, nImages);
    fftOffscreensRef.current = fftOffscreensRef.current.slice(0, nImages);
  }, [nImages]);

  // FFT of a single visible diff pair. Computes ref − other in JS at full image resolution,
  // feeds to FFT pipeline. Recomputes when raw data changes.
  React.useEffect(() => {
    if (!effectiveShowFft || !showDiffPanel || diffOtherIndices.length !== 1) return;
    const raw = rawDataRef.current;
    const otherIdx = diffOtherIndices[0];
    if (!raw || !raw[effectiveDiffReference] || !raw[otherIdx]) return;
    const refData = raw[effectiveDiffReference], otherData = raw[otherIdx];
    const difference = new Float32Array(width * height);
    for (let i = 0; i < difference.length; i++) difference[i] = refData[i] - otherData[i];
    const fftW = nextPow2(width), fftH = nextPow2(height);
    const real = new Float32Array(fftW * fftH);
    const imag = new Float32Array(fftW * fftH);
    if (fftWindow) applyHannWindow2D(difference, width, height);
    const padRows = Math.floor((fftH - height) / 2), padCols = Math.floor((fftW - width) / 2);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) real[(row + padRows) * fftW + col + padCols] = difference[row * width + col];
    }
    let cancelled = false;
    (async () => {
      const gpu = gpuFFTRef.current || await getDisplayFFT();
      gpuFFTRef.current = gpu;
      const result = await gpu.fft2D(real, imag, fftW, fftH, false);
      if (cancelled) return;
      fftshift(result.real, fftW, fftH);
      fftshift(result.imag, fftW, fftH);
      const magnitude = computeMagnitude(result.real, result.imag);
      diffFftMagRef.current = magnitude;
      diffFftDimsRef.current = { width: fftW, height: fftH };
      setDiffFftMagVersion(version => version + 1);
    })().catch(err => {
      if (!cancelled) console.warn("[Show2D] Diff FFT failed", err);
    });
    return () => { cancelled = true; };
  }, [effectiveShowFft, showDiffPanel, diffOtherIndices, effectiveDiffReference, dataVersion, width, height, fftWindow]);

  // Re-blit the cached diff FFT after a browser tab/page restore without
  // recomputing its magnitude.
  React.useLayoutEffect(() => {
    const canvas = diffFftCanvasRef.current;
    if (!effectiveShowFft || !showDiffPanel || !canvas) return;
    const magnitude = diffFftMagRef.current;
    const dims = diffFftDimsRef.current;
    if (!magnitude || !dims) return;
    const { min, max } = autoEnhanceFFT(magnitude, dims.width, dims.height);
    const offscreen = renderToOffscreen(
      magnitude,
      dims.width,
      dims.height,
      COLORMAPS[fftColormap] || COLORMAPS.inferno,
      min,
      max,
    );
    if (!offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = fftSmooth;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offscreen, 0, 0, offscreen.width, offscreen.height, 0, 0, canvasW, canvasH);
  }, [effectiveShowFft, showDiffPanel, diffFftMagVersion, fftColormap, canvasW, canvasH, fftSmooth, canvasRepaintSignal]);

  // Diff panels: one per non-reference image, image[ref] − image[i].
  // Computed at canvas resolution from raw float data, re-running on zoom/pan change.
  // Drift correction happens upstream, so the pair is subtracted without a shift.
  React.useEffect(() => {
    if (!showDiffPanel) return;
    const raw = rawDataRef.current;
    if (!raw || raw.length < 2) return;
    const ref = effectiveDiffReference;
    const refFrame = raw[ref];
    if (!refFrame) return;
    diffOtherIndices.forEach((otherIdx, slot) => {
      renderDiffPanel(slot, refFrame, raw[otherIdx], otherIdx);
    });
    // Hoisted function declaration: paints one diff slot.
    function renderDiffPanel(slot: number, refData: Float32Array, otherData: Float32Array | undefined, otherIdx: number) {
    if (!otherData) return;
    const canvas = diffCanvasRefs.current[slot];
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const refView = getZoomState(ref);
    const otherView = getZoomState(otherIdx);
    const a = refData, b = otherData;
    const cw = canvasW, ch = canvasH;
    const cx = cw / 2, cy = ch / 2;
    const sx = width / cw, sy = height / ch;
    const diff = new Float32Array(cw * ch);
    let diffMin = Infinity, diffMax = -Infinity;
    // Smooth: bilinear (slower, sub-pixel correct). !Smooth: nearest neighbor (faster, pixelated).
    const lastRow = height - 1, lastCol = width - 1;
    const refPanX = refView.panX, refPanY = refView.panY, refZoom = refView.zoom;
    const otherPanX = otherView.panX, otherPanY = otherView.panY, otherZoom = otherView.zoom;
    if (smooth) {
      for (let y = 0; y < ch; y++) {
        const ayu = (y - cy - refPanY) / refZoom + cy;
        const byu = (y - cy - otherPanY) / otherZoom + cy;
        const aRowF = ayu * sy;
        const bRowF = byu * sy;
        const aR0 = aRowF | 0, bR0 = bRowF | 0;
        const aFr = aRowF - aR0, bFr = bRowF - bR0;
        const aRowOOB = aR0 < 0 || aR0 >= lastRow;
        const bRowOOB = bR0 < 0 || bR0 >= lastRow;
        const aRowOff = aR0 * width;
        const bRowOff = bR0 * width;
        const rowOff = y * cw;
        for (let x = 0; x < cw; x++) {
          const axu = (x - cx - refPanX) / refZoom + cx;
          const bxu = (x - cx - otherPanX) / otherZoom + cx;
          const aColF = axu * sx;
          const bColF = bxu * sx;
          const aC0 = aColF | 0, bC0 = bColF | 0;
          let v = 0;
          if (!aRowOOB && !bRowOOB && aC0 >= 0 && aC0 < lastCol && bC0 >= 0 && bC0 < lastCol) {
            const aFc = aColF - aC0, bFc = bColF - bC0;
            const ai = aRowOff + aC0;
            const bi = bRowOff + bC0;
            const aV = (a[ai] * (1 - aFc) + a[ai + 1] * aFc) * (1 - aFr) +
                       (a[ai + width] * (1 - aFc) + a[ai + width + 1] * aFc) * aFr;
            const bV = (b[bi] * (1 - bFc) + b[bi + 1] * bFc) * (1 - bFr) +
                       (b[bi + width] * (1 - bFc) + b[bi + width + 1] * bFc) * bFr;
            v = aV - bV;
          }
          diff[rowOff + x] = v;
          if (v < diffMin) diffMin = v;
          if (v > diffMax) diffMax = v;
        }
      }
    } else {
      for (let y = 0; y < ch; y++) {
        const ayu = (y - cy - refPanY) / refZoom + cy;
        const byu = (y - cy - otherPanY) / otherZoom + cy;
        const aRow = (ayu * sy + 0.5) | 0;
        const bRow = (byu * sy + 0.5) | 0;
        const aRowOK = aRow >= 0 && aRow < height;
        const bRowOK = bRow >= 0 && bRow < height;
        const aRowOff = aRow * width;
        const bRowOff = bRow * width;
        const rowOff = y * cw;
        for (let x = 0; x < cw; x++) {
          const axu = (x - cx - refPanX) / refZoom + cx;
          const bxu = (x - cx - otherPanX) / otherZoom + cx;
          const aCol = (axu * sx + 0.5) | 0;
          const bCol = (bxu * sx + 0.5) | 0;
          let v = 0;
          if (aRowOK && bRowOK && aCol >= 0 && aCol < width && bCol >= 0 && bCol < width) {
            v = a[aRowOff + aCol] - b[bRowOff + bCol];
          }
          diff[rowOff + x] = v;
          if (v < diffMin) diffMin = v;
          if (v > diffMax) diffMax = v;
        }
      }
    }
    const diffLimit = Math.max(Math.abs(diffMin), Math.abs(diffMax));
    // Diff is signed around zero: use a diverging cmap (RdBu) if the user picked a sequential one.
    const sequentialCmaps = new Set(["inferno", "viridis", "plasma", "magma", "hot", "gray", "turbo"]);
    const diffCmap = sequentialCmaps.has(cmap) ? "RdBu" : cmap;
    const offscreen = renderToOffscreen(diff, cw, ch, COLORMAPS[diffCmap] || COLORMAPS.RdBu, -diffLimit, diffLimit);
    if (!offscreen) return;
    ctx.imageSmoothingEnabled = smooth;
    if (smooth) ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offscreen, 0, 0);
    }
  }, [showDiffPanel, diffOtherIndices, effectiveDiffReference, nImages, dataVersion, width, height, cmap, smooth, canvasW, canvasH,
      getZoomState, linkedZoom, linkPan, linkedZoomState, zoomStates, canvasRepaintSignal]);

  // --- Browser-side display filtering -------------------------------------
  // The kernel ships raw frames; the WGSL compute port (../display/filter.ts,
  // CPU fallback without WebGPU) applies the per-panel gaussian/bin/anscombe
  // knobs here, right before the arrays feed the colormap/FFT/histogram paths.
  // sigmaDraft feeds the filter directly during drag, so scrubbing sigma is
  // live with zero kernel round trips; the model commit happens on release.
  // Other mode names pass through untouched (browserFilterSupported is false).
  const sigmaDraftForFilter = browserFilterActive ? sigmaFilterDraft : null;
  const sigmaDraftPanel = sigmaDraftForFilter === null ? -1 : selectedIdx;
  const panelFilterKnobs = React.useCallback((panel: number) => {
    const { mode, sigma: resolvedSigma, bin } = resolvePanelDenoiseKnobs(
      panel, displayFilters, displaySigmas, spatialBins,
      { mode: displayFilter || "none", sigma: Number(displaySigma ?? 4), bin: Number(spatialBin || 1) },
    );
    let sigma = resolvedSigma;
    if (sigmaDraftForFilter !== null && (denoiseScopeAll || panel === sigmaDraftPanel)) {
      sigma = sigmaDraftForFilter;
    }
    return { mode, sigma, bin };
  }, [displayFilters, displaySigmas, spatialBins, displayFilter, displaySigma, spatialBin,
      sigmaDraftForFilter, sigmaDraftPanel, denoiseScopeAll]);
  const frequencyDraftPanel = frequencyDraft === null ? -1 : selectedIdx;
  const panelFrequencyKnobs = React.useCallback((panel: number) => {
    const mode = frequencyFilterModes?.[panel] ?? frequencyFilter ?? "none";
    let cutoff = Number(frequencyFilterCutoffs?.[panel] ?? frequencyFilterCutoff ?? 0.15);
    let center = Number(frequencyFilterCenters?.[panel] ?? frequencyFilterCenter ?? 0.30);
    const width = Number(frequencyFilterWidths?.[panel] ?? frequencyFilterWidth ?? 0.12);
    if (frequencyDraft !== null && (frequencyFilterScopeAll || panel === frequencyDraftPanel)) {
      if (normalizeFrequencyFilterMode(mode) === "bandpass") center = frequencyDraft;
      else cutoff = frequencyDraft;
    }
    return { mode: normalizeFrequencyFilterMode(mode), cutoff, center, width };
  }, [frequencyFilterModes, frequencyFilterCutoffs, frequencyFilterCenters, frequencyFilterWidths,
      frequencyFilter, frequencyFilterCutoff, frequencyFilterCenter, frequencyFilterWidth,
      frequencyDraft, frequencyFilterScopeAll, frequencyDraftPanel]);
  const filterFrameForPanel = React.useCallback(async (panel: number, frame: Float32Array): Promise<Float32Array> => {
    if (isRgbFlags && isRgbFlags[panel]) return frame;
    let displayed = frame;
    if (browserFilterActive) {
      const { mode, sigma, bin } = panelFilterKnobs(panel);
      if (filterKnobsActive(mode, bin) && browserFilterSupported(mode)) {
        displayed = await applyDisplayFilterBrowser(displayed, width, height, mode, sigma, bin);
      }
    }
    const frequency = panelFrequencyKnobs(panel);
    if (frequencyFilterEnabled && frequencyFilterActive(frequency.mode)) {
      displayed = await applyFrequencyFilterBrowser(displayed, width, height, frequency);
      setFrequencyFilterBackend(getFrequencyFilterBackend());
    }
    return displayed;
  }, [browserFilterActive, isRgbFlags, panelFilterKnobs, width, height, frequencyFilterEnabled,
      panelFrequencyKnobs]);
  // Generation token: any newer decode/scrub run invalidates pending async
  // filter commits, so a stale sigma never overwrites a fresher frame.
  const browserFilterGenerationRef = React.useRef(0);
  // Mirror of Python's _on_display_filter_scalar_change write-through, so
  // kernel-less pages (offline exports, saved notebooks) keep the per-panel
  // lists coherent when the scalar editor knobs change. With a kernel
  // attached Python performs the same update and converges to identical
  // lists, producing no extra change events.
  const mirrorFilterKnobEdit = React.useCallback((name: "mode" | "sigma" | "bin", value: string | number) => {
    if (!browserFilterActive || nImages <= 0) return;
    const idx = Math.min(Math.max(0, selectedIdx || 0), nImages - 1);
    if (name === "mode") setDisplayFilters(scopedKnobValues(displayFilters, String(value), "none", nImages, idx, denoiseScopeAll));
    else if (name === "sigma") setDisplaySigmas(scopedKnobValues(displaySigmas, Number(value), 4, nImages, idx, denoiseScopeAll));
    else setSpatialBins(scopedKnobValues(spatialBins, Number(value), 1, nImages, idx, denoiseScopeAll));
  }, [browserFilterActive, nImages, selectedIdx, denoiseScopeAll, displayFilters, displaySigmas,
      spatialBins, setDisplayFilters, setDisplaySigmas, setSpatialBins]);
  const mirrorFrequencyKnobEdit = React.useCallback((name: "mode" | "cutoff" | "center" | "width", value: string | number) => {
    if (nImages <= 0) return;
    const idx = Math.min(Math.max(0, selectedIdx || 0), nImages - 1);
    if (name === "mode") setFrequencyFilterModes(scopedKnobValues(frequencyFilterModes, String(value), "none", nImages, idx, frequencyFilterScopeAll));
    else if (name === "cutoff") setFrequencyFilterCutoffs(scopedKnobValues(frequencyFilterCutoffs, Number(value), 0.15, nImages, idx, frequencyFilterScopeAll));
    else if (name === "center") setFrequencyFilterCenters(scopedKnobValues(frequencyFilterCenters, Number(value), 0.30, nImages, idx, frequencyFilterScopeAll));
    else setFrequencyFilterWidths(scopedKnobValues(frequencyFilterWidths, Number(value), 0.12, nImages, idx, frequencyFilterScopeAll));
  }, [nImages, selectedIdx, frequencyFilterScopeAll, frequencyFilterModes, frequencyFilterCutoffs,
      frequencyFilterCenters, frequencyFilterWidths, setFrequencyFilterModes, setFrequencyFilterCutoffs,
      setFrequencyFilterCenters, setFrequencyFilterWidths]);
  // Local banner (format_display_filter_banner port): announcing an active
  // reduction is a house rule, and kernel-less pages have no Python to
  // refresh the synced banner trait. Live it also tracks sigmaDraft mid drag.
  const browserFilterBanner = React.useMemo(() => {
    if (!browserFilterActive || nImages <= 0) return null;
    const knobs = Array.from({ length: nImages }, (_, i) => panelFilterKnobs(i));
    const activePanels = knobs.map((_, i) => i).filter(i => filterKnobsActive(knobs[i].mode, knobs[i].bin));
    if (activePanels.length === 0) return "";
    const suffix = " (set denoise='none' for raw counts)";
    const uniform = knobs.every(knob =>
      normalizeFilterMode(knob.mode) === normalizeFilterMode(knobs[0].mode)
      && knob.sigma === knobs[0].sigma && knob.bin === knobs[0].bin);
    if (uniform) {
      const mode = normalizeFilterMode(knobs[0].mode);
      const parts = [mode === "none" ? "raw" : mode];
      if (mode !== "none") parts.push(`σ=${Number(knobs[0].sigma)}`);
      if (knobs[0].bin > 1) parts.push(`bin${knobs[0].bin}`);
      return `denoise: ${parts.join(" ")}${suffix}`;
    }
    const perPanel = activePanels.map(i =>
      `p${i}:${normalizeFilterMode(knobs[i].mode)} σ=${Number(knobs[i].sigma)}`
      + (knobs[i].bin > 1 ? ` bin${knobs[i].bin}` : "")).join(", ");
    return `denoise: ${perPanel}${suffix}`;
  }, [browserFilterActive, nImages, panelFilterKnobs]);
  const filterBannerText = browserFilterActive ? (browserFilterBanner ?? "") : (displayFilterBanner || "");

  React.useEffect(() => {
    if (!allFloats || allFloats.length === 0) return;
    const dataArrays: Float32Array[] = [];
    const rgbArrays: (Float32Array | null)[] = [];
    const perImage = width * height;
    for (let i = 0; i < nImages; i++) {
      const start = panelFloatOffsets[i];
      if (isRgbFlags && isRgbFlags[i]) {
        // RGB panel: keep the interleaved color pixels for direct painting and
        // reduce to luminance so the grayscale analysis paths stay valid.
        const rgb = new Float32Array(allFloats.subarray(start, start + 3 * perImage));
        rgbArrays.push(rgb);
        const luminance = new Float32Array(perImage);
        for (let k = 0; k < perImage; k++) {
          luminance[k] = 0.2126 * rgb[3 * k] + 0.7152 * rgb[3 * k + 1] + 0.0722 * rgb[3 * k + 2];
        }
        dataArrays.push(luminance);
      } else {
        rgbArrays.push(null);
        const frameCount = panelFrameCounts?.[i] || 1;
        const stackOffset = panelStackOffsets?.[i] ?? -1;
        const requestedFrame = normalizedPanelFrameIndices[i] || 0;
        const frameIndex = Math.max(0, Math.min(frameCount - 1, requestedFrame));
        if (frameCount > 1 && stackOffset >= 0 && allPanelStackFloats.length >= stackOffset + (frameIndex + 1) * perImage) {
          const frameStart = stackOffset + frameIndex * perImage;
          dataArrays.push(new Float32Array(allPanelStackFloats.subarray(frameStart, frameStart + perImage)));
        } else {
          const frame = allFloats.subarray(start, start + perImage);
          dataArrays.push(uint8FolderPreviewMode ? frame : new Float32Array(frame));
        }
      }
    }
    const generation = ++browserFilterGenerationRef.current;
    const previousSource = filterInputSourceRef.current;
    const sourceChanged = !previousSource
      || previousSource.allFloats !== allFloats
      || previousSource.allPanelStackFloats !== allPanelStackFloats
      || previousSource.frameSourceKey !== frameSourceKey
      || previousSource.width !== width
      || previousSource.height !== height
      || previousSource.nImages !== nImages;
    const panelViewSignatures = Array.from({ length: nImages }, (_, panel) => {
      if (isRgbFlags && isRgbFlags[panel]) return "rgb";
      const denoise = panelFilterKnobs(panel);
      const frequency = panelFrequencyKnobs(panel);
      const denoiseSignature = browserFilterActive && filterKnobsActive(denoise.mode, denoise.bin)
        ? `${normalizeFilterMode(denoise.mode)}:${denoise.sigma}:${denoise.bin}`
        : "off";
      const frequencySignature = frequencyFilterEnabled && frequencyFilterActive(frequency.mode)
        ? `${frequency.mode}:${frequency.cutoff}:${frequency.center}:${frequency.width}`
        : "off";
      return `${denoiseSignature}|${frequencySignature}`;
    });
    const changedPanels = sourceChanged
      ? Array.from({ length: nImages }, (_, panel) => panel)
      : panelViewSignatures
        .map((signature, panel) => signature !== appliedPanelViewSignaturesRef.current[panel] ? panel : -1)
        .filter(panel => panel >= 0);
    if (changedPanels.length === 0) return;
    const changedPanelSet = new Set(changedPanels);
    const commit = (arrays: Float32Array[]) => {
      rawDataRef.current = arrays;
      rgbDataRef.current = rgbArrays;
      lastAppliedPanelFrameIndicesRef.current = [...normalizedPanelFrameIndices];
      filterInputSourceRef.current = { allFloats, allPanelStackFloats, frameSourceKey, width, height, nImages };
      appliedPanelViewSignaturesRef.current = panelViewSignatures;
      const epochs = galleryFftPanelEpochsRef.current.length === nImages
        ? [...galleryFftPanelEpochsRef.current]
        : new Array(nImages).fill(0);
      const activeKeys = galleryFftActiveKeysRef.current.length === nImages
        ? [...galleryFftActiveKeysRef.current]
        : new Array(nImages).fill(null);
      const targetKeys = galleryFftTargetKeysRef.current.length === nImages
        ? [...galleryFftTargetKeysRef.current]
        : new Array(nImages).fill(null);
      const magnitudes = fftMagCacheGalleryRef.current.length === nImages
        ? [...fftMagCacheGalleryRef.current]
        : new Array(nImages).fill(null);
      const offscreens = fftOffscreensRef.current.length === nImages
        ? [...fftOffscreensRef.current]
        : new Array(nImages).fill(null);
      const pipelines = galleryFftPipelineRef.current.length === nImages
        ? [...galleryFftPipelineRef.current]
        : new Array(nImages).fill(null);
      for (const panel of changedPanels) {
        epochs[panel] += 1;
        activeKeys[panel] = null;
        targetKeys[panel] = null;
        magnitudes[panel] = null;
        offscreens[panel] = null;
        pipelines[panel] = null;
      }
      galleryFftPanelEpochsRef.current = epochs;
      galleryFftActiveKeysRef.current = activeKeys;
      galleryFftTargetKeysRef.current = targetKeys;
      fftMagCacheGalleryRef.current = magnitudes;
      fftOffscreensRef.current = offscreens;
      galleryFftPipelineRef.current = pipelines;
      if (sourceChanged) galleryFftMagnitudeLruRef.current.clear();
      // New pixels (fresh frame_bytes, rotation, ...) invalidate every fetched
      // detail tile; the request effect refetches for the current view.
      detailTilesRef.current.clear();
      detailSentKeysRef.current.clear();
      setDetailStreamStatus("preview");
      setDataVersion(version => version + 1);
    };
    const needsDenoise = browserFilterActive && dataArrays.some((_, i) => {
      if (isRgbFlags && isRgbFlags[i]) return false;
      const { mode, bin } = panelFilterKnobs(i);
      return filterKnobsActive(mode, bin) && browserFilterSupported(mode);
    });
    const needsFrequencyFilter = !!frequencyFilterEnabled && dataArrays.some((_, panel) => frequencyFilterActive(panelFrequencyKnobs(panel).mode));
    const needsBrowserFilter = needsDenoise || needsFrequencyFilter;
    if (!needsBrowserFilter) { commit(dataArrays); return; }
    const previousArrays = rawDataRef.current;
    Promise.all(dataArrays.map((frame, i) => changedPanelSet.has(i)
      ? filterFrameForPanel(i, frame)
      : Promise.resolve(previousArrays?.[i] ?? frame))).then(filtered => {
      if (browserFilterGenerationRef.current === generation) commit(filtered);
    }).catch(error => console.error("[Show2D] hardware WebGPU view pipeline failed", error));
  }, [
    allFloats,
    allPanelStackFloats,
    nImages,
    width,
    height,
    panelFloatOffsets,
    isRgbFlags,
    frameSourceKey,
    offlineMin,
    offlineMax,
    offlineMins,
    offlineMaxs,
    panelFrameCounts,
    panelStackOffsets,
    browserFilterActive,
    panelFilterKnobs,
    filterFrameForPanel,
    frequencyFilterEnabled,
    panelFrequencyKnobs,
    uint8FolderIdentityEncoding,
    uint8FolderPreviewMode,
  ]);

  React.useEffect(() => {
    const raw = rawDataRef.current;
    if (!raw || !hasLocalPanelStacks || allPanelStackFloats.length === 0) return;
    const previous = lastAppliedPanelFrameIndicesRef.current;
    const perImage = width * height;
    const changedFrames: { panel: number; frame: Float32Array }[] = [];
    for (let panel = 0; panel < nImages; panel++) {
      const frameCount = panelFrameCounts?.[panel] || 1;
      const stackOffset = panelStackOffsets?.[panel] ?? -1;
      const frameIndex = normalizedPanelFrameIndices[panel] || 0;
      if (frameCount <= 1 || stackOffset < 0 || previous[panel] === frameIndex) continue;
      const frameStart = stackOffset + frameIndex * perImage;
      if (allPanelStackFloats.length < frameStart + perImage) continue;
      changedFrames.push({
        panel,
        frame: new Float32Array(allPanelStackFloats.subarray(frameStart, frameStart + perImage)),
      });
    }
    lastAppliedPanelFrameIndicesRef.current = [...normalizedPanelFrameIndices];
    if (changedFrames.length === 0) return;
    // Stack frames ship raw when the browser owns the filter; apply the
    // panel's knobs before commit so scrubbing a filtered stack panel shows
    // the same view as the main frame. The decode effect re-runs on any newer
    // data, so a bumped generation just drops this pending commit.
    const generation = browserFilterGenerationRef.current;
    Promise.all(changedFrames.map(({ panel, frame }) =>
      filterFrameForPanel(panel, frame).then(filtered => ({ panel, filtered }))
    )).then(results => {
      if (browserFilterGenerationRef.current !== generation) return;
      for (const { panel, filtered } of results) {
        raw[panel] = filtered;
        if (fftMagCacheGalleryRef.current.length === nImages) {
          fftMagCacheGalleryRef.current[panel] = null;
        }
        if (galleryFftPipelineRef.current.length === nImages) {
          galleryFftPipelineRef.current[panel] = null;
        }
      }
      detailTilesRef.current.clear();
      detailSentKeysRef.current.clear();
      setDetailStreamStatus("preview");
      setDataVersion(version => version + 1);
    }).catch(error => console.error("[Show2D] hardware WebGPU view pipeline failed", error));
  }, [
    allPanelStackFloats,
    hasLocalPanelStacks,
    height,
    nImages,
    normalizedPanelFrameIndices,
    panelFrameCounts,
    panelStackOffsets,
    width,
    filterFrameForPanel,
  ]);

  React.useEffect(() => {
    const raw = rawDataRef.current;
    if (!raw || !hasLocalPanelStacks) {
      setLocalPanelFrameStats(previous => previous.size === 0 ? previous : new Map());
      return;
    }
    const next = new Map<number, { mean: number; min: number; max: number; std: number }>();
    for (let panel = 0; panel < nImages; panel++) {
      if ((panelFrameCounts?.[panel] || 1) <= 1 || !raw[panel]) continue;
      const values = raw[panel];
      let min = Infinity;
      let max = -Infinity;
      let sum = 0;
      let sumSquares = 0;
      for (let i = 0; i < values.length; i++) {
        const value = values[i];
        if (!Number.isFinite(value)) continue;
        min = Math.min(min, value);
        max = Math.max(max, value);
        sum += value;
        sumSquares += value * value;
      }
      const count = values.length;
      const mean = count > 0 ? sum / count : 0;
      const variance = count > 0 ? Math.max(0, sumSquares / count - mean * mean) : 0;
      next.set(panel, {
        mean,
        min: Number.isFinite(min) ? min : 0,
        max: Number.isFinite(max) ? max : 0,
        std: Math.sqrt(variance),
      });
    }
    setLocalPanelFrameStats(next);
  }, [dataVersion, hasLocalPanelStacks, nImages, panelFrameCounts]);

  // Initialize reusable offscreen canvases (one per image, resized when dimensions change)
  React.useEffect(() => {
    if (width <= 0 || height <= 0 || nImages <= 0) return;
    if (uint8FolderPreviewMode && (canvasW <= 0 || canvasH <= 0)) return;
    const offscreenW = uint8FolderPreviewMode ? Math.max(1, Math.round(canvasW)) : width;
    const offscreenH = uint8FolderPreviewMode ? Math.max(1, Math.round(canvasH)) : height;
    const current = mainOffscreensRef.current;
    if (
      current.length === nImages
      && current.every(canvas => canvas && canvas.width === offscreenW && canvas.height === offscreenH)
      && mainImgDatasRef.current.length === nImages
      && mainImgDatasRef.current.every(imgData => imgData.width === offscreenW && imgData.height === offscreenH)
    ) {
      return;
    }
    const canvases: HTMLCanvasElement[] = [];
    const imgDatas: ImageData[] = [];
    for (let i = 0; i < nImages; i++) {
      const canvas = document.createElement("canvas");
      canvas.width = offscreenW;
      canvas.height = offscreenH;
      canvases.push(canvas);
      imgDatas.push(canvas.getContext("2d")!.createImageData(offscreenW, offscreenH));
    }
    mainOffscreensRef.current = canvases;
    mainImgDatasRef.current = imgDatas;
  }, [width, height, nImages, canvasW, canvasH, uint8FolderPreviewMode]);

  // Compute histogram data for the displayed image (reflects log scale)
  // Persistent WebGPU slot histograms are the only numerical source.
  React.useEffect(() => {
    if (!rawDataRef.current) return;
    const idx = nImages > 1 ? selectedIdx : 0;
    const raw = rawDataRef.current[idx];
    if (!raw) return;

    const hasAbsoluteRange = traitVmin != null && traitVmax != null;
    const hasAnyPerImageRange = Array.from({ length: nImages }).some((_, i) => (
      traitVmins && traitVmaxs && traitVmins[i] != null && traitVmaxs[i] != null
    ));
    const linkedHistogram = linkedContrast && isGallery && !hasAbsoluteRange && !hasAnyPerImageRange;
    if (rawRangesRef.current.length < nImages) return;
    const imageRanges = Array.from({ length: nImages }, (_, i) => {
      const rawRange = rawRangesRef.current[i];
      return displayRange(rawRange.min, rawRange.max, logScale);
    });
    const range = linkedHistogram ? mergeDataRanges(imageRanges) : (imageRanges[idx] || { min: 0, max: 1 });
    setImageDataRange(range);

    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current && engine.slotCount > idx) {
      if (linkedHistogram && engine.slotCount >= nImages) {
        const indices = Array.from({ length: nImages }, (_, i) => i);
        engine.computeHistogramBatch(indices, indices.map(() => range), logScale).then(histograms => {
          const merged = mergeHistogramBins(histograms);
          // All-zero bins mean the GPU slots were not populated yet: keep the
          // previous bars rather than drawing an empty distribution.
          const hasSignal = histograms.length > 0 && merged.some(count => count > 0);
          if (hasSignal) {
            setImageHistogramBins(merged);
            setImageHistogramData(null);
          }
        });
      } else {
        // GPU histogram - single image, persistent buffers
        engine.computeHistogramWithRange(idx, range.min, range.max, logScale).then(bins => {
          // Zero-only bins mean the slot data is not populated yet: keep the
          // previous bars, as in the linked histogram path.
          const hasSignal = bins && bins.length > 0 && bins.some(count => count > 0);
          if (hasSignal) {
            setImageHistogramBins(bins);
            setImageHistogramData(null);
          }
        });
      }
    } else {
      setImageHistogramBins(null);
      setImageHistogramData(null);
    }
  }, [rangeVersion, nImages, logScale, selectedIdx, linkedContrast, isGallery, traitVmin, traitVmax, traitVmins, traitVmaxs]);

  // Prevent page scroll when scrolling on canvases (must use native listener with passive: false)
  // In gallery mode, only block scroll on the selected image (or all if linkedZoom)
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const elements: (HTMLElement | null)[] = isGallery
      ? (linkedZoom
          ? [
              ...imageContainerRefs.current,
              ...(effectiveShowFft ? fftContainerRefs.current : []),
            ]
          : [
              imageContainerRefs.current[selectedIdx],
              ...(effectiveShowFft ? [fftContainerRefs.current[selectedIdx]] : []),
            ])
      : [
          imageContainerRefs.current[0],
          ...(effectiveShowFft ? [singleFftContainerRef.current] : []),
        ];
    elements.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => elements.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, [canvasReady, effectiveShowFft, isGallery, selectedIdx, linkedZoom]);

  // The one place frames reach the colormap engine. It reconciles the two
  // independent async prerequisites of the first paint, parsed/filtered data
  // and the widget-owned engine: whichever becomes ready second uploads the
  // CURRENT rawDataRef (the display-filtered view when Denoise/Filter is
  // active). A panel is uploaded only when its array changed since the last
  // upload (every panel for a new engine, a new size, or packed uint8 frames
  // on a resized canvas), and gpuCmapVersion moves once per upload, so the
  // range, histogram, auto-contrast and colormap passes run once per frame.
  const uploadedFramesRef = React.useRef<{ engine: DisplayColormapEngine | null; key: string; arrays: (Float32Array | null)[] }>({ engine: null, key: "", arrays: [] });
  const uint8CanvasPixels = uint8FolderPreviewMode ? Math.max(1, Math.round(canvasW) * Math.round(canvasH)) : 0;
  React.useEffect(() => {
    const engine = gpuCmapRef.current;
    const arrays = rawDataRef.current;
    if (!engine || !gpuCmapReadyRef.current || !arrays || arrays.length === 0) return;
    const uploaded = uploadedFramesRef.current;
    const key = `${width}x${height}:${uint8CanvasPixels}`;
    if (uploaded.engine !== engine || uploaded.key !== key) uploadedFramesRef.current = { engine, key, arrays: [] };
    const record = uploadedFramesRef.current.arrays;
    let changed = false;
    for (let i = 0; i < arrays.length; i++) {
      const data = arrays[i];
      if (!data || record[i] === data) continue;
      if (uint8FolderPreviewMode) {
        if (!ArrayBuffer.isView(data) || data.BYTES_PER_ELEMENT !== 1) {
          // The engine-ready effect can beat the folder fetch/commit by one
          // render and still see the old float placeholder. The data-version
          // rerun uploads the authoritative byte frame; never reinterpret the
          // placeholder as packed uint8.
          continue;
        }
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        engine.uploadUint8Data(i, bytes, width, height, uint8CanvasPixels);
      } else {
        engine.uploadData(i, data, width, height);
      }
      record[i] = data;
      changed = true;
    }
    if (changed) setGpuCmapVersion(version => version + 1);
  }, [dataVersion, gpuCmapReadyVersion, width, height, uint8CanvasPixels, uint8FolderPreviewMode]);
  // Cached per-image data ranges: only recomputed when data or logScale changes, NOT on slider drag
  const dataRangesRef = React.useRef<{ min: number; max: number }[]>([]);
  // Ref mirrors for async GPU callbacks (avoid stale closures)
  const logScaleRef = React.useRef(logScale);
  logScaleRef.current = logScale;
  const panelCmapForRef = React.useRef(panelCmapFor);
  panelCmapForRef.current = panelCmapFor;
  const uint8FolderPreviewModeRef = React.useRef(uint8FolderPreviewMode);
  uint8FolderPreviewModeRef.current = uint8FolderPreviewMode;
  const autoContrastRequestRef = React.useRef(0);

  // Cache per-image data ranges (raw AND log) on data change only.
  // Log ranges are derived mathematically: log1p(rawMin), log1p(rawMax).
  // NO applyLogScale here: the GPU shader handles log1p per pixel.
  // Log toggle is now free: just pick the right cached ranges.
  // Uploaded frames' ranges are stale until this pass returns: the colormap
  // waits for them (an old window never paints new pixels), and Auto awaits
  // the same request instead of starting its own.
  const rawRangesRef = React.useRef<{ min: number; max: number }[]>([]);
  const rangesRequestRef = React.useRef<Promise<{ min: number; max: number }[]> | null>(null);
  React.useEffect(() => {
    if (!rawDataRef.current || rawDataRef.current.length === 0) return;
    autoContrastRequestRef.current += 1;
    autoContrastCacheRef.current = [];
    rawRangesRef.current = [];
    dataRangesRef.current = [];
    rangesRequestRef.current = null;
    const engine = gpuCmapRef.current;
    const panelCount = rawDataRef.current.length;
    if (!engine || !gpuCmapReadyRef.current || engine.slotCount < panelCount) return;
    const request = engine.computeRangeBatch(Array.from({ length: panelCount }, (_, i) => i));
    rangesRequestRef.current = request;
    request.then(rawRanges => {
      if (rangesRequestRef.current !== request) return;
      rawRangesRef.current = rawRanges;
      dataRangesRef.current = logScaleRef.current ? rawRanges.map(range => displayRange(range.min, range.max, true)) : rawRanges;
      setRangeVersion(version => version + 1);
    });
  }, [gpuCmapVersion, uint8FolderPreviewMode]);

  // When logScale toggles, just swap cached ranges (no data scan)
  React.useEffect(() => {
    if (rawRangesRef.current.length === 0) return;
    autoContrastRequestRef.current += 1;
    autoContrastCacheRef.current = [];
    const logRanges = rawRangesRef.current.map(range => displayRange(range.min, range.max, true));
    dataRangesRef.current = logScale ? logRanges : rawRangesRef.current;
  }, [logScale]);

  // GPU auto-contrast: batch-compute percentile ranges from GPU histograms.
  // One GPU submission for all images. Caches results for synchronous use in render.
  React.useEffect(() => {
    if (!autoContrast) { autoContrastCacheRef.current = []; return; }
    const engine = gpuCmapRef.current;
    const rangesRequest = rangesRequestRef.current;
    if (!engine || !gpuCmapReadyRef.current || !rawDataRef.current || !rangesRequest) return;
    const panelCount = Math.min(rawDataRef.current.length, engine.slotCount);
    if (panelCount === 0) return;
    const request = ++autoContrastRequestRef.current;

    (async () => {
      const indices = Array.from({ length: panelCount }, (_, i) => i);
      // The range pass may still be in flight when Auto first runs.
      const rawRanges = await rangesRequest;
      if (request !== autoContrastRequestRef.current || rawRanges.length < panelCount) return;
      const cachedRanges = logScale ? rawRanges.map(range => displayRange(range.min, range.max, true)) : rawRanges;
      const histRanges = indices.map(i => cachedRanges[i] || { min: 0, max: 1 });
      const allBins = await engine.computeHistogramBatch(indices, histRanges, logScale);

      const percentileLow = 2, percentileHigh = 98;
      const autoRanges: { vmin: number; vmax: number }[] = [];
      for (let k = 0; k < allBins.length; k++) {
        const bins = allBins[k];
        const histRange = histRanges[k];
        // Percentile from normalized histogram CDF
        let sum = 0;
        for (let bin = 0; bin < 256; bin++) sum += bins[bin];
        let binLow = 0, binHigh = 255;
        const targetLow = sum * percentileLow / 100;
        const targetHigh = sum * percentileHigh / 100;
        let running = 0;
        for (let bin = 0; bin < 256; bin++) {
          running += bins[bin];
          if (running >= targetLow && binLow === 0) binLow = bin;
          if (running >= targetHigh) { binHigh = bin; break; }
        }
        const range = histRange.max - histRange.min;
        autoRanges.push({ vmin: histRange.min + (binLow / 255) * range, vmax: histRange.min + (binHigh / 255) * range });
      }
      if (autoRanges.length < panelCount) return;
      if (request !== autoContrastRequestRef.current) return;
      autoContrastCacheRef.current = autoRanges;
      // Reflect the auto-computed range on the histogram dual-thumb slider so
      // the operator sees what's actually applied. Without this, the slider
      // sits at 0-100 (user's untouched state) while the image renders at
      // 2-98 percentile, which is confusing.
      // Histogram axis = full per-panel data range (histRanges, awaited above
      // when the range pass was still in flight).
      const autoPercents: { panel: number; vminPct: number; vmaxPct: number }[] = [];
      for (let k = 0; k < autoRanges.length; k++) {
        const histRange = histRanges[k];
        const autoRange = autoRanges[k];
        if (!autoRange) continue;
        if (!histRange || histRange.max <= histRange.min) continue;
        const vminPct = Math.max(0, Math.min(100, ((autoRange.vmin - histRange.min) / (histRange.max - histRange.min)) * 100));
        const vmaxPct = Math.max(0, Math.min(100, ((autoRange.vmax - histRange.min) / (histRange.max - histRange.min)) * 100));
        autoPercents.push({ panel: k, vminPct, vmaxPct });
      }
      // Skip the pct-write when explicit vmin/vmax traits are set: they
      // anchor the display range, so writing pcts derived from data range
      // produces a histogram-thumb mismatch (degenerate -0.3/-0.3 case).
      const traitsAnchor = traitVmin != null && traitVmax != null;
      const hasPerImageTraits = traitVmins && traitVmaxs && traitVmins.some((v, i) => v != null && traitVmaxs[i] != null);
      if (!traitsAnchor && !hasPerImageTraits) {
        const linkedAutoPercentRange = linkedContrast && autoPercents.length > 0
          ? linkedAutoContrastPercentRange(histRanges, autoRanges, indices, isRgbFlags)
          : undefined;
        // The slider preview bypasses React state for low-latency GPU paints.
        // Mirror auto-derived panel windows into that ref before publishing
        // state so the first manual drag starts from the visible ranges.
        const seededMirror = seedAutoContrastMirror(
          contrastRef.current,
          autoPercents,
          linkedContrast,
          linkedAutoPercentRange,
        );
        contrastRef.current = seededMirror;
        // Write all panel pcts in a single state update.
        setContrastStates(prev => {
          const next = new Map(prev);
          for (const entry of autoPercents) next.set(entry.panel, { vminPct: entry.vminPct, vmaxPct: entry.vmaxPct });
          return next;
        });
        // Linked-contrast mode reads `linkedContrastState`, not the per-panel
        // map. Mirror the auto range into it so the dual-thumb slider reflects
        // Auto when contrast is grouped. Use the widest envelope so all panels
        // still display within the active bars.
        if (linkedContrast && autoPercents.length > 0) {
          setLinkedContrastState(seededMirror.linked);
        }
      }
      setAutoContrastVersion(version => version + 1);
    })();
  }, [autoContrast, logScale, gpuCmapVersion, linkedContrast, traitVmin, traitVmax, traitVmins, traitVmaxs, uint8FolderPreviewMode, isRgbFlags]);

  // -------------------------------------------------------------------------
  // Data effect: normalize + colormap → reusable offscreen canvases.
  // The colormap engine (WebGPU, or its Canvas2D fallback) colors every visible
  // panel in one pass.
  // (does NOT depend on zoom/pan, so a pan or zoom never recolors 16M pixels)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!dataVersion || !rawDataRef.current || rawDataRef.current.length === 0) return;
    if (mainOffscreensRef.current.length === 0 || mainImgDatasRef.current.length === 0) return;

    const renderGeneration = ++mainCmapGenerationRef.current;
    let cancelled = false;
    let renderRaf: number | null = null;
    const isCurrentRender = () => (
      !cancelled
      && renderGeneration === mainCmapGenerationRef.current
      && (typeof document === "undefined" || !document.hidden)
    );
    const panelCmapNames = Array.from({ length: nImages }, (_, i) => panelCmapFor(i));
    const hasMixedPanelCmaps = panelCmapNames.some(name => name !== panelCmapNames[0]);
    const lutName = panelCmapNames[0] || cmap;
    const lut = COLORMAPS[lutName] || COLORMAPS.inferno;

    // RGB panels bypass the LUT + contrast pipeline entirely: paint their
    // display-ready (r, g, b) floats straight into the offscreen once per data
    // change. The colormap loops below skip these panels so cmap/contrast/log
    // changes never overwrite the color pixels.
    if (isRgbFlags && isRgbFlags.some(Boolean)) {
      for (const i of visibleImageIndices) {
        if (!isRgbFlags[i]) continue;
        const rgb = rgbDataRef.current[i];
        const offscreen = mainOffscreensRef.current[i];
        const imgData = mainImgDatasRef.current[i];
        if (!rgb || !offscreen || !imgData) continue;
        const pixels = imgData.data;
        for (let k = 0, n = width * height; k < n; k++) {
          pixels[4 * k] = Math.max(0, Math.min(255, Math.round(rgb[3 * k] * 255)));
          pixels[4 * k + 1] = Math.max(0, Math.min(255, Math.round(rgb[3 * k + 1] * 255)));
          pixels[4 * k + 2] = Math.max(0, Math.min(255, Math.round(rgb[3 * k + 2] * 255)));
          pixels[4 * k + 3] = 255;
        }
        offscreen.getContext("2d")?.putImageData(imgData, 0, 0);
      }
    }

    // Compute per-image vmin/vmax from CACHED data ranges (no findDataRange per tick).
    // dataRangesRef is precomputed when data or logScale changes.
    const cachedRanges = dataRangesRef.current;
    if (cachedRanges.length < nImages) return;
    const hasAbsoluteRange = traitVmin != null && traitVmax != null;
    const baseRanges: { min: number; max: number }[] = [];
    const hasPerImageRanges: boolean[] = [];
    for (let i = 0; i < nImages; i++) {
      const panelVmin = traitVmins && traitVmins[i] != null ? traitVmins[i] : null;
      const panelVmax = traitVmaxs && traitVmaxs[i] != null ? traitVmaxs[i] : null;
      const hasPerImage = panelVmin != null && panelVmax != null;
      hasPerImageRanges.push(hasPerImage);
      if (hasPerImage) {
        baseRanges.push(displayRange(panelVmin!, panelVmax!, logScale));
      } else if (hasAbsoluteRange) {
        baseRanges.push(displayRange(traitVmin!, traitVmax!, logScale));
      } else {
        const cached = cachedRanges[i];
        if (!cached) return;
        baseRanges.push(cached);
      }
    }
    const linkedSharedContrast = linkedContrast && isGallery && !hasAbsoluteRange && !hasPerImageRanges.some(Boolean);
    // Linked contrast merges GRAYSCALE panels only: an RGB overlay's [0, 1]
    // luminance range must never drag a counts-scaled panel's shared window.
    const grayOnly = <T,>(items: T[]): T[] => {
      const filtered = items.filter((_, i) => !(isRgbFlags && isRgbFlags[i]));
      return filtered.length > 0 ? filtered : items;
    };
    const sharedBaseRange = linkedSharedContrast ? mergeDataRanges(grayOnly(baseRanges)) : null;
    let sharedAutoRange: { vmin: number; vmax: number } | null = null;
    if (linkedSharedContrast && autoContrast) {
      const cachedAutoRanges = autoContrastCacheRef.current.slice(0, nImages);
      if (cachedAutoRanges.length === nImages && cachedAutoRanges.every(cachedRange => cachedRange && Number.isFinite(cachedRange.vmin) && Number.isFinite(cachedRange.vmax) && cachedRange.vmax > cachedRange.vmin)) {
        const merged = mergeDataRanges(grayOnly(cachedAutoRanges.map(cachedRange => ({ min: cachedRange.vmin, max: cachedRange.vmax }))));
        sharedAutoRange = { vmin: merged.min, vmax: merged.max };
      } else return;
    }
    const ranges: { vmin: number; vmax: number }[] = [];
    for (let i = 0; i < nImages; i++) {
      let vmin: number, vmax: number;
      const contrast = linkedContrast ? linkedContrastState : (contrastStates.get(i) || { vminPct: 0, vmaxPct: 100 });
      const hasPerImage = hasPerImageRanges[i];
      const range = sharedBaseRange || baseRanges[i] || { min: 0, max: 1 };
      const rangeMin = range.min;
      const rangeMax = range.max;

      if (!hasAbsoluteRange && !hasPerImage && autoContrast) {
        if (sharedAutoRange) {
          vmin = sharedAutoRange.vmin; vmax = sharedAutoRange.vmax;
          ranges.push({ vmin, vmax });
          continue;
        }
        // Auto-contrast uses the GPU-precomputed percentile range. Retain the
        // previous frame until that asynchronous range is ready.
        const autoRange = autoContrastCacheRef.current[i];
        if (autoRange && Number.isFinite(autoRange.vmin) && Number.isFinite(autoRange.vmax) && autoRange.vmax > autoRange.vmin) {
          vmin = autoRange.vmin; vmax = autoRange.vmax;
        } else return;
      } else if (rangeMin !== rangeMax && (contrast.vminPct > 0 || contrast.vmaxPct < 100)) {
        ({ vmin, vmax } = sliderRange(rangeMin, rangeMax, contrast.vminPct, contrast.vmaxPct));
      } else {
        vmin = rangeMin; vmax = rangeMax;
      }
      ranges.push({ vmin, vmax });
    }

    // Cache first image's vmin/vmax for colorbar/lens
    if (ranges.length > 0) {
      colorbarVminRef.current = ranges[0].vmin;
      colorbarVmaxRef.current = ranges[0].vmax;
    }
    panelRangesRef.current = ranges;  // keep detail tiles on the live contrast window

    // GPU colormap. A tab can be backgrounded while the
    // bitmap transfer is pending, so only the current visible generation may
    // commit into the retained offscreens. This also prevents a late black GPU
    // clear frame from overwriting a newer foreground repaint.
    const engine = gpuCmapRef.current;
    const gpuReady = engine && gpuCmapReadyRef.current && engine.slotCount >= nImages;
    if (gpuReady && (uint8FolderPreviewMode || hasMixedPanelCmaps)) {
      const capturedRanges = ranges.slice();
      const capturedLogScale = logScale;
      const capturedNImages = nImages;
      renderRaf = requestAnimationFrame(() => {
        renderRaf = null;
        void (async () => {
          const indices = visibleImageIndices.filter(i => i >= 0 && i < capturedNImages);
          let bitmaps: (ImageBitmap | null)[] = [];
          try {
            if (!isCurrentRender()) return;
            // Encode every visible panel before awaiting the queue. Awaiting
            // each panel serially made a 40-image native-4K gallery take long
            // enough for a newer React generation to cancel the whole paint,
            // leaving every scientific canvas transparent.
            const panels = indices.filter(i => !(isRgbFlags && isRgbFlags[i]));
            bitmaps = await renderScaledPanelBitmaps(
              engine!,
              panels,
              panels.map(i => capturedRanges[i] || { vmin: 0, vmax: 1 }),
              capturedLogScale,
              panelCmapNames,
              mainOffscreensRef.current,
            );
            if (!isCurrentRender()) return;
            let painted = !!(isRgbFlags && isRgbFlags.some(Boolean));
            for (let k = 0; k < bitmaps.length; k++) {
              const bitmap = bitmaps[k];
              if (!bitmap) continue;
              const offscreen = mainOffscreensRef.current[panels[k]];
              offscreen?.getContext("2d")?.drawImage(bitmap, 0, 0);
              painted = true;
            }
            if (painted && isCurrentRender()) setOffscreenVersion(version => version + 1);
          } catch (err) {
            if (isCurrentRender()) console.error("[Show2D] scaled WebGPU colormap repaint failed", err);
          } finally {
            bitmaps.forEach(bitmap => bitmap?.close());
          }
        })();
      });
    } else if (gpuReady) {
      const capturedRanges = ranges.slice();
      const capturedLogScale = logScale;
      const capturedNImages = nImages;
      const capturedIsRgb = isRgbFlags ? isRgbFlags.slice() : [];
      renderRaf = requestAnimationFrame(() => {
        renderRaf = null;
        void (async () => {
          const indices = visibleImageIndices.filter(i => i >= 0 && i < capturedNImages);
          let bitmaps: (ImageBitmap | null)[] | null = null;
          const closeBitmaps = () => {
            bitmaps?.forEach(bitmap => bitmap?.close());
            bitmaps = null;
          };
          try {
            if (!isCurrentRender()) return;
            // Await GPU completion before snapshotting so the OffscreenCanvas
            // contains the colormap instead of the render pass's black clear.
            const bitmapRanges = indices.map(i => capturedRanges[i] || { vmin: 0, vmax: 1 });
            bitmaps = await engine!.renderSlotsToImageBitmapAsync(indices, bitmapRanges, capturedLogScale, lutName, lut);
            if (!isCurrentRender()) {
              closeBitmaps();
              return;
            }
            const drawn = bitmaps ?? [];
            let painted = capturedIsRgb.some(Boolean);
            for (let k = 0; k < drawn.length; k++) {
              const bitmap = drawn[k];
              const i = indices[k];
              if (!bitmap || capturedIsRgb[i]) continue; // RGB offscreen already holds true color pixels
              const ctx = mainOffscreensRef.current[i]?.getContext("2d");
              if (!ctx) continue;
              ctx.drawImage(bitmap, 0, 0);
              painted = true;
            }
            closeBitmaps();
            if (painted && isCurrentRender()) setOffscreenVersion(version => version + 1);
          } catch (err) {
            closeBitmaps();
            if (isCurrentRender()) {
              console.error("[Show2D] WebGPU colormap repaint failed", err);
            }
          }
        })();
      });
    }
    return () => {
      cancelled = true;
      if (renderRaf !== null) window.cancelAnimationFrame(renderRaf);
    };
  }, [dataVersion, gpuCmapVersion, rangeVersion, autoContrastVersion, nImages, width, height, canvasW, canvasH, cmap, panelCmaps, panelCmapFor, logScale, autoContrast, linkedContrast, linkedContrastState, contrastStates, traitVmin, traitVmax, traitVmins, traitVmaxs, diffMode, isRgbFlags, canvasRepaintSignal, visibleImageIndices, uint8FolderPreviewMode]);

  // -------------------------------------------------------------------------
  // Maps-style detail fetch (preview binned only, _display_bin_factor > 1).
  // Request: when the user zooms past the preview's resolution, ask Python for
  // the VISIBLE window cropped from full-res and binned to ~canvas size.
  // Debounced so wheel/drag streams settle into one request per gesture.
  // -------------------------------------------------------------------------
  const currentDetailWindow = React.useCallback((panel: number) => {
    if (!displayBinFactor || displayBinFactor <= 1) return null;
    if (canvasW <= 0 || canvasH <= 0 || width <= 0 || height <= 0) return null;
    if (isRgbFlags && isRgbFlags[panel]) return null;
    if (hiddenPanelSet.has(panel)) return null;
    const view = getZoomState(panel);
    // Canvas px painted per preview px: at or below 1 the preview already
    // saturates the screen, so full-res detail adds nothing visible.
    if (view.zoom * displayScale <= 1.02) return null;
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    const row0 = Math.max(0, ((0 - cy - view.panY) / view.zoom + cy) / displayScale);
    const row1 = Math.min(height, ((canvasH - cy - view.panY) / view.zoom + cy) / displayScale);
    const col0 = Math.max(0, ((0 - cx - view.panX) / view.zoom + cx) / displayScale);
    const col1 = Math.min(width, ((canvasW - cx - view.panX) / view.zoom + cx) / displayScale);
    if (row1 <= row0 || col1 <= col0) return null;
    const visFullW = (col1 - col0) * displayBinFactor;
    const visFullH = (row1 - row0) * displayBinFactor;
    let bin = Math.max(1, Math.floor(Math.min(visFullW / canvasW, visFullH / canvasH)));
    while ((visFullW / bin) * (visFullH / bin) * 4 > DETAIL_BUDGET_BYTES) bin++;
    if (bin >= displayBinFactor) return null;
    return {
      row0, row1, col0, col1, bin,
      fullRow0: row0 * displayBinFactor,
      fullRow1: row1 * displayBinFactor,
      fullCol0: col0 * displayBinFactor,
      fullCol1: col1 * displayBinFactor,
    };
  }, [displayBinFactor, canvasW, canvasH, width, height, isRgbFlags, hiddenPanelSet, getZoomState, displayScale]);

  React.useEffect(() => {
    if (!displayBinFactor || displayBinFactor <= 1) return;
    const signature = visibleImageIndices.map((i) => {
      const detailWindow = currentDetailWindow(i);
      if (!detailWindow) return `${i}:preview`;
      return `${i}:${Math.round(detailWindow.fullRow0)},${Math.round(detailWindow.fullRow1)},${Math.round(detailWindow.fullCol0)},${Math.round(detailWindow.fullCol1)},${detailWindow.bin}`;
    }).join("|");
    if (signature === detailViewSignatureRef.current) return;
    detailViewSignatureRef.current = signature;
    // Drop any in-flight replies from the previous view before they can create
    // a sharp rectangular tile over the new preview.
    detailRequestIdRef.current++;
    detailSentKeysRef.current.clear();
  }, [displayBinFactor, visibleImageIndices, currentDetailWindow, linkedZoomState, zoomStates, dataVersion]);

  React.useEffect(() => {
    if (!displayBinFactor || displayBinFactor <= 1 || detailTilesRef.current.size === 0) return;
    let removed = false;
    detailTilesRef.current.forEach((tile, panel) => {
      if (!detailTileCoversWindow(tile, currentDetailWindow(panel))) removed = clearDetailTile(panel) || removed;
    });
    if (removed) {
      // Any in-flight reply was for a previous view; drop it and let the
      // debounced request below ask for the current window. This prevents a
      // sharp stale tile from flashing as a square over the binned preview.
      detailRequestIdRef.current++;
      setDetailStreamStatus(detailTilesRef.current.size > 0 ? "ready" : "preview");
      setDetailPaintVersion(version => version + 1);
    }
  }, [displayBinFactor, currentDetailWindow, clearDetailTile, linkedZoomState, zoomStates, dataVersion]);

  React.useEffect(() => {
    if (!displayBinFactor || displayBinFactor <= 1) return;
    if (canvasW <= 0 || canvasH <= 0 || width <= 0 || height <= 0) return;
    const timer = window.setTimeout(() => {
      const tiles: { panel: number; row0: number; row1: number; col0: number; col1: number; bin: number }[] = [];
      for (const i of visibleImageIndices) {
        const detailWindow = currentDetailWindow(i);
        if (!detailWindow) continue;
        const key = `${Math.round(detailWindow.row0)},${Math.round(detailWindow.row1)},${Math.round(detailWindow.col0)},${Math.round(detailWindow.col1)},${detailWindow.bin}`;
        if (detailSentKeysRef.current.get(i) === key) continue; // in flight or already shown
        detailSentKeysRef.current.set(i, key);
        tiles.push({ panel: i, row0: detailWindow.row0, row1: detailWindow.row1, col0: detailWindow.col0, col1: detailWindow.col1, bin: detailWindow.bin });
      }
      if (tiles.length === 0) {
        if (detailTilesRef.current.size === 0) setDetailStreamStatus("preview");
        return;
      }
      const id = ++detailRequestIdRef.current;
      setDetailStreamStatus("streaming");
      setDetailRequest(JSON.stringify({ id: String(id), tiles }));
    }, 150);
    return () => window.clearTimeout(timer);
  }, [displayBinFactor, canvasW, canvasH, width, height, visibleImageIndices,
      currentDetailWindow, dataVersion, setDetailRequest]);

  // Detail reply: decode the float32 tiles and stash per panel. Replies for
  // superseded requests (user kept zooming) are dropped by id.
  React.useEffect(() => {
    if (!detailMeta || !detailBytes || detailBytes.byteLength === 0) return;
    let meta: { id?: string; tiles?: { panel: number; row0: number; col0: number; rows: number; cols: number; bin: number; offset: number }[] };
    try { meta = JSON.parse(detailMeta); } catch { return; }
    if (String(meta.id) !== String(detailRequestIdRef.current)) return; // stale reply
    const incoming = meta.tiles ?? [];
    for (const tile of incoming) {
      const byteCount = tile.rows * tile.cols * 4;
      if (tile.offset + byteCount > detailBytes.byteLength) continue;
      // Copy out: the comm buffer's byteOffset is not guaranteed 4-aligned,
      // and Float32Array views require alignment.
      const copied = new Uint8Array(detailBytes.buffer.slice(
        detailBytes.byteOffset + tile.offset, detailBytes.byteOffset + tile.offset + byteCount));
      detailTilesRef.current.set(tile.panel, {
        row0: tile.row0, col0: tile.col0, rows: tile.rows, cols: tile.cols, bin: tile.bin,
        floats: new Float32Array(copied.buffer), canvas: null,
      });
    }
    if (incoming.length > 0) {
      setDetailStreamStatus("ready");
      setDetailVersion(version => version + 1);
    }
  }, [detailMeta, detailBytes]);

  // Colormap detail tiles through the SAME display window as the preview
  // (panelRangesRef, set by the data effect above and the slider fast path).
  // CPU renderToOffscreen is fine: tiles are ≤ 2 M pixels.
  React.useEffect(() => {
    if (detailTilesRef.current.size === 0) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    let painted = false;
    detailTilesRef.current.forEach((tile, i) => {
      const range = panelRangesRef.current[i];
      if (!range) return;
      const processed = logScale ? applyLogScale(tile.floats) : tile.floats;
      tile.canvas = renderToOffscreen(processed, tile.cols, tile.rows, lut, range.vmin, range.vmax);
      painted = true;
    });
    if (painted) setDetailPaintVersion(version => version + 1);
  }, [detailVersion, offscreenVersion, cmap, logScale]);

  // -------------------------------------------------------------------------
  // Draw: zoom/pan changes are cheap, just drawImage from cached offscreens.
  // paintMainPanels is also called directly by the live contrast preview, so a
  // histogram drag repaints its panels without re-rendering the widget tree.
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  // -------------------------------------------------------------------------
  const panelPlacement = (panel: number): PanelPlacement => {
    const view = getZoomState(panel);
    return {
      canvasW, canvasH, zoom: view.zoom, panX: view.panX, panY: view.panY,
      // Live notebook sessions still apply display rotations on the Python
      // side so static PNG/state exports see the same orientation. Kernel-less
      // standalone HTML cannot do that round trip, so it needs the canvas
      // transform here. Keep this split explicit to avoid double-rotating live
      // widgets after the backend sends rotated frame bytes.
      rotationTurns: offlineForTheme ? rotationForPanel(panel) : 0,
      flipX: Boolean(imageFlipsHorizontal?.[panel]),
      flipY: Boolean(imageFlipsVertical?.[panel]),
    };
  };
  const paintMainPanels = (paintIndices: number[]) => {
    for (const i of paintIndices) {
      const canvas = canvasRefs.current[i];
      const offscreen = mainOffscreensRef.current[i];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      ctx.imageSmoothingEnabled = smooth;
      if (smooth) ctx.imageSmoothingQuality = "high";
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      const { drawW, drawH } = applyPanelTransform(ctx, panelPlacement(i));
      ctx.drawImage(offscreen, 0, 0, offscreen.width, offscreen.height, 0, 0, drawW, drawH);
      // Detail tile on top of the preview (same zoom/pan transform): tile
      // coordinates are full-res pixels, so divide by the preview bin factor
      // to land in the preview's coordinate space. Outside the tile the
      // preview remains visible as the pan-away fallback.
      const tile = displayBinFactor > 1 ? detailTilesRef.current.get(i) : undefined;
      if (tile && tile.canvas) {
        if (detailTileCoversWindow(tile, currentDetailWindow(i))) {
          const scaleX = drawW / width;
          const scaleY = drawH / height;
          const binFactor = displayBinFactor;
          ctx.drawImage(tile.canvas, 0, 0, tile.cols, tile.rows,
            (tile.col0 / binFactor) * scaleX, (tile.row0 / binFactor) * scaleY,
            (tile.cols * tile.bin / binFactor) * scaleX, (tile.rows * tile.bin / binFactor) * scaleY);
        }
      }
      ctx.restore();
    }
  };
  React.useLayoutEffect(() => {
    if (mainOffscreensRef.current.length === 0) return;
    const activePanel = activeViewInteractionPanelRef.current;
    paintMainPanels(isGallery && !linkedZoom && !linkPan && activePanel !== null ? [activePanel] : viewportPaintImageIndices);
  }, [offscreenVersion, detailPaintVersion, displayBinFactor, nImages, width, height, displayScale, canvasW, canvasH, canvasReady, isGallery, linkedZoom, linkPan, linkedZoomState, zoomStates, smooth, currentDetailWindow, canvasRepaintSignal, imageFlipsHorizontal, imageFlipsVertical, offlineForTheme, rotationForPanel, viewportPaintImageIndices, settledViewPaintVersion]);
  // The live contrast preview: the offscreens of `panels` hold the dragged
  // window, so redraw just those canvases. A detail tile is recolored by its
  // own effect on offscreenVersion, so a binned preview still goes through it.
  const paintPreviewPanels = (panels: number[]) => {
    if (displayBinFactor > 1) setOffscreenVersion(version => version + 1);
    else paintMainPanels(panels);
  };
  const paintPreviewPanelsRef = React.useRef(paintPreviewPanels);
  paintPreviewPanelsRef.current = paintPreviewPanels;
  // A contrast tick recolors only the source pixels the canvases show
  // (sampledSourceIndex), not every pixel of the panels: the same canvas
  // pixels from 0.36 M lookups instead of 16.7 M at 4096^2, and no 64 MB
  // bitmap per tick on WebGPU. The release repaints the offscreens at full
  // resolution on the engine. Smooth display filters every source pixel, and
  // binned previews and packed uint8 folder frames are drawn through other
  // buffers, so those keep the full repaint.
  const sampledPreviewTargetsRef = React.useRef(new Map<number, { key: string; index: Int32Array; image: ImageData }>());
  const paintSampledPreview = (
    panels: number[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean,
    cmapNames: string[],
  ): boolean => {
    if (smooth || displayBinFactor > 1 || uint8FolderPreviewMode) return false;
    const targets = panels.map(panel => {
      const canvas = canvasRefs.current[panel];
      const data = rawDataRef.current?.[panel];
      if (!canvas || !data || data.length !== width * height) return null;
      const placement = panelPlacement(panel);
      const key = JSON.stringify([placement, width, height]);
      let target = sampledPreviewTargetsRef.current.get(panel);
      if (target?.key !== key) {
        const index = sampledSourceIndex(width, height, placement);
        const image = index ? canvas.getContext("2d")?.createImageData(canvasW, canvasH) : null;
        target = index && image ? { key, index, image } : undefined;
        if (target) sampledPreviewTargetsRef.current.set(panel, target);
      }
      return target ? { canvas, data, target } : null;
    });
    if (targets.some(target => !target)) return false;
    panels.forEach((panel, k) => {
      const { canvas, data, target } = targets[k]!;
      const name = cmapNames[panel];
      colormapSampledPixels(data, target.index, ranges[k], logScale, COLORMAPS[name] || COLORMAPS.inferno, target.image);
      canvas.getContext("2d")?.putImageData(target.image, 0, 0);
    });
    return true;
  };
  const paintSampledPreviewRef = React.useRef(paintSampledPreview);
  paintSampledPreviewRef.current = paintSampledPreview;

  // -------------------------------------------------------------------------
  // Render overlays: scale bar, colorbar, inset plots, panel overlays, ROIs,
  // line profile and distance measurement
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const activePanel = activeViewInteractionPanelRef.current;
    const paintIndices = (
      isGallery && !linkedZoom && !linkPan && activePanel !== null
    ) ? [activePanel] : viewportPaintImageIndices;
    for (const i of paintIndices) {
      const overlay = overlayRefs.current[i];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;

      ctx.clearRect(0, 0, overlay.width, overlay.height);
      if (panelHasScaleBar(i)) {
        const view = getZoomState(i);
        const panelPixelSize = pixelSizeForPanel(i);
        const unit = panelPixelSize > 0 ? pixelUnit : "px";
        const barPixelSize = panelPixelSize > 0 ? panelPixelSize : 1;
        const region = { x: 0, y: 0, width: overlay.width / DPR, height: overlay.height / DPR };
        const effectiveZoom = view.zoom * (region.width / width);
        const scaleBarOptions = show2dScaleBarOptions(scaleBarPosition, scaleBarLength, scaleBarLabel, scaleBarStyle, Boolean(showZoomIndicator));
        if (scaleBarGeometry(region, effectiveZoom, barPixelSize, unit, scaleBarOptions)) {
          ctx.save();
          ctx.scale(DPR, DPR);
          drawScaleBarInRegion(ctx, region, view.zoom, effectiveZoom, barPixelSize, unit, scaleBarOptions);
          ctx.restore();
        }
      }

      // Colorbar (single image mode only) uses the cached vmin/vmax from the data effect
      if (showColorbar && !isGallery) {
        const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
        const cssW = overlay.width / DPR;
        const cssH = overlay.height / DPR;
        const vmin = colorbarVminRef.current;
        const vmax = colorbarVmaxRef.current;

        ctx.save();
        ctx.scale(DPR, DPR);
        drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, logScale);
        ctx.restore();
      }

      if (showInsetPlots !== false) {
        ctx.save();
        ctx.scale(DPR, DPR);
        drawInsetPlot(
          ctx,
          insetPlotSpecFor(i),
          i,
          overlay.width / DPR,
          overlay.height / DPR,
          panelMarkerColor(i),
          panelHasScaleBar(i),
        );
        ctx.restore();
      }

      const panelOverlaySpecs = panelOverlays?.[i] || [];
      if (panelOverlaySpecs.length > 0) {
        const view = getZoomState(i);
        const { zoom, panX, panY } = view;
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        const toScreenX = (col: number) => (col * displayScale - cx) * zoom + cx + panX;
        const toScreenY = (row: number) => (row * displayScale - cy) * zoom + cy + panY;
        ctx.save();
        ctx.scale(DPR, DPR);
        drawPanelOverlays(ctx, panelOverlaySpecs, toScreenX, toScreenY, width, height);
        if (overlaySelection?.panel === i) {
          drawPanelOverlaySelection(ctx, panelOverlaySpecs[overlaySelection.overlay], toScreenX, toScreenY, width, height);
        }
        ctx.restore();
      }

      // ROI overlay: every ROI plus the highlight mask
      if (roiActive && roiList && roiList.length > 0) {
        const view = getZoomState(i);
        const { zoom, panX, panY } = view;
        const cx = canvasW / 2;
        const cy = canvasH / 2;

        // Highlight mask: dim everything outside highlighted ROIs
        const highlightedRois = roiList.filter(roi => roi.highlight);
        if (highlightedRois.length > 0) {
          ctx.save();
          ctx.scale(DPR, DPR);
          ctx.fillStyle = "rgba(0,0,0,0.6)";
          ctx.fillRect(0, 0, canvasW, canvasH);
          ctx.globalCompositeOperation = "destination-out";
          for (const roi of highlightedRois) {
            const screenX = (roi.col * displayScale - cx) * zoom + cx + panX;
            const screenY = (roi.row * displayScale - cy) * zoom + cy + panY;
            const screenRadius = roi.radius * displayScale * zoom;
            const shape = roi.shape || "circle";
            ctx.fillStyle = "rgba(0,0,0,1)";
            if (shape === "circle") {
              ctx.beginPath(); ctx.arc(screenX, screenY, screenRadius, 0, Math.PI * 2); ctx.fill();
            } else if (shape === "square") {
              ctx.fillRect(screenX - screenRadius, screenY - screenRadius, screenRadius * 2, screenRadius * 2);
            } else if (shape === "rectangle") {
              const screenW = roi.width * displayScale * zoom;
              const screenH = roi.height * displayScale * zoom;
              ctx.fillRect(screenX - screenW / 2, screenY - screenH / 2, screenW, screenH);
            } else if (shape === "annular") {
              ctx.beginPath(); ctx.arc(screenX, screenY, screenRadius, 0, Math.PI * 2); ctx.fill();
              // Re-darken inner ring
              ctx.globalCompositeOperation = "source-over";
              ctx.fillStyle = "rgba(0,0,0,0.6)";
              const screenRadiusInner = roi.radius_inner * displayScale * zoom;
              ctx.beginPath(); ctx.arc(screenX, screenY, screenRadiusInner, 0, Math.PI * 2); ctx.fill();
              ctx.globalCompositeOperation = "destination-out";
            }
          }
          ctx.restore();
        }

        ctx.save();
        ctx.scale(DPR, DPR);
        for (let roiIndex = 0; roiIndex < roiList.length; roiIndex++) {
          const roi = roiList[roiIndex];
          const isSelected = roiIndex === roiSelectedIdx;
          const screenX = (roi.col * displayScale - cx) * zoom + cx + panX;
          const screenY = (roi.row * displayScale - cy) * zoom + cy + panY;
          const screenRadius = roi.radius * displayScale * zoom;
          const screenW = roi.width * displayScale * zoom;
          const screenH = roi.height * displayScale * zoom;
          const screenRadiusInner = roi.radius_inner * displayScale * zoom;
          const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
          ctx.lineWidth = roi.line_width || 2;
          drawROI(ctx, screenX, screenY, shape, screenRadius, screenW, screenH, roi.color || ROI_COLORS[roiIndex % ROI_COLORS.length], roi.color || ROI_COLORS[roiIndex % ROI_COLORS.length], isSelected && isDraggingROI, screenRadiusInner);
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

      // Line profile overlay
      if (profileActive && profilePoints.length > 0) {
        const view = getZoomState(i);
        const { zoom, panX, panY } = view;
        ctx.save();
        ctx.scale(DPR, DPR);

        const cx = canvasW / 2;
        const cy = canvasH / 2;
        const toScreenX = (col: number) => (col * displayScale - cx) * zoom + cx + panX;
        const toScreenY = (row: number) => (row * displayScale - cy) * zoom + cy + panY;

        const ax = toScreenX(profilePoints[0].col);
        const ay = toScreenY(profilePoints[0].row);
        ctx.fillStyle = themeColors.accent;
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();

        if (profilePoints.length === 2) {
          const bx = toScreenX(profilePoints[1].col);
          const by = toScreenY(profilePoints[1].row);

          ctx.strokeStyle = themeColors.accent;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = themeColors.accent;
          ctx.beginPath();
          ctx.arc(bx, by, 4, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.restore();
      }

      // Distance measurement overlay
      if (measureActive && measurePoints.length >= 1) {
        const view = getZoomState(i);
        const { zoom, panX, panY } = view;
        ctx.save();
        ctx.scale(DPR, DPR);
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        const toScreenX = (col: number) => (col * displayScale - cx) * zoom + cx + panX;
        const toScreenY = (row: number) => (row * displayScale - cy) * zoom + cy + panY;

        ctx.shadowColor = "rgba(0,0,0,0.6)";
        ctx.shadowBlur = 3;

        const ax = toScreenX(measurePoints[0].col);
        const ay = toScreenY(measurePoints[0].row);
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();

        if (measurePoints.length === 2) {
          const bx = toScreenX(measurePoints[1].col);
          const by = toScreenY(measurePoints[1].row);

          // Solid white line (distinct from profile's dashed accent line)
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(bx, by, 4, 0, Math.PI * 2);
          ctx.fill();

          const deltaCol = measurePoints[1].col - measurePoints[0].col;
          const deltaRow = measurePoints[1].row - measurePoints[0].row;
          const distPx = Math.sqrt(deltaCol * deltaCol + deltaRow * deltaRow);
          const measurePixelSize = pixelSizeForPanel(selectedIdx);
          let label: string;
          if (measurePixelSize > 0) {
            label = formatLength(distPx * measurePixelSize, pixelUnit);
          } else {
            label = `${distPx.toFixed(1)} px`;
          }

          const midX = (ax + bx) / 2;
          const midY = (ay + by) / 2;
          ctx.font = `bold 13px ${SYSTEM_FONT}`;
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillStyle = "#fff";
          ctx.fillText(label, midX, midY - 8);
        }

        ctx.shadowBlur = 0;
        ctx.restore();
      }
    }
  }, [nImages, pixelSizeForPanel, pixelUnit, panelHasScaleBar, scaleBarPosition, scaleBarLength, scaleBarLabel, scaleBarStyle, showZoomIndicator, selectedIdx, isGallery, canvasW, canvasH, width, height, displayScale, linkedZoom, linkPan, linkedZoomState, zoomStates, dataVersion, showColorbar, cmap, offscreenVersion, logScale, profileActive, profilePoints, roiActive, roiList, roiSelectedIdx, isDraggingROI, themeColors, measureActive, measurePoints, canvasRepaintSignal, insetPlots, insetPlotSpecFor, insetDragVersion, showInsetPlots, panelMarkerColor, viewportPaintImageIndices, panelOverlays, overlaySelection, settledViewPaintVersion]);

  // -------------------------------------------------------------------------
  // Inset magnifier (lens): magnified region at the cursor, bottom-left unless dragged
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const lensCanvas = lensCanvasRef.current;
    if (lensCanvas) {
      const lensCtx = lensCanvas.getContext("2d");
      if (lensCtx) lensCtx.clearRect(0, 0, lensCanvas.width, lensCanvas.height);
    }
    if (!showLens || isGallery || !lensPos || !rawDataRef.current?.[0]) return;
    if (!lensCanvas) return;
    const ctx = lensCanvas.getContext("2d");
    if (!ctx) return;

    const raw = rawDataRef.current[0];
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    // Use cached vmin/vmax from data effect (avoids full-image applyLogScale + findDataRange)
    const vmin = colorbarVminRef.current;
    const vmax = colorbarVmaxRef.current;

    // Region around the cursor: regionSize = displaySize / magnification
    const regionSize = Math.max(4, Math.round(lensDisplaySize / lensMag));
    const lensSize = lensDisplaySize;
    const half = Math.floor(regionSize / 2);
    const startRow = lensPos.row - half;
    const startCol = lensPos.col - half;

    const regionCanvas = document.createElement("canvas");
    regionCanvas.width = regionSize;
    regionCanvas.height = regionSize;
    const regionCtx = regionCanvas.getContext("2d");
    if (!regionCtx) return;
    const imgData = regionCtx.createImageData(regionSize, regionSize);
    const range = vmax - vmin || 1;
    for (let rowOffset = 0; rowOffset < regionSize; rowOffset++) {
      for (let colOffset = 0; colOffset < regionSize; colOffset++) {
        const sourceRow = startRow + rowOffset;
        const sourceCol = startCol + colOffset;
        const pixelOffset = (rowOffset * regionSize + colOffset) * 4;
        if (sourceRow < 0 || sourceRow >= height || sourceCol < 0 || sourceCol >= width) {
          imgData.data[pixelOffset] = 0; imgData.data[pixelOffset + 1] = 0; imgData.data[pixelOffset + 2] = 0; imgData.data[pixelOffset + 3] = 255;
        } else {
          // Apply log scale inline per-pixel (only for the small region, not full image)
          const rawVal = raw[sourceRow * width + sourceCol];
          const val = logScale ? Math.log1p(rawVal) : rawVal;
          const normalized = Math.max(0, Math.min(1, (val - vmin) / range));
          const lutIndex = Math.round(normalized * 255);
          imgData.data[pixelOffset] = lut[lutIndex * 3]; imgData.data[pixelOffset + 1] = lut[lutIndex * 3 + 1]; imgData.data[pixelOffset + 2] = lut[lutIndex * 3 + 2]; imgData.data[pixelOffset + 3] = 255;
        }
      }
    }
    regionCtx.putImageData(imgData, 0, 0);

    // The lens inset sits at the dragged anchor, or bottom-left by default
    ctx.save();
    ctx.scale(DPR, DPR);
    const { x: lensX, y: lensY } = lensInsetOrigin(lensAnchor, canvasH, lensSize);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(regionCanvas, lensX, lensY, lensSize, lensSize);
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(lensX, lensY, lensSize, lensSize);
    const centerX = lensX + lensSize / 2;
    const centerY = lensY + lensSize / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX - 8, centerY); ctx.lineTo(centerX + 8, centerY);
    ctx.moveTo(centerX, centerY - 8); ctx.lineTo(centerX, centerY + 8);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "10px monospace";
    ctx.fillText(`${lensMag}×`, lensX + 4, lensY + lensSize - 4);
    ctx.restore();
  }, [showLens, lensPos, isGallery, cmap, logScale, offscreenVersion, width, height, canvasH, themeColors, lensMag, lensDisplaySize, lensAnchor, canvasRepaintSignal]);

  // -------------------------------------------------------------------------
  // Auto-compute profile when profile_line is set (e.g. from Python)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    let cancelled = false;
    if (profilePoints.length === 2 && rawDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      const frames = rawDataRef.current;
      void Promise.all(frames.map((raw, index) => (
        hiddenPanelSet.has(index) || !raw
          ? Promise.resolve(null)
          : sampleLineProfileBrowser(raw, width, height, p0.row, p0.col, p1.row, p1.col)
      ))).then((allProfiles) => {
        if (cancelled) return;
        setProfileDataAll(allProfiles);
        if (!profileActive) setProfileActive(true);
      }).catch((error) => {
        if (!cancelled) console.error("[Show2D] WebGPU line profile failed", error);
      });
    }
    return () => { cancelled = true; };
  }, [profilePoints, dataVersion, profileActive, hiddenPanelSet]);

  // -------------------------------------------------------------------------
  // Render sparkline for line profile
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = profileCanvasWidth;
    const cssH = profileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, cssW, cssH);

    const hasData = profileDataAll.some(profile => profile && profile.length >= 2);
    if (!hasData) {
      ctx.font = `10px ${SYSTEM_FONT}`;
      ctx.fillStyle = isDark ? "#555" : "#999";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Click two points on the image to draw a profile", cssW / 2, cssH / 2);
      return;
    }

    const padLeft = 40;
    const padRight = 8;
    const padTop = 6;
    const padBottom = 18;
    const plotW = cssW - padLeft - padRight;
    const plotH = cssH - padTop - padBottom;

    let globalMin = Infinity, globalMax = -Infinity;
    for (const profile of profileDataAll) {
      if (!profile) continue;
      for (let i = 0; i < profile.length; i++) {
        if (profile[i] < globalMin) globalMin = profile[i];
        if (profile[i] > globalMax) globalMax = profile[i];
      }
    }
    const range = globalMax - globalMin || 1;

    const colors = profileDataAll.length === 1 ? [themeColors.accent] : PROFILE_COLORS;
    for (let panel = 0; panel < profileDataAll.length; panel++) {
      const profile = profileDataAll[panel];
      if (!profile || profile.length < 2) continue;
      ctx.strokeStyle = colors[panel % colors.length];
      ctx.lineWidth = panel === selectedIdx || profileDataAll.length === 1 ? 1.5 : 1;
      ctx.globalAlpha = panel === selectedIdx || profileDataAll.length === 1 ? 1 : 0.5;
      ctx.beginPath();
      for (let i = 0; i < profile.length; i++) {
        const x = padLeft + (i / (profile.length - 1)) * plotW;
        const y = padTop + plotH - ((profile[i] - globalMin) / range) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Compute total distance for x-axis
    const firstProfile = profileDataAll.find(profile => profile);
    let totalDist = (firstProfile?.length ?? 2) - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const deltaCol = profilePoints[1].col - profilePoints[0].col;
      const deltaRow = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(deltaCol * deltaCol + deltaRow * deltaRow);
      if (pixelSize > 0) {
        totalDist = distPx * pixelSize;
        xUnit = pixelUnit;
      } else {
        totalDist = distPx;
      }
    }

    // Draw x-axis ticks
    const tickY = padTop + plotH;
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    const idealTicks = Math.max(2, Math.floor(plotW / 70));
    const tickStep = roundToNiceValue(totalDist / idealTicks);
    ctx.font = `9px ${SYSTEM_FONT}`;
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textBaseline = "top";
    const ticks: number[] = [];
    for (let tickValue = 0; tickValue <= totalDist + tickStep * 0.01; tickValue += tickStep) {
      if (tickValue > totalDist * 1.001) break;
      ticks.push(tickValue);
    }
    for (let i = 0; i < ticks.length; i++) {
      const tickValue = ticks[i];
      const frac = totalDist > 0 ? tickValue / totalDist : 0;
      const x = padLeft + frac * plotW;
      ctx.beginPath(); ctx.moveTo(x, tickY); ctx.lineTo(x, tickY + 3); ctx.stroke();
      ctx.textAlign = frac < 0.05 ? "left" : frac > 0.95 ? "right" : "center";
      const tickLabel = tickValue % 1 === 0 ? tickValue.toFixed(0) : tickValue.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${tickLabel} ${xUnit}` : tickLabel, x, tickY + 4);
    }

    // Draw y-axis min/max labels
    ctx.font = `9px ${SYSTEM_FONT}`;
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(globalMax), padLeft - 3, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(globalMin), padLeft - 3, padTop + plotH);

    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Legend (gallery mode with multiple images)
    if (profileDataAll.length > 1) {
      ctx.textAlign = "right";
      ctx.textBaseline = "top";
      ctx.font = `9px ${SYSTEM_FONT}`;
      let legendX = cssW - 4;
      for (let panel = profileDataAll.length - 1; panel >= 0; panel--) {
        if (!profileDataAll[panel]) continue;
        const label = labels?.[panel] || `#${panel + 1}`;
        const color = colors[panel % colors.length];
        const textW = ctx.measureText(label).width;
        ctx.globalAlpha = panel === selectedIdx ? 1 : 0.5;
        ctx.fillStyle = color;
        ctx.fillRect(legendX - textW - 10, 2, 6, 6);
        ctx.fillStyle = isDark ? "#aaa" : "#555";
        ctx.fillText(label, legendX, 1);
        legendX -= textW + 16;
      }
      ctx.globalAlpha = 1;
    }

    // Save base rendering + layout for hover overlay
    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, globalMin, globalMax, totalDist, xUnit };
  }, [profileDataAll, themeInfo.theme, themeColors.accent, profilePoints, pixelSize, selectedIdx, labels, profileCanvasWidth, profileHeight, canvasRepaintSignal]);

  // Profile hover: crosshair plus value readout over the saved base image
  const handleProfileMouseMove = React.useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    const layout = profileLayoutRef.current;
    if (!canvas || !base || !layout) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, globalMin, globalMax, totalDist, xUnit } = layout;
    const range = globalMax - globalMin || 1;

    ctx.putImageData(base, 0, 0);

    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Vertical crosshair
    ctx.strokeStyle = themeInfo.theme === "dark" ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on each profile line + collect values
    const colors = profileDataAll.length === 1 ? [themeColors.accent] : PROFILE_COLORS;
    const activeIdx = isGallery ? selectedIdx : 0;
    let displayVal: number | null = null;
    for (let panel = 0; panel < profileDataAll.length; panel++) {
      const profile = profileDataAll[panel];
      if (!profile || profile.length < 2) continue;
      const dataIdx = Math.min(profile.length - 1, Math.max(0, Math.round(frac * (profile.length - 1))));
      const value = profile[dataIdx];
      const y = padTop + plotH - ((value - globalMin) / range) * plotH;
      ctx.fillStyle = colors[panel % colors.length];
      ctx.globalAlpha = panel === activeIdx || profileDataAll.length === 1 ? 1 : 0.5;
      ctx.beginPath();
      ctx.arc(cssX, y, 3, 0, Math.PI * 2);
      ctx.fill();
      if (panel === activeIdx || profileDataAll.length === 1) displayVal = value;
    }
    ctx.globalAlpha = 1;

    // Value readout label
    if (displayVal !== null) {
      const dist = frac * totalDist;
      const label = `${formatNumber(displayVal)}  @  ${dist.toFixed(1)} ${xUnit}`;
      const isDark = themeInfo.theme === "dark";
      ctx.font = `bold 9px ${SYSTEM_FONT}`;
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
  }, [profileDataAll, themeInfo.theme, themeColors.accent, isGallery, selectedIdx]);

  const handleProfileMouseLeave = React.useCallback(() => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  }, []);

  // -------------------------------------------------------------------------
  // Compute FFT magnitude (cached: only recomputes when data changes)
  // Supports ROI-scoped FFT: when ROI is active with a selected ROI, compute
  // FFT of the cropped region instead of the full image.
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!effectiveShowFft || isGallery || !rawDataRef.current) return;
    if (!rawDataRef.current[selectedIdx]) return;
    // Generation counter: coalesces rapid ROI drag events so at most one
    // FFT runs per animation frame. The rAF yield lets the browser paint
    // the ROI position update before the (potentially blocking) FFT runs.
    const generation = ++fftGenRef.current;

    const doCompute = async () => {
      // Yield to the next animation frame: the browser paints the updated ROI first,
      // and stale requests (from earlier drag events) are discarded below.
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      if (generation !== fftGenRef.current) return;

      const gpu = gpuFFTRef.current || await getDisplayFFT();
      gpuFFTRef.current = gpu;
      if (generation !== fftGenRef.current) return;
      setFftComputing(true);
      setFftProgress(`Computing FFT… (${gpu.path})`);
      const data = rawDataRef.current![selectedIdx];
      let fftW = width;
      let fftH = height;
      let inputData = data;

      // ROI crop: extract bounding box and optionally zero-mask outside radius
      let origCropW = 0, origCropH = 0;
      if (roiFftActive && roiList && roiSelectedIdx >= 0 && roiSelectedIdx < roiList.length) {
        const roi = roiList[roiSelectedIdx];
        const crop = await cropMaskedRegionBrowser(data, width, height, roi);
        if (crop) {
          origCropW = crop.cropW;
          origCropH = crop.cropH;
          // Apply Hann window to crop at native dimensions BEFORE zero-padding
          if (fftWindow) applyHannWindow2D(crop.cropped, crop.cropW, crop.cropH);
          // Pad to next power-of-2 so fft2d doesn't truncate frequency data
          const padW = nextPow2(crop.cropW);
          const padH = nextPow2(crop.cropH);
          inputData = zeroPadTopLeft(crop.cropped, crop.cropW, crop.cropH, padW, padH);
          fftW = padW;
          fftH = padH;
        }
      }

      // Pre-pad non-power-of-2 full images so fft2d doesn't truncate frequency data
      if (origCropW === 0) {
        // Window at native dimensions before padding. Previously the ordinary
        // full-image path ignored fft_window entirely (ROI alone honored it).
        if (fftWindow) {
          inputData = data.slice();
          applyHannWindow2D(inputData, width, height);
        }
        const padW = nextPow2(fftW);
        const padH = nextPow2(fftH);
        if (padW !== fftW || padH !== fftH) {
          inputData = zeroPadTopLeft(inputData, fftW, fftH, padW, padH);
          fftW = padW;
          fftH = padH;
        }
      }

      const real = inputData.slice();
      const imag = new Float32Array(inputData.length);

      const result = await gpu.fft2D(real, imag, fftW, fftH, false);
      if (generation !== fftGenRef.current) return;
      fftshift(result.real, fftW, fftH);
      fftshift(result.imag, fftW, fftH);
      fftMagCacheRef.current = computeMagnitude(result.real, result.imag);
      // Track FFT dimensions when they differ from image dimensions (ROI crop or non-pow2 padding)
      if (origCropW > 0) {
        setFftCropDims({ cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH });
      } else if (fftW !== width || fftH !== height) {
        setFftCropDims({ cropWidth: width, cropHeight: height, fftWidth: fftW, fftHeight: fftH });
      } else {
        setFftCropDims(null);
      }
      setFftMagVersion(version => version + 1);
    };

    void doCompute().catch(err => {
      if (generation === fftGenRef.current) console.warn("[Show2D] FFT failed", err);
    }).finally(() => {
      if (generation === fftGenRef.current) {
        setFftComputing(false);
        setFftProgress("");
      }
    });
  }, [effectiveShowFft, isGallery, selectedIdx, width, height, dataVersion, roiFftKey, fftWindow]);

  // Clear FFT measurement when image, FFT state, or ROI changes
  React.useEffect(() => { setFftClickInfo(null); }, [selectedIdx, effectiveShowFft, roiFftActive, roiSelectedIdx]);

  // -------------------------------------------------------------------------
  // FFT data effect: normalize + colormap → cached offscreen canvas
  // (does NOT depend on fftZoom/fftPanX/fftPanY, so zoom/pan never reprocesses)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!effectiveShowFft || isGallery || !fftMagCacheRef.current) return;

    const fftMag = fftMagCacheRef.current;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;

    // Use crop dimensions when ROI FFT is active
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    if (fftMetricsEnabled) {
      const qualityKey = `${fftMagVersion}:${fftW}x${fftH}:${pixelSize || 0}:${pixelUnit || ""}`;
      if (fftQualityKeyRef.current !== qualityKey) {
        fftQualityKeyRef.current = qualityKey;
        setFftQuality(computeFftQualityMetrics(fftMag, fftW, fftH, { sampling: pixelSize, unit: pixelUnit }));
      }
    } else if (fftQualityKeyRef.current) {
      fftQualityKeyRef.current = "";
      setFftQuality(null);
    }

    // Heavy steps (log/power transform, range, stats, histogram-data copy) only
    // when source magnitude OR scale-mode changed, NOT on every contrast slider tick.
    // Cached values live in fftPipelineRef for cheap re-renders.
    const sourceChanged = (
      fftPipelineRef.current?.magVersion !== fftMagVersion ||
      fftPipelineRef.current?.scaleMode !== fftScaleMode ||
      fftPipelineRef.current?.fftAuto !== fftAuto
    );
    if (sourceChanged) {
      const magnitude = new Float32Array(fftMag.length);
      for (let i = 0; i < fftMag.length; i++) {
        if (fftScaleMode === "log") magnitude[i] = Math.log1p(fftMag[i]);
        else if (fftScaleMode === "power") magnitude[i] = Math.pow(fftMag[i], 0.5);
        else magnitude[i] = fftMag[i];
      }
      let displayMin: number, displayMax: number;
      if (fftAuto) ({ min: displayMin, max: displayMax } = autoEnhanceFFT(magnitude, fftW, fftH));
      else ({ min: displayMin, max: displayMax } = findDataRange(magnitude));
      const { mean, std } = computeStats(magnitude);
      setFftStats([mean, displayMin, displayMax, std]);
      setFftHistogramData(magnitude);  // no .slice(): magnitude is fresh
      setFftDataRange({ min: displayMin, max: displayMax });
      fftPipelineRef.current = { magnitude, displayMin, displayMax, magVersion: fftMagVersion, scaleMode: fftScaleMode, fftAuto };
    }

    const cache = fftPipelineRef.current!;
    const { vmin, vmax } = sliderRange(cache.displayMin, cache.displayMax, fftVminPct, fftVmaxPct);

    // GPU colormap path for FFT: a dedicated slot at index nImages.
    // Uploads magnitude only when source changed; contrast/cmap drag triggers cheap re-render.
    const engine = gpuCmapRef.current;
    const fftSlot = nImages;  // dedicate slot just past main image slots
    const colorGeneration = ++singleFftColorGenRef.current;
    let cancelled = false;
    if (engine && gpuCmapReadyRef.current) {
      try {
        if (sourceChanged) engine.uploadData(fftSlot, cache.magnitude, fftW, fftH);
        void engine.renderSlotsToImageBitmapAsync([fftSlot], [{ vmin, vmax }], false, fftColormap, lut).then(bitmaps => {
          if (!bitmaps || !bitmaps[0]) {
            console.error("[Show2D] WebGPU FFT colormap returned no bitmap");
            return;
          }
          try {
            if (cancelled || colorGeneration !== singleFftColorGenRef.current) return;
            const offscreen = fftOffscreenRef.current && fftOffscreenRef.current.width === fftW && fftOffscreenRef.current.height === fftH
              ? fftOffscreenRef.current
              : Object.assign(document.createElement("canvas"), { width: fftW, height: fftH });
            const ctx = offscreen.getContext("2d");
            if (ctx) {
              ctx.drawImage(bitmaps[0], 0, 0);
              fftOffscreenRef.current = offscreen;
              setFftOffscreenVersion(version => version + 1);
            }
          } finally {
            bitmaps.forEach(bitmap => bitmap?.close());
          }
        }).catch(err => {
          if (!cancelled) console.error("[Show2D] FFT colormap GPU render failed", err);
        });
        return () => { cancelled = true; };
      } catch (err) {
        console.error("[Show2D] FFT colormap GPU setup failed", err);
      }
    }
    return () => { cancelled = true; };
  }, [effectiveShowFft, isGallery, fftMagVersion, fftVminPct, fftVmaxPct, fftColormap, fftScaleMode, fftAuto, width, height, fftCropDims, nImages, pixelSize, pixelUnit, fftMetricsEnabled, canvasRepaintSignal]);

  // -------------------------------------------------------------------------
  // FFT draw effect: cheap drawImage from cached offscreen (zoom/pan changes)
  // -------------------------------------------------------------------------
  React.useLayoutEffect(() => {
    if (!effectiveShowFft || isGallery || !fftCanvasRef.current || !fftOffscreenRef.current) return;

    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const offscreen = fftOffscreenRef.current;
    const fftW = offscreen.width;
    const fftH = offscreen.height;

    ctx.imageSmoothingEnabled = fftSmooth;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();

    const centerOffsetX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
    const centerOffsetY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;

    ctx.translate(centerOffsetX, centerOffsetY);
    ctx.scale(fftZoom, fftZoom);
    // Stretch cropped FFT to fill the full canvas (no layout change during drag)
    ctx.drawImage(offscreen, 0, 0, fftW, fftH, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [effectiveShowFft, isGallery, fftOffscreenVersion, canvasW, canvasH, fftZoom, fftPanX, fftPanY, fftSmooth, canvasRepaintSignal]);

  // -------------------------------------------------------------------------
  // Render FFT overlay (colorbar + d-spacing marker)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const overlay = fftOverlayRef.current;
    if (!overlay || !effectiveShowFft || isGallery) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Use crop dimensions for reciprocal-space calculations
    const fftW = fftCropDims?.fftWidth ?? width;

    // FFT colorbar
    if (fftShowColorbar && fftDataRange.min !== fftDataRange.max) {
      const { vmin, vmax } = sliderRange(fftDataRange.min, fftDataRange.max, fftVminPct, fftVmaxPct);
      const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      const cssW = overlay.width / DPR;
      const cssH = overlay.height / DPR;
      drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, fftScaleMode === "log");
      ctx.restore();
    }

    // D-spacing crosshair marker, mapped with the crop dims
    const fftH = fftCropDims?.fftHeight ?? height;
    if (fftClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const centerOffsetX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
      const centerOffsetY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;
      const screenX = centerOffsetX + fftZoom * (fftClickInfo.col / fftW * canvasW);
      const screenY = centerOffsetY + fftZoom * (fftClickInfo.row / fftH * canvasH);
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
      if (fftClickInfo.dSpacing != null) {
        const dSpacing = fftClickInfo.dSpacing;
        const label = `d = ${formatLength(dSpacing, pixelUnit)}`;
        ctx.font = `bold 11px ${SYSTEM_FONT}`;
        ctx.fillStyle = "white";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(label, screenX + 10, screenY - 4);
      }
      ctx.restore();
    }
  }, [effectiveShowFft, isGallery, fftClickInfo, canvasW, canvasH, fftZoom, fftPanX, fftPanY, width, height, pixelSize, fftDataRange, fftVminPct, fftVmaxPct, fftColormap, fftScaleMode, fftShowColorbar, fftCropDims, canvasRepaintSignal]);

  // -------------------------------------------------------------------------
  // Compute FFT magnitudes for gallery mode (cache raw magnitudes)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!effectiveShowFft || !isGallery || !rawDataRef.current) return;
    if (rawDataRef.current.length === 0) return;
    let cancelled = false;
    const serial = ++galleryFftComputeSerialRef.current;

    const finishCurrentCompute = () => {
      if (serial !== galleryFftComputeSerialRef.current) return;
      setFftComputing(false);
      setFftProgress("");
    };

    const computeAllFFTs = async () => {
      if (fftMagCacheGalleryRef.current.length !== nImages) {
        fftMagCacheGalleryRef.current = new Array(nImages).fill(null);
      }
      if (galleryFftActiveKeysRef.current.length !== nImages) {
        galleryFftActiveKeysRef.current = new Array(nImages).fill(null);
      }
      if (galleryFftPipelineRef.current.length !== nImages) {
        galleryFftPipelineRef.current = new Array(nImages).fill(null);
      }

      const useRoiCrop = roiFftActive && roiList && roiSelectedIdx >= 0 && roiSelectedIdx < roiList.length;
      const roi = useRoiCrop ? roiList[roiSelectedIdx] : null;
      const overviewMaxDim = isPaged
        ? PAGED_GALLERY_FFT_OVERVIEW_MAX_DIM
        : GALLERY_FFT_OVERVIEW_MAX_DIM;
      const overviewDownsample = roi
        ? 1
        : Math.max(1, Math.ceil(Math.max(width, height) / overviewMaxDim));
      const fftPanelIndices = visibleImageIndices;
      const visibleFftSet = new Set(fftPanelIndices);
      for (let idx = 0; idx < nImages; idx++) {
        if (visibleFftSet.has(idx)) continue;
        fftMagCacheGalleryRef.current[idx] = null;
        galleryFftActiveKeysRef.current[idx] = null;
        galleryFftPipelineRef.current[idx] = null;
        fftOffscreensRef.current[idx] = null;
      }
      const dataEpochs = galleryFftPanelEpochsRef.current.length === nImages
        ? [...galleryFftPanelEpochsRef.current]
        : new Array(nImages).fill(0);
      const targetKeys: (string | null)[] = new Array(nImages).fill(null);
      for (const idx of fftPanelIndices) {
        targetKeys[idx] = makeGalleryFftCacheKey({
          dataEpoch: dataEpochs[idx],
          panel: idx,
          frame: normalizedPanelFrameIndices[idx] || 0,
          width,
          height,
          roiKey: roiFftKey,
          fftWindow,
          overviewDownsample,
        });
      }
      galleryFftTargetKeysRef.current = targetKeys;

      const missingIndices: number[] = [];
      let activatedCachedResult = false;
      for (const idx of fftPanelIndices) {
        const targetKey = targetKeys[idx];
        if (!targetKey) continue;
        if (
          galleryFftActiveKeysRef.current[idx] === targetKey
          && fftMagCacheGalleryRef.current[idx]
        ) {
          continue;
        }
        const cached = readGalleryFftCache(
          galleryFftMagnitudeLruRef.current,
          targetKey,
        );
        if (!cached) {
          missingIndices.push(idx);
          continue;
        }
        fftMagCacheGalleryRef.current[idx] = cached.mag;
        galleryFftActiveKeysRef.current[idx] = targetKey;
        galleryFftPipelineRef.current[idx] = null;
        galleryFftDimsRef.current = { w: cached.fftWidth, h: cached.fftHeight };
        activatedCachedResult = true;
      }
      if (activatedCachedResult) setGalleryFftMagVersion(version => version + 1);
      if (missingIndices.length === 0) {
        finishCurrentCompute();
        return;
      }

      setFftComputing(true);
      setFftProgress("FFT");
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      if (cancelled || serial !== galleryFftComputeSerialRef.current) return;

      const gpu = gpuFFTRef.current || await getDisplayFFT();
      gpuFFTRef.current = gpu;
      if (cancelled || serial !== galleryFftComputeSerialRef.current) return;
      setFftProgress(`FFT (${gpu.path})`);

      // Helper: prep one image for FFT (crop, pad, window)
      const prepOne = async (idx: number): Promise<{ real: Float32Array; imag: Float32Array; width: number; height: number } | null> => {
        const data = rawDataRef.current![idx];
        if (!data) return null;
        let inputData = data;
        let inputW = width, inputH = height;
        if (roi) {
          const crop = await cropMaskedRegionBrowser(data, width, height, roi);
          if (crop) {
            if (fftWindow) applyHannWindow2D(crop.cropped, crop.cropW, crop.cropH);
            const padW = nextPow2(crop.cropW), padH = nextPow2(crop.cropH);
            inputData = zeroPadTopLeft(crop.cropped, crop.cropW, crop.cropH, padW, padH);
            inputW = padW; inputH = padH;
          }
        } else {
          if (overviewDownsample > 1) {
            const down = meanDownsample2D(inputData, inputW, inputH, overviewDownsample);
            inputData = down.data;
            inputW = down.width;
            inputH = down.height;
          }
          // Apply the Hann taper at the actual FFT source dimensions before
          // power-of-two padding. The old full-image gallery path never used
          // fft_window, even though ROI and diff FFTs did.
          if (fftWindow) {
            inputData = inputData.slice();
            applyHannWindow2D(inputData, inputW, inputH);
          }
          const padW = nextPow2(inputW), padH = nextPow2(inputH);
          if (padW !== inputW || padH !== inputH) {
            inputData = zeroPadTopLeft(inputData, inputW, inputH, padW, padH);
            inputW = padW; inputH = padH;
          }
        }
        return { real: inputData.slice(), imag: new Float32Array(inputData.length), width: inputW, height: inputH };
      };

      // ── Prep only cache misses ──
      const missingSet = new Set(missingIndices);
      const inputs: { real: Float32Array; imag: Float32Array }[] = [];
      let fftW = width, fftH = height;
      for (let idx = 0; idx < nImages; idx++) {
        if (!missingSet.has(idx)) {
          inputs.push({ real: new Float32Array(0), imag: new Float32Array(0) });
          continue;
        }
        const input = await prepOne(idx);
        if (input) {
          fftW = input.width; fftH = input.height;
          inputs.push({ real: input.real, imag: input.imag });
        } else {
          inputs.push({ real: new Float32Array(0), imag: new Float32Array(0) });
        }
      }
      galleryFftDimsRef.current = { w: fftW, h: fftH };
      if (cancelled || serial !== galleryFftComputeSerialRef.current) return;

      const rememberResult = (idx: number, mag: Float32Array): boolean => {
        const targetKey = targetKeys[idx];
        if (!targetKey) return false;
        const protectedKeys = new Set(
          [...galleryFftActiveKeysRef.current, targetKey]
            .filter((key): key is string => !!key),
        );
        const totalFrameCount = (panelFrameCounts || []).reduce(
          (sum, count) => sum + Math.max(1, count || 1),
          0,
        );
        rememberGalleryFftCache(
          galleryFftMagnitudeLruRef.current,
          targetKey,
          { mag, fftWidth: fftW, fftHeight: fftH },
          {
            maxEntries: Math.max(nImages, Math.min(GALLERY_FFT_CACHE_MAX_ENTRIES, totalFrameCount || nImages)),
            maxBytes: GALLERY_FFT_CACHE_MAX_BYTES,
            protectedKeys,
          },
        );
        if (
          galleryFftPanelEpochsRef.current[idx] !== dataEpochs[idx]
          || galleryFftTargetKeysRef.current[idx] !== targetKey
          || serial !== galleryFftComputeSerialRef.current
        ) {
          return false;
        }
        fftMagCacheGalleryRef.current[idx] = mag;
        galleryFftActiveKeysRef.current[idx] = targetKey;
        galleryFftPipelineRef.current[idx] = null;
        return true;
      };

      // ── Batched progressive FFT: batch BATCH_SIZE at a time, display after each batch ──
      const BATCH_SIZE = 4;
      for (let batchStart = 0; batchStart < missingIndices.length; batchStart += BATCH_SIZE) {
        if (cancelled || serial !== galleryFftComputeSerialRef.current) return;
        const batchIndices = missingIndices.slice(batchStart, batchStart + BATCH_SIZE);
        const batchInputs = batchIndices.map(idx => inputs[idx]).filter(prepared => prepared.real.length > 0);
        if (batchInputs.length === 0) continue;
        setFftProgress(`FFT ${batchStart + 1}–${Math.min(batchStart + BATCH_SIZE, missingIndices.length)}/${missingIndices.length} visible (${gpu.path})`);
        let activatedBatchResult = false;

        if (batchInputs.length > 1) {
          const batchResults = await gpu.fft2DBatch(batchInputs, fftW, fftH);
          if (cancelled || serial !== galleryFftComputeSerialRef.current) return;
          let resultIndex = 0;
          for (const idx of batchIndices) {
            if (!inputs[idx] || inputs[idx].real.length === 0) continue;
            fftshift(batchResults[resultIndex].real, fftW, fftH);
            fftshift(batchResults[resultIndex].imag, fftW, fftH);
            const mag = computeMagnitude(batchResults[resultIndex].real, batchResults[resultIndex].imag);
            activatedBatchResult = rememberResult(idx, mag) || activatedBatchResult;
            resultIndex++;
          }
        } else {
          for (const idx of batchIndices) {
            if (!inputs[idx] || inputs[idx].real.length === 0) continue;
            if (cancelled || serial !== galleryFftComputeSerialRef.current) return;
            const { real, imag } = inputs[idx];
            const result = await gpu.fft2D(real, imag, fftW, fftH, false);
            if (cancelled || serial !== galleryFftComputeSerialRef.current) return;
            fftshift(result.real, fftW, fftH);
            fftshift(result.imag, fftW, fftH);
            const mag = computeMagnitude(result.real, result.imag);
            activatedBatchResult = rememberResult(idx, mag) || activatedBatchResult;
          }
        }
        // Show this batch immediately (progressive top-to-bottom). A stale
        // completion may enter the bounded cache but never becomes active.
        if (activatedBatchResult) setGalleryFftMagVersion(version => version + 1);
        // Yield to let the browser paint the batch
        await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      }
      finishCurrentCompute();
    };

    void computeAllFFTs().catch(err => {
      if (!cancelled && serial === galleryFftComputeSerialRef.current) {
        console.warn("[Show2D] Gallery FFT failed", err);
      }
    }).finally(() => {
      finishCurrentCompute();
    });

    return () => {
      cancelled = true;
      if (serial === galleryFftComputeSerialRef.current) finishCurrentCompute();
    };
  }, [effectiveShowFft, isGallery, nImages, width, height, dataVersion, roiFftKey, fftWindow, visibleImageIndices]);

  // Gallery FFT data effect: normalize + colormap → cached offscreen canvases
  // (does NOT depend on gallery zoom/pan states)
  React.useEffect(() => {
    if (!effectiveShowFft || !isGallery) return;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const fftW = galleryFftDimsRef.current?.w ?? width;
    const fftH = galleryFftDimsRef.current?.h ?? height;
    const generation = ++galleryFftColorGenRef.current;
    let cancelled = false;

    if (galleryFftPipelineRef.current.length !== nImages) {
      galleryFftPipelineRef.current = new Array(nImages).fill(null);
    }

    const ranges: { vmin: number; vmax: number }[] = [];
    const slots: number[] = [];
    const paintKeys: string[] = [];
    const uploadSlots: number[] = [];

    for (const idx of visibleImageIndices) {
      const magnitude = fftMagCacheGalleryRef.current[idx];
      if (!magnitude) continue;
      const sourceKey = galleryFftActiveKeysRef.current[idx]
        || `legacy:${idx}:${galleryFftMagVersion}`;

      // Heavy work (log/sqrt transform + range) is cached per FFT source/config.
      // Histogram/contrast drag below only changes vmin/vmax and does not touch
      // this block, nor does it recompute FFT magnitudes.
      let cache = galleryFftPipelineRef.current[idx];
      const sourceChanged = (
        !cache ||
        cache.sourceKey !== sourceKey ||
        cache.scaleMode !== fftScaleMode ||
        cache.fftAuto !== fftAuto
      );
      if (sourceChanged) {
        let displayData: Float32Array;
        if (fftScaleMode === "log") {
          displayData = applyLogScale(magnitude);
        } else if (fftScaleMode === "power") {
          displayData = new Float32Array(magnitude.length);
          for (let j = 0; j < magnitude.length; j++) displayData[j] = Math.sqrt(magnitude[j]);
        } else {
          // Auto enhancement masks the DC display pixel. Keep that display-only
          // mutation out of the cached scientific FFT magnitude used by metrics.
          displayData = magnitude.slice();
        }
        let displayMin: number, displayMax: number;
        if (fftAuto) {
          ({ min: displayMin, max: displayMax } = autoEnhanceFFT(magnitude, fftW, fftH));
          if (fftScaleMode === "log") { displayMin = Math.log1p(displayMin); displayMax = Math.log1p(displayMax); }
          else if (fftScaleMode === "power") { displayMin = Math.sqrt(displayMin); displayMax = Math.sqrt(displayMax); }
        } else {
          ({ min: displayMin, max: displayMax } = findDataRange(displayData));
        }
        cache = {
          displayData,
          displayMin,
          displayMax,
          sourceKey,
          scaleMode: fftScaleMode,
          fftAuto,
          uploadedKey: "",
        };
        galleryFftPipelineRef.current[idx] = cache;
      }
      if (!cache) continue;
      const fftContrast = fftContrastFor(idx);
      const { vmin, vmax } = sliderRange(cache.displayMin, cache.displayMax, fftContrast.vminPct, fftContrast.vmaxPct);
      const uploadKey = `${cache.sourceKey}:${fftW}x${fftH}:${fftScaleMode}:${fftAuto}`;
      // An unlinked contrast drag changes one panel's range: the panels whose
      // pixels would come out the same keep their offscreen and are not redrawn.
      const paintKey = `${uploadKey}:${fftColormap}:${vmin}:${vmax}:${canvasRepaintSignal}`;
      if (!sourceChanged && cache.paintedKey === paintKey && fftOffscreensRef.current[idx]) continue;
      ranges.push({ vmin, vmax });
      slots.push(nImages + idx);
      paintKeys.push(paintKey);
      if (sourceChanged || cache.uploadedKey !== uploadKey) {
        uploadSlots.push(idx);
      }
    }

    // Update FFT histogram from selected image
    const selectedCache = galleryFftPipelineRef.current[selectedIdx];
    if (selectedCache) {
      setFftHistogramData(selectedCache.displayData);
      setFftDataRange({ min: selectedCache.displayMin, max: selectedCache.displayMax });
    }

    const renderGalleryFft = async () => {
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      if (cancelled || generation !== galleryFftColorGenRef.current) return;

      const engine = gpuCmapRef.current;
      if (engine && gpuCmapReadyRef.current && slots.length > 0) {
        try {
          for (const idx of uploadSlots) {
            const cache = galleryFftPipelineRef.current[idx];
            if (!cache) continue;
            engine.uploadData(nImages + idx, cache.displayData, fftW, fftH);
            cache.uploadedKey = `${cache.sourceKey}:${fftW}x${fftH}:${fftScaleMode}:${fftAuto}`;
          }
          const bitmaps = await engine.renderSlotsToImageBitmapAsync(slots, ranges, false, fftColormap, lut);
          if (cancelled || generation !== galleryFftColorGenRef.current) {
            bitmaps?.forEach(bitmap => bitmap?.close());
            return;
          }
          if (bitmaps && bitmaps.length > 0) {
            let painted = false;
            try {
              for (let k = 0; k < bitmaps.length; k++) {
                const bitmap = bitmaps[k];
                const idx = slots[k] - nImages;
                if (!bitmap) continue;
                const offscreen = fftOffscreensRef.current[idx] && fftOffscreensRef.current[idx]!.width === fftW && fftOffscreensRef.current[idx]!.height === fftH
                  ? fftOffscreensRef.current[idx]!
                  : Object.assign(document.createElement("canvas"), { width: fftW, height: fftH });
                const ctx = offscreen.getContext("2d");
                if (!ctx) continue;
                ctx.drawImage(bitmap, 0, 0);
                fftOffscreensRef.current[idx] = offscreen;
                galleryFftPaintStampRef.current.stamps.set(offscreen, ++galleryFftPaintStampRef.current.next);
                const paintedCache = galleryFftPipelineRef.current[idx];
                if (paintedCache) paintedCache.paintedKey = paintKeys[k];
                painted = true;
              }
            } finally {
              bitmaps.forEach(bitmap => bitmap?.close());
            }
            if (painted) {
              setGalleryFftOffscreenVersion(version => version + 1);
              return;
            }
          }
        } catch (err) {
          console.error("[Show2D FFT] Gallery WebGPU colormap failed", err);
        }
      }
    };

    renderGalleryFft();
    return () => { cancelled = true; };
  }, [effectiveShowFft, isGallery, nImages, width, height, galleryFftMagVersion, fftColormap, fftScaleMode, fftAuto, fftVminPct, fftVmaxPct, selectedIdx, effectiveFftLinkedContrast, fftContrastStates, canvasRepaintSignal, visibleImageIndices]);

  React.useEffect(() => {
    if (!effectiveShowFft || !isGallery || !fftMetricsEnabled) {
      setGalleryFftQuality([]);
      return;
    }
    const fftW = galleryFftDimsRef.current?.w ?? width;
    const fftH = galleryFftDimsRef.current?.h ?? height;
    const next = new Array<FftQualityMetrics | null>(nImages).fill(null);
    for (const idx of visibleImageIndices) {
      const mag = fftMagCacheGalleryRef.current[idx];
      if (!mag) continue;
      next[idx] = computeFftQualityMetrics(mag, fftW, fftH, {
        sampling: pixelSizeForPanel(idx),
        unit: pixelUnit,
      });
    }
    setGalleryFftQuality(next);
  }, [effectiveShowFft, isGallery, fftMetricsEnabled, galleryFftMagVersion, nImages, width, height, pixelSizeForPanel, pixelUnit, visibleImageIndices]);

  // Gallery FFT draw effect: cheap drawImage from cached offscreens (zoom/pan changes)
  React.useLayoutEffect(() => {
    if (!effectiveShowFft || !isGallery) return;
    const fftW = galleryFftDimsRef.current?.w ?? width;
    const fftH = galleryFftDimsRef.current?.h ?? height;

    for (const idx of visibleImageIndices) {
      const offscreen = fftOffscreensRef.current[idx];
      const canvas = fftCanvasRefs.current[idx];
      if (!offscreen || !canvas) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      const { zoom, panX, panY } = getGalleryFftState(idx);
      const stamp = galleryFftPaintStampRef.current.stamps.get(offscreen) ?? 0;
      const drawKey = `${stamp}:${zoom}:${panX}:${panY}:${canvasW}x${canvasH}:${fftW}x${fftH}:${fftSmooth}:${canvasRepaintSignal}`;
      if (galleryFftDrawnKeysRef.current.get(canvas) === drawKey) continue;
      galleryFftDrawnKeysRef.current.set(canvas, drawKey);
      ctx.imageSmoothingEnabled = fftSmooth;
      ctx.clearRect(0, 0, canvasW, canvasH);
      ctx.save();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, fftW, fftH, 0, 0, canvasW, canvasH);
      ctx.restore();
    }
  }, [effectiveShowFft, isGallery, nImages, canvasW, canvasH, width, height, galleryFftOffscreenVersion, galleryFftStates, effectiveFftLinkedZoom, effectiveFftLinkPan, linkedFftZoomState, fftSmooth, canvasRepaintSignal, visibleImageIndices]);

  // -------------------------------------------------------------------------
  // Mouse Handlers for Zoom/Pan
  // -------------------------------------------------------------------------
  const handleWheel = (e: React.WheelEvent, idx: number) => {
    // In gallery mode, only allow zoom on the selected image (unless linked)
    if (isGallery && idx !== selectedIdx && !linkedZoom) return;

    const canvas = canvasRefs.current[idx];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    
    const view = getImmediateZoomState(idx);
    
    // Mouse position relative to canvas (in canvas pixel coordinates)
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    
    // Mouse position relative to the current view (accounting for pan and zoom)
    // The transformation is: translate(cx + panX, cy + panY) -> scale(zoom) -> translate(-cx, -cy)
    // So a point on screen at (screenX, screenY) maps to image space as:
    // imageX = (screenX - cx - panX) / zoom + cx
    const mouseImageX = (mouseCanvasX - cx - view.panX) / view.zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - view.panY) / view.zoom + cy;

    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, view.zoom * zoomFactor));
    
    // Calculate new pan to keep the mouse position fixed on the same image point
    // After zoom: screenX = (imageX - cx) * newZoom + cx + newPanX
    // We want screenX to stay at mouseCanvasX, so:
    // newPanX = mouseCanvasX - (imageX - cx) * newZoom - cx
    const newPanX = mouseCanvasX - (mouseImageX - cx) * newZoom - cx;
    const newPanY = mouseCanvasY - (mouseImageY - cy) * newZoom - cy;

    const nextState = { zoom: newZoom, panX: newPanX, panY: newPanY };
    beginViewInteraction(idx);
    scheduleWheelZoomState(idx, nextState);
    persistZoomState(nextState);
  };

  const resetViewState = React.useCallback((): ZoomState => resetZoomStateRef.current || { zoom: 1, panX: 0, panY: 0 }, []);

  const resetImageView = React.useCallback((idx: number) => {
    const resetState = resetViewState();
    setZoomState(idx, resetState);
    persistZoomState(resetState);
    clickStartRef.current = null;
    setIsDraggingPan(false);
    setPanStart(null);
    setPanningIdx(null);
  }, [persistZoomState, resetViewState, setZoomState]);

  const handleDoubleClick = (e: React.MouseEvent, idx: number) => {
    e.preventDefault();
    e.stopPropagation();
    resetImageView(idx);
  };

  type TouchPoint = { clientX: number; clientY: number };
  const touchDistance = (a: TouchPoint, b: TouchPoint) => Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
  const touchMidpoint = (a: TouchPoint, b: TouchPoint) => ({ x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 });
  const touchToCanvas = (clientX: number, clientY: number, canvas: HTMLCanvasElement) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (clientX - rect.left) * (canvas.width / Math.max(1, rect.width)),
      y: (clientY - rect.top) * (canvas.height / Math.max(1, rect.height)),
    };
  };

  const handleTouchStart = (e: React.TouchEvent, idx: number) => {
    if (profileActive || roiActive || measureActive) return;
    if (isGallery && idx !== selectedIdx) setSelectedIdx(idx);
    const now = Date.now();
    if (e.touches.length === 1) {
      const lastTap = lastTapRef.current;
      if (lastTap && lastTap.idx === idx && now - lastTap.time < 320) {
        e.preventDefault();
        resetImageView(idx);
        lastTapRef.current = null;
        touchStartRef.current = null;
        return;
      }
      lastTapRef.current = { time: now, idx };
      const touch = e.touches[0];
      touchStartRef.current = {
        idx,
        mode: "pan",
        startX: touch.clientX,
        startY: touch.clientY,
        startDistance: 0,
        startMidX: touch.clientX,
        startMidY: touch.clientY,
        startState: getZoomState(idx),
      };
      e.preventDefault();
      return;
    }
    if (e.touches.length >= 2) {
      const firstTouch = e.touches[0];
      const secondTouch = e.touches[1];
      const mid = touchMidpoint(firstTouch, secondTouch);
      touchStartRef.current = {
        idx,
        mode: "pinch",
        startX: mid.x,
        startY: mid.y,
        startDistance: Math.max(1, touchDistance(firstTouch, secondTouch)),
        startMidX: mid.x,
        startMidY: mid.y,
        startState: getZoomState(idx),
      };
      e.preventDefault();
    }
  };

  const handleTouchMove = (e: React.TouchEvent, idx: number) => {
    const start = touchStartRef.current;
    if (!start || start.idx !== idx) return;
    const canvas = canvasRefs.current[idx];
    if (!canvas) return;
    e.preventDefault();
    const base = start.startState;
    if (start.mode === "pinch" && e.touches.length >= 2) {
      const firstTouch = e.touches[0];
      const secondTouch = e.touches[1];
      const mid = touchMidpoint(firstTouch, secondTouch);
      const startCanvas = touchToCanvas(start.startMidX, start.startMidY, canvas);
      const currentCanvas = touchToCanvas(mid.x, mid.y, canvas);
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      const imageX = (startCanvas.x - cx - base.panX) / base.zoom + cx;
      const imageY = (startCanvas.y - cy - base.panY) / base.zoom + cy;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base.zoom * (touchDistance(firstTouch, secondTouch) / start.startDistance)));
      const nextState = {
        zoom: newZoom,
        panX: currentCanvas.x - (imageX - cx) * newZoom - cx,
        panY: currentCanvas.y - (imageY - cy) * newZoom - cy,
      };
      setZoomState(idx, nextState);
      return;
    }
    if (start.mode === "pan" && e.touches.length === 1) {
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / Math.max(1, rect.width);
      const scaleY = canvas.height / Math.max(1, rect.height);
      setZoomState(idx, {
        ...base,
        panX: base.panX + (touch.clientX - start.startX) * scaleX,
        panY: base.panY + (touch.clientY - start.startY) * scaleY,
      });
    }
  };

  const handleTouchEnd = (e: React.TouchEvent, idx: number) => {
    const start = touchStartRef.current;
    if (!start || start.idx !== idx) return;
    if (e.touches.length > 0) return;
    persistZoomState(getZoomState(idx));
    touchStartRef.current = null;
  };

  // Reset view (zoom/pan only; preserves profile, FFT state, etc.)
  const handleResetAll = () => {
    const resetState = resetViewState();
    setZoomStates(new Map(Array.from({ length: nImages }, (_, i) => [i, resetState])));
    setLinkedZoomState(resetState);
    persistZoomState(resetState);
    setGalleryFftStates(new Map());
    setLinkedFftZoomState({ zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 });
    setFftZoom(DEFAULT_FFT_ZOOM);
    setFftPanX(0);
    setFftPanY(0);
  };

  // FFT zoom/pan: cursor-anchored zoom matching FFT's own canvas transform.
  // FFT render: translate(centerOffsetX, centerOffsetY) → scale(zoom) where
  //   centerOffsetX = (canvasW - canvasW*zoom)/2 + panX
  // Solving for image-space u in [0,1]:
  //   u = (screenX - centerOffsetX) / (zoom * canvasW)
  // After zoom change, keep screenX of mouse at u:
  //   newPanX = mouseX - (canvasW - canvasW*newZoom)/2 - newZoom*u*canvasW
  const handleFftWheel = (e: React.WheelEvent) => {
    const canvas = fftCanvasRef.current;
    if (!canvas) {
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      setFftZoom(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftZoom * zoomFactor)));
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const canvasWidth = canvas.width, canvasHeight = canvas.height;
    const centerOffsetX = (canvasWidth - canvasWidth * fftZoom) / 2 + fftPanX;
    const centerOffsetY = (canvasHeight - canvasHeight * fftZoom) / 2 + fftPanY;
    const u = (mouseX - centerOffsetX) / (fftZoom * canvasWidth);
    const v = (mouseY - centerOffsetY) / (fftZoom * canvasHeight);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftZoom * zoomFactor));
    const newPanX = mouseX - (canvasWidth - canvasWidth * newZoom) / 2 - newZoom * u * canvasWidth;
    const newPanY = mouseY - (canvasHeight - canvasHeight * newZoom) / 2 - newZoom * v * canvasHeight;
    setFftZoom(newZoom);
    setFftPanX(newPanX);
    setFftPanY(newPanY);
  };

  const handleFftDoubleClick = () => {
    setFftZoom(DEFAULT_FFT_ZOOM);
    setFftPanX(0);
    setFftPanY(0);
    setFftClickInfo(null);
  };

  // Convert FFT canvas mouse position to FFT image pixel coordinates
  const fftScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const centerOffsetX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
    const centerOffsetY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    const imgCol = ((mouseX - centerOffsetX) / fftZoom) / canvasW * fftW;
    const imgRow = ((mouseY - centerOffsetY) / fftZoom) / canvasH * fftH;
    if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
      return { col: imgCol, row: imgRow };
    }
    return null;
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsDraggingFftPan(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, panX: fftPanX, panY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingFftPan || !fftPanStart) return;
    const dx = e.clientX - fftPanStart.x;
    const dy = e.clientY - fftPanStart.y;
    setFftPanX(fftPanStart.panX + dx);
    setFftPanY(fftPanStart.panY + dy);
  };

  const handleFftMouseUp = async (e: React.MouseEvent) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const fftPoint = fftScreenToImg(e);
        if (fftPoint) {
          // Use crop dimensions when ROI FFT is active
          const fftW = fftCropDims?.fftWidth ?? width;
          const fftH = fftCropDims?.fftHeight ?? height;
          let imgCol = fftPoint.col;
          let imgRow = fftPoint.row;
          // Snap to nearest Bragg spot (local max in FFT magnitude)
          if (fftMagCacheRef.current) {
            const snapped = await findFFTPeakBrowser(fftMagCacheRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
            imgCol = snapped.col;
            imgRow = snapped.row;
          }
          const halfW = Math.floor(fftW / 2);
          const halfH = Math.floor(fftH / 2);
          const deltaCol = imgCol - halfW;
          const deltaRow = imgRow - halfH;
          const distPx = Math.sqrt(deltaCol * deltaCol + deltaRow * deltaRow);
          if (distPx < 1) {
            setFftClickInfo(null);
          } else {
            let spatialFreq: number | null = null;
            let dSpacing: number | null = null;
            if (pixelSize > 0) {
              const paddedW = nextPow2(fftW);
              const paddedH = nextPow2(fftH);
              ({ spatialFrequency: spatialFreq, dSpacing } = reciprocalCoordinatesFromShiftedOffset(
                Math.round(imgRow) - halfH,
                Math.round(imgCol) - halfW,
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
    setIsDraggingFftPan(false);
    setFftPanStart(null);
  };

  const handleFftMouseLeave = () => {
    fftClickStartRef.current = null;
    setIsDraggingFftPan(false);
    setFftPanStart(null);
  };

  // Gallery FFT zoom/pan handlers (only selected image's FFT responds)
  const handleGalleryFftWheel = (e: React.WheelEvent, idx: number) => {
    if (isGallery && idx !== selectedIdx && !effectiveFftLinkedZoom) return;
    const fftView = getGalleryFftState(idx);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setGalleryFftState(idx, { ...fftView, zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftView.zoom * zoomFactor)) });
  };

  const handleGalleryFftMouseDown = (e: React.MouseEvent, idx: number) => {
    if (isGallery && idx !== selectedIdx && !effectiveFftLinkPan) {
      setSelectedIdx(idx);
      return; // Select first, don't start panning
    }
    const fftView = getGalleryFftState(idx);
    setFftPanningIdx(idx);
    setIsDraggingFftPan(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, panX: fftView.panX, panY: fftView.panY });
  };

  const handleGalleryFftMouseMove = (e: React.MouseEvent, idx: number) => {
    if (!isDraggingFftPan || !fftPanStart || fftPanningIdx !== idx) return;
    const dx = e.clientX - fftPanStart.x;
    const dy = e.clientY - fftPanStart.y;
    const fftView = getGalleryFftState(idx);
    setGalleryFftState(idx, { ...fftView, panX: fftPanStart.panX + dx, panY: fftPanStart.panY + dy });
  };

  const handleGalleryFftMouseUp = () => {
    setIsDraggingFftPan(false);
    setFftPanStart(null);
    setFftPanningIdx(null);
  };

  const resetFftView = React.useCallback((idx: number) => {
    if (idx < 0) {
      handleFftDoubleClick();
      return;
    }
    setGalleryFftState(idx, { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 });
  }, [handleFftDoubleClick, setGalleryFftState]);

  const setSingleFftState = React.useCallback((state: ZoomState) => {
    setFftZoom(state.zoom);
    setFftPanX(state.panX);
    setFftPanY(state.panY);
  }, []);

  const handleFftTouchStart = (e: React.TouchEvent, idx: number) => {
    const canvas = idx < 0 ? fftCanvasRef.current : fftCanvasRefs.current[idx];
    if (!canvas) return;
    if (isGallery && idx >= 0 && idx !== selectedIdx) setSelectedIdx(idx);
    const now = Date.now();
    const base = idx < 0 ? { zoom: fftZoom, panX: fftPanX, panY: fftPanY } : getGalleryFftState(idx);
    if (e.touches.length === 1) {
      const lastTap = lastFftTapRef.current;
      if (lastTap && lastTap.idx === idx && now - lastTap.time < 320) {
        e.preventDefault();
        resetFftView(idx);
        lastFftTapRef.current = null;
        fftTouchStartRef.current = null;
        return;
      }
      lastFftTapRef.current = { time: now, idx };
      const touch = e.touches[0];
      fftTouchStartRef.current = {
        idx,
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
      fftTouchStartRef.current = {
        idx,
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

  const handleFftTouchMove = (e: React.TouchEvent, idx: number) => {
    const start = fftTouchStartRef.current;
    if (!start || start.idx !== idx) return;
    const canvas = idx < 0 ? fftCanvasRef.current : fftCanvasRefs.current[idx];
    if (!canvas) return;
    e.preventDefault();
    const base = start.startState;
    const applyState = (state: ZoomState) => {
      if (idx < 0) setSingleFftState(state);
      else setGalleryFftState(idx, state);
    };
    if (start.mode === "pinch" && e.touches.length >= 2) {
      const firstTouch = e.touches[0];
      const secondTouch = e.touches[1];
      const mid = touchMidpoint(firstTouch, secondTouch);
      const startCanvas = touchToCanvas(start.startMidX, start.startMidY, canvas);
      const currentCanvas = touchToCanvas(mid.x, mid.y, canvas);
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, base.zoom * (touchDistance(firstTouch, secondTouch) / start.startDistance)));
      if (idx < 0) {
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        const centerOffsetX = (canvasWidth - canvasWidth * base.zoom) / 2 + base.panX;
        const centerOffsetY = (canvasHeight - canvasHeight * base.zoom) / 2 + base.panY;
        const u = (startCanvas.x - centerOffsetX) / (base.zoom * canvasWidth);
        const v = (startCanvas.y - centerOffsetY) / (base.zoom * canvasHeight);
        applyState({
          zoom: newZoom,
          panX: currentCanvas.x - (canvasWidth - canvasWidth * newZoom) / 2 - newZoom * u * canvasWidth,
          panY: currentCanvas.y - (canvasHeight - canvasHeight * newZoom) / 2 - newZoom * v * canvasHeight,
        });
      } else {
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const imageX = (startCanvas.x - cx - base.panX) / base.zoom + cx;
        const imageY = (startCanvas.y - cy - base.panY) / base.zoom + cy;
        applyState({
          zoom: newZoom,
          panX: currentCanvas.x - (imageX - cx) * newZoom - cx,
          panY: currentCanvas.y - (imageY - cy) * newZoom - cy,
        });
      }
      return;
    }
    if (start.mode === "pan" && e.touches.length === 1) {
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / Math.max(1, rect.width);
      const scaleY = canvas.height / Math.max(1, rect.height);
      applyState({
        ...base,
        panX: base.panX + (touch.clientX - start.startX) * scaleX,
        panY: base.panY + (touch.clientY - start.startY) * scaleY,
      });
    }
  };

  const handleFftTouchEnd = (e: React.TouchEvent, idx: number) => {
    const start = fftTouchStartRef.current;
    if (!start || start.idx !== idx) return;
    if (e.touches.length > 0) return;
    fftTouchStartRef.current = null;
  };

  // Track which image is being panned
  const [panningIdx, setPanningIdx] = React.useState<number | null>(null);
  const clickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const touchStartRef = React.useRef<TouchZoomState | null>(null);
  const fftTouchStartRef = React.useRef<TouchZoomState | null>(null);
  const lastTapRef = React.useRef<{ time: number; idx: number } | null>(null);
  const lastFftTapRef = React.useRef<{ time: number; idx: number } | null>(null);
  const [draggingProfileEndpoint, setDraggingProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingProfileLine, setIsDraggingProfileLine] = React.useState(false);
  const [hoveredProfileEndpoint, setHoveredProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringProfileLine, setIsHoveringProfileLine] = React.useState(false);
  const profileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);

  const screenToImg = (e: React.MouseEvent, idx: number): { imgCol: number; imgRow: number } => {
    const canvas = canvasRefs.current[idx];
    if (!canvas) return { imgCol: 0, imgRow: 0 };
    const rect = canvas.getBoundingClientRect();
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const view = getZoomState(idx);
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    return {
      imgCol: ((mouseCanvasX - cx - view.panX) / view.zoom + cx) / displayScale,
      imgRow: ((mouseCanvasY - cy - view.panY) / view.zoom + cy) / displayScale,
    };
  };

  const updateAllProfileData = async (p0: { row: number; col: number }, p1: { row: number; col: number }) => {
    if (!rawDataRef.current) return;
    const allProfiles = await Promise.all(rawDataRef.current.map((raw, index) => (
      hiddenPanelSet.has(index) || !raw
        ? Promise.resolve(null)
        : sampleLineProfileBrowser(raw, width, height, p0.row, p0.col, p1.row, p1.col)
    )));
    setProfileDataAll(allProfiles);
  };

  const updateROI = (e: React.MouseEvent, idx: number) => {
    const { imgCol, imgRow } = screenToImg(e, idx);
    updateSelectedRoi({ col: Math.max(0, Math.min(width - 1, Math.floor(imgCol))), row: Math.max(0, Math.min(height - 1, Math.floor(imgRow))) });
  };

  const hitTestROI = (imgCol: number, imgRow: number): number => {
    if (!roiActive || !roiList) return -1;
    for (let roiIndex = roiList.length - 1; roiIndex >= 0; roiIndex--) {
      const roi = roiList[roiIndex];
      const shape = roi.shape || "circle";
      if (shape === "circle" || shape === "annular") {
        if (Math.sqrt((imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2) <= roi.radius) return roiIndex;
      } else if (shape === "square") {
        if (Math.abs(imgCol - roi.col) <= roi.radius && Math.abs(imgRow - roi.row) <= roi.radius) return roiIndex;
      } else if (shape === "rectangle") {
        if (Math.abs(imgCol - roi.col) <= roi.width / 2 && Math.abs(imgRow - roi.row) <= roi.height / 2) return roiIndex;
      }
    }
    return -1;
  };

  const getHitArea = () => {
    const zoom = getZoomState(selectedIdx).zoom;
    return RESIZE_HIT_AREA_PX / (displayScale * zoom);
  };

  const isNearEdge = (imgCol: number, imgRow: number, roi: ROIItem): boolean => {
    const hitArea = getHitArea();
    const shape = roi.shape || "circle";
    if (shape === "circle" || shape === "annular") {
      const dist = Math.sqrt((imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2);
      return Math.abs(dist - roi.radius) < hitArea;
    }
    if (shape === "square") {
      const deltaCol = Math.abs(imgCol - roi.col);
      const deltaRow = Math.abs(imgRow - roi.row);
      const radius = roi.radius;
      return (deltaCol <= radius + hitArea && deltaRow <= radius + hitArea) && (Math.abs(deltaCol - radius) < hitArea || Math.abs(deltaRow - radius) < hitArea);
    }
    if (shape === "rectangle") {
      const deltaCol = Math.abs(imgCol - roi.col);
      const deltaRow = Math.abs(imgRow - roi.row);
      const halfWidth = roi.width / 2;
      const halfHeight = roi.height / 2;
      return (deltaCol <= halfWidth + hitArea && deltaRow <= halfHeight + hitArea) && (Math.abs(deltaCol - halfWidth) < hitArea || Math.abs(deltaRow - halfHeight) < hitArea);
    }
    return false;
  };

  const isNearResizeHandle = (imgCol: number, imgRow: number): boolean => {
    if (!roiActive || !selectedRoi) return false;
    return isNearEdge(imgCol, imgRow, selectedRoi);
  };

  const isNearAnyEdge = (imgCol: number, imgRow: number): boolean => {
    if (!roiActive || !roiList) return false;
    return roiList.some(roi => isNearEdge(imgCol, imgRow, roi));
  };

  const isNearResizeHandleInner = (imgCol: number, imgRow: number): boolean => {
    if (!roiActive || !selectedRoi || selectedRoi.shape !== "annular") return false;
    const hitArea = getHitArea();
    const dist = Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
    return Math.abs(dist - selectedRoi.radius_inner) < hitArea;
  };

  const beginInsetPlotDrag = React.useCallback((e: React.MouseEvent, idx: number): boolean => {
    if (showInsetPlots === false || !hasInsetPlots) return false;
    const canvas = canvasRefs.current[idx];
    if (!canvas) return false;
    const spec = insetPlotSpecFor(idx);
    const rect = canvas.getBoundingClientRect();
    const cssX = (e.clientX - rect.left) * (canvas.width / Math.max(1, rect.width));
    const cssY = (e.clientY - rect.top) * (canvas.height / Math.max(1, rect.height));
    const inset = insetPlotGeometry(spec, canvas.width, canvas.height, scaleBarVisible);
    if (!inset) return false;
    const grabPad = 10;
    if (
      cssX < inset.x0 - grabPad ||
      cssX > inset.x0 + inset.boxW + grabPad ||
      cssY < inset.y0 - grabPad ||
      cssY > inset.y0 + inset.boxH + grabPad
    ) {
      return false;
    }
    insetDragStateRef.current = {
      idx,
      offsetX: cssX - inset.x0,
      offsetY: cssY - inset.y0,
      boxW: inset.boxW,
      boxH: inset.boxH,
    };
    insetDragDraftRef.current.set(idx, {
      ...(spec || {}),
      box: [inset.x0 / canvas.width, inset.y0 / canvas.height, inset.boxW / canvas.width, inset.boxH / canvas.height],
    });
    insetHoverKeyRef.current = "";
    setInsetHoverInfo(null);
    scheduleInsetDragPaint();
    e.preventDefault();
    e.stopPropagation();
    return true;
  }, [hasInsetPlots, insetPlotSpecFor, scaleBarVisible, scheduleInsetDragPaint, showInsetPlots]);

  const updateInsetPlotDrag = React.useCallback((e: React.MouseEvent, idx: number): boolean => {
    const drag = insetDragStateRef.current;
    if (!drag || drag.idx !== idx) return false;
    const canvas = canvasRefs.current[idx];
    if (!canvas) return false;
    const rect = canvas.getBoundingClientRect();
    const cssX = (e.clientX - rect.left) * (canvas.width / Math.max(1, rect.width));
    const cssY = (e.clientY - rect.top) * (canvas.height / Math.max(1, rect.height));
    const x0 = Math.max(0, Math.min(canvas.width - drag.boxW, cssX - drag.offsetX));
    const y0 = Math.max(0, Math.min(canvas.height - drag.boxH, cssY - drag.offsetY));
    const base = insetPlotSpecFor(idx) || {};
    insetDragDraftRef.current.set(idx, {
      ...base,
      box: [x0 / canvas.width, y0 / canvas.height, drag.boxW / canvas.width, drag.boxH / canvas.height],
    });
    scheduleInsetDragPaint();
    e.preventDefault();
    e.stopPropagation();
    return true;
  }, [insetPlotSpecFor, scheduleInsetDragPaint]);

  const finishInsetPlotDrag = React.useCallback((e?: React.MouseEvent, idxArg?: number): boolean => {
    const drag = insetDragStateRef.current;
    if (!drag) return false;
    const idx = idxArg ?? drag.idx;
    const canvas = canvasRefs.current[idx];
    const draft = insetDragDraftRef.current.get(idx);
    if (canvas && draft) {
      const inset = insetPlotGeometry(draft, canvas.width, canvas.height, scaleBarVisible);
      if (inset) {
        const right = inset.x0 + inset.boxW / 2 >= canvas.width / 2;
        const bottom = inset.y0 + inset.boxH / 2 >= canvas.height / 2;
        const position = `${bottom ? "bottom" : "top"}-${right ? "right" : "left"}` as InsetPlotSpec["position"];
        const marginX = right ? canvas.width - inset.x0 - inset.boxW : inset.x0;
        const bottomScaleBarOffset = scaleBarVisible && position === "bottom-right" ? 34 : 0;
        const marginY = bottom ? canvas.height - inset.y0 - inset.boxH - bottomScaleBarOffset : inset.y0 - 18;
        const { box: _box, ...rest } = draft;
        const nextSpec: InsetPlotSpec = {
          ...rest,
          position,
          margin: [
            Math.max(0, Math.round(marginX)),
            Math.max(0, Math.round(marginY)),
          ],
        };
        const next = Array.isArray(insetPlots) ? insetPlots.slice() : [];
        next[idx] = nextSpec;
        setInsetPlots(next);
      }
    }
    insetDragStateRef.current = null;
    insetDragDraftRef.current.delete(idx);
    scheduleInsetDragPaint();
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    return true;
  }, [insetPlots, scaleBarVisible, scheduleInsetDragPaint, setInsetPlots]);

  const handleMouseDown = (e: React.MouseEvent, idx: number) => {
    if (e.detail >= 2) {
      handleDoubleClick(e, idx);
      return;
    }
    if (handlePanelSelectionMouseDown(e, idx)) return;
    const view = getZoomState(idx);
    if (isGallery && idx !== selectedIdx) {
      setSelectedIdx(idx);
      // Continue to pan setup so click-drag on unselected panel pans immediately
      // (no double-click required to select first then drag).
    }
    if (beginInsetPlotDrag(e, idx)) return;
    // A click on the lens inset: edge = resize, interior = drag
    if (showLens && !isGallery && idx === 0) {
      const canvas = canvasRefs.current[0];
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;
        const { x: lensX, y: lensY } = lensInsetOrigin(lensAnchor, canvasH, lensDisplaySize);
        const lensHit = lensInsetHit(cssX, cssY, lensX, lensY, lensDisplaySize);
        if (lensHit) {
          if (lensHit === "edge") {
            setIsResizingLens(true);
            lensResizeStartRef.current = { mouseY: e.clientY, startSize: lensDisplaySize };
          } else {
            setIsDraggingLens(true);
            lensDragStartRef.current = { mouseX: e.clientX, mouseY: e.clientY, anchorX: lensX, anchorY: lensY };
          }
          e.preventDefault();
          return;
        }
      }
    }
    clickStartRef.current = { x: e.clientX, y: e.clientY };
    if (overlayEditMode) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      const hitRadius = 10 / Math.max(0.01, displayScale * view.zoom);
      const hit = panelOverlayHit(panelOverlays?.[idx], imgRow, imgCol, width, height, hitRadius);
      if (hit) {
        const original = panelOverlays?.[idx]?.[hit.overlay];
        if (!original) return;
        setOverlaySelection({ panel: idx, overlay: hit.overlay });
        setAnnotationSelection(null);
        overlayDragRef.current = {
          mode: hit.mode,
          panel: idx,
          overlay: hit.overlay,
          handle: hit.handle,
          startRow: imgRow,
          startCol: imgCol,
          original,
        };
        setIsDraggingOverlay(true);
        setIsDraggingPan(false);
        setPanStart(null);
        setPanningIdx(null);
        e.preventDefault();
        return;
      }
      setOverlaySelection(null);
      setAnnotationSelection(null);
    }
    if (profileActive) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      if (profilePoints.length === 2) {
        const p0 = profilePoints[0];
        const p1 = profilePoints[1];
        const hitRadius = 10 / (displayScale * view.zoom);
        const distanceToStart = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
        const distanceToEnd = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
        if (distanceToStart <= hitRadius || distanceToEnd <= hitRadius) {
          setDraggingProfileEndpoint(distanceToStart <= distanceToEnd ? 0 : 1);
          setIsDraggingPan(false);
          setPanStart(null);
          setPanningIdx(null);
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
          setPanStart(null);
          setPanningIdx(null);
          return;
        }
      }
      setIsDraggingPan(true);
      setPanningIdx(idx);
      setPanStart({ x: e.clientX, y: e.clientY, panX: view.panX, panY: view.panY });
      return;
    }
    if (roiActive) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      // Check resize handles on selected ROI first
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
      // Edge of any ROI: auto-select it and start resizing
      if (roiList) {
        for (let roiIndex = 0; roiIndex < roiList.length; roiIndex++) {
          if (isNearEdge(imgCol, imgRow, roiList[roiIndex])) {
            e.preventDefault();
            const roi = roiList[roiIndex];
            resizeAspectRef.current = roi && (roi.shape === "rectangle") && roi.width > 0 && roi.height > 0 ? roi.width / roi.height : null;
            setRoiSelectedIdx(roiIndex);
            setIsDraggingResize(true);
            return;
          }
        }
      }
      // Hit-test existing ROIs (click inside to select + drag)
      const hitIdx = hitTestROI(imgCol, imgRow);
      if (hitIdx >= 0) {
        setRoiSelectedIdx(hitIdx);
        setIsDraggingROI(true);
        return;
      }
      // Click on empty space: deselect and allow panning
      setRoiSelectedIdx(-1);
    }
    // Start panning (works in both ROI-active and normal modes)
    setIsDraggingPan(true);
    setPanningIdx(idx);
    setPanStart({ x: e.clientX, y: e.clientY, panX: view.panX, panY: view.panY });
  };

  const handleMouseMove = (e: React.MouseEvent, idx: number) => {
    if (updateInsetPlotDrag(e, idx)) return;
    if (annotationDragRef.current) {
      const drag = annotationDragRef.current;
      if (idx !== drag.panel) return;
      updatePanelAnnotation(drag.panel, drag.annotation, updateAnnotationFromDrag(drag, e.clientX, e.clientY));
      e.preventDefault();
      return;
    }
    if (overlayDragRef.current) {
      const drag = overlayDragRef.current;
      if (idx !== drag.panel) return;
      const { imgCol, imgRow } = screenToImg(e, idx);
      updatePanelOverlay(
        drag.panel,
        drag.overlay,
        updateOverlayFromDrag(drag.original, drag.mode, drag.startRow, drag.startCol, imgRow, imgCol, width, height, drag.handle),
      );
      e.preventDefault();
      return;
    }
    // Fast path: during pan drag, skip all cursor/hover/lens work and only update pan
    if (isDraggingPan && panStart && panningIdx !== null) {
      const canvas = canvasRefs.current[idx];
      if (!canvas || idx !== panningIdx) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - panStart.x) * scaleX;
      const dy = (e.clientY - panStart.y) * scaleY;
      const view = getZoomState(idx);
      setZoomState(idx, { ...view, panX: panStart.panX + dx, panY: panStart.panY + dy });
      return;
    }

    // Cursor readout: convert screen position to image pixel coordinates
    const canvas = canvasRefs.current[idx];
    if (canvas && rawDataRef.current) {
      const rect = canvas.getBoundingClientRect();
      const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
      const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
      const insetHover = showInsetPlots !== false ? insetHoverAt(
        insetPlotSpecFor(idx),
        idx,
        canvas.width,
        canvas.height,
        mouseCanvasX,
        mouseCanvasY,
        scaleBarVisible,
      ) : null;
      const insetHoverKey = insetHover
        ? `${insetHover.idx}:${insetHover.text}:${insetHover.leftPct.toFixed(1)}:${insetHover.topPct.toFixed(1)}`
        : "";
      if (insetHoverKey !== insetHoverKeyRef.current) {
        insetHoverKeyRef.current = insetHoverKey;
        setInsetHoverInfo(insetHover);
      }
      const view = getZoomState(idx);
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const imageCanvasX = (mouseCanvasX - cx - view.panX) / view.zoom + cx;
      const imageCanvasY = (mouseCanvasY - cy - view.panY) / view.zoom + cy;
      const imgCol = Math.floor(imageCanvasX / displayScale);
      const imgRow = Math.floor(imageCanvasY / displayScale);
      if (imgCol >= 0 && imgCol < width && imgRow >= 0 && imgRow < height) {
        const rawData = rawDataRef.current[idx];
        if (rawData) {
          const binFactor = Math.max(1, displayBinFactor || 1);
          const nativeRow = Math.floor(imgRow * binFactor);
          const nativeCol = Math.floor(imgCol * binFactor);
          let value = rawData[imgRow * width + imgCol];
          let valueSource: "preview" | "detail" | "native" = binFactor > 1 ? "preview" : "native";
          const detailTile = binFactor > 1 ? detailTilesRef.current.get(idx) : undefined;
          if (detailTile) {
            const tileRow = Math.floor((nativeRow - detailTile.row0) / detailTile.bin);
            const tileCol = Math.floor((nativeCol - detailTile.col0) / detailTile.bin);
            if (tileRow >= 0 && tileRow < detailTile.rows && tileCol >= 0 && tileCol < detailTile.cols) {
              value = detailTile.floats[tileRow * detailTile.cols + tileCol];
              valueSource = detailTile.bin === 1 ? "native" : "detail";
            }
          }
          // RGB panels read out the (r, g, b) triplet instead of a scalar.
          const rgbData = isRgbPanel(idx) ? rgbDataRef.current[idx] : null;
          const rgbOffset = (imgRow * width + imgCol) * 3;
          setCursorInfo({
            idx, row: nativeRow, col: nativeCol, value, valueSource,
            rgb: rgbData ? [rgbData[rgbOffset], rgbData[rgbOffset + 1], rgbData[rgbOffset + 2]] : null,
          });
        }
        if (showLens && !isGallery) setLensPos({ row: imgRow, col: imgCol });
      } else {
        setCursorInfo(null);
        // Keep lensPos: the lens stays at its last position while the toggle is on
      }
    }

    if (isDraggingLens && lensDragStartRef.current) {
      const dx = e.clientX - lensDragStartRef.current.mouseX;
      const dy = e.clientY - lensDragStartRef.current.mouseY;
      setLensAnchor({ x: lensDragStartRef.current.anchorX + dx, y: lensDragStartRef.current.anchorY + dy });
      return;
    }
    if (isResizingLens && lensResizeStartRef.current) {
      const dy = e.clientY - lensResizeStartRef.current.mouseY;
      setLensDisplaySize(Math.max(64, Math.min(256, lensResizeStartRef.current.startSize + dy)));
      return;
    }

    if (profileActive && profilePoints.length === 2) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      const p0 = profilePoints[0];
      const p1 = profilePoints[1];
      const hitRadius = 10 / (displayScale * getZoomState(idx).zoom);
      const distanceToStart = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
      const distanceToEnd = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
      if (draggingProfileEndpoint !== null) {
        const clampedRow = Math.max(0, Math.min(height - 1, imgRow));
        const clampedCol = Math.max(0, Math.min(width - 1, imgCol));
        const next = [
          draggingProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
          draggingProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
        ];
        setProfilePoints(next);
        updateAllProfileData(next[0], next[1]);
        return;
      }
      if (isDraggingProfileLine && profileDragStartRef.current) {
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
        deltaCol = Math.min(deltaCol, (width - 1) - maxCol);
        const next = [
          { row: drag.p0.row + deltaRow, col: drag.p0.col + deltaCol },
          { row: drag.p1.row + deltaRow, col: drag.p1.col + deltaCol },
        ];
        setProfilePoints(next);
        updateAllProfileData(next[0], next[1]);
        return;
      }
      const nextHoveredEndpoint: 0 | 1 | null = distanceToStart <= hitRadius ? 0 : distanceToEnd <= hitRadius ? 1 : null;
      const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(imgCol, imgRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius;
      setHoveredProfileEndpoint(nextHoveredEndpoint);
      setIsHoveringProfileLine(nextHoverLine);
    } else {
      if (hoveredProfileEndpoint !== null) setHoveredProfileEndpoint(null);
      if (isHoveringProfileLine) setIsHoveringProfileLine(false);
    }

    // ROI resize drag (inner annular ring)
    if (isDraggingResizeInner && selectedRoi) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      const newRadius = Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
      updateSelectedRoi({ radius_inner: Math.max(1, Math.min(selectedRoi.radius - 1, Math.round(newRadius))) });
      return;
    }
    // ROI resize drag (outer)
    if (isDraggingResize && selectedRoi) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      const shape = selectedRoi.shape || "circle";
      if (shape === "rectangle") {
        let newWidth = Math.max(2, Math.round(Math.abs(imgCol - selectedRoi.col) * 2));
        let newHeight = Math.max(2, Math.round(Math.abs(imgRow - selectedRoi.row) * 2));
        if (e.shiftKey && resizeAspectRef.current != null) {
          const aspect = resizeAspectRef.current;
          if (newWidth / newHeight > aspect) newHeight = Math.max(2, Math.round(newWidth / aspect));
          else newWidth = Math.max(2, Math.round(newHeight * aspect));
        }
        updateSelectedRoi({ width: newWidth, height: newHeight });
      } else {
        const newRadius = shape === "square" ? Math.max(Math.abs(imgCol - selectedRoi.col), Math.abs(imgRow - selectedRoi.row)) : Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
        const minRadius = shape === "annular" ? selectedRoi.radius_inner + 1 : 1;
        updateSelectedRoi({ radius: Math.max(minRadius, Math.round(newRadius)) });
      }
      return;
    }
    // ROI drag (move center)
    if (isDraggingROI) {
      updateROI(e, idx);
      return;
    }
    // Lens edge hover detection
    if (showLens && !isGallery && canvas) {
      const rect = canvas.getBoundingClientRect();
      const cssX = e.clientX - rect.left;
      const cssY = e.clientY - rect.top;
      const { x: lensX, y: lensY } = lensInsetOrigin(lensAnchor, canvasH, lensDisplaySize);
      setIsHoveringLensEdge(lensInsetHit(cssX, cssY, lensX, lensY, lensDisplaySize) === "edge");
    } else {
      setIsHoveringLensEdge(false);
    }
    if (overlayEditMode && !isDraggingPan) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      const hitRadius = 10 / Math.max(0.01, displayScale * getZoomState(idx).zoom);
      const hit = panelOverlayHit(panelOverlays?.[idx], imgRow, imgCol, width, height, hitRadius);
      setIsHoveringOverlay(Boolean(hit));
      return;
    } else if (isHoveringOverlay) {
      setIsHoveringOverlay(false);
    }
    // Hover detection for resize handles (show cursor on any ROI edge)
    if (roiActive && !isDraggingPan) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      setIsHoveringResizeInner(isNearResizeHandleInner(imgCol, imgRow));
      setIsHoveringResize(isNearAnyEdge(imgCol, imgRow));
    }
  };

  const handleMouseUp = (e: React.MouseEvent, idx: number) => {
    if (finishInsetPlotDrag(e, idx)) return;
    if (annotationDragRef.current) {
      annotationDragRef.current = null;
      setIsDraggingAnnotation(false);
      return;
    }
    if (overlayDragRef.current) {
      overlayDragRef.current = null;
      setIsDraggingOverlay(false);
      return;
    }
    if (isDraggingLens) {
      setIsDraggingLens(false);
      lensDragStartRef.current = null;
      return;
    }
    if (isResizingLens) {
      setIsResizingLens(false);
      lensResizeStartRef.current = null;
      return;
    }
    if (draggingProfileEndpoint !== null || isDraggingProfileLine) {
      setDraggingProfileEndpoint(null);
      setIsDraggingProfileLine(false);
      profileDragStartRef.current = null;
      clickStartRef.current = null;
      setIsDraggingROI(false);
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      setIsDraggingPan(false);
      setPanStart(null);
      setPanningIdx(null);
      setHoveredProfileEndpoint(null);
      setIsHoveringProfileLine(false);
      return;
    }
    // Detect click (vs drag) for profile mode
    if (profileActive && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        // A click, not a drag: place a profile point
        if (canvasRefs.current[idx] && rawDataRef.current) {
          const { imgCol, imgRow } = screenToImg(e, idx);
          if (imgCol >= 0 && imgCol < width && imgRow >= 0 && imgRow < height) {
            const point = { row: imgRow, col: imgCol };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              // Start new line
              setProfilePoints([point]);
              setProfileDataAll([]);
            } else {
              // Complete the line
              const p0 = profilePoints[0];
              setProfilePoints([p0, point]);
              updateAllProfileData(p0, point);
            }
          }
        }
      }
    }
    // Detect click for measurement mode (only when profile is not active)
    if (measureActive && !profileActive && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        if (canvasRefs.current[idx]) {
          const { imgCol, imgRow } = screenToImg(e, idx);
          if (imgCol >= 0 && imgCol < width && imgRow >= 0 && imgRow < height) {
            const point = { row: imgRow, col: imgCol };
            if (measurePoints.length < 2) {
              setMeasurePoints([...measurePoints, point]);
            } else {
              setMeasurePoints([point]);
            }
          }
        }
      }
    }
    clickStartRef.current = null;
    if (isDraggingPan) persistZoomState(getZoomState(idx));
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    profileDragStartRef.current = null;
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingPan(false);
    setPanStart(null);
    setPanningIdx(null);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
  };

  const handleMouseLeave = (idx: number) => {
    setCursorInfo(null);
    insetHoverKeyRef.current = "";
    setInsetHoverInfo(null);
    overlayDragRef.current = null;
    setIsDraggingOverlay(false);
    setIsHoveringOverlay(false);
    if (insetDragStateRef.current?.idx === idx) {
      insetDragStateRef.current = null;
      insetDragDraftRef.current.delete(idx);
      scheduleInsetDragPaint();
    }
    // Keep lensPos: the lens stays at its last position while the toggle is on
    setIsDraggingLens(false);
    setIsResizingLens(false);
    lensDragStartRef.current = null;
    lensResizeStartRef.current = null;
    setIsHoveringLensEdge(false);
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    profileDragStartRef.current = null;
    setIsHoveringResize(false);
    setIsHoveringResizeInner(false);
    if (panningIdx === idx) {
      persistZoomState(getZoomState(idx));
      setIsDraggingPan(false);
      setPanStart(null);
      setPanningIdx(null);
    }
  };
  const clearTransientHoverInspection = () => {
    setCursorInfo(null);
    insetHoverKeyRef.current = "";
    setInsetHoverInfo(null);
    setIsHoveringOverlay(false);
    setIsHoveringResize(false);
    setIsHoveringResizeInner(false);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
  };
  const handleRootMouseMoveCapture = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cursorInfo && !insetHoverInfo && !isHoveringOverlay && !isHoveringResize && !isHoveringResizeInner && hoveredProfileEndpoint === null && !isHoveringProfileLine) return;
    const target = e.target instanceof Element ? e.target : null;
    if (target?.closest("[data-show2d-image-panel], [data-show2d-fft-panel]")) return;
    clearTransientHoverInspection();
  };
  const handleRootMouseLeave = () => {
    clearTransientHoverInspection();
  };

  // -------------------------------------------------------------------------
  // Copy to clipboard handler
  const handleCopy = React.useCallback(async () => {
    const canvas = canvasRefs.current[isGallery ? selectedIdx : 0];
    if (!canvas) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/png"));
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch {
      // Fallback: download if clipboard API unavailable
      canvas.toBlob((fallbackBlob) => { if (fallbackBlob) downloadBlob(fallbackBlob, `show2d_${labels?.[selectedIdx] || "image"}.png`); }, "image/png");
    }
  }, [isGallery, selectedIdx, labels]);

  const buildSvgExport = React.useCallback((scale: number = 3): Show2DSvgExport => {
    const exportScale = Math.max(1, Math.min(8, Math.round(Number(scale) || 2)));
    const panels = isGallery ? visibleImageIndices : [0];
    if (panels.length === 0 || canvasW <= 0 || canvasH <= 0) {
      throw new Error("no visible Show2D panels");
    }
    const cols = isGallery ? Math.max(1, Math.min(clampedNcols, panels.length)) : 1;
    const rows = Math.max(1, Math.ceil(panels.length / cols));
    const gap = isGallery ? galleryGapPx : 0;
    const frame = isGallery ? galleryOuterBorderPx : 0;
    const frameColor = galleryOuterBorderColor;
    const gapColor = galleryGapColorResolved;
    const titleHeight = showTitle !== false && title ? 30 : 0;
    const svgWidth = 2 * frame + cols * canvasW + (cols - 1) * gap;
    const svgHeight = titleHeight + 2 * frame + rows * canvasH + (rows - 1) * gap;
    const filename = makeSvgExportFilename(title, panels.length, height, width, exportScale);

    {
      const defs: string[] = [];
      const body: string[] = [];
      if (frame > 0 && frameColor) {
        body.push(`<rect x="0" y="${titleHeight}" width="${svgWidth}" height="${Math.max(0, svgHeight - titleHeight)}" fill="${escapeXmlAttr(svgColor(frameColor))}"/>`);
      }
      if (gap > 0 && gapColor) {
        body.push(`<rect x="${frame}" y="${titleHeight + frame}" width="${Math.max(0, svgWidth - 2 * frame)}" height="${Math.max(0, svgHeight - titleHeight - 2 * frame)}" fill="${escapeXmlAttr(svgColor(gapColor))}"/>`);
      }
      if (titleHeight > 0) {
        body.push(
          `<text x="${svgWidth / 2}" y="19" text-anchor="middle" font-family="${SYSTEM_FONT}" font-size="14" font-weight="700" fill="${escapeXmlAttr(svgColor(themeColors.text))}">${escapeXmlText(title)}</text>`
        );
      }

      const titleFontFamily = styleString(panelTitleStyle?.font_family, SYSTEM_FONT);
      const titleFg = svgColor(panelTitleStyle?.fg, "#fff");
      const titleWeight = panelTitleStyle?.font_weight ?? 700;
      const titleOpacity = Math.max(0, Math.min(1, styleNumber(panelTitleStyle?.opacity, 0.95)));
      const titleOutlineWidth = Math.max(0, styleNumber(panelTitleStyle?.outline_width, 0));
      const titleOutlineColor = svgColor(panelTitleStyle?.outline_color, "rgba(0,0,0,0.85)");
      const titleTextAnchor = (anchor: string): string => {
        if (anchor.includes("right")) return "end";
        if (anchor.includes("center")) return "middle";
        return "start";
      };
      const titleBaselineY = (anchor: string, top: number, fontSize: number): number => {
        if (anchor.includes("center")) return top + fontSize * 0.35;
        if (anchor.includes("bottom")) return top;
        return top + fontSize;
      };
      const titlePosition = (panelX: number, panelY: number, fontSize: number): { x: number; y: number; anchor: string } => {
        const rawX = Number(panelTitleStyle?.x);
        const rawY = Number(panelTitleStyle?.y);
        const offset = Array.isArray(panelTitleStyle?.offset) ? panelTitleStyle.offset.map(Number) : [0, 0];
        if (Number.isFinite(rawX) || Number.isFinite(rawY)) {
          const anchor = styleString(panelTitleStyle?.anchor, "top-center").toLowerCase();
          return {
            x: panelX + (Number.isFinite(rawX) ? rawX : 0.5) * canvasW + Number(offset[0] || 0),
            y: titleBaselineY(anchor, panelY + (Number.isFinite(rawY) ? rawY : 0) * canvasH + Number(offset[1] || 0), fontSize),
            anchor: titleTextAnchor(anchor),
          };
        }
        const align = styleString(panelTitleStyle?.align, "center").toLowerCase();
        if (align === "left" || align === "start") return { x: panelX + 28, y: panelY + 6 + fontSize, anchor: "start" };
        if (align === "right" || align === "end") return { x: panelX + canvasW - 28, y: panelY + 6 + fontSize, anchor: "end" };
        return { x: panelX + canvasW / 2, y: panelY + 6 + fontSize, anchor: "middle" };
      };
      const titleOutlineText = (xPos: number, yPos: number, anchor: string, fontSize: number, text: string): string =>
        `<text x="${xPos}" y="${yPos}" text-anchor="${anchor}" font-family="${escapeXmlAttr(titleFontFamily)}" font-size="${fontSize}" font-weight="${escapeXmlAttr(titleWeight)}" fill="none" stroke="${escapeXmlAttr(titleOutlineColor)}" stroke-width="${titleOutlineWidth}" stroke-linejoin="round">${escapeXmlText(text)}</text>`;

      groupMarkerOverlays.forEach((marker) => {
        body.push(
          `<rect x="${marker.left}" y="${titleHeight + marker.top}" width="${marker.width}" height="${marker.height}" fill="none" stroke="${escapeXmlAttr(marker.color)}" stroke-width="3"/>`,
          `<rect x="${marker.left + 3}" y="${titleHeight + marker.top + 3}" width="${Math.max(0, marker.width - 6)}" height="${Math.max(0, marker.height - 6)}" fill="none" stroke="#000" stroke-opacity="0.9" stroke-width="2"/>`
        );
      });

      for (let slot = 0; slot < panels.length; slot += 1) {
        const panel = panels[slot];
        const canvas = canvasRefs.current[panel];
        if (!canvas) continue;
        const col = slot % cols;
        const row = Math.floor(slot / cols);
        const x = frame + col * (canvasW + gap);
        const y = titleHeight + frame + row * (canvasH + gap);
        const clipId = `show2d-svg-clip-${slot}`;
        defs.push(`<clipPath id="${clipId}"><rect x="${x}" y="${y}" width="${canvasW}" height="${canvasH}"/></clipPath>`);
        const dataUrl = scaledCanvasPngDataUrl(canvas, exportScale, smooth);
        const panelStroke = svgColor(panelInnerBorderColor, themeColors.border);
        body.push(
          `<g id="show2d-panel-${panel}">`,
          `<rect x="${x}" y="${y}" width="${canvasW}" height="${canvasH}" fill="#000"/>`,
          `<image x="${x}" y="${y}" width="${canvasW}" height="${canvasH}" xlink:href="${escapeXmlAttr(dataUrl)}" preserveAspectRatio="none"/>`
        );
        if (panelInnerBorderPx > 0) {
          body.push(`<rect x="${x}" y="${y}" width="${canvasW}" height="${canvasH}" fill="none" stroke="${escapeXmlAttr(panelStroke)}" stroke-width="${panelInnerBorderPx}"/>`);
        }
        const markerColor = panelMarkerColor(panel);
        if (markerAround) {
          body.push(
            `<rect x="${x + 1.5}" y="${y + 1.5}" width="${Math.max(0, canvasW - 3)}" height="${Math.max(0, canvasH - 3)}" fill="none" stroke="${escapeXmlAttr(markerColor)}" stroke-width="3"/>`,
            `<rect x="${x + 4}" y="${y + 4}" width="${Math.max(0, canvasW - 8)}" height="${Math.max(0, canvasH - 8)}" fill="none" stroke="#000" stroke-opacity="0.9" stroke-width="2"/>`
          );
        } else {
          body.push(`<rect x="${x}" y="${y}" width="5" height="${canvasH}" fill="${escapeXmlAttr(markerColor)}"/>`);
        }

        if (showPanelTitles !== false) {
          const label = panelTitleText(panel);
          if (label) {
            const fontSize = Math.max(8, panelTitleFontSize || 11);
            const richTitle = panelTitleSpans?.[panel];
            body.push(
              `<g clip-path="url(#${clipId})">`
            );
            if (richTitle?.length) {
              const rich = svgTextFromRichSpans(richTitle, label);
              const titlePos = titlePosition(x, y, fontSize);
              const lineY = titlePos.y;
              if (titleOutlineWidth <= 0) {
                body.push(`<text x="${titlePos.x + 1}" y="${lineY + 1}" text-anchor="${titlePos.anchor}" font-family="${escapeXmlAttr(titleFontFamily)}" font-size="${fontSize}" font-weight="${escapeXmlAttr(titleWeight)}" fill="#000" fill-opacity="0.85">${escapeXmlText(rich.text)}</text>`);
              } else {
                body.push(titleOutlineText(titlePos.x, lineY, titlePos.anchor, fontSize, rich.text));
              }
              body.push(`<text x="${titlePos.x}" y="${lineY}" text-anchor="${titlePos.anchor}" font-family="${escapeXmlAttr(titleFontFamily)}" font-size="${fontSize}" font-weight="${escapeXmlAttr(titleWeight)}" fill="${escapeXmlAttr(titleFg)}" fill-opacity="${titleOpacity}">`);
              rich.spans.forEach((span) => body.push(`<tspan${span.color ? ` fill="${escapeXmlAttr(svgColor(span.color))}"` : ""}>${escapeXmlText(span.text)}</tspan>`));
              body.push("</text>");
            } else {
              const titleLines = wrapSvgTextLines(label, fontSize, Math.max(24, canvasW - 56), 3);
              titleLines.forEach((line, lineIdx) => {
                const titlePos = titlePosition(x, y, fontSize);
                const lineY = titlePos.y + lineIdx * fontSize * 1.2;
                if (titleOutlineWidth <= 0) {
                  body.push(`<text x="${titlePos.x + 1}" y="${lineY + 1}" text-anchor="${titlePos.anchor}" font-family="${escapeXmlAttr(titleFontFamily)}" font-size="${fontSize}" font-weight="${escapeXmlAttr(titleWeight)}" fill="#000" fill-opacity="0.85">${escapeXmlText(line)}</text>`);
                } else {
                  body.push(titleOutlineText(titlePos.x, lineY, titlePos.anchor, fontSize, line));
                }
                body.push(`<text x="${titlePos.x}" y="${lineY}" text-anchor="${titlePos.anchor}" font-family="${escapeXmlAttr(titleFontFamily)}" font-size="${fontSize}" font-weight="${escapeXmlAttr(titleWeight)}" fill="${escapeXmlAttr(titleFg)}" fill-opacity="${titleOpacity}">${escapeXmlText(line)}</text>`);
              });
            }
            body.push(`</g>`);
          }
        }

        const vectorLayer: string[] = [];
        if (showInsetPlots !== false) {
          vectorLayer.push(svgInsetPlotElement(insetPlotSpecFor(panel), x, y, canvasW, canvasH, markerColor, panelHasScaleBar(panel)));
        }
        const panelOverlaySpecs = panelOverlays?.[panel] || [];
        if (panelOverlaySpecs.length > 0) {
          const view = getZoomState(panel);
          const cx = canvasW / 2;
          const cy = canvasH / 2;
          const toScreenX = (imgCol: number) => x + (imgCol * displayScale - cx) * view.zoom + cx + view.panX;
          const toScreenY = (imgRow: number) => y + (imgRow * displayScale - cy) * view.zoom + cy + view.panY;
          panelOverlaySpecs.forEach((overlay) => vectorLayer.push(svgPanelOverlayElement(overlay, toScreenX, toScreenY, width, height)));
        }
        if (showColorbar && !isGallery) {
          const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
          const colorbarId = `show2d-svg-colorbar-${slot}`;
          const colorbar = svgColorbarElements(lut, x, y, canvasW, canvasH, colorbarVminRef.current, colorbarVmaxRef.current, colorbarId);
          defs.push(colorbar.def);
          vectorLayer.push(colorbar.body);
        }
        (panelAnnotations?.[panel] || []).forEach((annotation) => {
          vectorLayer.push(svgPanelAnnotationElement(annotation, x, y, canvasW, canvasH));
        });
        if (vectorLayer.some(Boolean)) {
          body.push(`<g data-show2d-vector-layer="true" clip-path="url(#${clipId})">${vectorLayer.join("")}</g>`);
        }

        if (panelHasScaleBar(panel)) {
          const view = getZoomState(panel);
          const panelPixelSize = pixelSizeForPanel(panel);
          const barPixelSize = panelPixelSize > 0 ? panelPixelSize : 1;
          const unit = panelPixelSize > 0 ? pixelUnit : "px";
          const scaleBar = scaleBarGeometry({ x: 0, y: 0, width: canvasW, height: canvasH }, view.zoom * (canvasW / width), barPixelSize, unit, show2dScaleBarOptions(scaleBarPosition, scaleBarLength, scaleBarLabel, scaleBarStyle, Boolean(showZoomIndicator)));
          if (scaleBar) {
            const barY = y + scaleBar.barY;
            const barX = x + scaleBar.barX;
            const barColor = svgColor(scaleBarStyle?.color, "#fff");
            const shadowColor = svgColor(scaleBarStyle?.shadow_color, "#000");
            const barShadowColor = scaleBarStyle?.shadow_color == null ? "" : shadowColor;
            const outlineColor = svgColor(scaleBarStyle?.outline_color, "#000");
            const outlineWidth = Math.max(0, styleNumber(scaleBarStyle?.outline_width, 0));
            const labelGap = styleNumber(scaleBarStyle?.label_gap, 4);
            const fontAttrs = scaleBarSvgFontAttrs(scaleBarStyle);
            const labelX = barX + scaleBar.barPx / 2;
            const labelY = barY - labelGap;
            if (barShadowColor) {
              body.push(`<rect x="${barX + 1}" y="${barY + 1}" width="${scaleBar.barPx}" height="${scaleBar.barHeight}" fill="${escapeXmlAttr(barShadowColor)}" fill-opacity="0.5"/>`);
            }
            body.push(`<rect x="${barX}" y="${barY}" width="${scaleBar.barPx}" height="${scaleBar.barHeight}" fill="${escapeXmlAttr(barColor)}"/>`);
            if (outlineWidth > 0) {
              body.push(
                `<text x="${labelX}" y="${labelY}" text-anchor="middle" ${fontAttrs} fill="none" stroke="${escapeXmlAttr(outlineColor)}" stroke-width="${outlineWidth}" stroke-linejoin="round">${escapeXmlText(scaleBar.label)}</text>`,
                `<text x="${labelX}" y="${labelY}" text-anchor="middle" ${fontAttrs} fill="${escapeXmlAttr(barColor)}">${escapeXmlText(scaleBar.label)}</text>`
              );
            } else {
              body.push(
                `<text x="${labelX + 1}" y="${labelY + 1}" text-anchor="middle" ${fontAttrs} fill="${escapeXmlAttr(shadowColor)}" fill-opacity="0.85">${escapeXmlText(scaleBar.label)}</text>`,
                `<text x="${labelX}" y="${labelY}" text-anchor="middle" ${fontAttrs} fill="${escapeXmlAttr(barColor)}">${escapeXmlText(scaleBar.label)}</text>`
              );
            }
            if (showZoomIndicator) {
              const zoomX = scaleBar.scaleLeft ? x + canvasW - 12 : x + 12;
              const anchor = scaleBar.scaleLeft ? "end" : "start";
              const zoomText = formatZoomLabel(view.zoom);
              body.push(
                `<text x="${zoomX + 1}" y="${y + canvasH - 12 + 5 + 1}" text-anchor="${anchor}" ${fontAttrs} fill="${escapeXmlAttr(shadowColor)}" fill-opacity="0.85">${escapeXmlText(zoomText)}</text>`,
                `<text x="${zoomX}" y="${y + canvasH - 12 + 5}" text-anchor="${anchor}" ${fontAttrs} fill="${escapeXmlAttr(barColor)}">${escapeXmlText(zoomText)}</text>`
              );
            }
          }
        }
        body.push("</g>");
      }

      const svg = [
        `<?xml version="1.0" encoding="UTF-8"?>`,
        `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${svgWidth} ${svgHeight}" role="img" aria-label="${escapeXmlAttr(title || "Show2D SVG export")}">`,
        `<defs>${defs.join("")}</defs>`,
        body.join(""),
        `</svg>`,
      ].join("\n");
      return { svg, width: svgWidth, height: svgHeight, filename, scale: exportScale };
    }
  }, [
    canvasH,
    canvasW,
    clampedNcols,
    cmap,
    galleryGapPx,
    galleryGapColorResolved,
    galleryOuterBorderColor,
    galleryOuterBorderPx,
    getZoomState,
    groupMarkerOverlays,
    height,
    displayScale,
    insetPlotSpecFor,
    isGallery,
    markerAround,
    panelMarkerColor,
    panelHasScaleBar,
    panelInnerBorderColor,
    panelInnerBorderPx,
    panelAnnotations,
    panelOverlays,
    panelTitleFontSize,
    panelTitleStyle,
    panelTitleSpans,
    panelTitleText,
    pixelSizeForPanel,
    pixelUnit,
    scaleBarPosition,
    scaleBarLength,
    scaleBarLabel,
    scaleBarStyle,
    showColorbar,
    showInsetPlots,
    showPanelTitles,
    showTitle,
    showZoomIndicator,
    smooth,
    themeColors.border,
    themeColors.text,
    title,
    visibleImageIndices,
    width,
  ]);

  const clearSvgPreview = React.useCallback(() => {
    if (svgPreviewUrlRef.current) {
      window.URL.revokeObjectURL(svgPreviewUrlRef.current);
      svgPreviewUrlRef.current = null;
    }
    setSvgPreview(null);
  }, []);

  React.useEffect(() => () => {
    if (svgPreviewUrlRef.current) {
      window.URL.revokeObjectURL(svgPreviewUrlRef.current);
      svgPreviewUrlRef.current = null;
    }
  }, []);

  const snapSvgPreviewToDevicePixels = React.useCallback(() => {
    const image = svgPreviewImageRef.current;
    if (!image) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = image.getBoundingClientRect();
    const current = svgPreviewSnapRef.current;
    const layoutLeft = rect.left - current.x;
    const layoutTop = rect.top - current.y;
    const x = Math.ceil(layoutLeft * dpr) / dpr - layoutLeft;
    const y = Math.ceil(layoutTop * dpr) / dpr - layoutTop;
    svgPreviewSnapRef.current = { x, y };
    setSvgPreviewSnap((prev) => (
      Math.abs(prev.x - x) < 0.001 && Math.abs(prev.y - y) < 0.001
        ? prev
        : { x, y }
    ));
  }, []);

  React.useLayoutEffect(() => {
    if (!svgPreview) {
      svgPreviewSnapRef.current = { x: 0, y: 0 };
      setSvgPreviewSnap((prev) => (prev.x === 0 && prev.y === 0 ? prev : { x: 0, y: 0 }));
      return undefined;
    }
    let frame = window.requestAnimationFrame(snapSvgPreviewToDevicePixels);
    const timers = [
      window.setTimeout(snapSvgPreviewToDevicePixels, 40),
      window.setTimeout(snapSvgPreviewToDevicePixels, 160),
      window.setTimeout(snapSvgPreviewToDevicePixels, 400),
    ];
    const onResize = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(snapSvgPreviewToDevicePixels);
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.cancelAnimationFrame(frame);
      timers.forEach((timer) => window.clearTimeout(timer));
      window.removeEventListener("resize", onResize);
    };
  }, [snapSvgPreviewToDevicePixels, svgPreview]);

  const handleSvgPreview = React.useCallback((scale: number = 3) => {
    setExportAnchor(null);
    try {
      const built = buildSvgExport(scale);
      const blob = new Blob([built.svg], { type: "image/svg+xml;charset=utf-8" });
      if (svgPreviewUrlRef.current) window.URL.revokeObjectURL(svgPreviewUrlRef.current);
      const url = window.URL.createObjectURL(blob);
      svgPreviewUrlRef.current = url;
      setSvgPreview({ ...built, url, size: blob.size });
      setLocalExportStatus(`Preview ${built.filename} (${formatSavedBytes(blob.size)})`);
    } catch (err) {
      setLocalExportStatus(`Preview failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [buildSvgExport]);

  const handleSvgExport = React.useCallback(async (scale: number = 3) => {
    setExportAnchor(null);
    try {
      const exportScale = Math.max(1, Math.min(8, Math.round(Number(scale) || 2)));
      const built = svgPreview && svgPreview.scale === exportScale ? svgPreview : buildSvgExport(exportScale);
      const blob = new Blob([built.svg], { type: "image/svg+xml;charset=utf-8" });
      const picker = (window as Show2DWindow).showSaveFilePicker;
      let handle: Show2DFileHandle | null = null;
      if (picker) {
        try {
          handle = await picker({
            suggestedName: built.filename,
            types: [{ description: "SVG image", accept: { "image/svg+xml": [".svg"] } }],
          });
        } catch (err) {
          if (isAbortLikeError(err)) {
            setLocalExportStatus("Export canceled");
            return;
          }
          throw err;
        }
      }
      if (handle) {
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
      } else {
        downloadBlob(blob, built.filename);
      }
      setLocalExportStatus(`Saved ${built.filename} (${formatSavedBytes(blob.size)})`);
    } catch (err) {
      setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [buildSvgExport, svgPreview]);

  // Resize Handlers
  // -------------------------------------------------------------------------
  const handleCanvasResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    canvasResizeCleanupRef.current?.();
    const perAnchors = new Map<number, ZoomAnchor>();
    zoomStates.forEach((state, idx) => perAnchors.set(idx, zoomStateToAnchor(state)));
    canvasResizeViewAnchorRef.current = {
      linked: zoomStateToAnchor(linkedZoomState),
      per: perAnchors,
      reset: resetZoomStateRef.current ? zoomStateToAnchor(resetZoomStateRef.current) : null,
    };
    const start = { x: e.clientX, y: e.clientY, size: canvasSize };
    let rafId = 0;
    let latestSize = start.size;

    const handleMouseMove = (event: MouseEvent) => {
      const delta = Math.max(event.clientX - start.x, event.clientY - start.y);
      latestSize = Math.max(200, start.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          applyResizeViewAnchor(latestSize);
          setCanvasSize(latestSize);
        });
      }
    };

    const finishResize = () => {
      cancelAnimationFrame(rafId);
      applyResizeViewAnchor(latestSize);
      setCanvasSize(latestSize);
      setCanvasSizeTrait(Math.round(latestSize));
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", finishResize);
      window.removeEventListener("blur", finishResize);
      canvasResizeCleanupRef.current = null;
      window.setTimeout(() => { canvasResizeViewAnchorRef.current = null; }, 0);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", finishResize);
    window.addEventListener("blur", finishResize);
    canvasResizeCleanupRef.current = finishResize;
  };

  React.useEffect(() => {
    return () => {
      canvasResizeCleanupRef.current?.();
      canvasResizeCleanupRef.current = null;
    };
  }, []);

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

  // -------------------------------------------------------------------------
  // Keyboard shortcuts
  // -------------------------------------------------------------------------
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (shouldIgnoreWidgetShortcut(e.target)) return;
    // image_rotations counts CCW quarter turns, so the clockwise key adds three.
    const rotateSelectedPanel = (quarterTurns: number) => {
      const panel = isGallery ? selectedIdx : 0;
      const rotations = [...(imageRotations || [])];
      while (rotations.length <= panel) rotations.push(0);
      rotations[panel] = (rotations[panel] + quarterTurns) % 4;
      setImageRotations(rotations);
    };
    // Number keys 1-9 select gallery images (avoids arrow key conflicts with Jupyter)
    if (isGallery && e.key >= "1" && e.key <= "9") {
      const target = pageShortcutTarget(
        visibleImageIndices,
        parseInt(e.key) - 1,
        isPaged,
        nImages,
      );
      if (target !== null) { e.preventDefault(); setSelectedIdx(target); }
      return;
    }
    switch (e.key) {
      case "ArrowLeft":
        if ((panelFrameCounts?.[selectedIdx] || 1) > 1) {
          e.preventDefault();
          stopPanelPlayback(selectedIdx);
          setPanelFrameIndex(selectedIdx, (panelFramePreviewIndices[selectedIdx] || 0) - 1, true);
        } else if (isGallery) {
          e.preventDefault();
          const position = visibleImageIndices.indexOf(selectedIdx);
          setSelectedIdx(visibleImageIndices[Math.max(0, position - 1)] ?? 0);
        }
        break;
      case "ArrowRight":
        if ((panelFrameCounts?.[selectedIdx] || 1) > 1) {
          e.preventDefault();
          stopPanelPlayback(selectedIdx);
          setPanelFrameIndex(selectedIdx, (panelFramePreviewIndices[selectedIdx] || 0) + 1, true);
        } else if (isGallery) {
          e.preventDefault();
          const position = visibleImageIndices.indexOf(selectedIdx);
          setSelectedIdx(visibleImageIndices[Math.min(visibleImageIndices.length - 1, position + 1)] ?? 0);
        }
        break;
      case "r":
      case "R":
        handleResetAll();
        break;
      case "m":
      case "M":
        setMeasureActive(!measureActive);
        setMeasurePoints([]);
        break;
      case "Escape":
        if (measureActive) {
          setMeasureActive(false);
          setMeasurePoints([]);
        }
        break;
      case "h":
      case "H":
        if (isGallery) {
          const candidates = selectedVisiblePanels.length > 0 ? selectedVisiblePanels : [selectedIdx];
          const hideable = candidates.filter((panel) => visibleImageIndices.includes(panel));
          if (hideable.length > 0 && visibleImageCount - hideable.length >= 1) {
            e.preventDefault();
            setPanelsHidden(hideable, true);
          }
        }
        break;
      case "]":
        e.preventDefault();
        rotateSelectedPanel(3);
        break;
      case "[":
        e.preventDefault();
        rotateSelectedPanel(1);
        break;
      case "Delete":
      case "Backspace":
        if (overlayEditMode && annotationSelection) {
          e.preventDefault();
          deleteSelectedAnnotation();
        } else if (overlayEditMode && overlaySelection) {
          e.preventDefault();
          deleteSelectedOverlay();
        } else if (roiActive && roiSelectedIdx >= 0 && roiList && roiSelectedIdx < roiList.length) {
          e.preventDefault();
          const newList = roiList.filter((_, i) => i !== roiSelectedIdx);
          setRoiList(newList);
          setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1);
        }
        break;
    }
  };
  const handleRootMouseDownCapture = (e: React.MouseEvent<HTMLDivElement>) => {
    if (shouldIgnoreWidgetShortcut(e.target)) return;
    staticFallbackRootRef.current?.focus({ preventScroll: true });
  };

  // -------------------------------------------------------------------------
  // Render (Show3D-style layout)
  // -------------------------------------------------------------------------
  const selectedView = getZoomState(isGallery ? selectedIdx : 0);
  const needsReset = selectedView.zoom !== 1 || selectedView.panX !== 0 || selectedView.panY !== 0;
  // Scientists inspect across galleries by moving the mouse, not by selecting
  // every panel first. Keep control state tied to selectedIdx, but let the
  // stats/readout strip follow the hovered panel while the cursor is over it.
  const hoverStatsIdx = isGallery && cursorInfo && visibleImageIndices.includes(cursorInfo.idx)
    ? cursorInfo.idx
    : null;
  const statsIdx = isGallery ? (hoverStatsIdx ?? selectedIdx) : 0;
  const currentFrameStats = localPanelFrameStats.get(statsIdx);
  const svgExportAvailable = canvasW > 0 && canvasH > 0 && visibleImageIndices.length > 0;
  const isDraggingEditableDecoration = isDraggingOverlay || isDraggingAnnotation;

  // Calibrated cursor position - unit is whatever the user passed via sampling/units.
  const calibratedUnit = pixelSize > 0 ? pixelUnit : "";
  const nativePixelSizeForPanel = (idx: number) => {
    const panelPixelSize = pixelSizeForPanel(idx);
    return panelPixelSize > 0 ? panelPixelSize / Math.max(1, displayBinFactor || 1) : 0;
  };
  const nativePixelSize = pixelSize > 0 ? pixelSize / Math.max(1, displayBinFactor || 1) : 0;
  const cursorValueSuffix = cursorInfo?.valueSource === "native"
    ? " native"
    : cursorInfo?.valueSource === "detail"
      ? " detail"
      : displayBinFactor > 1
        ? " preview"
        : "";

  // "More" overflow menu: ROI, Denoise, Filter, Diff and the display toggles live here to keep the primary
  // toolbar uncrowded. moreActiveCount drives the badge + per-item accent so an
  // active tool stays legible with the menu closed (house rule: a live
  // reduction is never hidden). ROI honors its !isGallery guard; Diff honors
  // the same availability condition it used inline.
  const roiControlAvailable = !isGallery;
  const diffControlAvailable = !isPaged && nImages >= 2 && (visibleGrayscaleIndices.length === 2 || diffMode);
  const moreActiveCount =
    (roiControlAvailable && roiActive ? 1 : 0) + (diffMode ? 1 : 0) + (denoiseEnabled ? 1 : 0)
    + (hasInsetPlots && showInsetPlots !== false ? 1 : 0)
    + (overlayEditMode ? 1 : 0)
    + (frequencyFilterEnabled && Array.from({ length: nImages }, (_, panel) => panelFrequencyKnobs(panel)).some(knobs => frequencyFilterActive(knobs.mode)) ? 1 : 0)
    + (isGallery && !colorShared ? 1 : 0)
    + (Array.from({ length: nImages }, (_, panel) => rotationForPanel(panel)).some(turns => turns !== 0) ? 1 : 0);
  const rotationActive = Array.from({ length: nImages }, (_, panel) => rotationForPanel(panel)).some(turns => turns !== 0);
  const clearRotations = React.useCallback(() => {
    setImageRotations(Array.from({ length: Math.max(1, nImages || 1) }, () => 0));
    setShowRotationSettings(false);
  }, [nImages, setImageRotations]);
  const frequencyUiKnobs = panelFrequencyKnobs(Math.min(Math.max(0, selectedIdx || 0), Math.max(0, nImages - 1)));
  const frequencyValueLabel = (value: number, panel = selectedIdx) => {
    const sampling = pixelSizeForPanel(Math.min(Math.max(0, panel || 0), Math.max(0, nImages - 1)));
    const unit = String(pixelUnit || "").trim().toLowerCase();
    if (sampling > 0 && (unit === "nm" || unit.includes("nanometer"))) return `${(value / (2 * sampling)).toFixed(3)} nm⁻¹`;
    if (sampling > 0 && (unit === "a" || unit === "å" || unit.includes("angstrom"))) return `${(value * 10 / (2 * sampling)).toFixed(3)} nm⁻¹`;
    return `${value.toFixed(3)} Nyq`;
  };
  const frequencyBannerText = frequencyFilterEnabled
    ? (frequencyUiKnobs.mode === "none" ? "" : frequencyUiKnobs.mode === "bandpass"
      ? `Filter: Band-pass center ${frequencyValueLabel(frequencyUiKnobs.center)}, width ${frequencyValueLabel(frequencyUiKnobs.width)} (view only; raw counts unchanged)`
      : `Filter: ${frequencyUiKnobs.mode === "lowpass" ? "Low-pass" : "High-pass"} cutoff ${frequencyValueLabel(frequencyUiKnobs.cutoff)} (view only; raw counts unchanged)`)
    : "";
  const setFrequencyMaster = (enabled: boolean) => {
    if (enabled && !Array.from({ length: nImages }, (_, panel) => panelFrequencyKnobs(panel)).some(knobs => frequencyFilterActive(knobs.mode))) {
      setFrequencyFilter("lowpass");
      mirrorFrequencyKnobEdit("mode", "lowpass");
    }
    setFrequencyFilterEnabled(enabled);
    setShowFrequencyFilter(enabled); // reveal the settings row while filtering; hide it when off (mirrors Denoise)
  };
  // Collapse-safe reduction badge: when the controls (and their inline denoise
  // banners) are hidden, surface any active reduction in the always-on
  // title row. Strip the trailing "how to undo" hint for the compact label and
  // keep the full text in the tooltip.
  const collapsedBannerParts = [
    filterBannerText ? filterBannerText.split(" (")[0] : "",
    frequencyBannerText ? frequencyBannerText.split(" (")[0] : "",
  ].filter(Boolean);
  const collapsedBannerLabel = collapsedBannerParts.join(" · ");
  const collapsedBannerTitle = [filterBannerText, frequencyBannerText].filter(Boolean).join("   |   ");
  const commitFrequencyRing = (panel: number, mode: string, value: number) => {
    const updatePanel = (values: number[] | undefined, fallback: number) => {
      if (frequencyFilterScopeAll) return new Array<number>(nImages).fill(value);
      const next = values?.length === nImages ? [...values] : new Array<number>(nImages).fill(fallback);
      next[panel] = value;
      return next;
    };
    setSelectedIdx(panel);
    if (mode === "bandpass") {
      setFrequencyFilterCenter(value);
      setFrequencyFilterCenters(updatePanel(frequencyFilterCenters, frequencyUiKnobs.center));
    } else {
      setFrequencyFilterCutoff(value);
      setFrequencyFilterCutoffs(updatePanel(frequencyFilterCutoffs, frequencyUiKnobs.cutoff));
    }
    setFrequencyDraft(null);
  };
  const frequencyRingOverlayForPanel = (panel: number) => {
    const knobs = panelFrequencyKnobs(panel);
    const ringValue = knobs.mode === "bandpass" ? knobs.center : knobs.cutoff;
    return frequencyFilterEnabled && frequencyFilterActive(knobs.mode) ? (
    <Box
      className="quantem-frequency-filter-ring"
      data-frequency-filter={knobs.mode}
      aria-label={`Draggable ${knobs.mode} frequency ring at ${frequencyValueLabel(ringValue, panel)}`}
      title="Drag the ring to choose a frequency from the FFT"
      onMouseDown={(event: React.MouseEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.stopPropagation();
        const parent = event.currentTarget.parentElement;
        if (!parent) return;
        const rect = parent.getBoundingClientRect();
        const valueAt = (clientX: number, clientY: number) => Math.max(0, Math.min(1, Math.hypot(clientX - (rect.left + rect.width / 2), clientY - (rect.top + rect.height / 2)) / (Math.min(rect.width, rect.height) / 2)));
        setSelectedIdx(panel);
        const onMove = (moveEvent: MouseEvent) => setFrequencyDraft(valueAt(moveEvent.clientX, moveEvent.clientY));
        const onUp = (upEvent: MouseEvent) => {
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
          commitFrequencyRing(panel, knobs.mode, valueAt(upEvent.clientX, upEvent.clientY));
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
        const value = Math.max(0, Math.min(1, Math.hypot(event.clientX - (rect.left + rect.width / 2), event.clientY - (rect.top + rect.height / 2)) / (Math.min(rect.width, rect.height) / 2)));
        setFrequencyDraft(value);
      }}
      onPointerUp={(event: React.PointerEvent<HTMLDivElement>) => {
        event.stopPropagation();
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        commitFrequencyRing(panel, knobs.mode, ringValue);
      }}
      sx={{
        position: "absolute", left: "50%", top: "50%",
        width: `${ringValue * 100}%`, height: `${ringValue * 100}%`,
        transform: "translate(-50%, -50%)", borderRadius: "50%",
        border: "2px solid rgba(0, 229, 255, 0.95)",
        bgcolor: knobs.mode === "highpass" ? "rgba(0,0,0,0.55)" : "transparent",
        boxShadow: knobs.mode === "lowpass"
          ? "0 0 0 1px rgba(0,0,0,0.75), 0 0 0 9999px rgba(0,0,0,0.55)"
          : "0 0 0 1px rgba(0,0,0,0.75)",
        cursor: "crosshair", touchAction: "none", zIndex: 6,
      }}
    >
      {knobs.mode === "bandpass" && [
        Math.max(0, knobs.center - knobs.width / 2),
        Math.min(1, knobs.center + knobs.width / 2),
      ].map((radius, index) => (
        <Box key={index} sx={{ position: "absolute", left: "50%", top: "50%", width: `${radius / Math.max(0.001, ringValue) * 100}%`, height: `${radius / Math.max(0.001, ringValue) * 100}%`, transform: "translate(-50%, -50%)", borderRadius: "50%", border: "1px dashed rgba(255,255,255,0.95)", bgcolor: index === 0 ? "rgba(0,0,0,0.55)" : "transparent", boxShadow: index === 1 ? "0 0 0 9999px rgba(0,0,0,0.55)" : "none", pointerEvents: "none" }} />
      ))}
      <Box sx={{ position: "absolute", left: "50%", top: -24, transform: "translateX(-50%)", px: 0.75, py: 0.25, borderRadius: 0.75, bgcolor: "rgba(0,0,0,0.78)", color: "rgba(200,250,255,0.98)", fontSize: 9, lineHeight: 1.2, fontWeight: 700, whiteSpace: "nowrap", pointerEvents: "none", textShadow: "0 1px 1px #000" }}>
        {knobs.mode === "lowpass" ? "Inside kept" : knobs.mode === "highpass" ? "Outside kept" : "Band kept"}
      </Box>
    </Box>
    ) : null;
  };
  const handleToggleRoi = (on: boolean) => {
    setRoiActive(on);
    if (on) {
      setProfileActive(false);
      setProfilePoints([]);
      setProfileDataAll([]);
      setHoveredProfileEndpoint(null);
      setIsHoveringProfileLine(false);
    } else {
      setRoiSelectedIdx(-1);
    }
  };

  // Panel overlays drawn the same way on a gallery panel and on the single image.
  const renderInsetHoverReadout = (panel: number) => panelChromeVisible && insetHoverInfo && insetHoverInfo.idx === panel && (
    <Box
      sx={{
        position: "absolute",
        left: `${insetHoverInfo.leftPct}%`,
        top: `${insetHoverInfo.topPct}%`,
        transform: "translate(-8px, -100%)",
        bgcolor: "rgba(0,0,0,0.78)",
        color: "rgba(255,255,255,0.94)",
        border: "1px solid rgba(255,255,255,0.25)",
        px: 0.6,
        py: 0.25,
        pointerEvents: "none",
        zIndex: 12,
        boxShadow: "0 1px 3px rgba(0,0,0,0.45)",
      }}
    >
      <Typography sx={{ fontSize: 9, fontFamily: "monospace", whiteSpace: "nowrap", lineHeight: 1.2 }}>
        {insetHoverInfo.text}
      </Typography>
    </Box>
  );
  const renderPanelAnnotations = (panel: number) => (panelAnnotations?.[panel] || []).map((annotation, annotationIdx) => (
    <Box
      key={`panel-annotation-${panel}-${annotationIdx}`}
      className={annotation.class_name}
      data-show2d-panel-annotation={panel}
      data-show2d-panel-annotation-index={annotationIdx}
      data-show2d-panel-annotation-position={annotation.position || "top-left"}
      data-show2d-panel-annotation-variant={annotation.variant || "badge"}
      title={annotation.text}
      onMouseDown={(event: React.MouseEvent<HTMLElement>) => beginPanelAnnotationDrag(event, panel, annotationIdx)}
      sx={{
        ...panelAnnotationSx(annotation),
        pointerEvents: overlayEditMode ? "auto" : "none",
        cursor: overlayEditMode ? (isDraggingAnnotation ? "grabbing" : "grab") : "inherit",
        ...(overlayEditMode && annotationSelection?.panel === panel && annotationSelection.annotation === annotationIdx ? {
          outline: "1px dashed rgba(255,255,255,0.9)",
          outlineOffset: 2,
        } : {}),
      }}
    >
      {renderPanelAnnotation(annotation)}
    </Box>
  ));
  const renderPanelFrameControls = (panel: number) => panelChromeVisible && (panelFrameCounts?.[panel] || 1) > 1 && (
    <Box
      data-show2d-panel-frame-controls={panel}
      onPointerDown={(event: React.PointerEvent) => {
        event.stopPropagation();
        if (isGallery) setSelectedIdx(panel);
      }}
      onMouseDown={(event: React.MouseEvent) => event.stopPropagation()}
      onTouchStart={(event: React.TouchEvent) => event.stopPropagation()}
      onWheel={(event: React.WheelEvent) => event.stopPropagation()}
      sx={{
        position: "absolute",
        left: 7,
        right: 7,
        bottom: 24,
        minHeight: 24,
        px: 0.5,
        display: "flex",
        alignItems: "center",
        gap: "8px",
        borderRadius: 0.75,
        bgcolor: "rgba(0,0,0,0.58)",
        boxShadow: "0 1px 3px rgba(0,0,0,0.35)",
        zIndex: 4,
      }}
    >
      <IconButton
        size="small"
        onClick={(event) => {
          event.stopPropagation();
          togglePanelPlayback(panel);
        }}
        title={playingPanelFrames.has(panel) ? `Pause ${panelLabel(panel)} frames` : `Play ${panelLabel(panel)} frames`}
        aria-label={playingPanelFrames.has(panel) ? `Pause frames for ${panelLabel(panel)}` : `Play frames for ${panelLabel(panel)}`}
        sx={{
          width: 20,
          height: 20,
          p: 0,
          flex: "0 0 20px",
          zIndex: 1,
          color: "rgba(255,255,255,0.92)",
          "&:hover": { bgcolor: "rgba(255,255,255,0.14)" },
        }}
      >
        {playingPanelFrames.has(panel)
          ? <PauseIcon sx={{ fontSize: 14 }} />
          : <PlayArrowIcon sx={{ fontSize: 14 }} />}
      </IconButton>
      <Slider
        value={panelFramePreviewIndices[panel] ?? normalizedPanelFrameIndices[panel] ?? 0}
        min={0}
        max={Math.max(1, (panelFrameCounts?.[panel] || 1) - 1)}
        step={1}
        onPointerDownCapture={() => stopPanelPlayback(panel)}
        onKeyDown={() => stopPanelPlayback(panel)}
        onChange={(_, value) => {
          const raw = Array.isArray(value) ? value[0] : value;
          setPanelFrameIndex(panel, Number(raw));
        }}
        onChangeCommitted={(_, value) => {
          const raw = Array.isArray(value) ? value[0] : value;
          setPanelFrameIndex(panel, Number(raw), true);
        }}
        size="small"
        sx={{
          ...sliderStyles.small,
          minWidth: 34,
          mx: 0.25,
          flex: "1 1 auto",
          color: "rgba(255,255,255,0.92)",
          "& .MuiSlider-rail": { opacity: 0.45 },
        }}
        aria-label={`Frame for ${panelLabel(panel)}`}
      />
      <Typography
        component="span"
        sx={{
          minWidth: "4.8ch",
          flex: "0 0 auto",
          color: "rgba(255,255,255,0.9)",
          fontSize: 9,
          lineHeight: 1,
          fontVariantNumeric: "tabular-nums",
          textAlign: "right",
          textShadow: "0 1px 2px rgba(0,0,0,0.8)",
        }}
      >
        {(panelFramePreviewIndices[panel] ?? normalizedPanelFrameIndices[panel] ?? 0) + 1}/{panelFrameCounts?.[panel] || 1}
      </Typography>
    </Box>
  );

  return (
    <Box
      className="show2d-root"
      ref={staticFallbackRootRef}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
      onMouseMoveCapture={handleRootMouseMoveCapture}
      onMouseLeave={handleRootMouseLeave}
      data-show2d-panel-playback-fps={panelPlaybackFps}
      data-show2d-selected-panel={selectedIdx}
      data-show2d-selected-panels={(selectedPanels || []).join(",")}
      data-show2d-visible-panel-count={visibleImageCount}
      data-show2d-canvas-repaint-signal={canvasRepaintSignal}
      data-frequency-filter-backend={frequencyFilterEnabled ? frequencyFilterBackend : "off"}
      sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, width: "100%", maxWidth: "100%", boxSizing: "border-box", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", "& canvas": { display: "block" }, "@media (max-width: 700px)": { p: 0, ".jp-OutputArea-output &, .jp-OutputArea-child &": { width: "calc(100vw - 96px)", maxWidth: "calc(100vw - 96px)" } } }}
    >
      {overlayEditMode && (
        <style>
          {".MuiModal-backdrop.MuiBackdrop-invisible { pointer-events: none !important; }"}
        </style>
      )}
      <FolderWatchBadge
        state={folderWatchState}
        detail={folderWatchDetail}
        live={folderWatchLive}
      />
      {folderWaiting && (
        <Box
          role="region"
          aria-label="Show2D folder waiting view"
          data-show2d-folder-waiting="true"
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
            {folderStatus || "Waiting for the first stable image"}
          </Typography>
        </Box>
      )}
      {!folderWaiting && !canRenderLive && hasSavedStaticFallback && (
        <Box sx={{ width: "100%", maxWidth: galleryGridWidth, boxSizing: "border-box" }}>
          <Box
            component="img"
            src={staticFallbackUrl}
            alt={`${title || "Show2D"} saved preview`}
            sx={{
              display: "block",
              width: "100%",
              maxWidth: galleryGridWidth,
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
        spacing={`${SPACING.LG}px`}
        alignItems="flex-start"
        sx={{
          width: "100%",
          maxWidth: "100%",
          minWidth: 0,
          boxSizing: "border-box",
          "@media (max-width: 700px)": {
            flexDirection: "column",
            alignItems: "stretch",
            "& > :not(style) + :not(style)": {
              marginLeft: "0 !important",
              marginTop: effectiveShowFft && !isGallery ? "0 !important" : `${SPACING.LG}px`,
            },
          },
        }}
      >
        {/* Main panel */}
        <Box sx={{ width: "100%", maxWidth: galleryGridWidth, boxSizing: "border-box" }}>
          {/* Title row */}
          {(showTitle || showControls) && <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block", minHeight: 16, lineHeight: "16px", overflow: "visible" }}>
            {showTitle && <>{title || (isGallery ? "Gallery" : "Image")}</>}
            {showTitle && <RenderPathBadge colors={themeColors} />}
            {showTitle && displayBinFactor > 1 && (
              <Box component="span" sx={{ ml: 0.5, px: 0.5, py: 0, fontSize: 9, fontWeight: 600, borderRadius: "3px", backgroundColor: themeColors.accent + "22", color: themeColors.accent, border: `1px solid ${themeColors.accent}44` }}>
                {displayBinFactor}× binned
              </Box>
            )}
            {controlsCollapsed && collapsedBannerLabel && (
              /* House rule: an active reduction is never invisible. The denoise
                 / view banners live inside the (now-hidden) controls block, so
                 mirror the surviving "× binned" badge here while collapsed. */
              <Box component="span" title={collapsedBannerTitle} sx={{ ml: 0.5, px: 0.5, py: 0, fontSize: 9, fontWeight: 600, borderRadius: "3px", backgroundColor: themeColors.accent + "22", color: themeColors.accent, border: `1px solid ${themeColors.accent}44` }}>
                {collapsedBannerLabel}
              </Box>
            )}
            {showTitle && displayBinFactor > 1 && (
              <Box component="span" sx={{ ml: 0.4, px: 0.5, py: 0, fontSize: 9, fontWeight: 500, borderRadius: "3px", backgroundColor: detailStreamStatus === "streaming" ? "rgba(255,193,7,0.18)" : themeColors.controlBg, color: detailStreamStatus === "streaming" ? "#b26a00" : themeColors.textMuted, border: `1px solid ${detailStreamStatus === "streaming" ? "rgba(255,193,7,0.45)" : themeColors.border}` }}>
                {detailStreamStatus === "streaming" ? "streaming detail..." : detailStreamStatus === "ready" ? "detail ready" : "preview; streams on zoom"}
              </Box>
            )}
	            {showControls && <InfoTooltip toggleOnClick text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <MetadataSection rows={[
                ["Shape", isGallery ? `${nImages} x ${height} x ${width}` : `${height} x ${width}`],
                ["Panels", isGallery ? `${nImages} images, ${clampedNcols} columns` : "single image"],
                ["Sampling", pixelSize > 0 ? `${formatNumber(pixelSize)} ${pixelUnit || "px"}/px` : ""],
                ["Display", displayBinFactor > 1 ? `${displayBinFactor}x preview, detail streams on zoom` : "full resolution"],
              ]} />
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum (Fourier transform) alongside image.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on image to draw a line intensity profile.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>ROI: Region of Interest. Click to place, drag to move.</Typography>
              {!isGallery && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Lens: Magnifier inset that follows the cursor.</Typography>}
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile-based contrast (2nd–98th percentile). FFT Auto masks DC + clips to 99.9th.</Typography>
              {isGallery && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Link Zoom / Contrast: Sync zoom or histogram range across all gallery images.</Typography>}
              {isGallery && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Cols / Panels: Change the gallery grid or hide panels without changing the source data.</Typography>}
              {isGallery && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Pinning: Click a panel to select or pin it for keyboard actions, per-panel zoom, ROI edits, and delete shortcuts.</Typography>}
              {hasLocalPanelStacks && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Local stacks: A 3D item inside a list gets its own in-panel slider and play button. Select it and use left/right arrows to change only that panel; hiding or reordering it preserves its frame.</Typography>}
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Pan: With Pan enabled, drag the image to move the zoomed view. With Link Zoom on, pan and zoom move together across gallery panels.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Large data: A binned preview is shown first. When you zoom in, Show2D requests full-resolution detail only for the visible window; small high-zoom windows use native pixels, while larger windows stay lightly binned to keep each reply responsive. The title badge shows whether you are seeing preview, streaming detail, or detail-ready data. Cursor row/column are reported in native full-resolution coordinates; the value is tagged preview/detail/native. The full native stack is not sent to the browser at once. Reduce columns, hide panels, turn off FFT/Profile/Stats, or zoom less to keep interaction faster.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Export / Copy: Save or copy the current panel view using the toolbar actions.</Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Denoise (display only)</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>A view filter for sparse counting maps (EDS, low dose). It never changes the stored array, the Mean/Min/Max/Std stats, or exports of raw counts; set Denoise to None to see raw counts. Any active filter is announced in the banner below the image.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Bin: averages each N by N block of neighbouring pixels, then upsamples back to the display size. Averaging N&sup2; independent pixels cuts the noise by &radic;(N&sup2;) = N, so Bin 2 gives about 2x and Bin 4 about 4x signal-to-noise, at the cost of resolution (features smaller than the bin blur together).</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Poisson (Anscombe): counting noise grows with signal (variance = mean), so a plain blur over-smooths bright regions. The Anscombe transform y = 2&radic;(x + 3/8) makes the noise variance about 1 everywhere, a Gaussian of width &sigma; is applied, then the inverse (y/2)&sup2; - 3/8 maps back to counts. Best for sparse EDS; pair with Bin 2 and &sigma; 6 to 10.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Gaussian: a simple blur of width &sigma;, fine for decent-dose images. None: raw counts, for any quantitative measurement.</Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
              <KeyboardShortcuts items={isGallery ? [["← / →", hasLocalPanelStacks ? "Prev / Next frame in a selected stack; otherwise select panel" : "Prev / Next image"], ["1 – 9", "Select image"], ["Shift-click", "Select panel range"], ["Ctrl/⌘-click", "Toggle panel selection"], ["H", "Hide selected panels"], ["] / [", "Rotate CW / CCW 90°"], ["Del / ⌫", "Delete selected ROI"], ["M", "Measure distance"], ["Esc", "Exit measure"], ["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]] : [["← / →", hasLocalPanelStacks ? "Prev / Next frame" : "No action"], ["] / [", "Rotate CW / CCW 90°"], ["Del / ⌫", "Delete selected ROI"], ["M", "Measure distance"], ["Esc", "Exit measure"], ["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
	            </Box>} theme={themeInfo.theme} />}
	            {showControls && (
	              <Button
	                size="small"
	                sx={{
	                  ...compactButton,
	                  ml: showTitle ? 0.75 : 0,
	                  py: 0,
	                  px: 0.5,
	                  minHeight: 16,
	                  lineHeight: "16px",
	                  verticalAlign: "baseline",
	                  "& .MuiButton-endIcon": { ml: 0.25 },
	                }}
	                onClick={() => setControlsCollapsed(!controlsCollapsed)}
	                aria-label={controlsCollapsed ? "Show controls" : "Hide controls"}
	                aria-pressed={!controlsCollapsed}
	                aria-expanded={!controlsCollapsed}
	                title={controlsCollapsed ? "Show controls" : "Hide controls"}
	                endIcon={<ExpandMoreIcon sx={{ fontSize: 14, transform: controlsCollapsed ? "rotate(-90deg)" : "rotate(0deg)", transition: "transform 0.15s ease" }} />}
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
	                    color: themeColors.accent,
	                  }}
	                  disabled={exportBusy || (!exportEnabled && !canDownloadCurrentHtml)}
	                  onClick={(e) => { setExportAnchor(e.currentTarget); }}
	                  title={localExportStatus || exportStatus || (exportEnabled ? "Export standalone HTML" : canDownloadCurrentHtml ? "Export this standalone HTML" : "Export unavailable for this data")}
	                >
	                  {exportBusy ? "Exporting" : "Export"}
	                </Button>
	                <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "right" }} transformOrigin={{ vertical: "top", horizontal: "right" }} {...themedTopMenuProps}>
	                  {exportEnabled && <MenuItem onClick={() => handleHtmlExportSelect("exact")} sx={{ fontSize: 12 }}>HTML exact float32 ({exactHtmlSize})</MenuItem>}
	                  {exportEnabled && <MenuItem onClick={() => handleHtmlExportSelect("quantized")} sx={{ fontSize: 12 }}>HTML quantized uint8 ({quantizedHtmlSize})</MenuItem>}
	                  {canDownloadCurrentHtml && offline && <MenuItem disabled sx={{ fontSize: 12 }} title="This standalone export contains quantized uint8 data, not the original float32 array. Open the live widget to export exact float32.">{unavailableStandaloneHtmlLabel}</MenuItem>}
	                  {canDownloadCurrentHtml && <MenuItem onClick={handleStandaloneHtmlDownload} sx={{ fontSize: 12 }}>{standaloneHtmlLabel}</MenuItem>}
	                  {canDownloadCurrentHtml && !offline && <MenuItem disabled sx={{ fontSize: 12 }} title="Quantized export requires the Python backend to repack the current float32 data.">{unavailableStandaloneHtmlLabel}</MenuItem>}
	                </Menu>
	              </>
	            )}
	          </Typography>}
	          {/* Page navigation sits above the analysis toolbar, matching Show3D. */}
	          {controlsVisible && isPaged && (
	            <Box
	              data-show2d-page-controls={pageKind || "comparison"}
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
	                  data-show2d-page-status="true"
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
	                  onChange={(e) => setPagePlayFps(Number(e.target.value) || 8)}
	                  size="small"
	                  sx={{ ...themedSelect, minWidth: 48, fontSize: 10 }}
	                  MenuProps={themedTopMenuProps}
	                  inputProps={{ "aria-label": "Page playback frames per second" }}
	                  title="Page playback speed"
	                >
	                  {PAGE_PLAY_FPS_OPTIONS.map((fps) => (
	                    <MenuItem key={fps} value={String(fps)}>{fps} fps</MenuItem>
	                  ))}
	                </Select>
	                {pageKind !== "items" && (
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
	                )}
	              </Box>
	            </Box>
	          )}
	          {/* Analysis and display controls row. */}
	          {controlsVisible && (
	          <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} useFlexGap sx={{ mb: `${SPACING.XS}px`, minHeight: 28, flexWrap: "wrap", rowGap: `${SPACING.XS}px`, width: "100%", maxWidth: "100%", boxSizing: "border-box" }}>
            {isGallery && (
              <Box sx={controlPairSx}>
                <Typography sx={compactLabelSx}>Cols</Typography>
                <Select
                  value={String(clampedNcols)}
                  onChange={(e) => setNcols(Math.max(1, Math.min(Number(e.target.value) || 1, isPaged ? activePanelCount : nImages, MAX_PANEL_COLUMNS)))}
                  size="small"
                  sx={{ ...themedSelect, minWidth: 48, fontSize: 10 }}
                  MenuProps={themedTopMenuProps}
                  inputProps={{ "aria-label": "Gallery columns" }}
                  title="Number of gallery columns"
                >
                  {galleryColumnOptions.map((cols) => (
                    <MenuItem key={cols} value={String(cols)}>{cols}</MenuItem>
                  ))}
                </Select>
              </Box>
            )}
            <Box sx={controlPairSx}>
              <Typography sx={compactLabelSx} title="Line profile: draw a line on the image and read intensity along it.">Profile</Typography>
              <Switch
                checked={profileActive}
                onChange={(e) => {
                  const on = e.target.checked;
                  setProfileActive(on);
                  if (on) {
                    setRoiActive(false);
                  } else {
                    setProfilePoints([]);
                    setProfileDataAll([]);
                    setHoveredProfileEndpoint(null);
                    setIsHoveringProfileLine(false);
                  }
                }}
                size="small"
                sx={switchStyles.small}
              />
            </Box>
            {!isGallery && (
              <Box sx={controlPairSx}>
                <Typography sx={compactLabelSx} title="Magnifier lens: hover the image to zoom a small region.">Lens</Typography>
                <Switch
                  checked={showLens}
                  onChange={() => {
                    if (!showLens) {
                      setShowLens(true);
                      setLensPos({ row: Math.floor(height / 2), col: Math.floor(width / 2) });
                    } else {
                      setShowLens(false);
                      setLensPos(null);
                    }
                  }}
                  size="small"
                  sx={switchStyles.small}
                />
              </Box>
            )}
            <Box sx={controlPairSx}>
              <Typography sx={compactLabelSx} title="Fourier transform view of the current image.">FFT</Typography>
              <Switch
                checked={showFft}
                onChange={(e) => {
                  const on = e.target.checked;
                  if (on && width * height > 2048 * 2048) {
                    console.warn(`Show2D: FFT on ${width}×${height} image (${(width * height / 1e6).toFixed(1)}M pixels) may be slow`);
                  }
                  setShowFft(on);
                }}
                size="small"
                sx={switchStyles.small}
              />
            </Box>
            <Box sx={{ flex: "1 1 24px", minWidth: 8 }} />
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: `${SPACING.SM}px`, flexWrap: "wrap", flex: "0 0 auto", ml: "auto" }}>
              {isGallery && (
                <>
                  {!isPaged && <Button
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
                  </Button>}
                  <Button
                    size="small"
                    sx={{ ...compactButton, "& .MuiButton-startIcon": { mr: 0.4 } }}
                    startIcon={<VisibilityIcon sx={{ fontSize: 14 }} />}
                    onClick={(e) => setPanelMenuAnchor(e.currentTarget)}
                    aria-label="Choose visible panels"
                    aria-controls={panelMenuAnchor ? "show2d-panels-menu" : undefined}
                    aria-expanded={panelMenuAnchor ? "true" : undefined}
                    aria-haspopup="menu"
                  >
                    {allCurrentPanelsVisible ? "Panels" : `Panels ${visibleImageCount}/${panelMenuTotal}`}
                  </Button>
                  {selectedVisibleCount > 1 && selectedVisibleCount < visibleImageCount && (
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
                    id="show2d-panels-menu"
                    anchorEl={panelMenuAnchor}
                    open={Boolean(panelMenuAnchor)}
                    onClose={() => setPanelMenuAnchor(null)}
                    MenuListProps={{ "aria-label": "Panel visibility options" }}
                    {...themedTopMenuProps}
                  >
                    {orderedImageIndices.map((panel) => {
                      const hidden = hiddenPanelSet.has(panel);
                      const disabled = !hidden && visibleImageCount <= 1;
                      return (
                        <MenuItem
                          key={`show2d-panel-menu-${panel}`}
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
                      }}
                    >
                      <VisibilityIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />
                      <Typography sx={{ fontSize: 11 }}>Show all panels</Typography>
                    </MenuItem>
                    <MenuItem
                      dense
                      disabled={selectedVisibleCount <= 1 || selectedVisibleCount >= visibleImageCount}
                      onClick={() => setPanelsHidden(selectedVisiblePanels, true)}
                      title={selectedVisibleCount >= visibleImageCount ? "At least one panel must remain visible" : undefined}
                    >
                      <VisibilityOffIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />
                      <Typography sx={{ fontSize: 11 }}>Hide selected ({selectedVisibleCount})</Typography>
                    </MenuItem>
                    <MenuItem
                      dense
                      disabled={selectedVisibleCount <= 1}
                      onClick={() => setSelectedPanels([selectedIdx])}
                    >
                      <VisibilityIcon sx={{ fontSize: 16, mr: 1, color: themeColors.textMuted }} />
                      <Typography sx={{ fontSize: 11 }}>Clear selection</Typography>
                    </MenuItem>
                    <MenuItem
                      dense
                      disabled={(panelOrder || []).length === 0}
                      onClick={resetPanelOrder}
                    >
                      <DragIndicatorIcon sx={{ fontSize: 16, mr: 1, color: themeColors.accent }} />
                      <Typography sx={{ fontSize: 11 }}>Reset order</Typography>
                    </MenuItem>
                  </Menu>
                </>
              )}
              {/* Overflow "More" menu: ROI, Denoise, Filter, Diff and display toggles. The badge keeps
                  an active tool legible while the menu is closed; active items
                  are accented inside (reuses the Panels N/M + reorder idioms). */}
              <Badge
                badgeContent={moreActiveCount}
                invisible={moreActiveCount === 0}
                sx={{ "& .MuiBadge-badge": { bgcolor: themeColors.accent, color: "#fff", fontSize: 9, fontWeight: 600, minWidth: 14, height: 14, px: 0.25 } }}
              >
                <Button
                  size="small"
                  sx={{ ...compactButton, color: moreActiveCount > 0 ? themeColors.accent : themeColors.text }}
                  onClick={(e) => setMoreMenuAnchor(e.currentTarget)}
                  aria-label="More tools"
                  aria-controls={moreMenuAnchor ? "show2d-more-menu" : undefined}
                  aria-expanded={moreMenuAnchor ? "true" : undefined}
                  aria-haspopup="menu"
                  title="More tools: ROI, Inset Chart, Denoise, Filter, Diff, Color"
                >
                  More
                </Button>
              </Badge>
              <Menu
                id="show2d-more-menu"
                anchorEl={moreMenuAnchor}
                open={Boolean(moreMenuAnchor)}
                onClose={() => setMoreMenuAnchor(null)}
                MenuListProps={{ "aria-label": "More tools" }}
                {...themedTopMenuProps}
                BackdropProps={{
                  sx: { pointerEvents: overlayEditMode ? "none" : "auto" },
                }}
              >
                {hasEditablePanelDecorations && (
                  <MenuItem
                    dense
                    onMouseDown={handleOverlayEditMenuToggle}
                    onClick={(event) => { event.preventDefault(); event.stopPropagation(); }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") handleOverlayEditMenuToggle(event);
                    }}
                    sx={{ fontSize: 12, gap: 1, color: overlayEditMode ? themeColors.accent : themeColors.text }}
                  >
                    <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Edit API-defined labels, circles, and rectangles. Drag labels or shape interiors to move; drag shape edges to resize.">
                      Overlay Edit
                    </Typography>
                    <Switch
                      checked={overlayEditMode}
                      onClick={(event) => { event.preventDefault(); event.stopPropagation(); }}
                      onChange={(event) => { event.preventDefault(); event.stopPropagation(); }}
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
                <Box
                  onClick={(event) => event.stopPropagation()}
                  sx={{
                    px: 1.5,
                    py: 0.75,
                    minWidth: 220,
                    display: "grid",
                    gridTemplateColumns: "1fr auto",
                    gap: 1,
                    alignItems: "center",
                  }}
                >
                  <Typography sx={{ fontSize: 12, color: themeColors.text }} title="Apply a percentile contrast window. The histogram stays visible below the image.">
                    Contrast
                  </Typography>
                  <Select
                    size="small"
                    value={contrastPreset || "manual"}
                    onChange={(event) => applyContrastPreset(String(event.target.value))}
                    renderValue={(value) => CONTRAST_PRESETS.find((preset) => preset.value === value)?.label || "Manual"}
                    MenuProps={themedMenuProps}
                    sx={{ ...themedSelect, minWidth: 76 }}
                    inputProps={{ "aria-label": "Contrast percentile preset" }}
                  >
                    {CONTRAST_PRESETS.map((preset) => (
                      <MenuItem key={preset.value} value={preset.value}>{preset.label}</MenuItem>
                    ))}
                  </Select>
                </Box>
                {roiControlAvailable && (
                  <MenuItem
                    dense
                    onClick={() => handleToggleRoi(!roiActive)}
                    sx={{ fontSize: 12, gap: 1, color: roiActive ? themeColors.accent : themeColors.text }}
                  >
                    <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Region of Interest: click to place, drag to move.">ROI</Typography>
                    <Switch
                      checked={roiActive}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => handleToggleRoi(e.target.checked)}
                      size="small"
                      sx={switchStyles.small}
                      slotProps={{ input: { "aria-label": "Toggle region of interest" } }}
                    />
                  </MenuItem>
                )}
                {hasInsetPlots && (
                  <MenuItem
                    dense
                    onClick={() => setShowInsetPlots(!(showInsetPlots !== false))}
                    sx={{ fontSize: 12, gap: 1, color: showInsetPlots !== false ? themeColors.accent : themeColors.text }}
                  >
                    <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Show/hide initialized inset charts. Drag a chart inside the image to move it; release snaps to the nearest corner.">Inset Chart</Typography>
                    <Switch
                      checked={showInsetPlots !== false}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => setShowInsetPlots(e.target.checked)}
                      size="small"
                      sx={switchStyles.small}
                      slotProps={{ input: { "aria-label": "Toggle inset chart" } }}
                    />
                  </MenuItem>
                )}
                <MenuItem
                  dense
                  onClick={toggleDenoise}
                  sx={{ fontSize: 12, gap: 1, color: denoiseEnabled ? themeColors.accent : themeColors.text }}
                >
                  <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Denoise the display only: ON shows the denoised view, OFF shows raw. Config is preserved across the toggle. Raw data, stats, and exports keep original counts.">Denoise</Typography>
                  <Switch
                    checked={denoiseEnabled ?? false}
                    onClick={(e) => e.stopPropagation()}
                    onChange={toggleDenoise}
                    size="small"
                    sx={switchStyles.small}
                    slotProps={{ input: { "aria-label": "Toggle denoise on/off" } }}
                  />
                </MenuItem>
                <MenuItem
                  dense
                  onClick={() => setFrequencyMaster(!frequencyFilterEnabled)}
                  sx={{ fontSize: 12, gap: 1, color: frequencyFilterEnabled && frequencyFilterActive(frequencyFilter) ? themeColors.accent : themeColors.text }}
                >
                  <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Off by default. Turn on to remove a background or isolate a periodicity; raw counts remain unchanged.">Filter</Typography>
                  <Switch
                    checked={frequencyFilterEnabled ?? false}
                    onClick={(event) => event.stopPropagation()}
                    onChange={() => setFrequencyMaster(!frequencyFilterEnabled)}
                    size="small"
                    sx={switchStyles.small}
                    slotProps={{ input: { "aria-label": "Toggle frequency filter effect" } }}
                  />
                </MenuItem>
                {diffControlAvailable && (
                  <MenuItem
                    dense
                    disabled={!diffMode && visibleGrayscaleIndices.length < 2}
                    onClick={() => setDiffMode(!diffMode)}
                    sx={{ fontSize: 12, gap: 1, color: diffMode ? themeColors.accent : themeColors.text }}
                  >
                    <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Show visible reference − comparison as a derived panel">Diff</Typography>
                    <Switch
                      checked={diffMode}
                      disabled={!diffMode && visibleGrayscaleIndices.length < 2}
                      onClick={(e) => e.stopPropagation()}
                      onChange={() => setDiffMode(!diffMode)}
                      size="small"
                      sx={switchStyles.small}
                      slotProps={{ input: { "aria-label": "Toggle difference of visible panels" } }}
                    />
                  </MenuItem>
                )}
                <MenuItem
                  dense
                  onClick={() => togglePanelFlip(selectedIdx, "h")}
                  sx={{ fontSize: 12, gap: 1, color: imageFlipsHorizontal?.[selectedIdx] ? themeColors.accent : themeColors.text }}
                >
                  <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only horizontal flip for the selected panel. Raw data and coordinates stay unchanged.">Flip H</Typography>
                  <Switch
                    checked={Boolean(imageFlipsHorizontal?.[selectedIdx])}
                    onClick={(e) => e.stopPropagation()}
                    onChange={() => togglePanelFlip(selectedIdx, "h")}
                    size="small"
                    sx={switchStyles.small}
                    slotProps={{ input: { "aria-label": "Toggle horizontal flip for selected panel" } }}
                  />
                </MenuItem>
                <MenuItem
                  dense
                  onClick={() => togglePanelFlip(selectedIdx, "v")}
                  sx={{ fontSize: 12, gap: 1, color: imageFlipsVertical?.[selectedIdx] ? themeColors.accent : themeColors.text }}
                >
                  <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Display-only vertical flip for the selected panel. Raw data and coordinates stay unchanged.">Flip V</Typography>
                  <Switch
                    checked={Boolean(imageFlipsVertical?.[selectedIdx])}
                    onClick={(e) => e.stopPropagation()}
                    onChange={() => togglePanelFlip(selectedIdx, "v")}
                    size="small"
                    sx={switchStyles.small}
                    slotProps={{ input: { "aria-label": "Toggle vertical flip for selected panel" } }}
                  />
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
                      minWidth: 250,
                      display: "grid",
                      gridTemplateColumns: "auto 1fr",
                      gap: 0.75,
                      alignItems: "center",
                    }}
                  >
                    <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Angle</Typography>
                    <Select
                      value={String(((rotationScope || "all") === "panel" ? rotationForPanel(selectedIdx) : rotationForPanel(0)) * 90)}
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
                    {isGallery && (
                      <>
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
                          <MenuItem value="panel">Panel</MenuItem>
                        </Select>
                      </>
                    )}
                    <Typography sx={{ gridColumn: "1 / -1", fontSize: 10, color: themeColors.textMuted }}>
                      {isGallery && (rotationScope || "all") === "panel"
                        ? `Selected panel: ${panelLabel(selectedIdx)}`
                        : "Applies to every visible panel"}
                    </Typography>
                  </Box>
                )}
                {isGallery && (
                  <MenuItem
                    dense
                    onClick={() => setColorShared(!colorShared)}
                    sx={{ fontSize: 12, gap: 1, color: !colorShared ? themeColors.accent : themeColors.text }}
                  >
                    <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }} title="Shared keeps one colormap for every panel. Turn off to let the Color dropdown edit only the selected panel.">Color shared</Typography>
                    <Switch
                      checked={colorShared}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => setColorShared(e.target.checked)}
                      size="small"
                      sx={switchStyles.small}
                      slotProps={{ input: { "aria-label": "Toggle shared panel colormap" } }}
                    />
                  </MenuItem>
                )}
              </Menu>
              <Button
                size="small"
                sx={{ ...compactButton, color: themeColors.accent }}
                disabled={exportBusy || (!exportEnabled && !canDownloadCurrentHtml && !svgExportAvailable)}
                onClick={(e) => { setExportAnchor(e.currentTarget); }}
                title={localExportStatus || exportStatus || (svgExportAvailable ? "Export SVG or standalone HTML" : exportEnabled ? "Export standalone HTML" : canDownloadCurrentHtml ? "Export this standalone HTML" : "Export unavailable for this data")}
              >
                {exportBusy ? "Exporting" : "Export"}
              </Button>
              <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "right" }} transformOrigin={{ vertical: "top", horizontal: "right" }} {...themedTopMenuProps}>
                {svgExportAvailable && !svgPreview && <MenuItem onClick={() => handleSvgPreview(3)} sx={{ fontSize: 12 }}>Preview SVG</MenuItem>}
                {svgPreview && <MenuItem onClick={() => { setExportAnchor(null); clearSvgPreview(); setLocalExportStatus("Exited SVG preview"); }} sx={{ fontSize: 12 }}>Exit SVG preview</MenuItem>}
                {svgExportAvailable && <MenuItem onClick={() => handleSvgExport(3)} sx={{ fontSize: 12 }}>SVG</MenuItem>}
                {exportEnabled && <MenuItem onClick={() => handleHtmlExportSelect("exact")} sx={{ fontSize: 12 }}>HTML exact float32 ({exactHtmlSize})</MenuItem>}
                {exportEnabled && <MenuItem onClick={() => handleHtmlExportSelect("quantized")} sx={{ fontSize: 12 }}>HTML quantized uint8 ({quantizedHtmlSize})</MenuItem>}
                {canDownloadCurrentHtml && offline && <MenuItem disabled sx={{ fontSize: 12 }} title="This standalone export contains quantized uint8 data, not the original float32 array. Open the live widget to export exact float32.">{unavailableStandaloneHtmlLabel}</MenuItem>}
                {canDownloadCurrentHtml && <MenuItem onClick={handleStandaloneHtmlDownload} sx={{ fontSize: 12 }}>{standaloneHtmlLabel}</MenuItem>}
                {canDownloadCurrentHtml && !offline && <MenuItem disabled sx={{ fontSize: 12 }} title="Quantized export requires the Python backend to repack the current float32 data.">{unavailableStandaloneHtmlLabel}</MenuItem>}
              </Menu>
              {(localExportStatus || exportStatus) && (
                <Typography
                  sx={{
                    ...typography.label,
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
              <Button size="small" sx={compactButton} disabled={!needsReset} onClick={handleResetAll}>Reset</Button>
              <Button size="small" sx={compactButton} onClick={handleCopy}>Copy</Button>
            </Box>
	          </Stack>
	          )}

          {svgPreview ? (
            <Box
              data-show2d-svg-preview="true"
              sx={{
                width: svgPreview.width,
                boxSizing: "content-box",
                bgcolor: "#fff",
                overflowX: "auto",
              }}
            >
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 1,
                  px: 0.75,
                  py: 0.4,
                  bgcolor: themeColors.bgAlt,
                  borderBottom: `1px solid ${themeColors.border}`,
                  boxSizing: "border-box",
                  width: svgPreview.width,
                }}
              >
                <Typography sx={{ ...typography.label, color: themeColors.accent }}>
                  SVG preview
                </Typography>
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                  <Button size="small" sx={compactButton} onClick={() => handleSvgPreview(svgPreview.scale)}>Refresh</Button>
                  <Button size="small" sx={compactButton} onClick={() => { clearSvgPreview(); setLocalExportStatus("Exited SVG preview"); }}>Exit</Button>
                </Box>
              </Box>
              <img
                ref={svgPreviewImageRef}
                data-show2d-svg-preview-image="true"
                src={svgPreview.url}
                alt={`${title || "Show2D"} SVG preview`}
                onLoad={snapSvgPreviewToDevicePixels}
                style={{
                  display: "block",
                  width: svgPreview.width,
                  height: svgPreview.height,
                  maxWidth: "none",
                  objectFit: "contain",
                  marginLeft: svgPreviewSnap.x,
                  marginTop: svgPreviewSnap.y,
                  marginRight: -svgPreviewSnap.x,
                  marginBottom: -svgPreviewSnap.y,
                }}
              />
            </Box>
          ) : isGallery ? (
            /* Gallery mode */
            <Box sx={{
              position: "relative",
              maxWidth: galleryGridWidth,
              width: "100%",
              boxSizing: "border-box",
              p: galleryOuterBorderPx > 0 ? `${galleryOuterBorderPx}px` : 0,
              bgcolor: galleryOuterBorderPx > 0 ? galleryOuterBorderColor : "transparent",
            }}>
            <Box sx={{
              display: "grid",
              gridTemplateColumns: galleryGridColumns,
              gap: `${galleryGapPx}px`,
              width: "100%",
              boxSizing: "border-box",
              justifyContent: "start",
              bgcolor: galleryGapPx > 0 ? galleryGapColorResolved : "transparent",
            }}>
              {visibleImageIndices.map((i) => (
                <Box
                  key={i}
                  sx={{
                    minWidth: 0,
                    cursor: reorderMode ? "grab" : i === selectedIdx ? (overlayEditMode ? (isDraggingEditableDecoration ? "grabbing" : isHoveringOverlay ? "nwse-resize" : "crosshair") : (isDraggingResize || isDraggingResizeInner || isHoveringResize || isHoveringResizeInner) ? "nwse-resize" : isDraggingROI ? "move" : (draggingProfileEndpoint !== null || isDraggingProfileLine) ? "grabbing" : (profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine)) ? "grab" : (profileActive || roiActive || measureActive) ? "crosshair" : "grab") : ("pointer"),
                    opacity: draggedPanelRef.current === i ? 0.62 : 1,
                    transform: dragOverPanel === i ? "translateY(-2px)" : "translateY(0)",
                    transition: "transform 120ms ease, opacity 120ms ease",
                    ...(reorderMode ? {
                      "@keyframes show2d-reorder-jiggle": {
                        "0%": { rotate: "-0.25deg" },
                        "100%": { rotate: "0.25deg" },
                      },
                      animation: "show2d-reorder-jiggle 180ms ease-in-out infinite alternate",
                    } : {}),
                  }}
                >
                  <Box
                    data-show2d-image-panel={i}
                    draggable={reorderMode}
                    onDragStart={(event) => handlePanelDragStart(event, i)}
                    onDragOver={(event) => handlePanelDragOver(event, i)}
                    onDrop={(event) => handlePanelDrop(event, i)}
                    onDragEnd={handlePanelDragEnd}
                    onPointerDown={reorderMode ? (event) => handlePanelReorderPointerDown(event, i) : undefined}
                    onPointerEnter={reorderMode ? (event) => handlePanelReorderPointerEnter(event, i) : undefined}
                    onPointerUp={reorderMode ? (event) => handlePanelReorderPointerUp(event, i) : undefined}
                    ref={(el: HTMLDivElement | null) => { imageContainerRefs.current[i] = el; }}
                    sx={{
                      ...responsivePanelSx,
	                      "&::after": {
	                        content: '""',
	                        position: "absolute",
                        inset: 0,
                        pointerEvents: "none",
	                        zIndex: 5,
	                        boxShadow: `inset 0 0 0 ${selectedPanelSet.has(i) ? 3 : 2}px ${reorderMode && dragOverPanel === i ? themeColors.accent : panelChromeVisible && (i === selectedIdx || selectedPanelSet.has(i)) ? themeColors.accent : "transparent"}`,
	                      },
                      "&::before": {
                        content: '""',
                        position: "absolute",
                        inset: 0,
                        pointerEvents: "none",
                        zIndex: 4,
                        boxShadow: panelInnerBorderPx > 0 ? `inset 0 0 0 ${panelInnerBorderPx}px ${panelInnerBorderColor}` : "none",
                      },
                      "&:hover .show2d-panel-hide-button, &:focus-within .show2d-panel-hide-button": {
                        opacity: 1,
                        pointerEvents: "auto",
                        transform: "translateY(0)",
                      },
                      "&:hover .show2d-panel-star-button, &:focus-within .show2d-panel-star-button": {
                        opacity: 1,
                        pointerEvents: "auto",
                        transform: "translateY(0)",
                      },
                      "@media (hover: none), (pointer: coarse)": {
                        "& .show2d-panel-hide-button": { display: "none" },
                        // no hover on touch: keep the star reachable at all times
                        "& .show2d-panel-star-button": { opacity: 1, pointerEvents: "auto", transform: "translateY(0)" },
                      },
                    }}
                    onMouseDown={reorderMode ? undefined : (e) => handleMouseDown(e, i)}
                    onMouseMove={reorderMode ? undefined : (e) => handleMouseMove(e, i)}
                    onMouseUp={reorderMode ? undefined : (e) => handleMouseUp(e, i)}
                    onMouseLeave={reorderMode ? undefined : () => handleMouseLeave(i)}
                    onWheel={!reorderMode && (i === selectedIdx || linkedZoom) ? (e) => handleWheel(e, i) : undefined}
                    onDoubleClick={reorderMode ? undefined : (e) => handleDoubleClick(e, i)}
                    onTouchStart={reorderMode ? undefined : (e) => handleTouchStart(e, i)}
                    onTouchMove={reorderMode ? undefined : (e) => handleTouchMove(e, i)}
                    onTouchEnd={reorderMode ? undefined : (e) => handleTouchEnd(e, i)}
                    onTouchCancel={reorderMode ? undefined : (e) => handleTouchEnd(e, i)}
                  >
                    {hasPanelMarkers && (markerAround ? (
                      <Box
                        data-show2d-marker-color={panelMarkerColor(i)}
                        data-show2d-marker-style="around"
                        title={`Panel marker ${panelMarkerColor(i)} · ${panelTitleText(i)}`}
                        sx={{
                          position: "absolute",
                          inset: 0,
                          boxSizing: "border-box",
                          boxShadow: `inset 0 0 0 3px ${panelMarkerColor(i)}, inset 0 0 0 5px rgba(0,0,0,0.9)`,
                          pointerEvents: "none",
                          zIndex: 8,
                        }}
                      />
                    ) : (
                      <Box
                        data-show2d-marker-color={panelMarkerColor(i)}
                        data-show2d-marker-style="left"
                        title={`Panel marker ${panelMarkerColor(i)} · ${panelTitleText(i)}`}
                        sx={{
                          position: "absolute",
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: 5,
                          bgcolor: panelMarkerColor(i),
                          boxShadow: "0 0 0 1px rgba(0,0,0,0.45)",
                          pointerEvents: "none",
                          zIndex: 8,
                        }}
                      />
                    ))}
                    <canvas
                      data-quantem-scientific-output={`show2d-image-${i}`}
                      data-show2d-main-canvas={i}
                      ref={(el) => { if (el && canvasRefs.current[i] !== el) { canvasRefs.current[i] = el; setCanvasReady(count => count + 1); } }}
                      width={canvasW} height={canvasH}
                      style={responsiveCanvasStyle}
                    />
                    <canvas
                      ref={(el) => { overlayRefs.current[i] = el; }}
                      width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)}
                      style={responsiveOverlayStyle}
                    />
	                    {panelChromeVisible && cursorInfo && cursorInfo.idx === i && (
                      /* Show4DSTEM readout spec; top-right, dropped 25px below the star button row */
                      <Box sx={{ position: "absolute", top: 28, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right", zIndex: 2 }}>
                        <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                          ({cursorInfo.row}, {cursorInfo.col}){nativePixelSizeForPanel(i) > 0 ? ` = (${(cursorInfo.row * nativePixelSizeForPanel(i)).toFixed(1)}, ${(cursorInfo.col * nativePixelSizeForPanel(i)).toFixed(1)} ${pixelUnit})` : ""} {cursorInfo.rgb ? `(${cursorInfo.rgb[0].toFixed(2)}, ${cursorInfo.rgb[1].toFixed(2)}, ${cursorInfo.rgb[2].toFixed(2)})` : `${formatNumber(cursorInfo.value)}${cursorValueSuffix}`}
                        </Typography>
                      </Box>
                    )}
                    {renderInsetHoverReadout(i)}
                    {renderPanelAnnotations(i)}
                    {showPanelTitles !== false && (
                      <Box
                        data-show2d-panel-title={i}
                        sx={{
                          ...panelTitleChromeSx(panelTitleStyle, {
                          position: "absolute",
                          top: 6,
                          left: 28,
                          right: 28,
                          px: 0.5,
                          color: "rgba(255,255,255,0.95)",
                          fontFamily: SYSTEM_FONT,
                          fontSize: Math.max(8, panelTitleFontSize || 11),
                          fontWeight: 700,
                          lineHeight: 1.2,
                          textAlign: "center",
                          textShadow: "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px rgba(0,0,0,0.75)",
                          pointerEvents: "none",
                          userSelect: "none",
                          whiteSpace: "normal",
                          overflow: "visible",
                          textOverflow: "clip",
                          overflowWrap: "anywhere",
                          zIndex: 2,
                          }),
                        }}
                      >
                        {panelTitleContent(i)}
                      </Box>
                    )}
                    {renderPanelFrameControls(i)}
                    {panelChromeVisible && reorderMode && (
                      <Box
                        sx={{
                          position: "absolute",
                          bottom: 6,
                          left: "50%",
                          transform: "translateX(-50%)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          width: 28,
                          height: 20,
                          borderRadius: 1,
                          bgcolor: "rgba(0,0,0,0.35)",
                          color: "rgba(255,255,255,0.9)",
                          pointerEvents: "none",
                          zIndex: 3,
                        }}
                      >
                        <DragIndicatorIcon sx={{ fontSize: 18 }} />
                      </Box>
                    )}
	                    {panelChromeVisible && (
	                    <IconButton
                      className="show2d-panel-star-button"
                      size="small"
                      onPointerDown={(event) => event.stopPropagation()}
                      onMouseDown={(event) => event.stopPropagation()}
                      onMouseUp={(event) => event.stopPropagation()}
                      onClick={(event) => {
                        event.stopPropagation();
                        togglePanelStar(i);
                      }}
                      title={(starred?.[i] ? "Unstar " : "Star ") + panelLabel(i)}
                      aria-label={(starred?.[i] ? "Unstar " : "Star ") + panelLabel(i)}
                      sx={{
                        position: "absolute",
                        top: 5,
                        right: 5,
                        width: 22,
                        height: 22,
                        p: 0,
                        border: "none",
                        bgcolor: "transparent",
                        cursor: "pointer",
                        fontSize: 18,
                        lineHeight: "20px",
                        textAlign: "center",
                        color: starred?.[i] ? "#ffc107" : "rgba(255,255,255,0.58)",
                        textShadow: "0 0 3px rgba(0,0,0,0.8)",
                        // hidden until the panel is hovered; starred panels stay visible
                        opacity: starred?.[i] ? 1 : 0,
                        pointerEvents: "auto",
                        transform: starred?.[i] ? "translateY(0)" : "translateY(-3px)",
                        transition: "opacity 120ms ease, transform 120ms ease, background-color 120ms ease, color 120ms ease",
                        userSelect: "none",
                        zIndex: 3,
                        "&:hover, &:focus-visible": {
                          bgcolor: "rgba(0,0,0,0.22)",
                          color: starred?.[i] ? "#ffc107" : "rgba(255,255,255,0.9)",
                        },
                      }}
                    >
                      {starred?.[i] ? "★" : "☆"}
	                    </IconButton>
	                    )}
	                    {panelChromeVisible && (
	                    <IconButton
                      className="show2d-panel-hide-button"
                      size="small"
                      disabled={visibleImageCount <= 1}
                      onMouseDown={(event) => event.stopPropagation()}
                      onClick={(event) => {
                        event.stopPropagation();
                        setPanelHidden(i, true);
                      }}
                      aria-label={visibleImageCount <= 1 ? "Cannot hide the last visible panel" : `Hide ${panelLabel(i)}`}
                      title={visibleImageCount <= 1 ? "Cannot hide the last visible panel" : `Hide ${panelLabel(i)}`}
                      sx={{
                        position: "absolute",
                        top: 5,
                        left: 5,
                        width: 22,
                        height: 22,
                        p: 0,
                        opacity: 0,
                        transform: "translateY(-3px)",
                        transition: "opacity 120ms ease, transform 120ms ease, background-color 120ms ease, color 120ms ease",
                        color: visibleImageCount <= 1 ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.75)",
                        bgcolor: "rgba(0,0,0,0.22)",
                        pointerEvents: "none",
                        zIndex: 3,
                        "&:hover, &:focus-visible": {
                          bgcolor: "rgba(0,0,0,0.42)",
                          color: "rgba(255,255,255,0.95)",
                        },
                      }}
                    >
                      <VisibilityOffIcon sx={{ fontSize: 15 }} />
	                    </IconButton>
	                    )}
                    {showResizeControls && (
                      <Box
                        onMouseDown={handleCanvasResizeStart}
                        title="Resize panels"
                        sx={resizeGripSx}
                      />
                    )}
                    {rotationForPanel(i) !== 0 && (
                      <Box
                        onClick={(event: React.MouseEvent) => {
                          event.stopPropagation();
                          setRotationForPanel(i, 0);
                        }}
                        title="Display rotation active; click to clear"
                        aria-label="Clear display rotation"
                        sx={{
                          position: "absolute",
                          left: 43,
                          bottom: 8,
                          zIndex: 4,
                          px: 0.6,
                          py: 0.15,
                          borderRadius: "5px",
                          bgcolor: "rgba(0,0,0,0.50)",
                          color: "rgba(255,255,255,0.95)",
                          fontSize: 10,
                          fontWeight: 700,
                          lineHeight: 1.25,
                          cursor: "pointer",
                          userSelect: "none",
                          textShadow: "0 1px 1px rgba(0,0,0,0.8)",
                          "&:hover": { bgcolor: "rgba(0,0,0,0.72)" },
                        }}
                      >
                        {rotationGlyph(rotationForPanel(i))}
                      </Box>
                    )}
                  </Box>
                  {effectiveShowFft && (
                    <Box
                      ref={(el: HTMLDivElement | null) => { fftContainerRefs.current[i] = el; }}
                      data-show2d-fft-panel={i}
                      sx={{
                        ...responsivePanelSx,
	                        "&::after": {
	                          content: '""',
	                          position: "absolute",
	                          inset: 0,
	                          pointerEvents: "none",
	                          zIndex: 5,
	                          boxShadow: `inset 0 0 0 2px ${panelChromeVisible && i === selectedIdx ? themeColors.accent : "transparent"}`,
	                        },
                        cursor: "grab",
                      }}
                      onWheel={(i === selectedIdx || effectiveFftLinkedZoom) ? (e) => handleGalleryFftWheel(e, i) : undefined}
                      onDoubleClick={() => setGalleryFftState(i, { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 })}
                      onMouseDown={(e) => handleGalleryFftMouseDown(e, i)}
                      onMouseMove={(e) => handleGalleryFftMouseMove(e, i)}
                      onMouseUp={handleGalleryFftMouseUp}
                      onMouseLeave={handleGalleryFftMouseUp}
                      onTouchStart={(e) => handleFftTouchStart(e, i)}
                      onTouchMove={(e) => handleFftTouchMove(e, i)}
                      onTouchEnd={(e) => handleFftTouchEnd(e, i)}
                      onTouchCancel={(e) => handleFftTouchEnd(e, i)}
                    >
                      <canvas
                        data-quantem-scientific-output={`show2d-fft-${i}`}
                        ref={(el) => { fftCanvasRefs.current[i] = el; }}
                        width={canvasW} height={canvasH}
                        style={responsiveCanvasStyle}
                      />
                      {frequencyRingOverlayForPanel(i)}
                      <Box
                        className="quantem-fft-zoom-label"
                        data-show2d-fft-zoom-indicator={i}
                        data-fft-zoom={formatZoomLabel(getGalleryFftState(i).zoom)}
                        aria-label={`FFT zoom for ${panelLabel(i)}: ${formatZoomLabel(getGalleryFftState(i).zoom)}`}
                        sx={{
                          position: "absolute",
                          left: 12,
                          bottom: 7,
                          color: "white",
                          fontFamily: SYSTEM_FONT,
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
                        {formatZoomLabel(getGalleryFftState(i).zoom)}
                      </Box>
                      {(showPanelTitles !== false || (fftMetricsEnabled && galleryFftQuality[i])) && (
                        <Box
                          className="quantem-fft-panel-label-stack"
                          sx={{
                            position: "absolute",
                            top: 6,
                            left: 8,
                            right: 8,
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "flex-start",
                            rowGap: "2px",
                            minWidth: 0,
                            maxWidth: "calc(100% - 16px)",
                            overflow: "hidden",
                            pointerEvents: "none",
                            userSelect: "none",
                            zIndex: 6,
                          }}
                        >
                          {showPanelTitles !== false && (
                            <Box
                              className="quantem-fft-panel-title"
                              data-show2d-panel-title={i}
                              sx={{
                                ...panelTitleChromeSx(panelTitleStyle, {
                                px: 0.5,
                                minWidth: 0,
                                color: "rgba(255,255,255,0.95)",
                                fontFamily: SYSTEM_FONT,
                                fontSize: Math.max(8, panelTitleFontSize || 11),
                                fontWeight: 700,
                                lineHeight: 1.2,
                                textAlign: "left",
                                textShadow: "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px rgba(0,0,0,0.75)",
                                bgcolor: "rgba(0,0,0,0.42)",
                                borderRadius: "3px",
                                maxWidth: "100%",
                                boxSizing: "border-box",
                                whiteSpace: "nowrap",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                }),
                              }}
                            >
                              FFT · {panelTitleContent(i)}
                            </Box>
                          )}
                          {fftMetricsEnabled && galleryFftQuality[i] && (
                            <Box
                              className="quantem-fft-quality-label"
                              aria-label={`FFT quality for ${panelLabel(i)}: ${formatFftQualityLabel(galleryFftQuality[i])}`}
                              sx={{
                                px: 0.5,
                                py: 0.15,
                                minWidth: 0,
                                maxWidth: "100%",
                                boxSizing: "border-box",
                                color: "rgba(255,255,255,0.96)",
                                bgcolor: "rgba(0,0,0,0.58)",
                                borderRadius: "3px",
                                fontFamily: SYSTEM_FONT,
                                fontSize: Math.max(9, Math.min(12, panelTitleFontSize || 11)),
                                fontWeight: 700,
                                lineHeight: 1.2,
                                whiteSpace: "nowrap",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                textShadow: "1px 1px 0 rgba(0,0,0,0.9), 0 0 3px rgba(0,0,0,0.85)",
                                alignSelf: "flex-start",
                              }}
                            >
                              {formatFftQualityLabel(galleryFftQuality[i])}
                            </Box>
                          )}
                        </Box>
                      )}
                      {fftComputing && !fftOffscreensRef.current[i] && (
                        <Box sx={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "rgba(0,0,0,0.6)", pointerEvents: "none" }}>
                          <Typography sx={{ fontSize: 10, color: "#aaa", fontFamily: "monospace", "@keyframes pulse": { "0%,100%": { opacity: 0.4 }, "50%": { opacity: 1 } }, animation: "pulse 1.2s ease-in-out infinite" }}>FFT…</Typography>
                        </Box>
                      )}
                      {showResizeControls && (
                        <Box
                          onMouseDown={(e: React.MouseEvent) => {
                            e.stopPropagation();
                            handleCanvasResizeStart(e);
                          }}
                          title="Resize FFT panels"
                          aria-label={`Resize FFT panel for ${panelLabel(i)}`}
                          data-show2d-fft-resize-handle={i}
                          sx={{ ...resizeGripSx, zIndex: 7 }}
                        />
                      )}
                    </Box>
                  )}
                </Box>
              ))}
              {showDiffPanel && diffOtherIndices.map((otherIdx, slot) => (
                <Box key={`diff_${slot}`} sx={{ minWidth: 0 }}>
                  <Box sx={{ ...responsivePanelSx, border: `2px solid ${themeColors.border}` }}>
                    <canvas
                      ref={(el) => { diffCanvasRefs.current[slot] = el; }}
                      width={canvasW} height={canvasH}
                      style={responsiveCanvasStyle}
                    />
                  </Box>
                  <Typography sx={{ fontSize: 10, color: themeColors.textMuted, textAlign: "center", mt: 0.25 }}>
                    {visibleGrayscaleIndices.length === 2
                      ? `Diff (${panelLabel(effectiveDiffReference)} − ${panelLabel(otherIdx)})`
                      : `Diff (#${effectiveDiffReference + 1} − #${otherIdx + 1})`}
                  </Typography>
                  {/* FFT of one visible diff pair */}
                  {effectiveShowFft && diffOtherIndices.length === 1 && slot === 0 && (
                    <Box sx={{ mt: 0, ...responsivePanelSx, border: `2px solid ${themeColors.border}` }}>
                      <canvas
                        ref={(el) => { diffFftCanvasRef.current = el; }}
                        width={canvasW} height={canvasH}
                        style={responsiveCanvasStyle}
                      />
                    </Box>
                  )}
                </Box>
              ))}
            </Box>
            {groupMarkerOverlays.map((marker) => (
              <Box
                key={marker.key}
                data-show2d-row-marker={marker.axis === "row" ? marker.key.slice(4) : undefined}
                data-show2d-col-marker={marker.axis === "col" ? marker.key.slice(4) : undefined}
                data-show2d-group-marker-color={marker.color}
                sx={{
                  position: "absolute",
                  left: marker.left,
                  top: marker.top,
                  width: marker.width,
                  height: marker.height,
                  boxSizing: "border-box",
                  boxShadow: `inset 0 0 0 3px ${marker.color}, inset 0 0 0 5px rgba(0,0,0,0.9)`,
                  pointerEvents: "none",
                  zIndex: 10,
                }}
              />
            ))}
            </Box>
          ) : (
            /* Single image mode */
            <Box
              ref={(el: HTMLDivElement | null) => { imageContainerRefs.current[0] = el; }}
              sx={{ ...responsivePanelSx, border: `1px solid ${themeColors.border}`, cursor: overlayEditMode ? (isDraggingEditableDecoration ? "grabbing" : isHoveringOverlay ? "nwse-resize" : "crosshair") : isHoveringLensEdge ? "nwse-resize" : isDraggingROI ? "move" : (isDraggingResize || isDraggingResizeInner || isHoveringResize || isHoveringResizeInner) ? "nwse-resize" : (draggingProfileEndpoint !== null || isDraggingProfileLine) ? "grabbing" : (profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine)) ? "grab" : (profileActive || roiActive || measureActive) ? "crosshair" : "grab" }}
              onMouseDown={(e) => handleMouseDown(e, 0)}
              onMouseMove={(e) => handleMouseMove(e, 0)}
              onMouseUp={(e) => handleMouseUp(e, 0)}
              onMouseLeave={() => handleMouseLeave(0)}
              onWheel={(e) => handleWheel(e, 0)}
              onDoubleClick={(e) => handleDoubleClick(e, 0)}
              onTouchStart={(e) => handleTouchStart(e, 0)}
              onTouchMove={(e) => handleTouchMove(e, 0)}
              onTouchEnd={(e) => handleTouchEnd(e, 0)}
              onTouchCancel={(e) => handleTouchEnd(e, 0)}
            >
            <canvas
              data-quantem-scientific-output="show2d-image-0"
              data-show2d-main-canvas={0}
              ref={(el) => { if (el && canvasRefs.current[0] !== el) { canvasRefs.current[0] = el; setCanvasReady(count => count + 1); } }}
                width={canvasW} height={canvasH}
                style={responsiveCanvasStyle}
              />
              <canvas
                ref={(el) => { overlayRefs.current[0] = el; }}
                width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)}
                style={responsiveOverlayStyle}
              />
              <canvas
                ref={lensCanvasRef}
                width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)}
                style={responsiveOverlayStyle}
              />
	              {panelChromeVisible && cursorInfo && (
                /* Show4DSTEM readout spec verbatim (single-image mode has no star button) */
                <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                  <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                    ({cursorInfo.row}, {cursorInfo.col}){nativePixelSize > 0 ? ` = (${(cursorInfo.row * nativePixelSize).toFixed(1)}, ${(cursorInfo.col * nativePixelSize).toFixed(1)} ${calibratedUnit})` : ""} {cursorInfo.rgb ? `(${cursorInfo.rgb[0].toFixed(2)}, ${cursorInfo.rgb[1].toFixed(2)}, ${cursorInfo.rgb[2].toFixed(2)})` : `${formatNumber(cursorInfo.value)}${cursorValueSuffix}`}
                  </Typography>
	                </Box>
	              )}
              {renderInsetHoverReadout(0)}
              {renderPanelAnnotations(0)}
              {renderPanelFrameControls(0)}
              {showResizeControls && (
                <Box onMouseDown={handleCanvasResizeStart} title="Resize image" sx={resizeGripSx} />
              )}
            </Box>
          )}

          {/* Stats bar - right below canvas (Show3D style) */}
          {showStats && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap", maxWidth: "100%", boxSizing: "border-box", opacity: 1 }}>
              {isGallery && (
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>{panelTitleContent(statsIdx)}</Typography>
              )}
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(currentFrameStats?.mean ?? statsMean?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(currentFrameStats?.min ?? statsMin?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(currentFrameStats?.max ?? statsMax?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(currentFrameStats?.std ?? statsStd?.[statsIdx] ?? 0)}</Box></Typography>
              {measureActive && (
                <>
                  <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                  <Typography sx={{ fontSize: 11, color: "#fff", fontWeight: "bold" }}>Measuring</Typography>
                </>
              )}
            </Box>
          )}

          {/* Line profile sparkline: always reserve space when profile is active */}
          {profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, width: "100%", maxWidth: profileCanvasWidth, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                onMouseMove={handleProfileMouseMove}
                onMouseLeave={handleProfileMouseLeave}
                style={{ width: "100%", height: profileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
              />
              {showResizeControls && (
                <div
                  onMouseDown={(e) => {
                    e.preventDefault();
                    setIsResizingProfile(true);
                    setProfileResizeStart({ y: e.clientY, height: profileHeight });
                  }}
                  style={{ width: "100%", height: 4, cursor: "ns-resize", borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, background: `linear-gradient(to bottom, ${themeColors.border}, transparent)`, opacity: 1, pointerEvents: "auto" }}
                />
              )}
            </Box>
          )}

          {/* Controls: two rows left + histogram right, ROI below */}
	          {controlsVisible && (
            <Box sx={{ mt: (effectiveShowFft && isGallery) ? `${SPACING.XS}px` : `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, boxSizing: "border-box" }}>
              {/* Top: control rows + histogram side by side */}
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: `${SPACING.SM}px`, width: "100%", maxWidth: galleryGridWidth, minWidth: 0, boxSizing: "border-box" }}>
                <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: "1 1 260px", minWidth: 0, justifyContent: "flex-start" }}>
                  {/* Row 1: Scale + Color */}
                  {(
                    <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                      <Box sx={controlPairSx}>
                        <Typography sx={compactLabelSx}>Scale</Typography>
                        <Select value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45 }} MenuProps={themedMenuProps}>
                          <MenuItem value="linear">Lin</MenuItem>
                          <MenuItem value="log">Log</MenuItem>
                        </Select>
                      </Box>
                      <Box sx={controlPairSx}>
                        <Typography sx={compactLabelSx}>Color</Typography>
                        <Select
                          size="small"
                          value={selectedCmap}
                          onChange={(e) => setSelectedCmap(String(e.target.value))}
                          MenuProps={themedMenuProps}
                          sx={{ ...themedSelect, minWidth: 60 }}
                          inputProps={{ "aria-label": colorShared ? "Shared colormap for all panels" : "Selected panel colormap" }}
                        >
                          {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                        </Select>
                      </Box>
                      {!isGallery && (
                        <Box sx={controlPairSx}>
                          <Typography sx={compactLabelSx}>Colorbar</Typography>
                          <Switch checked={showColorbar} onChange={() => { setShowColorbar(!showColorbar); }} size="small" sx={switchStyles.small} />
                        </Box>
                      )}
                    </Box>
                  )}
                  {/* Row 2: Auto + Smooth + Link toggles (gallery) + zoom indicator */}
                  {(
                    <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                      <Box sx={controlPairSx}>
                        <Typography sx={compactLabelSx} title="Auto-contrast: recompute the display range from the current view's percentiles. Turn off to set the histogram range by hand.">Auto</Typography>
                        <Switch
                          checked={autoContrast}
                          onChange={(event) => {
                            setAutoContrast(event.target.checked);
                            if (event.target.checked) setContrastPreset("manual");
                          }}
                          size="small"
                          sx={switchStyles.small}
                        />
                      </Box>
                      <Box sx={controlPairSx}>
                        <Typography sx={compactLabelSx} title="CSS bilinear interpolation. Same data, the browser smooths visually, useful when upscaling small images on a large canvas.">Smooth</Typography>
                        <Switch checked={smooth} onChange={() => { setSmooth(!smooth); }} size="small" sx={switchStyles.small} />
                      </Box>
                      {!showDenoise && filterBannerText && (
                        /* House rule: an active reduction is never invisible,
                           even with the denoise controls row hidden. */
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent }} title={filterBannerText}>
                          {filterBannerText.split(" (")[0]}
                        </Typography>
                      )}
                      {isGallery && (
                        /* All link toggles live in one bordered "Link" sub-group so
                           the governing word stays with Zoom/Pan/Contrast/Denoise
                           when the row wraps on a narrow viewport. */
                        <Box sx={controlSubGroupSx}>
                          <Typography sx={compactLabelSx}>Link</Typography>
                          <Box sx={controlPairSx}>
                            <Typography sx={compactLabelSx} title="Zoom together across panels.">Zoom</Typography>
                            <Switch checked={linkedZoom} onChange={() => { setLinkedZoom(!linkedZoom); }} size="small" sx={switchStyles.small} />
                          </Box>
                          <Box sx={controlPairSx}>
                            <Typography sx={compactLabelSx} title="Pan together (independent of zoom).">Pan</Typography>
                            <Switch checked={linkPan} onChange={() => { setLinkPan(!linkPan); }} size="small" sx={switchStyles.small} />
                          </Box>
                          <Box sx={controlPairSx}>
                            <Typography sx={compactLabelSx} title="Share contrast slider across panels.">Contrast</Typography>
                            <Switch checked={linkedContrast} onChange={() => { setLinkedContrast(!linkedContrast); }} size="small" sx={switchStyles.small} />
                          </Box>
                          <Box sx={controlPairSx}>
                            <Typography sx={compactLabelSx} title="Linked: denoise edits apply to every panel. Unlinked: edits apply to the selected panel only.">Denoise</Typography>
                            <Switch checked={denoiseScopeAll} onChange={() => setDenoiseScope(denoiseScopeAll ? "panel" : "all")} size="small" sx={switchStyles.small} />
                          </Box>
                          <Box sx={controlPairSx}>
                            <Typography sx={compactLabelSx} title="Linked: frequency-filter edits apply to every panel. Unlinked: edits apply to the selected panel only.">Filter edits</Typography>
                            <Switch checked={frequencyFilterScopeAll} onChange={() => setFrequencyFilterScope(frequencyFilterScopeAll ? "panel" : "all")} size="small" sx={switchStyles.small} />
                          </Box>
                        </Box>
                      )}
                      {selectedView.zoom !== 1 && (
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold" }}>{selectedView.zoom.toFixed(1)}x</Typography>
                      )}
                    </Box>
                  )}
                  {/* Lens row (toggle-gated): magnifier strength + window size on
                      their own line so Row 2 stays clean, mirroring the denoise row. */}
                  {!isGallery && showLens && (
                    <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                      <Box sx={controlPairSx}>
                        <Typography sx={compactLabelSx} title="Magnifier zoom factor.">Lens {lensMag}×</Typography>
                        <Slider value={lensMag} min={2} max={8} step={1} onChange={(_, value) => setLensMag(value as number)} size="small" sx={{ ...sliderStyles.small, width: 60 }} aria-label="Lens magnification" />
                      </Box>
                      <Box sx={controlPairSx}>
                        <Typography sx={compactLabelSx} title="Magnifier window size in display pixels.">Size {lensDisplaySize}px</Typography>
                        <Slider value={lensDisplaySize} min={64} max={256} step={16} onChange={(_, value) => setLensDisplaySize(value as number)} size="small" sx={{ ...sliderStyles.small, width: 60 }} aria-label="Lens window size" />
                      </Box>
                    </Box>
                  )}
                  {/* Row 3 (toggle-gated): display-only denoise for sparse maps (EDS, low dose) */}
                  {showDenoise && (
                    <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                      <Box sx={controlPairSx}>
                        <Typography sx={compactLabelSx} title="Poisson (Anscombe): count-respecting smoothing for sparse EDS/counting data - recommended with Bin 2, sigma 6-10. Gaussian: simple smooth for decent-dose images. None: raw counts (use for anything quantitative).">Denoise</Typography>
                        <Select size="small" value={denoiseBaseMode} onChange={(e) => { const mode = e.target.value; setDisplayFilter(mode); mirrorFilterKnobEdit("mode", mode); if (normalizeFilterMode(mode) !== "none" || (spatialBin || 1) > 1) setDenoiseEnabled(true); }} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 88 }}>
                          {[["none", "None"], ["gaussian", "Gaussian"], ["anscombe", "Poisson (Anscombe)"]].map(([mode, label]) => (
                            <MenuItem key={mode} value={mode}>{label}</MenuItem>
                          ))}
                        </Select>
                      </Box>
                      <Box sx={controlPairSx}>
                        <Typography sx={{ ...compactLabelSx, minWidth: 40, display: "inline-block" }}>σ {(sigmaDraft ?? Number(displaySigma ?? 4)).toFixed(1)}</Typography>
                        <Slider
                          value={sigmaDraft ?? Number(displaySigma ?? 4)}
                          min={0} max={20} step={0.5}
                          onChange={(_, value) => { if (denoiseBaseMode === "none") { setDisplayFilter("gaussian"); mirrorFilterKnobEdit("mode", "gaussian"); } setSigmaDraftDuringDrag(value as number); }}
                          onChangeCommitted={(_, value) => { setDisplaySigma(value as number); mirrorFilterKnobEdit("sigma", value as number); setSigmaDraft(null); setSigmaFilterDraft(null); sigmaFilterDraftPendingRef.current = null; if (denoiseBaseMode === "none") { setDisplayFilter("gaussian"); mirrorFilterKnobEdit("mode", "gaussian"); } setDenoiseEnabled(true); }}
                          size="small" sx={{ ...sliderStyles.small, width: 60 }}
                        />
                      </Box>
                      <Box sx={controlPairSx}>
                        <Typography sx={compactLabelSx} title="Display-side 2x bin passes for SNR, combined with the denoise method. 1 is lossless.">Bin</Typography>
                        <Select size="small" value={String(spatialBin || 1)} onChange={(e) => { const bin = parseInt(e.target.value, 10); setSpatialBin(bin); mirrorFilterKnobEdit("bin", bin); if (bin > 1 || normalizeFilterMode(denoiseBaseMode) !== "none") setDenoiseEnabled(true); }} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 40 }}>
                          {[1, 2, 4].map((bin) => (<MenuItem key={bin} value={String(bin)}>{bin}</MenuItem>))}
                        </Select>
                      </Box>
                    </Box>
                  )}
                  {showFrequencyFilter && (
                    <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                      <Box sx={controlPairSx}>
                        <Typography sx={compactLabelSx} title="Low-pass removes fine detail; High-pass removes slow background; Band-pass isolates a periodicity.">Filter</Typography>
                        <Select size="small" value={frequencyUiKnobs.mode} onChange={(event) => { const mode = String(event.target.value); setFrequencyFilter(mode); mirrorFrequencyKnobEdit("mode", mode); if (mode !== "none") setFrequencyFilterEnabled(true); }} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 84 }}>
                          <MenuItem value="none">None</MenuItem>
                          <MenuItem value="lowpass">Low-pass</MenuItem>
                          <MenuItem value="highpass">High-pass</MenuItem>
                          <MenuItem value="bandpass">Band-pass</MenuItem>
                        </Select>
                      </Box>
                      {frequencyUiKnobs.mode === "bandpass" ? (
                        <>
                          <Box sx={controlPairSx}>
                            <Typography sx={{ ...compactLabelSx, minWidth: 84, display: "inline-block" }}>Center {frequencyValueLabel(frequencyUiKnobs.center)}</Typography>
                            <Slider value={frequencyUiKnobs.center} min={0} max={1} step={0.005} onChange={(_, value) => setFrequencyDraft(value as number)} onChangeCommitted={(_, value) => { setFrequencyFilterCenter(value as number); mirrorFrequencyKnobEdit("center", value as number); setFrequencyDraft(null); }} size="small" sx={{ ...sliderStyles.small, width: 72 }} aria-label="Band-pass center as fraction of Nyquist" />
                          </Box>
                          <Box sx={controlPairSx}>
                            <Typography sx={{ ...compactLabelSx, minWidth: 80, display: "inline-block" }}>Width {frequencyValueLabel(frequencyUiKnobs.width)}</Typography>
                            <Slider value={frequencyUiKnobs.width} min={0.01} max={1} step={0.005} onChange={(_, value) => { setFrequencyFilterWidth(value as number); mirrorFrequencyKnobEdit("width", value as number); }} size="small" sx={{ ...sliderStyles.small, width: 72 }} aria-label="Band-pass width as fraction of Nyquist" />
                          </Box>
                        </>
                      ) : (
                        <Box sx={controlPairSx}>
                          <Typography sx={{ ...compactLabelSx, minWidth: 84, display: "inline-block" }}>Cutoff {frequencyValueLabel(frequencyUiKnobs.cutoff)}</Typography>
                          <Slider value={frequencyUiKnobs.cutoff} min={0} max={1} step={0.005} disabled={!frequencyFilterActive(frequencyUiKnobs.mode)} onChange={(_, value) => setFrequencyDraft(value as number)} onChangeCommitted={(_, value) => { setFrequencyFilterCutoff(value as number); mirrorFrequencyKnobEdit("cutoff", value as number); setFrequencyDraft(null); }} size="small" sx={{ ...sliderStyles.small, width: 72 }} aria-label="Frequency cutoff as fraction of Nyquist" />
                        </Box>
                      )}
                    </Box>
                  )}
                </Box>
                {/* Right: histograms. Unlinked + gallery → grid matching gallery layout
                    (same effectiveNcols × rows). Linked or single image → one histogram. */}
                {(imageHistogramData || imageHistogramBins || (isGallery && !linkedContrast && rawDataRef.current)) && (
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", justifyContent: "flex-start", gap: 0.5, flex: "0 1 auto", maxWidth: "100%", opacity: 1, pointerEvents: "auto" }}>
                    {(!linkedContrast && isGallery && rawDataRef.current) ? (
                      <Box sx={{ display: "grid", gridTemplateColumns: histogramGridColumns, gap: `${histogramGapPx}px`, width: "100%", maxWidth: histogramGridMaxWidth, justifyContent: "start" }}>
                        {/* Only the panels on screen get a histogram: the current
                            page's slice when paged, hidden panels skipped. */}
                        {visibleImageIndices.map((i) => {
                          const contrast = contrastStates.get(i) || { vminPct: 0, vmaxPct: 100 };
                          const raw = rawDataRef.current?.[i] || null;
                          const histData = raw && logScale ? applyLogScale(raw) : raw;
                          const histRange = histData ? findDataRange(histData) : (dataRangesRef.current[i] || imageDataRange);
                          return (
                            /* RGB panels bypass the contrast pipeline: grey out their histogram */
                            <Box key={i} sx={isRgbPanel(i) ? { opacity: 0.35, pointerEvents: "none" } : undefined}
                              title={isRgbPanel(i) ? "RGB panel: contrast controls do not apply" : undefined}>
                              <Histogram data={histData} vminPct={contrast.vminPct} vmaxPct={contrast.vmaxPct}
                                {...histogramRangeHandlers(i)}
                                width={110} height={58} theme={themeInfo.theme === "dark" ? "dark" : "light"}
                                dataMin={histRange?.min ?? imageDataRange.min}
                                dataMax={histRange?.max ?? imageDataRange.max} />
                            </Box>
                          );
                        })}
                      </Box>
                    ) : (
                      <Histogram bins={imageHistogramBins} vminPct={imageVminPct} vmaxPct={imageVmaxPct} {...histogramRangeHandlers(activeContrastIdx)} width={110} height={58} theme={themeInfo.theme === "dark" ? "dark" : "light"} dataMin={traitVmin != null && traitVmax != null ? displayValue(traitVmin, logScale) : imageDataRange.min} dataMax={traitVmin != null && traitVmax != null ? displayValue(traitVmax, logScale) : imageDataRange.max} />
                    )}
                  </Box>
                )}
              </Box>
              {/* ROI Section (own box, below control rows) */}
              {roiActive && (
                <Box sx={{ border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, px: 1, py: 0.5, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, opacity: 1, pointerEvents: "auto" }}>
                  {/* ROI shape and actions */}
                  <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>ROI</Typography>
                    <Select
                      size="small"
                      value={newRoiShape}
                      onChange={(e) => setNewRoiShape(e.target.value as "circle" | "square" | "rectangle" | "annular")}
                      MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                    >
                      {(["square", "rectangle", "circle", "annular"] as const).map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                    </Select>
                    <Button size="small" sx={compactButton} onClick={() => {
                      const defaultRadius = Math.max(10, Math.round(Math.min(width, height) * 0.05));
                      const newRoi: ROIItem = { row: Math.floor(height / 2), col: Math.floor(width / 2), shape: newRoiShape, radius: defaultRadius, radius_inner: Math.max(5, Math.round(defaultRadius * 0.5)), width: defaultRadius * 2, height: defaultRadius * 2, color: ROI_COLORS[(roiList?.length ?? 0) % ROI_COLORS.length], line_width: 2, highlight: false };
                      const newList = [...(roiList || []), newRoi];
                      setRoiList(newList);
                      setRoiSelectedIdx(newList.length - 1);
                    }}>Add</Button>
                    <Box sx={{ flex: 1 }} />
                    <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 24, color: "#ef5350" }} disabled={!roiList?.length} onClick={() => { setRoiList([]); setRoiSelectedIdx(-1); }}>Clear</Button>
                  </Box>
                  {/* Selected ROI details */}
                  {selectedRoi && (
                    <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: `${SPACING.SM}px`, borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: selectedRoi.color }}>#{roiSelectedIdx + 1}/{roiList?.length ?? 0}</Typography>
                      <Select
                        size="small"
                        value={selectedRoi.shape || "circle"}
                        onChange={(e) => updateSelectedRoi({ shape: e.target.value })}
                        MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                      >
                        {(["square", "rectangle", "circle", "annular"] as const).map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                      </Select>
                      {selectedRoi.shape === "rectangle" && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>W</Typography>
                          <Slider value={selectedRoi.width} min={5} max={width} onChange={(_, value) => updateSelectedRoi({ width: value as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>H</Typography>
                          <Slider value={selectedRoi.height} min={5} max={height} onChange={(_, value) => updateSelectedRoi({ height: value as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                        </>
                      )}
                      {selectedRoi.shape === "annular" && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Inner</Typography>
                          <Slider value={selectedRoi.radius_inner} min={1} max={selectedRoi.radius - 1} onChange={(_, value) => updateSelectedRoi({ radius_inner: value as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Outer</Typography>
                          <Slider value={selectedRoi.radius} min={selectedRoi.radius_inner + 1} max={Math.max(width, height)} onChange={(_, value) => updateSelectedRoi({ radius: value as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                        </>
                      )}
                      {selectedRoi.shape !== "rectangle" && selectedRoi.shape !== "annular" && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Size</Typography>
                          <Slider value={selectedRoi.radius} min={5} max={Math.max(width, height)} onChange={(_, value) => updateSelectedRoi({ radius: value as number })} size="small" sx={{ ...sliderStyles.small, width: 50 }} />
                        </>
                      )}
                      <Box sx={{ display: "flex", gap: "2px" }}>
                        {ROI_COLORS.map(swatch => (
                          <Box key={swatch} onClick={() => updateSelectedRoi({ color: swatch })} sx={{ width: 12, height: 12, bgcolor: swatch, cursor: "pointer", border: swatch === selectedRoi.color ? `2px solid ${themeColors.text}` : "1px solid transparent", "&:hover": { opacity: 0.8 } }} />
                        ))}
                      </Box>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Border</Typography>
                      <Slider value={selectedRoi.line_width} min={1} max={6} step={1} onChange={(_, value) => updateSelectedRoi({ line_width: value as number })} size="small" sx={{ ...sliderStyles.small, width: 30 }} />
                      <Box
                        onClick={() => updateSelectedRoi({ highlight: !selectedRoi.highlight })}
                        sx={{ cursor: "pointer", fontSize: 10, color: selectedRoi.highlight ? themeColors.accentGreen : themeColors.textMuted, "&:hover": { opacity: 0.8 } }}
                        title="Focus (dim outside)"
                      >{selectedRoi.highlight ? "\u25C9 Focus" : "\u25CB Focus"}</Box>
                      <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 20, color: "#ef5350" }} onClick={() => {
                        const newList = roiList!.filter((_, j) => j !== roiSelectedIdx);
                        setRoiList(newList);
                        setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1);
                      }}>&times;</Button>
                    </Box>
                  )}
                  {/* ROI list */}
                  {roiList && roiList.length > 0 && (
                    <Box sx={{ display: "flex", flexDirection: "column", borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                      {roiList.map((roi, i) => {
                        const roiColor = roi.color || ROI_COLORS[i % ROI_COLORS.length];
                        const isSelected = i === roiSelectedIdx;
                        const shapeLabel = roi.shape === "rectangle" ? `${roi.width}×${roi.height}` : roi.shape === "annular" ? `r${roi.radius_inner}-${roi.radius}` : `r${roi.radius}`;
                        return (
                          <Box key={i} onClick={() => setRoiSelectedIdx(i)} sx={{ display: "flex", alignItems: "center", gap: "3px", lineHeight: 1.6, cursor: "pointer", "&:hover .roi-delete": { opacity: 1 } }}>
                            <Box sx={{ width: 8, height: 8, borderRadius: roi.shape === "square" || roi.shape === "rectangle" ? 0 : "50%", bgcolor: roiColor, border: isSelected ? "2px solid #fff" : "1px solid transparent", flexShrink: 0 }} />
                            <Typography component="span" sx={{ fontSize: 10, fontFamily: "monospace", color: isSelected ? themeColors.text : themeColors.textMuted, fontWeight: isSelected ? "bold" : "normal" }}>
                              <Box component="span" sx={{ color: roiColor }}>{i + 1}</Box>{" "}
                              {roi.shape} ({roi.row}, {roi.col}) {shapeLabel}
                            </Typography>
                            <Box
                              onClick={(e) => { e.stopPropagation(); const newList = roiList.map((item, j) => ({ ...item, highlight: j === i ? !item.highlight : false })); setRoiList(newList); }}
                              sx={{ cursor: "pointer", fontSize: 10, color: roi.highlight ? themeColors.accentGreen : themeColors.textMuted, lineHeight: 1, opacity: roi.highlight ? 1 : 0.5, "&:hover": { opacity: 1 } }}
                              title="Focus (dim outside)"
                            >{roi.highlight ? "\u25C9" : "\u25CB"}</Box>
                            <Box
                              className="roi-delete"
                              onClick={(e) => { e.stopPropagation(); const newList = roiList.filter((_, j) => j !== i); setRoiList(newList); setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1); }}
                              sx={{ opacity: 0, cursor: "pointer", fontSize: 10, color: themeColors.textMuted, ml: 0.5, lineHeight: 1, "&:hover": { color: "#f44336" } }}
                            >&times;</Box>
                          </Box>
                        );
                      })}
                    </Box>
                  )}
                </Box>
              )}
            </Box>
          )}

          {/* Gallery FFT Controls - below regular image controls */}
	          {controlsVisible && effectiveShowFft && isGallery && (
            <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexWrap: "wrap", gap: `${SPACING.SM}px`, width: "100%", maxWidth: galleryGridWidth, minWidth: 0, boxSizing: "border-box" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: "1 1 260px", minWidth: 0, justifyContent: "flex-start" }}>
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT Scale</Typography>
                  <Select value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  {roiFftActive && fftCropDims && (
                    <>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Win</Typography>
                      <Switch checked={fftWindow} onChange={(e) => { setFftWindow(e.target.checked); }} size="small" sx={switchStyles.small} />
                    </>
                  )}
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Color</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                </Box>
                {/* FFT Row 2: Auto + Smooth + Link Zoom/Pan/Contrast (mirrors main image Row 2) */}
                <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                  <Box sx={controlPairSx}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto</Typography>
                    <Switch checked={fftAuto} onChange={(e) => { setFftAuto(e.target.checked); }} size="small" sx={switchStyles.small} />
                  </Box>
                  <Box sx={controlPairSx}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }} title="CSS bilinear interpolation on the FFT canvas.">Smooth</Typography>
                    <Switch checked={fftSmooth} onChange={(e) => { setFftSmooth(e.target.checked); }} size="small" sx={switchStyles.small} />
                  </Box>
                  {isGallery && (
                    <>
                      <Box sx={controlPairSx}>
                        <Typography sx={{ ...typography.label, fontSize: 10 }}>Link</Typography>
                        <Typography sx={{ ...typography.label, fontSize: 10 }} title="Zoom together across image and FFT panels.">Zoom</Typography>
                        <Switch checked={effectiveFftLinkedZoom} onChange={() => { setLinkedZoom(!linkedZoom); }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Link FFT zoom across panels" } }} />
                      </Box>
                      <Box sx={controlPairSx}>
                        <Typography sx={{ ...typography.label, fontSize: 10 }} title="Pan image and FFT panels together.">Pan</Typography>
                        <Switch checked={effectiveFftLinkPan} onChange={() => { setLinkPan(!linkPan); }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Link FFT pan across panels" } }} />
                      </Box>
                      <Box sx={controlPairSx}>
                        <Typography sx={{ ...typography.label, fontSize: 10 }} title="Share image and FFT contrast sliders across panels.">Contrast</Typography>
                        <Switch checked={effectiveFftLinkedContrast} onChange={() => { setLinkedContrast(!linkedContrast); }} size="small" sx={switchStyles.small} slotProps={{ input: { "aria-label": "Link FFT contrast across panels" } }} />
                      </Box>
                    </>
                  )}
                </Box>
              </Box>
              {(
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-start", justifyContent: "center", flex: "0 1 auto", maxWidth: "100%", opacity: 1, pointerEvents: "auto" }}>
                  {fftHistogramData && (
                    !effectiveFftLinkedContrast && isGallery ? (
                      <Box sx={{ display: "grid", gridTemplateColumns: histogramGridColumns, gap: `${histogramGapPx}px`, width: "100%", maxWidth: histogramGridMaxWidth, justifyContent: "start" }}>
                        {/* Match the image-histogram grid: current page only, hidden panels skipped. */}
                        {visibleImageIndices.map((i) => {
                          const fftContrast = fftContrastFor(i);
                          const cache = galleryFftPipelineRef.current[i];
                          const range = cache ? { min: cache.displayMin, max: cache.displayMax } : fftDataRange;
                          return (
                            <Histogram
                              key={i}
                              data={cache?.displayData ?? fftHistogramData}
                              pinBinsToRange
                              vminPct={fftContrast.vminPct} vmaxPct={fftContrast.vmaxPct}
                              onRangeChange={(min, max) => { setFftContrastFor(i, { vminPct: min, vmaxPct: max }); }}
                              width={110} height={58}
                              theme={themeInfo.theme === "dark" ? "dark" : "light"}
                              dataMin={range.min} dataMax={range.max}
                            />
                          );
                        })}
                      </Box>
                    ) : (() => {
                      const fftContrast = fftContrastFor(selectedIdx);
                      return (
                        <Histogram
                          data={fftHistogramData}
                          pinBinsToRange
                          vminPct={fftContrast.vminPct}
                          vmaxPct={fftContrast.vmaxPct}
                          onRangeChange={(min, max) => { setFftContrastFor(selectedIdx, { vminPct: min, vmaxPct: max }); }}
                          width={110} height={58}
                          theme={themeInfo.theme === "dark" ? "dark" : "light"}
                          dataMin={fftDataRange.min} dataMax={fftDataRange.max}
                        />
                      );
                    })()
                  )}
                </Box>
              )}
            </Box>
          )}
        </Box>

        {/* FFT Panel - canvas + stats (single mode only) */}
        {effectiveShowFft && !isGallery && (
          <Box sx={{ ...responsivePanelWidthSx }}>
            {/* Spacer: matches the main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16, "@media (max-width: 700px)": { display: "none" } }} />
            {/* Controls row: matches the main panel controls row height */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 28, height: "auto", flexWrap: "wrap", gap: `${SPACING.XS}px`, "@media (max-width: 700px)": { display: "none" } }}>
              {fftComputing ? (
                <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: themeColors.textMuted, "@keyframes pulse": { "0%,100%": { opacity: 0.4 }, "50%": { opacity: 1 } }, animation: "pulse 1.2s ease-in-out infinite" }}>
                  {fftProgress || "Computing FFT…"}</Typography>
              ) : roiFftActive && fftCropDims ? (
                <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: themeColors.accentGreen }}>
                  ROI FFT ({fftCropDims.cropWidth}&times;{fftCropDims.cropHeight})
                </Typography>
              ) : <Box />}
              {(
                <Button size="small" sx={compactButton} disabled={(fftZoom === DEFAULT_FFT_ZOOM && fftPanX === 0 && fftPanY === 0)} onClick={handleFftDoubleClick}>Reset</Button>
              )}
            </Stack>
            <Box
              ref={singleFftContainerRef}
              sx={{ ...responsivePanelSx, border: `1px solid ${themeColors.border}`, cursor: "crosshair" }}
              onWheel={handleFftWheel}
              onDoubleClick={handleFftDoubleClick}
              onMouseDown={handleFftMouseDown}
              onMouseMove={handleFftMouseMove}
              onMouseUp={handleFftMouseUp}
              onMouseLeave={handleFftMouseLeave}
              onTouchStart={(e) => handleFftTouchStart(e, -1)}
              onTouchMove={(e) => handleFftTouchMove(e, -1)}
              onTouchEnd={(e) => handleFftTouchEnd(e, -1)}
              onTouchCancel={(e) => handleFftTouchEnd(e, -1)}
            >
              <canvas data-quantem-scientific-output="show2d-fft-0" ref={fftCanvasRef} width={canvasW} height={canvasH} style={responsiveCanvasStyle} />
              <canvas ref={fftOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={responsiveOverlayStyle} />
              {frequencyRingOverlayForPanel(0)}
              <Box
                className="quantem-fft-zoom-label"
                data-show2d-fft-zoom-indicator="single"
                data-fft-zoom={formatZoomLabel(fftZoom)}
                aria-label={`FFT zoom: ${formatZoomLabel(fftZoom)}`}
                sx={{
                  position: "absolute",
                  left: 12,
                  bottom: 7,
                  color: "white",
                  fontFamily: SYSTEM_FONT,
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
                {formatZoomLabel(fftZoom)}
              </Box>
              {fftMetricsEnabled && fftQuality && (
                <Box
                  className="quantem-fft-quality-label"
                  aria-label={`FFT quality: ${formatFftQualityLabel(fftQuality)}`}
                  sx={{
                    position: "absolute",
                    top: 8,
                    left: 8,
                    maxWidth: "calc(100% - 16px)",
                    px: 0.5,
                    py: 0.15,
                    boxSizing: "border-box",
                    color: "rgba(255,255,255,0.96)",
                    bgcolor: "rgba(0,0,0,0.58)",
                    borderRadius: "3px",
                    fontFamily: SYSTEM_FONT,
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
              {fftComputing && !fftOffscreenRef.current && (
                <Box sx={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "rgba(0,0,0,0.6)", pointerEvents: "none" }}>
                  <Typography sx={{ fontSize: 11, color: "#aaa", fontFamily: "monospace", "@keyframes pulse": { "0%,100%": { opacity: 0.4 }, "50%": { opacity: 1 } }, animation: "pulse 1.2s ease-in-out infinite" }}>
                    {fftProgress || "Computing FFT…"}
                  </Typography>
                </Box>
              )}
              {showResizeControls && (
                <Box onMouseDown={handleCanvasResizeStart} title="Resize image" sx={resizeGripSx} />
              )}
            </Box>
            {/* FFT Stats Bar */}
            {fftStats && fftStats.length === 4 && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, flexWrap: "wrap", maxWidth: "100%", boxSizing: "border-box" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[0])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[1])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[2])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[3])}</Box></Typography>
                {fftClickInfo && (
                  <>
                    <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                      {fftClickInfo.dSpacing != null ? (
                        <>d = <Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{formatLength(fftClickInfo.dSpacing, pixelUnit)}</Box>{" | |g| = "}<Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.spatialFreq!.toFixed(4)} {unitSymbol(pixelUnit)}⁻¹</Box></>
                      ) : (
                        <>dist = <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                      )}
                    </Typography>
                  </>
                )}
              </Box>
            )}
            {/* FFT Controls - two rows + histogram (matching main panel layout) */}
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, ...responsivePanelWidthSx }}>
              <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
                <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "flex-start" }}>
                  {/* Row 1: Scale + Color + Colorbar */}
                  <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale</Typography>
                    <Select value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Color</Typography>
                    <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                      {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Colorbar</Typography>
                    <Switch checked={fftShowColorbar} onChange={(e) => { setFftShowColorbar(e.target.checked); }} size="small" sx={switchStyles.small} />
                  </Box>
                  {/* Row 2: Auto + window (ROI FFT) */}
                  <Box sx={{ ...controlRow, ...mobileControlRowSx, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                    <Box sx={controlPairSx}>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto</Typography>
                      <Switch checked={fftAuto} onChange={(e) => { setFftAuto(e.target.checked); }} size="small" sx={switchStyles.small} />
                    </Box>
                    {fftCropDims && (
                      <Box sx={controlPairSx}>
                        <Typography sx={{ ...typography.label, fontSize: 10 }}>Win</Typography>
                        <Switch checked={fftWindow} onChange={(e) => { setFftWindow(e.target.checked); }} size="small" sx={switchStyles.small} />
                      </Box>
                    )}
                  </Box>
                </Box>
                {/* Right: FFT Histogram */}
                {(
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: 1, pointerEvents: "auto" }}>
                    {fftHistogramData && (
                      <Histogram data={fftHistogramData} pinBinsToRange vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme === "dark" ? "dark" : "light"} dataMin={fftDataRange.min} dataMax={fftDataRange.max} />
                    )}
                  </Box>
                )}
              </Box>
            </Box>
          </Box>
        )}
      </Stack>
      </>
      )}
    </Box>
  );
}

export const render = createRender(Show2D);
