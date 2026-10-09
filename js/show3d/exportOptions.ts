/** Show3D export options: file names and types, GIF presets, output scale and the frame list of an animation. */
import { exportTitleSlug } from "../shared/exportFormat";

export function makeExportFilename(
  title: string,
  nSlices: number,
  height: number,
  width: number,
  mode: string,
  quality = "medium",
  downsample = 1,
): string {
  const slug = exportTitleSlug(title, "show3d");
  if (mode === "gif") return `${slug}_${nSlices}x${height}x${width}_${quality}.gif`;
  const binSuffix = mode === "quantized" && downsample > 1 ? `_${downsample}xbin` : "";
  const suffix = mode === "quantized" ? `quantized${binSuffix}` : "exact";
  return `${slug}_${nSlices}x${height}x${width}_${suffix}.html`;
}

export function exportPickerType(mode: string): { description: string; accept: Record<string, string[]> } {
  if (mode === "gif") return { description: "Animated GIF", accept: { "image/gif": [".gif"] } };
  return { description: "Standalone HTML", accept: { "text/html": [".html"] } };
}

export function exportBlobType(mode: string): string {
  if (mode === "gif") return "image/gif";
  return "text/html;charset=utf-8";
}

export const DEFAULT_ANIMATION_EXPORT_FPS = 8;

export const MIN_ANIMATION_TITLE_FONT_PX = 12;

export const MIN_ANIMATION_SCALE_FONT_PX = 12;

export const MIN_ANIMATION_SCALE_BAR_THICKNESS_PX = 5;

export const MIN_ANIMATION_OVERLAY_MARGIN_PX = 12;

export const ANIMATION_QUALITY_SCALE: Record<string, number> = { low: 0.35, medium: 0.6, high: 1.0 };

export const ANIMATION_QUALITY_OPTIONS = ["low", "medium", "high"] as const;

export type AnimationQuality = typeof ANIMATION_QUALITY_OPTIONS[number];

export type ExportPanelMode = "home" | "gif" | "html";

export type ExportSpatialPreset = "full" | "down2" | "down4" | "edge512" | "edge1024";

export type GifExportPreset = "slides" | "compact" | "full" | "custom";

export const EXPORT_SPATIAL_OPTIONS: Array<{
  value: ExportSpatialPreset;
  label: string;
  downsample: number;
  maxEdgePx: number | null;
}> = [
  { value: "full", label: "Full", downsample: 1, maxEdgePx: null },
  { value: "down2", label: "2x downsample", downsample: 2, maxEdgePx: null },
  { value: "down4", label: "4x downsample", downsample: 4, maxEdgePx: null },
  { value: "edge512", label: "Max edge 512 px", downsample: 1, maxEdgePx: 512 },
  { value: "edge1024", label: "Max edge 1024 px", downsample: 1, maxEdgePx: 1024 },
];

export const GIF_EXPORT_PRESETS: Array<{ value: GifExportPreset; label: string }> = [
  { value: "slides", label: "Slides" },
  { value: "compact", label: "Compact" },
  { value: "full", label: "Full" },
];

/**
 * Source-to-GIF pixel scale: the quality preset divided by the downsample
 * factor (at most 1), shrunk further so the whole panel grid fits maxEdgePx,
 * and never so small that the frame drops below one output pixel.
 */
export function animationOutputScale(
  width: number,
  height: number,
  quality: string,
  downsample = 1,
  maxEdgePx: number | null = null,
  visiblePanels = 1,
  maxCols = 1,
  panelGap = 0,
): number {
  const base = ANIMATION_QUALITY_SCALE[quality] ?? ANIMATION_QUALITY_SCALE.medium;
  const factor = Math.max(1, Math.round(downsample || 1));
  let scale = Math.min(1, base / factor);
  if (maxEdgePx && maxEdgePx > 0) {
    const panels = Math.max(1, Math.round(visiblePanels || 1));
    const cols = maxCols <= 0 ? panels : Math.max(1, Math.min(Math.round(maxCols || 1), panels));
    const rows = Math.max(1, Math.ceil(panels / cols));
    const gap = Math.max(0, Number(panelGap) || 0);
    const layoutW = cols * Math.max(1, width) + Math.max(0, cols - 1) * gap;
    const layoutH = rows * Math.max(1, height) + Math.max(0, rows - 1) * gap;
    scale = Math.min(scale, maxEdgePx / Math.max(1, layoutW, layoutH));
  }
  return Math.max(scale, 1 / Math.max(1, width, height));
}

export function spatialOptionFor(value: ExportSpatialPreset) {
  return EXPORT_SPATIAL_OPTIONS.find((option) => option.value === value) || EXPORT_SPATIAL_OPTIONS[0];
}

/**
 * 0-based frame indices for an export: every `everyN`-th frame of the 1-based
 * inclusive range [startOne, endOne], evenly subsampled to at most maxFrames
 * (0 means no cap).
 */
export function buildAnimationFrameIndices(
  nSlices: number,
  startOne: number,
  endOne: number,
  everyN: number,
  maxFrames: number,
): number[] {
  const total = Math.max(1, Math.floor(nSlices || 1));
  const start = Math.max(0, Math.min(total - 1, Math.round(startOne || 1) - 1));
  const end = Math.max(start, Math.min(total - 1, Math.round(endOne || total) - 1));
  const step = Math.max(1, Math.round(everyN || 1));
  const frames: number[] = [];
  for (let idx = start; idx <= end; idx += step) frames.push(idx);
  const cap = Math.max(0, Math.round(maxFrames || 0));
  if (cap > 0 && frames.length > cap) {
    if (cap === 1) return [frames[0]];
    const sampled: number[] = [];
    for (let i = 0; i < cap; i++) {
      sampled.push(frames[Math.round((i * (frames.length - 1)) / (cap - 1))]);
    }
    return sampled;
  }
  return frames;
}

export function formatEstimatedAnimationWork(
  width: number,
  height: number,
  nSlices: number,
  visiblePanels: number,
  maxCols: number,
  panelGap: number,
  quality: string,
  downsample = 1,
  maxEdgePx: number | null = null,
): string {
  const scale = animationOutputScale(width, height, quality, downsample, maxEdgePx, visiblePanels, maxCols, panelGap);
  const panelW = Math.max(1, Math.floor(Math.max(1, width) * scale));
  const panelH = Math.max(1, Math.floor(Math.max(1, height) * scale));
  const panels = Math.max(1, visiblePanels);
  const cols = maxCols <= 0 ? panels : Math.max(1, Math.min(maxCols, panels));
  const rows = Math.max(1, Math.ceil(panels / cols));
  const gap = Math.max(0, Math.round((panelGap || 0) * scale));
  const outW = cols * panelW + Math.max(0, cols - 1) * gap;
  const outH = rows * panelH + Math.max(0, rows - 1) * gap;
  const rgbBytes = outW * outH * Math.max(1, nSlices) * 3;
  const megabytes = rgbBytes / (1024 * 1024);
  if (megabytes >= 100) return `~${Math.round(megabytes)} MB before compression`;
  if (megabytes >= 10) return `~${megabytes.toFixed(1)} MB before compression`;
  return `~${megabytes.toFixed(2)} MB before compression`;
}
