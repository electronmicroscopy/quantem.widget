/**
 * ShowPtycho — interactive ptychography aberration explorer widget.
 *
 * Two panels (Phase + optional FFT) following Show2D/Live design.
 * Uses full selected BF by default; a smaller BF fraction can still be selected
 * from the same BF-count control for faster exploratory review.
 * Features: zoom/pan, scale bar, histogram contrast, colormap selector,
 * pin system, resize handle.
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Switch from "@mui/material/Switch";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Badge from "@mui/material/Badge";
import Tooltip from "@mui/material/Tooltip";
import LinearProgress from "@mui/material/LinearProgress";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import PushPinIcon from "@mui/icons-material/PushPin";
import ShuffleIcon from "@mui/icons-material/Shuffle";
import SaveIcon from "@mui/icons-material/Save";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import { useTheme, type ThemeColors } from "../theme";
import { extractFloat32, formatNumber } from "../format";
import { percentileClip } from "../stats";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen, GPUColormapEngine, getGPUColormapEngine } from "../colormaps";
import { fft2d, nextPow2, fftshift, computeMagnitude, applyHannWindow2D, getWebGPUFFT, WebGPUFFT } from "../fft";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI } from "../figure";
import { computeHistogramFromBytes } from "../stats";
import { WebGPUSSBBackend, deleteSSBFolderFile, readSSBFolderBytes, readSSBFolderJson, setSSBLocalDirectory, setSSBLocalFiles, ssbFolderWritable, ssbNeedsLocalSource, writeSSBFolderFile, type WebGPULoadProgress } from "../.generated/engine/ssb/compute/webgpu/backend";

/* ================================================================
   Design tokens (matching Live / Show2D)
   ================================================================ */

const typography = {
  label: { fontSize: 11 } as const,
  labelSmall: { fontSize: 10 } as const,
  value: { fontSize: 10, fontFamily: "monospace" } as const,
};

const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const DEFAULT_BF_FRACTION = 1.0;

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
  imageBox: { overflow: "hidden", position: "relative" as const },
};

function compactButton(tc: ThemeColors) {
  return {
    fontSize: 10,
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

function compactPathLabel(path: string, depth = 2): string {
  const parts = path.split("/").filter(Boolean);
  if (parts.length <= depth) return path;
  return parts.slice(-depth).join("/");
}

const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};

function formatStat(v: number): string {
  if (!isFinite(v)) return "--";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 0.01) return v.toExponential(2);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(3);
  return v.toFixed(4);
}

/* ================================================================
   Constants
   ================================================================ */

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 20;
const ZOOM_FACTOR = 1.15;
const DPR = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
const MIN_PANEL = 200;
const DEFAULT_PANEL = 380;
const MAX_PINS = 20;
// Thumb bitmap is rendered at the largest size and CSS-scaled down — keeps quality
// when the user flips to a larger preset without re-making the thumbnail.
const THUMB_BITMAP_PX = 128;
type ThumbSize = "S" | "M" | "L";
const THUMB_SIZE_PX: Record<ThumbSize, number> = { S: 44, M: 72, L: 120 };
const PANEL_GAP = 0;
const ACTION_CONTROL_HEIGHT = 26;
const ABERRATION_SLIDER_WIDTH = 144;
const ROTATION_SLIDER_WIDTH = 170;

// Semantic status colors — used only for loss delta (better/worse) and the play
// indicator (active sweep).  Kept as constants so there's one place to retune.
const STATUS_GOOD = "#4caf50";
const STATUS_BAD = "#f44336";

type ZoomState = { zoom: number; panX: number; panY: number };
const ZOOM_RESET: ZoomState = { zoom: 1, panX: 0, panY: 0 };
type RealViewMode = "phase" | "amp" | "complex";
type ExtraRealViewMode = Exclude<RealViewMode, "phase">;
type FFTPlacement = "panel" | "inset";
type FFTInsetBox = { x: number; y: number; size: number };
type GPUColormapSlot = 0 | 1 | 2;

interface PinnedEntry {
  id: number;
  C10: number;
  C12: number;
  phi12_deg: number;
  rotation_deg: number;
  flip_phase: boolean;
  loss: number;
  starred: boolean;
  timestamp: string;
  phaseData: Float32Array;
  displayMode: RealViewMode;
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
): { canvas: HTMLCanvasElement | null; min: number; max: number; clipMs: number; renderMs: number } {
  const t0 = performance.now();
  const { vmin, vmax, min, max } = percentileClip(data, pctLo, pctHi);
  const tClip = performance.now();
  const canvas = renderToOffscreen(data, w, h, lut, vmin, vmax);
  const tRender = performance.now();
  return { canvas, min, max, clipMs: tClip - t0, renderMs: tRender - tClip };
}

function phaseAbsData(data: Float32Array): Float32Array {
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Math.abs(data[i]);
  return out;
}

function phaseDisplayData(
  phase: Float32Array,
  mode: RealViewMode,
  cache: React.MutableRefObject<{ source: Float32Array | null; abs: Float32Array | null }>,
): Float32Array {
  if (mode === "phase") return phase;
  if (cache.current.source !== phase || !cache.current.abs) {
    cache.current = { source: phase, abs: phaseAbsData(phase) };
  }
  return cache.current.abs ?? phase;
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
): { canvas: HTMLCanvasElement | null; min: number; max: number; clipMs: number; renderMs: number; amp: Float32Array } {
  const t0 = performance.now();
  const amp = phaseAbsData(phase);
  const { vmin, vmax, min, max } = percentileClip(amp, pctLo, pctHi);
  const tClip = performance.now();
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return { canvas: null, min, max, clipMs: tClip - t0, renderMs: performance.now() - tClip, amp };
  const img = ctx.createImageData(w, h);
  const range = vmax > vmin ? vmax - vmin : 1;
  for (let i = 0; i < phase.length; i++) {
    const hue = ((phase[i] + Math.PI) / (2 * Math.PI) % 1 + 1) % 1;
    const clipped = Math.max(vmin, Math.min(vmax, amp[i]));
    const value = Math.max(0, Math.min(1, (clipped - vmin) / range));
    const [r, g, b] = hsvToRgb(hue, 0.92, value);
    const j = i * 4;
    img.data[j] = Math.round(r * 255);
    img.data[j + 1] = Math.round(g * 255);
    img.data[j + 2] = Math.round(b * 255);
    img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const tRender = performance.now();
  return { canvas, min, max, clipMs: tClip - t0, renderMs: tRender - tClip, amp };
}

/** Mean-subtract, Hann-window, zero-pad — shared prelude for both FFT paths. */
function prepareFFTInput(data: Float32Array, w: number, h: number) {
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i];
  const mean = sum / data.length;
  const centered = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) centered[i] = data[i] - mean;
  applyHannWindow2D(centered, w, h);
  const pw = nextPow2(w), ph = nextPow2(h);
  const real = new Float32Array(pw * ph);
  const imag = new Float32Array(pw * ph);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      real[y * pw + x] = centered[y * w + x];
  return { real, imag, pw, ph };
}

/** Crop log-magnitude back to original size + fftshift. Shared postlude. */
function finalizeFFTMag(
  real: Float32Array, imag: Float32Array,
  w: number, h: number, pw: number,
): Float32Array {
  const fullMag = computeMagnitude(real, imag);
  const cropped = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      cropped[y * w + x] = fullMag[y * pw + x];
  fftshift(cropped, w, h);
  for (let i = 0; i < cropped.length; i++) cropped[i] = Math.log1p(cropped[i]);
  return cropped;
}

/** CPU fallback: synchronous JS FFT. */
/** Find brightest pixel in FFT mag within radius of (col,row), then sub-pixel
 *  refine via 3×3 weighted centroid. Mirrors show2d's d-spacing snap. */
function findFFTPeak(mag: Float32Array, width: number, height: number, col: number, row: number, radius: number): { row: number; col: number } {
  const c0 = Math.max(0, Math.floor(col) - radius);
  const r0 = Math.max(0, Math.floor(row) - radius);
  const c1 = Math.min(width - 1, Math.floor(col) + radius);
  const r1 = Math.min(height - 1, Math.floor(row) + radius);
  let bestCol = Math.round(col), bestRow = Math.round(row), bestVal = -Infinity;
  for (let ir = r0; ir <= r1; ir++) {
    for (let ic = c0; ic <= c1; ic++) {
      const v = mag[ir * width + ic];
      if (v > bestVal) { bestVal = v; bestCol = ic; bestRow = ir; }
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

function computeFFTMag(data: Float32Array, w: number, h: number): { mag: Float32Array; pw: number; ph: number } {
  const { real, imag, pw, ph } = prepareFFTInput(data, w, h);
  fft2d(real, imag, pw, ph, false);
  return { mag: finalizeFFTMag(real, imag, w, h, pw), pw, ph };
}

/** WebGPU FFT when available — offloads from the main thread. */
async function computeFFTMagGPU(
  gpu: WebGPUFFT,
  data: Float32Array, w: number, h: number,
): Promise<{ mag: Float32Array; pw: number; ph: number }> {
  const { real, imag, pw, ph } = prepareFFTInput(data, w, h);
  const res = await gpu.fft2D(real, imag, pw, ph, false);
  return { mag: finalizeFFTMag(res.real, res.imag, w, h, pw), pw, ph };
}

function renderFFTOffscreen(
  mag: Float32Array, w: number, h: number,
  lut: Uint8Array,
  pLo = 1, pHi = 99,
): { canvas: HTMLCanvasElement | null; min: number; max: number } {
  const { vmin, vmax, min, max } = percentileClip(mag, pLo, pHi);
  return { canvas: renderToOffscreen(mag, w, h, lut, vmin, vmax), min, max };
}

function makeThumbnail(data: Float32Array, w: number, h: number, cmapName: string, mode: RealViewMode = "phase") {
  const lut = COLORMAPS[cmapName as keyof typeof COLORMAPS] || COLORMAPS.viridis;
  const { canvas: full } = mode === "complex"
    ? renderComplexPhaseOffscreen(data, w, h, 1, 99)
    : renderPhaseOffscreen(mode === "amp" ? phaseAbsData(data) : data, w, h, lut, 1, 99);
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
    if (typeof canvas.toBlob !== "function") {
      try {
        const dataUrl = canvas.toDataURL("image/png");
        const [meta, base64] = dataUrl.split(",");
        const mime = meta.match(/data:([^;]+)/)?.[1] || "image/png";
        const raw = atob(base64);
        const bytes = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
        resolve(new Blob([bytes], { type: mime }));
      } catch (err) {
        reject(err);
      }
      return;
    }
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Could not encode pinned image as PNG."));
    }, "image/png");
  });
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

let browserGifPaletteCache: Uint8Array | null = null;

function asciiBytes(value: string): Uint8Array {
  const out = new Uint8Array(value.length);
  for (let i = 0; i < value.length; i++) out[i] = value.charCodeAt(i) & 0xff;
  return out;
}

function u16Bytes(value: number): Uint8Array {
  const v = Math.max(0, Math.min(65535, Math.round(value)));
  return new Uint8Array([v & 0xff, (v >> 8) & 0xff]);
}

function concatUint8(parts: Uint8Array[]): Uint8Array {
  let total = 0;
  for (const part of parts) total += part.byteLength;
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.byteLength;
  }
  return out;
}

function browserGifPalette(): Uint8Array {
  if (browserGifPaletteCache) return browserGifPaletteCache;
  const palette = new Uint8Array(256 * 3);
  let idx = 0;
  for (let r = 0; r < 6; r++) {
    for (let g = 0; g < 6; g++) {
      for (let b = 0; b < 6; b++) {
        const j = idx * 3;
        palette[j] = r * 51;
        palette[j + 1] = g * 51;
        palette[j + 2] = b * 51;
        idx++;
      }
    }
  }
  const grayCount = 256 - idx;
  for (let i = 0; idx < 256; idx++, i++) {
    const v = grayCount <= 1 ? 0 : Math.round((i / (grayCount - 1)) * 255);
    const j = idx * 3;
    palette[j] = v;
    palette[j + 1] = v;
    palette[j + 2] = v;
  }
  browserGifPaletteCache = palette;
  return palette;
}

function quantizeRgbaForBrowserGif(rgba: Uint8ClampedArray): Uint8Array {
  const out = new Uint8Array(Math.floor(rgba.length / 4));
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    const a = rgba[j + 3];
    const r = a === 255 ? rgba[j] : Math.round((rgba[j] * a + 255 * (255 - a)) / 255);
    const g = a === 255 ? rgba[j + 1] : Math.round((rgba[j + 1] * a + 255 * (255 - a)) / 255);
    const b = a === 255 ? rgba[j + 2] : Math.round((rgba[j + 2] * a + 255 * (255 - a)) / 255);
    const rq = Math.max(0, Math.min(5, Math.round(r / 51)));
    const gq = Math.max(0, Math.min(5, Math.round(g / 51)));
    const bq = Math.max(0, Math.min(5, Math.round(b / 51)));
    out[i] = rq * 36 + gq * 6 + bq;
  }
  return out;
}

function gifLzwEncode(indices: Uint8Array): Uint8Array {
  const minCodeSize = 8;
  const clearCode = 1 << minCodeSize;
  const endCode = clearCode + 1;
  const codeSize = minCodeSize + 1;
  const bytes: number[] = [];
  let bitBuffer = 0;
  let bitCount = 0;
  const writeCode = (code: number) => {
    bitBuffer |= code << bitCount;
    bitCount += codeSize;
    while (bitCount >= 8) {
      bytes.push(bitBuffer & 0xff);
      bitBuffer >>= 8;
      bitCount -= 8;
    }
  };
  writeCode(clearCode);
  let sinceClear = 0;
  for (let i = 0; i < indices.length; i++) {
    if (sinceClear >= 250) {
      writeCode(clearCode);
      sinceClear = 0;
    }
    writeCode(indices[i]);
    sinceClear++;
  }
  writeCode(endCode);
  if (bitCount > 0) bytes.push(bitBuffer & 0xff);
  return new Uint8Array(bytes);
}

function pushGifSubBlocks(parts: Uint8Array[], data: Uint8Array): void {
  for (let offset = 0; offset < data.length; offset += 255) {
    const chunk = data.subarray(offset, Math.min(offset + 255, data.length));
    parts.push(new Uint8Array([chunk.length]));
    parts.push(chunk);
  }
  parts.push(new Uint8Array([0]));
}

function encodeIndexedGif(width: number, height: number, frames: Uint8Array[], delayCs: number): Uint8Array {
  if (width <= 0 || height <= 0 || frames.length === 0) {
    throw new Error("GIF export needs at least one frame.");
  }
  const parts: Uint8Array[] = [
    asciiBytes("GIF89a"),
    u16Bytes(width),
    u16Bytes(height),
    new Uint8Array([0xf7, 0, 0]),
    browserGifPalette(),
    new Uint8Array([0x21, 0xff, 0x0b]),
    asciiBytes("NETSCAPE2.0"),
    new Uint8Array([0x03, 0x01, 0x00, 0x00, 0x00]),
  ];
  const delay = Math.max(1, Math.min(65535, Math.round(delayCs)));
  for (const frame of frames) {
    parts.push(new Uint8Array([0x21, 0xf9, 0x04, 0x04]));
    parts.push(u16Bytes(delay));
    parts.push(new Uint8Array([0, 0]));
    parts.push(new Uint8Array([0x2c]));
    parts.push(u16Bytes(0), u16Bytes(0), u16Bytes(width), u16Bytes(height));
    parts.push(new Uint8Array([0, 8]));
    pushGifSubBlocks(parts, gifLzwEncode(frame));
  }
  parts.push(new Uint8Array([0x3b]));
  return concatUint8(parts);
}

function formatSavedBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${Math.round(bytes)} B`;
}

function makeShowPtychoExportFilename(mode: "gif" | "mp4", sweepParam: string, frameCount: number): string {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const target = slugPart(sweepParam.replace(/^bundle:/, "").replace(/^ho:/, ""));
  return `showptycho_${target}_${frameCount}f_${stamp}.${mode}`;
}

function publishShowPtychoTestPhase(data: Float32Array, w: number, h: number): void {
  const win = window as typeof window & {
    __QUANTEM_SHOWPTYCHO_CAPTURE__?: boolean;
    __QUANTEM_SHOWPTYCHO_LAST_PHASE__?: { data: Float32Array; w: number; h: number; updatedAt: number };
  };
  if (!win.__QUANTEM_SHOWPTYCHO_CAPTURE__) return;
  win.__QUANTEM_SHOWPTYCHO_LAST_PHASE__ = { data, w, h, updatedAt: performance.now() };
}

function publishShowPtychoTestFFT(
  data: Float32Array,
  w: number,
  h: number,
  pw: number,
  ph: number,
): void {
  const win = window as typeof window & {
    __QUANTEM_SHOWPTYCHO_CAPTURE__?: boolean;
    __QUANTEM_SHOWPTYCHO_LAST_FFT__?: {
      data: Float32Array;
      w: number;
      h: number;
      pw: number;
      ph: number;
      updatedAt: number;
    };
  };
  if (!win.__QUANTEM_SHOWPTYCHO_CAPTURE__) return;
  win.__QUANTEM_SHOWPTYCHO_LAST_FFT__ = { data, w, h, pw, ph, updatedAt: performance.now() };
}

/* ================================================================
   drawCanvas — DPR-aware (matching Live)
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
  const targetW = Math.round(size * DPR);
  const targetH = Math.round(size * DPR);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
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

  // Smooth on: bilinear via canvas2d. Smooth off: nearest-neighbor (auto
  // also nearest above 4× zoom to keep pixel structure visible).
  if (smooth) {
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
  } else {
    ctx.imageSmoothingEnabled = false;
  }
  ctx.drawImage(offscreen, (size - drawW) / 2 + panX, (size - drawH) / 2 + panY, drawW, drawH);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

/* ================================================================
   Histogram (copied from Live)
   ================================================================ */

interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  tc: ThemeColors;
  dataMin?: number;
  dataMax?: number;
}

function Histogram({
  data, vminPct, vmaxPct, onRangeChange,
  width = 110, height = 40, tc, dataMin = 0, dataMax = 1,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);

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
      let s = 0;
      for (let j = 0; j < binRatio; j++) s += bins[i * binRatio + j] || 0;
      reduced.push(s / binRatio);
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
        onChange={(_, v) => {
          const [lo, hi] = v as number[];
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
   ImagePanel — header + canvas with zoom/pan + scale bar overlay
   ================================================================ */

function ImagePanel({
  offscreen, offscreenVersion, label, size, tc, zoom, onZoomChange,
  showResize, onResizeStart,
  pixelSize, imageWidth, isFFT,
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
  smooth?: boolean;
  imageWidth?: number;
  isFFT?: boolean;
  rawData?: Float32Array | null;
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
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);

  React.useEffect(() => {
    drawCanvas(canvasRef.current, offscreen, size, zoom, tc.bgAlt, tc.textMuted, !!smooth);
  }, [offscreen, offscreenVersion, size, zoom, tc, smooth]);

  // Scale bar overlay
  React.useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay || !pixelSize || pixelSize <= 0 || !imageWidth || imageWidth <= 0) return;
    overlay.width = size * DPR;
    overlay.height = size * DPR;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (isFFT) {
      drawFFTScaleBarHiDPI(overlay, DPR, zoom.zoom, pixelSize, imageWidth);
      // d-spacing marker — Show3D style: open circle + gap-cross, white with
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
        ctx.scale(DPR, DPR);
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.shadowColor = "rgba(0,0,0,0.6)";
        ctx.shadowBlur = 2;
        ctx.lineWidth = 1.5;
        const r = 8;
        ctx.beginPath();
        ctx.moveTo(cssX - r, cssY); ctx.lineTo(cssX - 3, cssY);
        ctx.moveTo(cssX + 3, cssY); ctx.lineTo(cssX + r, cssY);
        ctx.moveTo(cssX, cssY - r); ctx.lineTo(cssX, cssY - 3);
        ctx.moveTo(cssX, cssY + 3); ctx.lineTo(cssX, cssY + r);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cssX, cssY, 4, 0, Math.PI * 2);
        ctx.stroke();
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
      drawScaleBarHiDPI(overlay, DPR, zoom.zoom, pixelSize, "Å", imageWidth);
    }
  }, [zoom, size, pixelSize, imageWidth, isFFT, fftClickInfo, offscreen]);

  // Native wheel listener with { passive: false } — React's synthetic onWheel
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
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = clientX - rect.left, mouseY = clientY - rect.top;
    const baseFit = Math.min(size / offscreen.width, size / offscreen.height);
    const scale = baseFit * zoomRef.current.zoom;
    const xOffset = (size - offscreen.width * scale) / 2 + zoomRef.current.panX;
    const yOffset = (size - offscreen.height * scale) / 2 + zoomRef.current.panY;
    const col = Math.floor((mouseX - xOffset) / scale);
    const row = Math.floor((mouseY - yOffset) / scale);
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
  const onFFTUp = React.useCallback((e: React.MouseEvent) => {
    if (!isFFT || !fftClickStartRef.current || !offscreen || !rawData) return;
    const dx = e.clientX - fftClickStartRef.current.x;
    const dy = e.clientY - fftClickStartRef.current.y;
    fftClickStartRef.current = null;
    if (Math.sqrt(dx * dx + dy * dy) >= 3) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const mouseX = e.clientX - rect.left, mouseY = e.clientY - rect.top;
    const baseFit = Math.min(size / offscreen.width, size / offscreen.height);
    const scale = baseFit * zoomRef.current.zoom;
    const xOffset = (size - offscreen.width * scale) / 2 + zoomRef.current.panX;
    const yOffset = (size - offscreen.height * scale) / 2 + zoomRef.current.panY;
    let imgCol = (mouseX - xOffset) / scale;
    let imgRow = (mouseY - yOffset) / scale;
    const fftW = offscreen.width, fftH = offscreen.height;
    if (imgCol < 0 || imgCol >= fftW || imgRow < 0 || imgRow >= fftH) return;
    // Snap to nearest local-max — same as Show3D's findFFTPeak.
    const snapped = findFFTPeak(rawData, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
    imgCol = snapped.col; imgRow = snapped.row;
    // After fftshift, DC sits at (fftW/2, fftH/2). Pixel-distance from DC.
    const halfW = Math.floor(fftW / 2);
    const halfH = Math.floor(fftH / 2);
    const dcol = imgCol - halfW;
    const drow = imgRow - halfH;
    const distPx = Math.sqrt(dcol * dcol + drow * drow);
    if (distPx < 1) { setFftClickInfo(null); return; }
    // Bin → frequency. FFT computed at zero-padded next-pow2 size, cropped
    // back to fftW. Bin index is mod-fftW (cropped), divisor is paddedW
    // (true FFT resolution). Mirrors Show3D's d-spacing math exactly.
    let spatialFreq: number | null = null;
    let dSpacing: number | null = null;
    if (pixelSize && pixelSize > 0) {
      const paddedW = nextPow2(fftW);
      const paddedH = nextPow2(fftH);
      const binC = ((Math.round(imgCol) - halfW) % fftW + fftW) % fftW;
      const binR = ((Math.round(imgRow) - halfH) % fftH + fftH) % fftH;
      const freqC = binC <= paddedW / 2 ? binC / (paddedW * pixelSize) : (binC - paddedW) / (paddedW * pixelSize);
      const freqR = binR <= paddedH / 2 ? binR / (paddedH * pixelSize) : (binR - paddedH) / (paddedH * pixelSize);
      spatialFreq = Math.sqrt(freqC * freqC + freqR * freqR);
      dSpacing = spatialFreq > 0 ? 1 / spatialFreq : null;
    }
    setFftClickInfo({ row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
  }, [isFFT, offscreen, rawData, size, pixelSize, imageWidth]);
  const onLeaveFFT = React.useCallback(() => { fftClickStartRef.current = null; }, []);

  // Cursor readout: map mouse position through zoom/pan to raw-data indices
  const onMove = React.useCallback((e: React.MouseEvent) => {
    if (!canvasRef.current || !offscreen || !rawData) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left, mouseY = e.clientY - rect.top;
    const baseFit = Math.min(size / offscreen.width, size / offscreen.height);
    const scale = baseFit * zoomRef.current.zoom;
    const xOffset = (size - offscreen.width * scale) / 2 + zoomRef.current.panX;
    const yOffset = (size - offscreen.height * scale) / 2 + zoomRef.current.panY;
    const col = Math.floor((mouseX - xOffset) / scale);
    const row = Math.floor((mouseY - yOffset) / scale);
    if (row < 0 || row >= offscreen.height || col < 0 || col >= offscreen.width) {
      setCursor(null);
      return;
    }
    setCursor({ row, col, val: rawData[row * offscreen.width + col] });
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
    const [r0, r1, c0, c1] = cropRegion;
    const baseFit = Math.min(size / offscreen.width, size / offscreen.height);
    const scale = baseFit * zoom.zoom;
    return {
      left: (size - offscreen.width * scale) / 2 + zoom.panX + c0 * scale,
      top: (size - offscreen.height * scale) / 2 + zoom.panY + r0 * scale,
      width: (c1 - c0) * scale,
      height: (r1 - r0) * scale,
    };
  }, [cropRegion, isFFT, offscreen, size, zoom]);

  return (
    <Box sx={{ width: size, flexShrink: 0, display: "flex", flexDirection: "column" }}>
      <Box sx={{ ...container.imageBox, bgcolor: tc.bgAlt, border: `1px solid ${tc.border}`, width: size, height: size }}>
        <canvas
          aria-label={label}
          ref={canvasRef}
          onMouseDown={onDown}
          onMouseMove={onMove}
          onMouseUp={onUp}
          onMouseLeave={() => { onLeave(); if (isFFT) onLeaveFFT(); }}
          onDoubleClick={() => { onZoomChange(ZOOM_RESET); if (isFFT) setFftClickInfo(null); }}
          style={{ width: size, height: size, cursor: isFFT || cropSelecting ? "crosshair" : "grab", display: "block", imageRendering: smooth ? "auto" : "pixelated" }}
        />
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
  offscreen, offscreenVersion, rawData, size, panelSize, box, onBoxChange, tc, smooth, pixelSize,
}: {
  offscreen: HTMLCanvasElement | null;
  offscreenVersion?: number;
  rawData?: Float32Array | null;
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
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; dSpacing: number | null;
  } | null>(null);

  React.useEffect(() => {
    drawCanvas(canvasRef.current, offscreen, size, ZOOM_RESET, "#000", "#fff", !!smooth);
  }, [offscreen, offscreenVersion, size, smooth]);

  React.useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    overlay.width = size * DPR;
    overlay.height = size * DPR;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!fftClickInfo || !offscreen) return;
    const scale = size / Math.max(offscreen.width, offscreen.height);
    const cssX = (size - offscreen.width * scale) / 2 + fftClickInfo.col * scale;
    const cssY = (size - offscreen.height * scale) / 2 + fftClickInfo.row * scale;
    ctx.save();
    ctx.scale(DPR, DPR);
    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.shadowColor = "rgba(0,0,0,0.7)";
    ctx.shadowBlur = 2;
    ctx.lineWidth = 1.4;
    const r = 7;
    ctx.beginPath();
    ctx.moveTo(cssX - r, cssY); ctx.lineTo(cssX - 3, cssY);
    ctx.moveTo(cssX + 3, cssY); ctx.lineTo(cssX + r, cssY);
    ctx.moveTo(cssX, cssY - r); ctx.lineTo(cssX, cssY - 3);
    ctx.moveTo(cssX, cssY + 3); ctx.lineTo(cssX, cssY + r);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cssX, cssY, 3.5, 0, Math.PI * 2);
    ctx.stroke();
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

  const measurePeak = React.useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas || !offscreen || !rawData) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = clientX - rect.left;
    const mouseY = clientY - rect.top;
    const scale = size / Math.max(offscreen.width, offscreen.height);
    const xOffset = (size - offscreen.width * scale) / 2;
    const yOffset = (size - offscreen.height * scale) / 2;
    let imgCol = (mouseX - xOffset) / scale;
    let imgRow = (mouseY - yOffset) / scale;
    const fftW = offscreen.width;
    const fftH = offscreen.height;
    if (imgCol < 0 || imgCol >= fftW || imgRow < 0 || imgRow >= fftH) return;
    const snapped = findFFTPeak(rawData, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
    imgCol = snapped.col; imgRow = snapped.row;
    const halfW = Math.floor(fftW / 2);
    const halfH = Math.floor(fftH / 2);
    const dcol = imgCol - halfW;
    const drow = imgRow - halfH;
    if (Math.sqrt(dcol * dcol + drow * drow) < 1) {
      setFftClickInfo(null);
      return;
    }
    let dSpacing: number | null = null;
    if (pixelSize && pixelSize > 0) {
      const paddedW = nextPow2(fftW);
      const paddedH = nextPow2(fftH);
      const binC = ((Math.round(imgCol) - halfW) % fftW + fftW) % fftW;
      const binR = ((Math.round(imgRow) - halfH) % fftH + fftH) % fftH;
      const freqC = binC <= paddedW / 2 ? binC / (paddedW * pixelSize) : (binC - paddedW) / (paddedW * pixelSize);
      const freqR = binR <= paddedH / 2 ? binR / (paddedH * pixelSize) : (binR - paddedH) / (paddedH * pixelSize);
      const spatialFreq = Math.sqrt(freqC * freqC + freqR * freqR);
      dSpacing = spatialFreq > 0 ? 1 / spatialFreq : null;
    }
    setFftClickInfo({ row: imgRow, col: imgCol, dSpacing });
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
      if (!moved) measurePeak(upEvent.clientX, upEvent.clientY);
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
        boxShadow: `0 2px 10px ${tc.shadow}`,
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

// Aberration layout.  Magnitude ranges are empirical — they cover typical
// Cs-corrected 300 kV aberration budgets (C3 ~1 μm, C5 ~1 mm).  Users on
// uncorrected scopes will live at the low end of each range.
type HOEntry = {
  name: string;          // Krivanek name (C21, C23, ...)
  hasAngle: boolean;     // false for rotationally-symmetric (m=0) terms
  mag_max: number;       // nm (internal storage; engine convention)
  step_nm: number;       // slider step in nm — tuned so snapping lands on "nice" display values
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
      tooltip: "C21 — axial coma.  Comet-like tails on features.  Range ±100 nm, step 2 nm." },
    { name: "C23", hasAngle: true,  mag_max: 100,          step_nm: 2,       unit_display: "nm", display_scale: 1,
      tooltip: "C23 — 3-fold astigmatism.  3-lobed probe shape.  Range ±100 nm, step 2 nm." },
  ],
  3: [
    { name: "C30", hasAngle: false, mag_max: 100000,       step_nm: 1000,    unit_display: "μm", display_scale: 1000,
      tooltip: "C30 — spherical aberration (Cs).  Primary resolution limiter.  Range ±100 μm, step 1 μm." },
    { name: "C32", hasAngle: true,  mag_max: 100000,       step_nm: 1000,    unit_display: "μm", display_scale: 1000,
      tooltip: "C32 — star aberration.  2-fold star pattern.  Range ±100 μm, step 1 μm." },
    { name: "C34", hasAngle: true,  mag_max: 1000,         step_nm: 10,      unit_display: "μm", display_scale: 1000,
      tooltip: "C34 — 4-fold astigmatism.  Range ±1 μm, step 0.01 μm (10 nm)." },
  ],
  4: [
    { name: "C41", hasAngle: true,  mag_max: 100000,       step_nm: 1000,    unit_display: "μm", display_scale: 1000,
      tooltip: "C41 — 4th order coma.  Subtle directional blur.  Range ±100 μm, step 1 μm." },
    { name: "C43", hasAngle: true,  mag_max: 100000,       step_nm: 1000,    unit_display: "μm", display_scale: 1000,
      tooltip: "C43 — 3-lobe aberration (4th order).  Range ±100 μm, step 1 μm." },
    { name: "C45", hasAngle: true,  mag_max: 100000,       step_nm: 1000,    unit_display: "μm", display_scale: 1000,
      tooltip: "C45 — 5-fold astigmatism (4th order).  Range ±100 μm, step 1 μm." },
  ],
  5: [
    { name: "C50", hasAngle: false, mag_max: 100000000,    step_nm: 100000,  unit_display: "mm", display_scale: 1000000,
      tooltip: "C50 — 5th order spherical.  Dominant on Cs-corrected scopes.  Range ±100 mm, step 0.1 mm." },
    { name: "C52", hasAngle: true,  mag_max: 100000000,    step_nm: 1000000, unit_display: "mm", display_scale: 1000000,
      tooltip: "C52 — 5th order star.  Range ±100 mm, step 1 mm." },
    { name: "C54", hasAngle: true,  mag_max: 100000000,    step_nm: 1000000, unit_display: "mm", display_scale: 1000000,
      tooltip: "C54 — rosette aberration; 4-fold rosette pattern.  Range ±100 mm, step 1 mm." },
    { name: "C56", hasAngle: true,  mag_max: 100000000,    step_nm: 1000000, unit_display: "mm", display_scale: 1000000,
      tooltip: "C56 — 6-fold astigmatism (5th order).  Range ±100 mm, step 1 mm." },
  ],
};

function formatHOValue(v_nm: number, max_nm: number, display_scale: number): string {
  // Convert nm → native display unit (μm divides by 1000, mm by 1e6)
  const v = v_nm / display_scale;
  const max = max_nm / display_scale;
  const a = Math.abs(v);
  if (a === 0) return "0";
  // Choose precision based on range in display units.  For sub-unit
  // ranges (e.g. C34 with max 0.1 μm) need extra decimals so small values
  // don't round to "0.00".
  if (max >= 100) return v.toFixed(0);
  if (max >= 10) return v.toFixed(1);
  if (max >= 1) return v.toFixed(2);
  if (max >= 0.1) return v.toFixed(3);
  return v.toFixed(4);
}

function HigherOrderPanel({
  tc, open, onToggle, activeCount, values, setValues,
}: {
  tc: ThemeColors;
  open: boolean;
  onToggle: () => void;
  activeCount: number;
  values: Record<string, number>;
  setValues: React.Dispatch<React.SetStateAction<Record<string, number>>>;
}) {
  const updateMag = React.useCallback((name: string, v: number) => {
    setValues(prev => ({ ...prev, [name]: v }));
  }, [setValues]);
  const updateAngle = React.useCallback((name: string, deg: number) => {
    setValues(prev => ({ ...prev, [`${name}_angle`]: deg }));
  }, [setValues]);
  const resetAll = React.useCallback(() => setValues({}), [setValues]);
  const resetMag = React.useCallback((name: string, hasAngle: boolean) => {
    setValues(prev => {
      const next = { ...prev };
      const magKey = hasAngle ? `${name}_mag` : name;
      delete next[magKey];
      return next;
    });
  }, [setValues]);
  const resetAngle = React.useCallback((name: string) => {
    setValues(prev => {
      const next = { ...prev };
      delete next[`${name}_angle`];
      return next;
    });
  }, [setValues]);
  const orderColumns = React.useMemo(() => [[2, 3], [4, 5]], []);
  const renderEntry = React.useCallback((e: HOEntry) => {
    const magKey = e.hasAngle ? `${e.name}_mag` : e.name;
    const mag = values[magKey] ?? 0;
    const ang = e.hasAngle ? (values[`${e.name}_angle`] ?? 0) : 0;
    const isActive = Math.abs(mag) > 0;
    const hasAngleOffset = e.hasAngle && Math.abs(ang) > 0;
    return (
      <Box
        key={e.name}
        sx={{
          display: "flex", alignItems: "center", gap: `${SPACING.SM}px`,
          py: 0.25,
        }}
      >
        <Tooltip title={e.tooltip} placement="right" arrow>
          <Typography
            sx={{
              ...typography.value,
              color: isActive ? tc.accent : tc.textMuted,
              minWidth: 34,
            }}
          >
            {e.name}
          </Typography>
        </Tooltip>
        {/* Cap slider width so rows stay compact in the two-column panel. */}
        <Box sx={{ flex: 1, maxWidth: 140, minWidth: 80 }}>
          <Slider
            value={mag} min={-e.mag_max} max={e.mag_max} step={e.step_nm}
            onChange={(_, v) => updateMag(magKey, v as number)}
            size="small" sx={{ py: 0.5 }}
          />
        </Box>
        <Typography sx={{
          ...typography.value, color: isActive ? tc.accent : tc.textMuted,
          minWidth: 58, textAlign: "right",
        }}>
          {formatHOValue(mag, e.mag_max, e.display_scale)} {e.unit_display}
        </Typography>
        <IconButton
          size="small"
          aria-label={`Reset ${e.name} magnitude`}
          disabled={!isActive}
          onClick={() => resetMag(e.name, e.hasAngle)}
          sx={compactBareIconButton(tc)}
        >
          <RestartAltIcon sx={{ fontSize: 13 }} />
        </IconButton>
        {e.hasAngle && (
          <>
            <Box sx={{ flex: 1, maxWidth: 110, minWidth: 60 }}>
              <Slider
                value={ang} min={-180} max={180} step={1}
                onChange={(_, v) => updateAngle(e.name, v as number)}
                size="small" sx={{ py: 0.5 }}
                disabled={!isActive}
              />
            </Box>
            <Typography sx={{
              ...typography.value,
              color: hasAngleOffset ? tc.accent : tc.textMuted,
              minWidth: 38, textAlign: "right",
            }}>
              {ang.toFixed(0)}°
            </Typography>
            <IconButton
              size="small"
              aria-label={`Reset ${e.name} angle`}
              disabled={!hasAngleOffset}
              onClick={() => resetAngle(e.name)}
              sx={compactBareIconButton(tc)}
            >
              <RestartAltIcon sx={{ fontSize: 13 }} />
            </IconButton>
          </>
        )}
        {!e.hasAngle && (
          // Filler to keep column alignment consistent across rows.
          <Box sx={{ flex: 1, maxWidth: 176, minWidth: 0 }} />
        )}
      </Box>
    );
  }, [resetAngle, resetMag, tc, updateAngle, updateMag, values]);

  return (
    <Box sx={{
      display: "inline-flex",
      flexDirection: "column",
      width: open ? "max-content" : "fit-content",
      maxWidth: open ? "none" : "100%",
      border: `1px solid ${tc.border}`,
      bgcolor: tc.controlBg,
    }}>
      {/* Header row — click to expand/collapse, shows badge when active. */}
      <Box
        sx={{
          display: "inline-flex", alignItems: "center", gap: `${SPACING.MD}px`,
          px: 1, py: 0.5, cursor: "pointer", userSelect: "none",
          borderBottom: open ? `1px solid ${tc.border}` : "none",
        }}
        onClick={onToggle}
      >
        <Typography sx={{ ...typography.label, color: tc.accent, fontFamily: "monospace" }}>
          {open ? "▾" : "▸"} Higher-order (n=2..5)
        </Typography>
        {activeCount > 0 && (
          <Tooltip title={`${activeCount} higher-order magnitude(s) non-zero.  Reconstruction uses the 14-coef kernel; loss is not computed (optimizer only tracks C10/C12/phi12).`} placement="top" arrow>
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
  const { colors: tc } = useTheme();

  /* --- Model state (traitlets) --- */
  const [c10Min] = useModelState<number>("c10_min");
  const [c10Max] = useModelState<number>("c10_max");
  const [c12Min] = useModelState<number>("c12_min");
  const [c12Max] = useModelState<number>("c12_max");
  const [phi12Min] = useModelState<number>("phi12_min");
  const [phi12Max] = useModelState<number>("phi12_max");
  const [rotationMin] = useModelState<number>("rotation_min");
  const [rotationMax] = useModelState<number>("rotation_max");
  const [rotationDeg, setRotationDeg] = useModelState<number>("rotation_deg");
  const [flipPhase, setFlipPhase] = useModelState<boolean>("flip_phase");

  const [autoC10] = useModelState<number>("auto_c10");
  const [autoC12] = useModelState<number>("auto_c12");
  const [autoPhi12] = useModelState<number>("auto_phi12_deg");
  const [autoRotation] = useModelState<number>("auto_rotation_deg");
  const [autoLoss] = useModelState<number>("auto_loss");
  const [pixelSize] = useModelState<number>("pixel_size");

  const [, setRequestJson] = useModelState<string>("request_json");
  const [phaseBytes] = useModelState<DataView>("phase_bytes");
  const [phaseWidth] = useModelState<number>("phase_width");
  const [phaseHeight] = useModelState<number>("phase_height");
  const [resultJson] = useModelState<string>("result_json");
  const [, setPinJson] = useModelState<string>("pin_json");
  const [dragBfTrait, setDragBfTrait] = useModelState<number>("drag_bf");
  const [totalBf] = useModelState<number>("total_bf");
  const [starsPath] = useModelState<string>("stars_path");
  const [saveTrigger, setSaveTrigger] = useModelState<number>("save_trigger");
  const [notes, setNotes] = useModelState<string>("notes");
  const [calibrationPath] = useModelState<string>("calibration_path");
  const [calibrationSavedAt] = useModelState<string>("calibration_saved_at");
  const [trialsJson] = useModelState<string>("trials_json");
  const [initialPanelSize] = useModelState<number>("initial_panel_size");
  const [initialFftOn] = useModelState<boolean>("initial_fft_on");
  const [, setHigherOrderJson] = useModelState<string>("higher_order_json");
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

  /* --- Local state --- */
  const [c10, setC10] = React.useState(0);
  const [c12, setC12] = React.useState(0);
  const [phi12, setPhi12] = React.useState(0);
  type AberKey = "c10" | "c12" | "phi12" | "rot";
  type MainRanges = Record<AberKey, [number, number]>;
  const [uiRanges, setUiRanges] = React.useState<MainRanges>({
    c10: [-300, 300],
    c12: [-100, 100],
    phi12: [-180, 180],
    rot: [-180, 180],
  });
  const uiRangesSeededRef = React.useRef(false);
  React.useEffect(() => {
    if (uiRangesSeededRef.current) return;
    const arrived =
      c10Min !== 0 || c10Max !== 0 ||
      c12Min !== 0 || c12Max !== 0 ||
      phi12Min !== 0 || phi12Max !== 0 ||
      rotationMin !== 0 || rotationMax !== 0;
    if (!arrived) return;
    uiRangesSeededRef.current = true;
    setUiRanges({
      c10: [c10Min, c10Max],
      c12: [c12Min, c12Max],
      phi12: [phi12Min, phi12Max],
      rot: [rotationMin, rotationMax],
    });
  }, [c10Min, c10Max, c12Min, c12Max, phi12Min, phi12Max, rotationMin, rotationMax]);
  const c10UiMin = uiRanges.c10[0];
  const c10UiMax = uiRanges.c10[1];
  const c12UiMin = uiRanges.c12[0];
  const c12UiMax = uiRanges.c12[1];
  const phi12UiMin = uiRanges.phi12[0];
  const phi12UiMax = uiRanges.phi12[1];
  const rotationUiMin = uiRanges.rot[0];
  const rotationUiMax = uiRanges.rot[1];

  /* --- Higher-order aberration state.  Keys: C21_mag/C21_angle, ..., C56_mag/C56_angle,
         plus C30/C50 as bare magnitudes (no angle).  All values default to 0. */
  const [higherOrder, setHigherOrder] = React.useState<Record<string, number>>({});
  const [hoOpen, setHoOpen] = React.useState(false);
  // Count non-zero magnitudes so the UI can show "active" + user knows when
  // the loss readout stops reflecting the 3-param auto reference.
  const hoActiveCount = React.useMemo(() => {
    let n = 0;
    for (const key of Object.keys(higherOrder)) {
      if (!key.endsWith("_angle") && Math.abs(higherOrder[key] || 0) > 0) n += 1;
    }
    return n;
  }, [higherOrder]);
  const hoActive = hoActiveCount > 0;
  const hoActiveRef = React.useRef(false);
  hoActiveRef.current = hoActive;
  const hoActiveCountRef = React.useRef(0);
  hoActiveCountRef.current = hoActiveCount;
  const higherOrderRef = React.useRef<Record<string, number>>({});
  higherOrderRef.current = higherOrder;
  const [loss, setLoss] = React.useState<number | null>(null);
  const [gpuMs, setGpuMs] = React.useState<number | null>(null);
  const [uiMs, setUiMs] = React.useState<number | null>(null);
  const [jsMs, setJsMs] = React.useState<number | null>(null);
  const [stageTiming, setStageTiming] = React.useState<{
    d2h: number; bytes: number; trait: number; pyTotal: number;
    clip: number; render: number; setState: number; paint: number;
  } | null>(null);
  const lastStagesRef = React.useRef<{
    d2h: number; bytes: number; trait: number; pyTotal: number;
    clip: number; render: number; setState: number; paint: number;
  } | null>(null);
  const lastGpuMsRef = React.useRef<number | null>(null);
  const lastUiMsRef = React.useRef<number | null>(null);
  const [busy, setBusy] = React.useState(false);
  // Mirror the trait so the UI and send-path both read from the same state.
  // Local state tracks the slider thumb during drag so we don't thrash Python
  // rebuilding BF state on every pixel — we only commit on release.
  const [localDragBf, setLocalDragBf] = React.useState(dragBfTrait ?? 0);
  const dragBfRef = React.useRef(dragBfTrait ?? 0);
  const defaultBfCount = React.useCallback((total: number) => {
    const n = Math.max(1, Math.round(total || 1));
    return Math.max(1, Math.min(n, Math.round(n * DEFAULT_BF_FRACTION)));
  }, []);
  const webgpuCalLogicalBf = React.useMemo(() => {
    if (!webgpuCalJson) return 0;
    try {
      const cal = JSON.parse(webgpuCalJson);
      const numBf = Number(cal?.num_bf || 0);
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
  const standaloneBfSeededRef = React.useRef(false);
  React.useEffect(() => {
    const total = effectiveTotalBf;
    const raw = Math.round(dragBfTrait ?? 0);
    const seedStandaloneDefault = webgpuStandalone && !standaloneBfSeededRef.current && total > 0;
    const nextRaw = total > 0 && (raw <= 0 || seedStandaloneDefault) ? defaultBfCount(total) : raw;
    const next = total > 0 ? Math.max(1, Math.min(total, nextRaw)) : nextRaw;
    setLocalDragBf(next);
    dragBfRef.current = next;
    if (seedStandaloneDefault) standaloneBfSeededRef.current = true;
    if (total > 0 && (raw <= 0 || seedStandaloneDefault) && raw !== next) {
      setDragBfTrait(next);
    }
  }, [defaultBfCount, dragBfTrait, effectiveTotalBf, setDragBfTrait, webgpuStandalone]);
  const [pinned, setPinned] = React.useState<PinnedEntry[]>([]);
  const [viewPin, setViewPin] = React.useState<number | null>(null);
  const [exportStatus, setExportStatus] = React.useState("");
  const [animationExportStatus, setAnimationExportStatus] = React.useState("");
  const [animationExportBusy, setAnimationExportBusy] = React.useState(false);
  const [toolbarMoreAnchor, setToolbarMoreAnchor] = React.useState<HTMLElement | null>(null);
  const [dragOverPin, setDragOverPin] = React.useState<number | null>(null);
  const [draggingPin, setDraggingPin] = React.useState<number | null>(null);
  const [newPinId, setNewPinId] = React.useState<number | null>(null);
  const [moreMenuAnchor, setMoreMenuAnchor] = React.useState<HTMLElement | null>(null);
  const draggedPinRef = React.useRef<number | null>(null);
  const didDragPinRef = React.useRef(false);
  const pinPointerDragRef = React.useRef<{ id: number; x: number; y: number; active: boolean } | null>(null);
  const pinLayoutBeforeRef = React.useRef<Map<number, DOMRect> | null>(null);

  const [phaseZoom, setPhaseZoom] = React.useState(ZOOM_RESET);
  const [ampZoom, setAmpZoom] = React.useState(ZOOM_RESET);
  const [complexZoom, setComplexZoom] = React.useState(ZOOM_RESET);
  const [fftZoom, setFFTZoom] = React.useState(ZOOM_RESET);
  const [phaseOff, setPhaseOff] = React.useState<HTMLCanvasElement | null>(null);
  const [ampOff, setAmpOff] = React.useState<HTMLCanvasElement | null>(null);
  const [complexOff, setComplexOff] = React.useState<HTMLCanvasElement | null>(null);
  const [fftOff, setFFTOff] = React.useState<HTMLCanvasElement | null>(null);
  const [cropSelecting, setCropSelecting] = React.useState(false);
  const [scanCrop, setScanCrop] = React.useState<[number, number, number, number] | null>(null);
  const [cropRefitPending, setCropRefitPending] = React.useState(false);
  React.useEffect(() => {
    const rows = Math.round(scanRows || phaseHeight || 0);
    const cols = Math.round(scanCols || phaseWidth || 0);
    setScanCrop(previous => {
      if (!previous) return null;
      const [r0, r1, c0, c1] = previous;
      if (r0 >= 0 && c0 >= 0 && r1 <= rows && c1 <= cols) return previous;
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
    const minSpan = Math.min(32, rows, cols);
    let r0 = Math.max(0, Math.min(startRow, endRow));
    let r1 = Math.min(rows, Math.max(startRow, endRow) + 1);
    let c0 = Math.max(0, Math.min(startCol, endCol));
    let c1 = Math.min(cols, Math.max(startCol, endCol) + 1);
    if (r1 - r0 < minSpan) r1 = Math.min(rows, r0 + minSpan);
    if (r1 - r0 < minSpan) r0 = Math.max(0, r1 - minSpan);
    if (c1 - c0 < minSpan) c1 = Math.min(cols, c0 + minSpan);
    if (c1 - c0 < minSpan) c0 = Math.max(0, c1 - minSpan);
    setScanCrop([r0, r1, c0, c1]);
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
      id: Date.now(), scan_region: scanCrop, n_trials: 200,
    }));
  }, [cropRefitAvailable, scanCrop, setCropRefitRequestJson]);
  const cropSummary = React.useMemo(() => {
    if (!scanCrop) return "Draw a region";
    const [r0, r1, c0, c1] = scanCrop;
    return `r ${r0}:${r1}  c ${c0}:${c1}  ${r1 - r0}x${c1 - c0}`;
  }, [scanCrop]);

  // Cached data for re-rendering without recomputing FFT
  const rawPhaseRef = React.useRef<{ data: Float32Array; w: number; h: number } | null>(null);
  const fftMagRef = React.useRef<{ mag: Float32Array; w: number; h: number; pw: number; ph: number } | null>(null);

  // UI toggles — FFT defaults OFF so the drag path is phase-render-only (fastest).
  // ``fft_on=`` / ``size=`` kwargs from Python override the defaults via a
  // one-shot effect below; useState's lazy initializer can't see the trait
  // value yet because anywidget traits arrive asynchronously on mount.
  const [showFFT, setShowFFT] = React.useState<boolean>(false);
  const [fftPlacement, setFFTPlacement] = React.useState<FFTPlacement>("panel");
  const [fftInsetBox, setFFTInsetBox] = React.useState<FFTInsetBox>({ x: 0.72, y: 0.02, size: 0.28 });
  const [extraRealViews, setExtraRealViews] = React.useState<Record<ExtraRealViewMode, boolean>>({ amp: false, complex: false });
  const toolbarMoreActiveCount =
    Number(extraRealViews.amp) + Number(extraRealViews.complex) + Number(cropSelecting || Boolean(scanCrop));
  const [smooth, setSmooth] = React.useState<boolean>(false);
  const [cmap, setCmap] = React.useState("viridis");
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
  const flipPhaseRef = React.useRef(flipPhase);
  flipPhaseRef.current = !!flipPhase;
  const displayDataCacheRef = React.useRef<{ source: Float32Array | null; abs: Float32Array | null }>({ source: null, abs: null });
  const activeRealDataRef = React.useRef<{ data: Float32Array; w: number; h: number; mode: RealViewMode } | null>(null);
  const ampDataRef = React.useRef<{ data: Float32Array; w: number; h: number } | null>(null);

  const [panel, setPanel] = React.useState<number>(DEFAULT_PANEL);

  // One-shot seed from Python kwargs ``size=`` and ``fft_on=``.  Runs once
  // (guarded by a ref) after the traits arrive from the kernel, so user
  // interactions with the resize handle / FFT switch afterwards stick.
  const kwargsAppliedRef = React.useRef(false);
  React.useEffect(() => {
    if (kwargsAppliedRef.current) return;
    if (initialPanelSize == null || initialFftOn == null) return;
    kwargsAppliedRef.current = true;
    const n = Number(initialPanelSize);
    if (Number.isFinite(n) && n >= MIN_PANEL) setPanel(n);
    if (initialFftOn) setShowFFT(true);
  }, [initialPanelSize, initialFftOn]);
  const resizeDrag = React.useRef({ on: false, y0: 0, s0: 0 });
  const requestIdRef = React.useRef(0);
  const frontendPreviewRef = React.useRef<((c10Val: number, c12Val: number, phi12Val: number, rotationVal: number) => boolean) | null>(null);
  const frontendFullRef = React.useRef<((c10Val: number, c12Val: number, phi12Val: number, rotationVal: number) => boolean) | null>(null);
  const shouldCommitOnReleaseRef = React.useRef(false);
  const [webgpuRuntimeStatus, setWebgpuRuntimeStatus] = React.useState("");
  const [webgpuLoadProgress, setWebgpuLoadProgress] = React.useState<WebGPULoadProgress | null>(null);
  const webgpuSsbRef = React.useRef<WebGPUSSBBackend | null>(null);
  const webgpuInFlightRef = React.useRef(false);
  const webgpuPendingRef = React.useRef<[number, number, number, number] | null>(null);
  const webgpuPendingFullRef = React.useRef(false);
  const selectedDragBfCount = React.useCallback(() => {
    const total = Math.max(1, effectiveTotalBf || 1);
    const requested = Math.round(dragBfRef.current || 0);
    if (requested <= 0) return defaultBfCount(total);
    return Math.max(1, Math.min(total, requested));
  }, [defaultBfCount, effectiveTotalBf]);

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
        // Validate old exports with root cal.json and clean exports with
        // snapshots/cal.json; calibration is embedded, this only catches a
        // wrong folder selection before source/saves reads fail later.
        try {
          await handle.getFileHandle("cal.json");
        } catch {
          try {
            const snapshots = await handle.getDirectoryHandle("snapshots");
            await snapshots.getFileHandle("cal.json");
          } catch {
            setLocalSourceError(
              `That folder has no calibration snapshot - select the folder named "${localFolderName || "the one containing this page"}" (the folder this index.html lives in).`,
            );
            return;
          }
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
    timestamp: string;
    C10: number;
    C12: number;
    phi12_deg: number;
    rotation_deg: number;
    flip_phase: boolean;
    loss: number | null;
    bf: number;
    notes: string;
    image: string;
  };
  const [folderSaves, setFolderSaves] = React.useState<FolderSaveRecord[]>([]);
  const [folderSaveStatus, setFolderSaveStatus] = React.useState("");
  const refreshFolderSaves = React.useCallback(async () => {
    const current = await readSSBFolderJson<FolderSaveRecord[]>("snapshots/snapshots.json");
    setFolderSaves(Array.isArray(current) ? current : []);
  }, []);
  React.useEffect(() => {
    if (webgpuStandalone && localSourceGranted) void refreshFolderSaves();
  }, [webgpuStandalone, localSourceGranted, refreshFolderSaves]);
  const onLocalDirInput = React.useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const names = new Set(Array.from(files).flatMap((f) => [
        f.name,
        (f as File & { webkitRelativePath?: string }).webkitRelativePath || "",
      ]));
      const hasCalibration = names.has("cal.json")
        || Array.from(names).some((name) => name.endsWith("snapshots/cal.json"));
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
    const hasH5Source = !!webgpuH5SourceJson && webgpuH5SourceJson.trim() !== "" && webgpuH5SourceJson.trim() !== "{}";
    if (!localSourceGranted) {
      webgpuSsbRef.current = null;
      setWebgpuRuntimeStatus("No-server mode: grant the folder containing this HTML to start");
      setWebgpuLoadProgress(null);
      return;
    }
    if (!webgpuPreviewEnabled || !webgpuCalJson || !hasH5Source) {
      webgpuSsbRef.current = null;
      setWebgpuRuntimeStatus(webgpuPreviewStatus || "");
      setWebgpuLoadProgress(null);
      return;
    }
    try {
      const parsedSource = JSON.parse(webgpuH5SourceJson) as { kind?: string };
      const source = {
        kind: parsedSource.kind === "bf_columns" ? "bf_columns" as const : "hdf5" as const,
        json: webgpuH5SourceJson,
      };
      const engine = new WebGPUSSBBackend(webgpuCalJson, source);
      engine.setProgressHandler((progress) => {
        setWebgpuLoadProgress(progress.stage === "ready" ? null : progress);
        setWebgpuRuntimeStatus(webgpuProgressLabel(progress));
      });
      webgpuSsbRef.current = engine;
      setWebgpuRuntimeStatus(engine.readyLabel);
      setWebgpuLoadProgress(null);
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
  }, [localSourceGranted, webgpuPreviewEnabled, webgpuCalJson, webgpuH5SourceJson, webgpuPreviewStatus]);

  // WebGPU FFT (async, off main thread). Null until init resolves or if unsupported.
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  // Generation counter — discards stale async FFT results when newer data arrives.
  const fftGenRef = React.useRef(0);
  React.useEffect(() => {
    let cancelled = false;
    getWebGPUFFT().then(fft => {
      if (!cancelled && fft) gpuFFTRef.current = fft;
    });
    return () => { cancelled = true; };
  }, []);

  // WebGPU colormap engine.
  // ShowPtycho uses the shared WebGPU colormap engine only for the primary
  // phase panel.  The engine has one LUT binding, so using it concurrently for
  // FFT/amplitude can make those panels leak their colormap back into phase.
  // Extra panels use the deterministic CPU path until the shared engine grows
  // true per-slot LUT bindings.
  // Generation counter discards stale async GPU results if newer data lands first.
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const gpuCmapGenRef = React.useRef<Record<GPUColormapSlot, number>>({ 0: 0, 1: 0, 2: 0 });
  const gpuSlotCanvasRef = React.useRef<Record<GPUColormapSlot, HTMLCanvasElement | null>>({ 0: null, 1: null, 2: null });
  const gpuCmapBusyRef = React.useRef<Record<GPUColormapSlot, boolean>>({ 0: false, 1: false, 2: false });
  const gpuCmapPendingRef = React.useRef<Record<GPUColormapSlot, (() => void) | null>>({ 0: null, 1: null, 2: null });
  const [phaseVersion, setPhaseVersion] = React.useState(0);
  const [fftVersion, setFftVersion] = React.useState(0);
  const [ampVersion, setAmpVersion] = React.useState(0);
  const gpuCmapReadyRef = React.useRef(false);
  // Track the LUT currently uploaded to each slot so we only re-upload when the cmap changes.
  const gpuCmapCurrentLutRef = React.useRef<Record<GPUColormapSlot, string | null>>({ 0: null, 1: null, 2: null });
  // Track the data uploaded to each slot so we only re-upload on real data change.
  const gpuSlotDataRef = React.useRef<Record<GPUColormapSlot, Float32Array | null>>({ 0: null, 1: null, 2: null });
  React.useEffect(() => {
    let cancelled = false;
    getGPUColormapEngine().then(engine => {
      if (cancelled) return;
      if (engine) {
        gpuCmapRef.current = engine;
        gpuCmapReadyRef.current = true;
        console.log("[showptycho] WebGPU colormap engine initialized");
      } else {
        console.log("[showptycho] no WebGPU colormap — falling back to CPU renderToOffscreen");
      }
    });
    return () => { cancelled = true; };
  }, []);

  /* --- Init sliders from auto --- */
  const initRef = React.useRef(false);
  React.useEffect(() => {
    if (!initRef.current && autoC10 !== undefined) {
      setC10(autoC10); setC12(autoC12); setPhi12(autoPhi12);
      initRef.current = true;
    }
  }, [autoC10, autoC12, autoPhi12]);

  /* --- Push higher-order state to Python as JSON whenever it changes.
         In exported WebGPU folders there is no Python kernel, so a second
         debounced effect below sends the same current slider state through the
         browser-side 14-coef path. */
  const hoDebounceRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    if (hoDebounceRef.current != null) window.clearTimeout(hoDebounceRef.current);
    hoDebounceRef.current = window.setTimeout(() => {
      setHigherOrderJson(JSON.stringify(higherOrder));
    }, 20);
    return () => {
      if (hoDebounceRef.current != null) {
        window.clearTimeout(hoDebounceRef.current);
        hoDebounceRef.current = null;
      }
    };
  }, [higherOrder, setHigherOrderJson]);

  /* --- Request plumbing ---
   * Default: slider events use a BF fraction for responsive exploration. Set
   * the BF control to the full count when the current view is ready for final
   * review.
   * In-flight throttling: only one drag request outstanding at a time. */
  const sendTimesRef = React.useRef<Map<number, number>>(new Map());
  const inflightDragRef = React.useRef(false);
  const pendingDragRef = React.useRef<[number, number, number] | null>(null);

  const fireRequest = React.useCallback((c10Val: number, c12Val: number, phi12Val: number, committed: boolean) => {
    const id = ++requestIdRef.current;
    sendTimesRef.current.set(id, performance.now());
    setBusy(true);
    setViewPin(null);
    setRequestJson(JSON.stringify({ id, c10: c10Val, c12: c12Val, phi12_deg: phi12Val, committed }));
  }, [setRequestJson]);

  const sendDrag = React.useCallback((c10Val: number, c12Val: number, phi12Val: number, rotationVal = rotationDegRef.current) => {
    if (frontendPreviewRef.current?.(c10Val, c12Val, phi12Val, rotationVal)) return;
    if (inflightDragRef.current) {
      pendingDragRef.current = [c10Val, c12Val, phi12Val];
      return;
    }
    inflightDragRef.current = true;
    fireRequest(c10Val, c12Val, phi12Val, false);
  }, [fireRequest]);

  const sendCommit = React.useCallback((c10Val: number, c12Val: number, phi12Val: number, rotationVal = rotationDegRef.current) => {
    pendingDragRef.current = null;
    if (frontendFullRef.current?.(c10Val, c12Val, phi12Val, rotationVal)) return;
    fireRequest(c10Val, c12Val, phi12Val, true);
  }, [fireRequest]);

  const commitDragBf = React.useCallback((val: number) => {
    const total = Math.max(1, effectiveTotalBf || 1);
    const count = Math.max(1, Math.min(total, Math.round(val)));
    dragBfRef.current = count;
    setLocalDragBf(count);
    setDragBfTrait(count);
    const engine = webgpuSsbRef.current;
    if (!engine) return;
    setWebgpuLoadProgress({
      stage: "gather",
      message: count >= total ? "Preparing full-BF ptychography view" : "Preparing preview BF ptychography view",
      detail: `${count}/${total} BF pixels selected`,
      current: count,
      total,
      percent: Math.min(95, Math.max(5, (count / total) * 90)),
      activeBf: count,
      totalBf: total,
    });
    setWebgpuRuntimeStatus(`Preparing ${count}/${total} BF reducer…`);
    engine.prepareBfCount(count).then(prepared => {
      setWebgpuLoadProgress(null);
      setWebgpuRuntimeStatus(`WebGPU folder ready: ${prepared}/${total} BF reducer`);
      const current = sliderVals.current;
      const fn = prepared >= total ? frontendFullRef.current : frontendPreviewRef.current;
      fn?.(current.c10, current.c12, current.phi12, rotationDegRef.current);
    }).catch(err => {
      const message = err instanceof Error ? err.message : String(err);
      setWebgpuLoadProgress({
        stage: "error",
        message: "BF reducer unavailable",
        detail: message,
        percent: 0,
      });
      setWebgpuRuntimeStatus(`BF reducer unavailable: ${message}`);
    });
  }, [effectiveTotalBf, setDragBfTrait]);

  /* --- Slider values ref (so callbacks read fresh values without re-creating) --- */
  const sliderVals = React.useRef({ c10: 0, c12: 0, phi12: 0 });
  sliderVals.current = { c10, c12, phi12 };
  const hoFrontendDebounceRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    if (hoFrontendDebounceRef.current != null) window.clearTimeout(hoFrontendDebounceRef.current);
    hoFrontendDebounceRef.current = window.setTimeout(() => {
      const engine = webgpuSsbRef.current;
      const preview = frontendPreviewRef.current;
      if (!engine || !preview || !initRef.current) return;
      const current = sliderVals.current;
      preview(current.c10, current.c12, current.phi12, rotationDegRef.current);
    }, 20);
    return () => {
      if (hoFrontendDebounceRef.current != null) {
        window.clearTimeout(hoFrontendDebounceRef.current);
        hoFrontendDebounceRef.current = null;
      }
    };
  }, [higherOrder]);

  /* --- Play control: sweep one parameter over its range ---
     Like Show3D playback: pick which aberration to sweep, set FPS, hit play.
     Each tick advances the sweep index and fires a normal drag/commit request
     so the image updates with the same pipeline users see when they drag a slider.
     For 30 FPS responsiveness on very large full-BF disks, reduce BF subset via
     the existing drag_bf slider. */
  // SweepParam: one of the main-slider keys, or a higher-order key of the form
  // "ho:<hoKey>" where <hoKey> is either a magnitude key (e.g. "C32_mag",
  // "C30") or an angle key ("C32_angle").
  type SweepParam = string;
  const [playing, setPlaying] = React.useState(false);
  const [playFps, setPlayFps] = React.useState(10);

  /* --- Optuna trials panel state.  Parsed once from the trait, renders as
     a loss-ranked strip the user can click through to preview any trial.
     Collapsed by default when there are many trials so the widget doesn't
     dominate the notebook scroll. --- */
  interface OptunaTrial { rank: number; C10: number; C12: number; phi12_deg: number; loss: number; }
  const trials: OptunaTrial[] = React.useMemo(() => {
    if (!trialsJson) return [];
    try {
      const parsed = JSON.parse(trialsJson);
      return Array.isArray(parsed) ? parsed : [];
    } catch { return []; }
  }, [trialsJson]);
  const [trialsExpanded, setTrialsExpanded] = React.useState(false);
  const [activeTrialRank, setActiveTrialRank] = React.useState<number | null>(null);

  const doViewTrial = React.useCallback((t: OptunaTrial) => {
    setActiveTrialRank(t.rank);
    setViewPin(null);   // trials + pins are mutually exclusive selections
    setC10(t.C10); setC12(t.C12); setPhi12(t.phi12_deg);
    sendCommit(t.C10, t.C12, t.phi12_deg);
  }, [sendCommit]);
  const [sweepParam, setSweepParam] = React.useState<SweepParam>("c10");
  const [sweepSteps] = React.useState(40);  // fixed 40-frame sweep; good balance
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
      if (hoKey.endsWith("_angle")) return [-180, 180];
      const baseName = hoKey.endsWith("_mag") ? hoKey.slice(0, -4) : hoKey;
      for (const order of [2, 3, 4, 5]) {
        const e = HO_BY_ORDER[order].find(x => x.name === baseName);
        if (e) return [-e.mag_max, e.mag_max];
      }
      return [0, 0];
    }
    switch (param) {
      case "c10":   return [c10UiMin, c10UiMax];
      case "c12":   return [c12UiMin, c12UiMax];
      case "phi12": return [phi12UiMin, phi12UiMax];
      case "rot":   return [rotationUiMin, rotationUiMax];
    }
    return [0, 0];  // unreachable; keeps TS happy with future SweepParam variants
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
  // thumb at zero — rotation drag looks dead until the ranges refresh.  Gate
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
      const frac = sweepSteps <= 1 ? 0 : sweepIdxRef.current / (sweepSteps - 1);
      const val = lo + frac * (hi - lo);
      // Apply the value, using the same code path the sliders would
      if (hoKey) {
        // Update the higher-order state.  The observer in Python fires when
        // higher_order_json changes (debounced 20 ms in the other effect), so
        // this indirectly triggers reconstruct_full through the 14-coef path.
        setHigherOrder(prev => ({ ...prev, [hoKey!]: val }));
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
        setC10(val); sendDrag(val, sliderVals.current.c12, sliderVals.current.phi12);
      } else if (sweepParam === "c12") {
        setC12(val); sendDrag(sliderVals.current.c10, val, sliderVals.current.phi12);
      } else if (sweepParam === "phi12") {
        setPhi12(val); sendDrag(sliderVals.current.c10, sliderVals.current.c12, val);
      } else if (sweepParam === "rot") {
        setRotationDeg(val);
        if (webgpuSsbRef.current) sendDrag(sliderVals.current.c10, sliderVals.current.c12, sliderVals.current.phi12, val);
      }
      // Advance index (boomerang — bounces between 0 and sweepSteps-1)
      sweepIdxRef.current += sweepDirRef.current;
      if (sweepIdxRef.current >= sweepSteps - 1) {
        sweepIdxRef.current = sweepSteps - 1;
        sweepDirRef.current = -1;
      } else if (sweepIdxRef.current <= 0) {
        sweepIdxRef.current = 0;
        sweepDirRef.current = 1;
      }
    };
    tick();  // fire once immediately so user sees motion without a delay
    const iv = window.setInterval(tick, Math.max(33, 1000 / playFps));
    return () => window.clearInterval(iv);
    // sendDrag, setRotationDeg, and the slider setters are stable callbacks;
    // c10Min etc. come from traits that only change at init, so their
    // dependency is safe to include implicitly via the closure.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, playFps, sweepParam, sweepSteps]);

  /* --- Process result from Python --- */
  React.useEffect(() => {
    if (!resultJson) return;
    let r: any;
    try { r = JSON.parse(resultJson); } catch { return; }
    const rid = r.id;
    const sendT = sendTimesRef.current.get(rid);
    if (sendT !== undefined) {
      const measured = performance.now() - sendT;
      setUiMs(measured);
      lastUiMsRef.current = measured;
      sendTimesRef.current.delete(rid);
    }
    inflightDragRef.current = false;
    if (r.error) { setBusy(false); return; }
    if (r.loss != null) setLoss(r.loss);
    setGpuMs(r.time_ms);
    lastGpuMsRef.current = r.time_ms;
    setBusy(false);
    // Carry Python stage times forward for the stats readout
    if (typeof r.d2h_ms === "number") {
      lastStagesRef.current = {
        ...(lastStagesRef.current || { clip: 0, render: 0, setState: 0, paint: 0 }),
        d2h: r.d2h_ms, bytes: r.bytes_ms, trait: r.trait_ms, pyTotal: r.py_total_ms,
      };
    }

    // Drain pending drag value if user moved the slider while we were busy.
    const pending = pendingDragRef.current;
    if (pending) {
      pendingDragRef.current = null;
      inflightDragRef.current = true;
      fireRequest(pending[0], pending[1], pending[2], false);
    }
  }, [resultJson, fireRequest]);

  /* --- Data ranges for histograms (populated alongside render) --- */
  const [dataRange, setDataRange] = React.useState({ min: 0, max: 1 });
  const [fftDataRange, setFftDataRange] = React.useState({ min: 0, max: 1 });

  /* --- GPU colormap render for phase — uploads data once, re-applies cmap
         shader when contrast/cmap changes.  Falls back to the CPU path if
         the WebGPU engine isn't available on this browser. --- */
  const renderPhaseGPU = React.useCallback((data: Float32Array, w: number, h: number, slot: GPUColormapSlot, cmapName: string, pctLo: number, pctHi: number) => {
    if (slot !== 0) return false;
    const engine = gpuCmapRef.current;
    if (!engine || !gpuCmapReadyRef.current) return false;
    const gen = ++gpuCmapGenRef.current[slot];
    const launch = () => {
      gpuCmapBusyRef.current[slot] = true;
      // Upload data only if it actually changed (new reconstruction), not on every contrast tick.
      const prev = gpuSlotDataRef.current[slot];
      if (prev !== data) {
        engine.uploadData(slot, data, w, h);
        gpuSlotDataRef.current[slot] = data;
      }
      // Upload LUT only when cmap changes (cheap but still avoid per-frame work).
      const fullLutKey = `${slot}:${cmapName}`;
      if (gpuCmapCurrentLutRef.current[slot] !== fullLutKey) {
        const lut = COLORMAPS[cmapName as keyof typeof COLORMAPS] || COLORMAPS.viridis;
        engine.uploadLUT(cmapName, lut);
        gpuCmapCurrentLutRef.current[slot] = fullLutKey;
      }
      // Compute vmin/vmax from data+percentiles on CPU (cheap, ~1 ms for 512²).
      const { vmin, vmax, min, max } = percentileClip(data, pctLo, pctHi);
      // Keep at most one GPU colormap pass in flight per slot. Histogram input
      // can arrive faster than GPU readback; queueing every tick makes old
      // ranges drain later and looks like the phase contrast flips/jitters.
      // applySingle keeps the colormap compute on WebGPU but avoids the flaky
      // OffscreenCanvas/ImageBitmap snapshot path.
      engine.applySingle(slot, vmin, vmax, false).then(rgba => {
        if (gen !== gpuCmapGenRef.current[slot] || !rgba) return;
        let canvas = gpuSlotCanvasRef.current[slot];
        const fresh = !canvas || canvas.width !== w || canvas.height !== h;
        if (fresh) {
          canvas = document.createElement("canvas");
          canvas.width = w; canvas.height = h;
          gpuSlotCanvasRef.current[slot] = canvas;
        }
        const ctx = canvas!.getContext("2d");
        if (!ctx) return;
        const image = ctx.createImageData(w, h);
        image.data.set(rgba);
        ctx.putImageData(image, 0, 0);
        if (slot === 0) {
          if (fresh) setPhaseOff(canvas!);
          setPhaseVersion(v => v + 1);
          setDataRange({ min, max });
        } else if (slot === 1) {
          if (fresh) setFFTOff(canvas!);
          setFftVersion(v => v + 1);
          setFftDataRange({ min, max });
        } else {
          if (fresh) setAmpOff(canvas!);
          setAmpVersion(v => v + 1);
        }
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
    return true;
  }, []);

  /* --- Re-render real-space displays when additive views/contrast/cmap changes. --- */
  const renderRealDisplay = React.useCallback((phase: Float32Array, w: number, h: number) => {
    const displayData = phaseDisplayData(phase, "phase", displayDataCacheRef);
    activeRealDataRef.current = { data: displayData, w, h, mode: "phase" };
    // GPU-first: the applySingle shader is ~300× faster than the CPU loop.
    if (renderPhaseGPU(displayData, w, h, 0, cmapRef.current, contrastRange[0], contrastRange[1])) {
      const extras = extraRealViewsRef.current;
      if (extras.amp || extras.complex) {
        const ampData = phaseDisplayData(phase, "amp", displayDataCacheRef);
        ampDataRef.current = { data: ampData, w, h };
        if (extras.amp) {
          if (!renderPhaseGPU(ampData, w, h, 2, cmapRef.current, contrastRange[0], contrastRange[1])) {
            const lut = COLORMAPS[cmapRef.current as keyof typeof COLORMAPS] || COLORMAPS.viridis;
            setAmpOff(renderPhaseOffscreen(ampData, w, h, lut, contrastRange[0], contrastRange[1]).canvas);
          }
        } else {
          setAmpOff(null);
        }
        if (extras.complex) {
          setComplexOff(renderComplexPhaseOffscreen(phase, w, h, contrastRange[0], contrastRange[1]).canvas);
        } else {
          setComplexOff(null);
        }
      } else {
        ampDataRef.current = null;
        setAmpOff(null);
        setComplexOff(null);
      }
      return { clipMs: 0, renderMs: 0, min: 0, max: 1, canvas: null, gpuHandled: true };
    }
    // CPU fallback — WebGPU colormap unavailable in this browser/context.
    const lut = COLORMAPS[cmapRef.current as keyof typeof COLORMAPS] || COLORMAPS.viridis;
    const cpu = renderPhaseOffscreen(displayData, w, h, lut, contrastRange[0], contrastRange[1]);
    setPhaseOff(cpu.canvas);
    setDataRange({ min: cpu.min, max: cpu.max });
    const extras = extraRealViewsRef.current;
    if (extras.amp || extras.complex) {
      const ampData = phaseDisplayData(phase, "amp", displayDataCacheRef);
      ampDataRef.current = { data: ampData, w, h };
      setAmpOff(extras.amp ? renderPhaseOffscreen(ampData, w, h, lut, contrastRange[0], contrastRange[1]).canvas : null);
      setComplexOff(extras.complex ? renderComplexPhaseOffscreen(phase, w, h, contrastRange[0], contrastRange[1]).canvas : null);
    } else {
      ampDataRef.current = null;
      setAmpOff(null);
      setComplexOff(null);
    }
    return { clipMs: cpu.clipMs, renderMs: cpu.renderMs, min: cpu.min, max: cpu.max, canvas: cpu.canvas, gpuHandled: false };
  }, [contrastRange, renderPhaseGPU]);

  /* --- Re-render FFT when its contrast/cmap changes (no FFT recompute).
         GPU-first; CPU fallback.  Uses the user-selected FFT colormap
         (defaults to inferno but follows the dropdown state). --- */
  const rerenderFFT = React.useCallback(() => {
    const f = fftMagRef.current;
    if (!f) return;
    // Try GPU path (slot 1 reserved for FFT magnitude).
    if (renderPhaseGPU(f.mag, f.w, f.h, 1, fftCmap, fftContrastRange[0], fftContrastRange[1])) return;
    // CPU fallback
    const lut = COLORMAPS[fftCmap as keyof typeof COLORMAPS] || COLORMAPS.inferno;
    const { canvas, min, max } = renderFFTOffscreen(f.mag, f.w, f.h, lut, fftContrastRange[0], fftContrastRange[1]);
    setFFTOff(canvas);
    setFftDataRange({ min, max });
  }, [fftContrastRange, fftCmap, renderPhaseGPU]);

  /* --- Full render: real-space view synchronous, FFT only when requested. --- */
  const renderPreviewPhase = React.useCallback((data: Float32Array, w: number, h: number) => {
    const t0 = performance.now();
    publishShowPtychoTestPhase(data, w, h);
    rawPhaseRef.current = { data, w, h };
    if (autoContrastRef.current) {
      const r = contrastRef.current;
      if (r[0] !== 1 || r[1] !== 99) setContrastRange([1, 99]);
    }
    const rendered = renderRealDisplay(data, w, h);
    const { clipMs, renderMs, canvas, min, max, gpuHandled } = rendered;
    const tAfterRender = performance.now();
    if (!gpuHandled) {
      setPhaseOff(canvas);
      setDataRange({ min, max });
    }
    const tAfterState = performance.now();
    setJsMs(tAfterState - t0);
    lastStagesRef.current = {
      ...(lastStagesRef.current || { d2h: 0, bytes: 0, trait: 0, pyTotal: 0 }),
      clip: clipMs,
      render: renderMs,
      setState: tAfterState - tAfterRender,
      paint: 0,
    };
    requestAnimationFrame(() => {
      const paintMs = performance.now() - tAfterState;
      lastStagesRef.current = { ...lastStagesRef.current!, paint: paintMs };
      setStageTiming({ ...lastStagesRef.current! });
    });
  }, [renderRealDisplay]);

  const renderAll = React.useCallback((data: Float32Array, w: number, h: number) => {
    const t0 = performance.now();
    publishShowPtychoTestPhase(data, w, h);
    rawPhaseRef.current = { data, w, h };
    // Auto-contrast: new data snaps clip back to 1-99 percentile on every
    // reconstruction.  The user can override via the histogram slider, which
    // flips auto OFF until they re-enable it.
    if (autoContrastRef.current) {
      const r = contrastRef.current;
      if (r[0] !== 1 || r[1] !== 99) setContrastRange([1, 99]);
    }
    if (showFFTRef.current && fftAutoContrastRef.current) {
      const fr = fftContrastRef.current;
      if (fr[0] !== 1 || fr[1] !== 99) setFftContrastRange([1, 99]);
    }
    const rendered = renderRealDisplay(data, w, h);
    const { clipMs, renderMs, canvas, min, max, gpuHandled } = rendered;
    const tAfterRender = performance.now();
    if (!gpuHandled) {
      setPhaseOff(canvas);
      setDataRange({ min, max });
    }
    const tAfterState = performance.now();
    setJsMs(tAfterState - t0);
    lastStagesRef.current = {
      ...(lastStagesRef.current || { d2h: 0, bytes: 0, trait: 0, pyTotal: 0 }),
      clip: clipMs,
      render: renderMs,
      setState: tAfterState - tAfterRender,
      paint: 0,
    };
    // Measure paint latency: from setState dispatch to next frame commit.
    requestAnimationFrame(() => {
      const paintMs = performance.now() - tAfterState;
      lastStagesRef.current = { ...lastStagesRef.current!, paint: paintMs };
      setStageTiming({ ...lastStagesRef.current! });
      // Log every frame to DevTools so the user can read real per-drag numbers.
      // Format: [showptycho] GPU=XX d2h=X bytes=X trait=X clip=X render=X setState=X paint=X | UI=XX gap=XX
      const s = lastStagesRef.current!;
      const gSize = w * h * 4;
      const ui = (typeof lastUiMsRef.current === "number") ? lastUiMsRef.current : null;
      const gap = ui != null ? ui - (s.d2h + s.bytes + s.trait + s.clip + s.render + s.setState + s.paint) : null;
      console.log(
        `[showptycho] ${w}×${h} (${(gSize/1024).toFixed(0)}KB) | ` +
        `GPU=${(lastGpuMsRef.current ?? 0).toFixed(1)} ` +
        `d2h=${s.d2h.toFixed(1)} bytes=${s.bytes.toFixed(1)} trait=${s.trait.toFixed(1)} | ` +
        `clip=${s.clip.toFixed(1)} render=${s.render.toFixed(1)} setState=${s.setState.toFixed(1)} paint=${s.paint.toFixed(1)} | ` +
        `UI=${ui?.toFixed(0) ?? "--"} gap=${gap?.toFixed(0) ?? "--"}`,
      );
    });

    // Skip FFT compute entirely when hidden — saves 30-80 ms/frame.
    if (!showFFTRef.current) {
      fftMagRef.current = null;
      return;
    }

    const applyFFT = (fftResult: { mag: Float32Array; pw: number; ph: number }) => {
      fftMagRef.current = { mag: fftResult.mag, w, h, pw: fftResult.pw, ph: fftResult.ph };
      publishShowPtychoTestFFT(fftResult.mag, w, h, fftResult.pw, fftResult.ph);
      const fr = fftContrastRef.current;
      const fftCmapName = fftCmapRef.current;
      // GPU-first for the FFT panel too (slot 1).
      if (renderPhaseGPU(fftResult.mag, w, h, 1, fftCmapName, fr[0], fr[1])) return;
      const lut = COLORMAPS[fftCmapName as keyof typeof COLORMAPS] || COLORMAPS.inferno;
      const fftRender = renderFFTOffscreen(fftResult.mag, w, h, lut, fr[0], fr[1]);
      setFFTOff(fftRender.canvas);
      setFftDataRange({ min: fftRender.min, max: fftRender.max });
    };

    const gen = ++fftGenRef.current;
    const gpu = gpuFFTRef.current;
    if (gpu) {
      computeFFTMagGPU(gpu, data, w, h).then(fftResult => {
        if (gen !== fftGenRef.current) return;  // stale — newer data already rendered
        applyFFT(fftResult);
      }).catch(() => {
        if (gen !== fftGenRef.current) return;
        applyFFT(computeFFTMag(data, w, h));
      });
    } else {
      applyFFT(computeFFTMag(data, w, h));
    }
  }, [renderPhaseGPU, renderRealDisplay]);

  const runFrontendPreview = React.useCallback((c10Val: number, c12Val: number, phi12Val: number, rotationVal: number): boolean => {
    const engine = webgpuSsbRef.current;
    if (!engine) return false;
    if (webgpuInFlightRef.current) {
      webgpuPendingRef.current = [c10Val, c12Val, phi12Val, rotationVal];
      webgpuPendingFullRef.current = false;
      return true;
    }
    webgpuInFlightRef.current = true;
    setBusy(true);
    setViewPin(null);
    const t0 = performance.now();
    const bfCount = selectedDragBfCount();
    const total = Math.max(1, effectiveTotalBf || bfCount);
    const isFull = bfCount >= total;
    engine.reconstruct(c10Val, c12Val, phi12Val * Math.PI / 180, {
      preview: !isFull,
      bfCount,
      computeLoss: false,
      rotationDeg: rotationVal,
      higherOrder: higherOrderRef.current,
    }).then(result => {
      const phase = result.phase;
      if (flipPhaseRef.current) {
        for (let i = 0; i < phase.length; i++) phase[i] = -phase[i];
      }
      if (result.loss != null) {
        setLoss(result.loss);
      } else if (!isFull) {
        setLoss(null);
      }
      setGpuMs(result.gpuMs);
      lastGpuMsRef.current = result.gpuMs;
      const measuredUi = performance.now() - t0;
      setUiMs(measuredUi);
      lastUiMsRef.current = measuredUi;
      const modeLabel = isFull ? "full BF phase" : "drag";
      const hoLabel = hoActiveRef.current ? `, HO=${hoActiveCountRef.current}` : "";
      setWebgpuLoadProgress(null);
      setWebgpuRuntimeStatus(
        `${result.adapterInfo}${result.softwareAdapter ? " software" : ""} WebGPU folder ${modeLabel} (${result.bfCount}/${total} BF, C10=${c10Val.toFixed(1)} nm, rot=${result.rotationDeg.toFixed(1)}°${hoLabel})`,
      );
      if (isFull || showFFTRef.current) {
        renderAll(phase, result.width, result.height);
      } else {
        renderPreviewPhase(phase, result.width, result.height);
      }
    }).catch(err => {
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
        setWebgpuLoadProgress(null);
        setWebgpuRuntimeStatus(`WebGPU preview failed; using Python path: ${message}`);
        fireRequest(c10Val, c12Val, phi12Val, true);
      }
    }).finally(() => {
      webgpuInFlightRef.current = false;
      setBusy(false);
      const pending = webgpuPendingRef.current;
      if (pending) {
        const pendingFull = webgpuPendingFullRef.current;
        webgpuPendingRef.current = null;
        webgpuPendingFullRef.current = false;
        (pendingFull ? frontendFullRef.current : frontendPreviewRef.current)?.(pending[0], pending[1], pending[2], pending[3]);
      }
    });
    return true;
  }, [effectiveTotalBf, fireRequest, renderAll, renderPreviewPhase, selectedDragBfCount, webgpuStandalone]);

  frontendPreviewRef.current = runFrontendPreview;
  const runFrontendFull = React.useCallback((c10Val: number, c12Val: number, phi12Val: number, rotationVal: number): boolean => {
    const engine = webgpuSsbRef.current;
    if (!webgpuStandalone || !engine) return false;
    if (webgpuInFlightRef.current) {
      webgpuPendingRef.current = [c10Val, c12Val, phi12Val, rotationVal];
      webgpuPendingFullRef.current = true;
      return true;
    }
    webgpuInFlightRef.current = true;
    setBusy(true);
    setViewPin(null);
    const t0 = performance.now();
    const bfCount = selectedDragBfCount();
    const total = Math.max(1, effectiveTotalBf || bfCount);
    const isFull = bfCount >= total;
    engine.reconstruct(c10Val, c12Val, phi12Val * Math.PI / 180, {
      preview: !isFull,
      bfCount,
      computeLoss: isFull,
      rotationDeg: rotationVal,
      higherOrder: higherOrderRef.current,
    }).then(result => {
      const phase = result.phase;
      if (flipPhaseRef.current) {
        for (let i = 0; i < phase.length; i++) phase[i] = -phase[i];
      }
      setLoss(isFull ? result.loss : null);
      setGpuMs(result.gpuMs);
      lastGpuMsRef.current = result.gpuMs;
      const measuredUi = performance.now() - t0;
      setUiMs(measuredUi);
      lastUiMsRef.current = measuredUi;
      const modeLabel = isFull ? "full BF + loss" : "selected BF";
      const hoLabel = hoActiveRef.current ? `, HO=${hoActiveCountRef.current}` : "";
      setWebgpuLoadProgress(null);
      setWebgpuRuntimeStatus(
        `${result.adapterInfo}${result.softwareAdapter ? " software" : ""} WebGPU folder ${modeLabel} (${result.bfCount}/${total} BF, C10=${c10Val.toFixed(1)} nm, rot=${result.rotationDeg.toFixed(1)}°${hoLabel})`,
      );
      renderAll(phase, result.width, result.height);
    }).catch(err => {
      const message = err instanceof Error ? err.message : String(err);
      console.error("[showptycho] WebGPU folder full failed", err);
      setWebgpuLoadProgress({
        stage: "error",
        message: "Full-BF WebGPU ptychography failed",
        detail: message,
        percent: 0,
      });
      setWebgpuRuntimeStatus(`WebGPU folder full failed: ${message}`);
    }).finally(() => {
      webgpuInFlightRef.current = false;
      setBusy(false);
      const pending = webgpuPendingRef.current;
      if (pending) {
        const pendingFull = webgpuPendingFullRef.current;
        webgpuPendingRef.current = null;
        webgpuPendingFullRef.current = false;
        (pendingFull ? frontendFullRef.current : frontendPreviewRef.current)?.(pending[0], pending[1], pending[2], pending[3]);
      }
    });
    return true;
  }, [effectiveTotalBf, renderAll, selectedDragBfCount, webgpuStandalone]);

  frontendFullRef.current = runFrontendFull;
  shouldCommitOnReleaseRef.current = dragBfRef.current > 0 || !!webgpuSsbRef.current;
  const standaloneInitialRenderRef = React.useRef(false);
  React.useEffect(() => {
    if (!webgpuStandalone || standaloneInitialRenderRef.current) return;
    if (!webgpuSsbRef.current || !initRef.current) return;
    if (rawPhaseRef.current) return;
    standaloneInitialRenderRef.current = true;
    const total = Math.max(1, effectiveTotalBf || 1);
    const count = selectedDragBfCount();
    setWebgpuLoadProgress({
      stage: "device",
      message: "Starting browser-side ptychography",
      detail: `Loading compressed HDF5 and preparing ${count}/${total} BF pixels`,
      current: 0,
      total: count,
      percent: 0,
      activeBf: count,
      totalBf: total,
    });
    setWebgpuRuntimeStatus("Preparing compressed HDF5 source on WebGPU");
    frontendPreviewRef.current?.(autoC10, autoC12, autoPhi12, autoRotation ?? rotationDegRef.current);
  }, [autoC10, autoC12, autoPhi12, autoRotation, effectiveTotalBf, selectedDragBfCount, webgpuRuntimeStatus, webgpuStandalone]);

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
    const p = rawPhaseRef.current;
    if (!p) return;
    // Re-render for the same data, now with FFT path enabled (showFFTRef is already true).
    renderAll(p.data, p.w, p.h);
  }, [showFFT, fftPlacement, renderAll]);

  /* --- Flip is a display-sign convention, not a new reconstruction.
         The live notebook path also receives a Python trait update, but the
         exported WebGPU HTML has no kernel observer. Negate the currently
         displayed phase immediately so both paths feel identical. --- */
  const flipInitializedRef = React.useRef(false);
  const renderAllRef = React.useRef(renderAll);
  renderAllRef.current = renderAll;
  React.useEffect(() => {
    if (!flipInitializedRef.current) {
      flipInitializedRef.current = true;
      return;
    }
    const p = rawPhaseRef.current;
    if (!p) return;
    const flipped = new Float32Array(p.data.length);
    for (let i = 0; i < p.data.length; i++) flipped[i] = -p.data[i];
    renderAllRef.current(flipped, p.w, p.h);
  }, [flipPhase]);

  /* --- Render when new data arrives --- */
  React.useEffect(() => {
    if (!phaseBytes || phaseWidth === 0 || phaseHeight === 0) return;
    const data = extractFloat32(phaseBytes);
    if (!data || data.length === 0) return;
    renderAll(data, phaseWidth, phaseHeight);
  }, [phaseBytes, phaseWidth, phaseHeight, renderAll]);

  /* --- Buttons --- */
  const doReset = () => {
    // auto values arrive asynchronously from Python; skip until init landed
    if (!initRef.current) return;
    setC10(autoC10); setC12(autoC12); setPhi12(autoPhi12);
    if (autoRotation != null) setRotationDeg(autoRotation);
    sendCommit(autoC10, autoC12, autoPhi12, autoRotation ?? rotationDegRef.current);
  };
  // RAND scope: which parameters to randomize.  Default = all on.  Users can
  // uncheck any (e.g. randomize only phi12 to sweep angular space at a fixed
  // C10/C12 they've dialed in).
  const [randScope, setRandScope] = React.useState({
    c10: true, c12: true, phi12: true, rotation: true,
  });
  const doRandom = () => {
    const c10Rand   = randScope.c10   ? Math.round(c10UiMin   + Math.random() * (c10UiMax   - c10UiMin))   : sliderVals.current.c10;
    const c12Rand   = randScope.c12   ? Math.round(c12UiMin   + Math.random() * (c12UiMax   - c12UiMin))   : sliderVals.current.c12;
    const phi12Rand = randScope.phi12 ? Math.round(phi12UiMin + Math.random() * (phi12UiMax - phi12UiMin)) : sliderVals.current.phi12;
    setC10(c10Rand); setC12(c12Rand); setPhi12(phi12Rand);
    let rotationRand = rotationDegRef.current;
    if (randScope.rotation) {
      rotationRand = rotationUiMin + Math.random() * (rotationUiMax - rotationUiMin);
      setRotationDeg(rotationRand);
    }
    sendCommit(c10Rand, c12Rand, phi12Rand, rotationRand);
  };
  const resetZoom = () => {
    setPhaseZoom(ZOOM_RESET);
    setAmpZoom(ZOOM_RESET);
    setComplexZoom(ZOOM_RESET);
    setFFTZoom(ZOOM_RESET);
  };

  const capturePinLayout = React.useCallback(() => {
    const rects = new Map<number, DOMRect>();
    document.querySelectorAll<HTMLElement>("[data-showptycho-pin-id]").forEach(el => {
      const id = Number(el.getAttribute("data-showptycho-pin-id"));
      if (Number.isFinite(id)) rects.set(id, el.getBoundingClientRect());
    });
    pinLayoutBeforeRef.current = rects;
  }, []);

  React.useLayoutEffect(() => {
    const before = pinLayoutBeforeRef.current;
    if (!before) return;
    pinLayoutBeforeRef.current = null;
    window.requestAnimationFrame(() => {
      document.querySelectorAll<HTMLElement>("[data-showptycho-pin-id]").forEach(el => {
        const id = Number(el.getAttribute("data-showptycho-pin-id"));
        const previous = before.get(id);
        if (!previous && id === newPinId) {
          el.animate(
            [
              { opacity: 0, transform: "scale(0.82)" },
              { opacity: 1, transform: "scale(1.06)" },
              { opacity: 1, transform: "scale(1)" },
            ],
            { duration: 320, easing: "cubic-bezier(0.2, 0, 0, 1)" },
          );
          document.documentElement.setAttribute("data-showptycho-last-pin-animation", `add:${id}`);
          return;
        }
        if (!previous) return;
        const next = el.getBoundingClientRect();
        const dx = previous.left - next.left;
        const dy = previous.top - next.top;
        if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) return;
        el.animate(
          [
            { transform: `translate(${dx}px, ${dy}px) scale(1)`, zIndex: 2 },
            { transform: "translate(0, 0) scale(1)", zIndex: 2 },
          ],
          { duration: 180, easing: "cubic-bezier(0.2, 0, 0, 1)" },
        );
        document.documentElement.setAttribute("data-showptycho-last-pin-animation", `move:${id}`);
      });
    });
  }, [newPinId, pinned]);

  /* --- Pin --- */
  const doPin = React.useCallback(() => {
    const phase = rawPhaseRef.current;
    if (!phase || busy) return;
    capturePinLayout();
    const entry: PinnedEntry = {
      id: Date.now(),
      C10: sliderVals.current.c10,
      C12: sliderVals.current.c12,
      phi12_deg: sliderVals.current.phi12,
      rotation_deg: rotationDeg ?? 0,
      flip_phase: !!flipPhase,
      loss: loss ?? 0,
      starred: false,
      timestamp: new Date().toISOString(),
      phaseData: new Float32Array(phase.data),
      displayMode: extraRealViewsRef.current.complex ? "complex" : extraRealViewsRef.current.amp ? "amp" : "phase",
      w: phase.w, h: phase.h,
      thumb: makeThumbnail(
        phase.data,
        phase.w,
        phase.h,
        cmapRef.current,
        extraRealViewsRef.current.complex ? "complex" : extraRealViewsRef.current.amp ? "amp" : "phase",
      ),
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
      C10: entry.C10, C12: entry.C12, phi12_deg: entry.phi12_deg,
      rotation_deg: entry.rotation_deg, flip_phase: entry.flip_phase,
      display_mode: entry.displayMode,
      loss: entry.loss, timestamp: entry.timestamp,
    }));
  }, [busy, capturePinLayout, loss, rotationDeg, flipPhase, setPinJson]);

  React.useEffect(() => {
    setPinned(prev => {
      if (prev.length === 0) return prev;
      return prev.map(p => ({
        ...p,
        thumb: makeThumbnail(p.phaseData, p.w, p.h, cmap, p.displayMode),
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
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("JPEG encode failed"))), "image/jpeg", 0.92);
      });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const record: FolderSaveRecord = {
        id: stamp,
        timestamp: new Date().toISOString(),
        C10: sliderVals.current.c10,
        C12: sliderVals.current.c12,
        phi12_deg: sliderVals.current.phi12,
        rotation_deg: rotationDeg ?? 0,
        flip_phase: !!flipPhase,
        loss: loss ?? null,
        bf: Math.round(dragBfRef.current || 0),
        notes: String(notes ?? ""),
        image: `snapshots/snapshot_${stamp}_C10_${slugPart(sliderVals.current.c10.toFixed(0))}.jpg`,
      };
      const next = [...folderSaves, record];
      if (ssbFolderWritable()) {
        await writeSSBFolderFile(record.image, jpeg);
        await writeSSBFolderFile("snapshots/snapshots.json", JSON.stringify(next, null, 2));
        setFolderSaves(next);
        setFolderSaveStatus(`Snapshot saved (${next.length})`);
      } else {
        setFolderSaveStatus("Folder is read-only here - use Open data folder for persistent snapshots");
      }
    } catch (err) {
      setFolderSaveStatus(`Save failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [busy, folderSaves, loss, notes, rotationDeg, flipPhase]);
  const loadFolderSave = React.useCallback((record: FolderSaveRecord) => {
    setC10(record.C10);
    setC12(record.C12);
    setPhi12(record.phi12_deg);
    setRotationDeg(record.rotation_deg);
    setFlipPhase(record.flip_phase);
    sliderVals.current = { c10: record.C10, c12: record.C12, phi12: record.phi12_deg };
    frontendFullRef.current?.(record.C10, record.C12, record.phi12_deg, record.rotation_deg);
  }, [setRotationDeg, setFlipPhase]);
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
      const next = folderSaves.filter((r) => r.id !== record.id);
      await writeSSBFolderFile("snapshots/snapshots.json", JSON.stringify(next, null, 2));
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
        pins: pinned.map((p, i) => ({
          index: i + 1,
          filename: `showptycho_pin_${String(i + 1).padStart(2, "0")}_C10_${slugPart(p.C10.toFixed(0))}_C12_${slugPart(p.C12.toFixed(0))}_phi_${slugPart(p.phi12_deg.toFixed(0))}.png`,
          C10: p.C10,
          C12: p.C12,
          phi12_deg: p.phi12_deg,
          rotation_deg: p.rotation_deg,
          flip_phase: p.flip_phase,
          display_mode: p.displayMode,
          loss: p.loss,
          starred: p.starred,
          timestamp: p.timestamp,
          shape: [p.h, p.w],
        })),
      };

      let exported = 0;
      for (const [i, p] of pinned.entries()) {
        const { canvas } = p.displayMode === "complex"
          ? renderComplexPhaseOffscreen(p.phaseData, p.w, p.h, 1, 99)
          : renderPhaseOffscreen(p.displayMode === "amp" ? phaseAbsData(p.phaseData) : p.phaseData, p.w, p.h, lut, 1, 99);
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
      const message = `Exported ${exported} PNG + JSON`;
      setExportStatus(message);
      document.documentElement.setAttribute("data-showptycho-last-export", JSON.stringify({
        count: exported,
        manifest: manifestName,
        colormap: cmapRef.current,
      }));
    } catch (err) {
      const message = `Export failed ${err instanceof Error ? err.message : String(err)}`;
      setExportStatus(message);
      document.documentElement.setAttribute("data-showptycho-last-export", JSON.stringify({ error: message }));
    }
  }, [pinned]);

  const animationFrameValues = React.useCallback((frameCount = Math.min(24, sweepSteps)) => {
    const count = Math.max(2, Math.min(120, Math.round(frameCount || sweepSteps)));
    const frames: Array<{
      c10: number; c12: number; phi12: number; rotation: number; higherOrder: Record<string, number>; label: string;
    }> = [];
    const mainKeys = sweepMainKeys(sweepParam);
    const hoKey = sweepParam.startsWith("ho:") ? sweepParam.slice(3) : null;
    const hoRange = hoKey ? getFullSweepRange(sweepParam) : null;
    for (let i = 0; i < count; i++) {
      const frac = count <= 1 ? 0 : i / (count - 1);
      const nextHo = { ...higherOrderRef.current };
      let nextC10 = sliderVals.current.c10;
      let nextC12 = sliderVals.current.c12;
      let nextPhi12 = sliderVals.current.phi12;
      let nextRot = rotationDegRef.current;
      if (mainKeys.includes("c10")) nextC10 = sweepRangesRef.current.c10[0] + frac * (sweepRangesRef.current.c10[1] - sweepRangesRef.current.c10[0]);
      if (mainKeys.includes("c12")) nextC12 = sweepRangesRef.current.c12[0] + frac * (sweepRangesRef.current.c12[1] - sweepRangesRef.current.c12[0]);
      if (mainKeys.includes("phi12")) nextPhi12 = sweepRangesRef.current.phi12[0] + frac * (sweepRangesRef.current.phi12[1] - sweepRangesRef.current.phi12[0]);
      if (mainKeys.includes("rot")) nextRot = sweepRangesRef.current.rot[0] + frac * (sweepRangesRef.current.rot[1] - sweepRangesRef.current.rot[0]);
      if (hoKey && hoRange) nextHo[hoKey] = hoRange[0] + frac * (hoRange[1] - hoRange[0]);
      frames.push({
        c10: nextC10,
        c12: nextC12,
        phi12: nextPhi12,
        rotation: nextRot,
        higherOrder: nextHo,
        label: `C10 ${nextC10.toFixed(0)} nm · C12 ${nextC12.toFixed(0)} nm · φ12 ${nextPhi12.toFixed(0)}° · rot ${nextRot.toFixed(1)}°`,
      });
    }
    return frames;
  }, [getFullSweepRange, sweepMainKeys, sweepParam, sweepSteps]);

  const renderAnimationFrameCanvas = React.useCallback(async (
    frame: { c10: number; c12: number; phi12: number; rotation: number; higherOrder: Record<string, number>; label: string },
    maxPanelEdge = 512,
  ): Promise<HTMLCanvasElement> => {
    const engine = webgpuSsbRef.current;
    if (!engine) throw new Error("GIF/MP4 export needs the WebGPU ShowPtycho engine.");
    const bfCount = selectedDragBfCount();
    const total = Math.max(1, effectiveTotalBf || bfCount);
    const result = await engine.reconstruct(
      frame.c10,
      frame.c12,
      frame.phi12 * Math.PI / 180,
      {
      preview: bfCount < total,
      bfCount,
      computeLoss: false,
      rotationDeg: frame.rotation,
      higherOrder: frame.higherOrder,
      },
    );
    const phase = result.phase;
    if (flipPhaseRef.current) {
      for (let i = 0; i < phase.length; i++) phase[i] = -phase[i];
    }
    const phaseLut = COLORMAPS[cmapRef.current as keyof typeof COLORMAPS] || COLORMAPS.viridis;
    const cr = contrastRef.current;
    const phaseRender = renderPhaseOffscreen(phase, result.width, result.height, phaseLut, cr[0], cr[1]);
    if (!phaseRender.canvas) throw new Error("Could not render phase export frame.");
    let fftCanvas: HTMLCanvasElement | null = null;
    if (showFFTRef.current) {
      const fft = computeFFTMag(phase, result.width, result.height);
      const fftLut = COLORMAPS[fftCmapRef.current as keyof typeof COLORMAPS] || COLORMAPS.inferno;
      const fr = fftContrastRef.current;
      fftCanvas = renderFFTOffscreen(fft.mag, result.width, result.height, fftLut, fr[0], fr[1]).canvas;
    }
    const panelEdge = Math.max(64, Math.min(maxPanelEdge, result.width, result.height));
    const labelH = 22;
    const out = document.createElement("canvas");
    out.width = panelEdge * (fftCanvas ? 2 : 1);
    out.height = panelEdge + labelH;
    const ctx = out.getContext("2d");
    if (!ctx) throw new Error("Could not create export canvas.");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(phaseRender.canvas, 0, 0, panelEdge, panelEdge);
    if (fftCanvas) ctx.drawImage(fftCanvas, panelEdge, 0, panelEdge, panelEdge);
    ctx.fillStyle = "rgba(0,0,0,0.84)";
    ctx.fillRect(0, panelEdge, out.width, labelH);
    ctx.fillStyle = "#e8f0ff";
    ctx.font = "11px monospace";
    ctx.textBaseline = "middle";
    ctx.fillText(frame.label, 6, panelEdge + labelH / 2);
    return out;
  }, [effectiveTotalBf, selectedDragBfCount]);

  const exportAnimationGif = React.useCallback(async () => {
    const frames = animationFrameValues();
    const filename = makeShowPtychoExportFilename("gif", sweepParam, frames.length);
    setAnimationExportBusy(true);
    setAnimationExportStatus(`Preparing ${filename}`);
    try {
      const indexed: Uint8Array[] = [];
      let outW = 0;
      let outH = 0;
      for (let i = 0; i < frames.length; i++) {
        const canvas = await renderAnimationFrameCanvas(frames[i], 512);
        outW = canvas.width;
        outH = canvas.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Could not read export frame.");
        indexed.push(quantizeRgbaForBrowserGif(ctx.getImageData(0, 0, outW, outH).data));
        if (i === 0 || i === frames.length - 1 || (i + 1) % 4 === 0) {
          setAnimationExportStatus(`Encoding ${filename} ${i + 1}/${frames.length}`);
          await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
        }
      }
      const gif = encodeIndexedGif(outW, outH, indexed, 100 / Math.max(1, Math.min(30, playFps)));
      const blob = new Blob([gif as BlobPart], { type: "image/gif" });
      downloadBlob(blob, filename);
      setAnimationExportStatus(`Downloaded ${filename} (${formatSavedBytes(blob.size)})`);
      document.documentElement.setAttribute("data-showptycho-last-animation-export", JSON.stringify({ mode: "gif", frames: frames.length, bytes: blob.size }));
    } catch (err) {
      setAnimationExportStatus(`Export failed ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setAnimationExportBusy(false);
    }
  }, [animationFrameValues, playFps, renderAnimationFrameCanvas, sweepParam]);

  const exportAnimationMp4 = React.useCallback(async () => {
    const mime = "video/mp4";
    if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported(mime)) {
      setAnimationExportStatus("MP4 export unavailable in this browser; use GIF here.");
      return;
    }
    const frames = animationFrameValues();
    const filename = makeShowPtychoExportFilename("mp4", sweepParam, frames.length);
    setAnimationExportBusy(true);
    setAnimationExportStatus(`Preparing ${filename}`);
    try {
      const first = await renderAnimationFrameCanvas(frames[0], 768);
      const even = document.createElement("canvas");
      even.width = first.width + (first.width % 2);
      even.height = first.height + (first.height % 2);
      const ctx = even.getContext("2d");
      if (!ctx) throw new Error("Could not create MP4 canvas.");
      const stream = even.captureStream(Math.max(1, Math.min(30, playFps)));
      const recorder = new MediaRecorder(stream, { mimeType: mime });
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (event) => { if (event.data.size > 0) chunks.push(event.data); };
      const stopped = new Promise<void>((resolve, reject) => {
        recorder.onstop = () => resolve();
        recorder.onerror = () => reject(new Error("MP4 recorder failed."));
      });
      recorder.start();
      for (let i = 0; i < frames.length; i++) {
        const frameCanvas = i === 0 ? first : await renderAnimationFrameCanvas(frames[i], 768);
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, even.width, even.height);
        ctx.drawImage(frameCanvas, 0, 0);
        setAnimationExportStatus(`Recording ${filename} ${i + 1}/${frames.length}`);
        await new Promise<void>((resolve) => window.setTimeout(resolve, Math.max(34, 1000 / Math.max(1, Math.min(30, playFps)))));
      }
      recorder.stop();
      await stopped;
      const blob = new Blob(chunks, { type: mime });
      downloadBlob(blob, filename);
      setAnimationExportStatus(`Downloaded ${filename} (${formatSavedBytes(blob.size)})`);
      document.documentElement.setAttribute("data-showptycho-last-animation-export", JSON.stringify({ mode: "mp4", frames: frames.length, bytes: blob.size }));
    } catch (err) {
      setAnimationExportStatus(`Export failed ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setAnimationExportBusy(false);
    }
  }, [animationFrameValues, playFps, renderAnimationFrameCanvas, sweepParam]);

  const reorderPinned = React.useCallback((fromId: number, toId: number) => {
    if (fromId === toId) return;
    capturePinLayout();
    setPinned(prev => {
      const from = prev.findIndex(p => p.id === fromId);
      const to = prev.findIndex(p => p.id === toId);
      if (from < 0 || to < 0 || from === to) return prev;
      const next = [...prev];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });
  }, [capturePinLayout]);

  const pinIdFromPoint = React.useCallback((clientX: number, clientY: number) => {
    const el = document.elementFromPoint(clientX, clientY)?.closest("[data-showptycho-pin-id]");
    const raw = el?.getAttribute("data-showptycho-pin-id");
    const id = raw ? Number(raw) : NaN;
    return Number.isFinite(id) ? id : null;
  }, []);

  const beginPinDrag = React.useCallback((id: number, clientX: number, clientY: number) => {
    pinPointerDragRef.current = { id, x: clientX, y: clientY, active: false };
    draggedPinRef.current = id;
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
    draggedPinRef.current = null;
    setDraggingPin(null);
    setDragOverPin(null);
    if (drag.active) window.setTimeout(() => { didDragPinRef.current = false; }, 0);
  }, [pinIdFromPoint, reorderPinned]);

  const cancelPinDrag = React.useCallback(() => {
    pinPointerDragRef.current = null;
    draggedPinRef.current = null;
    setDraggingPin(null);
    setDragOverPin(null);
    window.setTimeout(() => { didDragPinRef.current = false; }, 0);
  }, []);

  React.useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      const drag = pinPointerDragRef.current;
      if (drag) updatePinDrag(drag.id, event.clientX, event.clientY);
    };
    const handleMouseUp = (event: MouseEvent) => {
      const drag = pinPointerDragRef.current;
      if (drag) finishPinDrag(drag.id, event.clientX, event.clientY);
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [finishPinDrag, updatePinDrag]);

  const doUnpin = React.useCallback((id: number) => {
    capturePinLayout();
    setPinned(prev => prev.filter(p => p.id !== id));
    setViewPin(current => current === id ? null : current);
    setPinJson(JSON.stringify({ action: "unpin", id }));
  }, [capturePinLayout, setPinJson]);

  // Toggle star on a pin; Python persists the new state to disk.
  const doToggleStar = React.useCallback((id: number) => {
    let nextStarred = false;
    setPinned(prev => prev.map(p => {
      if (p.id !== id) return p;
      nextStarred = !p.starred;
      return { ...p, starred: nextStarred };
    }));
    // The map() runs synchronously in the setState callback, so the updated
    // nextStarred flag is correct when we fire the trait below.
    setPinJson(JSON.stringify({ action: nextStarred ? "star" : "unstar", id }));
  }, [setPinJson]);

  const doViewPin = React.useCallback((entry: PinnedEntry) => {
    setViewPin(entry.id);
    setActiveTrialRank(null);  // pins + trials are mutually exclusive selections
    const mode = entry.displayMode ?? "phase";
    const extras = { amp: mode === "amp", complex: mode === "complex" };
    setExtraRealViews(extras);
    extraRealViewsRef.current = extras;
    setC10(entry.C10); setC12(entry.C12); setPhi12(entry.phi12_deg);
    setLoss(entry.loss);
    setGpuMs(null); setUiMs(null); setJsMs(null);
    renderAll(entry.phaseData, entry.w, entry.h);
  }, [renderAll]);

  /* --- Keyboard: window-level so shortcuts work regardless of focus --- */
  React.useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if ((e.target as HTMLElement).closest?.(".MuiSlider-root")) return;

      // R → reset zoom (works without pins)
      if (e.key === "r" || e.key === "R") { resetZoom(); return; }
      // P → pin current snapshot (oldest is evicted once MAX_PINS hit)
      if (e.key === "p" || e.key === "P") { e.preventDefault(); doPin(); return; }
      // Space → toggle play/pause of the aberration sweep
      if (e.key === " " || e.code === "Space") { e.preventDefault(); setPlaying(p => !p); return; }

      if (e.key === "Escape") {
        setViewPin(null); setActiveTrialRank(null); return;
      }
      // S → star/unstar the currently-viewed pin (Python persists to disk)
      if ((e.key === "s" || e.key === "S") && viewPin != null) {
        e.preventDefault(); doToggleStar(viewPin); return;
      }
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        // Dispatch based on which browser was most recently touched.  Clicking
        // a trial tile sets activeTrialRank; clicking a pin sets viewPin; they
        // clear each other so only one is "active" at a time.
        const hasActiveTrial = activeTrialRank != null && trials.length > 0;
        const hasActivePin = pinned.length > 0;
        if (!hasActiveTrial && !hasActivePin && trials.length === 0) return;
        // Capture-phase + stopPropagation so JupyterLab's notebook cell nav
        // can't steal our arrow keys.  Only do this when we're actually going
        // to act on the arrow, otherwise leave arrows free for normal use.
        e.preventDefault();
        e.stopPropagation();
        if (hasActiveTrial) {
          const cur = trials.findIndex(t => t.rank === activeTrialRank);
          const next = e.key === "ArrowRight"
            ? (cur < 0 ? 0 : (cur >= trials.length - 1 ? 0 : cur + 1))
            : (cur < 0 ? trials.length - 1 : (cur <= 0 ? trials.length - 1 : cur - 1));
          doViewTrial(trials[next]);
          setTrialsExpanded(true);  // so the selected tile is visible
          return;
        }
        if (hasActivePin) {
          const idx = viewPin ? pinned.findIndex(p => p.id === viewPin) : -1;
          const next = e.key === "ArrowRight"
            ? (idx < 0 ? 0 : (idx >= pinned.length - 1 ? 0 : idx + 1))
            : (idx < 0 ? pinned.length - 1 : (idx <= 0 ? pinned.length - 1 : idx - 1));
          doViewPin(pinned[next]);
          return;
        }
        // Neither collection touched but trials exist → enter trials list at #1.
        if (trials.length > 0) {
          doViewTrial(trials[0]);
          setTrialsExpanded(true);
        }
      }
    };
    // Capture phase so JupyterLab's notebook-level cell-navigation handler
    // (also bound to keydown) doesn't consume the event before we see it.
    window.addEventListener("keydown", handleKey, { capture: true });
    return () => window.removeEventListener("keydown", handleKey, { capture: true });
  }, [pinned, viewPin, doViewPin, doPin, doToggleStar, trials, activeTrialRank, doViewTrial]);

  /* --- Resize handle --- */
  React.useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (resizeDrag.current.on) {
        setPanel(Math.max(MIN_PANEL, resizeDrag.current.s0 + e.clientY - resizeDrag.current.y0));
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
    resizeDrag.current = { on: true, y0: e.clientY, s0: panel };
  }, [panel]);

  /* --- Scale bar: reciprocal pixel size for FFT --- */
  const fftPixelSize = React.useMemo(() => {
    const f = fftMagRef.current;
    if (!f || !pixelSize || pixelSize <= 0) return 0;
    // Reciprocal pixel size: 1 / (padded_N * real_space_pixel_size)
    // We use the padded width since FFT was computed on padded grid, but
    // we crop back to original size, so use original width for display.
    return 1.0 / (f.w * pixelSize);
  }, [pixelSize, fftOff]);

  /* --- Derived --- */
  const delta = loss != null && autoLoss > 0 ? loss - autoLoss : null;
  const fftAsPanel = showFFT && fftPlacement === "panel";
  const fftAsInset = showFFT && fftPlacement === "inset";
  const realPanelCount = 1 + (extraRealViews.amp ? 1 : 0) + (extraRealViews.complex ? 1 : 0);
  const visiblePanelCount = realPanelCount + (fftAsPanel ? 1 : 0);
  const totalW = panel * visiblePanelCount + PANEL_GAP * Math.max(0, visiblePanelCount - 1);
  const fftInsetSize = Math.max(96, Math.min(180, Math.round(panel * 0.28)));
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
    height: ACTION_CONTROL_HEIGHT,
    fontSize: 10,
    "& .MuiSelect-select": {
      ...themedSelect["& .MuiSelect-select, & .MuiSelect-select.MuiSelect-outlined"],
      minHeight: 0,
      py: 0,
      display: "flex",
      alignItems: "center",
    },
  };
  const themedMenuProps = {
    PaperProps: {
      sx: {
        bgcolor: tc.bgAlt,
        color: tc.text,
        border: `1px solid ${tc.border}`,
        boxShadow: `0 8px 24px ${tc.shadow}`,
        "& .MuiMenuItem-root": {
          fontFamily: "monospace",
          fontSize: 11,
          minHeight: 28,
        },
      },
    },
  };
  const canRecordMp4 = typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported("video/mp4");
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
    c12: [c12Min, c12Max],
    phi12: [phi12Min, phi12Max],
    rot: [rotationMin, rotationMax],
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

  return (
    <Box sx={{ ...container.root, bgcolor: tc.bg, color: tc.text }}>
      {/* ---- Top toolbar: FFT toggle + image dims + Reset view (matching Show2D) ---- */}
      <Stack
        direction="row"
        alignItems="center"
        spacing={`${SPACING.SM}px`}
        sx={{
          mb: `${SPACING.XS}px`, minHeight: 28, height: "auto", width: "100%",
          flexWrap: "wrap", rowGap: `${SPACING.XS}px`,
        }}
      >
        <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted }}>FFT</Typography>
        <Switch
          checked={showFFT}
          onChange={(e) => setShowFFT(e.target.checked)}
          size="small"
          sx={switchStyles.small}
        />
        {showFFT && (
          <Select
            value={fftPlacement}
            onChange={(e) => setFFTPlacement(e.target.value as FFTPlacement)}
            size="small"
            sx={{ ...actionSelect, width: 72, flex: "0 0 72px" }}
            MenuProps={themedMenuProps}
            aria-label="FFT placement"
          >
            <MenuItem value="inset">Inset</MenuItem>
            <MenuItem value="panel">Panel</MenuItem>
          </Select>
        )}
        <Typography sx={{ ...typography.label, fontSize: 10, color: tc.textMuted, ml: 1 }}>
          BF
        </Typography>
        <Slider
          value={localDragBf}
          min={effectiveTotalBf > 0 ? 1 : 0}
          max={effectiveTotalBf > 0 ? effectiveTotalBf : 0}
          step={1}
          disabled={effectiveTotalBf <= 0}
          onChange={(_, val) => setLocalDragBf(val as number)}
          onChangeCommitted={(_, val) => commitDragBf(val as number)}
          size="small"
          valueLabelDisplay="auto"
          valueLabelFormat={(v) => effectiveTotalBf > 0 ? `${(v / effectiveTotalBf).toFixed(2)} (${v}/${effectiveTotalBf})` : "--"}
          aria-label={effectiveTotalBf > 0 ? `BF pixels ${localDragBf} of ${effectiveTotalBf}` : "BF pixels"}
          sx={{ width: 120, flex: "0 0 120px", mx: 0.5 }}
        />
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
        <Badge
          badgeContent={toolbarMoreActiveCount}
          invisible={toolbarMoreActiveCount === 0}
          sx={{ "& .MuiBadge-badge": { bgcolor: tc.accent, color: "#fff", fontSize: 9, fontWeight: 600, minWidth: 14, height: 14, px: 0.25 } }}
        >
          <Button
            size="small"
            onClick={(e) => setToolbarMoreAnchor(e.currentTarget)}
            aria-label="More tools"
            aria-controls={toolbarMoreAnchor ? "showptycho-toolbar-more-menu" : undefined}
            aria-expanded={toolbarMoreAnchor ? "true" : undefined}
            aria-haspopup="menu"
            title="More tools: views, crop refit, export"
            sx={{ ...compactButton(tc), color: toolbarMoreActiveCount > 0 ? tc.accent : tc.text }}
          >
            More
          </Button>
        </Badge>
        <Menu
          id="showptycho-toolbar-more-menu"
          anchorEl={toolbarMoreAnchor}
          open={Boolean(toolbarMoreAnchor)}
          onClose={() => setToolbarMoreAnchor(null)}
          MenuListProps={{ "aria-label": "More tools" }}
          {...themedMenuProps}
        >
          <Box sx={{ px: 1.5, pt: 0.75, pb: 0.35, minWidth: 260 }}>
            <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: tc.textMuted, textTransform: "uppercase" }}>Views</Typography>
          </Box>
          {(["amp", "complex"] as ExtraRealViewMode[]).map((mode) => {
            const label = mode === "amp" ? "Amplitude" : "Complex";
            const active = extraRealViews[mode];
            return (
              <MenuItem key={mode} dense onClick={() => setExtraRealViews(prev => ({ ...prev, [mode]: !prev[mode] }))} sx={{ fontSize: 12, gap: 1, color: active ? tc.accent : tc.text }}>
                <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }}>{label}</Typography>
                <Switch checked={active} onClick={(e) => e.stopPropagation()} onChange={() => setExtraRealViews(prev => ({ ...prev, [mode]: !prev[mode] }))} size="small" sx={switchStyles.small} />
              </MenuItem>
            );
          })}
          {cropRefitAvailable && <>
            <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${tc.border}` }} />
            <Box sx={{ px: 1.5, pt: 0.35, pb: 0.35 }}>
              <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: tc.textMuted, textTransform: "uppercase" }}>SSB Crop</Typography>
            </Box>
            <MenuItem dense onClick={() => { setCropSelecting(value => !value); setToolbarMoreAnchor(null); }} sx={{ fontSize: 12, gap: 1, color: cropSelecting ? STATUS_GOOD : tc.text }}>
              <Typography sx={{ flex: 1, fontSize: 12, color: "inherit" }}>Draw crop region</Typography>
              <Switch checked={cropSelecting} onClick={(e) => e.stopPropagation()} onChange={() => { setCropSelecting(value => !value); setToolbarMoreAnchor(null); }} size="small" sx={switchStyles.small} />
            </MenuItem>
            {(cropSelecting || scanCrop) && (
              <Box sx={{ px: 1.5, py: 0.6, minWidth: 260 }}>
                <Typography sx={{ ...typography.value, fontVariantNumeric: "tabular-nums", color: scanCrop ? tc.text : tc.textMuted }}>{cropSummary}</Typography>
              </Box>
            )}
            <MenuItem dense onClick={() => { resetCrop(); setToolbarMoreAnchor(null); }} disabled={!scanCrop && !cropSelecting} sx={{ fontSize: 12 }}>
              Crop Reset
            </MenuItem>
            <MenuItem dense onClick={() => { requestCropRefit(); setToolbarMoreAnchor(null); }} disabled={busy || cropRefitPending || !scanCrop} sx={{ fontSize: 12, color: STATUS_GOOD }}>
              Refit SSB (200 trials)
            </MenuItem>
          </>}
          <Box sx={{ mx: 1.5, my: 0.5, borderTop: `1px solid ${tc.border}` }} />
          <Box sx={{ px: 1.5, pt: 0.35, pb: 0.35 }}>
            <Typography sx={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.04em", color: tc.textMuted, textTransform: "uppercase" }}>Export</Typography>
          </Box>
          <MenuItem dense onClick={() => { void exportAnimationGif(); setToolbarMoreAnchor(null); }} disabled={animationExportBusy || !rawPhaseRef.current} sx={{ fontSize: 12 }}>
            Export GIF
          </MenuItem>
          <MenuItem dense onClick={() => { void exportAnimationMp4(); setToolbarMoreAnchor(null); }} disabled={animationExportBusy || !rawPhaseRef.current || !canRecordMp4} sx={{ fontSize: 12 }}>
            Export MP4
          </MenuItem>
          {animationExportStatus && (
            <Box sx={{ px: 1.5, py: 0.6, minWidth: 260 }}>
              <Typography sx={{ ...typography.value, color: animationExportStatus.startsWith("Export failed") ? STATUS_BAD : tc.textMuted }}>{animationExportStatus}</Typography>
            </Box>
          )}
        </Menu>
        <Box sx={{ flex: 1 }} />
        {phaseWidth > 0 && phaseHeight > 0 && (
          <Typography sx={{ ...typography.value, color: tc.textMuted, opacity: 0.7 }}>
            {phaseWidth}×{phaseHeight}
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
        <Button
          size="small"
          variant="outlined"
          onClick={resetZoom}
          disabled={phaseZoom.zoom === 1 && ampZoom.zoom === 1 && complexZoom.zoom === 1 && fftZoom.zoom === 1}
          sx={{ ...compactButton(tc), width: 52, minWidth: 52, color: tc.textMuted, borderColor: tc.border }}
        >
          Reset
        </Button>
      </Stack>

      {/* ---- Two panels ---- */}
      <Stack direction="row" spacing={`${PANEL_GAP}px`}>
        <ImagePanel
          offscreen={phaseOff} offscreenVersion={phaseVersion} label="Phase" size={panel} tc={tc}
          zoom={phaseZoom} onZoomChange={setPhaseZoom}
          showResize onResizeStart={startResize}
          pixelSize={pixelSize} imageWidth={rawPhaseRef.current?.w}
          rawData={activeRealDataRef.current?.data ?? rawPhaseRef.current?.data}
          smooth={smooth}
          cropRegion={cropRefitAvailable ? scanCrop : null}
          cropSelecting={cropRefitAvailable && cropSelecting}
          onCropChange={chooseCropRectangle}
          inset={fftAsInset ? (
            <FFTInset
              offscreen={fftOff} offscreenVersion={fftVersion}
              rawData={fftMagRef.current?.mag}
              size={fftInsetSize}
              panelSize={panel}
              box={fftInsetBox}
              onBoxChange={setFFTInsetBox}
              tc={tc}
              smooth={smooth}
              pixelSize={pixelSize}
            />
          ) : undefined}
        />
        {extraRealViews.amp && (
          <ImagePanel
            offscreen={ampOff} offscreenVersion={ampVersion} label="Amplitude" size={panel} tc={tc}
            zoom={ampZoom} onZoomChange={setAmpZoom}
            showResize onResizeStart={startResize}
            pixelSize={pixelSize} imageWidth={rawPhaseRef.current?.w}
            rawData={ampDataRef.current?.data}
            smooth={smooth}
          />
        )}
        {extraRealViews.complex && (
          <ImagePanel
            offscreen={complexOff} label="Complex" size={panel} tc={tc}
            zoom={complexZoom} onZoomChange={setComplexZoom}
            showResize onResizeStart={startResize}
            pixelSize={pixelSize} imageWidth={rawPhaseRef.current?.w}
            rawData={ampDataRef.current?.data ?? rawPhaseRef.current?.data}
            smooth={smooth}
          />
        )}
        {fftAsPanel && (
          <ImagePanel
            offscreen={fftOff} offscreenVersion={fftVersion} label="FFT" size={panel} tc={tc}
            zoom={fftZoom} onZoomChange={setFFTZoom}
            showResize onResizeStart={startResize}
            pixelSize={fftPixelSize} imageWidth={rawPhaseRef.current?.w} isFFT
            rawData={fftMagRef.current?.mag}
          />
        )}
      </Stack>

      {/* ---- Controls ---- */}
      <Box sx={{ width: totalW, mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>

        {/* Controls + Histogram row */}
        <Box sx={{ display: "flex", gap: `${SPACING.SM}px`, width: "100%" }}>
          {/* Left: stats + controls */}
          <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
            {/* Stats row — each readout gets a hover tooltip so new users know what the numbers mean */}
            <Box sx={controlBand(tc)}>
              <Tooltip title="Defocus in nanometers.  Primary aberration — sets the probe's axial focus point relative to the sample.  Usually the first knob to get right." placement="top" arrow>
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
              {loss != null && (
                <Tooltip title={hoActive
                  ? "BF-disk phase variance from the full 14-coef kernel — the same metric the 3-param optimizer minimizes.  Drag a higher-order slider and watch this number: lower = sharper reconstruction.  Compare to the HO=0 baseline to confirm your tuning actually helps."
                  : "Variance of the reconstructed phase inside the BF (brightfield) disk — the central region of the diffraction pattern where the direct beam lands.  SSB uses that region's scattered intensity to recover phase; lower loss ≈ sharper reconstruction."} placement="top" arrow>
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
                  .map(e => {
                    const magKey = e.hasAngle ? `${e.name}_mag` : e.name;
                    const mag = higherOrder[magKey] ?? 0;
                    if (Math.abs(mag) === 0) return null;
                    const ang = e.hasAngle ? (higherOrder[`${e.name}_angle`] ?? 0) : null;
                    const label = `${formatHOValue(mag, e.mag_max, e.display_scale)} ${e.unit_display}${ang != null ? ` / ${ang.toFixed(0)}°` : ""}`;
                    return (
                      <Tooltip key={e.name} title={e.tooltip} placement="top" arrow>
                        <Typography sx={{ ...typography.label, color: tc.textMuted, fontFamily: "monospace", cursor: "help" }}>
                          {e.name} <Box component="span" sx={{ color: tc.accent }}>{label}</Box>
                        </Typography>
                      </Tooltip>
                    );
                  })
                  .filter(Boolean)
              )}
              {(gpuMs != null || uiMs != null) && (
                <Typography
                  sx={{ ...typography.value, ...statChip(tc) }}
                  title={
                    stageTiming
                      ? `GPU=${gpuMs?.toFixed(0)}ms (Python kernel)\n` +
                        `d2h=${stageTiming.d2h.toFixed(1)}ms (cp.asnumpy)\n` +
                        `bytes=${stageTiming.bytes.toFixed(1)}ms (ndarray.tobytes)\n` +
                        `trait=${stageTiming.trait.toFixed(1)}ms (Comm enqueue)\n` +
                        `clip=${stageTiming.clip.toFixed(1)}ms (percentileClip)\n` +
                        `render=${stageTiming.render.toFixed(1)}ms (LUT→offscreen)\n` +
                        `setState=${stageTiming.setState.toFixed(1)}ms\n` +
                        `paint=${stageTiming.paint.toFixed(1)}ms (rAF after setState)\n` +
                        `UI=${uiMs?.toFixed(0)}ms (slider → next paint)\n` +
                        `gap=${(uiMs! - (gpuMs||0) - (stageTiming.d2h+stageTiming.bytes+stageTiming.trait+stageTiming.clip+stageTiming.render+stageTiming.setState+stageTiming.paint)).toFixed(0)}ms (Comm wire + React scheduling)`
                      : "hover for breakdown"
                  }
                >
                  {gpuMs != null && `GPU ${gpuMs.toFixed(0)}`}
                  {jsMs != null && ` / JS ${jsMs.toFixed(0)}`}
                  {uiMs != null && ` / UI ${uiMs.toFixed(0)}`}ms
                </Typography>
              )}
              {busy && <Box component="span" sx={{ color: tc.accent }}>●</Box>}
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
              <Typography sx={{ ...typography.value, color: tc.textMuted, opacity: 0.7 }}>
                auto {autoC10?.toFixed(0)} / {autoC12?.toFixed(0)} / {autoPhi12?.toFixed(0)}° = {autoLoss?.toFixed(8)}
              </Typography>
            </Box>

            {!localSourceGranted && (
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
                  Alternative: `quantem showptycho &lt;folder&gt;` serves and opens this automatically.
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
              <Typography sx={{ ...typography.label, fontSize: 10 }} title="CSS bilinear interpolation on the phase canvas. Same data, browser softens pixel grid — useful when upscaling small phase images on a large canvas.">Smooth</Typography>
              <Switch
                checked={smooth}
                onChange={(_, v) => setSmooth(v)}
                size="small" sx={switchStyles.small}
              />
              {showFFT && (
                <>
                  <Box sx={{ width: "1px", height: 18, bgcolor: tc.border, mx: 0.5 }} />
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT</Typography>
                  <Select
                    value={fftCmap} onChange={(e) => setFftCmap(e.target.value)}
                    size="small" sx={{ ...themedSelect, minWidth: 78, fontSize: 10 }}
                    MenuProps={themedMenuProps}
                  >
                    {COLORMAP_NAMES.map(n => (
                      <MenuItem key={n} value={n}>{n.charAt(0).toUpperCase() + n.slice(1)}</MenuItem>
                    ))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }} title="Auto-contrast for FFT magnitude. Same as Phase Auto.">Auto</Typography>
                  <Switch
                    checked={fftAutoContrast}
                    onChange={(e) => { const on = e.target.checked; setFftAutoContrast(on); if (on) setFftContrastRange([1, 99]); }}
                    size="small" sx={switchStyles.small}
                  />
                </>
              )}
            </Box>
          </Box>

          {/* Right: Histograms (Phase + FFT, stacked).  Dragging either slider
              flips the matching Auto switch OFF so the user's range persists. */}
          <Box sx={{ display: "flex", flexDirection: "row", alignItems: "flex-end", gap: 1, flexShrink: 0 }}>
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0.25 }}>
              <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>
                Phase
              </Typography>
              <Histogram
                data={activeRealDataRef.current?.data ?? rawPhaseRef.current?.data ?? null}
                vminPct={contrastRange[0]}
                vmaxPct={contrastRange[1]}
                onRangeChange={(lo, hi) => { setContrastRange([lo, hi]); setAutoContrast(false); }}
                width={110}
                height={48}
                tc={tc}
                dataMin={dataRange.min}
                dataMax={dataRange.max}
              />
            </Box>
            {showFFT && (
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0.25 }}>
                <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>FFT</Typography>
                <Histogram
                  data={fftMagRef.current?.mag ?? null}
                  vminPct={fftContrastRange[0]}
                  vmaxPct={fftContrastRange[1]}
                  onRangeChange={(lo, hi) => { setFftContrastRange([lo, hi]); setFftAutoContrast(false); }}
                  width={110}
                  height={48}
                  tc={tc}
                  dataMin={fftDataRange.min}
                  dataMax={fftDataRange.max}
                />
              </Box>
            )}
          </Box>
        </Box>

        {/* Sliders.  When a slider's parameter is the ACTIVE sweep target, it
            renders as a 3-thumb slider: outer thumbs = PLAY sweep bounds, middle
            thumb = current value.  Everything else stays single-thumb.  CSS below
            dims the outer thumbs so the current-value thumb stays visually
            dominant — the range markers are meant to look like brackets, not
            competing handles. */}
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
          {/* Each aberration column = main single-thumb slider PLUS a tiny 2-thumb
              stopper slider underneath.  Main slider value is unconstrained — the
              stoppers only affect PLAY bounds.  Moving the middle value never
              collides with the stopper thumbs because they're separate MUI
              Slider instances with independent thumbs. */}
          {([
            {
              key: "c10" as AberKey, label: "C10", unit: "nm", value: c10, displayPrec: 0,
              tipFull: "C10 — defocus.  Positive = overfocus, negative = underfocus.  The dominant aberration.",
              min: c10UiMin, max: c10UiMax, step: 1,
              setValue: (v: number) => { setC10(v); sendDrag(v, sliderVals.current.c12, sliderVals.current.phi12); },
              commitValue: (v: number) => { if (shouldCommitOnReleaseRef.current) sendCommit(v, sliderVals.current.c12, sliderVals.current.phi12); },
            },
            {
              key: "c12" as AberKey, label: "C12", unit: "nm", value: c12, displayPrec: 0,
              tipFull: "C12 — 2-fold astigmatism magnitude.  Paired with φ₁₂.",
              min: c12UiMin, max: c12UiMax, step: 1,
              setValue: (v: number) => { setC12(v); sendDrag(sliderVals.current.c10, v, sliderVals.current.phi12); },
              commitValue: (v: number) => { if (shouldCommitOnReleaseRef.current) sendCommit(sliderVals.current.c10, v, sliderVals.current.phi12); },
            },
            {
              key: "phi12" as AberKey, label: "φ₁₂", unit: "°", value: phi12, displayPrec: 0,
              tipFull: "φ₁₂ — angle of 2-fold astigmatism.  Meaningful when C12 ≠ 0.",
              min: phi12UiMin, max: phi12UiMax, step: 1,
              setValue: (v: number) => { setPhi12(v); sendDrag(sliderVals.current.c10, sliderVals.current.c12, v); },
              commitValue: (v: number) => { if (shouldCommitOnReleaseRef.current) sendCommit(sliderVals.current.c10, sliderVals.current.c12, v); },
            },
            {
              key: "rot" as AberKey, label: "scan-det rot", unit: "°", value: rotationDeg ?? 0, displayPrec: 1,
              tipFull: "Scan-detector rotation — angle between scan axes and detector axes.  Sweep to find the sharpest reconstruction.",
              min: rotationUiMin, max: rotationUiMax, step: 0.1,
              setValue: (v: number) => {
                setRotationDeg(v);
                if (webgpuSsbRef.current) sendDrag(sliderVals.current.c10, sliderVals.current.c12, sliderVals.current.phi12, v);
              },
              commitValue: (v: number) => {
                if (shouldCommitOnReleaseRef.current) sendCommit(sliderVals.current.c10, sliderVals.current.c12, sliderVals.current.phi12, v);
              },
            },
          ] as const).map(cfg => {
            const range = sweepRanges[cfg.key];
            const isActiveSweep = sweepMainKeys(sweepParam).includes(cfg.key);
            const isAtFull = range[0] === cfg.min && range[1] === cfg.max;
            const sliderWidth = cfg.key === "rot" ? ROTATION_SLIDER_WIDTH : ABERRATION_SLIDER_WIDTH;
            const visibleValue = Math.max(cfg.min, Math.min(cfg.max, cfg.value));
            return (
              <Box key={cfg.key} sx={{ flex: `0 0 ${sliderWidth}px`, width: sliderWidth, minWidth: 0 }}>
                <Tooltip title={cfg.tipFull} placement="top" arrow>
                  <Typography sx={{ ...typography.labelSmall, color: tc.textMuted, mb: -0.5, cursor: "help" }}>
                    {cfg.label} <b>{cfg.value.toFixed(cfg.displayPrec)}</b> {cfg.unit}
                    {!isAtFull && (
                      <span style={{ color: isActiveSweep ? tc.accent : tc.textMuted, opacity: 0.65 }}>
                        {" ["}{range[0].toFixed(cfg.displayPrec)}, {range[1].toFixed(cfg.displayPrec)}{"]"}
                      </span>
                    )}
                  </Typography>
                </Tooltip>
                {/* Show3D-style 3-thumb slider: outer thumbs = sweep stoppers,
                    middle thumb = current value.  disableSwap keeps order stable;
                    to push the value outside the stopper window, drag the relevant
                    stopper first.  Zero separate sub-slider — everything on one
                    track. */}
                <Slider
                  value={[range[0], visibleValue, range[1]]}
                  min={cfg.min} max={cfg.max} step={cfg.step}
                  disableSwap
                  onChange={(_, v, activeThumb) => {
                    const arr = v as number[];
                    if (activeThumb === 1) {
                      cfg.setValue(arr[1]);
                    } else {
                      updateSweepRange(cfg.key, [arr[0], arr[2]]);
                    }
                  }}
                  onChangeCommitted={(_, v) => {
                    const arr = v as number[];
                    cfg.commitValue(arr[1]);
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
          <Tooltip title="Flip phase sign.  SSB's phase is defined only up to ± (sign-ambiguous).  Use this to match the expected contrast of your sample.  Does not re-run reconstruction — just negates the displayed phase." placement="top" arrow>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, cursor: "help" }}>
              <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>flip</Typography>
              <Switch checked={!!flipPhase} onChange={(_, v) => setFlipPhase(v)} size="small" />
            </Box>
          </Tooltip>
          <Tooltip title="Reset all aberrations to the automatic Nelder-Mead result and restore every sweep-stopper to its full range." placement="top" arrow>
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

        {/* Higher-order aberration panel — collapsible.  Shows 11 Krivanek
            coefficients (C21 through C56), grouped by order.  When any
            magnitude is non-zero the engine routes through the 14-coef
            kernel and the loss readout is set to "manual" (the optimizer
            only knows the 3-param space). */}
        <HigherOrderPanel
          tc={tc}
          open={hoOpen}
          onToggle={() => setHoOpen(v => !v)}
          activeCount={hoActiveCount}
          values={higherOrder}
          setValues={setHigherOrder}
        />

        {/* Single compact action row — PLAY + PIN + RAND + RESET + SAVE CALIBRATION.
            Dropdowns and scope chips live inline.  Wraps to a second line only
            on narrow widgets (below ~520 px panel size). */}
        <Box sx={controlBand(tc)}>
          <Tooltip title={playing ? "Pause sweep (Space)" : "Play: sweep the selected aberration (Space)"} placement="top" arrow>
            <IconButton
              size="small"
              disableRipple
              disableFocusRipple
              disableTouchRipple
              onClick={() => setPlaying(p => !p)}
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
            <MenuItem value="rot">rot</MenuItem>
            <MenuItem value="bundle:c10c12">C10+C12</MenuItem>
            <MenuItem value="bundle:c10c12phi12">C10+C12+φ₁₂</MenuItem>
            <MenuItem value="bundle:c10c12phi12rot">C10+C12+φ₁₂+rot</MenuItem>
            {Object.keys(higherOrder)
              .filter(k => (k.endsWith("_angle")
                ? (higherOrder[k] !== undefined)
                : Math.abs(higherOrder[k] || 0) > 0))
              .sort()
              .map(k => (
                <MenuItem key={`ho:${k}`} value={`ho:${k}`}>
                  {k.endsWith("_mag") ? `${k.slice(0, -4)} m`
                   : k.endsWith("_angle") ? `${k.slice(0, -6)} φ`
                   : `${k}`}
                </MenuItem>
              ))}
          </Select>
          <Select
            value={playFps}
            onChange={(e) => setPlayFps(Number(e.target.value))}
            size="small"
            sx={{ ...actionSelect, width: 52, minWidth: 52, flex: "0 0 52px" }}
            title="Sweep FPS — ≥15 needs the drag BF subset"
          >
            {[1, 3, 5, 10, 15, 30].map(fps => (<MenuItem key={fps} value={fps}>{fps}</MenuItem>))}
          </Select>
          <Badge
            badgeContent={editedRangeCount}
            invisible={editedRangeCount === 0}
            sx={{ "& .MuiBadge-badge": { bgcolor: tc.accent, color: "#fff", fontSize: 9, fontWeight: 600, minWidth: 14, height: 14, px: 0.25 } }}
          >
            <Button
              size="small"
              sx={{ ...compactButton(tc), height: ACTION_CONTROL_HEIGHT, width: 52, minWidth: 52, color: editedRangeCount > 0 ? tc.accent : tc.text }}
              onClick={(e) => setMoreMenuAnchor(e.currentTarget)}
              aria-label="More sweep controls"
              aria-controls={moreMenuAnchor ? "showptycho-more-menu" : undefined}
              aria-expanded={moreMenuAnchor ? "true" : undefined}
              aria-haspopup="menu"
              title="More sweep controls"
            >
              More
            </Button>
          </Badge>
          <Menu
            id="showptycho-more-menu"
            anchorEl={moreMenuAnchor}
            open={Boolean(moreMenuAnchor)}
            onClose={() => setMoreMenuAnchor(null)}
            MenuListProps={{ "aria-label": "More sweep controls" }}
            {...themedMenuProps}
          >
            <Box
              onClick={(event) => event.stopPropagation()}
              onKeyDown={(event) => event.stopPropagation()}
              sx={{ px: 1.25, py: 1, minWidth: 280, display: "flex", flexDirection: "column", gap: 0.75 }}
            >
              <Stack direction="row" alignItems="center" spacing={1}>
                <Typography sx={{ ...typography.label, color: tc.text, fontWeight: 700, flex: 1 }}>
                  Sweep Ranges
                </Typography>
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
          </Menu>

          {/* Divider: PLAY group | PIN */}
          <Box sx={{ width: "1px", height: 20, bgcolor: tc.border, mx: 0.5, alignSelf: "center" }} />
          <Tooltip title="Pin the current snapshot (P)" placement="top" arrow>
            <IconButton
              size="small"
              disableRipple
              disableFocusRipple
              disableTouchRipple
              onClick={doPin}
              disabled={busy || !rawPhaseRef.current}
              sx={{ ...compactIconButton(tc, tc.accent), height: ACTION_CONTROL_HEIGHT }}
              aria-label="Pin current snapshot"
            >
              <PushPinIcon sx={{ fontSize: 15 }} />
            </IconButton>
          </Tooltip>
          {/* Divider: PIN | RAND */}
          <Box sx={{ width: "1px", height: 20, bgcolor: tc.border, mx: 0.5, alignSelf: "center" }} />
          <Tooltip title="Jump to random aberration values.  Click chips to scope." placement="top" arrow>
            <IconButton
              size="small"
              disableRipple
              disableFocusRipple
              disableTouchRipple
              onClick={doRandom}
              sx={{ ...compactIconButton(tc, tc.textMuted), height: ACTION_CONTROL_HEIGHT }}
              aria-label="Randomize aberrations"
            >
              <ShuffleIcon sx={{ fontSize: 15 }} />
            </IconButton>
          </Tooltip>
          {[
            { key: "c10",      label: "C10"  },
            { key: "c12",      label: "C12"  },
            { key: "phi12",    label: "φ₁₂"  },
            { key: "rotation", label: "rot"  },
          ].map(({ key, label }) => (
            <Box
              key={key}
              onClick={() => setRandScope(s => ({ ...s, [key]: !s[key as keyof typeof s] }))}
              sx={{
                height: ACTION_CONTROL_HEIGHT,
                width: key === "phi12" ? 34 : 30,
                flex: `0 0 ${key === "phi12" ? 34 : 30}px`,
                px: 0.5, py: 0,
                fontSize: 9, fontFamily: "monospace",
                display: "inline-flex", alignItems: "center", justifyContent: "center",
                cursor: "pointer", userSelect: "none",
                border: `1px solid ${tc.border}`,
                color: randScope[key as keyof typeof randScope] ? tc.accent : tc.textMuted,
                opacity: randScope[key as keyof typeof randScope] ? 1 : 0.4,
                bgcolor: randScope[key as keyof typeof randScope] ? tc.controlBg : "transparent",
              }}
            >
              {label}
            </Box>
          ))}
          {/* Divider: RAND chips | SAVE CALIBRATION.  The big RESET button is
              gone — each aberration slider now has its own inline ↺. */}
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
              flex: 1, minWidth: 120, fontFamily: "monospace", fontSize: 11,
              bgcolor: tc.bgAlt, color: tc.text,
              border: `1px solid ${tc.border}`, borderRadius: 0,
              px: 0.75, py: 0, outline: "none",
              "&:focus": { borderColor: tc.accent },
            }}
          />
          {calibrationPath && (
            <Tooltip title={`Last saved at ${calibrationSavedAt || "—"}`} placement="top" arrow>
              <Typography sx={{
                ...typography.value, color: STATUS_GOOD,
                maxWidth: 260, overflowWrap: "anywhere",
              }}>
                ✓ {compactPathLabel(calibrationPath, 2)}
              </Typography>
            </Tooltip>
          )}
        </Box>

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
                  C10 {record.C10.toFixed(1)} · C12 {record.C12.toFixed(1)} · φ12 {record.phi12_deg.toFixed(0)}° · rot {record.rotation_deg.toFixed(1)}°
                  {record.loss != null ? ` · loss ${record.loss.toFixed(6)}` : ""}
                  {` · ${record.timestamp.slice(0, 16).replace("T", " ")}`}
                  {record.notes ? ` · ${record.notes}` : ""}
                </Typography>
                <Button size="small" onClick={() => loadFolderSave(record)} sx={{ ...compactButton(tc), minWidth: 44 }}>Load</Button>
                <Button size="small" onClick={() => void downloadFolderSave(record)} sx={{ ...compactButton(tc), minWidth: 30 }}>⬇</Button>
                <Button size="small" onClick={() => void deleteFolderSave(record)} sx={{ ...compactButton(tc), minWidth: 30, color: STATUS_BAD }}>✕</Button>
              </Box>
            ))}
          </Box>
        )}

        {/* Optuna trials strip — scrollable, loss-ranked.  Click any tile to
            preview that trial's aberrations.  Best-loss tile is highlighted so
            the user can immediately verify that Nelder-Mead's refined point is near
            (or past) Optuna's best. */}
        {trials.length > 0 && (
          <Box sx={{ pt: `${SPACING.XS}px`, borderTop: `1px solid ${tc.border}` }}>
            <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} sx={{ mb: `${SPACING.XS}px` }}>
              <Box
                onClick={() => setTrialsExpanded(v => !v)}
                sx={{ cursor: "pointer", userSelect: "none", color: tc.textMuted, fontSize: 10, fontFamily: "monospace" }}
              >
                {trialsExpanded ? "▼" : "▶"} Optuna trials ({trials.length}) + Nelder-Mead · loss-sorted · click to preview
              </Box>
              <Box sx={{ flex: 1 }} />
              {trials.length > 0 && (
                <Tooltip title={`Optuna explored ${trials.length} trials. Nelder-Mead then refined the best one. Auto loss = ${autoLoss?.toFixed(8) ?? '—'}; best Optuna loss = ${formatNumber(trials[0].loss, 8)}.`} placement="top" arrow>
                  <Typography sx={{ ...typography.value, color: tc.textMuted, cursor: "help" }}>
                    Nelder-Mead {autoLoss?.toFixed(8) ?? '—'}  ·  best Optuna {formatNumber(trials[0].loss, 8)}
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
                  <Tooltip title="Nelder-Mead refined the best Optuna trial.  Click to jump sliders back to this result.  This is what 'auto' in the stats bar refers to." placement="top" arrow>
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
                          Nelder-Mead
                        </Typography>
                        <Typography sx={{ fontSize: 11, color: STATUS_GOOD, fontFamily: "monospace", fontWeight: "bold" }}>
                          {autoLoss?.toFixed(8) ?? '—'}
                        </Typography>
                      </Stack>
                      <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: tc.textMuted, whiteSpace: "nowrap" }}>
                        {autoC10.toFixed(0)} / {autoC12.toFixed(0)} / {autoPhi12.toFixed(0)}°
                      </Typography>
                    </Box>
                  </Tooltip>
                )}
                {trials.map((t, i) => {
                  // Color ramp: best trial = accent, worst trial = border.  Interpolate
                  // via opacity so the CSS reads simply from a single accent token.
                  const frac = trials.length > 1 ? i / (trials.length - 1) : 0;
                  const isActive = activeTrialRank === t.rank;
                  return (
                    <Box
                      key={t.rank}
                      onClick={() => doViewTrial(t)}
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
                          #{t.rank + 1}
                        </Typography>
                        <Typography sx={{ fontSize: 11, color: tc.accent, fontFamily: "monospace", fontWeight: "bold" }}>
                          {formatNumber(t.loss, 8)}
                        </Typography>
                      </Stack>
                      <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: tc.textMuted, whiteSpace: "nowrap" }}>
                        {t.C10.toFixed(0)} / {t.C12.toFixed(0)} / {t.phi12_deg.toFixed(0)}°
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
            Viewing pinned — drag slider or Esc to return
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
              {starsPath && pinned.some(p => p.starred) && (
                <Tooltip title={`Starred snapshots auto-save to ${starsPath}.  Load in the next cell with json.load(open(path)).`} placement="top" arrow>
                  <Typography sx={{ ...typography.value, color: tc.textMuted, opacity: 0.65, cursor: "help", overflowWrap: "anywhere" }}>
                    ★ → {compactPathLabel(starsPath, 2)}
                  </Typography>
                </Tooltip>
              )}
            </Stack>
            <Stack direction="row" spacing={`${SPACING.XS}px`} sx={{ overflowX: "auto", pb: 0.5 }}>
              {pinned.map(p => {
                const px = THUMB_SIZE_PX[thumbSize];
                const isDraggingPin = draggingPin === p.id;
                const isNewPin = newPinId === p.id;
                return (
                  <Box
                    key={p.id}
                    data-showptycho-pin-id={p.id}
                    data-showptycho-pin-state={isDraggingPin ? "dragging" : isNewPin ? "new" : undefined}
                    title="Drag to reorder pinned images. Click to view."
                    onClick={() => {
                      if (didDragPinRef.current) {
                        didDragPinRef.current = false;
                        return;
                      }
                      doViewPin(p);
                    }}
                    onPointerDown={(e) => {
                      if (e.button !== 0) return;
                      beginPinDrag(p.id, e.clientX, e.clientY);
                      (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
                    }}
                    onPointerMove={(e) => {
                      if (updatePinDrag(p.id, e.clientX, e.clientY)) e.preventDefault();
                    }}
                    onPointerUp={(e) => {
                      finishPinDrag(p.id, e.clientX, e.clientY);
                    }}
                    onPointerCancel={() => {
                      cancelPinDrag();
                    }}
                    onMouseDown={(e) => {
                      if (e.button !== 0) return;
                      beginPinDrag(p.id, e.clientX, e.clientY);
                    }}
                    onMouseMove={(e) => {
                      if (updatePinDrag(p.id, e.clientX, e.clientY)) e.preventDefault();
                    }}
                    onMouseUp={(e) => {
                      finishPinDrag(p.id, e.clientX, e.clientY);
                    }}
                    onMouseLeave={(e) => {
                      if (pinPointerDragRef.current?.active) {
                        updatePinDrag(p.id, e.clientX, e.clientY);
                      }
                    }}
                    onDragStart={(e) => {
                      draggedPinRef.current = p.id;
                      didDragPinRef.current = true;
                      setDragOverPin(p.id);
                      e.dataTransfer.effectAllowed = "move";
                      e.dataTransfer.setData("text/plain", String(p.id));
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      e.dataTransfer.dropEffect = "move";
                      setDragOverPin(p.id);
                    }}
                    onDrop={(e) => {
                      e.preventDefault();
                      const fromId = Number(e.dataTransfer.getData("text/plain") || draggedPinRef.current);
                      if (Number.isFinite(fromId)) reorderPinned(fromId, p.id);
                      draggedPinRef.current = null;
                      setDragOverPin(null);
                    }}
                    onDragEnd={() => {
                      draggedPinRef.current = null;
                      setDragOverPin(null);
                      window.setTimeout(() => { didDragPinRef.current = false; }, 0);
                    }}
                    sx={{
                    position: "relative", cursor: "grab", flexShrink: 0,
                    border: viewPin === p.id
                      ? `2px solid ${tc.accent}`
                      : p.starred ? `2px solid ${STATUS_GOOD}` : "2px solid transparent",
                    outline: dragOverPin === p.id ? `2px solid ${tc.accent}` : "none",
                    outlineOffset: 1,
                    boxShadow: isDraggingPin ? `0 6px 16px ${tc.shadow}` : "none",
                    transform: isDraggingPin ? "scale(1.045)" : "scale(1)",
                    transition: "border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease, outline-color 120ms ease",
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
                      key={`${p.id}-${cmap}`}
                      data-cmap={cmap}
                      ref={el => {
                        if (el && p.thumb) {
                          el.width = THUMB_BITMAP_PX; el.height = THUMB_BITMAP_PX;
                          el.getContext("2d")!.drawImage(p.thumb, 0, 0);
                        }
                      }}
                      style={{ width: px, height: px, display: "block", pointerEvents: "none" }}
                    />
                    {/* Bottom metadata strip — sized so small thumbs stay readable. */}
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
                        {p.C10.toFixed(0)}/{p.C12.toFixed(0)}/{p.phi12_deg.toFixed(0)}°
                      </Typography>
                    </Box>
                    {/* Star toggle (top-left).  Gold when active, muted otherwise. */}
                    <Tooltip title={p.starred ? "Unstar (remove from time-series JSON)" : "Star — save to JSON time-series"} placement="top" arrow>
                      <Box
                        onClick={e => { e.stopPropagation(); doToggleStar(p.id); }}
                        sx={{
                          position: "absolute", top: 0, left: 0,
                          width: thumbSize === "L" ? 20 : 16, height: thumbSize === "L" ? 20 : 16,
                          bgcolor: "rgba(0,0,0,0.6)",
                          display: "flex", alignItems: "center", justifyContent: "center",
                          cursor: "pointer", borderRadius: "0 0 4px 0",
                          fontSize: thumbSize === "L" ? 13 : 11,
                          color: p.starred ? "#ffd54a" : tc.textMuted,
                          "&:hover": { color: "#ffd54a" },
                        }}
                      >{p.starred ? "★" : "☆"}</Box>
                    </Tooltip>
                    {/* Unpin (top-right) */}
                    <Box onClick={e => { e.stopPropagation(); doUnpin(p.id); }} sx={{
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

const render = createRender(Explore);

export default { render };
