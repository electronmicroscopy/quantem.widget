// Canvas2D twin of the GPUColormapEngine slot API that Show2D drives: data
// ranges, 256-bin histograms and colormapped bitmaps per image slot. It runs
// when the browser has no hardware WebGPU, so the widget code path is the same
// on both backends. Every step repeats the WGSL arithmetic in float32
// (Math.fround) so both paths pick the same LUT entry and histogram bin.
// Every paint names its colormap: the image panels and the FFT panels paint
// from separate animation frames, so a LUT shared by the engine would let one
// pipeline repaint the other's panels in its colormap.

import { createGPUColormapEngine, displayLog, displayNormalize } from "./colormaps";
import { type RenderPath } from "./device";

/** The slot calls Show2D makes; GPUColormapEngine and CPUColormapEngine provide them. */
export interface DisplayColormapEngine {
  readonly path: RenderPath;
  readonly slotCount: number;
  uploadData(idx: number, data: Float32Array, width?: number, height?: number): void;
  uploadUint8Data(idx: number, data: Uint8Array | Uint8ClampedArray, width: number, height: number, rgbaCapacityHint: number): void;
  computeRangeBatch(indices: number[]): Promise<{ min: number; max: number }[]>;
  computeHistogramBatch(indices: number[], ranges: { min: number; max: number }[], logScale?: boolean): Promise<number[][]>;
  computeHistogramWithRange(idx: number, dmin: number, dmax: number, logScale?: boolean): Promise<number[]>;
  renderSlotsToImageBitmapAsync(indices: number[], ranges: { vmin: number; vmax: number }[], logScale: boolean, lutName: string, lut: Uint8Array): Promise<ImageBitmap[] | null>;
  renderSlotScaledToImageBitmapAsync(idx: number, range: { vmin: number; vmax: number }, logScale: boolean, outW: number, outH: number, lutName: string, lut: Uint8Array): Promise<ImageBitmap | null>;
  destroy(): void;
}

type CPUSlot = { data: Float32Array | Uint8Array; width: number; height: number; finite: boolean };

/**
 * 256-bin histogram of the WGSL histogram pass: skip non-finite samples, apply
 * the signed log1p when asked, bin = min(trunc(normalize(v) * 256), 255) in
 * float32, then scale so the tallest bin is 1.
 */
export function histogramBins(
  data: ArrayLike<number>,
  dmin: number,
  dmax: number,
  logScale = false,
): number[] {
  const counts = new Uint32Array(256);
  const low = Math.fround(dmin);
  const high = Math.fround(dmax);
  for (let i = 0; i < data.length; i++) {
    let value = data[i];
    if (!Number.isFinite(value)) continue;
    if (logScale) value = displayLog(value);
    counts[Math.min(Math.trunc(Math.fround(displayNormalize(value, low, high) * 256)), 255)] += 1;
  }
  let maxCount = 0;
  for (let i = 0; i < 256; i++) if (counts[i] > maxCount) maxCount = counts[i];
  return Array.from(counts, count => maxCount > 0 ? count / maxCount : 0);
}

/**
 * Colormap engine on WebGPU when a hardware adapter exists and builds the
 * display shaders, else on Canvas2D.
 */
export async function createColormapEngine(): Promise<DisplayColormapEngine> {
  return (await createGPUColormapEngine()) ?? new CPUColormapEngine();
}

/** The Canvas2D engine, for a widget whose WebGPU device was lost mid-session. */
export function createCPUColormapEngine(): DisplayColormapEngine {
  return new CPUColormapEngine();
}

/**
 * Pack a 256-entry RGB LUT as one RGBA word per entry, so a pixel is one 32-bit
 * store. The words are built from bytes, so they land in the right byte order
 * on any platform.
 */
function packLut(lut: Uint8Array): Uint32Array {
  const bytes = new Uint8Array(256 * 4);
  for (let entry = 0; entry < 256; entry++) {
    bytes[entry * 4] = lut[entry * 3];
    bytes[entry * 4 + 1] = lut[entry * 3 + 1];
    bytes[entry * 4 + 2] = lut[entry * 3 + 2];
    bytes[entry * 4 + 3] = 255;
  }
  return new Uint32Array(bytes.buffer);
}

/**
 * Colormap `data` into `target`: target pixel k shows source pixel
 * sourceIndex[k] (identity when sourceIndex is null; a negative index stays
 * transparent). LUT entry = min(trunc(normalize(v) * 255), 255) in float32,
 * the WGSL colormap pass. The common window (finite, non-empty float32 span)
 * runs displayNormalize's own arithmetic inline: this loop touches every
 * displayed pixel on every contrast tick.
 */
function colormapPixels(
  target: Uint32Array,
  data: Float32Array | Uint8Array,
  sourceIndex: Int32Array | null,
  range: { vmin: number; vmax: number },
  logScale: boolean,
  packedLut: Uint32Array,
): void {
  const f32 = Math.fround;
  const low = f32(range.vmin);
  const high = f32(range.vmax);
  const span = f32(high - low);
  const plainWindow = high > low && Number.isFinite(span);
  for (let k = 0; k < target.length; k++) {
    const index = sourceIndex ? sourceIndex[k] : k;
    if (index < 0) { target[k] = 0; continue; }
    const raw = data[index];
    const value = logScale ? displayLog(raw) : raw;
    let normalized: number;
    if (!plainWindow) normalized = displayNormalize(value, low, high);
    else if (Number.isFinite(value)) normalized = Math.min(1, Math.max(0, f32(f32(value - low) / span)));
    else normalized = value === Infinity ? 1 : 0;
    target[k] = packedLut[Math.min(Math.trunc(f32(normalized * 255)), 255)];
  }
}

/**
 * Source pixel of every output pixel of an outW x outH nearest resample, the
 * WGSL scaled-colormap sampling: column min(trunc((outCol + 0.5) * srcW / outW), srcW - 1),
 * rows alike, each step rounded to float32. Null for an equal-size render (the identity).
 */
function nearestSourceIndex(srcW: number, srcH: number, outW: number, outH: number): Int32Array | null {
  if (srcW === outW && srcH === outH) return null;
  const f32 = Math.fround;
  const columns = Int32Array.from({ length: outW }, (_, outCol) => Math.min(Math.trunc(f32(f32(f32(outCol + 0.5) * srcW) / outW)), srcW - 1));
  const index = new Int32Array(outW * outH);
  for (let outRow = 0; outRow < outH; outRow++) {
    const rowStart = Math.min(Math.trunc(f32(f32(f32(outRow + 0.5) * srcH) / outH)), srcH - 1) * srcW;
    for (let outCol = 0; outCol < outW; outCol++) index[outRow * outW + outCol] = rowStart + columns[outCol];
  }
  return index;
}

/**
 * Colormap only the source pixels a canvas shows into `target` (canvas size):
 * target pixel k shows data[sourceIndex[k]], or nothing where the index is
 * negative. A live contrast drag repaints a 600 px canvas from 0.36 M lookups
 * this way instead of recoloring all 16.7 M pixels of a 4096^2 panel per tick,
 * on either engine: the arithmetic is the WGSL colormap pass's.
 */
export function colormapSampledPixels(
  data: Float32Array,
  sourceIndex: Int32Array,
  range: { vmin: number; vmax: number },
  logScale: boolean,
  lut: Uint8Array,
  target: ImageData,
): void {
  colormapPixels(new Uint32Array(target.data.buffer), data, sourceIndex, range, logScale, packLut(lut));
}

class CPUColormapEngine implements DisplayColormapEngine {
  readonly path = "CPU" as const;
  private slots: (CPUSlot | null)[] = [];
  // One output image per size, reused: createImageBitmap copies the pixels
  // when it is called, and a 4096^2 panel would otherwise allocate 64 MB per paint.
  private outputs = new Map<string, ImageData>();
  private packedLuts = new Map<Uint8Array, Uint32Array>();

  get slotCount(): number { return this.slots.filter(slot => slot).length; }

  uploadData(idx: number, data: Float32Array, width?: number, height?: number): void {
    // Same dimension repair as GPUColormapEngine.uploadData (stale mount-time dims).
    const validDims = width && height && width > 1 && height > 1 && width * height === data.length;
    const slotWidth = validDims ? width : Math.round(Math.sqrt(data.length));
    const slotHeight = validDims ? height : Math.round(data.length / slotWidth);
    while (this.slots.length <= idx) this.slots.push(null);
    // Copy, as the GPU upload does: later edits to the caller's array must not repaint.
    this.slots[idx] = { data: Float32Array.from(data), width: slotWidth, height: slotHeight, finite: false };
  }

  uploadUint8Data(idx: number, data: Uint8Array | Uint8ClampedArray, width: number, height: number): void {
    if (width <= 0 || height <= 0 || width * height !== data.length) {
      throw new Error(`uint8 display dimensions ${width}x${height} do not match ${data.length} pixels`);
    }
    while (this.slots.length <= idx) this.slots.push(null);
    this.slots[idx] = { data: Uint8Array.from(data), width, height, finite: true };
  }

  async computeRangeBatch(indices: number[]): Promise<{ min: number; max: number }[]> {
    const results: { min: number; max: number }[] = [];
    for (const idx of indices) {
      const slot = this.slots[idx];
      if (!slot) continue;
      let low = Infinity;
      let high = -Infinity;
      for (let i = 0; i < slot.data.length; i++) {
        const value = slot.data[i];
        if (!slot.finite && !Number.isFinite(value)) continue;
        if (value < low) low = value;
        if (value > high) high = value;
      }
      results.push(high >= low ? { min: low, max: high } : { min: 0, max: 0 });
    }
    return results;
  }

  async computeHistogramBatch(indices: number[], ranges: { min: number; max: number }[], logScale = false): Promise<number[][]> {
    return indices.map((idx, k) => {
      const range = ranges[k] || { min: 0, max: 1 };
      const slot = this.slots[idx];
      return slot ? histogramBins(slot.data, range.min, range.max, logScale) : new Array(256).fill(0);
    });
  }

  async computeHistogramWithRange(idx: number, dmin: number, dmax: number, logScale = false): Promise<number[]> {
    const slot = this.slots[idx];
    return slot ? histogramBins(slot.data, dmin, dmax, logScale) : new Array(256).fill(0);
  }

  async renderSlotsToImageBitmapAsync(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean,
    _lutName: string,
    lut: Uint8Array,
  ): Promise<ImageBitmap[] | null> {
    if (indices.length === 0) return null;
    return Promise.all(indices.map((idx, k) => {
      const slot = this.slots[idx];
      return slot ? this.paint(slot, ranges[k] || { vmin: 0, vmax: 1 }, logScale, slot.width, slot.height, lut) : null as never;
    }));
  }

  async renderSlotScaledToImageBitmapAsync(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    outW: number,
    outH: number,
    _lutName: string,
    lut: Uint8Array,
  ): Promise<ImageBitmap | null> {
    const slot = this.slots[idx];
    if (!slot) return null;
    return this.paint(slot, range, logScale, Math.max(1, Math.round(outW)), Math.max(1, Math.round(outH)), lut);
  }

  destroy(): void {
    this.slots = [];
    this.outputs.clear();
    this.packedLuts.clear();
  }

  private packedLut(lut: Uint8Array): Uint32Array {
    let packed = this.packedLuts.get(lut);
    if (!packed) {
      packed = packLut(lut);
      this.packedLuts.set(lut, packed);
    }
    return packed;
  }

  /** Colormap one slot into an outW x outH bitmap (nearestSourceIndex sampling). */
  private paint(
    slot: CPUSlot,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    outW: number,
    outH: number,
    lut: Uint8Array,
  ): Promise<ImageBitmap> {
    const key = `${outW}x${outH}`;
    let image = this.outputs.get(key);
    if (!image) {
      // Image panels and FFT panels keep their buffers; stale sizes (a resize) go.
      if (this.outputs.size >= 4) this.outputs.clear();
      image = new ImageData(new Uint8ClampedArray(outW * outH * 4), outW, outH);
      this.outputs.set(key, image);
    }
    const sourceIndex = nearestSourceIndex(slot.width, slot.height, outW, outH);
    colormapPixels(new Uint32Array(image.data.buffer), slot.data, sourceIndex, range, logScale, this.packedLut(lut));
    return createImageBitmap(image);
  }
}
