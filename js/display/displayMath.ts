// CPU colormap (Float32 -> RGBA via 256-entry LUT): the float32 normalize and
// log the WGSL display shaders repeat, and the Canvas2D paint built on them.

/**
 * Map a display value to [0, 1] exactly as the WGSL display_normalize does, in
 * float32: NaN and -inf go to 0, +inf to 1, an empty window to 0.5, and a
 * window whose float32 span overflows is halved first (exact), so
 * (value - low) / (high - low) is computed the same way for every window.
 * Every step is a correctly rounded float32 operation on both paths (the WGSL
 * divides through display_divide, not the GPU's inexact `/`), which is what
 * lets the Canvas2D and WebGPU paths choose the same LUT entry for a value that
 * sits on a level edge.
 */
export function displayNormalize(value: number, low: number, high: number): number {
  const f32 = Math.fround;
  if (!Number.isFinite(value)) return value === Infinity ? 1 : 0;
  if (!(high > low)) return 0.5;
  const scale = Number.isFinite(f32(high - low)) ? 1 : 0.5;
  const span = f32(f32(high * scale) - f32(low * scale));
  return Math.min(1, Math.max(0, f32(f32(f32(value * scale) - f32(low * scale)) / span)));
}

/** Signed log1p in float32, the WGSL log_scale branch: log(1 + v), or -log(1 - v) below zero. */
export function displayLog(value: number): number {
  const f32 = Math.fround;
  return value >= 0 ? f32(Math.log(f32(1 + value))) : f32(-Math.log(f32(1 - value)));
}

// Packed LUTs by LUT array: every paint of a colormap reuses one.
const packedLuts = new WeakMap<Uint8Array, Uint32Array>();

/**
 * A 256-entry RGB LUT as one RGBA word per entry, so a pixel is one 32-bit
 * store. The words are built from bytes, so they land in the right byte order
 * on any platform.
 */
export function packedLut(lut: Uint8Array): Uint32Array {
  const cached = packedLuts.get(lut);
  if (cached) return cached;
  const bytes = new Uint8Array(256 * 4);
  for (let entry = 0; entry < 256; entry++) {
    bytes[entry * 4] = lut[entry * 3];
    bytes[entry * 4 + 1] = lut[entry * 3 + 1];
    bytes[entry * 4 + 2] = lut[entry * 3 + 2];
    bytes[entry * 4 + 3] = 255;
  }
  const packed = new Uint32Array(bytes.buffer);
  packedLuts.set(lut, packed);
  return packed;
}

/**
 * Colormap `data` into `target`: target pixel k shows source pixel
 * sourceIndex[k] (identity when sourceIndex is null; a negative index stays
 * transparent). LUT entry = min(trunc(normalize(v) * 255), 255) in float32,
 * the WGSL colormap pass. The common window (finite, non-empty float32 span)
 * runs displayNormalize's own arithmetic inline: this loop touches every
 * displayed pixel on every contrast tick.
 */
export function colormapPixels(
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

/** Apply colormap LUT to float data, writing into an RGBA Uint8ClampedArray. */
export function applyColormap(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
): void {
  colormapPixels(new Uint32Array(rgba.buffer, rgba.byteOffset, data.length), data, null, { vmin, vmax }, false, packedLut(lut));
}

/** Create an offscreen canvas with colormapped data. Returns null if context unavailable. */
export function renderToOffscreen(
  data: Float32Array,
  width: number,
  height: number,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
): HTMLCanvasElement | null {
  const offscreen = document.createElement("canvas");
  offscreen.width = width;
  offscreen.height = height;
  const ctx = offscreen.getContext("2d");
  if (!ctx) return null;
  const imgData = ctx.createImageData(width, height);
  applyColormap(data, imgData.data, lut, vmin, vmax);
  ctx.putImageData(imgData, 0, 0);
  return offscreen;
}

/** Render colormapped data to a reusable offscreen canvas + ImageData (avoids per-frame allocation). */
export function renderToOffscreenReuse(
  data: Float32Array,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  offscreen: HTMLCanvasElement,
  imgData: ImageData,
): void {
  applyColormap(data, imgData.data, lut, vmin, vmax);
  offscreen.getContext("2d")!.putImageData(imgData, 0, 0);
}
