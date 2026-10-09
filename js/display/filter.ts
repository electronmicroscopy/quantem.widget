/// <reference types="@webgpu/types" />
// Browser-side port of the display-only filter pipeline in
// src/quantem/widget/utils/display_filter.py (apply_display_filter).
//
// Covered modes: none, gaussian and anscombe, each with the spatial_bin
// (1, 2 or 4) pre-passes. Other mode names pass the frame through unfiltered.
//
// Two implementations share the same kernel weights, resample coordinates and
// percentile math:
//   - applyDisplayFilterCPU: scalar TypeScript reference, tested against the
//     Python/SciPy implementation; it runs when the browser has no WebGPU.
//   - GPUDisplayFilterEngine: the accelerated WGSL compute passes (separable gaussian,
//     corner-aligned bilinear resample, pointwise Anscombe pre/post) so the
//     sigma slider filters live during drag with zero kernel round trips.
//
// Numerical contract (matches display_filter.py exactly):
//   gaussian_filter  -> scipy default: reflect boundary, truncate=4.0,
//                       radius = int(4*sigma + 0.5), normalized exp weights.
//   _bin2            -> ndimage.zoom(0.5, order=1): output shape is
//                       python-round(n*0.5) (half-even) and coordinates are
//                       corner aligned, in = out * (in-1)/(out-1); optional
//                       gaussian(max(1, sigma/2.5)) on the binned image; zoom
//                       back to the original shape with the same mapping.
//   _anscombe_gauss  -> scale = percentile(image, 99.5) + 1e-9 (numpy linear
//                       interpolation), counts = clip(x/scale*30, 0, inf),
//                       stabilized = 2*sqrt(counts + 3/8), gaussian with
//                       max(1, sigma*0.85), inverse, * scale/30.

import { getHardwareGPUDevice, onGPULost } from "./device";

/** Filter modes the browser can evaluate; the denoise menus offer exactly these. */
const BROWSER_FILTER_MODES = new Set(["none", "gaussian", "anscombe"]);

/** Lowercase mode name; the identity spellings off/raw/"" collapse to none. */
export function normalizeFilterMode(mode: string): string {
  const normalized = String(mode ?? "").trim().toLowerCase();
  return normalized === "" || normalized === "off" || normalized === "raw" ? "none" : normalized;
}

export function browserFilterSupported(mode: string): boolean {
  return BROWSER_FILTER_MODES.has(normalizeFilterMode(mode));
}

/** True when the (mode, spatialBin) knobs change pixels at all. */
export function filterKnobsActive(mode: string, spatialBin: number): boolean {
  return normalizeFilterMode(mode) !== "none" || (spatialBin | 0) > 1;
}

/** Resolve the denoise editor/render knobs owned by one gallery panel. */
export function resolvePanelDenoiseKnobs(
  panel: number,
  modes: string[] | null | undefined,
  sigmas: number[] | null | undefined,
  bins: number[] | null | undefined,
  fallback: { mode: string; sigma: number; bin: number },
): { mode: string; sigma: number; bin: number } {
  const idx = Math.max(0, Math.round(panel));
  return {
    mode: modes && idx < modes.length ? modes[idx] : fallback.mode,
    sigma: Number(sigmas && idx < sigmas.length ? sigmas[idx] : fallback.sigma),
    bin: Number(bins && idx < bins.length ? bins[idx] : fallback.bin),
  };
}

/** scipy _gaussian_kernel1d(order=0): radius int(truncate*sigma + 0.5), normalized. */
function gaussianKernel1d(sigma: number, truncate = 4.0): Float32Array {
  const radius = Math.max(0, Math.trunc(truncate * sigma + 0.5));
  const weights = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const weight = sigma > 0 ? Math.exp(-0.5 * (i * i) / (sigma * sigma)) : (i === 0 ? 1 : 0);
    weights[i + radius] = weight;
    sum += weight;
  }
  const out = new Float32Array(weights.length);
  for (let i = 0; i < weights.length; i++) out[i] = weights[i] / sum;
  return out;
}

/** Python round() (half to even), used by ndimage.zoom for the output shape. */
function pythonRound(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** In-place Hoare quickselect: exact k-th order statistic, O(n) average. */
function selectKth(values: Float32Array, k: number): number {
  let lo = 0;
  let hi = values.length - 1;
  while (lo < hi) {
    const pivot = values[(lo + hi) >> 1];
    let i = lo;
    let j = hi;
    while (i <= j) {
      while (values[i] < pivot) i++;
      while (values[j] > pivot) j--;
      if (i <= j) { const swap = values[i]; values[i] = values[j]; values[j] = swap; i++; j--; }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  return values[k];
}

/**
 * numpy.percentile(..., method="linear"), but selection instead of a full
 * sort: the same value bit-for-bit, O(n) instead of O(n log n). At 2048^2 the
 * sort cost about 370 ms on the main thread and ran on every filter call, which
 * is what made the denoise knobs feel slow even with the GPU doing the blur.
 */
function percentileLinear(data: Float32Array, q: number): number {
  const n = data.length;
  if (n === 0) return 0;
  const scratch = Float32Array.from(data);
  const rank = (q / 100) * (n - 1);
  const lo = Math.floor(rank);
  const frac = rank - lo;
  const vlo = selectKth(scratch, lo);
  if (lo + 1 >= n) return vlo;
  // selectKth leaves everything > index lo in the right partition, so the next
  // order statistic is that partition's minimum.
  let vhi = Infinity;
  for (let i = lo + 1; i < n; i++) if (scratch[i] < vhi) vhi = scratch[i];
  return vlo + (vhi - vlo) * frac;
}

// scipy 'reflect' boundary: (d c b a | a b c d | d c b a), applied repeatedly.
function reflectIndex(i: number, n: number): number {
  if (n === 1) return 0;
  const period = 2 * n;
  let m = i % period;
  if (m < 0) m += period;
  return m < n ? m : period - 1 - m;
}

/** Reflected source offset (times stride) for every (position, tap) pair, so the blur loops skip the modulo. */
function reflectLookup(n: number, radius: number, stride: number): Int32Array {
  const out = new Int32Array(n * (2 * radius + 1));
  for (let i = 0; i < n; i++) {
    const base = i * (2 * radius + 1);
    for (let k = -radius; k <= radius; k++) {
      out[base + k + radius] = reflectIndex(i + k, n) * stride;
    }
  }
  return out;
}

/** Separable gaussian, rows axis first then cols axis, matching scipy's axis order. */
function gaussianBlurCPU(data: Float32Array, width: number, height: number, sigma: number): Float32Array {
  const kernel = gaussianKernel1d(sigma);
  const radius = (kernel.length - 1) / 2;
  if (radius === 0) return Float32Array.from(data);
  const mid = new Float32Array(data.length);
  const rowOffsets = reflectLookup(height, radius, width);
  const colOffsets = reflectLookup(width, radius, 1);
  const kernelSize = kernel.length;
  for (let row = 0; row < height; row++) {
    const lookupOffset = row * kernelSize;
    const outOffset = row * width;
    for (let col = 0; col < width; col++) {
      let acc = 0;
      for (let tap = 0; tap < kernelSize; tap++) {
        acc += kernel[tap] * data[rowOffsets[lookupOffset + tap] + col];
      }
      mid[outOffset + col] = acc;
    }
  }
  const out = new Float32Array(data.length);
  for (let row = 0; row < height; row++) {
    const rowOffset = row * width;
    for (let col = 0; col < width; col++) {
      let acc = 0;
      const lookupOffset = col * kernelSize;
      for (let tap = 0; tap < kernelSize; tap++) {
        acc += kernel[tap] * mid[rowOffset + colOffsets[lookupOffset + tap]];
      }
      out[rowOffset + col] = acc;
    }
  }
  return out;
}

/** Corner-aligned bilinear resample, the order-1 ndimage.zoom mapping. */
function resampleBilinearCPU(
  src: Float32Array, inW: number, inH: number, outW: number, outH: number,
): Float32Array {
  const out = new Float32Array(outW * outH);
  const scaleCol = outW > 1 ? (inW - 1) / (outW - 1) : 0;
  const scaleRow = outH > 1 ? (inH - 1) / (outH - 1) : 0;
  for (let row = 0; row < outH; row++) {
    const srcRow = row * scaleRow;
    const row0 = Math.min(Math.floor(srcRow), inH - 1);
    const row1 = Math.min(row0 + 1, inH - 1);
    const fr = srcRow - row0;
    for (let col = 0; col < outW; col++) {
      const srcCol = col * scaleCol;
      const col0 = Math.min(Math.floor(srcCol), inW - 1);
      const col1 = Math.min(col0 + 1, inW - 1);
      const fc = srcCol - col0;
      const top = src[row0 * inW + col0] * (1 - fc) + src[row0 * inW + col1] * fc;
      const bottom = src[row1 * inW + col0] * (1 - fc) + src[row1 * inW + col1] * fc;
      out[row * outW + col] = top * (1 - fr) + bottom * fr;
    }
  }
  return out;
}

type Plane = { data: Float32Array; width: number; height: number };

function bin2CPU(plane: Plane, sigma: number | null): Plane {
  const outW = pythonRound(plane.width * 0.5);
  const outH = pythonRound(plane.height * 0.5);
  let binned = resampleBilinearCPU(plane.data, plane.width, plane.height, outW, outH);
  if (sigma !== null) binned = gaussianBlurCPU(binned, outW, outH, Math.max(1.0, sigma / 2.5));
  const up = resampleBilinearCPU(binned, outW, outH, plane.width, plane.height);
  return { data: up, width: plane.width, height: plane.height };
}

function anscombeGaussCPU(plane: Plane, sigma: number): Plane {
  const { data, width, height } = plane;
  const scale = percentileLinear(data, 99.5) + 1e-9;
  const stabilized = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const counts = Math.max((data[i] / scale) * 30.0, 0);
    stabilized[i] = 2.0 * Math.sqrt(counts + 0.375);
  }
  const smoothed = gaussianBlurCPU(stabilized, width, height, Math.max(1.0, sigma * 0.85));
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const inverse = Math.max((smoothed[i] * 0.5) ** 2 - 0.375, 0);
    out[i] = (inverse * scale) / 30.0;
  }
  return { data: out, width, height };
}

/**
 * CPU reference for apply_display_filter(image, mode=mode, sigma=sigma, spatial_bin=spatialBin).
 * Output length equals width*height (bin passes zoom back to the input shape).
 */
export function applyDisplayFilterCPU(
  data: Float32Array, width: number, height: number,
  mode: string, sigma: number, spatialBin = 1,
): Float32Array {
  const resolved = { mode: normalizeFilterMode(mode), bin: spatialBin | 0 || 1 };
  let plane: Plane = { data: Float32Array.from(data), width, height };
  if (resolved.mode === "gaussian") {
    // Binned gaussian keeps the reference _bin2 semantics: smooth on the
    // binned grid with the lighter max(1, sigma/2.5) kernel, zoom back.
    if (resolved.bin === 4) return bin2CPU(bin2CPU(plane, null), sigma).data;
    if (resolved.bin === 2) return bin2CPU(plane, sigma).data;
    return gaussianBlurCPU(plane.data, plane.width, plane.height, sigma);
  }
  if (resolved.mode === "anscombe") {
    if (resolved.bin >= 2) plane = bin2CPU(plane, null);
    if (resolved.bin === 4) plane = bin2CPU(plane, null);
    const binnedSigma = resolved.bin >= 2 ? Math.max(2.0, sigma * 0.75) : sigma;
    return anscombeGaussCPU(plane, binnedSigma).data;
  }
  if (resolved.mode === "none") {
    if (resolved.bin >= 2) plane = bin2CPU(plane, null);
    if (resolved.bin === 4) plane = bin2CPU(plane, null);
    return plane.data;
  }
  throw new Error(`display filter mode not supported in the browser: ${mode}`);
}

// ---------------------------------------------------------------------------
// WGSL compute pipeline
// ---------------------------------------------------------------------------

const GAUSS_SHADER = /* wgsl */ `
struct Params { width: u32, height: u32, radius: u32, dir: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

fn reflect_index(i: i32, n: i32) -> i32 {
  if (n == 1) { return 0; }
  let period = 2 * n;
  var m = i % period;
  if (m < 0) { m = m + period; }
  if (m < n) { return m; }
  return period - 1 - m;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let radius = i32(p.radius);
  var acc = 0.0;
  if (p.dir == 0u) {
    // rows axis (vertical neighbourhood), scipy filters axis 0 first
    for (var k = -radius; k <= radius; k = k + 1) {
      let row = reflect_index(i32(gid.y) + k, i32(p.height));
      acc = acc + weights[k + radius] * src[u32(row) * p.width + gid.x];
    }
  } else {
    for (var k = -radius; k <= radius; k = k + 1) {
      let col = reflect_index(i32(gid.x) + k, i32(p.width));
      acc = acc + weights[k + radius] * src[gid.y * p.width + u32(col)];
    }
  }
  dst[gid.y * p.width + gid.x] = acc;
}
`;

const RESAMPLE_SHADER = /* wgsl */ `
struct Params { inW: u32, inH: u32, outW: u32, outH: u32, scaleCol: f32, scaleRow: f32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.outW || gid.y >= p.outH) { return; }
  let src_row = f32(gid.y) * p.scaleRow;
  let src_col = f32(gid.x) * p.scaleCol;
  let row0 = min(u32(floor(src_row)), p.inH - 1u);
  let col0 = min(u32(floor(src_col)), p.inW - 1u);
  let row1 = min(row0 + 1u, p.inH - 1u);
  let col1 = min(col0 + 1u, p.inW - 1u);
  let fr = src_row - f32(row0);
  let fc = src_col - f32(col0);
  let top = mix(src[row0 * p.inW + col0], src[row0 * p.inW + col1], fc);
  let bottom = mix(src[row1 * p.inW + col0], src[row1 * p.inW + col1], fc);
  dst[gid.y * p.outW + gid.x] = mix(top, bottom, fr);
}
`;

// mode 0: Anscombe forward 2*sqrt(clip(x/scale*30) + 3/8)
// mode 1: inverse clip((x/2)^2 - 3/8) * scale / 30
const ANSCOMBE_SHADER = /* wgsl */ `
struct Params { count: u32, mode: u32, scale: f32, _pad: f32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.count) { return; }
  let x = src[gid.x];
  if (p.mode == 0u) {
    let counts = max(x / p.scale * 30.0, 0.0);
    dst[gid.x] = 2.0 * sqrt(counts + 0.375);
  } else {
    let inverse = max((x * 0.5) * (x * 0.5) - 0.375, 0.0);
    dst[gid.x] = inverse * p.scale / 30.0;
  }
}
`;

/**
 * A float32 image living in a GPU storage buffer, flowing between stages.
 * Every stage is (GPUPlane in) -> (GPUPlane out) on the same device, so the
 * display filter composes by chaining stage calls:
 *
 *   raw frame -> uploadSourcePlane
 *             -> spatial stages (stageBin2, stageGaussian, stageAnscombeGauss)
 *             -> readPlane (CPU floats)
 */
type GPUPlane = { buffer: GPUBuffer; width: number; height: number };

type CachedSourcePlane = {
  source: Float32Array;
  plane: GPUPlane;
  generation: number;
};

/**
 * WGSL port of the display filter chain. One instance per page (shares the
 * engine's GPU device). Stage outputs are transient storage buffers tracked
 * per engine and destroyed by releasePlanes(), so idle cost is three pipelines.
 */
class GPUDisplayFilterEngine {
  /**
   * Source-frame cache. Dragging sigma over the same scientific frame should
   * not upload 16 MB again on every tick. The source Float32Array already owns
   * the frame identity in Show2D/Show3D, so keep its GPU storage buffer attached
   * and reuse it until a new frame object arrives. Cap the strong LRU so a
   * scientist scrubbing many Show3D frames does not quietly pin every frame on
   * the GPU.
   */
  private sourcePlaneCache = new WeakMap<Float32Array, CachedSourcePlane>();
  private sourcePlaneLru: CachedSourcePlane[] = [];
  private sourcePlaneGeneration = 0;
  private readonly maxSourcePlanes = 16;

  /**
   * Anscombe scale cache. The 99.5th percentile depends only on the frame and
   * the bin factor, never on sigma or mode, so dragging the sigma slider must
   * not recompute it. Keyed weakly on the source frame so freed frames drop out.
   */
  private scaleCache = new WeakMap<Float32Array, Map<number, number>>();

  private cachedScale(source: Float32Array, bin: number): number | undefined {
    return this.scaleCache.get(source)?.get(bin);
  }

  private storeScale(source: Float32Array, bin: number, scale: number): void {
    let perBin = this.scaleCache.get(source);
    if (!perBin) { perBin = new Map(); this.scaleCache.set(source, perBin); }
    perBin.set(bin, scale);
  }

  private device: GPUDevice;
  private gaussPipeline: GPUComputePipeline;
  private resamplePipeline: GPUComputePipeline;
  private anscombePipeline: GPUComputePipeline;
  private transient: GPUBuffer[] = [];
  private filterQueue: Promise<void> = Promise.resolve();

  constructor(device: GPUDevice) {
    this.device = device;
    this.gaussPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: GAUSS_SHADER }), entryPoint: "main" },
    });
    this.resamplePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: RESAMPLE_SHADER }), entryPoint: "main" },
    });
    this.anscombePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: ANSCOMBE_SHADER }), entryPoint: "main" },
    });
  }

  /** Transient float32 storage buffer for one stage output, destroyed by releasePlanes(). */
  private storage(count: number): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.transient.push(buffer);
    return buffer;
  }

  /** Transient uniform buffer holding one stage's params, destroyed by releasePlanes(). */
  private uniform(bytes: ArrayBuffer): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: Math.max(16, bytes.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, bytes);
    this.transient.push(buffer);
    return buffer;
  }

  private dispatch2d(pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[], width: number, height: number): void {
    const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Cached upload for source frames reused across live denoise knob drags. */
  uploadSourcePlane(data: Float32Array, width: number, height: number): GPUPlane {
    const cached = this.sourcePlaneCache.get(data);
    if (cached && cached.plane.width === width && cached.plane.height === height) {
      cached.generation = ++this.sourcePlaneGeneration;
      return cached.plane;
    }
    if (cached) {
      cached.plane.buffer.destroy();
      cached.generation = -1;
      this.sourcePlaneCache.delete(data);
    }
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    const plane = { buffer, width, height };
    const entry = { source: data, plane, generation: ++this.sourcePlaneGeneration };
    this.sourcePlaneCache.set(data, entry);
    this.sourcePlaneLru.push(entry);
    this.pruneSourcePlaneCache();
    return plane;
  }

  /** Destroy the least recently used source planes beyond maxSourcePlanes. */
  private pruneSourcePlaneCache(): void {
    this.sourcePlaneLru = this.sourcePlaneLru.filter((entry) => entry.generation >= 0);
    if (this.sourcePlaneLru.length <= this.maxSourcePlanes) return;
    this.sourcePlaneLru.sort((a, b) => a.generation - b.generation);
    while (this.sourcePlaneLru.length > this.maxSourcePlanes) {
      const evicted = this.sourcePlaneLru.shift();
      if (!evicted || evicted.generation < 0) continue;
      if (this.sourcePlaneCache.get(evicted.source) === evicted) {
        this.sourcePlaneCache.delete(evicted.source);
      }
      evicted.generation = -1;
      evicted.plane.buffer.destroy();
    }
  }

  /** Destroy every transient stage buffer created since the last release. */
  releasePlanes(): void {
    for (const buffer of this.transient) buffer.destroy();
    this.transient.length = 0;
  }

  /** Separable gaussian stage: two 1D passes, scipy reflect boundary. */
  stageGaussian(plane: GPUPlane, sigma: number): GPUPlane {
    const kernel = gaussianKernel1d(sigma);
    const radius = (kernel.length - 1) / 2;
    if (radius === 0) return plane;
    const weights = this.device.createBuffer({
      size: kernel.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.transient.push(weights);
    this.device.queue.writeBuffer(weights, 0, kernel.buffer as ArrayBuffer, kernel.byteOffset, kernel.byteLength);
    const mid = this.storage(plane.width * plane.height);
    const out = this.storage(plane.width * plane.height);
    for (const [dir, src, dst] of [[0, plane.buffer, mid], [1, mid, out]] as const) {
      const params = this.uniform(new Uint32Array([plane.width, plane.height, radius, dir]).buffer);
      this.dispatch2d(this.gaussPipeline, [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: weights } },
        { binding: 3, resource: { buffer: params } },
      ], plane.width, plane.height);
    }
    return { buffer: out, width: plane.width, height: plane.height };
  }

  /** Corner-aligned bilinear resample stage (ndimage.zoom order=1 mapping). */
  stageResample(plane: GPUPlane, outW: number, outH: number): GPUPlane {
    const out = this.storage(outW * outH);
    const raw = new ArrayBuffer(24);
    const paramsU32 = new Uint32Array(raw);
    const paramsF32 = new Float32Array(raw);
    paramsU32[0] = plane.width; paramsU32[1] = plane.height; paramsU32[2] = outW; paramsU32[3] = outH;
    paramsF32[4] = outW > 1 ? (plane.width - 1) / (outW - 1) : 0;
    paramsF32[5] = outH > 1 ? (plane.height - 1) / (outH - 1) : 0;
    const params = this.uniform(raw);
    this.dispatch2d(this.resamplePipeline, [
      { binding: 0, resource: { buffer: plane.buffer } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: params } },
    ], outW, outH);
    return { buffer: out, width: outW, height: outH };
  }

  /** Pointwise Anscombe pass: mode 0 is the forward transform, mode 1 its inverse. */
  private anscombePointwise(plane: GPUPlane, mode: 0 | 1, scale: number): GPUPlane {
    const count = plane.width * plane.height;
    const out = this.storage(count);
    const raw = new ArrayBuffer(16);
    const paramsU32 = new Uint32Array(raw);
    paramsU32[0] = count;
    paramsU32[1] = mode;
    new Float32Array(raw)[2] = scale;
    const params = this.uniform(raw);
    const bindGroup = this.device.createBindGroup({
      layout: this.anscombePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: plane.buffer } },
        { binding: 1, resource: { buffer: out } },
        { binding: 2, resource: { buffer: params } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.anscombePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return { buffer: out, width: plane.width, height: plane.height };
  }

  /** Read a stage output back to CPU float32 (does not release the chain). */
  async readPlane(plane: GPUPlane): Promise<Float32Array> {
    const count = plane.width * plane.height;
    const staging = this.device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.transient.push(staging);
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(plane.buffer, 0, staging, 0, count * 4);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    return out;
  }

  /** _bin2 stage: 0.5x bilinear down, optional smooth, bilinear back up. */
  stageBin2(plane: GPUPlane, sigma: number | null): GPUPlane {
    const outW = pythonRound(plane.width * 0.5);
    const outH = pythonRound(plane.height * 0.5);
    let binned = this.stageResample(plane, outW, outH);
    if (sigma !== null) binned = this.stageGaussian(binned, Math.max(1.0, sigma / 2.5));
    return this.stageResample(binned, plane.width, plane.height);
  }

  /** Anscombe stage: variance stabilize, gaussian(max(1, sigma*0.85)), inverse. */
  async stageAnscombeGauss(
    plane: GPUPlane, sigma: number, cpuShadow: Float32Array | null = null,
    cacheKey: { source: Float32Array; bin: number } | null = null,
  ): Promise<GPUPlane> {
    // The Anscombe scale is the 99.5th percentile of THIS stage's input. It is
    // independent of sigma, so a cache hit makes sigma edits pure GPU work.
    let scale = cacheKey ? this.cachedScale(cacheKey.source, cacheKey.bin) : undefined;
    if (scale === undefined) {
      const values = cpuShadow ?? await this.readPlane(plane);
      scale = percentileLinear(values, 99.5) + 1e-9;
      if (cacheKey) this.storeScale(cacheKey.source, cacheKey.bin, scale);
    }
    const stabilized = this.anscombePointwise(plane, 0, scale);
    const smoothed = this.stageGaussian(stabilized, Math.max(1.0, sigma * 0.85));
    return this.anscombePointwise(smoothed, 1, scale);
  }

  /** GPU apply_display_filter; resolves to a fresh Float32Array of width*height. */
  async filter(
    data: Float32Array, width: number, height: number,
    mode: string, sigma: number, spatialBin = 1,
  ): Promise<Float32Array> {
    // The engine reuses a shared transient buffer list for stage outputs. A
    // real Show2D/Show3D widget may ask for another filter while an earlier
    // readback is still mapping; serialize calls so one cleanup cannot destroy
    // another call's staging buffer and force a slow CPU fallback.
    const previous = this.filterQueue;
    let releaseQueue: () => void = () => {};
    this.filterQueue = new Promise<void>((resolve) => { releaseQueue = resolve; });
    await previous;
    const resolved = { mode: normalizeFilterMode(mode), bin: spatialBin | 0 || 1 };
    if (!BROWSER_FILTER_MODES.has(resolved.mode)) {
      releaseQueue();
      throw new Error(`display filter mode not supported in the browser: ${mode}`);
    }
    try {
      let plane: GPUPlane = this.uploadSourcePlane(data, width, height);
      if (resolved.mode === "gaussian") {
        // Binned gaussian = the reference _bin2 semantics (light smooth on
        // the binned grid); plain separable gaussian otherwise.
        if (resolved.bin === 4) plane = this.stageBin2(this.stageBin2(plane, null), sigma);
        else if (resolved.bin === 2) plane = this.stageBin2(plane, sigma);
        else plane = this.stageGaussian(plane, sigma);
      } else if (resolved.mode === "anscombe") {
        // cpuShadow: at bin 1 the plane still equals the raw input, so the
        // percentile skips a GPU readback.
        let cpuShadow: Float32Array | null = data;
        if (resolved.bin >= 2) { plane = this.stageBin2(plane, null); cpuShadow = null; }
        if (resolved.bin === 4) { plane = this.stageBin2(plane, null); }
        const binnedSigma = resolved.bin >= 2 ? Math.max(2.0, sigma * 0.75) : sigma;
        plane = await this.stageAnscombeGauss(plane, binnedSigma, cpuShadow, { source: data, bin: resolved.bin });
      } else if (resolved.bin >= 2) {
        plane = this.stageBin2(plane, null);
        if (resolved.bin === 4) plane = this.stageBin2(plane, null);
      }
      return await this.readPlane(plane);
    } finally {
      this.releasePlanes();
      releaseQueue();
    }
  }
}

let enginePromise: Promise<GPUDisplayFilterEngine | null> | null = null;
onGPULost(() => { enginePromise = null; });

/** Singleton GPU filter engine on the shared hardware device, or null without one. */
function getGPUDisplayFilterEngine(): Promise<GPUDisplayFilterEngine | null> {
  if (!enginePromise) {
    enginePromise = getHardwareGPUDevice().then((device) => device ? new GPUDisplayFilterEngine(device) : null);
  }
  return enginePromise;
}

/**
 * Filter one frame in the browser: WebGPU when a hardware adapter exists,
 * otherwise the scalar reference (same weights, coordinates and percentiles).
 */
export async function applyDisplayFilterBrowser(
  data: Float32Array, width: number, height: number,
  mode: string, sigma: number, spatialBin = 1,
): Promise<Float32Array> {
  const engine = await getGPUDisplayFilterEngine();
  if (!engine) return applyDisplayFilterCPU(data, width, height, mode, sigma, spatialBin);
  return engine.filter(data, width, height, mode, sigma, spatialBin);
}
