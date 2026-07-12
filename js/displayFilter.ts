/// <reference types="@webgpu/types" />
// Browser-side port of the display-only filter pipeline in
// src/quantem/widget/utils/display_filter.py (apply_display_filter).
//
// Covered modes: none, gaussian, bin2, anscombe, bin2_anscombe, bin4_anscombe,
// plus the spatial_bin pre-passes. tv and denova* need scikit-image / the
// denova package and stay on the Python path (the kernel keeps filtering those
// panels; see Show2D._panel_browser_filtered).
//
// Two implementations share the same kernel weights, resample coordinates and
// percentile math:
//   - applyDisplayFilterCPU: scalar TypeScript reference, tested against the
//     Python/scipy implementation in displayFilter.numpy.test.ts. Also the
//     fallback for kernel-less offline HTML pages on browsers without WebGPU.
//   - GPUDisplayFilterEngine: WGSL compute passes (separable gaussian,
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

import { getGPUDevice, getGPUInfo, isSoftwareGPUAdapter } from "./engine/device";

/** Filter modes the browser can evaluate; mirrors BROWSER_DISPLAY_FILTER_MODES in Python. */
export const BROWSER_FILTER_MODES = new Set([
  "none", "gaussian", "bin2", "anscombe", "bin2_anscombe", "bin4_anscombe",
]);

/** Mirror of display_filter._normalize_mode for the mode spellings the UI can produce. */
export function normalizeFilterMode(mode: string): string {
  const m = String(mode ?? "").trim().toLowerCase().replace(/-/g, "_");
  if (m === "" || m === "none" || m === "off" || m === "raw") return "none";
  const aliases: Record<string, string> = {
    bin2anscombe: "bin2_anscombe",
    bin_anscombe: "bin2_anscombe",
    bin4anscombe: "bin4_anscombe",
    poisson: "anscombe",
    anscombe_gaussian: "anscombe",
    denova_tv1_2: "denova_tv12",
  };
  return aliases[m] ?? m;
}

export function browserFilterSupported(mode: string): boolean {
  return BROWSER_FILTER_MODES.has(normalizeFilterMode(mode));
}

/**
 * Mirror of display_filter.resolve_denoise_mode: the canonical menu is three
 * orthogonal methods (none/gaussian/anscombe) with binning as its own knob;
 * compound spellings fold into (mode, bin).
 */
export function resolveDenoiseMode(mode: string, spatialBin = 1): { mode: string; bin: number } {
  const normalized = normalizeFilterMode(mode);
  const bin = spatialBin | 0 || 1;
  if (normalized === "bin2") return { mode: "gaussian", bin: Math.max(bin, 2) };
  if (normalized === "bin2_anscombe") return { mode: "anscombe", bin: Math.max(bin, 2) };
  if (normalized === "bin4_anscombe") return { mode: "anscombe", bin: Math.max(bin, 4) };
  return { mode: normalized, bin };
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
export function gaussianKernel1d(sigma: number, truncate = 4.0): Float32Array {
  const radius = Math.max(0, Math.trunc(truncate * sigma + 0.5));
  const weights = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const w = sigma > 0 ? Math.exp(-0.5 * (i * i) / (sigma * sigma)) : (i === 0 ? 1 : 0);
    weights[i + radius] = w;
    sum += w;
  }
  const out = new Float32Array(weights.length);
  for (let i = 0; i < weights.length; i++) out[i] = weights[i] / sum;
  return out;
}

/** Python round() (half to even), used by ndimage.zoom for the output shape. */
export function pythonRound(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** numpy.percentile(data, q) with the default linear interpolation. */
export function percentileLinear(data: Float32Array, q: number): number {
  const n = data.length;
  if (n === 0) return 0;
  const sorted = Float32Array.from(data).sort();
  const rank = (q / 100) * (n - 1);
  const lo = Math.floor(rank);
  const frac = rank - lo;
  if (lo + 1 >= n) return sorted[n - 1];
  return sorted[lo] + (sorted[lo + 1] - sorted[lo]) * frac;
}

// scipy 'reflect' boundary: (d c b a | a b c d | d c b a), applied repeatedly.
function reflectIndex(i: number, n: number): number {
  if (n === 1) return 0;
  const period = 2 * n;
  let m = i % period;
  if (m < 0) m += period;
  return m < n ? m : period - 1 - m;
}

/** Separable gaussian, rows axis first then cols axis, matching scipy's axis order. */
export function gaussianBlurCPU(data: Float32Array, width: number, height: number, sigma: number): Float32Array {
  const kernel = gaussianKernel1d(sigma);
  const radius = (kernel.length - 1) / 2;
  if (radius === 0) return Float32Array.from(data);
  const mid = new Float32Array(data.length);
  for (let col = 0; col < width; col++) {
    for (let row = 0; row < height; row++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        acc += kernel[k + radius] * data[reflectIndex(row + k, height) * width + col];
      }
      mid[row * width + col] = acc;
    }
  }
  const out = new Float32Array(data.length);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        acc += kernel[k + radius] * mid[row * width + reflectIndex(col + k, width)];
      }
      out[row * width + col] = acc;
    }
  }
  return out;
}

/** Corner-aligned bilinear resample, the order-1 ndimage.zoom mapping. */
export function resampleBilinearCPU(
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
 * CPU reference for apply_display_filter(image, filter=mode, sigma, spatial_bin).
 * Output length equals width*height (bin passes zoom back to the input shape).
 */
export function applyDisplayFilterCPU(
  data: Float32Array, width: number, height: number,
  mode: string, sigma: number, spatialBin = 1,
): Float32Array {
  const resolved = resolveDenoiseMode(mode, spatialBin);
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

// Pointwise display stage: percentile stretch + gamma in one pass.
// out = pow(clamp((x - lo) / (hi - lo), 0, 1), gamma). With gamma=1 this is a
// plain percentile stretch; lo/hi come from percentileLinear on the CPU side.
const STRETCH_GAMMA_SHADER = /* wgsl */ `
struct Params { count: u32, _pad: u32, lo: f32, inv_range: f32, gamma: f32, _pad2: f32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.count) { return; }
  let t = clamp((src[gid.x] - p.lo) * p.inv_range, 0.0, 1.0);
  dst[gid.x] = pow(t, p.gamma);
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
 * Every stage is (GPUPlane in) -> (GPUPlane out) on the same device, so
 * display pipelines compose by chaining stage calls:
 *
 *   raw frame -> uploadPlane
 *             -> spatial stages   (stageBin2, stageGaussian, stageAnscombeGauss)
 *             -> pointwise stages (stageStretchGamma; future: two-input
 *                HAADF blend / dual-channel composite stages that take a
 *                second GPUPlane plus alpha/gain uniforms and emit RGB)
 *             -> readPlane (CPU floats) or hand the buffer to
 *                GPUColormapEngine's LUT pass for direct canvas paint.
 *
 * A blend stage plugs in exactly here: it is one more pointwise WGSL entry
 * with `@binding` slots for two source planes, no restructuring needed.
 */
export type GPUPlane = { buffer: GPUBuffer; width: number; height: number };

/**
 * WGSL port of the display filter chain. One instance per page (shares the
 * engine's GPU device). Stage outputs are transient storage buffers tracked
 * per engine and destroyed by releasePlanes(), so idle cost is four pipelines.
 */
export class GPUDisplayFilterEngine {
  private device: GPUDevice;
  private gaussPipeline: GPUComputePipeline;
  private resamplePipeline: GPUComputePipeline;
  private anscombePipeline: GPUComputePipeline;
  private stretchPipeline: GPUComputePipeline;
  private transient: GPUBuffer[] = [];

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
    this.stretchPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: STRETCH_GAMMA_SHADER }), entryPoint: "main" },
    });
  }

  private storage(count: number): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.transient.push(buffer);
    return buffer;
  }

  private uniform(bytes: ArrayBuffer): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: Math.max(16, bytes.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, bytes);
    this.transient.push(buffer);
    return buffer;
  }

  private dispatch2d(pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[], w: number, h: number): void {
    const bind = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Upload a float32 frame as the head plane of a stage chain. */
  uploadPlane(data: Float32Array, width: number, height: number): GPUPlane {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.transient.push(buffer);
    this.device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    return { buffer, width, height };
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
    const u32 = new Uint32Array(raw);
    const f32 = new Float32Array(raw);
    u32[0] = plane.width; u32[1] = plane.height; u32[2] = outW; u32[3] = outH;
    f32[4] = outW > 1 ? (plane.width - 1) / (outW - 1) : 0;
    f32[5] = outH > 1 ? (plane.height - 1) / (outH - 1) : 0;
    const params = this.uniform(raw);
    this.dispatch2d(this.resamplePipeline, [
      { binding: 0, resource: { buffer: plane.buffer } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: params } },
    ], outW, outH);
    return { buffer: out, width: outW, height: outH };
  }

  private anscombePointwise(plane: GPUPlane, mode: 0 | 1, scale: number): GPUPlane {
    const count = plane.width * plane.height;
    const out = this.storage(count);
    const raw = new ArrayBuffer(16);
    new Uint32Array(raw)[0] = count;
    new Uint32Array(raw)[1] = mode;
    new Float32Array(raw)[2] = scale;
    const params = this.uniform(raw);
    const bind = this.device.createBindGroup({
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
    pass.setBindGroup(0, bind);
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

  /**
   * Pointwise percentile-stretch + gamma stage:
   * out = pow(clamp((x - lo) / (hi - lo), 0, 1), gamma). Compute lo/hi with
   * percentileLinear on the CPU frame (the same reduction the Anscombe scale
   * uses). Slot it after the denoise stages and before the colormap LUT.
   */
  stageStretchGamma(plane: GPUPlane, lo: number, hi: number, gamma = 1.0): GPUPlane {
    const count = plane.width * plane.height;
    const out = this.storage(count);
    const raw = new ArrayBuffer(24);
    const u32 = new Uint32Array(raw);
    const f32 = new Float32Array(raw);
    u32[0] = count;
    f32[2] = lo;
    f32[3] = 1.0 / Math.max(hi - lo, 1e-30);
    f32[4] = gamma;
    const params = this.uniform(raw);
    const bind = this.device.createBindGroup({
      layout: this.stretchPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: plane.buffer } },
        { binding: 1, resource: { buffer: out } },
        { binding: 2, resource: { buffer: params } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.stretchPipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(count / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return { buffer: out, width: plane.width, height: plane.height };
  }

  /** Anscombe stage: variance stabilize, gaussian(max(1, sigma*0.85)), inverse. */
  async stageAnscombeGauss(plane: GPUPlane, sigma: number, cpuShadow: Float32Array | null = null): Promise<GPUPlane> {
    // The Anscombe scale is the 99.5th percentile of THIS stage's input; a
    // sort-based percentile on the GPU buys nothing at these sizes, so read
    // the (bin-reduced) plane back when a bin pass already ran on the GPU.
    const values = cpuShadow ?? await this.readPlane(plane);
    const scale = percentileLinear(values, 99.5) + 1e-9;
    const stabilized = this.anscombePointwise(plane, 0, scale);
    const smoothed = this.stageGaussian(stabilized, Math.max(1.0, sigma * 0.85));
    return this.anscombePointwise(smoothed, 1, scale);
  }

  /** GPU apply_display_filter; resolves to a fresh Float32Array of width*height. */
  async filter(
    data: Float32Array, width: number, height: number,
    mode: string, sigma: number, spatialBin = 1,
  ): Promise<Float32Array> {
    const resolved = resolveDenoiseMode(mode, spatialBin);
    if (!BROWSER_FILTER_MODES.has(resolved.mode)) {
      throw new Error(`display filter mode not supported in the browser: ${mode}`);
    }
    try {
      let plane: GPUPlane = this.uploadPlane(data, width, height);
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
        plane = await this.stageAnscombeGauss(plane, binnedSigma, cpuShadow);
      } else if (resolved.bin >= 2) {
        plane = this.stageBin2(plane, null);
        if (resolved.bin === 4) plane = this.stageBin2(plane, null);
      }
      // stageBin2 zooms back to the input shape, so this is a safety net only.
      if (plane.width !== width || plane.height !== height) {
        plane = this.stageResample(plane, width, height);
      }
      return await this.readPlane(plane);
    } finally {
      this.releasePlanes();
    }
  }
}

let enginePromise: Promise<GPUDisplayFilterEngine | null> | null = null;

/**
 * Singleton GPU filter engine on the shared engine device. Resolves null when
 * WebGPU is missing OR the adapter is a software rasterizer (SwiftShader /
 * llvmpipe): a CPU-fallback "GPU" is slower than Python's scipy path, so the
 * kernel keeps filtering in that case.
 */
export function getGPUDisplayFilterEngine(): Promise<GPUDisplayFilterEngine | null> {
  if (!enginePromise) {
    enginePromise = getGPUDevice().then((device) => {
      if (!device) return null;
      if (isSoftwareGPUAdapter()) {
        console.warn(`[Show2D] display filter: software WebGPU adapter (${getGPUInfo()}); leaving filtering to the kernel/CPU`);
        return null;
      }
      console.log(`[Show2D] WebGPU display filter engine ready - adapter: ${getGPUInfo()}`);
      return new GPUDisplayFilterEngine(device);
    }).catch(() => null);
  }
  return enginePromise;
}

/**
 * Filter one frame in the browser: WGSL compute when a real GPU adapter is
 * available, CPU TypeScript otherwise (kernel-less offline pages must still
 * honor the knobs). Both paths share the same math; parity is tested in
 * displayFilter.numpy.test.ts.
 */
export async function applyDisplayFilterBrowser(
  data: Float32Array, width: number, height: number,
  mode: string, sigma: number, spatialBin = 1,
): Promise<Float32Array> {
  const engine = await getGPUDisplayFilterEngine();
  if (engine) {
    try {
      return await engine.filter(data, width, height, mode, sigma, spatialBin);
    } catch (err) {
      console.warn("[Show2D] GPU display filter failed; falling back to CPU", err);
    }
  }
  return applyDisplayFilterCPU(data, width, height, mode, sigma, spatialBin);
}
