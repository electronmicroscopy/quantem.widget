// The display work of one reconstructed phase on the GPU it is drawn with: the
// percentile histograms behind auto contrast and the histogram widgets, and the
// Fourier panel (mean removed, Hann window, zero pad, 2D FFT, shifted log
// magnitude). The phase is uploaded once into its colormap slot; the spectrum is
// written straight into the Fourier slot, and only the two 1024-bin histograms
// (8 KB) come back to JavaScript. Without it every frame copied the phase to a
// statistics worker, windowed and padded it in a second worker, uploaded and read
// back a 2048 x 2048 complex grid and took its magnitude in a third pass on the
// CPU: about 200 ms per 1536 x 1536 frame.
import type { GPUColormapEngine } from "../display/colormapEngine";
import type { FrameStats } from "../display/displayStats";
import type { GridRegion } from "./fftFraming";
import { nextPow2, type WebGPUFFT } from "../display/fft";
import { computePipeline, renderPipeline } from "../display/pipelines";
import { DISPLAY_NORMALIZE_WGSL } from "../display/displayShaders";

const NUM_BINS = 1024;
const DISPLAY_BINS = 256;
const REDUCE_GROUPS = 1024;
// One stats record: min, max, sum, count (4 floats), then 1024 u32 bins, padded to the
// 256-byte storage binding alignment so the second record can be bound on its own.
export const RECORD_BYTES = Math.ceil((16 + NUM_BINS * 4) / 256) * 256;

const REDUCE_WGSL = /* wgsl */ `
struct Params { count: u32, groups: u32, row_width: u32, stride: u32, x0: u32, y0: u32, _a: u32, _b: u32 };
// The index-th value of the block (x0, y0, row_width wide) of a grid stride wide.
fn at(index: u32) -> u32 {
  let row = index / params.row_width;
  return (params.y0 + row) * params.stride + params.x0 + index - row * params.row_width;
}
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> partials: array<vec4<f32>>;
var<workgroup> lo: array<f32, 256>;
var<workgroup> hi: array<f32, 256>;
var<workgroup> total: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  var vmin = 3.4e38;
  var vmax = -3.4e38;
  var sum = 0.0;
  for (var index = group.x * 256u + lane; index < params.count; index += params.groups * 256u) {
    let value = data[at(index)];
    vmin = min(vmin, value);
    vmax = max(vmax, value);
    sum += value;
  }
  lo[lane] = vmin; hi[lane] = vmax; total[lane] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      lo[lane] = min(lo[lane], lo[lane + stride]);
      hi[lane] = max(hi[lane], hi[lane + stride]);
      total[lane] += total[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) { partials[group.x] = vec4<f32>(lo[0], hi[0], total[0], 0.0); }
}
`;

// Folds the partials into the record head: (min, max, sum, count).
const FINISH_WGSL = /* wgsl */ `
struct Params { count: u32, groups: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> partials: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> head: array<f32>;
var<workgroup> lo: array<f32, 256>;
var<workgroup> hi: array<f32, 256>;
var<workgroup> total: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane: u32) {
  var vmin = 3.4e38;
  var vmax = -3.4e38;
  var sum = 0.0;
  for (var index = lane; index < params.groups; index += 256u) {
    vmin = min(vmin, partials[index].x);
    vmax = max(vmax, partials[index].y);
    sum += partials[index].z;
  }
  lo[lane] = vmin; hi[lane] = vmax; total[lane] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      lo[lane] = min(lo[lane], lo[lane + stride]);
      hi[lane] = max(hi[lane], hi[lane + stride]);
      total[lane] += total[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) { head[0] = lo[0]; head[1] = hi[0]; head[2] = total[0]; head[3] = f32(params.count); }
}
`;

// percentileHistogram's binning: bin i spans [min + i*range/1024, min + (i+1)*range/1024),
// max lands in the last bin. Bins are counted in workgroup memory, then added once per group.
const HISTOGRAM_WGSL = /* wgsl */ `
struct Params { count: u32, groups: u32, row_width: u32, stride: u32, x0: u32, y0: u32, _a: u32, _b: u32 };
// The index-th value of the block (x0, y0, row_width wide) of a grid stride wide.
fn at(index: u32) -> u32 {
  let row = index / params.row_width;
  return (params.y0 + row) * params.stride + params.x0 + index - row * params.row_width;
}
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> record: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> head: array<f32>;
var<workgroup> local_bins: array<atomic<u32>, ${NUM_BINS}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  for (var bin = lane; bin < ${NUM_BINS}u; bin += 256u) { atomicStore(&local_bins[bin], 0u); }
  workgroupBarrier();
  let vmin = head[0];
  let range = head[1] - head[0];
  if (range > 0.0) {
    let scale = ${NUM_BINS}.0 / range;
    for (var index = group.x * 256u + lane; index < params.count; index += params.groups * 256u) {
      let bin = min(${NUM_BINS - 1}u, u32(floor((data[at(index)] - vmin) * scale)));
      atomicAdd(&local_bins[bin], 1u);
    }
  }
  workgroupBarrier();
  for (var bin = lane; bin < ${NUM_BINS}u; bin += 256u) {
    let count = atomicLoad(&local_bins[bin]);
    if (count > 0u) { atomicAdd(&record[4u + bin], count); }
  }
}
`;

// displayFFTInput: subtract the mean, Hann-window (np.hanning) and zero-pad to pw x ph,
// interleaved (real, 0) as the FFT reads it.
const PREPARE_WGSL = /* wgsl */ `
struct Params { width: u32, height: u32, pw: u32, ph: u32 };
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> complex: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> head: array<f32>;
const TWO_PI = 6.283185307179586;
fn hann(index: u32, size: u32) -> f32 {
  let denom = f32(max(size, 2u) - 1u);
  return 0.5 * (1.0 - cos(TWO_PI * f32(index) / denom));
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let col = id.x; let row = id.y;
  if (col >= params.pw || row >= params.ph) { return; }
  var value = 0.0;
  if (col < params.width && row < params.height) {
    let mean = head[2] / head[3];
    value = (data[row * params.width + col] - mean) * (hann(row, params.height) * hann(col, params.width));
  }
  complex[row * params.pw + col] = vec2<f32>(value, 0.0);
}
`;

// shiftedMagnitude(logScale): DC to the centre, log1p(|F|). log1p keeps its relative precision for
// small magnitudes with the u = 1 + x correction.
const MAGNITUDE_WGSL = /* wgsl */ `
struct Params { width: u32, height: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> spectrum: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> magnitude: array<f32>;
fn log1p(x: f32) -> f32 {
  let u = 1.0 + x;
  if (u == 1.0) { return x; }
  return log(u) * (x / (u - 1.0));
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let col = id.x; let row = id.y;
  if (col >= params.width || row >= params.height) { return; }
  let value = spectrum[row * params.width + col];
  let out_row = (row + params.height / 2u) % params.height;
  let out_col = (col + params.width / 2u) % params.width;
  magnitude[out_row * params.width + out_col] = log1p(sqrt(value.x * value.x + value.y * value.y));
}
`;

/** FrameStats from one read-back record, laid out as percentileHistogram and computeHistogramFromBytes give them. */
export function recordStats(record: ArrayBuffer, offset: number): FrameStats {
  const head = new Float32Array(record, offset, 4);
  const min = head[0], max = head[1], finiteCount = head[3];
  const counts = new Uint32Array(record, offset + 16, NUM_BINS).slice();
  // computeHistogramFromBytes' 256 display bins are the 1024 percentile bins taken four at a time.
  const displayBins = new Array(DISPLAY_BINS).fill(0);
  if (!(max > min)) {
    if (finiteCount > 0) displayBins[DISPLAY_BINS >> 1] = 1;
    return { histogram: { min, max, finiteCount, bins: null }, displayBins };
  }
  const group = NUM_BINS / DISPLAY_BINS;
  for (let bin = 0; bin < NUM_BINS; bin++) displayBins[Math.floor(bin / group)] += counts[bin];
  const peak = Math.max(...displayBins);
  if (peak > 0) for (let bin = 0; bin < DISPLAY_BINS; bin++) displayBins[bin] /= peak;
  return { histogram: { min, max, finiteCount, bins: counts }, displayBins };
}

/** Map one read-back record and decode its statistics. */
async function readRecord(buffer: GPUBuffer): Promise<FrameStats> {
  await buffer.mapAsync(GPUMapMode.READ);
  const stats = recordStats(buffer.getMappedRange().slice(0), 0);
  buffer.unmap();
  return stats;
}

export type GPUFrameResult = { phase: FrameStats; fft: { width: number; height: number; stats: FrameStats } | null };

/** Display statistics and the Fourier panel of the phase in colormap slot `phaseSlot`, on the engine's device. */
export class GPUFrameDisplay {
  private readonly engine: GPUColormapEngine;
  private readonly fft: WebGPUFFT;
  private readonly device: GPUDevice;
  private readonly reduce: GPUComputePipeline;
  private readonly finish: GPUComputePipeline;
  private readonly histogram: GPUComputePipeline;
  private readonly prepare: GPUComputePipeline;
  private readonly magnitude: GPUComputePipeline;
  private readonly partials: GPUBuffer;
  private readonly records: GPUBuffer;
  private complex: GPUBuffer | null = null;

  constructor(engine: GPUColormapEngine, fft: WebGPUFFT) {
    this.engine = engine;
    this.fft = fft;
    this.device = engine.getDevice();
    this.reduce = computePipeline(this.device, REDUCE_WGSL);
    this.finish = computePipeline(this.device, FINISH_WGSL);
    this.histogram = computePipeline(this.device, HISTOGRAM_WGSL);
    this.prepare = computePipeline(this.device, PREPARE_WGSL);
    this.magnitude = computePipeline(this.device, MAGNITUDE_WGSL);
    this.partials = this.device.createBuffer({ size: REDUCE_GROUPS * 16, usage: GPUBufferUsage.STORAGE });
    this.records = this.device.createBuffer({ size: 2 * RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  /** True when the FFT runs on the engine's own device, so its buffers can be shared. */
  static compatible(engine: GPUColormapEngine, fft: { path: string; device?: GPUDevice }): boolean {
    return fft.path === "WebGPU" && (fft as unknown as { device: GPUDevice }).device === engine.getDevice();
  }

  private uniform(values: number[]): GPUBuffer {
    const buffer = this.device.createBuffer({ size: values.length * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, new Uint32Array(values));
    return buffer;
  }

  private dispatch(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, buffers: GPUBuffer[], x: number, y = 1, offsets: number[] = []) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer, offset: offsets[binding] ?? 0 } })),
    }));
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }

  /** Range, sum and 1024-bin histogram of `data` (count floats, or `region` of a grid `stride`
   *  wide) into record `index`; returns the (min, max, sum, count) head for later passes of the same frame. */
  private encodeStats(encoder: GPUCommandEncoder, data: GPUBuffer, count: number, index: number, temporaries: GPUBuffer[],
    region: GridRegion | null = null, stride = count): GPUBuffer {
    if (region) count = region.width * region.height;
    const groups = Math.min(REDUCE_GROUPS, Math.max(1, Math.ceil(count / 256)));
    const params = this.uniform(region
      ? [count, groups, region.width, stride, region.x0, region.y0, 0, 0]
      : [count, groups, count, count, 0, 0, 0, 0]);
    temporaries.push(params);
    const offset = index * RECORD_BYTES;
    encoder.clearBuffer(this.records, offset, RECORD_BYTES);
    this.dispatch(encoder, this.reduce, [data, params, this.partials], groups);
    const head = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });  // phase mean for the FFT input
    temporaries.push(head);
    this.dispatch(encoder, this.finish, [this.partials, params, head], 1);
    this.dispatch(encoder, this.histogram, [data, params, this.records, head], groups, 1, [0, 0, offset, 0]);
    encoder.copyBufferToBuffer(head, 0, this.records, offset, 16);
    return head;
  }

  /** The Fourier slot's dimensions for a width x height phase: the next powers of two. */
  static fftShape(width: number, height: number): [number, number] { return [nextPow2(width), nextPow2(height)]; }

  /**
   * Statistics of slot `phaseSlot` (a width x height phase already uploaded) and, when `fftSlot`
   * is given, its display spectrum written into that slot (already sized pw x ph).
   */
  async process(phaseSlot: number, width: number, height: number, fftSlot: number | null,
    onPhase?: (stats: FrameStats) => void, fftRegion: GridRegion | null = null): Promise<GPUFrameResult> {
    const phase = this.engine.slots[phaseSlot];
    const temporaries: GPUBuffer[] = [];
    const encoder = this.device.createCommandEncoder();
    const head = this.encodeStats(encoder, phase.dataBuffer, width * height, 0, temporaries);
    // The phase panel paints from its own record as soon as it lands, before the spectrum is done.
    const phaseReadback = this.device.createBuffer({ size: RECORD_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    temporaries.push(phaseReadback);
    encoder.copyBufferToBuffer(this.records, 0, phaseReadback, 0, RECORD_BYTES);
    let shape: [number, number] | null = null;
    if (fftSlot !== null) {
      shape = GPUFrameDisplay.fftShape(width, height);
      const [pw, ph] = shape;
      const bytes = pw * ph * 8;
      if (!this.complex || this.complex.size !== bytes) {
        this.complex?.destroy();
        this.complex = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      }
      const params = this.uniform([width, height, pw, ph]);
      temporaries.push(params);
      this.dispatch(encoder, this.prepare, [phase.dataBuffer, params, this.complex, head], Math.ceil(pw / 16), Math.ceil(ph / 16));
    }
    this.device.queue.submit([encoder.finish()]);
    const phaseStats = readRecord(phaseReadback).then(stats => { onPhase?.(stats); return stats; });
    let fftStats: Promise<FrameStats | null> = Promise.resolve(null);
    if (shape && fftSlot !== null) {
      const [pw, ph] = shape;
      await this.fft.fft2DResident(this.complex!, pw, ph, false);
      const after = this.device.createCommandEncoder();
      const params = this.uniform([pw, ph, 0, 0]);
      temporaries.push(params);
      const target = this.engine.slots[fftSlot].dataBuffer;
      this.dispatch(after, this.magnitude, [this.complex!, params, target], Math.ceil(pw / 16), Math.ceil(ph / 16));
      // The FFT panel's contrast comes from the block its auto frame shows (display only).
      this.encodeStats(after, target, pw * ph, 1, temporaries, fftRegion, pw);
      const fftReadback = this.device.createBuffer({ size: RECORD_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      temporaries.push(fftReadback);
      after.copyBufferToBuffer(this.records, RECORD_BYTES, fftReadback, 0, RECORD_BYTES);
      this.device.queue.submit([after.finish()]);
      fftStats = readRecord(fftReadback);
    }
    try {
      const [phaseResult, fftResult] = await Promise.all([phaseStats, fftStats]);
      return { phase: phaseResult, fft: shape && fftResult ? { width: shape[0], height: shape[1], stats: fftResult } : null };
    } finally {
      for (const buffer of temporaries) buffer.destroy();
    }
  }


  /** The spectrum in `fftSlot` as float32, for cursor readouts and peak picks, read back on demand. */
  async readSlot(slot: number, out: Float32Array): Promise<void> {
    const source = this.engine.slots[slot].dataBuffer;
    const readback = this.device.createBuffer({ size: out.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, out.byteLength);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    out.set(new Float32Array(readback.getMappedRange()));
    readback.unmap();
    readback.destroy();
  }

  destroy() {
    this.partials.destroy();
    this.records.destroy();
    this.complex?.destroy();
  }
}

// ---------------------------------------------------------------------------
// Panel paint: colormap, window, zoom and pan in one render pass from the slot's float32 data
// into the panel's own WebGPU canvas. A contrast, colormap, zoom or pan change repaints here;
// nothing returns to the CPU and no image bitmap is made per frame.

// Geometry of drawCanvas: the image fits the square panel, zoom about the centre, pan in CSS px,
// all scaled to device pixels. Nearest sampling unless smooth; smooth averages the colours a
// device pixel covers (box filter) when the image is shrunk and interpolates them when enlarged,
// as the 2D canvas' high-quality smoothing does.
const PANEL_PAINT_WGSL = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Params {
  src_width: u32, src_height: u32, smoothing: u32, bg_rgb: u32,
  scale: f32, offset_x: f32, offset_y: f32, vmin: f32,
  vmax: f32, _a: f32, _b: f32, _c: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
struct VSOut { @builtin(position) pos: vec4f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  out.pos = vec4f(f32(i32(vi & 1u)) * 4.0 - 1.0, f32(i32(vi >> 1u)) * 4.0 - 1.0, 0.0, 1.0);
  return out;
}
fn unpack(rgb: u32) -> vec3f {
  return vec3f(f32(rgb & 0xFFu), f32((rgb >> 8u) & 0xFFu), f32((rgb >> 16u) & 0xFFu)) / 255.0;
}
fn color_at(col: u32, row: u32) -> vec3f {
  let t = display_normalize(data[row * params.src_width + col], params.vmin, params.vmax);
  return unpack(lut[min(u32(t * 255.0), 255u)]);
}
@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let width = f32(params.src_width);
  let height = f32(params.src_height);
  let x = (in.pos.x - params.offset_x) / params.scale;
  let y = (in.pos.y - params.offset_y) / params.scale;
  if (x < 0.0 || y < 0.0 || x >= width || y >= height) { return vec4f(unpack(params.bg_rgb), 1.0); }
  if (params.smoothing == 0u) { return vec4f(color_at(u32(x), u32(y)), 1.0); }
  if (params.scale >= 1.0) {
    let fx = clamp(x - 0.5, 0.0, width - 1.0);
    let fy = clamp(y - 0.5, 0.0, height - 1.0);
    let c0 = u32(fx); let r0 = u32(fy);
    let c1 = min(c0 + 1u, params.src_width - 1u); let r1 = min(r0 + 1u, params.src_height - 1u);
    let tx = fx - f32(c0); let ty = fy - f32(r0);
    let top = mix(color_at(c0, r0), color_at(c1, r0), tx);
    let bottom = mix(color_at(c0, r1), color_at(c1, r1), tx);
    return vec4f(mix(top, bottom, ty), 1.0);
  }
  // footprint of this device pixel in source pixels, sampled at most 8 x 8 times
  let reach = 0.5 / params.scale;
  let x0 = max(x - reach, 0.0); let x1 = min(x + reach, width);
  let y0 = max(y - reach, 0.0); let y1 = min(y + reach, height);
  let steps = min(8u, u32(ceil(1.0 / params.scale)) + 1u);
  var sum = vec3f(0.0);
  var weight = 0.0;
  for (var j = 0u; j < steps; j++) {
    let ya = y0 + (y1 - y0) * f32(j) / f32(steps);
    let yb = y0 + (y1 - y0) * f32(j + 1u) / f32(steps);
    for (var i = 0u; i < steps; i++) {
      let xa = x0 + (x1 - x0) * f32(i) / f32(steps);
      let xb = x0 + (x1 - x0) * f32(i + 1u) / f32(steps);
      let area = (xb - xa) * (yb - ya);
      let col = min(u32(0.5 * (xa + xb)), params.src_width - 1u);
      let row = min(u32(0.5 * (ya + yb)), params.src_height - 1u);
      sum += color_at(col, row) * area;
      weight += area;
    }
  }
  return vec4f(sum / max(weight, 1e-12), 1.0);
}
`;

/** What a panel shows from the GPU: a slot, its contrast window and its colormap. */
export type GPUPanelSource = { painter: GPUPanelPainter; slot: number; vmin: number; vmax: number; lutName: string; lut: Uint8Array };

/** The GPU source behind each panel's frame canvas (the canvas carries size and frame events). */
export const gpuPanelSources = new WeakMap<HTMLCanvasElement, GPUPanelSource>();

/** Paints engine slots straight into WebGPU panel canvases. */
export class GPUPanelPainter {
  private readonly engine: GPUColormapEngine;
  private readonly device: GPUDevice;
  private readonly pipeline: GPURenderPipeline;
  private readonly format: GPUTextureFormat;
  private readonly contexts = new WeakMap<HTMLCanvasElement, GPUCanvasContext>();

  constructor(engine: GPUColormapEngine) {
    this.engine = engine;
    this.device = engine.getDevice();
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.pipeline = renderPipeline(this.device, PANEL_PAINT_WGSL, this.format);
  }

  /**
   * Draw `source` into `canvas` at CSS size `size` with the panel's zoom and pan, as drawCanvas
   * places it. Returns false when the slot or the canvas context is unavailable.
   */
  paint(canvas: HTMLCanvasElement, source: GPUPanelSource, size: number,
    zoom: { zoom: number; panX: number; panY: number }, bgColor: string, smooth: boolean): boolean {
    const slot = this.engine.slots[source.slot];
    if (!slot || slot.dataKind !== "f32") return false;
    const dpr = window.devicePixelRatio || 1;
    const target = Math.round(size * dpr);
    if (canvas.width !== target || canvas.height !== target) { canvas.width = target; canvas.height = target; }
    let context = this.contexts.get(canvas);
    if (!context) {
      context = canvas.getContext("webgpu") as GPUCanvasContext | null ?? undefined;
      if (!context) return false;
      context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
      this.contexts.set(canvas, context);
    }
    const baseFit = Math.min(size / slot.width, size / slot.height);
    const scale = baseFit * zoom.zoom;
    const params = new ArrayBuffer(48);
    const u32 = new Uint32Array(params), f32 = new Float32Array(params);
    u32[0] = slot.width; u32[1] = slot.height; u32[2] = smooth ? 1 : 0; u32[3] = cssRgb(bgColor);
    f32[4] = scale * dpr;
    f32[5] = ((size - slot.width * scale) / 2 + zoom.panX) * dpr;
    f32[6] = ((size - slot.height * scale) / 2 + zoom.panY) * dpr;
    f32[7] = source.vmin; f32[8] = source.vmax;
    const uniform = this.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(uniform, 0, params);
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: this.engine.namedLutBuffer(source.lutName, source.lut) } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    uniform.destroy();
    return true;
  }
}

/** A CSS colour as the shader's packed 0xBBGGRR, through a 1-pixel 2D canvas (any CSS syntax). */
const rgbCache = new Map<string, number>();
function cssRgb(color: string): number {
  const cached = rgbCache.get(color);
  if (cached !== undefined) return cached;
  const probe = document.createElement("canvas").getContext("2d");
  let packed = 0;
  if (probe) {
    probe.fillStyle = color;
    probe.fillRect(0, 0, 1, 1);
    const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
    packed = r | (g << 8) | (b << 16);
  }
  rgbCache.set(color, packed);
  return packed;
}
