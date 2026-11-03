/**
 * Browser-side global slice-alignment estimation for Show3DSlices.
 *
 * Mirrors `Show3DSlices.estimate_slice_alignment` in Python: adjacent slices are
 * registered after median subtraction, Gaussian high-pass filtering, and Hann
 * windowing; the adjacent shifts are accumulated and a straight line is fit
 * versus slice index. The fitted slope is the display shift per deeper slice.
 *
 * Why this exists: the Python estimator needs a live kernel, so an exported
 * standalone HTML could never estimate alignment - the toolbar could only tell
 * the reader to go back to a notebook. The volume is already resident on the
 * GPU for slice rendering, so the same estimate runs in WebGPU with no kernel
 * and no round-trip, which makes the Align control work offline and removes a
 * comm hop when a kernel IS attached.
 *
 * On a hardware WebGPU device the estimate is GPU-resident: each slice is
 * uploaded once, and blur -> Hann window -> FFT -> cross-power -> inverse FFT ->
 * peak search then chain through device buffers via `WebGPUFFT.fft2DResident`.
 * Only the scalars come back - one energy per slice, one peak per pair, and the
 * small subpixel refinement region - so a 16 x 1688 x 1688 stack moves a few
 * hundred KB back instead of the ~1.2 GB that a readback per stage would cost.
 *
 * Without one, the CPU path below runs the same algorithm; the parity tests
 * check it against the Python estimator.
 */

import { WebGPUFFT, fft2d, nextPow2, requireWebGPUFFT } from "../display/fft";
import { selectKth } from "../display/filter";

// Matches src/quantem/widget/show3dslices/alignment.py. Changing one
// side without the other makes the browser and the kernel disagree on the shift.
const HIGHPASS_SIGMA_PX = 12.0;
const UPSAMPLE_FACTOR = 20;
const DFT_REGION_FACTOR = 1.5;
// Blur taps smaller than this fraction of the centre tap are dropped: they sit
// below float32 resolution of the blurred value.
const BLUR_TAP_CUTOFF = 1e-8;

interface SliceAlignmentEstimate {
  rowShiftPxPerSlice: number;
  colShiftPxPerSlice: number;
  adjacentShiftPx: number[][];
  cumulativeShiftPx: number[][];
  fitR2: { row: number; col: number };
  quality: number[];
  backend: "webgpu" | "cpu";
}

/** numpy "reflect" padding: mirror about the edge sample without repeating it (period 2(n - 1)). */
function reflectIndex(index: number, length: number): number {
  const period = 2 * (length - 1);
  const folded = Math.abs(index) % period;
  return folded < length ? folded : period - folded;
}

/**
 * One axis of the high-pass blur, with the numbers alignment.py computes.
 *
 * Python reflect-pads the slice by ceil(4 sigma) and multiplies its FFT by
 * exp(-2 pi^2 sigma^2 f^2) on the padded grid. That product is a circular
 * convolution of each padded axis with the inverse DFT of the transfer
 * function: output sample i is sum over offsets m of taps[m - low] times the
 * unpadded sample sourceIndex[i - m + high], where the lookup folds the circular
 * wrap and the reflect pad together. A spatial Gaussian truncated at 4 sigma
 * with mirrored edges (the previous browser blur) differed from it by up to
 * 1e-2 near the borders, enough to move a shift estimate by a refinement step.
 */
interface BlurAxis {
  low: number;
  high: number;
  taps: Float64Array;
  sourceIndex: Uint32Array<ArrayBuffer>;
}

/** A BlurAxis with its taps and source lookup on the device. */
type BlurAxisBuffers = { low: number; high: number; taps: GPUBuffer; sourceIndex: GPUBuffer };

function blurAxis(length: number, sigma: number): BlurAxis {
  const pad = Math.max(1, Math.ceil(4 * sigma));
  const padded = length + 2 * pad;
  const transfer = Float64Array.from({ length: padded }, (_, bin) => {
    const frequency = (bin < (padded + 1) >> 1 ? bin : bin - padded) / padded;
    return Math.exp(-2 * Math.PI ** 2 * sigma ** 2 * frequency ** 2);
  });
  // The kernel is even, so one tap per distance; distances past half the
  // padded length wrap onto the other side.
  const half = padded >> 1;
  const byDistance = Float64Array.from({ length: half + 1 }, (_, distance) => {
    let sum = 0;
    for (let bin = 0; bin < padded; bin++) sum += transfer[bin] * Math.cos((2 * Math.PI * bin * distance) / padded);
    return sum / padded;
  });
  let radius = half;
  while (radius > 0 && Math.abs(byDistance[radius]) < BLUR_TAP_CUTOFF * byDistance[0]) radius--;
  // A kernel reaching half the padded length covers the whole circle: one tap per residue.
  const wraps = radius >= (padded - 1) >> 1;
  const low = wraps ? -((padded - 1) >> 1) : -radius;
  const high = wraps ? half : radius;
  const taps = Float64Array.from({ length: high - low + 1 }, (_, index) => byDistance[Math.abs(index + low)]);
  const sourceIndex = Uint32Array.from({ length: length + high - low }, (_, index) => {
    const paddedIndex = (((index - high + pad) % padded) + padded) % padded;
    return reflectIndex(paddedIndex - pad, length);
  });
  return { low, high, taps, sourceIndex };
}

/**
 * Median of a copy, used to center each slice before registration.
 *
 * Quickselect rather than a full sort: this runs once per slice on every pixel
 * of the plane, and sorting 2.9M values per slice dominated the whole alignment
 * estimate. The returned order statistic is identical either way.
 */
export function median(values: Float32Array): number {
  const scratch = Float32Array.from(values);
  const mid = scratch.length >> 1;
  const upper = selectKth(scratch, mid);
  if (scratch.length % 2) return upper;
  // Even length averages the two central order statistics; everything below
  // `mid` is already <= upper after the select, so the lower one is their max.
  let lower = -Infinity;
  for (let i = 0; i < mid; i++) if (scratch[i] > lower) lower = scratch[i];
  return (lower + upper) / 2;
}

// One axis of the high-pass blur (see BlurAxis): the taps and the folded
// source lookup come from the CPU, so both paths read the same numbers.
const BLUR_SHADER = /* wgsl */ `
struct Params { width: u32, height: u32, low: i32, high: i32, horizontal: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> taps: array<f32>;
@group(0) @binding(3) var<storage, read> sourceIndex: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  var acc = 0.0;
  for (var offset = p.low; offset <= p.high; offset = offset + 1) {
    let tap = taps[u32(offset - p.low)];
    if (p.horizontal == 1u) {
      acc = acc + tap * src[gid.y * p.width + sourceIndex[u32(i32(gid.x) - offset + p.high)]];
    } else {
      acc = acc + tap * src[sourceIndex[u32(i32(gid.y) - offset + p.high)] * p.width + gid.x];
    }
  }
  dst[gid.y * p.width + gid.x] = acc;
}
`;

// Resident-pipeline kernels. Each consumes and produces GPU buffers so the
// estimate can run blur -> window -> FFT -> cross-power -> inverse FFT -> peak
// without the array ever returning to the CPU.
const RESIDENT_SHADER = /* wgsl */ `
struct Params {
  width: u32, height: u32, paddedWidth: u32, paddedHeight: u32,
  count: u32, groups: u32, _p0: u32, _p1: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const PI = 3.141592653589793;

/** centered - blurred, Hann windowed, written into a zero-padded complex frame. */
@compute @workgroup_size(16, 16)
fn prepare(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.paddedWidth || gid.y >= p.paddedHeight) { return; }
  let dst = (gid.y * p.paddedWidth + gid.x) * 2u;
  if (gid.x >= p.width || gid.y >= p.height) {
    out[dst] = 0.0;
    out[dst + 1u] = 0.0;
    return;
  }
  let src = gid.y * p.width + gid.x;
  var rowWin = 1.0;
  if (p.height > 1u) { rowWin = 0.5 - 0.5 * cos(2.0 * PI * f32(gid.y) / f32(p.height - 1u)); }
  var colWin = 1.0;
  if (p.width > 1u) { colWin = 0.5 - 0.5 * cos(2.0 * PI * f32(gid.x) / f32(p.width - 1u)); }
  out[dst] = (a[src] - b[src]) * rowWin * colWin;
  out[dst + 1u] = 0.0;
}

/**
 * product = ref * conj(mov), the cross-power spectrum feeding the inverse FFT.
 *
 * Strided rather than one invocation per element: a 2048x2048 spectrum needs
 * 65536 workgroups of 64, one past the 65535 maxComputeWorkgroupsPerDimension
 * limit, and an over-limit dispatch is rejected so the output stays zero.
 */
@compute @workgroup_size(64)
fn crossPower(@builtin(global_invocation_id) gid: vec3u) {
  var e = gid.x;
  loop {
    if (e >= p.count) { break; }
    let i = e * 2u;
    let refRe = a[i]; let refIm = a[i + 1u];
    let movRe = b[i]; let movIm = b[i + 1u];
    out[i] = refRe * movRe + refIm * movIm;
    out[i + 1u] = refIm * movRe - refRe * movIm;
    e = e + 64u * p.groups;
  }
}
`;

// Reductions are split from the elementwise kernels because they need a
// different binding shape (one input, one small output).
const REDUCE_SHADER = /* wgsl */ `
struct Params { count: u32, groups: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;

var<workgroup> shared_value: array<f32, 256>;
var<workgroup> shared_index: array<u32, 256>;

/** Sum of squares of the real components - the registration image energy. */
@compute @workgroup_size(256)
fn energy(@builtin(global_invocation_id) gid: vec3u,
          @builtin(local_invocation_id) lid: vec3u,
          @builtin(workgroup_id) wid: vec3u) {
  var acc = 0.0;
  var i = gid.x;
  loop {
    if (i >= p.count) { break; }
    let v = src[i * 2u];
    acc = acc + v * v;
    i = i + 256u * p.groups;
  }
  shared_value[lid.x] = acc;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { shared_value[lid.x] = shared_value[lid.x] + shared_value[lid.x + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lid.x == 0u) { partials[wid.x] = shared_value[0]; }
}

/**
 * Largest |value|^2 and its index. Ties keep the LOWEST index so the result
 * matches a scalar scan with a strict greater-than, which is what the CPU path
 * and the Python reference both do.
 */
@compute @workgroup_size(256)
fn peak(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  var bestValue = -1.0;
  var bestIndex = 0u;
  var i = gid.x;
  loop {
    if (i >= p.count) { break; }
    let re = src[i * 2u];
    let im = src[i * 2u + 1u];
    let m = re * re + im * im;
    if (m > bestValue) { bestValue = m; bestIndex = i; }
    i = i + 256u * p.groups;
  }
  shared_value[lid.x] = bestValue;
  shared_index[lid.x] = bestIndex;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      let other = shared_value[lid.x + stride];
      let otherIdx = shared_index[lid.x + stride];
      if (other > shared_value[lid.x] || (other == shared_value[lid.x] && otherIdx < shared_index[lid.x])) {
        shared_value[lid.x] = other;
        shared_index[lid.x] = otherIdx;
      }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lid.x == 0u) {
    partials[wid.x * 2u] = shared_value[0];
    partials[wid.x * 2u + 1u] = bitcast<f32>(shared_index[0]);
  }
}
`;

// Separable upsampled inverse DFT (Guizar-Sicairos). Pass A contracts the row
// axis of the full spectrum down to the small region; pass B contracts the
// column axis. Doing it separably turns an O(region^2 * N^2) evaluation into
// O(region * N^2), which is what makes subpixel refinement affordable per pair.
const UPSAMPLED_DFT_SHADER = /* wgsl */ `
struct Params {
  rows: u32, cols: u32, region: u32, upsample: u32,
  offsetRow: f32, offsetCol: f32, _p0: f32, _p1: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;      // interleaved complex
@group(0) @binding(2) var<storage, read_write> dst: array<f32>; // interleaved complex

const TAU = 6.283185307179586;

/** fftfreq(n, d=upsample)[i] - the sample frequency numpy pairs with this bin. */
fn freq(i: u32, n: u32, upsample: f32) -> f32 {
  var k = f32(i);
  if (i >= (n + 1u) / 2u) { k = k - f32(n); }
  return k / (f32(n) * upsample);
}

// A[r, c] = sum_u src[u, c] * exp(-i*TAU*(r - offsetRow)*freq(u))
@compute @workgroup_size(1, 64)
fn rowPass(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.region || gid.y >= p.cols) { return; }
  let coord = f32(gid.x) - p.offsetRow;
  let upsample = f32(p.upsample);
  var accRe = 0.0;
  var accIm = 0.0;
  for (var u = 0u; u < p.rows; u = u + 1u) {
    let angle = -TAU * coord * freq(u, p.rows, upsample);
    let c = cos(angle);
    let s = sin(angle);
    let idx = (u * p.cols + gid.y) * 2u;
    let re = src[idx];
    let im = src[idx + 1u];
    accRe = accRe + re * c - im * s;
    accIm = accIm + re * s + im * c;
  }
  let out = (gid.x * p.cols + gid.y) * 2u;
  dst[out] = accRe;
  dst[out + 1u] = accIm;
}

// R[r, s] = sum_c A[r, c] * exp(-i*TAU*(s - offsetCol)*freq(c))
@compute @workgroup_size(8, 8)
fn colPass(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.region || gid.y >= p.region) { return; }
  let coord = f32(gid.y) - p.offsetCol;
  let upsample = f32(p.upsample);
  var accRe = 0.0;
  var accIm = 0.0;
  for (var c = 0u; c < p.cols; c = c + 1u) {
    let angle = -TAU * coord * freq(c, p.cols, upsample);
    let cc = cos(angle);
    let ss = sin(angle);
    let idx = (gid.x * p.cols + c) * 2u;
    let re = src[idx];
    let im = src[idx + 1u];
    accRe = accRe + re * cc - im * ss;
    accIm = accIm + re * ss + im * cc;
  }
  let out = (gid.x * p.region + gid.y) * 2u;
  dst[out] = accRe;
  dst[out + 1u] = accIm;
}
`;

/**
 * GPU helper holding the blur, registration, reduction and upsampled-DFT
 * pipelines for one estimate run. Built only when a device exists; without one
 * the estimate takes the CPU path.
 */
class SliceAlignmentGPU {
  private device: GPUDevice;
  private blurPipeline: GPUComputePipeline;
  private rowPassPipeline: GPUComputePipeline;
  private colPassPipeline: GPUComputePipeline;
  private preparePipeline: GPUComputePipeline;
  private crossPowerPipeline: GPUComputePipeline;
  private energyPipeline: GPUComputePipeline;
  private peakPipeline: GPUComputePipeline;

  constructor(device: GPUDevice) {
    this.device = device;
    const blurModule = device.createShaderModule({ code: BLUR_SHADER });
    this.blurPipeline = device.createComputePipeline({
      layout: "auto", compute: { module: blurModule, entryPoint: "main" },
    });
    const dftModule = device.createShaderModule({ code: UPSAMPLED_DFT_SHADER });
    this.rowPassPipeline = device.createComputePipeline({
      layout: "auto", compute: { module: dftModule, entryPoint: "rowPass" },
    });
    this.colPassPipeline = device.createComputePipeline({
      layout: "auto", compute: { module: dftModule, entryPoint: "colPass" },
    });
    const residentModule = device.createShaderModule({ code: RESIDENT_SHADER });
    this.preparePipeline = device.createComputePipeline({
      layout: "auto", compute: { module: residentModule, entryPoint: "prepare" },
    });
    this.crossPowerPipeline = device.createComputePipeline({
      layout: "auto", compute: { module: residentModule, entryPoint: "crossPower" },
    });
    const reduceModule = device.createShaderModule({ code: REDUCE_SHADER });
    this.energyPipeline = device.createComputePipeline({
      layout: "auto", compute: { module: reduceModule, entryPoint: "energy" },
    });
    this.peakPipeline = device.createComputePipeline({
      layout: "auto", compute: { module: reduceModule, entryPoint: "peak" },
    });
  }

  buffer(byteLength: number, extraUsage: GPUBufferUsageFlags = 0): GPUBuffer {
    return this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extraUsage,
    });
  }

  write(buffer: GPUBuffer, data: Float32Array<ArrayBuffer>): void {
    this.device.queue.writeBuffer(buffer, 0, data);
  }

  /** Device-to-device copy, so intermediate results never touch the CPU. */
  copyBuffer(src: GPUBuffer, dst: GPUBuffer, byteLength: number): void {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, dst, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
  }

  /** Read the first `byteLength` bytes of a device buffer back as float32 through a mappable copy. */
  private async readBack(source: GPUBuffer, byteLength: number): Promise<Float32Array<ArrayBuffer>> {
    const readBuffer = this.device.createBuffer({
      size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readBuffer, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();
    return out;
  }

  /** Bind `entries` to `pipeline` and submit one dispatch. */
  private dispatch(
    pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[], groupsX: number, groupsY = 1,
  ): void {
    const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groupsX, groupsY);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Uniform for RESIDENT_SHADER: plane and padded sizes, element count and dispatched groups. */
  private residentParams(
    width: number, height: number, paddedWidth: number, paddedHeight: number,
    count: number, groups = 0,
  ): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      buffer, 0, new Uint32Array([width, height, paddedWidth, paddedHeight, count, groups, 0, 0]),
    );
    return buffer;
  }

  /** Taps and source lookup of one blur axis, uploaded once per estimate. */
  blurAxisBuffers(axis: BlurAxis): BlurAxisBuffers {
    const taps = this.device.createBuffer({ size: axis.taps.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(taps, 0, Float32Array.from(axis.taps));
    const sourceIndex = this.device.createBuffer({ size: axis.sourceIndex.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(sourceIndex, 0, axis.sourceIndex);
    return { low: axis.low, high: axis.high, taps, sourceIndex };
  }

  /** Blur one real-valued plane in place: along columns (each row), then along rows. */
  blurResident(
    src: GPUBuffer, scratch: GPUBuffer, width: number, height: number, rowAxis: BlurAxisBuffers, colAxis: BlurAxisBuffers,
  ): void {
    const pass = (from: GPUBuffer, to: GPUBuffer, axis: BlurAxisBuffers, horizontal: boolean) => {
      const params = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const values = new ArrayBuffer(32);
      new Uint32Array(values).set([width, height, 0, 0, horizontal ? 1 : 0]);
      new Int32Array(values, 8, 2).set([axis.low, axis.high]);
      this.device.queue.writeBuffer(params, 0, values);
      this.dispatch(this.blurPipeline, [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: from } },
        { binding: 2, resource: { buffer: axis.taps } },
        { binding: 3, resource: { buffer: axis.sourceIndex } },
        { binding: 4, resource: { buffer: to } },
      ], Math.ceil(width / 16), Math.ceil(height / 16));
      params.destroy();
    };
    pass(src, scratch, colAxis, true);
    pass(scratch, src, rowAxis, false);
  }

  /** centered - blurred, Hann windowed, into a zero-padded complex frame. */
  prepareResident(
    centered: GPUBuffer, blurred: GPUBuffer, out: GPUBuffer,
    width: number, height: number, paddedWidth: number, paddedHeight: number,
  ): void {
    const params = this.residentParams(width, height, paddedWidth, paddedHeight, 0);
    this.dispatch(this.preparePipeline, [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: centered } },
      { binding: 2, resource: { buffer: blurred } },
      { binding: 3, resource: { buffer: out } },
    ], Math.ceil(paddedWidth / 16), Math.ceil(paddedHeight / 16));
    params.destroy();
  }

  crossPowerResident(ref: GPUBuffer, mov: GPUBuffer, out: GPUBuffer, count: number): void {
    // Capped well under maxComputeWorkgroupsPerDimension; the kernel strides.
    const groups = Math.min(4096, Math.max(1, Math.ceil(count / 64)));
    const params = this.residentParams(0, 0, 0, 0, count, groups);
    this.dispatch(this.crossPowerPipeline, [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: ref } },
      { binding: 2, resource: { buffer: mov } },
      { binding: 3, resource: { buffer: out } },
    ], groups);
    params.destroy();
  }

  /** Reduce a complex buffer to per-workgroup partials, then finish on the CPU. */
  private async reduce(
    pipeline: GPUComputePipeline, src: GPUBuffer, count: number, stride: number,
  ): Promise<Float32Array<ArrayBuffer>> {
    const groups = 64;
    const partials = this.device.createBuffer({
      size: groups * stride * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const params = this.device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(params, 0, new Uint32Array([count, groups, 0, 0]));
    this.dispatch(pipeline, [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: src } },
      { binding: 2, resource: { buffer: partials } },
    ], groups);
    const out = await this.readBack(partials, groups * stride * 4);
    partials.destroy(); params.destroy();
    return out;
  }

  /** Total sum of squares of the real components. */
  async energyResident(src: GPUBuffer, count: number): Promise<number> {
    const partials = await this.reduce(this.energyPipeline, src, count, 1);
    let total = 0;
    for (let i = 0; i < partials.length; i++) total += partials[i];
    return total;
  }

  /** Peak |value|^2 and its flat index, lowest index winning ties. */
  async peakResident(src: GPUBuffer, count: number): Promise<{ value: number; index: number }> {
    const partials = await this.reduce(this.peakPipeline, src, count, 2);
    const indices = new Uint32Array(partials.buffer);
    let value = -1;
    let index = 0;
    for (let i = 0; i < partials.length / 2; i++) {
      const candidate = partials[i * 2];
      const candidateIndex = indices[i * 2 + 1];
      if (candidate > value || (candidate === value && candidateIndex < index)) {
        value = candidate; index = candidateIndex;
      }
    }
    return { value, index };
  }

  /** Upsampled inverse DFT reading a spectrum that is already on the device. */
  async upsampledDftResident(
    spectrum: GPUBuffer, rows: number, cols: number, region: number,
    upsample: number, offsetRow: number, offsetCol: number,
  ): Promise<Float32Array<ArrayBuffer>> {
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const midBuffer = this.device.createBuffer({ size: region * cols * 2 * 4, usage: storage });
    const dstBuffer = this.device.createBuffer({ size: region * region * 2 * 4, usage: storage });
    const paramsBuffer = this.device.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const params = new ArrayBuffer(32);
    new Uint32Array(params, 0, 4).set([rows, cols, region, upsample]);
    new Float32Array(params, 16, 2).set([offsetRow, offsetCol]);
    this.device.queue.writeBuffer(paramsBuffer, 0, params);
    this.dispatch(this.rowPassPipeline, [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: spectrum } },
      { binding: 2, resource: { buffer: midBuffer } },
    ], region, Math.ceil(cols / 64));
    this.dispatch(this.colPassPipeline, [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: midBuffer } },
      { binding: 2, resource: { buffer: dstBuffer } },
    ], Math.ceil(region / 8), Math.ceil(region / 8));
    const out = await this.readBack(dstBuffer, region * region * 2 * 4);
    midBuffer.destroy(); dstBuffer.destroy(); paramsBuffer.destroy();
    return out;
  }
}

/**
 * CPU twin of the WGSL upsampled inverse DFT, separable in the same order.
 * Without it the fallback path would stop at whole-pixel peaks, and a stack
 * drifting by a fraction of a pixel per slice would estimate a zero slope.
 */
function upsampledDftCpu(
  spectrum: Float32Array,
  rows: number,
  cols: number,
  region: number,
  upsample: number,
  offsetRow: number,
  offsetCol: number,
): Float32Array<ArrayBuffer> {
  const freq = (index: number, n: number) => {
    const k = index >= (n + 1) >> 1 ? index - n : index;
    return k / (n * upsample);
  };
  const middle = new Float32Array(region * cols * 2);
  for (let r = 0; r < region; r++) {
    const coord = r - offsetRow;
    for (let u = 0; u < rows; u++) {
      const angle = -2 * Math.PI * coord * freq(u, rows);
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      for (let c = 0; c < cols; c++) {
        const src = (u * cols + c) * 2;
        const dst = (r * cols + c) * 2;
        const re = spectrum[src];
        const im = spectrum[src + 1];
        middle[dst] += re * cosine - im * sine;
        middle[dst + 1] += re * sine + im * cosine;
      }
    }
  }
  const out = new Float32Array(region * region * 2);
  for (let r = 0; r < region; r++) {
    for (let s = 0; s < region; s++) {
      const coord = s - offsetCol;
      let accRe = 0;
      let accIm = 0;
      for (let c = 0; c < cols; c++) {
        const angle = -2 * Math.PI * coord * freq(c, cols);
        const cosine = Math.cos(angle);
        const sine = Math.sin(angle);
        const src = (r * cols + c) * 2;
        const re = middle[src];
        const im = middle[src + 1];
        accRe += re * cosine - im * sine;
        accIm += re * sine + im * cosine;
      }
      const dst = (r * region + s) * 2;
      out[dst] = accRe;
      out[dst + 1] = accIm;
    }
  }
  return out;
}

/** The high-pass blur of one plane on the CPU: along columns (each row), then along rows. */
function blurCpu(image: Float32Array, width: number, height: number, rowAxis: BlurAxis, colAxis: BlurAxis): Float32Array<ArrayBuffer> {
  const horizontal = new Float64Array(image.length);
  for (let row = 0; row < height; row++) {
    const base = row * width;
    for (let col = 0; col < width; col++) {
      let sum = 0;
      for (let offset = colAxis.low; offset <= colAxis.high; offset++) {
        sum += colAxis.taps[offset - colAxis.low] * image[base + colAxis.sourceIndex[col - offset + colAxis.high]];
      }
      horizontal[base + col] = sum;
    }
  }
  const out = new Float32Array(image.length);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      let sum = 0;
      for (let offset = rowAxis.low; offset <= rowAxis.high; offset++) {
        sum += rowAxis.taps[offset - rowAxis.low] * horizontal[rowAxis.sourceIndex[row - offset + rowAxis.high] * width + col];
      }
      out[row * width + col] = sum;
    }
  }
  return out;
}

/** np.hanning(length) in float64. */
function hannWindow(length: number): Float64Array {
  return Float64Array.from({ length }, (_, index) => (length > 1 ? 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / (length - 1)) : 1));
}

/** The high-pass sigma alignment.py uses: 12 px, capped at a sixth of the short side. */
function highPassSigma(width: number, height: number): number {
  return Math.min(HIGHPASS_SIGMA_PX, Math.max(1.0, Math.min(width, height) / 6.0));
}

/**
 * Build the registration image for one slice: median-centered, Gaussian
 * high-passed, Hann windowed (alignment.py ``registration_image``). The high
 * pass removes the slowly varying background that would otherwise dominate
 * the correlation peak; the window removes the edge discontinuity that a
 * periodic FFT would read as structure. Python skips the high pass below 3
 * samples per side, and so does this. The subtraction and window product
 * round to float32 as numpy's float32 arrays do.
 */
export function registrationImage(slice: Float32Array, width: number, height: number): Float32Array<ArrayBuffer> {
  const centered = new Float32Array(slice.length);
  for (let i = 0; i < slice.length; i++) centered[i] = Number.isFinite(slice[i]) ? slice[i] : 0;
  const medianValue = median(centered);
  for (let i = 0; i < centered.length; i++) centered[i] -= medianValue;
  const sigma = highPassSigma(width, height);
  const blurred = width >= 3 && height >= 3
    ? blurCpu(centered, width, height, blurAxis(height, sigma), blurAxis(width, sigma))
    : new Float32Array(slice.length);
  const rowWindow = hannWindow(height);
  const colWindow = hannWindow(width);
  const out = new Float32Array(slice.length);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const i = row * width + col;
      out[i] = Math.fround(centered[i] - blurred[i]) * Math.fround(rowWindow[row] * colWindow[col]);
    }
  }
  return out;
}

/** Zero-pad into a power-of-two frame so the FFT helper never crops the spectrum. */
function padToPow2(
  image: Float32Array<ArrayBuffer>, width: number, height: number,
): { data: Float32Array<ArrayBuffer>; width: number; height: number } {
  const paddedWidth = nextPow2(width);
  const paddedHeight = nextPow2(height);
  if (paddedWidth === width && paddedHeight === height) return { data: image, width, height };
  const data = new Float32Array(paddedWidth * paddedHeight);
  for (let row = 0; row < height; row++) data.set(image.subarray(row * width, (row + 1) * width), row * paddedWidth);
  return { data, width: paddedWidth, height: paddedHeight };
}

/** Least-squares slope/intercept of shift versus slice index, plus the fit R^2. */
function linearFit(sliceIndices: number[], shifts: number[]): { slope: number; intercept: number; r2: number } {
  const n = sliceIndices.length;
  const meanSlice = sliceIndices.reduce((a, b) => a + b, 0) / n;
  const meanShift = shifts.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (sliceIndices[i] - meanSlice) * (shifts[i] - meanShift);
    den += (sliceIndices[i] - meanSlice) * (sliceIndices[i] - meanSlice);
  }
  const slope = den > 0 ? num / den : 0;
  const intercept = meanShift - slope * meanSlice;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const predicted = slope * sliceIndices[i] + intercept;
    ssRes += (shifts[i] - predicted) * (shifts[i] - predicted);
    ssTot += (shifts[i] - meanShift) * (shifts[i] - meanShift);
  }
  return { slope, intercept, r2: ssTot > 0 ? 1 - ssRes / ssTot : 0 };
}

/**
 * Unwrap a correlation peak's flat index into a signed (row, col) shift, then
 * round it onto the 1/UPSAMPLE_FACTOR grid the subpixel refinement starts from.
 * Indices past the midpoint are negative shifts wrapped around the FFT frame.
 */
function coarseShift(peakIndex: number, paddedWidth: number, paddedHeight: number): [number, number] {
  let shiftRow = Math.floor(peakIndex / paddedWidth);
  let shiftCol = peakIndex % paddedWidth;
  if (shiftRow > Math.trunc(paddedHeight / 2)) shiftRow -= paddedHeight;
  if (shiftCol > Math.trunc(paddedWidth / 2)) shiftCol -= paddedWidth;
  return [
    Math.round(shiftRow * UPSAMPLE_FACTOR) / UPSAMPLE_FACTOR,
    Math.round(shiftCol * UPSAMPLE_FACTOR) / UPSAMPLE_FACTOR,
  ];
}

/**
 * Subpixel (row, col) correction from the upsampled region: the brightest
 * |value|^2 sample (first one on ties), offset from the region center and
 * divided by UPSAMPLE_FACTOR.
 */
function subpixelOffset(refined: Float32Array, region: number, dftShift: number): [number, number] {
  let bestIndex = 0;
  let bestValue = -1;
  for (let i = 0; i < region * region; i++) {
    const magnitude = refined[i * 2] * refined[i * 2] + refined[i * 2 + 1] * refined[i * 2 + 1];
    if (magnitude > bestValue) { bestValue = magnitude; bestIndex = i; }
  }
  return [
    (Math.floor(bestIndex / region) - dftShift) / UPSAMPLE_FACTOR,
    ((bestIndex % region) - dftShift) / UPSAMPLE_FACTOR,
  ];
}

/** Accumulate the per-pair shifts and fit each axis's shift per slice by least squares. */
function fitFromAdjacent(adjacent: number[][], numSlices: number, backend: "webgpu" | "cpu", quality: number[]) {
  const cumulative: number[][] = [[0, 0]];
  for (let i = 0; i < adjacent.length; i++) {
    const previous = cumulative[i];
    cumulative.push([previous[0] + adjacent[i][0], previous[1] + adjacent[i][1]]);
  }
  const sliceIndices = Array.from({ length: numSlices }, (_, i) => i);
  const rowFit = linearFit(sliceIndices, cumulative.map((entry) => entry[0]));
  const colFit = linearFit(sliceIndices, cumulative.map((entry) => entry[1]));
  return {
    rowShiftPxPerSlice: rowFit.slope,
    colShiftPxPerSlice: colFit.slope,
    adjacentShiftPx: adjacent,
    cumulativeShiftPx: cumulative,
    fitR2: { row: rowFit.r2, col: colFit.r2 },
    quality,
    backend,
  };
}

/**
 * GPU-resident estimate: the volume crosses the bus once per slice and the
 * spectra never come back.
 *
 * The stage chain blur -> window -> FFT -> cross-power -> inverse FFT -> peak
 * runs entirely in device buffers. Only three small results are read back per
 * pair - the peak value and index, the energy scalars, and the tiny refinement
 * region - instead of the full padded spectrum at every stage boundary. On a
 * 16 x 1688 x 1688 stack that is a few hundred KB rather than ~1.2 GB.
 */
async function estimateResident(
  volume: Float32Array,
  numCols: number,
  numRows: number,
  numSlices: number,
  fft: WebGPUFFT,
  gpu: SliceAlignmentGPU,
): Promise<SliceAlignmentEstimate> {
  const planeSize = numCols * numRows;
  const paddedWidth = nextPow2(numCols);
  const paddedHeight = nextPow2(numRows);
  const complexCount = paddedWidth * paddedHeight;
  const sigma = highPassSigma(numCols, numRows);
  const highPass = numCols >= 3 && numRows >= 3;
  const rowAxis = gpu.blurAxisBuffers(blurAxis(numRows, sigma));
  const colAxis = gpu.blurAxisBuffers(blurAxis(numCols, sigma));

  // Stays zero when the plane is too small to high-pass, as in alignment.py.
  const planeBuffer = gpu.buffer(planeSize * 4);
  const scratchBuffer = gpu.buffer(planeSize * 4);
  // Two spectra are live at once (the pair being correlated), so a rolling pair
  // of buffers is enough no matter how deep the stack is.
  const spectra = [gpu.buffer(complexCount * 2 * 4), gpu.buffer(complexCount * 2 * 4)];
  const productBuffer = gpu.buffer(complexCount * 2 * 4);
  const correlationBuffer = gpu.buffer(complexCount * 2 * 4);
  const conjugateBuffer = gpu.buffer(complexCount * 2 * 4);

  const centeredBuffer = gpu.buffer(planeSize * 4);
  const centered = new Float32Array(planeSize);
  const buildSpectrum = async (sliceIndex: number, target: GPUBuffer): Promise<number> => {
    const slice = volume.subarray(sliceIndex * planeSize, (sliceIndex + 1) * planeSize);
    for (let i = 0; i < planeSize; i++) centered[i] = Number.isFinite(slice[i]) ? slice[i] : 0;
    const medianValue = median(centered);
    for (let i = 0; i < planeSize; i++) centered[i] -= medianValue;
    // The one upload per slice. Everything downstream stays on the device.
    gpu.write(centeredBuffer, centered);
    // blurResident overwrites its source, so blur a device-side copy and keep
    // the centered plane intact for the subtraction.
    if (highPass) {
      gpu.copyBuffer(centeredBuffer, planeBuffer, planeSize * 4);
      gpu.blurResident(planeBuffer, scratchBuffer, numCols, numRows, rowAxis, colAxis);
    }
    gpu.prepareResident(centeredBuffer, planeBuffer, target, numCols, numRows, paddedWidth, paddedHeight);
    const energy = await gpu.energyResident(target, complexCount);
    await fft.fft2DResident(target, paddedWidth, paddedHeight, false);
    return energy;
  };

  const region = Math.ceil(UPSAMPLE_FACTOR * DFT_REGION_FACTOR);
  const dftShift = Math.trunc(region / 2.0);
  const adjacent: number[][] = [];
  const quality: number[] = [];
  let previousEnergy = await buildSpectrum(0, spectra[0]);
  for (let sliceIndex = 0; sliceIndex < numSlices - 1; sliceIndex++) {
    const refBuffer = spectra[sliceIndex % 2];
    const movBuffer = spectra[(sliceIndex + 1) % 2];
    const movEnergy = await buildSpectrum(sliceIndex + 1, movBuffer);
    if (previousEnergy === 0 || movEnergy === 0) {
      adjacent.push([0, 0]);
      quality.push(0);
      previousEnergy = movEnergy;
      continue;
    }
    gpu.crossPowerResident(refBuffer, movBuffer, productBuffer, complexCount);
    // The inverse FFT is destructive, so correlate on a copy and keep the
    // product for the subpixel refinement.
    gpu.copyBuffer(productBuffer, correlationBuffer, complexCount * 2 * 4);
    await fft.fft2DResident(correlationBuffer, paddedWidth, paddedHeight, true);
    const peak = await gpu.peakResident(correlationBuffer, complexCount);
    const [refinedRow, refinedCol] = coarseShift(peak.index, paddedWidth, paddedHeight);
    // The refinement samples conj(product), exactly as the CPU path does.
    // conj(ref * conj(mov)) == mov * conj(ref), so the same cross-power kernel
    // with the operands swapped produces it without a dedicated conjugate pass.
    gpu.crossPowerResident(movBuffer, refBuffer, conjugateBuffer, complexCount);
    const refined = await gpu.upsampledDftResident(
      conjugateBuffer, paddedHeight, paddedWidth, region, UPSAMPLE_FACTOR,
      dftShift - refinedRow * UPSAMPLE_FACTOR,
      dftShift - refinedCol * UPSAMPLE_FACTOR,
    );
    const [rowCorrection, colCorrection] = subpixelOffset(refined, region, dftShift);
    adjacent.push([refinedRow + rowCorrection, refinedCol + colCorrection]);
    const norm = Math.sqrt(previousEnergy * movEnergy);
    quality.push(norm > 0 ? Math.sqrt(peak.value) / norm : 0);
    previousEnergy = movEnergy;
  }

  for (const buffer of [rowAxis.taps, rowAxis.sourceIndex, colAxis.taps, colAxis.sourceIndex, planeBuffer, scratchBuffer, centeredBuffer, productBuffer, correlationBuffer, conjugateBuffer, ...spectra]) {
    buffer.destroy();
  }
  return fitFromAdjacent(adjacent, numSlices, "webgpu", quality);
}

/**
 * Estimate the global row/col shift per slice for one volume.
 *
 * @param volume Contiguous slice-major float volume (numSlices planes of numRows x numCols).
 * @param device The page's hardware WebGPU device (getHardwareGPUDevice) for the
 *   GPU-resident estimate, or null for the CPU path.
 */
export async function estimateSliceAlignment(
  volume: Float32Array,
  numCols: number,
  numRows: number,
  numSlices: number,
  device: GPUDevice | null,
): Promise<SliceAlignmentEstimate> {
  if (numSlices < 2) throw new Error("slice alignment requires at least 2 slices");
  if (device) return estimateResident(volume, numCols, numRows, numSlices, await requireWebGPUFFT("Slice alignment"), new SliceAlignmentGPU(device));
  const planeSize = numCols * numRows;

  const spectra: { real: Float32Array; imag: Float32Array }[] = [];
  const energies: number[] = [];
  let paddedWidth = numCols;
  let paddedHeight = numRows;
  for (let sliceIndex = 0; sliceIndex < numSlices; sliceIndex++) {
    const prepared = registrationImage(volume.subarray(sliceIndex * planeSize, (sliceIndex + 1) * planeSize), numCols, numRows);
    let energy = 0;
    for (let i = 0; i < prepared.length; i++) energy += prepared[i] * prepared[i];
    energies.push(energy);
    const padded = padToPow2(prepared, numCols, numRows);
    paddedWidth = padded.width;
    paddedHeight = padded.height;
    const real = Float32Array.from(padded.data);
    const imag = new Float32Array(padded.data.length);
    fft2d(real, imag, padded.width, padded.height, false);
    spectra.push({ real, imag });
  }

  const region = Math.ceil(UPSAMPLE_FACTOR * DFT_REGION_FACTOR);
  const dftShift = Math.trunc(region / 2.0);
  const adjacent: number[][] = [];
  const quality: number[] = [];
  for (let sliceIndex = 0; sliceIndex < numSlices - 1; sliceIndex++) {
    const ref = spectra[sliceIndex];
    const mov = spectra[sliceIndex + 1];
    if (energies[sliceIndex] === 0 || energies[sliceIndex + 1] === 0) {
      adjacent.push([0, 0]);
      quality.push(0);
      continue;
    }
    // Cross power spectrum: ref * conj(mov). Its inverse transform peaks at the
    // shift that carries mov onto ref.
    const count = ref.real.length;
    const productReal = new Float32Array(count);
    const productImag = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      productReal[i] = ref.real[i] * mov.real[i] + ref.imag[i] * mov.imag[i];
      productImag[i] = ref.imag[i] * mov.real[i] - ref.real[i] * mov.imag[i];
    }
    const corrReal = Float32Array.from(productReal);
    const corrImag = Float32Array.from(productImag);
    fft2d(corrReal, corrImag, paddedWidth, paddedHeight, true);
    let peakIndex = 0;
    let peakValue = -1;
    for (let i = 0; i < count; i++) {
      const magnitude = corrReal[i] * corrReal[i] + corrImag[i] * corrImag[i];
      if (magnitude > peakValue) { peakValue = magnitude; peakIndex = i; }
    }
    const [refinedRow, refinedCol] = coarseShift(peakIndex, paddedWidth, paddedHeight);
    // Refine against conj(product) and conjugate back, matching the Python
    // path; only the peak location is used, so the conjugation is about
    // keeping the sampled grid identical rather than the values themselves.
    const conjugated = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      conjugated[i * 2] = productReal[i];
      conjugated[i * 2 + 1] = -productImag[i];
    }
    const refined = upsampledDftCpu(
      conjugated, paddedHeight, paddedWidth, region, UPSAMPLE_FACTOR,
      dftShift - refinedRow * UPSAMPLE_FACTOR, dftShift - refinedCol * UPSAMPLE_FACTOR,
    );
    const [rowCorrection, colCorrection] = subpixelOffset(refined, region, dftShift);
    adjacent.push([refinedRow + rowCorrection, refinedCol + colCorrection]);
    const norm = Math.sqrt(energies[sliceIndex] * energies[sliceIndex + 1]);
    quality.push(norm > 0 ? Math.sqrt(peakValue) / norm : 0);
  }
  return fitFromAdjacent(adjacent, numSlices, "cpu", quality);
}
