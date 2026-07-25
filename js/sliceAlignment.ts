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
 * The two expensive stages run in WGSL: the separable reflect-boundary Gaussian
 * blur that builds each registration image, and the separable upsampled inverse
 * DFT that refines each correlation peak to subpixel precision. The FFTs reuse
 * the shared `WebGPUFFT` helper. Elementwise cross-power and peak search stay on
 * the CPU: they are O(N^2) scalar work that costs a few milliseconds, far below
 * the transfer they would need to run anywhere else.
 */

import { WebGPUFFT, fft2d, nextPow2 } from "./fft";

// Matches _SLICE_ALIGNMENT_* in src/quantem/widget/show3dslices.py. Changing one
// side without the other makes the browser and the kernel disagree on the shift.
const HIGHPASS_SIGMA_PX = 12.0;
const UPSAMPLE_FACTOR = 20;
const DFT_REGION_FACTOR = 1.5;

export interface SliceAlignmentEstimate {
  rowShiftPxPerSlice: number;
  colShiftPxPerSlice: number;
  adjacentShiftPx: number[][];
  cumulativeShiftPx: number[][];
  fitR2: { row: number; col: number };
  quality: number[];
  backend: "webgpu" | "cpu";
}

/** Reflect an out-of-range sample index back inside [0, n), matching numpy's "reflect" pad. */
function reflectIndex(index: number, n: number): number {
  if (n <= 1) return 0;
  let i = index;
  while (i < 0 || i >= n) {
    if (i < 0) i = -i - 1;
    if (i >= n) i = 2 * n - i - 1;
  }
  return i;
}

/** Sampled 1D Gaussian truncated at 4 sigma, normalized to unit sum. */
function gaussianKernel1d(sigma: number): Float32Array<ArrayBuffer> {
  const radius = Math.max(1, Math.ceil(4.0 * sigma));
  const kernel = new Float32Array(radius * 2 + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const value = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = value;
    sum += value;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  return kernel;
}

/** Median of a copy, used to center each slice before registration. */
function median(values: Float32Array): number {
  const sorted = Float32Array.from(values).sort();
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

const BLUR_SHADER = /* wgsl */ `
struct Params { width: u32, height: u32, radius: u32, horizontal: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> kern: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

fn reflect(i: i32, n: i32) -> i32 {
  var v = i;
  loop {
    if (v >= 0 && v < n) { break; }
    if (v < 0) { v = -v - 1; }
    if (v >= n) { v = 2 * n - v - 1; }
  }
  return v;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let radius = i32(p.radius);
  var acc = 0.0;
  for (var t = -radius; t <= radius; t = t + 1) {
    var sx = i32(gid.x);
    var sy = i32(gid.y);
    if (p.horizontal == 1u) { sx = reflect(sx + t, i32(p.width)); }
    else { sy = reflect(sy + t, i32(p.height)); }
    acc = acc + src[u32(sy) * p.width + u32(sx)] * kern[u32(t + radius)];
  }
  dst[gid.y * p.width + gid.x] = acc;
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
 * GPU helper holding the blur and upsampled-DFT pipelines for one estimate run.
 * Falls back to null when WebGPU is unavailable so callers can take the CPU path.
 */
class SliceAlignmentGPU {
  private device: GPUDevice;
  private blurPipeline: GPUComputePipeline;
  private rowPassPipeline: GPUComputePipeline;
  private colPassPipeline: GPUComputePipeline;

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
  }

  /** Reflect-boundary Gaussian blur, run as two separable passes. */
  async blur(image: Float32Array<ArrayBuffer>, width: number, height: number, sigma: number): Promise<Float32Array<ArrayBuffer>> {
    const kernel = gaussianKernel1d(sigma);
    const radius = (kernel.length - 1) / 2;
    const byteLength = image.byteLength;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const bufferA = this.device.createBuffer({ size: byteLength, usage });
    const bufferB = this.device.createBuffer({ size: byteLength, usage });
    const kernelBuffer = this.device.createBuffer({
      size: kernel.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const paramsBuffer = this.device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const readBuffer = this.device.createBuffer({
      size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(bufferA, 0, image);
    this.device.queue.writeBuffer(kernelBuffer, 0, kernel);

    const pass = (src: GPUBuffer, dst: GPUBuffer, horizontal: boolean) => {
      this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([width, height, radius, horizontal ? 1 : 0]));
      const bindGroup = this.device.createBindGroup({
        layout: this.blurPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: src } },
          { binding: 2, resource: { buffer: kernelBuffer } },
          { binding: 3, resource: { buffer: dst } },
        ],
      });
      const encoder = this.device.createCommandEncoder();
      const compute = encoder.beginComputePass();
      compute.setPipeline(this.blurPipeline);
      compute.setBindGroup(0, bindGroup);
      compute.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
      compute.end();
      this.device.queue.submit([encoder.finish()]);
    };
    // Two submits: the params uniform changes between the horizontal and
    // vertical pass, so they cannot share one command buffer.
    pass(bufferA, bufferB, true);
    pass(bufferB, bufferA, false);

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(bufferA, 0, readBuffer, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    bufferA.destroy(); bufferB.destroy(); kernelBuffer.destroy();
    paramsBuffer.destroy(); readBuffer.destroy();
    return out;
  }

  /** Evaluate the small upsampled inverse-DFT region of an interleaved-complex spectrum. */
  async upsampledDft(
    spectrum: Float32Array<ArrayBuffer>,
    rows: number,
    cols: number,
    region: number,
    upsample: number,
    offsetRow: number,
    offsetCol: number,
  ): Promise<Float32Array<ArrayBuffer>> {
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const srcBuffer = this.device.createBuffer({ size: spectrum.byteLength, usage: storage });
    const midBuffer = this.device.createBuffer({ size: region * cols * 2 * 4, usage: storage });
    const dstBuffer = this.device.createBuffer({ size: region * region * 2 * 4, usage: storage });
    const paramsBuffer = this.device.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const readBuffer = this.device.createBuffer({
      size: region * region * 2 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(srcBuffer, 0, spectrum);
    const params = new ArrayBuffer(32);
    new Uint32Array(params, 0, 4).set([rows, cols, region, upsample]);
    new Float32Array(params, 16, 2).set([offsetRow, offsetCol]);
    this.device.queue.writeBuffer(paramsBuffer, 0, params);

    const runPass = (
      pipeline: GPUComputePipeline, src: GPUBuffer, dst: GPUBuffer, groupsX: number, groupsY: number,
    ) => {
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: src } },
          { binding: 2, resource: { buffer: dst } },
        ],
      });
      const encoder = this.device.createCommandEncoder();
      const compute = encoder.beginComputePass();
      compute.setPipeline(pipeline);
      compute.setBindGroup(0, bindGroup);
      compute.dispatchWorkgroups(groupsX, groupsY);
      compute.end();
      this.device.queue.submit([encoder.finish()]);
    };
    runPass(this.rowPassPipeline, srcBuffer, midBuffer, region, Math.ceil(cols / 64));
    runPass(this.colPassPipeline, midBuffer, dstBuffer, Math.ceil(region / 8), Math.ceil(region / 8));

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(dstBuffer, 0, readBuffer, 0, region * region * 2 * 4);
    this.device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    srcBuffer.destroy(); midBuffer.destroy(); dstBuffer.destroy();
    paramsBuffer.destroy(); readBuffer.destroy();
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

/** CPU reflect-boundary separable Gaussian blur, used when WebGPU is unavailable. */
function blurCpu(image: Float32Array, width: number, height: number, sigma: number): Float32Array<ArrayBuffer> {
  const kernel = gaussianKernel1d(sigma);
  const radius = (kernel.length - 1) / 2;
  const horizontal = new Float32Array(image.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let t = -radius; t <= radius; t++) acc += image[y * width + reflectIndex(x + t, width)] * kernel[t + radius];
      horizontal[y * width + x] = acc;
    }
  }
  const out = new Float32Array(image.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let t = -radius; t <= radius; t++) acc += horizontal[reflectIndex(y + t, height) * width + x] * kernel[t + radius];
      out[y * width + x] = acc;
    }
  }
  return out;
}

/**
 * Build the registration image for one slice: median-centered, Gaussian
 * high-passed, Hann windowed. The high pass removes the slowly varying
 * background that would otherwise dominate the correlation peak; the window
 * removes the edge discontinuity that a periodic FFT would read as structure.
 */
async function registrationImage(
  slice: Float32Array, width: number, height: number, gpu: SliceAlignmentGPU | null,
): Promise<Float32Array<ArrayBuffer>> {
  const centered = new Float32Array(slice.length);
  for (let i = 0; i < slice.length; i++) centered[i] = Number.isFinite(slice[i]) ? slice[i] : 0;
  const mid = median(centered);
  for (let i = 0; i < centered.length; i++) centered[i] -= mid;
  const sigma = Math.min(HIGHPASS_SIGMA_PX, Math.max(1.0, Math.min(width, height) / 6.0));
  const blurred = gpu
    ? await gpu.blur(centered, width, height, sigma)
    : blurCpu(centered, width, height, sigma);
  const rowWindow = new Float32Array(height);
  for (let y = 0; y < height; y++) rowWindow[y] = height > 1 ? 0.5 - 0.5 * Math.cos((2 * Math.PI * y) / (height - 1)) : 1;
  const colWindow = new Float32Array(width);
  for (let x = 0; x < width; x++) colWindow[x] = width > 1 ? 0.5 - 0.5 * Math.cos((2 * Math.PI * x) / (width - 1)) : 1;
  const out = new Float32Array(slice.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      out[i] = (centered[i] - blurred[i]) * rowWindow[y] * colWindow[x];
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
  for (let y = 0; y < height; y++) data.set(image.subarray(y * width, (y + 1) * width), y * paddedWidth);
  return { data, width: paddedWidth, height: paddedHeight };
}

/** Least-squares slope/intercept of y versus z, plus the fit R^2. */
function linearFit(z: number[], y: number[]): { slope: number; intercept: number; r2: number } {
  const n = z.length;
  const meanZ = z.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (z[i] - meanZ) * (y[i] - meanY);
    den += (z[i] - meanZ) * (z[i] - meanZ);
  }
  const slope = den > 0 ? num / den : 0;
  const intercept = meanY - slope * meanZ;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const predicted = slope * z[i] + intercept;
    ssRes += (y[i] - predicted) * (y[i] - predicted);
    ssTot += (y[i] - meanY) * (y[i] - meanY);
  }
  return { slope, intercept, r2: ssTot > 0 ? 1 - ssRes / ssTot : 0 };
}

/**
 * Estimate the global row/col shift per slice for one volume.
 *
 * @param volume Contiguous z-major float volume (nz planes of ny x nx).
 * @param fft Shared WebGPU FFT helper; when null the CPU FFT is used.
 * @param device WebGPU device for the blur and refinement passes, or null.
 */
export async function estimateSliceAlignment(
  volume: Float32Array,
  nx: number,
  ny: number,
  nz: number,
  fft: WebGPUFFT | null,
  device: GPUDevice | null,
): Promise<SliceAlignmentEstimate> {
  if (nz < 2) throw new Error("slice alignment requires at least 2 slices");
  const gpu = device ? new SliceAlignmentGPU(device) : null;
  const planeSize = nx * ny;

  const spectra: { real: Float32Array; imag: Float32Array }[] = [];
  const energies: number[] = [];
  let paddedWidth = nx;
  let paddedHeight = ny;
  for (let z = 0; z < nz; z++) {
    const prepared = await registrationImage(volume.subarray(z * planeSize, (z + 1) * planeSize), nx, ny, gpu);
    let energy = 0;
    for (let i = 0; i < prepared.length; i++) energy += prepared[i] * prepared[i];
    energies.push(energy);
    const padded = padToPow2(prepared, nx, ny);
    paddedWidth = padded.width;
    paddedHeight = padded.height;
    const imag = new Float32Array(padded.data.length);
    if (fft) {
      spectra.push(await fft.fft2D(padded.data, imag, padded.width, padded.height, false));
    } else {
      const real = Float32Array.from(padded.data);
      fft2d(real, imag, padded.width, padded.height, false);
      spectra.push({ real, imag });
    }
  }

  const region = Math.ceil(UPSAMPLE_FACTOR * DFT_REGION_FACTOR);
  const dftShift = Math.trunc(region / 2.0);
  const adjacent: number[][] = [];
  const quality: number[] = [];
  for (let z = 0; z < nz - 1; z++) {
    const ref = spectra[z];
    const mov = spectra[z + 1];
    if (energies[z] === 0 || energies[z + 1] === 0) {
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
    if (fft) {
      const inverse = await fft.fft2D(corrReal, corrImag, paddedWidth, paddedHeight, true);
      corrReal.set(inverse.real);
      corrImag.set(inverse.imag);
    } else {
      fft2d(corrReal, corrImag, paddedWidth, paddedHeight, true);
    }
    let peakIndex = 0;
    let peakValue = -1;
    for (let i = 0; i < count; i++) {
      const magnitude = corrReal[i] * corrReal[i] + corrImag[i] * corrImag[i];
      if (magnitude > peakValue) { peakValue = magnitude; peakIndex = i; }
    }
    let shiftRow = Math.floor(peakIndex / paddedWidth);
    let shiftCol = peakIndex % paddedWidth;
    // Correlation indices past the midpoint are negative shifts wrapped around.
    if (shiftRow > Math.trunc(paddedHeight / 2)) shiftRow -= paddedHeight;
    if (shiftCol > Math.trunc(paddedWidth / 2)) shiftCol -= paddedWidth;

    let refinedRow = Math.round(shiftRow * UPSAMPLE_FACTOR) / UPSAMPLE_FACTOR;
    let refinedCol = Math.round(shiftCol * UPSAMPLE_FACTOR) / UPSAMPLE_FACTOR;
    // Refine against conj(product) and conjugate back, matching the Python
    // path; only the peak location is used, so the conjugation is about
    // keeping the sampled grid identical rather than the values themselves.
    const conjugated = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      conjugated[i * 2] = productReal[i];
      conjugated[i * 2 + 1] = -productImag[i];
    }
    const offsetRow = dftShift - refinedRow * UPSAMPLE_FACTOR;
    const offsetCol = dftShift - refinedCol * UPSAMPLE_FACTOR;
    const refined = gpu
      ? await gpu.upsampledDft(
        conjugated, paddedHeight, paddedWidth, region, UPSAMPLE_FACTOR, offsetRow, offsetCol,
      )
      : upsampledDftCpu(
        conjugated, paddedHeight, paddedWidth, region, UPSAMPLE_FACTOR, offsetRow, offsetCol,
      );
    let bestIndex = 0;
    let bestValue = -1;
    for (let i = 0; i < region * region; i++) {
      const magnitude = refined[i * 2] * refined[i * 2] + refined[i * 2 + 1] * refined[i * 2 + 1];
      if (magnitude > bestValue) { bestValue = magnitude; bestIndex = i; }
    }
    refinedRow += (Math.floor(bestIndex / region) - dftShift) / UPSAMPLE_FACTOR;
    refinedCol += ((bestIndex % region) - dftShift) / UPSAMPLE_FACTOR;
    adjacent.push([refinedRow, refinedCol]);
    const norm = Math.sqrt(energies[z] * energies[z + 1]);
    quality.push(norm > 0 ? Math.sqrt(peakValue) / norm : 0);
  }

  const cumulative: number[][] = [[0, 0]];
  for (let i = 0; i < adjacent.length; i++) {
    const previous = cumulative[i];
    cumulative.push([previous[0] + adjacent[i][0], previous[1] + adjacent[i][1]]);
  }
  const z = Array.from({ length: nz }, (_, i) => i);
  const rowFit = linearFit(z, cumulative.map((entry) => entry[0]));
  const colFit = linearFit(z, cumulative.map((entry) => entry[1]));
  return {
    rowShiftPxPerSlice: rowFit.slope,
    colShiftPxPerSlice: colFit.slope,
    adjacentShiftPx: adjacent,
    cumulativeShiftPx: cumulative,
    fitR2: { row: rowFit.r2, col: colFit.r2 },
    quality,
    backend: gpu ? "webgpu" : "cpu",
  };
}
