/// <reference types="@webgpu/types" />

/**
 * Display 2D FFT. WebGPUFFT runs on a hardware adapter; CPUFFT runs the scalar
 * reference (same radix-2 algorithm and zero padding) when the browser has no
 * WebGPU. Handles non-power-of-2 dimensions via zero-padding.
 */
import { getHardwareGPUDevice, onGPULost, requireHardwareGPUDevice, type RenderPath } from "./device";
import { FFT_2D_SHADER } from "./fftShader";

// The GPU device is owned by device.ts. Every widget and the compute engine must
// share ONE device: a bind group built on one device and submitted on another
// throws "BindGroupLayout is associated with [Device], cannot be used with
// [Device]" and crashes the tab's GPU process. These re-exports keep the FFT
// module's public device accessors pointing at that one device.
export { getGPUDevice, getGPUInfo } from "./device";

// ============================================================================
// CPU FFT reference
// ============================================================================

export function nextPow2(n: number): number { return Math.pow(2, Math.ceil(Math.log2(n))); }

function fft1d(real: Float32Array, imag: Float32Array, inverse: boolean = false) {
  const n = real.length;
  if (n <= 1) return;
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) { [real[i], real[j]] = [real[j], real[i]]; [imag[i], imag[j]] = [imag[j], imag[i]]; }
    let k = n >> 1;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (sign * 2 * Math.PI) / len;
    const wReal = Math.cos(angle), wImag = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curReal = 1, curImag = 0;
      for (let k = 0; k < halfLen; k++) {
        const evenIdx = i + k, oddIdx = i + k + halfLen;
        const tReal = curReal * real[oddIdx] - curImag * imag[oddIdx];
        const tImag = curReal * imag[oddIdx] + curImag * real[oddIdx];
        real[oddIdx] = real[evenIdx] - tReal; imag[oddIdx] = imag[evenIdx] - tImag;
        real[evenIdx] += tReal; imag[evenIdx] += tImag;
        const newReal = curReal * wReal - curImag * wImag;
        curImag = curReal * wImag + curImag * wReal; curReal = newReal;
      }
    }
  }
  if (inverse) { for (let i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; } }
}

export function fft2d(real: Float32Array, imag: Float32Array, width: number, height: number, inverse: boolean = false) {
  const paddedW = nextPow2(width), paddedH = nextPow2(height);
  const needsPadding = paddedW !== width || paddedH !== height;
  let workReal: Float32Array, workImag: Float32Array;
  if (needsPadding) {
    workReal = new Float32Array(paddedW * paddedH); workImag = new Float32Array(paddedW * paddedH);
    for (let row = 0; row < height; row++) for (let col = 0; col < width; col++) {
      workReal[row * paddedW + col] = real[row * width + col]; workImag[row * paddedW + col] = imag[row * width + col];
    }
  } else { workReal = real; workImag = imag; }
  const rowReal = new Float32Array(paddedW), rowImag = new Float32Array(paddedW);
  for (let row = 0; row < paddedH; row++) {
    const offset = row * paddedW;
    for (let col = 0; col < paddedW; col++) { rowReal[col] = workReal[offset + col]; rowImag[col] = workImag[offset + col]; }
    fft1d(rowReal, rowImag, inverse);
    for (let col = 0; col < paddedW; col++) { workReal[offset + col] = rowReal[col]; workImag[offset + col] = rowImag[col]; }
  }
  const colReal = new Float32Array(paddedH), colImag = new Float32Array(paddedH);
  for (let col = 0; col < paddedW; col++) {
    for (let row = 0; row < paddedH; row++) { colReal[row] = workReal[row * paddedW + col]; colImag[row] = workImag[row * paddedW + col]; }
    fft1d(colReal, colImag, inverse);
    for (let row = 0; row < paddedH; row++) { workReal[row * paddedW + col] = colReal[row]; workImag[row * paddedW + col] = colImag[row]; }
  }
  if (needsPadding) {
    for (let row = 0; row < height; row++) for (let col = 0; col < width; col++) {
      real[row * width + col] = workReal[row * paddedW + col]; imag[row * width + col] = workImag[row * paddedW + col];
    }
  }
}

export function fftshift(data: Float32Array, width: number, height: number): void {
  const halfW = width >> 1, halfH = height >> 1;
  const shifted = new Float32Array(width * height);
  for (let row = 0; row < height; row++) for (let col = 0; col < width; col++) {
    shifted[((row + halfH) % height) * width + ((col + halfW) % width)] = data[row * width + col];
  }
  data.set(shifted);
}

// ============================================================================
// CPU FFT Web Worker: fft2d + fftshift + magnitude off the main thread
// ============================================================================

// Build worker source by stringifying the same fft1d/fft2d/fftshift defined
// above. Single source of truth: fix a bug once, both paths get it. Pure
// functions only (no module-state closures), so .toString() captures the full
// behavior. Use Function.name in the onmessage body so minified names still
// match (esbuild may rename `fft2d` -> `a`; the .name property tracks rename).
const FFT_WORKER_CODE = `
${nextPow2.toString()}
${fft1d.toString()}
${fft2d.toString()}
${fftshift.toString()}
self.onmessage = function(event) {
  const job = event.data;
  ${fft2d.name}(job.real, job.imag, job.width, job.height, job.inverse);
  ${fftshift.name}(job.real, job.width, job.height);
  ${fftshift.name}(job.imag, job.width, job.height);
  const n = job.real.length, mag = new Float32Array(n);
  for (let i = 0; i < n; i++) mag[i] = Math.sqrt(job.real[i]*job.real[i] + job.imag[i]*job.imag[i]);
  self.postMessage({ id: job.id, magnitude: mag, real: job.real, imag: job.imag }, [mag.buffer, job.real.buffer, job.imag.buffer]);
};
`;

let _fftWorker: Worker | null = null;
const _fftCallbacks = new Map<number, (data: { magnitude: Float32Array; real: Float32Array; imag: Float32Array }) => void>();
let _fftWorkerId = 0;

function getFFTWorker(): Worker {
  if (!_fftWorker) {
    const blob = new Blob([FFT_WORKER_CODE], { type: 'application/javascript' });
    _fftWorker = new Worker(URL.createObjectURL(blob));
    _fftWorker.onmessage = (event: MessageEvent) => {
      const resolve = _fftCallbacks.get(event.data.id);
      if (resolve) {
        _fftCallbacks.delete(event.data.id);
        resolve(event.data);
      }
    };
  }
  return _fftWorker;
}

/**
 * CPU FFT in a Web Worker: fft2d + fftshift + computeMagnitude off the main thread.
 * Transfers Float32Arrays to the worker (zero-copy) so the main thread is never blocked.
 * The input arrays become detached after this call; pass copies if you need to keep them.
 */
export function fft2dAsync(
  real: Float32Array, imag: Float32Array,
  width: number, height: number,
  inverse: boolean = false,
): Promise<{ magnitude: Float32Array; real: Float32Array; imag: Float32Array }> {
  const worker = getFFTWorker();
  const id = ++_fftWorkerId;
  return new Promise((resolve) => {
    _fftCallbacks.set(id, resolve);
    worker.postMessage(
      { id, real, imag, width, height, inverse },
      [real.buffer, imag.buffer],
    );
  });
}

// ============================================================================
// WebGPU FFT (compute shader, GPU-resident)
// ============================================================================

/** The FFT calls display code makes; WebGPUFFT and CPUFFT both provide them. */
export interface DisplayFFT {
  readonly path: RenderPath;
  fft2D(realData: Float32Array, imagData: Float32Array, width: number, height: number, inverse?: boolean): Promise<{ real: Float32Array, imag: Float32Array }>;
  fft2DBatch(images: { real: Float32Array; imag: Float32Array }[], width: number, height: number): Promise<{ real: Float32Array; imag: Float32Array }[]>;
}

/** Scalar FFT with the WebGPUFFT interface, for browsers without WebGPU. */
export class CPUFFT implements DisplayFFT {
  readonly path = "CPU" as const;
  async fft2D(realData: Float32Array, imagData: Float32Array, width: number, height: number, inverse: boolean = false): Promise<{ real: Float32Array, imag: Float32Array }> {
    const real = Float32Array.from(realData.subarray(0, width * height));
    const imag = Float32Array.from(imagData.subarray(0, width * height));
    fft2d(real, imag, width, height, inverse);
    return { real, imag };
  }
  async fft2DBatch(images: { real: Float32Array; imag: Float32Array }[], width: number, height: number): Promise<{ real: Float32Array; imag: Float32Array }[]> {
    return Promise.all(images.map(image => this.fft2D(image.real, image.imag, width, height)));
  }
}

/**
 * Zero-pad a width x height complex image into a paddedWidth x paddedHeight frame
 * of interleaved (real, imag) floats, the layout the FFT shader reads.
 */
function interleaveComplex(
  real: Float32Array, imag: Float32Array,
  width: number, height: number, paddedWidth: number, paddedHeight: number,
): Float32Array<ArrayBuffer> {
  const complexData = new Float32Array(paddedWidth * paddedHeight * 2);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const target = (row * paddedWidth + col) * 2;
      complexData[target] = real[row * width + col];
      complexData[target + 1] = imag[row * width + col];
    }
  }
  return complexData;
}

/** Crop the width x height corner of a padded interleaved spectrum back into real and imag planes. */
function deinterleaveComplex(
  complexData: Float32Array, width: number, height: number, paddedWidth: number,
): { real: Float32Array; imag: Float32Array } {
  const real = new Float32Array(width * height), imag = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const source = (row * paddedWidth + col) * 2;
      real[row * width + col] = complexData[source];
      imag[row * width + col] = complexData[source + 1];
    }
  }
  return { real, imag };
}

export class WebGPUFFT implements DisplayFFT {
  readonly path = "WebGPU" as const;
  private device: GPUDevice;
  private pipelines2D: { bitReverseRows: GPUComputePipeline; bitReverseCols: GPUComputePipeline; butterflyRows: GPUComputePipeline; butterflyCols: GPUComputePipeline; normalize: GPUComputePipeline } | null = null;
  private initialized = false;
  constructor(device: GPUDevice) { this.device = device; }
  async init(): Promise<void> {
    if (this.initialized) return;
    const module2D = this.device.createShaderModule({ code: FFT_2D_SHADER });
    this.pipelines2D = {
      bitReverseRows: this.device.createComputePipeline({ layout: 'auto', compute: { module: module2D, entryPoint: 'bitReverseRows' } }),
      bitReverseCols: this.device.createComputePipeline({ layout: 'auto', compute: { module: module2D, entryPoint: 'bitReverseCols' } }),
      butterflyRows: this.device.createComputePipeline({ layout: 'auto', compute: { module: module2D, entryPoint: 'butterflyRows' } }),
      butterflyCols: this.device.createComputePipeline({ layout: 'auto', compute: { module: module2D, entryPoint: 'butterflyCols' } }),
      normalize: this.device.createComputePipeline({ layout: 'auto', compute: { module: module2D, entryPoint: 'normalize2D' } })
    };
    this.initialized = true;
  }

  /**
   * Dispatch every pass of one in-place 2D FFT on `dataBuffer`: bit reversal and
   * log2(width) butterfly stages along rows, the same along columns, then the 1/N
   * normalization for an inverse transform. Each pass is its own submit because
   * the shared params uniform changes between passes.
   */
  private dispatchFFT2D(
    dataBuffer: GPUBuffer, paramsBuffer: GPUBuffer, width: number, height: number, inverse: boolean,
  ): void {
    const log2Width = Math.log2(width), log2Height = Math.log2(height);
    const workgroupsX = Math.ceil(width / 16), workgroupsY = Math.ceil(height / 16);
    const pipelines = this.pipelines2D!;
    const runPass = (pipeline: GPUComputePipeline) => {
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: paramsBuffer } }, { binding: 1, resource: { buffer: dataBuffer } }],
      });
      const encoder = this.device.createCommandEncoder(); const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupsX, workgroupsY); pass.end();
      this.device.queue.submit([encoder.finish()]);
    };
    const params = new ArrayBuffer(24);
    const paramsU32 = new Uint32Array(params); const paramsF32 = new Float32Array(params);
    paramsU32[0] = width; paramsU32[1] = height; paramsU32[2] = log2Width; paramsU32[3] = 0;
    paramsF32[4] = inverse ? 1.0 : -1.0; paramsU32[5] = 1;
    this.device.queue.writeBuffer(paramsBuffer, 0, params); runPass(pipelines.bitReverseRows);
    for (let stage = 0; stage < log2Width; stage++) {
      paramsU32[3] = stage; this.device.queue.writeBuffer(paramsBuffer, 0, params);
      runPass(pipelines.butterflyRows);
    }
    paramsU32[2] = log2Height; paramsU32[3] = 0; paramsU32[5] = 0;
    this.device.queue.writeBuffer(paramsBuffer, 0, params); runPass(pipelines.bitReverseCols);
    for (let stage = 0; stage < log2Height; stage++) {
      paramsU32[3] = stage; this.device.queue.writeBuffer(paramsBuffer, 0, params);
      runPass(pipelines.butterflyCols);
    }
    if (inverse) runPass(pipelines.normalize);
  }

  async fft2D(realData: Float32Array, imagData: Float32Array, width: number, height: number, inverse: boolean = false): Promise<{ real: Float32Array, imag: Float32Array }> {
    await this.init();
    const paddedWidth = nextPow2(width), paddedHeight = nextPow2(height);
    const complexData = interleaveComplex(realData, imagData, width, height, paddedWidth, paddedHeight);
    const dataBuffer = this.device.createBuffer({ size: complexData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(dataBuffer, 0, complexData);
    const paramsBuffer = this.device.createBuffer({ size: 24, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readBuffer = this.device.createBuffer({ size: complexData.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.dispatchFFT2D(dataBuffer, paramsBuffer, paddedWidth, paddedHeight, inverse);
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, complexData.byteLength);
    this.device.queue.submit([encoder.finish()]); await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0)); readBuffer.unmap();
    dataBuffer.destroy(); paramsBuffer.destroy(); readBuffer.destroy();
    return deinterleaveComplex(result, width, height, paddedWidth);
  }

  /**
   * In-place 2D FFT on a caller-owned GPU buffer of interleaved complex data.
   *
   * `fft2D` owns its buffers: it uploads, dispatches, reads back, and destroys
   * on every call, so a multi-stage pipeline built on it round-trips the whole
   * array to the CPU at every stage boundary. This entry point does only the
   * dispatches, letting a caller chain FFTs with other compute passes and read
   * back once at the end. Width and height must already be powers of two and
   * the buffer must hold width*height*2 floats.
   */
  async fft2DResident(
    buffer: GPUBuffer, width: number, height: number, inverse: boolean = false,
  ): Promise<void> {
    await this.init();
    if (width !== nextPow2(width) || height !== nextPow2(height)) {
      throw new Error(`fft2DResident needs power-of-two dims, got ${width}x${height}`);
    }
    const paramsBuffer = this.device.createBuffer({
      size: 24, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.dispatchFFT2D(buffer, paramsBuffer, width, height, inverse);
    paramsBuffer.destroy();
  }

  /**
   * Batched 2D FFT: compute N forward FFTs with pipelined GPU submissions.
   * All images must have the same dimensions. Each image gets its own
   * submit (required because the params uniform changes per-pass), but
   * all readbacks are batched into a single Promise.all at the end.
   */
  async fft2DBatch(
    images: { real: Float32Array; imag: Float32Array }[],
    width: number, height: number,
  ): Promise<{ real: Float32Array; imag: Float32Array }[]> {
    await this.init();
    if (images.length === 0) return [];
    const paddedWidth = nextPow2(width), paddedHeight = nextPow2(height);
    const byteSize = paddedWidth * paddedHeight * 2 * 4;
    // One params buffer serves every image: each pass is submitted before the next write.
    const paramsBuffer = this.device.createBuffer({ size: 24, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readBuffers: GPUBuffer[] = [];
    const dataBuffers: GPUBuffer[] = [];
    for (const image of images) {
      const complexData = interleaveComplex(image.real, image.imag, width, height, paddedWidth, paddedHeight);
      const dataBuffer = this.device.createBuffer({ size: byteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(dataBuffer, 0, complexData);
      dataBuffers.push(dataBuffer);
      const readBuffer = this.device.createBuffer({ size: byteSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      readBuffers.push(readBuffer);
      this.dispatchFFT2D(dataBuffer, paramsBuffer, paddedWidth, paddedHeight, false);
      const copyEncoder = this.device.createCommandEncoder();
      copyEncoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, byteSize);
      this.device.queue.submit([copyEncoder.finish()]);
    }
    // One sync point for all images.
    await Promise.all(readBuffers.map(buffer => buffer.mapAsync(GPUMapMode.READ)));
    const results = readBuffers.map((readBuffer, i) => {
      const result = new Float32Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();
      dataBuffers[i].destroy();
      readBuffer.destroy();
      return deinterleaveComplex(result, width, height, paddedWidth);
    });
    paramsBuffer.destroy();
    return results;
  }

  destroy(): void { this.initialized = false; }
}

// ============================================================================
// FFT pre-processing helpers
// ============================================================================

/**
 * Apply 2D Hann window in-place to reduce spectral leakage in ROI FFT.
 *
 * When an ROI is cropped from an image, the sharp rectangular boundary acts as
 * a rect window whose sinc sidelobes produce streak artifacts in the FFT,
 * obscuring real spectral features (Bragg spots, lattice frequencies).
 * The Hann window smoothly tapers data to zero at all edges, suppressing
 * sidelobes by ~31 dB at the cost of a slightly wider main lobe.
 *
 * Separable: window2D = outer(hann_h, hann_w), applied as element-wise multiply.
 * Symmetric formula: w(i) = 0.5*(1 - cos(2πi/(N-1))), matching np.hanning:
 * both endpoints are exactly zero for seamless transition to zero-padded regions.
 * (Periodic variant ÷N is for overlapping STFT windows, not for zero-padding.)
 *
 * IMPORTANT: Must be called on the crop at its native dimensions BEFORE
 * zero-padding to power-of-2. Window-then-pad ensures no discontinuity at the
 * crop/pad boundary. Pad-then-window applies the wrong taper and reintroduces
 * leakage. Validated against np.hanning in test_widget_show2d.py.
 */
export function applyHannWindow2D(data: Float32Array, width: number, height: number): void {
  const hannW = new Float32Array(width);
  const hannH = new Float32Array(height);
  const wDenom = width > 1 ? width - 1 : 1;
  const hDenom = height > 1 ? height - 1 : 1;
  for (let i = 0; i < width; i++) hannW[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / wDenom));
  for (let i = 0; i < height; i++) hannH[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / hDenom));
  for (let row = 0; row < height; row++) {
    const rowWeight = hannH[row];
    const offset = row * width;
    for (let col = 0; col < width; col++) data[offset + col] *= rowWeight * hannW[col];
  }
}

// ============================================================================
// FFT post-processing helpers
// ============================================================================

/** Compute magnitude from complex FFT output: sqrt(real² + imag²). */
export function computeMagnitude(
  real: Float32Array,
  imag: Float32Array,
  out: Float32Array = new Float32Array(real.length),
): Float32Array {
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
  }
  return out;
}

/** Return the complete shifted magnitude of a complex Fourier grid.
 *
 * The output dimensions are exactly ``width × height``; this helper never
 * crops a padded spectrum. Set ``logScale`` for display-oriented ``log1p``
 * magnitude while retaining the same Fourier coordinates.
 */
export function shiftedMagnitude(
  real: Float32Array,
  imag: Float32Array,
  width: number,
  height: number,
  logScale = false,
): Float32Array {
  if (real.length < width * height || imag.length < width * height) {
    throw new Error("shiftedMagnitude input is shorter than width * height");
  }
  const magnitude = computeMagnitude(real, imag, new Float32Array(width * height));
  fftshift(magnitude, width, height);
  if (logScale) {
    for (let index = 0; index < magnitude.length; index++) {
      magnitude[index] = Math.log1p(magnitude[index]);
    }
  }
  return magnitude;
}

type ReciprocalCoordinates = {
  rowFrequency: number;
  columnFrequency: number;
  spatialFrequency: number;
  dSpacing: number | null;
};

/** Convert an offset from the shifted FFT center into reciprocal coordinates.
 * Dimensions describe the Fourier grid and sampling follows ``(row, column)``.
 */
export function reciprocalCoordinatesFromShiftedOffset(
  rowOffset: number,
  columnOffset: number,
  rows: number,
  columns: number,
  rowSampling: number,
  columnSampling: number,
): ReciprocalCoordinates {
  const rowFrequency = rowOffset / (rows * rowSampling);
  const columnFrequency = columnOffset / (columns * columnSampling);
  const spatialFrequency = Math.hypot(rowFrequency, columnFrequency);
  return {
    rowFrequency,
    columnFrequency,
    spatialFrequency,
    dSpacing: spatialFrequency > 0 ? 1 / spatialFrequency : null,
  };
}

/** Find the strongest local FFT-magnitude pixel and refine it by a 3×3 centroid.
 * Non-finite samples are ignored and equal maxima keep row-major first-hit order.
 */
export function findFFTPeak(
  magnitude: Float32Array,
  width: number,
  height: number,
  column: number,
  row: number,
  radius: number,
): { row: number; col: number } {
  if (width < 1 || height < 1 || magnitude.length < width * height) {
    return { row: 0, col: 0 };
  }
  const baseColumn = Math.max(0, Math.min(width - 1, Math.floor(column)));
  const baseRow = Math.max(0, Math.min(height - 1, Math.floor(row)));
  const column0 = Math.max(0, baseColumn - radius);
  const row0 = Math.max(0, baseRow - radius);
  const column1 = Math.min(width - 1, baseColumn + radius);
  const row1 = Math.min(height - 1, baseRow + radius);
  let bestColumn = baseColumn;
  let bestRow = baseRow;
  let bestValue = -Infinity;
  for (let candidateRow = row0; candidateRow <= row1; candidateRow++) {
    for (let candidateColumn = column0; candidateColumn <= column1; candidateColumn++) {
      const value = magnitude[candidateRow * width + candidateColumn];
      if (Number.isFinite(value) && value > bestValue) {
        bestValue = value;
        bestColumn = candidateColumn;
        bestRow = candidateRow;
      }
    }
  }
  let weight = 0;
  let weightedColumn = 0;
  let weightedRow = 0;
  for (let candidateRow = Math.max(0, bestRow - 1); candidateRow <= Math.min(height - 1, bestRow + 1); candidateRow++) {
    for (let candidateColumn = Math.max(0, bestColumn - 1); candidateColumn <= Math.min(width - 1, bestColumn + 1); candidateColumn++) {
      const value = magnitude[candidateRow * width + candidateColumn];
      if (!Number.isFinite(value) || value <= 0) continue;
      weight += value;
      weightedColumn += candidateColumn * value;
      weightedRow += candidateRow * value;
    }
  }
  return weight > 0
    ? { row: weightedRow / weight, col: weightedColumn / weight }
    : { row: bestRow, col: bestColumn };
}

/** Mask DC component (center pixel) and return 99.9% percentile-clipped range. Mutates `mag`. */
export function autoEnhanceFFT(
  mag: Float32Array, width: number, height: number,
): { min: number; max: number } {
  const len = mag.length;
  if (len === 0) return { min: 0, max: 0 };
  const centerIdx = Math.floor(height / 2) * width + Math.floor(width / 2);
  const neighbors = [
    mag[Math.max(0, centerIdx - 1)],
    mag[Math.min(mag.length - 1, centerIdx + 1)],
    mag[Math.max(0, centerIdx - width)],
    mag[Math.min(mag.length - 1, centerIdx + width)],
  ];
  mag[centerIdx] = neighbors.reduce((a, b) => a + b, 0) / 4;
  // Use two O(n) histogram passes instead of an O(n log n) sort. One linear
  // histogram is not enough for microscopy FFTs: a 1e9 DC neighborhood can
  // put every useful 1e3-1e6 value in bin zero, making the old fallback choose
  // the absolute maximum and paint the spectrum black.
  let dMin = Infinity, dMax = -Infinity;
  for (let i = 0; i < len; i++) {
    const value = mag[i];
    if (value < dMin) dMin = value;
    if (value > dMax) dMax = value;
  }
  if (dMin === dMax) return { min: dMin, max: dMax };
  const NUM_BINS = 1024;
  const bins = new Uint32Array(NUM_BINS);
  const range = dMax - dMin;
  const binIndex = (value: number, lo: number, span: number) => Math.min(
    NUM_BINS - 1,
    Math.max(0, Math.floor(((value - lo) / span) * NUM_BINS)),
  );
  for (let i = 0; i < len; i++) bins[binIndex(mag[i], dMin, range)]++;
  // Find the coarse bin containing the 99.9th percentile.
  const target = Math.ceil(len * 0.999);
  let cumSum = 0;
  let coarseBin = NUM_BINS - 1;
  let countBefore = 0;
  for (let i = 0; i < NUM_BINS; i++) {
    cumSum += bins[i];
    if (cumSum >= target) {
      coarseBin = i;
      countBefore = cumSum - bins[i];
      break;
    }
  }
  const coarseMin = dMin + (coarseBin / NUM_BINS) * range;
  const coarseMax = coarseBin === NUM_BINS - 1
    ? dMax
    : dMin + ((coarseBin + 1) / NUM_BINS) * range;
  const coarseSpan = coarseMax - coarseMin;
  if (!(coarseSpan > 0)) return { min: dMin, max: dMax };

  // Re-bin only the selected coarse interval. This resolves the useful range
  // even when the first interval spans several million intensity units.
  const refinedBins = new Uint32Array(NUM_BINS);
  for (let i = 0; i < len; i++) {
    const value = mag[i];
    const inCoarseBin = value >= coarseMin && (
      coarseBin === NUM_BINS - 1 ? value <= coarseMax : value < coarseMax
    );
    if (inCoarseBin) refinedBins[binIndex(value, coarseMin, coarseSpan)]++;
  }
  const refinedTarget = Math.max(1, target - countBefore);
  let refinedCumSum = 0;
  let refinedBin = NUM_BINS - 1;
  for (let i = 0; i < NUM_BINS; i++) {
    refinedCumSum += refinedBins[i];
    if (refinedCumSum >= refinedTarget) {
      refinedBin = i;
      break;
    }
  }
  const percentileMax = Math.min(
    dMax,
    coarseMin + ((refinedBin + 1) / NUM_BINS) * coarseSpan,
  );
  return { min: dMin, max: percentileMax };
}

// ============================================================================
// Singleton
// ============================================================================

let gpuFFT: WebGPUFFT | null = null;
// A WebGPUFFT on a lost device can never dispatch again; the next caller rebuilds it.
onGPULost(() => { gpuFFT = null; });

let cpuFFT: CPUFFT | null = null;

/** The display FFT: WebGPU on a hardware adapter, the scalar reference otherwise. */
export async function getDisplayFFT(): Promise<DisplayFFT> {
  if (await getHardwareGPUDevice()) return requireWebGPUFFT("Browser FFT");
  cpuFFT ??= new CPUFFT();
  return cpuFFT;
}

/** Return the shared hardware FFT or fail without running a CPU/Worker FFT. */
export async function requireWebGPUFFT(operation = "Browser FFT"): Promise<WebGPUFFT> {
  if (gpuFFT) return gpuFFT;
  const device = await requireHardwareGPUDevice(operation);
  try {
    gpuFFT = new WebGPUFFT(device);
    await gpuFFT.init();
    return gpuFFT;
  } catch (error) {
    gpuFFT = null;
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`${operation} could not initialize WebGPU FFT: ${detail}`);
  }
}
