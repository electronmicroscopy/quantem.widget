/// <reference types="@webgpu/types" />

/**
 * Display 2D FFT. WebGPUFFT runs on a hardware adapter; CPUFFT runs the scalar
 * reference (the same radix-2 algorithm) when the browser has no WebGPU. Both
 * take power-of-two sizes: callers zero-pad an image first, because a padded
 * spectrum cropped back to the image size is not the image's DFT.
 */
import { getHardwareGPUDevice, onGPULost, requireHardwareGPUDevice, type RenderPath } from "./device";
import { FFT_2D_SHADER } from "./fftShader";

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

/** Throw unless both sizes are powers of two, the only sizes the radix-2 FFT transforms. */
export function requirePow2(width: number, height: number): void {
  if (width !== nextPow2(width) || height !== nextPow2(height)) {
    throw new Error(`The display FFT needs power-of-two sizes; got ${width}x${height}. Zero-pad the image first.`);
  }
}

/** In-place 2D FFT (inverse when `inverse`, scaled by 1/N) of a power-of-two width x height image. */
export function fft2d(real: Float32Array, imag: Float32Array, width: number, height: number, inverse: boolean = false) {
  requirePow2(width, height);
  const rowReal = new Float32Array(width), rowImag = new Float32Array(width);
  for (let row = 0; row < height; row++) {
    const offset = row * width;
    for (let col = 0; col < width; col++) { rowReal[col] = real[offset + col]; rowImag[col] = imag[offset + col]; }
    fft1d(rowReal, rowImag, inverse);
    for (let col = 0; col < width; col++) { real[offset + col] = rowReal[col]; imag[offset + col] = rowImag[col]; }
  }
  const colReal = new Float32Array(height), colImag = new Float32Array(height);
  for (let col = 0; col < width; col++) {
    for (let row = 0; row < height; row++) { colReal[row] = real[row * width + col]; colImag[row] = imag[row * width + col]; }
    fft1d(colReal, colImag, inverse);
    for (let row = 0; row < height; row++) { real[row * width + col] = colReal[row]; imag[row * width + col] = colImag[row]; }
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
${requirePow2.toString()}
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
  /** fft2D on interleaved (real, imag) floats, returning the interleaved spectrum; no split planes. */
  fft2DInterleaved(complexData: Float32Array, width: number, height: number, inverse?: boolean): Promise<Float32Array>;
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
  async fft2DInterleaved(complexData: Float32Array, width: number, height: number, inverse: boolean = false): Promise<Float32Array> {
    const { real, imag } = deinterleaveComplex(complexData.subarray(0, width * height * 2));
    fft2d(real, imag, width, height, inverse);
    return interleaveComplex(real, imag, width * height);
  }
}

/** Interleaved (real, imag) floats, the layout the FFT shader reads. */
function interleaveComplex(real: Float32Array, imag: Float32Array, count: number): Float32Array<ArrayBuffer> {
  const complexData = new Float32Array(count * 2);
  for (let index = 0; index < count; index++) {
    complexData[2 * index] = real[index];
    complexData[2 * index + 1] = imag[index];
  }
  return complexData;
}

/** Real and imag planes of an interleaved spectrum. */
function deinterleaveComplex(complexData: Float32Array): { real: Float32Array; imag: Float32Array } {
  const count = complexData.length / 2;
  const real = new Float32Array(count), imag = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    real[index] = complexData[2 * index];
    imag[index] = complexData[2 * index + 1];
  }
  return { real, imag };
}

export class WebGPUFFT implements DisplayFFT {
  readonly path = "WebGPU" as const;
  private device: GPUDevice;
  private pipelines2D: { bitReverseRows: GPUComputePipeline; bitReverseCols: GPUComputePipeline; butterflyRows: GPUComputePipeline; butterflyCols: GPUComputePipeline; normalize: GPUComputePipeline };
  constructor(device: GPUDevice) {
    this.device = device;
    const module2D = device.createShaderModule({ code: FFT_2D_SHADER });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ layout: 'auto', compute: { module: module2D, entryPoint } });
    this.pipelines2D = {
      bitReverseRows: pipeline('bitReverseRows'), bitReverseCols: pipeline('bitReverseCols'),
      butterflyRows: pipeline('butterflyRows'), butterflyCols: pipeline('butterflyCols'), normalize: pipeline('normalize2D'),
    };
  }

  /**
   * Uniform parameters of every pass of one 2D FFT, one aligned slot per pass:
   * bit reversal and log2(width) butterfly stages along rows, the same along
   * columns, then the 1/N normalization of an inverse transform. Images of one
   * size share it, so a batch writes it once.
   */
  private fftPassParams(width: number, height: number, inverse: boolean): { buffer: GPUBuffer; passes: { pipeline: GPUComputePipeline; offset: number }[] } {
    const pipelines = this.pipelines2D;
    const log2Width = Math.log2(width), log2Height = Math.log2(height);
    const passes: { pipeline: GPUComputePipeline; rows: boolean; stage: number }[] = [
      { pipeline: pipelines.bitReverseRows, rows: true, stage: 0 },
      ...Array.from({ length: log2Width }, (_, stage) => ({ pipeline: pipelines.butterflyRows, rows: true, stage })),
      { pipeline: pipelines.bitReverseCols, rows: false, stage: 0 },
      ...Array.from({ length: log2Height }, (_, stage) => ({ pipeline: pipelines.butterflyCols, rows: false, stage })),
      ...(inverse ? [{ pipeline: pipelines.normalize, rows: false, stage: 0 }] : []),
    ];
    const slot = Math.max(24, this.device.limits.minUniformBufferOffsetAlignment);
    const params = new ArrayBuffer(slot * passes.length);
    passes.forEach(({ rows, stage }, index) => {
      const paramsU32 = new Uint32Array(params, index * slot, 6), paramsF32 = new Float32Array(params, index * slot, 6);
      paramsU32[0] = width; paramsU32[1] = height; paramsU32[2] = rows ? log2Width : log2Height; paramsU32[3] = stage;
      paramsF32[4] = inverse ? 1.0 : -1.0; paramsU32[5] = rows ? 1 : 0;
    });
    const buffer = this.device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, params);
    return { buffer, passes: passes.map(({ pipeline }, index) => ({ pipeline, offset: index * slot })) };
  }

  /**
   * Record every pass of one in-place 2D FFT on `dataBuffer` into one compute
   * pass. Dispatches in a pass run in order and each sees the previous one's
   * writes, so the whole transform is one submit instead of one per stage.
   */
  private encodeFFT2D(
    encoder: GPUCommandEncoder, dataBuffer: GPUBuffer, width: number, height: number,
    params: { buffer: GPUBuffer; passes: { pipeline: GPUComputePipeline; offset: number }[] },
  ): void {
    const pass = encoder.beginComputePass();
    for (const { pipeline, offset } of params.passes) {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: params.buffer, offset, size: 24 } }, { binding: 1, resource: { buffer: dataBuffer } }],
      }));
      pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    }
    pass.end();
  }

  async fft2D(realData: Float32Array, imagData: Float32Array, width: number, height: number, inverse: boolean = false): Promise<{ real: Float32Array, imag: Float32Array }> {
    requirePow2(width, height);
    const complexData = interleaveComplex(realData, imagData, width * height);
    const dataBuffer = this.device.createBuffer({ size: complexData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(dataBuffer, 0, complexData);
    const readBuffer = this.device.createBuffer({ size: complexData.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const params = this.fftPassParams(width, height, inverse);
    const encoder = this.device.createCommandEncoder();
    this.encodeFFT2D(encoder, dataBuffer, width, height, params);
    encoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, complexData.byteLength);
    this.device.queue.submit([encoder.finish()]);
    dataBuffer.destroy(); params.buffer.destroy();
    await readBuffer.mapAsync(GPUMapMode.READ);
    const spectrum = deinterleaveComplex(new Float32Array(readBuffer.getMappedRange()));
    readBuffer.unmap(); readBuffer.destroy();
    return spectrum;
  }

  async fft2DInterleaved(complexData: Float32Array, width: number, height: number, inverse: boolean = false): Promise<Float32Array> {
    requirePow2(width, height);
    const byteSize = width * height * 2 * 4;
    const dataBuffer = this.device.createBuffer({ size: byteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(dataBuffer, 0, complexData.buffer, complexData.byteOffset, byteSize);
    const readBuffer = this.device.createBuffer({ size: byteSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const params = this.fftPassParams(width, height, inverse);
    const encoder = this.device.createCommandEncoder();
    this.encodeFFT2D(encoder, dataBuffer, width, height, params);
    encoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, byteSize);
    this.device.queue.submit([encoder.finish()]);
    dataBuffer.destroy(); params.buffer.destroy();
    await readBuffer.mapAsync(GPUMapMode.READ);
    const spectrum = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap(); readBuffer.destroy();
    return spectrum;
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
    requirePow2(width, height);
    const params = this.fftPassParams(width, height, inverse);
    const encoder = this.device.createCommandEncoder();
    this.encodeFFT2D(encoder, buffer, width, height, params);
    this.device.queue.submit([encoder.finish()]);
    params.buffer.destroy();
  }

  /**
   * Batched 2D FFT: N forward FFTs of one size with one readback wait for all.
   * Each image is its own submit so the next upload overlaps this transform.
   */
  async fft2DBatch(
    images: { real: Float32Array; imag: Float32Array }[],
    width: number, height: number,
  ): Promise<{ real: Float32Array; imag: Float32Array }[]> {
    if (images.length === 0) return [];
    requirePow2(width, height);
    const byteSize = width * height * 2 * 4;
    const params = this.fftPassParams(width, height, false);
    // one submit per image lets the next upload overlap this image's transform
    const readBuffers = images.map(image => {
      const dataBuffer = this.device.createBuffer({ size: byteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(dataBuffer, 0, interleaveComplex(image.real, image.imag, width * height));
      const readBuffer = this.device.createBuffer({ size: byteSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const encoder = this.device.createCommandEncoder();
      this.encodeFFT2D(encoder, dataBuffer, width, height, params);
      encoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, byteSize);
      this.device.queue.submit([encoder.finish()]);
      dataBuffer.destroy();
      return readBuffer;
    });
    params.buffer.destroy();
    await Promise.all(readBuffers.map(buffer => buffer.mapAsync(GPUMapMode.READ)));
    return readBuffers.map(readBuffer => {
      const spectrum = deinterleaveComplex(new Float32Array(readBuffer.getMappedRange()));
      readBuffer.unmap();
      readBuffer.destroy();
      return spectrum;
    });
  }

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

/** Mean-subtract, Hann-window and zero-pad to powers of two, so the display FFT shows no DC spike or edge
 *  streaks. The result is already interleaved (real, 0) in the layout the FFT shader reads. */
export function displayFFTInput(data: Float32Array, w: number, h: number): { complex: Float32Array; pw: number; ph: number } {
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i];
  const mean = sum / data.length;
  const centered = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) centered[i] = data[i] - mean;
  applyHannWindow2D(centered, w, h);
  const pw = nextPow2(w), ph = nextPow2(h);
  const complex = new Float32Array(pw * ph * 2);
  for (let row = 0; row < h; row++)
    for (let col = 0; col < w; col++)
      complex[2 * (row * pw + col)] = centered[row * w + col];
  return { complex, pw, ph };
}

/** shiftedMagnitude (log scale) of an interleaved spectrum, split into the same real and imag planes first. */
export function interleavedShiftedMagnitude(complex: Float32Array, width: number, height: number): Float32Array {
  const count = width * height;
  const real = new Float32Array(count), imag = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    real[i] = complex[2 * i];
    imag[i] = complex[2 * i + 1];
  }
  return shiftedMagnitude(real, imag, width, height, true);
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

/** Copy a ``height x width`` image into the top-left corner of a zero ``paddedHeight x paddedWidth`` grid. */
export function zeroPadTopLeft(source: Float32Array, width: number, height: number, paddedWidth: number, paddedHeight: number): Float32Array {
  const padded = new Float32Array(paddedWidth * paddedHeight);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      padded[row * paddedWidth + col] = source[row * width + col];
    }
  }
  return padded;
}

/** Readout of a clicked pixel of a shifted ``height x width`` FFT: its distance from the center
 * and, with a real-space pixel size, its spatial frequency and d-spacing on the padded grid; null
 * within a pixel of the center.
 */
export function fftClickReadout(row: number, col: number, height: number, width: number, pixelSize: number) {
  const halfW = Math.floor(width / 2);
  const halfH = Math.floor(height / 2);
  const deltaCol = col - halfW;
  const deltaRow = row - halfH;
  const distPx = Math.sqrt(deltaCol * deltaCol + deltaRow * deltaRow);
  if (distPx < 1) return null;
  let spatialFreq: number | null = null;
  let dSpacing: number | null = null;
  if (pixelSize > 0) {
    ({ spatialFrequency: spatialFreq, dSpacing } = reciprocalCoordinatesFromShiftedOffset(
      Math.round(row) - halfH,
      Math.round(col) - halfW,
      nextPow2(height),
      nextPow2(width),
      pixelSize,
      pixelSize,
    ));
  }
  return { distPx, spatialFreq, dSpacing };
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

// One WebGPUFFT per page, memoized while it is being built: concurrent first
// callers (every panel of a widget asks at once) share it instead of each
// compiling the pipelines again. A WebGPUFFT on a lost device can never
// dispatch again; the next caller rebuilds it.
let gpuFFT: Promise<WebGPUFFT> | null = null;
onGPULost(() => { gpuFFT = null; });

let cpuFFT: CPUFFT | null = null;

/** The display FFT: WebGPU on a hardware adapter, the scalar reference otherwise. */
export async function getDisplayFFT(): Promise<DisplayFFT> {
  if (await getHardwareGPUDevice()) return requireWebGPUFFT("Browser FFT");
  cpuFFT ??= new CPUFFT();
  return cpuFFT;
}

/** Return the shared hardware FFT or fail without running a CPU/Worker FFT. */
export function requireWebGPUFFT(operation = "Browser FFT"): Promise<WebGPUFFT> {
  gpuFFT ??= requireHardwareGPUDevice(operation).then((device) => new WebGPUFFT(device));
  return gpuFFT;
}
