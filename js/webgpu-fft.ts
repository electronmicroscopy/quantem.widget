/// <reference types="@webgpu/types" />

/**
 * WebGPU FFT — shared 2D FFT with GPU acceleration and CPU fallback.
 * Handles non-power-of-2 dimensions via zero-padding.
 */

// ============================================================================
// CPU FFT fallback
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
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      workReal[y * paddedW + x] = real[y * width + x]; workImag[y * paddedW + x] = imag[y * width + x];
    }
  } else { workReal = real; workImag = imag; }
  const rowReal = new Float32Array(paddedW), rowImag = new Float32Array(paddedW);
  for (let y = 0; y < paddedH; y++) {
    const offset = y * paddedW;
    for (let x = 0; x < paddedW; x++) { rowReal[x] = workReal[offset + x]; rowImag[x] = workImag[offset + x]; }
    fft1d(rowReal, rowImag, inverse);
    for (let x = 0; x < paddedW; x++) { workReal[offset + x] = rowReal[x]; workImag[offset + x] = rowImag[x]; }
  }
  const colReal = new Float32Array(paddedH), colImag = new Float32Array(paddedH);
  for (let x = 0; x < paddedW; x++) {
    for (let y = 0; y < paddedH; y++) { colReal[y] = workReal[y * paddedW + x]; colImag[y] = workImag[y * paddedW + x]; }
    fft1d(colReal, colImag, inverse);
    for (let y = 0; y < paddedH; y++) { workReal[y * paddedW + x] = colReal[y]; workImag[y * paddedW + x] = colImag[y]; }
  }
  if (needsPadding) {
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      real[y * width + x] = workReal[y * paddedW + x]; imag[y * width + x] = workImag[y * paddedW + x];
    }
  }
}

export function fftshift(data: Float32Array, width: number, height: number): void {
  const halfW = width >> 1, halfH = height >> 1;
  const temp = new Float32Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    temp[((y + halfH) % height) * width + ((x + halfW) % width)] = data[y * width + x];
  }
  data.set(temp);
}

// ============================================================================
// CPU FFT Web Worker — runs fft2d + fftshift + computeMagnitude off main thread
// ============================================================================

const FFT_WORKER_CODE = `
function nextPow2(n) { return Math.pow(2, Math.ceil(Math.log2(n))); }
function fft1d(real, imag, inverse) {
  var n = real.length; if (n <= 1) return;
  var j = 0;
  for (var i = 0; i < n - 1; i++) {
    if (i < j) { var t = real[i]; real[i] = real[j]; real[j] = t; t = imag[i]; imag[i] = imag[j]; imag[j] = t; }
    var k = n >> 1; while (k <= j) { j -= k; k >>= 1; } j += k;
  }
  var sign = inverse ? 1 : -1;
  for (var len = 2; len <= n; len <<= 1) {
    var halfLen = len >> 1, angle = (sign * 2 * Math.PI) / len;
    var wR = Math.cos(angle), wI = Math.sin(angle);
    for (var i = 0; i < n; i += len) {
      var cR = 1, cI = 0;
      for (var k = 0; k < halfLen; k++) {
        var eI = i + k, oI = i + k + halfLen;
        var tR = cR * real[oI] - cI * imag[oI], tI = cR * imag[oI] + cI * real[oI];
        real[oI] = real[eI] - tR; imag[oI] = imag[eI] - tI;
        real[eI] += tR; imag[eI] += tI;
        var nR = cR * wR - cI * wI; cI = cR * wI + cI * wR; cR = nR;
      }
    }
  }
  if (inverse) { for (var i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; } }
}
function fft2d(real, imag, width, height, inverse) {
  var pW = nextPow2(width), pH = nextPow2(height), pad = pW !== width || pH !== height;
  var wR, wI;
  if (pad) {
    wR = new Float32Array(pW * pH); wI = new Float32Array(pW * pH);
    for (var y = 0; y < height; y++) for (var x = 0; x < width; x++) { wR[y*pW+x] = real[y*width+x]; wI[y*pW+x] = imag[y*width+x]; }
  } else { wR = real; wI = imag; }
  var rR = new Float32Array(pW), rI = new Float32Array(pW);
  for (var y = 0; y < pH; y++) {
    var o = y * pW; for (var x = 0; x < pW; x++) { rR[x] = wR[o+x]; rI[x] = wI[o+x]; }
    fft1d(rR, rI, inverse); for (var x = 0; x < pW; x++) { wR[o+x] = rR[x]; wI[o+x] = rI[x]; }
  }
  var cR = new Float32Array(pH), cI = new Float32Array(pH);
  for (var x = 0; x < pW; x++) {
    for (var y = 0; y < pH; y++) { cR[y] = wR[y*pW+x]; cI[y] = wI[y*pW+x]; }
    fft1d(cR, cI, inverse); for (var y = 0; y < pH; y++) { wR[y*pW+x] = cR[y]; wI[y*pW+x] = cI[y]; }
  }
  if (pad) { for (var y = 0; y < height; y++) for (var x = 0; x < width; x++) { real[y*width+x] = wR[y*pW+x]; imag[y*width+x] = wI[y*pW+x]; } }
}
function fftshift(data, width, height) {
  var hW = width >> 1, hH = height >> 1, temp = new Float32Array(width * height);
  for (var y = 0; y < height; y++) for (var x = 0; x < width; x++) temp[((y+hH)%height)*width+((x+hW)%width)] = data[y*width+x];
  data.set(temp);
}
self.onmessage = function(e) {
  var d = e.data, real = d.real, imag = d.imag, w = d.width, h = d.height;
  fft2d(real, imag, w, h, d.inverse);
  fftshift(real, w, h); fftshift(imag, w, h);
  var n = real.length, mag = new Float32Array(n);
  for (var i = 0; i < n; i++) mag[i] = Math.sqrt(real[i]*real[i] + imag[i]*imag[i]);
  self.postMessage({ id: d.id, magnitude: mag, real: real, imag: imag }, [mag.buffer, real.buffer, imag.buffer]);
};
`;

let _fftWorker: Worker | null = null;
const _fftCallbacks = new Map<number, (data: { magnitude: Float32Array; real: Float32Array; imag: Float32Array }) => void>();
let _fftWorkerId = 0;

function getFFTWorker(): Worker {
  if (!_fftWorker) {
    const blob = new Blob([FFT_WORKER_CODE], { type: 'application/javascript' });
    _fftWorker = new Worker(URL.createObjectURL(blob));
    _fftWorker.onmessage = (e: MessageEvent) => {
      const cb = _fftCallbacks.get(e.data.id);
      if (cb) {
        _fftCallbacks.delete(e.data.id);
        cb(e.data);
      }
    };
  }
  return _fftWorker;
}

/**
 * CPU FFT in a Web Worker — does fft2d + fftshift + computeMagnitude off main thread.
 * Transfers Float32Arrays to the worker (zero-copy) so the main thread is never blocked.
 * The input arrays become detached after this call — pass copies if you need to keep them.
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
// WebGPU FFT — GPU-accelerated 2D FFT
// ============================================================================

const FFT_2D_SHADER = /* wgsl */`
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
fn twiddle(k: u32, N: u32, inverse: f32) -> vec2<f32> { let angle = inverse * 2.0 * 3.14159265359 * f32(k) / f32(N); return vec2<f32>(cos(angle), sin(angle)); }
fn bitReverse(x: u32, log2N: u32) -> u32 { var result: u32 = 0u; var val = x; for (var i: u32 = 0u; i < log2N; i = i + 1u) { result = (result << 1u) | (val & 1u); val = val >> 1u; } return result; }
struct FFT2DParams { width: u32, height: u32, log2Size: u32, stage: u32, inverse: f32, isRowWise: u32, }
@group(0) @binding(0) var<uniform> params: FFT2DParams;
@group(0) @binding(1) var<storage, read_write> data: array<vec2<f32>>;
fn getIndex(row: u32, col: u32) -> u32 { return row * params.width + col; }
@compute @workgroup_size(16, 16) fn bitReverseRows(@builtin(global_invocation_id) gid: vec3<u32>) { let row = gid.y; let col = gid.x; if (row >= params.height || col >= params.width) { return; } let rev = bitReverse(col, params.log2Size); if (col < rev) { let idx1 = getIndex(row, col); let idx2 = getIndex(row, rev); let temp = data[idx1]; data[idx1] = data[idx2]; data[idx2] = temp; } }
@compute @workgroup_size(16, 16) fn bitReverseCols(@builtin(global_invocation_id) gid: vec3<u32>) { let row = gid.y; let col = gid.x; if (row >= params.height || col >= params.width) { return; } let rev = bitReverse(row, params.log2Size); if (row < rev) { let idx1 = getIndex(row, col); let idx2 = getIndex(rev, col); let temp = data[idx1]; data[idx1] = data[idx2]; data[idx2] = temp; } }
@compute @workgroup_size(16, 16) fn butterflyRows(@builtin(global_invocation_id) gid: vec3<u32>) { let row = gid.y; let idx = gid.x; if (row >= params.height || idx >= params.width / 2u) { return; } let stage = params.stage; let halfSize = 1u << stage; let fullSize = halfSize << 1u; let group = idx / halfSize; let pos = idx % halfSize; let col_i = group * fullSize + pos; let col_j = col_i + halfSize; if (col_j >= params.width) { return; } let w = twiddle(pos, fullSize, params.inverse); let i = getIndex(row, col_i); let j = getIndex(row, col_j); let u = data[i]; let t = cmul(w, data[j]); data[i] = u + t; data[j] = u - t; }
@compute @workgroup_size(16, 16) fn butterflyCols(@builtin(global_invocation_id) gid: vec3<u32>) { let col = gid.x; let idx = gid.y; if (col >= params.width || idx >= params.height / 2u) { return; } let stage = params.stage; let halfSize = 1u << stage; let fullSize = halfSize << 1u; let group = idx / halfSize; let pos = idx % halfSize; let row_i = group * fullSize + pos; let row_j = row_i + halfSize; if (row_j >= params.height) { return; } let w = twiddle(pos, fullSize, params.inverse); let i = getIndex(row_i, col); let j = getIndex(row_j, col); let u = data[i]; let t = cmul(w, data[j]); data[i] = u + t; data[j] = u - t; }
@compute @workgroup_size(16, 16) fn normalize2D(@builtin(global_invocation_id) gid: vec3<u32>) { let row = gid.y; let col = gid.x; if (row >= params.height || col >= params.width) { return; } let idx = getIndex(row, col); let scale = 1.0 / f32(params.width * params.height); data[idx] = data[idx] * scale; }`;

export class WebGPUFFT {
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
  async fft2D(realData: Float32Array, imagData: Float32Array, width: number, height: number, inverse: boolean = false): Promise<{ real: Float32Array, imag: Float32Array }> {
    await this.init();
    const paddedWidth = nextPow2(width), paddedHeight = nextPow2(height);
    const needsPadding = paddedWidth !== width || paddedHeight !== height;
    const log2Width = Math.log2(paddedWidth), log2Height = Math.log2(paddedHeight);
    const paddedSize = paddedWidth * paddedHeight, originalSize = width * height;
    let workReal: Float32Array, workImag: Float32Array;
    if (needsPadding) {
      workReal = new Float32Array(paddedSize); workImag = new Float32Array(paddedSize);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) { workReal[y * paddedWidth + x] = realData[y * width + x]; workImag[y * paddedWidth + x] = imagData[y * width + x]; }
    } else { workReal = realData; workImag = imagData; }
    const complexData = new Float32Array(paddedSize * 2);
    for (let i = 0; i < paddedSize; i++) { complexData[i * 2] = workReal[i]; complexData[i * 2 + 1] = workImag[i]; }
    const dataBuffer = this.device.createBuffer({ size: complexData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(dataBuffer, 0, complexData);
    const paramsBuffer = this.device.createBuffer({ size: 24, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readBuffer = this.device.createBuffer({ size: complexData.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const inverseVal = inverse ? 1.0 : -1.0;
    const workgroupsX = Math.ceil(paddedWidth / 16), workgroupsY = Math.ceil(paddedHeight / 16);
    const runPass = (pipeline: GPUComputePipeline) => {
      const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: paramsBuffer } }, { binding: 1, resource: { buffer: dataBuffer } }] });
      const encoder = this.device.createCommandEncoder(); const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(workgroupsX, workgroupsY); pass.end();
      this.device.queue.submit([encoder.finish()]);
    };
    const params = new ArrayBuffer(24); const paramsU32 = new Uint32Array(params); const paramsF32 = new Float32Array(params);
    paramsU32[0] = paddedWidth; paramsU32[1] = paddedHeight; paramsU32[2] = log2Width; paramsU32[3] = 0; paramsF32[4] = inverseVal; paramsU32[5] = 1;
    this.device.queue.writeBuffer(paramsBuffer, 0, params); runPass(this.pipelines2D!.bitReverseRows);
    for (let stage = 0; stage < log2Width; stage++) { paramsU32[3] = stage; this.device.queue.writeBuffer(paramsBuffer, 0, params); runPass(this.pipelines2D!.butterflyRows); }
    paramsU32[2] = log2Height; paramsU32[3] = 0; paramsU32[5] = 0;
    this.device.queue.writeBuffer(paramsBuffer, 0, params); runPass(this.pipelines2D!.bitReverseCols);
    for (let stage = 0; stage < log2Height; stage++) { paramsU32[3] = stage; this.device.queue.writeBuffer(paramsBuffer, 0, params); runPass(this.pipelines2D!.butterflyCols); }
    if (inverse) runPass(this.pipelines2D!.normalize);
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, complexData.byteLength);
    this.device.queue.submit([encoder.finish()]); await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0)); readBuffer.unmap();
    dataBuffer.destroy(); paramsBuffer.destroy(); readBuffer.destroy();
    if (needsPadding) {
      const realResult = new Float32Array(originalSize), imagResult = new Float32Array(originalSize);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) { realResult[y * width + x] = result[(y * paddedWidth + x) * 2]; imagResult[y * width + x] = result[(y * paddedWidth + x) * 2 + 1]; }
      return { real: realResult, imag: imagResult };
    }
    const realResult = new Float32Array(paddedSize), imagResult = new Float32Array(paddedSize);
    for (let i = 0; i < paddedSize; i++) { realResult[i] = result[i * 2]; imagResult[i] = result[i * 2 + 1]; }
    return { real: realResult, imag: imagResult };
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
 * Symmetric formula: w(i) = 0.5*(1 - cos(2πi/(N-1))), matching np.hanning —
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
  for (let r = 0; r < height; r++) {
    const hr = hannH[r];
    const offset = r * width;
    for (let c = 0; c < width; c++) data[offset + c] *= hr * hannW[c];
  }
}

// ============================================================================
// FFT post-processing helpers
// ============================================================================

/** Compute magnitude from complex FFT output: sqrt(real² + imag²). */
export function computeMagnitude(real: Float32Array, imag: Float32Array): Float32Array {
  const mag = new Float32Array(real.length);
  for (let i = 0; i < mag.length; i++) {
    mag[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
  }
  return mag;
}

/** Mask DC component (center pixel) and return 99.9% percentile-clipped range. Mutates `mag`. */
export function autoEnhanceFFT(
  mag: Float32Array, width: number, height: number,
): { min: number; max: number } {
  const centerIdx = Math.floor(height / 2) * width + Math.floor(width / 2);
  const neighbors = [
    mag[Math.max(0, centerIdx - 1)],
    mag[Math.min(mag.length - 1, centerIdx + 1)],
    mag[Math.max(0, centerIdx - width)],
    mag[Math.min(mag.length - 1, centerIdx + width)],
  ];
  mag[centerIdx] = neighbors.reduce((a, b) => a + b, 0) / 4;
  // Use O(n) histogram approach instead of O(n log n) sort
  const len = mag.length;
  if (len === 0) return { min: 0, max: 0 };
  let dMin = Infinity, dMax = -Infinity;
  for (let i = 0; i < len; i++) {
    const v = mag[i];
    if (v < dMin) dMin = v;
    if (v > dMax) dMax = v;
  }
  if (dMin === dMax) return { min: dMin, max: dMax };
  const NUM_BINS = 1024;
  const bins = new Uint32Array(NUM_BINS);
  const range = dMax - dMin;
  const scale = (NUM_BINS - 1) / range;
  for (let i = 0; i < len; i++) bins[Math.floor((mag[i] - dMin) * scale)]++;
  // Find 99.9th percentile
  const target = Math.ceil(len * 0.999);
  let cumSum = 0;
  let pMax = dMax;
  for (let i = 0; i < NUM_BINS; i++) {
    cumSum += bins[i];
    if (cumSum >= target) { pMax = dMin + (i / (NUM_BINS - 1)) * range; break; }
  }
  // If percentile collapsed to min (sparse spectra), fall back to actual max
  if (pMax <= dMin) pMax = dMax;
  return { min: dMin, max: pMax };
}

// ============================================================================
// Singleton
// ============================================================================

let gpuFFT: WebGPUFFT | null = null;
let gpuDevice: GPUDevice | null = null;
let gpuInfo = "GPU";

export async function getGPUDevice(): Promise<GPUDevice | null> {
  if (gpuDevice) return gpuDevice;
  if (!navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    try {
      // @ts-ignore - requestAdapterInfo is not yet in all type definitions
      const info = await adapter.requestAdapterInfo?.();
      if (info) {
        gpuInfo = info.description || `${info.vendor} ${info.architecture || ""} ${info.device || ""}`.trim() || "Generic WebGPU Adapter";
      }
    } catch (_e) { /* adapter info not available */ }
    gpuDevice = await adapter.requestDevice();
    return gpuDevice;
  } catch { return null; }
}

export async function getWebGPUFFT(): Promise<WebGPUFFT | null> {
  if (gpuFFT) return gpuFFT;
  const device = await getGPUDevice();
  if (!device) { console.warn('WebGPU not supported, falling back to CPU FFT'); return null; }
  try {
    gpuFFT = new WebGPUFFT(device);
    await gpuFFT.init();
    return gpuFFT;
  } catch (e) { console.warn('WebGPU init failed:', e); return null; }
}

export function getGPUInfo(): string { return gpuInfo; }
