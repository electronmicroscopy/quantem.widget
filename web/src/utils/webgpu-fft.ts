/// <reference types="@webgpu/types" />
/**
 * WebGPU FFT — GPU-accelerated 2D FFT for real-time ROI analysis.
 * Ported from quantem.widget/js/webgpu-fft.ts
 */
import { getGPUDevice as engineGetGPUDevice, onGPULost } from "../engine/device";

function nextPow2(n: number): number { return Math.pow(2, Math.ceil(Math.log2(n))); }
type WebGPUFftOptions = {
  shouldDefer?: () => boolean;
};

async function waitWhileDeferred(shouldDefer?: () => boolean): Promise<void> {
  while (shouldDefer?.()) {
    await new Promise<void>(resolve => globalThis.setTimeout(resolve, 80));
  }
}

import { FFT_2D_SHADER } from "../engine/fft-shader";

export class WebGPUFFT {
  private device: GPUDevice;
  private pipelines: {
    bitReverseRows: GPUComputePipeline; bitReverseCols: GPUComputePipeline;
    butterflyRows: GPUComputePipeline; butterflyCols: GPUComputePipeline;
    normalize: GPUComputePipeline;
  } | null = null;
  private initialized = false;

  constructor(device: GPUDevice) { this.device = device; }

  async init(): Promise<void> {
    if (this.initialized) return;
    const module = this.device.createShaderModule({ code: FFT_2D_SHADER });
    this.pipelines = {
      bitReverseRows: this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "bitReverseRows" } }),
      bitReverseCols: this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "bitReverseCols" } }),
      butterflyRows: this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "butterflyRows" } }),
      butterflyCols: this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "butterflyCols" } }),
      normalize: this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "normalize2D" } }),
    };
    this.initialized = true;
  }

  async fft2D(realData: Float32Array, imagData: Float32Array, width: number, height: number): Promise<{ real: Float32Array; imag: Float32Array }> {
    await this.init();
    const pw = nextPow2(width), ph = nextPow2(height);
    const needsPad = pw !== width || ph !== height;
    const log2W = Math.log2(pw), log2H = Math.log2(ph);
    const ps = pw * ph;

    let wR: Float32Array, wI: Float32Array;
    if (needsPad) {
      wR = new Float32Array(ps); wI = new Float32Array(ps);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        wR[y * pw + x] = realData[y * width + x]; wI[y * pw + x] = imagData[y * width + x];
      }
    } else { wR = realData; wI = imagData; }

    const complex = new Float32Array(ps * 2);
    for (let i = 0; i < ps; i++) { complex[i * 2] = wR[i]; complex[i * 2 + 1] = wI[i]; }

    const dataBuf = this.device.createBuffer({ size: complex.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(dataBuf, 0, complex);
    // WGSL requires uniform struct size to be a multiple of 16 bytes.
    // FFT2DParams has 6×u32 = 24 bytes; bump buffer to 32 bytes (16-byte
    // multiple) so writes round-trip. Prior 24-byte buffer produced
    // silently-truncated uniform reads on large inputs (manifested as
    // all-zeros FFT output for 1024×1024 padded obj_phase).
    const paramsBuf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readBuf = this.device.createBuffer({ size: complex.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const wgX = Math.ceil(pw / 16), wgY = Math.ceil(ph / 16);
    // The FFT is a linear chain of ~2×log2(N) butterfly passes, each
    // reading+writing the SAME dataBuf via a different uniform `stage`
    // index. Previous version did one commandEncoder.finish + queue.submit
    // per pass (~21 submits for a 1024 FFT). That meant ~21 driver flushes
    // and forced GPU to serialize between passes via a pipeline bubble.
    //
    // Folding everything into ONE encoder + one submit eliminates ~20 ms
    // of per-call overhead on Apple M5. Each pass writes a fresh params
    // uniform via staging buffers (writeBuffer before/between passes on
    // the same encoder is fine; they're ordered on the queue).
    //
    // BUT — a single paramsBuf can't hold N different values of `stage`
    // at once, so we allocate one paramsBuf PER PASS (cheap: 32 bytes each)
    // and let the encoder batch all dispatches together.
    const TOTAL_PASSES = 1 + log2W + 1 + log2H;
    const paramsBufs: GPUBuffer[] = [];
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();

    const writeParams = (idx: number, values: [number, number, number, number, number, number]) => {
      const pb = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      paramsBufs[idx] = pb;
      const p = new ArrayBuffer(24); const u = new Uint32Array(p); const f = new Float32Array(p);
      u[0] = values[0]; u[1] = values[1]; u[2] = values[2]; u[3] = values[3];
      f[4] = values[4]; u[5] = values[5];
      this.device.queue.writeBuffer(pb, 0, p);
      return pb;
    };
    const runOn = (pipeline: GPUComputePipeline, paramsBuf: GPUBuffer) => {
      const bg = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: paramsBuf } }, { binding: 1, resource: { buffer: dataBuf } }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX, wgY);
    };

    let passIdx = 0;
    runOn(this.pipelines!.bitReverseRows, writeParams(passIdx++, [pw, ph, log2W, 0, -1.0, 1]));
    for (let s = 0; s < log2W; s++) {
      runOn(this.pipelines!.butterflyRows, writeParams(passIdx++, [pw, ph, log2W, s, -1.0, 1]));
    }
    runOn(this.pipelines!.bitReverseCols, writeParams(passIdx++, [pw, ph, log2H, 0, -1.0, 0]));
    for (let s = 0; s < log2H; s++) {
      runOn(this.pipelines!.butterflyCols, writeParams(passIdx++, [pw, ph, log2H, s, -1.0, 0]));
    }
    pass.end();
    enc.copyBufferToBuffer(dataBuf, 0, readBuf, 0, complex.byteLength);
    this.device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuf.getMappedRange().slice(0)); readBuf.unmap();
    dataBuf.destroy();
    for (const pb of paramsBufs) pb.destroy();
    readBuf.destroy();
    // paramsBuf from the original signature is kept for compat but unused
    // in the batched path — destroy the allocation we made earlier.
    paramsBuf.destroy();

    if (needsPad) {
      const rR = new Float32Array(width * height), rI = new Float32Array(width * height);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        rR[y * width + x] = result[(y * pw + x) * 2]; rI[y * width + x] = result[(y * pw + x) * 2 + 1];
      }
      return { real: rR, imag: rI };
    }
    const rR = new Float32Array(ps), rI = new Float32Array(ps);
    for (let i = 0; i < ps; i++) { rR[i] = result[i * 2]; rI[i] = result[i * 2 + 1]; }
    return { real: rR, imag: rI };
  }

  /** Like fft2D but returns the PADDED-size result instead of un-padding to
   *  (width×height). Callers doing spectral display on non-power-of-2 inputs
   *  need the full padded spectrum so they can do a proper centered fftshift;
   *  the plain fft2D's "grab top-left corner" un-pad truncates 70% of the
   *  frequency content and produces a solid-color panel. */
  async fft2DPadded(realData: Float32Array, imagData: Float32Array, width: number, height: number): Promise<{ real: Float32Array; imag: Float32Array; pw: number; ph: number } | null> {
    const pw = nextPow2(width), ph = nextPow2(height);
    // Fast path — already pow-2. Single FFT, fft2D's internal un-pad is a no-op.
    if (pw === width && ph === height) {
      const out = await this.fft2D(realData, imagData, width, height);
      return { real: out.real, imag: out.imag, pw, ph };
    }
    // Non-pow2: pre-pad to (pw, ph) and call fft2D with the PADDED size so
    // fft2D treats our input as already pow2 (needsPad=false), skips its
    // internal copy+pad and its trailing un-pad. Previously this function
    // also ran a first fft2D at (width, height) whose padded result was
    // thrown away — doubling the GPU work for every non-pow2 FFT.
    const padR = new Float32Array(pw * ph);
    // imagData is all-zero in our callers (real-valued input) so skip the
    // interleave there — allocating the zero-filled buffer is enough.
    const padI = new Float32Array(pw * ph);
    for (let y = 0; y < height; y++) {
      padR.set(realData.subarray(y * width, y * width + width), y * pw);
    }
    const out = await this.fft2D(padR, padI, pw, ph);
    return { real: out.real, imag: out.imag, pw, ph };
  }
}

// Singleton FFT instance. The GPU DEVICE is owned by engine/device.ts - this module
// must NOT create its own. Decode/compute (engine), colormap, and FFT have to share
// ONE device, or a buffer/bind-group from one device submitted on another throws
// "BindGroupLayout is associated with [Device], cannot be used with [Device]" and
// crashes the tab GPU process. So getGPUDevice here just delegates to the engine.
let _gpuFFT: WebGPUFFT | null = null;
let _gpuUnavailableReason = "WebGPU has not been checked yet.";
// Drop the FFT pipeline cache when the shared device is lost so it rebuilds on the new one.
onGPULost(() => { _gpuFFT = null; });
// Module-scope warm-up promise. Shared across every component that needs FFT
// so shader compile (50-500 ms) happens at most ONCE per app-load, not
// per-component-mount. Per-CLAUDE.md: "GPU pipelines must be pre-warmed."
let _warmed: Promise<void> | null = null;

export function getGPUUnavailableReason(): string {
  return _gpuUnavailableReason;
}

/** Shared singleton GPU device - delegates to engine/device.ts (the one device for everything). */
export async function getGPUDevice(): Promise<GPUDevice | null> {
  const device = await engineGetGPUDevice();
  _gpuUnavailableReason = device ? "" : "WebGPU unavailable (no adapter / requestDevice failed / not enabled).";
  return device;
}

export async function getWebGPUFFT(): Promise<WebGPUFFT | null> {
  if (_gpuFFT) return _gpuFFT;
  const device = await getGPUDevice();
  if (!device) return null;
  try {
    _gpuFFT = new WebGPUFFT(device);
    await _gpuFFT.init();
    return _gpuFFT;
  } catch { return null; }
}

/** GPU-backed equivalent of utils/fft.ts `fft2dMagnitude`. Returns log1p
 *  magnitude with fftshift + DC suppression in the same layout as the JS
 *  version, so callers can swap without changing downstream rendering.
 *  Resolves to null when WebGPU isn't available OR when the GPU output
 *  fails a sanity check (e.g. all-zeros from a broken shader). */
// One-shot "WebGPU is live" indicator so sessions can tell whether the GPU
// path is actually running or silently falling back to CPU.
let _loggedGPUOnce = false;
export async function fft2dMagnitudeGPU(
  data: Float32Array,
  rows: number,
  cols: number,
  options: WebGPUFftOptions = {},
): Promise<Float32Array | null> {
  const t0 = performance.now();
  const fft = await getWebGPUFFT();
  if (!fft) {
    if (!_loggedGPUOnce) { console.warn("[FFT] WebGPU unavailable — navigator.gpu or adapter missing"); _loggedGPUOnce = true; }
    return null;
  }
  if (!_loggedGPUOnce) { console.info("[FFT] WebGPU adapter active — FFT runs on GPU"); _loggedGPUOnce = true; }
  const imag = new Float32Array(data.length);
  let out: { real: Float32Array; imag: Float32Array; pw: number; ph: number } | null;
  try {
    // fft2DPadded gives us the FULL padded spectrum (pw × ph) — we must
    // shift and crop-center it OURSELVES. fft2D's un-pad grabs only the
    // low-frequency corner and produces garbage on 686×686 inputs.
    out = await fft.fft2DPadded(data, imag, cols, rows);
  } catch { return null; }
  if (!out) return null;
  await waitWhileDeferred(options.shouldDefer);
  const { real: re, imag: im, pw: pc, ph: pr } = out;
  // Sanity check: a well-formed FFT of a natural image has spectrum
  // spread over many orders of magnitude (DC spike + broad noise floor).
  // If the variance of the raw FFT output is ~0, the shader returned
  // garbage — let the caller fall back to CPU. 2026-04-20: the WebGPU
  // shader on 686×686 inputs was producing a uniform-magnitude output
  // that rendered as solid pink after colormapping.
  let maxMag = 0;
  const sampleN = Math.min(re.length, 4096);
  const stride = Math.max(1, Math.floor(re.length / sampleN));
  for (let i = 0; i < re.length; i += stride) {
    const m = re[i] * re[i] + im[i] * im[i];
    if (m > maxMag) maxMag = m;
  }
  if (maxMag < 1e-12) {
    console.warn(`[FFT] WebGPU shader returned flat output (input ${rows}x${cols}, padded ${pr}x${pc})`);
    return null;
  }
  // fftshift + log-magnitude + crop to (rows × cols) in one pass.
  // Precompute row remap so the hot loop is a single mod-free index arithmetic.
  // Keep the log definition identical to the CPU/server renderer so FFTs have
  // one visual style across SSB, DPC, BF/DF, CoM, thumbnails, and viewers.
  const result = new Float32Array(rows * cols);
  const halfR = rows >> 1, halfC = cols >> 1;
  // rowRemap[r] = unshifted padded-FFT row for output row r
  const rowRemap = new Int32Array(rows);
  for (let r = 0; r < rows; r++) {
    const x = r - halfR;
    rowRemap[r] = ((x % pr) + pr) % pr;
  }
  const colRemap = new Int32Array(cols);
  for (let c = 0; c < cols; c++) {
    const x = c - halfC;
    colRemap[c] = ((x % pc) + pc) % pc;
  }
  for (let r = 0; r < rows; r++) {
    const rowBase = rowRemap[r] * pc;
    const outBase = r * cols;
    for (let c = 0; c < cols; c++) {
      const i = rowBase + colRemap[c];
      const a = re[i], b = im[i];
      result[outBase + c] = Math.log1p(Math.sqrt(a * a + b * b));
    }
  }
  const cR = halfR;
  const cC = halfC;
  if (cR > 0 && cC > 0 && cR + 1 < rows && cC + 1 < cols) {
    const n = [
      result[(cR - 1) * cols + (cC - 1)], result[(cR - 1) * cols + cC], result[(cR - 1) * cols + (cC + 1)],
      result[cR * cols + (cC - 1)],                                      result[cR * cols + (cC + 1)],
      result[(cR + 1) * cols + (cC - 1)], result[(cR + 1) * cols + cC], result[(cR + 1) * cols + (cC + 1)],
    ].sort((a, b) => a - b);
    result[cR * cols + cC] = 0.5 * (n[3] + n[4]);
  }
  let sampledMin = Infinity;
  let sampledMax = -Infinity;
  for (let i = 0; i < result.length; i += Math.max(1, Math.floor(result.length / 4096))) {
    const v = result[i];
    if (!Number.isFinite(v)) continue;
    if (v < sampledMin) sampledMin = v;
    if (v > sampledMax) sampledMax = v;
  }
  if (!(sampledMax > sampledMin)) {
    console.warn(`[FFT] WebGPU shader returned unusable log-magnitude output (input ${rows}x${cols})`);
    return null;
  }
  if ((window as typeof window & { __quantemDebugFft?: boolean }).__quantemDebugFft) {
    console.debug(`[FFT] WebGPU OK (input ${rows}x${cols}, ${(performance.now()-t0).toFixed(1)}ms)`);
  }
  return result;
}

/** Pre-warm the WebGPU FFT pipeline by running a throwaway 16×16 FFT.
 *  Idempotent: every caller awaits the SAME promise, so shader compile
 *  (50-500 ms on first call) happens once per app-load no matter how
 *  many components mount/unmount. Consumers decide WHEN to warm:
 *  App.tsx can call this on idle to mask compile latency while the
 *  user reads the table; on-demand callers can await it just-in-time.
 *  Both patterns share the single promise. */
export function ensureFFTWarmed(): Promise<void> {
  if (_warmed) return _warmed;
  _warmed = (async () => {
    try {
      const dummy = new Float32Array(16 * 16);
      await fft2dMagnitudeGPU(dummy, 16, 16);
    } catch (e) {
      // First-call may fail if WebGPU device is unavailable; reset so a
      // later call can retry (rare — usually fatal).
      _warmed = null;
      throw e;
    }
  })();
  return _warmed;
}

/** True after ensureFFTWarmed() has been kicked off (whether or not the
 *  underlying promise has resolved). Components that want a "warming"
 *  placeholder can poll this. */
export function isFFTWarmed(): boolean { return _warmed !== null; }
