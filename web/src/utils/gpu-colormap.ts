/// <reference types="@webgpu/types" />
/**
 * GPU Colormap Engine — WebGPU-accelerated colormapping + histogram.
 * Ported from quantem.widget/js/colormaps.ts for the dashboard.
 *
 * Provides instant colormap/contrast changes without server round-trips.
 * Falls back to CPU (applyColormap from colormaps.ts) when WebGPU is unavailable.
 */

import { getGPUDevice } from "./webgpu-fft";
import { COLORMAPS, applyColormap } from "./colormaps";

// ── WGSL Shaders ──

const COLORMAP_SHADER = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  vmin: f32,
  vmax: f32,
  log_scale: u32,
  _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

// rgba mode (entry=main): pack as R | G<<8 | B<<16 | A<<24 — consumed via
// JS Uint8ClampedArray + ImageData (RGBA byte order).
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var val = data[idx];
  if (params.log_scale == 1u) { val = log(1.0 + max(val, 0.0)); }
  let range = max(params.vmax - params.vmin, 1e-30);
  let t = clamp((val - params.vmin) / range, 0.0, 1.0);
  let bin = min(u32(t * 255.0), 255u);
  rgba[idx] = lut[bin] | 0xFF000000u;
}

// bgra mode (entry=main_bgra): pack as B | G<<8 | R<<16 | A<<24 for direct
// copyBufferToTexture into a canvas configured with format=bgra8unorm.
// Uses params._pad repurposed as row_stride_u32 (= bytesPerRow / 4) so
// output rows align to 256 bytes as required by copyBufferToTexture.
// Avoids the mapAsync readback entirely.
@compute @workgroup_size(16, 16)
fn main_bgra(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let src_idx = gid.y * params.width + gid.x;
  let dst_idx = gid.y * params._pad + gid.x;
  var val = data[src_idx];
  if (params.log_scale == 1u) { val = log(1.0 + max(val, 0.0)); }
  let range = max(params.vmax - params.vmin, 1e-30);
  let t = clamp((val - params.vmin) / range, 0.0, 1.0);
  let bin = min(u32(t * 255.0), 255u);
  let rgb = lut[bin];
  let r = rgb & 0xFFu;
  let g = (rgb >> 8u) & 0xFFu;
  let b = (rgb >> 16u) & 0xFFu;
  rgba[dst_idx] = b | (g << 8u) | (r << 16u) | 0xFF000000u;
}
`;

// Complex colormap matching quantem-core's `array_to_rgba` (JCh perceptual
// via colorspacious CAM16 UCS). Formula from quantem/core/visualization/
// visualization_utils.py lines 61-67:
//   J = amp_normalized * 61.5            (lightness, 0-100 scale)
//   C = min(98 * J / 123, 110)           (chroma)
//   h = rad2deg(angle) + 180             (hue, degrees)
// Normalization: QuantileInterval(2%, 98%) + LinearStretch (CustomNormalization
// defaults). Passed as amin/amax — caller computes them from the data.
//
// CAM16-UCS JCh → sRGB is ~300 lines; we use CIELAB L*C*h* → XYZ → sRGB which
// is a closed-form approximation with matching J/C/h semantics and
// visually indistinguishable output for the probe/complex use case.
const COMPLEX_JCH_SHADER = /* wgsl */ `
struct JchParams {
  width: u32,
  height: u32,
  amin: f32,      // |z| low clip (2nd percentile in quantem's default)
  amax: f32,      // |z| high clip (98th percentile)
  _pad0: u32,
  _pad1: u32,
};
@group(0) @binding(0) var<uniform> params: JchParams;
@group(0) @binding(1) var<storage, read> data: array<f32>;   // interleaved re, im
@group(0) @binding(2) var<storage, read_write> rgba: array<u32>;

// CIE Lab → XYZ (D65 whitepoint), then XYZ → linear sRGB, then sRGB gamma.
fn lab_finv(t: f32) -> f32 {
  // f_inv(t) = t^3 if t > 6/29 else 3 * (6/29)^2 * (t - 4/29)
  let delta = 6.0 / 29.0;
  if (t > delta) { return t * t * t; }
  return 3.0 * delta * delta * (t - 4.0 / 29.0);
}

fn lab_to_rgb(L: f32, a: f32, b: f32) -> vec3<f32> {
  let fy = (L + 16.0) / 116.0;
  let fx = a / 500.0 + fy;
  let fz = fy - b / 200.0;
  // D65 whitepoint
  let X = 0.95047 * lab_finv(fx);
  let Y = 1.00000 * lab_finv(fy);
  let Z = 1.08883 * lab_finv(fz);
  // XYZ → linear sRGB (Bradford-adapted D65 matrix)
  let rl =  3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  let gl = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  let bl =  0.0557 * X - 0.2040 * Y + 1.0570 * Z;
  let lin = vec3<f32>(rl, gl, bl);
  // sRGB gamma companding — gamma FIRST, then clamp. Matches
  // colorspacious output order: cspace_convert() → .clip(0, 1).
  // Clamping before gamma would crush out-of-gamut negatives into 0 and
  // then the gamma would still fire; order differs for wide-gamut colors.
  var out = vec3<f32>(0.0, 0.0, 0.0);
  for (var i = 0; i < 3; i = i + 1) {
    let c = lin[i];
    var g: f32;
    if (c <= 0.0031308) {
      g = 12.92 * c;
    } else {
      g = 1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055;
    }
    out[i] = clamp(g, 0.0, 1.0);
  }
  return out;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let re = data[idx * 2u];
  let im = data[idx * 2u + 1u];
  let amp = sqrt(re * re + im * im);
  // Quantile interval: map [amin, amax] → [0, 1] (linear stretch).
  let range = max(params.amax - params.amin, 1e-30);
  let scaled = clamp((amp - params.amin) / range, 0.0, 1.0);
  // quantem-core formula (visualization_utils.py:61-63)
  let J = scaled * 61.5;
  let C = min(98.0 * J / 123.0, 110.0);
  let phase = atan2(im, re);
  let h_deg = degrees(phase) + 180.0;
  let h_rad = radians(h_deg);
  // JCh (CAM16) ≈ LCh (CIELAB) for this application — closed-form conversion.
  let L = J;
  let a = C * cos(h_rad);
  let b_lab = C * sin(h_rad);
  let c = lab_to_rgb(L, a, b_lab);
  let r = u32(clamp(c.r, 0.0, 1.0) * 255.0);
  let g = u32(clamp(c.g, 0.0, 1.0) * 255.0);
  let b_out = u32(clamp(c.b, 0.0, 1.0) * 255.0);
  rgba[idx] = r | (g << 8u) | (b_out << 16u) | 0xFF000000u;
}
`;

const HISTOGRAM_SHADER = /* wgsl */ `
struct HistParams {
  width: u32,
  height: u32,
  dmin: f32,
  dmax: f32,
  log_scale: u32,
  _pad: u32,
};
@group(0) @binding(0) var<uniform> params: HistParams;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>>;

@compute @workgroup_size(16, 16)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var val = data[idx];
  if (params.log_scale == 1u) { val = log(1.0 + max(val, 0.0)); }
  let range = max(params.dmax - params.dmin, 1e-30);
  let t = clamp((val - params.dmin) / range, 0.0, 1.0);
  let bin = min(u32(t * 256.0), 255u);
  atomicAdd(&bins[bin], 1u);
}

@compute @workgroup_size(256)
fn clear_bins(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < 256u) { atomicStore(&bins[gid.x], 0u); }
}
`;

// ── GPU Colormap Engine ──

interface Slot {
  width: number;
  height: number;
  count: number;
  dataBuffer: GPUBuffer;
  rgbaBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  paramsBuffer: GPUBuffer;
  histBinsBuffer: GPUBuffer;
  histReadBuffer: GPUBuffer;
}

function destroySlot(slot: Slot): void {
  slot.dataBuffer.destroy();
  slot.rgbaBuffer.destroy();
  slot.readBuffer.destroy();
  slot.paramsBuffer.destroy();
  slot.histBinsBuffer.destroy();
  slot.histReadBuffer.destroy();
}

/** Canvas format — a canvas configured with bgra8unorm. Prefer this because
 *  most desktop platforms use BGRA as the native surface format. */
export const PREFERRED_CANVAS_FORMAT: GPUTextureFormat = "bgra8unorm";

/** Round up `n` to the next multiple of `align`. */
function alignUp(n: number, align: number): number {
  return Math.ceil(n / align) * align;
}

export class GPUColormapEngine {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline | null = null;       // RGBA output (mapAsync readback path)
  private pipelineBgra: GPUComputePipeline | null = null;   // BGRA output (copyBufferToTexture path)
  private pipelineJch: GPUComputePipeline | null = null;    // Complex → JCh-like HSL colormap
  private histPipeline: GPUComputePipeline | null = null;
  private histClearPipeline: GPUComputePipeline | null = null;
  private lutBuffer: GPUBuffer | null = null;
  private currentLutName = "";
  private lutBuffers: Map<string, GPUBuffer> = new Map();
  private slots: (Slot | null)[] = [];
  /** Per-slot padded display buffer for applyToCanvas. Keyed by slot idx.
   *  Separate from rgbaBuffer because canvas copies need 256-byte row alignment. */
  private displayBuffers: Map<number, { buf: GPUBuffer; bytesPerRow: number; width: number; height: number }> = new Map();
  /** Configured canvas contexts — tracked so we don't reconfigure on every call. */
  private configuredCtxs: WeakSet<GPUCanvasContext> = new WeakSet();
  private retiredSlots: Slot[] = [];

  private _retireSlot(slot: Slot): void {
    this.retiredSlots.push(slot);
    void this.device.queue.onSubmittedWorkDone()
      .catch(() => {})
      .finally(() => {
        const idx = this.retiredSlots.indexOf(slot);
        if (idx >= 0) this.retiredSlots.splice(idx, 1);
        destroySlot(slot);
      });
  }

  constructor(device: GPUDevice) {
    this.device = device;
    const cmModule = device.createShaderModule({ code: COLORMAP_SHADER });
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: cmModule, entryPoint: "main" },
    });
    this.pipelineBgra = device.createComputePipeline({
      layout: "auto",
      compute: { module: cmModule, entryPoint: "main_bgra" },
    });
    const histModule = device.createShaderModule({ code: HISTOGRAM_SHADER });
    this.histPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: histModule, entryPoint: "histogram" },
    });
    this.histClearPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: histModule, entryPoint: "clear_bins" },
    });
    const jchModule = device.createShaderModule({ code: COMPLEX_JCH_SHADER });
    this.pipelineJch = device.createComputePipeline({
      layout: "auto",
      compute: { module: jchModule, entryPoint: "main" },
    });
  }

  /** Expose the underlying device so callers can configure a GPUCanvasContext. */
  getDevice(): GPUDevice { return this.device; }

  private _lutBufferFor(name: string): GPUBuffer | null {
    const cached = this.lutBuffers.get(name);
    if (cached) return cached;
    const reversed = name.endsWith("_r");
    const baseName = reversed ? name.slice(0, -2) : name;
    const lut = COLORMAPS[baseName];
    if (!lut) return null;
    // Pack RGB triplets into u32: R | (G << 8) | (B << 16).
    const packed = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      const j = reversed ? (255 - i) : i;
      packed[i] = lut[j * 3] | (lut[j * 3 + 1] << 8) | (lut[j * 3 + 2] << 16);
    }
    const buf = this.device.createBuffer({
      size: packed.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, packed);
    this.lutBuffers.set(name, buf);
    return buf;
  }

  /** Select a colormap LUT on the GPU. Buffers are cached per colormap instead
   *  of replacing/destroying one shared buffer; concurrent phase/FFT renders
   *  can therefore bind different stable LUTs without racing each other.
   *  ``_r`` suffix (e.g. ``"magma_r"``) reverses the colormap direction so
   *  bright-on-dark inverts to dark-on-bright — used by the F-flip toggle so
   *  flipped phase is visually unambiguous on near-symmetric distributions. */
  uploadLUT(name: string): void {
    const lutBuffer = this._lutBufferFor(name);
    if (!lutBuffer) return;
    this.lutBuffer = lutBuffer;
    this.currentLutName = name;
  }

  /** Upload float32 image data to a slot. Destroys previous slot buffers. */
  uploadData(idx: number, data: Float32Array, width: number, height: number): void {
    const old = this.slots[idx];
    if (old) {
      this._retireSlot(old);
    }
    // Display buffer is size-dependent; drop it if dimensions change.
    const displayExisting = this.displayBuffers.get(idx);
    if (displayExisting && (displayExisting.width !== width || displayExisting.height !== height)) {
      displayExisting.buf.destroy();
      this.displayBuffers.delete(idx);
    }
    const count = width * height;
    const dataBuffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(dataBuffer, 0, data.buffer, data.byteOffset, data.byteLength);

    const rgbaBuffer = this.device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readBuffer = this.device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const paramsBuffer = this.device.createBuffer({
      size: 24,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const histBinsBuffer = this.device.createBuffer({
      size: 256 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const histReadBuffer = this.device.createBuffer({
      size: 256 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.slots[idx] = { width, height, count, dataBuffer, rgbaBuffer, readBuffer, paramsBuffer, histBinsBuffer, histReadBuffer };
  }

  /** Point a slot at an EXTERNAL GPU float buffer (e.g. the maskedSum virtual-image buffer) with
   *  NO CPU upload. The slot OWNS `buffer` now (destroyed on the next adopt/upload/dispose), so
   *  the caller must hand off a fresh buffer each call. `applyToCanvas` then colormaps it straight
   *  to the canvas with zero readback - the GPU-resident 60fps drag path. The aux buffers (rgba/
   *  read/hist) are unused here, so they're allocated tiny just to keep the Slot shape uniform. */
  adoptBuffer(idx: number, buffer: GPUBuffer, width: number, height: number): void {
    const old = this.slots[idx];
    if (old) this._retireSlot(old);
    const displayExisting = this.displayBuffers.get(idx);
    if (displayExisting && (displayExisting.width !== width || displayExisting.height !== height)) {
      displayExisting.buf.destroy(); this.displayBuffers.delete(idx);
    }
    const tiny = () => this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    this.slots[idx] = {
      width, height, count: width * height, dataBuffer: buffer,
      rgbaBuffer: tiny(), readBuffer: this.device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      paramsBuffer: this.device.createBuffer({ size: 24, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      histBinsBuffer: tiny(), histReadBuffer: this.device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    };
  }

  private _writeParams(buf: ArrayBuffer, w: number, h: number, vmin: number, vmax: number, logScale: boolean, rowStrideU32 = 0): void {
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u[0] = w; u[1] = h; f[2] = vmin; f[3] = vmax; u[4] = logScale ? 1 : 0; u[5] = rowStrideU32;
  }

  /**
   * Apply colormap to a single slot and return RGBA as Uint8ClampedArray.
   * Uses mapAsync readback — ~3-5ms for typical images. Perfect for
   * real-time slider interaction.
   */
  async apply(idx: number, vmin: number, vmax: number, logScale = false): Promise<Uint8ClampedArray | null> {
    const slot = this.slots[idx];
    if (!slot || !this.pipeline || !this.lutBuffer) return null;

    const params = new ArrayBuffer(24);
    this._writeParams(params, slot.width, slot.height, vmin, vmax, logScale);
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

    const encoder = this.device.createCommandEncoder();
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: this.lutBuffer } },
        { binding: 3, resource: { buffer: slot.rgbaBuffer } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
    pass.end();
    encoder.copyBufferToBuffer(slot.rgbaBuffer, 0, slot.readBuffer, 0, slot.count * 4);
    this.device.queue.submit([encoder.finish()]);

    try {
      await slot.readBuffer.mapAsync(GPUMapMode.READ);
    } catch {
      return null;
    }
    const mapped = slot.readBuffer.getMappedRange();
    const result = new Uint8ClampedArray(mapped.slice(0));
    slot.readBuffer.unmap();
    return result;
  }

  /** Upload INTERLEAVED (re, im) complex float32 data for JCh rendering.
   *  The slot's dataBuffer is sized for 2× the pixel count. Separate from
   *  uploadData() so mixing scalar + complex usage on the same slot is
   *  explicit at the call site (mistakenly JCh-rendering a non-complex
   *  buffer produces garbage pixels). */
  uploadComplex(idx: number, data: Float32Array, width: number, height: number): void {
    const old = this.slots[idx];
    if (old) {
      this._retireSlot(old);
    }
    const existingDisplay = this.displayBuffers.get(idx);
    if (existingDisplay && (existingDisplay.width !== width || existingDisplay.height !== height)) {
      existingDisplay.buf.destroy();
      this.displayBuffers.delete(idx);
    }
    const count = width * height;
    const dataBuffer = this.device.createBuffer({
      // 2x float32 per pixel (re, im)
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(dataBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
    const rgbaBuffer = this.device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readBuffer = this.device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const paramsBuffer = this.device.createBuffer({
      size: 24,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Histogram buffers unused for complex slots but allocated so the
    // Slot shape stays uniform across the slot table.
    const histBinsBuffer = this.device.createBuffer({
      size: 256 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const histReadBuffer = this.device.createBuffer({
      size: 256 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.slots[idx] = { width, height, count, dataBuffer, rgbaBuffer, readBuffer, paramsBuffer, histBinsBuffer, histReadBuffer };
  }

  /** Render the JCh colormap from complex data uploaded via uploadComplex().
   *  `amin`/`amax` are the 2%-98% quantile amplitude clips (not log-amp) —
   *  matches quantem-core's default CustomNormalization. Caller samples a
   *  few thousand pixels of |z| and passes the percentile bounds directly. */
  async applyComplexJch(idx: number, amin: number, amax: number): Promise<Uint8ClampedArray | null> {
    const slot = this.slots[idx];
    if (!slot || !this.pipelineJch) return null;
    const params = new ArrayBuffer(24);
    const u = new Uint32Array(params);
    const f = new Float32Array(params);
    u[0] = slot.width; u[1] = slot.height; f[2] = amin; f[3] = amax; u[4] = 0; u[5] = 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);
    const encoder = this.device.createCommandEncoder();
    const bindGroup = this.device.createBindGroup({
      layout: this.pipelineJch.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: slot.rgbaBuffer } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelineJch);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
    pass.end();
    encoder.copyBufferToBuffer(slot.rgbaBuffer, 0, slot.readBuffer, 0, slot.count * 4);
    this.device.queue.submit([encoder.finish()]);
    try {
      await slot.readBuffer.mapAsync(GPUMapMode.READ);
    } catch {
      return null;
    }
    const result = new Uint8ClampedArray(slot.readBuffer.getMappedRange().slice(0));
    slot.readBuffer.unmap();
    return result;
  }

  /** Ensure the given slot has a padded display buffer matching its dimensions.
   *  Row-aligned to 256 bytes (the WebGPU copyBufferToTexture requirement). */
  private _ensureDisplayBuffer(idx: number, width: number, height: number): { buf: GPUBuffer; bytesPerRow: number } {
    const bytesPerRow = alignUp(width * 4, 256);
    const existing = this.displayBuffers.get(idx);
    if (existing && existing.width === width && existing.height === height && existing.bytesPerRow === bytesPerRow) {
      return { buf: existing.buf, bytesPerRow };
    }
    existing?.buf.destroy();
    const buf = this.device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.displayBuffers.set(idx, { buf, bytesPerRow, width, height });
    return { buf, bytesPerRow };
  }

  /**
   * Apply colormap to a slot and blit the RGBA result directly to a WebGPU
   * canvas context. Skips the mapAsync readback entirely — the GPU writes
   * into a padded buffer and we copyBufferToTexture into the canvas's
   * current swap-chain texture. Canvas is configured on first call.
   *
   * The canvas's CSS width/height are respected; its internal size is set
   * to (cols, rows) of the raw image so the compute shader writes 1:1 and
   * the 2D draw path scales as needed via drawImage.
   */
  applyToCanvas(
    idx: number, vmin: number, vmax: number,
    ctx: GPUCanvasContext, canvas: HTMLCanvasElement | OffscreenCanvas,
    logScale = false,
  ): void {
    const slot = this.slots[idx];
    if (!slot || !this.pipelineBgra || !this.lutBuffer) return;

    // Size the canvas's drawing buffer to the image so we get 1:1 pixels.
    // Reconfigure only when dimensions actually change.
    if (canvas.width !== slot.width) canvas.width = slot.width;
    if (canvas.height !== slot.height) canvas.height = slot.height;
    if (!this.configuredCtxs.has(ctx)) {
      ctx.configure({
        device: this.device,
        format: PREFERRED_CANVAS_FORMAT,
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        alphaMode: "opaque",
      });
      this.configuredCtxs.add(ctx);
    }

    const { buf: displayBuf, bytesPerRow } = this._ensureDisplayBuffer(idx, slot.width, slot.height);
    const rowStrideU32 = bytesPerRow / 4;

    const params = new ArrayBuffer(24);
    this._writeParams(params, slot.width, slot.height, vmin, vmax, logScale, rowStrideU32);
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

    const encoder = this.device.createCommandEncoder();
    const bindGroup = this.device.createBindGroup({
      layout: this.pipelineBgra.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: this.lutBuffer } },
        { binding: 3, resource: { buffer: displayBuf } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelineBgra);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
    pass.end();

    const tex = ctx.getCurrentTexture();
    encoder.copyBufferToTexture(
      { buffer: displayBuf, bytesPerRow, rowsPerImage: slot.height },
      { texture: tex },
      { width: slot.width, height: slot.height, depthOrArrayLayers: 1 },
    );
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Compute a 256-bin histogram on the GPU for the given data range.
   * Returns normalized bins (0–1, divided by max count).
   */
  async computeHistogram(idx: number, dmin: number, dmax: number, logScale = false): Promise<number[]> {
    const slot = this.slots[idx];
    if (!slot || !this.histPipeline || !this.histClearPipeline || dmin === dmax) {
      return new Array(256).fill(0);
    }

    const params = new ArrayBuffer(24);
    this._writeParams(params, slot.width, slot.height, dmin, dmax, logScale);
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

    const encoder = this.device.createCommandEncoder();

    // Clear bins
    const clearGroup = this.device.createBindGroup({
      layout: this.histClearPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: { buffer: slot.histBinsBuffer } },
      ],
    });
    const clearPass = encoder.beginComputePass();
    clearPass.setPipeline(this.histClearPipeline);
    clearPass.setBindGroup(0, clearGroup);
    clearPass.dispatchWorkgroups(1);
    clearPass.end();

    // Histogram
    const histGroup = this.device.createBindGroup({
      layout: this.histPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: slot.histBinsBuffer } },
      ],
    });
    const histPass = encoder.beginComputePass();
    histPass.setPipeline(this.histPipeline);
    histPass.setBindGroup(0, histGroup);
    histPass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
    histPass.end();

    encoder.copyBufferToBuffer(slot.histBinsBuffer, 0, slot.histReadBuffer, 0, 256 * 4);
    this.device.queue.submit([encoder.finish()]);

    try {
      await slot.histReadBuffer.mapAsync(GPUMapMode.READ);
    } catch {
      return new Array(256).fill(0);
    }
    const rawBins = new Uint32Array(slot.histReadBuffer.getMappedRange().slice(0));
    slot.histReadBuffer.unmap();

    let maxCount = 0;
    for (let i = 0; i < 256; i++) if (rawBins[i] > maxCount) maxCount = rawBins[i];
    const result = new Array(256);
    for (let i = 0; i < 256; i++) result[i] = maxCount > 0 ? rawBins[i] / maxCount : 0;
    return result;
  }

  /**
   * Release all GPU resources held for a single slot index. Call this from
   * a React component's `useEffect` cleanup when the component that owns
   * the slot unmounts. Without it, every mount/unmount cycle (page nav,
   * popup open/close) accumulates GPU buffers forever — eventually Chrome
   * runs out of GPU memory and kills the tab. Safe to call with an
   * already-released or never-allocated index.
   */
  releaseSlot(idx: number): void {
    if (idx < 0) return;
    const slot = this.slots[idx];
    if (slot) {
      this._retireSlot(slot);
      this.slots[idx] = null;
    }
    const display = this.displayBuffers.get(idx);
    if (display) {
      display.buf.destroy();
      this.displayBuffers.delete(idx);
    }
  }

  /** Release all GPU resources. */
  destroy(): void {
    for (const slot of this.slots) {
      if (slot) destroySlot(slot);
    }
    this.slots = [];
    for (const slot of this.retiredSlots) destroySlot(slot);
    this.retiredSlots = [];
    for (const d of this.displayBuffers.values()) d.buf.destroy();
    this.displayBuffers.clear();
    for (const buf of this.lutBuffers.values()) buf.destroy();
    this.lutBuffers.clear();
    this.lutBuffer = null;
    this.currentLutName = "";
  }
}

// ── Slot Allocator ──

// The GPU engine is a singleton keyed by numeric slot. Every concurrent
// consumer (ImageViewer instances, CompareDialog tiles, PanelViewer,
// Gallery, etc.) MUST hold its own unique slot or buffers will collide.
// Route ALL slot allocation through `allocateSlot()` — never hardcode an
// integer and never maintain a module-local counter in a consumer file.
let _globalSlot = 0;
export function allocateSlot(): number { return _globalSlot++; }

// ── Singleton ──

let _engine: GPUColormapEngine | null = null;
// Singleton Promise — started eagerly at module load so the engine is ready
// before the first ZoomDialog mounts. Callers await the same Promise
// regardless of how many concurrent waiters there are.
let _enginePromise: Promise<GPUColormapEngine | null> | null = null;

/** Get or create the singleton GPU colormap engine. Returns null if WebGPU unavailable. */
export function getGPUColormapEngine(): Promise<GPUColormapEngine | null> {
  if (_enginePromise) return _enginePromise;
  _enginePromise = (async () => {
    try {
      const device = await getGPUDevice();
      if (!device) return null;
      _engine = new GPUColormapEngine(device);
      return _engine;
    } catch {
      return null;
    }
  })();
  return _enginePromise;
}

// Kick off GPU init as soon as this module is imported — ZoomDialog mounts
// ~100ms after module load, so the engine is typically ready before the
// first open.
getGPUColormapEngine();

// ── CPU Fallback Utilities ──

/** Find min/max of a Float32Array (for percentile computation). */
export function findDataRange(data: Float32Array): { min: number; max: number } {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (min === Infinity) return { min: 0, max: 0 };
  return { min, max };
}

/**
 * Given sorted values and a percentile (0–100), return the value at that percentile.
 * sortedValues should be pre-sorted ascending for O(1) lookup.
 */
export function percentileFromSorted(sorted: Float32Array, p: number): number {
  if (sorted.length === 0) return 0;
  const pct = Number.isFinite(p) ? Math.max(0, Math.min(100, p)) : 0;
  const idx = Math.min(Math.floor((pct / 100) * sorted.length), sorted.length - 1);
  const value = sorted[Math.max(0, idx)];
  return Number.isFinite(value) ? value : 0;
}

/**
 * CPU fallback: apply colormap and return an ImageBitmap.
 * Used when WebGPU is unavailable.
 */
export function cpuColormapToImageData(
  data: Float32Array, width: number, height: number,
  vmin: number, vmax: number, cmapName: string,
): ImageData {
  const lut = COLORMAPS[cmapName] || COLORMAPS.inferno;
  const rgba = new Uint8ClampedArray(width * height * 4);
  const finiteMin = Number.isFinite(vmin) ? vmin : 0;
  const finiteMax = Number.isFinite(vmax) && vmax > finiteMin ? vmax : finiteMin + 1;
  applyColormap(data, rgba, lut, finiteMin, finiteMax);
  return new ImageData(rgba, width, height);
}

/**
 * CPU fallback: compute a 256-bin histogram, normalized to 0–1.
 */
export function cpuHistogram(data: Float32Array, dmin: number, dmax: number): number[] {
  const bins = new Uint32Array(256);
  const min = Number.isFinite(dmin) ? dmin : 0;
  const max = Number.isFinite(dmax) && dmax > min ? dmax : min + 1;
  const range = max - min;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    const t = Math.max(0, Math.min(1, (v - min) / range));
    bins[Math.min(255, Math.floor(t * 256))]++;
  }
  let maxCount = 0;
  for (let i = 0; i < 256; i++) if (bins[i] > maxCount) maxCount = bins[i];
  const result = new Array(256);
  for (let i = 0; i < 256; i++) result[i] = maxCount > 0 ? bins[i] / maxCount : 0;
  return result;
}
