// GPU min/max reductions and 256-bin histograms of the engine's slots, and the
// range-aware colormap passes that consume a reduced range without a readback.
import type { GPUColormapEngine, GPUSlot } from "./colormapEngine";
import { DISPLAY_NORMALIZE_WGSL, RANGE_REDUCE_WGSL } from "./displayShaders";
import { computePipeline } from "./pipelines";

/** Range and histogram passes over the engine's slots, with their pipelines and scratch buffers. */
export class GPURanges {
  private readonly engine: GPUColormapEngine;
  private readonly device: GPUDevice;

  constructor(engine: GPUColormapEngine) {
    this.engine = engine;
    this.device = engine.getDevice();
  }

  // ── GPU min/max reduction ──

  private rangePipeline: GPUComputePipeline | null = null;
  private rangeUint8Pipeline: GPUComputePipeline | null = null;
  private RANGE_WG_SIZE = 256;
  private rangeFinishPipeline: GPUComputePipeline | null = null;

  /** Reduce the current scalar frame and keep both passes and their storage resident. */
  recordLiveRange(encoder: GPUCommandEncoder, slot: GPUSlot, logScale: boolean): void {
    this.ensureRangePipeline();
    if (!this.rangeFinishPipeline) {
      this.rangeFinishPipeline = computePipeline(this.device, /* wgsl */ `
struct Config { count: u32, groups: u32, logScale: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> partials: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> config: Config;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
var<workgroup> lower: array<f32, 256>;
var<workgroup> upper: array<f32, 256>;
fn scaled(value: f32) -> f32 {
  if (config.logScale == 0u) { return value; }
  if (value >= 0.0) { return log(1.0 + value); }
  return -log(1.0 - value);
}
@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_index) lane: u32) {
  var lo = 1.0e38;
  var hi = -1.0e38;
  for (var index = lane; index < config.groups; index += 256u) {
    lo = min(lo, partials[index].x);
    hi = max(hi, partials[index].y);
  }
  lower[lane] = lo;
  upper[lane] = hi;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      lower[lane] = min(lower[lane], lower[lane + stride]);
      upper[lane] = max(upper[lane], upper[lane + stride]);
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    if (upper[0] >= lower[0]) {
      output[0] = vec4<f32>(scaled(lower[0]), scaled(upper[0]), 0.0, 0.0);
    } else { output[0] = vec4<f32>(0.0); }
  }
}
`, "reduce");
    }
    if (!slot.liveRange) {
      const groups = Math.min(Math.ceil(slot.count / this.RANGE_WG_SIZE), this.device.limits.maxComputeWorkgroupsPerDimension);
      const partials = this.device.createBuffer({ size: Math.max(8, groups * 8), usage: GPUBufferUsage.STORAGE });
      // One block for both passes: the partial pass reads it as RangeParams
      // { count, groups } (its grid stride), the finish pass as Config.
      const parameters = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const values = new Uint32Array([slot.count, groups, logScale ? 1 : 0, 0]);
      this.device.queue.writeBuffer(parameters, 0, values);
      slot.liveRange = {
        groups, partials, parameters, values, renderGroup: null,
        reduceGroup: this.device.createBindGroup({
          layout: this.rangePipeline!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: slot.dataBuffer } },
            { binding: 1, resource: { buffer: partials } },
            { binding: 2, resource: { buffer: parameters } },
          ],
        }),
        finishGroup: this.device.createBindGroup({
          layout: this.rangeFinishPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: partials } },
            { binding: 1, resource: { buffer: parameters } },
            { binding: 2, resource: { buffer: this.ensureSlotRangeBuffer(slot) } },
          ],
        }),
      };
    }
    const state = slot.liveRange;
    if (state.values[2] !== Number(logScale)) {
      state.values[2] = Number(logScale);
      this.device.queue.writeBuffer(state.parameters, 0, state.values);
    }
    const partialPass = encoder.beginComputePass();
    partialPass.setPipeline(this.rangePipeline!);
    partialPass.setBindGroup(0, state.reduceGroup);
    partialPass.dispatchWorkgroups(state.groups);
    partialPass.end();
    const finishPass = encoder.beginComputePass();
    finishPass.setPipeline(this.rangeFinishPipeline);
    finishPass.setBindGroup(0, state.finishGroup);
    finishPass.dispatchWorkgroups(1);
    finishPass.end();
  }

  private ensureRangePipeline(): void {
    if (this.rangePipeline) return;
    // Two-pass parallel reduction: each workgroup reduces a chunk to one min/max pair.
    // Output: array of [min, max] pairs (one per workgroup). JS reduces the partials.
    const code = DISPLAY_NORMALIZE_WGSL + RANGE_REDUCE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: RangeParams;

var<workgroup> sMin: array<f32, 256>;
var<workgroup> sMax: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  var local_min = 1.0e38;
  var local_max = -1.0e38;
  let stride = range_stride();
  let steps = range_steps(stride);
  for (var step = 0u; step < steps; step = step + 1u) {
    let i = gid.x + step * stride;
    if (i >= params.count) { break; }
    let value = data[i];
    if (display_is_finite(value)) {
      local_min = min(local_min, value);
      local_max = max(local_max, value);
    }
  }
  sMin[lid.x] = local_min;
  sMax[lid.x] = local_max;

  workgroupBarrier();

  // Tree reduction in shared memory
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      sMin[lid.x] = min(sMin[lid.x], sMin[lid.x + s]);
      sMax[lid.x] = max(sMax[lid.x], sMax[lid.x + s]);
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    out[wid.x * 2u] = sMin[0];
    out[wid.x * 2u + 1u] = sMax[0];
  }
}
`;
    this.rangePipeline = computePipeline(this.device, code, "reduce");
  }

  private ensureRangeUint8Pipeline(): void {
    if (this.rangeUint8Pipeline) return;
    const code = RANGE_REDUCE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<storage, read> packed_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: RangeParams;

var<workgroup> sMin: array<f32, 256>;
var<workgroup> sMax: array<f32, 256>;

fn read_u8(index: u32) -> f32 {
  let word = packed_data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return f32((word >> shift) & 255u);
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  var local_min = 1.0e38;
  var local_max = -1.0e38;
  let stride = range_stride();
  let steps = range_steps(stride);
  for (var step = 0u; step < steps; step = step + 1u) {
    let i = gid.x + step * stride;
    if (i >= params.count) { break; }
    let value = read_u8(i);
    local_min = min(local_min, value);
    local_max = max(local_max, value);
  }
  sMin[lid.x] = local_min;
  sMax[lid.x] = local_max;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      sMin[lid.x] = min(sMin[lid.x], sMin[lid.x + s]);
      sMax[lid.x] = max(sMax[lid.x], sMax[lid.x + s]);
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    out[wid.x * 2u] = sMin[0];
    out[wid.x * 2u + 1u] = sMax[0];
  }
}
`;
    this.rangeUint8Pipeline = computePipeline(this.device, code, "reduce");
  }

  /**
   * Batch-compute min/max for multiple slots on GPU.
   * Returns { min, max } per slot. One GPU submission for all slots.
   */
  async computeRangeBatch(indices: number[]): Promise<{ min: number; max: number }[]> {
    this.ensureRangePipeline();
    this.ensureRangeUint8Pipeline();
    if (!this.rangePipeline || !this.rangeUint8Pipeline || indices.length === 0) return [];
    const encoder = this.device.createCommandEncoder();
    const jobs: { idx: number; nGroups: number; outBuf: GPUBuffer; readBuf: GPUBuffer; countBuf: GPUBuffer }[] = [];

    for (const idx of indices) {
      const slot = this.engine.slots[idx];
      if (!slot) continue;
      const pipeline = slot.dataKind === "u8" ? this.rangeUint8Pipeline : this.rangePipeline;
      const nGroups = Math.min(
        Math.ceil(slot.count / this.RANGE_WG_SIZE),
        this.device.limits.maxComputeWorkgroupsPerDimension,
      );
      const outSize = nGroups * 2 * 4; // 2 floats (min, max) per workgroup
      const outBuf = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readBuf = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      // RangeParams { count, groups }: the shader's grid stride is groups * 256.
      const countBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(countBuf, 0, new Uint32Array([slot.count, nGroups, 0, 0]));

      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.dataBuffer } },
          { binding: 1, resource: { buffer: outBuf } },
          { binding: 2, resource: { buffer: countBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(nGroups);
      pass.end();
      encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outSize);
      jobs.push({ idx, nGroups, outBuf, readBuf, countBuf });
    }

    this.device.queue.submit([encoder.finish()]);
    await Promise.all(jobs.map(job => job.readBuf.mapAsync(GPUMapMode.READ)));

    const results: { min: number; max: number }[] = [];
    for (const job of jobs) {
      const partials = new Float32Array(job.readBuf.getMappedRange().slice(0));
      job.readBuf.unmap();
      job.outBuf.destroy(); job.readBuf.destroy(); job.countBuf.destroy();
      // The partials are at most 65535 pairs, so finishing the reduction in JS is cheap.
      let dmin = Infinity, dmax = -Infinity;
      for (let k = 0; k < job.nGroups; k++) {
        if (partials[k * 2] < dmin) dmin = partials[k * 2];
        if (partials[k * 2 + 1] > dmax) dmax = partials[k * 2 + 1];
      }
      if (!(dmax >= dmin)) { dmin = 0; dmax = 0; }
      results.push({ min: dmin, max: dmax });
    }
    return results;
  }

  // ── GPU region min/max → range-aware colormap (no CPU readback) ──
  //
  // Used by Show3D per-panel contrast: each panel is a sub-region of one full
  // frame buffer. We avoid the JS slab-extract + findDataRange loop entirely
  // by reducing on GPU and feeding the result straight into the colormap pass
  // via a small storage buffer (no mapAsync between the two passes).

  private rangeRegionPipeline: GPUComputePipeline | null = null;
  private colormapRangePipeline: GPUComputePipeline | null = null;
  // Per-panel scratch state for renderPerPanelGpuExplicit and
  // computeHistogramRegions when N panels share ONE GPU slot (full frame).
  // Each entry holds the panel-sized rgba output buffer and the 16-byte range
  // buffer. Keyed by panel index.
  private panelRgbaBuffers: Map<number, { rgba: GPUBuffer; range: GPUBuffer; size: number }> = new Map();

  private ensurePanelScratch(panel: number, panelPixels: number): { rgba: GPUBuffer; range: GPUBuffer } {
    const want = panelPixels * 4;
    const existing = this.panelRgbaBuffers.get(panel);
    if (existing && existing.size === want) return existing;
    if (existing) { existing.rgba.destroy(); existing.range.destroy(); }
    const rgba = this.device.createBuffer({
      size: want,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const range = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const entry = { rgba, range, size: want };
    this.panelRgbaBuffers.set(panel, entry);
    return entry;
  }

  /** One workgroup reduces a rectangular region of a slot to its finite (optionally log) min/max. */
  private ensureRangeRegionPipeline(): void {
    this.rangeRegionPipeline ??= computePipeline(this.device, DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct RangeOut { vmin: f32, vmax: f32, _p0: f32, _p1: f32 };
struct RegionParams { region: vec4u, fullWidth: u32, log_scale: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<uniform> params: RegionParams;
@group(0) @binding(2) var<storage, read_write> out: RangeOut;
var<workgroup> sMin: array<f32, 256>;
var<workgroup> sMax: array<f32, 256>;
@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_index) lid: u32) {
  var low = 1.0e38; var high = -1.0e38;
  let width = params.region.z;
  let count = width * params.region.w;
  for (var index = lid; index < count; index += 256u) {
    let row = index / width; let col = index - row * width;
    var value = data[(params.region.y + row) * params.fullWidth + params.region.x + col];
    if (!display_is_finite(value)) { continue; }
    if (params.log_scale == 1u) {
      if (value >= 0.0) { value = log(1.0 + value); } else { value = -log(1.0 - value); }
    }
    if (value < low) { low = value; }
    if (value > high) { high = value; }
  }
  sMin[lid] = low; sMax[lid] = high;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lid < stride) {
      sMin[lid] = min(sMin[lid], sMin[lid + stride]);
      sMax[lid] = max(sMax[lid], sMax[lid + stride]);
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    if (sMax[0] >= sMin[0]) { out.vmin = sMin[0]; out.vmax = sMax[0]; }
    else { out.vmin = 0.0; out.vmax = 0.0; }
    out._p0 = 0.0; out._p1 = 0.0;
  }
}
`, "reduce");
  }

  private ensureColormapRangePipeline(): void {
    if (this.colormapRangePipeline) return;
    // Same as COLORMAP_SHADER but reads vmin/vmax from a storage buffer
    // and applies the user slider percentages on GPU so those scalars never
    // round-trip through JS.
    // Also accepts a region (offset + size into the full data buffer) and a
    // stride so the colormap output is the panel sub-image, sourced from
    // the full frame in place, with no slab extraction in JS.
    const code = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct Params {
  width: u32,        // output (panel) width
  height: u32,       // output (panel) height
  vmin_pct: f32,
  vmax_pct: f32,
  log_scale: u32,
  src_x: u32,        // region offset x in source data
  src_y: u32,        // region offset y in source data
  src_stride: u32,   // row stride of source data
};
struct RangeOut { vmin: f32, vmax: f32, _p0: f32, _p1: f32 };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;
@group(0) @binding(4) var<storage, read> range_in: RangeOut;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let out_idx = gid.y * params.width + gid.x;
  let src_idx = (params.src_y + gid.y) * params.src_stride + (params.src_x + gid.x);
  var val = data[src_idx];
  if (params.log_scale == 1u) {
    if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); }
  }
  let span = range_in.vmax - range_in.vmin;
  let vmin = range_in.vmin + span * (params.vmin_pct / 100.0);
  let vmax = range_in.vmin + span * (params.vmax_pct / 100.0);
  let t = display_normalize(val, vmin, vmax);
  let lutIdx = min(u32(t * 255.0), 255u);
  let rgb = lut[lutIdx];
  rgba[out_idx] = rgb | 0xFF000000u;
}
`;
    this.colormapRangePipeline = computePipeline(this.device, code);
  }

  ensureSlotRangeBuffer(slot: GPUSlot): GPUBuffer {
    if (!slot.rangeBuffer) {
      slot.rangeBuffer = this.device.createBuffer({
        // 4 floats: vmin, vmax, _p0, _p1
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }
    return slot.rangeBuffer;
  }

  /**
   * Render panel sub-regions with explicit per-panel ranges. Used when
   * Show3D contrast is unlinked and each histogram owns its own clip state.
   */
  renderPerPanelGpuExplicit(
    slotIdx: number,
    regions: { x: number; y: number; width: number; height: number }[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean | boolean[] = false,
  ): ImageBitmap[] | null {
    this.ensureColormapRangePipeline();
    if (!this.colormapRangePipeline || !this.engine.lutBuffer) return null;
    const slot = this.engine.slots[slotIdx];
    if (!slot || regions.length === 0) return null;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.engine.ensureBlitPipeline(format);
    if (!this.engine.blitPipeline) return null;

    const encoder = this.device.createCommandEncoder();
    const colormapParams = new ArrayBuffer(32);
    const canvases: (OffscreenCanvas | null)[] = [];
    const tempBuffers: GPUBuffer[] = [];

    for (let k = 0; k < regions.length; k++) {
      const region = regions[k];
      const panelRange = ranges[k] ?? ranges[0];
      if (!region || !panelRange) { canvases.push(null); continue; }
      const scratch = this.ensurePanelScratch(k, region.width * region.height);
      this.device.queue.writeBuffer(
        scratch.range,
        0,
        new Float32Array([panelRange.vmin, panelRange.vmax, 0, 0]),
      );

      const paramsU32 = new Uint32Array(colormapParams);
      const paramsF32 = new Float32Array(colormapParams);
      paramsU32[0] = region.width; paramsU32[1] = region.height;
      paramsF32[2] = 0; paramsF32[3] = 100;
      paramsU32[4] = Array.isArray(logScale) ? (logScale[k] ? 1 : 0) : (logScale ? 1 : 0);
      paramsU32[5] = region.x; paramsU32[6] = region.y; paramsU32[7] = slot.width;
      const colormapParamsBuffer = this.device.createBuffer({
        size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(colormapParamsBuffer, 0, colormapParams);
      tempBuffers.push(colormapParamsBuffer);

      const colormapGroup = this.device.createBindGroup({
        layout: this.colormapRangePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: colormapParamsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.engine.lutBuffer } },
          { binding: 3, resource: { buffer: scratch.rgba } },
          { binding: 4, resource: { buffer: scratch.range } },
        ],
      });
      const colormapPass = encoder.beginComputePass();
      colormapPass.setPipeline(this.colormapRangePipeline);
      colormapPass.setBindGroup(0, colormapGroup);
      colormapPass.dispatchWorkgroups(Math.ceil(region.width / 16), Math.ceil(region.height / 16));
      colormapPass.end();
      const blit = this.engine.encodeBlitToOffscreen(encoder, scratch.rgba, region.width, region.height, format)!;
      tempBuffers.push(blit.blitParamsBuffer);
      canvases.push(blit.canvas);
    }

    this.device.queue.submit([encoder.finish()]);
    for (const buffer of tempBuffers) buffer.destroy();
    return this.engine.transferOffscreens(canvases);
  }

  // ── GPU histogram ──

  private histPipeline: GPUComputePipeline | null = null;
  private histUint8Pipeline: GPUComputePipeline | null = null;
  private histClearPipeline: GPUComputePipeline | null = null;
  private histRegionPipeline: GPUComputePipeline | null = null;

  private ensureHistPipeline(): void {
    if (this.histPipeline) return;
    const code = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
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
  if (!display_is_finite(val)) { return; }
  if (params.log_scale == 1u) { if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); } }
  let t = display_normalize(val, params.dmin, params.dmax);
  let bin = min(u32(t * 256.0), 255u);
  atomicAdd(&bins[bin], 1u);
}

@compute @workgroup_size(256)
fn clear_bins(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < 256u) { atomicStore(&bins[gid.x], 0u); }
}
`;
    this.histPipeline = computePipeline(this.device, code, "histogram");
    this.histClearPipeline = computePipeline(this.device, code, "clear_bins");
  }

  private ensureHistUint8Pipeline(): void {
    if (this.histUint8Pipeline) return;
    const code = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct HistParams {
  width: u32,
  height: u32,
  dmin: f32,
  dmax: f32,
  log_scale: u32,
  _pad: u32,
};
@group(0) @binding(0) var<uniform> params: HistParams;
@group(0) @binding(1) var<storage, read> packed_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>>;

fn read_u8(index: u32) -> f32 {
  let word = packed_data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return f32((word >> shift) & 255u);
}

@compute @workgroup_size(16, 16)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var val = read_u8(idx);
  if (params.log_scale == 1u) { val = log(1.0 + val); }
  let t = display_normalize(val, params.dmin, params.dmax);
  let bin = min(u32(t * 256.0), 255u);
  atomicAdd(&bins[bin], 1u);
}
`;
    this.histUint8Pipeline = computePipeline(this.device, code, "histogram");
  }

  private ensureHistRegionPipeline(): void {
    if (this.histRegionPipeline) return;
    const code = DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct RegionParams {
  region: vec4u,
  full_width: u32,
  log_scale: u32,
  _pad0: u32,
  _pad1: u32,
};
struct RangeIn { vmin: f32, vmax: f32, _p0: f32, _p1: f32 };

@group(0) @binding(0) var<uniform> params: RegionParams;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> range_in: RangeIn;
@group(0) @binding(3) var<storage, read_write> bins: array<atomic<u32>>;

@compute @workgroup_size(16, 16)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.region.z || gid.y >= params.region.w) { return; }
  let idx = (params.region.y + gid.y) * params.full_width + params.region.x + gid.x;
  var val = data[idx];
  if (!display_is_finite(val)) { return; }
  if (params.log_scale == 1u) { if (val >= 0.0) { val = log(1.0 + val); } else { val = -log(1.0 - val); } }
  let t = display_normalize(val, range_in.vmin, range_in.vmax);
  let bin = min(u32(t * 256.0), 255u);
  atomicAdd(&bins[bin], 1u);
}
`;
    this.histRegionPipeline = computePipeline(this.device, code, "histogram");
  }

  /**
   * Compute ranges and 256-bin histograms for rectangular regions of one slot.
   *
   * Show3D packs independent panels side-by-side in one resident frame slot.
   * This keeps playback histogram refreshes on WebGPU: range reduction and bin
   * accumulation happen in one submission, with only the small ranges and bins
   * read back for drawing the histogram controls.
   */
  async computeHistogramRegions(
    idx: number,
    regions: { x: number; y: number; width: number; height: number }[],
    logScale: boolean = false,
  ): Promise<{ range: { min: number; max: number }; bins: number[] }[]> {
    this.ensureRangeRegionPipeline();
    this.ensureHistRegionPipeline();
    const slot = this.engine.slots[idx];
    if (!slot || !this.rangeRegionPipeline || !this.histRegionPipeline || regions.length === 0) {
      return [];
    }

    const validRegions = regions.map(region => {
      const x = Math.max(0, Math.min(slot.width - 1, Math.round(region.x)));
      const y = Math.max(0, Math.min(slot.height - 1, Math.round(region.y)));
      return {
        x,
        y,
        width: Math.max(1, Math.min(slot.width - x, Math.round(region.width))),
        height: Math.max(1, Math.min(slot.height - y, Math.round(region.height))),
      };
    });
    const binsBytes = validRegions.length * 256 * 4;
    const rangesBytes = validRegions.length * 16;
    const binsBuffer = this.device.createBuffer({
      size: binsBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const binsReadBuffer = this.device.createBuffer({
      size: binsBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const rangesReadBuffer = this.device.createBuffer({
      size: rangesBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const paramsBuffers: GPUBuffer[] = [];
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(binsBuffer);

    for (let k = 0; k < validRegions.length; k++) {
      const region = validRegions[k];
      const scratch = this.ensurePanelScratch(k, region.width * region.height);
      const paramsBuffer = this.device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(
        paramsBuffer,
        0,
        new Uint32Array([
          region.x,
          region.y,
          region.width,
          region.height,
          slot.width,
          logScale ? 1 : 0,
          0,
          0,
        ]),
      );
      paramsBuffers.push(paramsBuffer);

      const rangeGroup = this.device.createBindGroup({
        layout: this.rangeRegionPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.dataBuffer } },
          { binding: 1, resource: { buffer: paramsBuffer } },
          { binding: 2, resource: { buffer: scratch.range } },
        ],
      });
      const rangePass = encoder.beginComputePass();
      rangePass.setPipeline(this.rangeRegionPipeline);
      rangePass.setBindGroup(0, rangeGroup);
      rangePass.dispatchWorkgroups(1);
      rangePass.end();

      const histogramGroup = this.device.createBindGroup({
        layout: this.histRegionPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: scratch.range } },
          {
            binding: 3,
            resource: { buffer: binsBuffer, offset: k * 256 * 4, size: 256 * 4 },
          },
        ],
      });
      const histogramPass = encoder.beginComputePass();
      histogramPass.setPipeline(this.histRegionPipeline);
      histogramPass.setBindGroup(0, histogramGroup);
      histogramPass.dispatchWorkgroups(
        Math.ceil(region.width / 16),
        Math.ceil(region.height / 16),
      );
      histogramPass.end();
      encoder.copyBufferToBuffer(scratch.range, 0, rangesReadBuffer, k * 16, 16);
    }

    encoder.copyBufferToBuffer(binsBuffer, 0, binsReadBuffer, 0, binsBytes);
    this.device.queue.submit([encoder.finish()]);

    for (const buffer of paramsBuffers) buffer.destroy();
    binsBuffer.destroy();
    await Promise.all([
      binsReadBuffer.mapAsync(GPUMapMode.READ),
      rangesReadBuffer.mapAsync(GPUMapMode.READ),
    ]);
    const rawBins = new Uint32Array(binsReadBuffer.getMappedRange().slice(0));
    const rawRanges = new Float32Array(rangesReadBuffer.getMappedRange().slice(0));
    binsReadBuffer.destroy();
    rangesReadBuffer.destroy();
    return validRegions.map((_, k) => {
      const offset = k * 256;
      let maxCount = 0;
      for (let bin = 0; bin < 256; bin++) {
        maxCount = Math.max(maxCount, rawBins[offset + bin]);
      }
      const bins = new Array<number>(256);
      for (let bin = 0; bin < 256; bin++) {
        bins[bin] = maxCount > 0 ? rawBins[offset + bin] / maxCount : 0;
      }
      return {
        range: { min: rawRanges[k * 4], max: rawRanges[k * 4 + 1] },
        bins,
      };
    });
  }

  /**
   * Batch-compute 256-bin histograms for multiple slots in ONE GPU submission.
   * Uses persistent per-slot histogram buffers (zero create/destroy overhead).
   * Returns normalized bins per image.
   */
  async computeHistogramBatch(
    indices: number[],
    ranges: { min: number; max: number }[],
    logScale: boolean = false,
  ): Promise<number[][]> {
    this.ensureHistPipeline();
    this.ensureHistUint8Pipeline();
    if (!this.histPipeline || !this.histUint8Pipeline || !this.histClearPipeline || indices.length === 0) return [];

    const encoder = this.device.createCommandEncoder();
    const activeSlots: { k: number; slot: GPUSlot; readBuffer: GPUBuffer }[] = [];
    const params = new ArrayBuffer(24);

    for (let k = 0; k < indices.length; k++) {
      const slot = this.engine.slots[indices[k]];
      if (!slot) continue;
      const histogramPipeline = slot.dataKind === "u8" ? this.histUint8Pipeline : this.histPipeline;
      const range = ranges[k] || { min: 0, max: 1 };
      // The histogram params share the colormap params layout.
      this.engine.writeColormapParams(params, slot.width, slot.height, range.min, range.max, logScale);
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

      const clearGroup = this.device.createBindGroup({
        layout: this.histClearPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 2, resource: { buffer: slot.histBinsBuffer } },
        ],
      });
      const clearPass = encoder.beginComputePass();
      clearPass.setPipeline(this.histClearPipeline!);
      clearPass.setBindGroup(0, clearGroup);
      clearPass.dispatchWorkgroups(1);
      clearPass.end();

      const histGroup = this.device.createBindGroup({
        layout: histogramPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: slot.histBinsBuffer } },
        ],
      });
      const histPass = encoder.beginComputePass();
      histPass.setPipeline(histogramPipeline);
      histPass.setBindGroup(0, histGroup);
      histPass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
      histPass.end();

      // A per-call readback buffer: one persistent buffer per slot made
      // overlapping histogram requests fail with "outstanding map pending"
      // when Show2D refreshed auto-contrast and the visible histogram during
      // fast interaction.
      const readBuffer = this.device.createBuffer({
        size: 256 * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      encoder.copyBufferToBuffer(slot.histBinsBuffer, 0, readBuffer, 0, 256 * 4);
      activeSlots.push({ k, slot, readBuffer });
    }

    this.device.queue.submit([encoder.finish()]);

    const results: number[][] = indices.map(() => new Array(256).fill(0));
    await Promise.all(activeSlots.map(active => active.readBuffer.mapAsync(GPUMapMode.READ)));
    for (const active of activeSlots) {
      const rawBins = new Uint32Array(active.readBuffer.getMappedRange().slice(0));
      active.readBuffer.destroy();
      let maxCount = 0;
      for (let bin = 0; bin < 256; bin++) if (rawBins[bin] > maxCount) maxCount = rawBins[bin];
      const norm = new Array(256);
      for (let bin = 0; bin < 256; bin++) norm[bin] = maxCount > 0 ? rawBins[bin] / maxCount : 0;
      results[active.k] = norm;
    }
    return results;
  }

  /**
   * Compute a 256-bin histogram for slot `idx` on GPU, given known data range.
   * Returns normalized bins (0–1) matching `computeHistogramFromBytes`.
   */
  async computeHistogramWithRange(
    idx: number, dmin: number, dmax: number, logScale: boolean = false,
  ): Promise<number[]> {
    const [bins] = await this.computeHistogramBatch([idx], [{ min: dmin, max: dmax }], logScale);
    return bins ?? new Array(256).fill(0);
  }

  /** Release the per-panel scratch buffers; the engine's destroy calls this. */
  destroy(): void {
    for (const scratch of this.panelRgbaBuffers.values()) { scratch.rgba.destroy(); scratch.range.destroy(); }
    this.panelRgbaBuffers.clear();
  }
}
