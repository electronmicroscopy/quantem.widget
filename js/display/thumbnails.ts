// Frame thumbnails drawn on the GPU straight from a resident slot.
//
// A thumbnail never copies the full-size frame: one compute pass averages each
// source block of every panel into a small buffer (the thumbnail's own pixels),
// one workgroup per panel finds that panel's range, and a render pass colormaps
// the small buffer into the thumbnail canvas. Nothing is read back to the CPU,
// and the viewer's slots, ranges and data are left as they are.
import type { GPUColormapEngine } from "./colormapEngine";
import { computePipeline, renderPipeline } from "./pipelines";

// Samples per block side: a 2048 px panel shown 100 px wide averages a 20x20 block from 8x8 samples.
const SAMPLES_PER_SIDE = 8;
const MAX_PANELS = 64;

const THUMBNAIL_SHADER = /* wgsl */ `
struct Params {
  src_width: u32,        // slot row length (all panels side by side)
  src_height: u32,
  panel_src_width: u32,  // one panel's width in the slot
  panel_stride: u32,     // slot columns between panels (0: every panel reads the same source)
  panels: u32,
  thumb_width: u32,      // one panel's thumbnail width
  thumb_height: u32,
  log_scale: u32,
  gap: u32,
  bg_rgb: u32,
  samples: u32,
  panel_offset: u32,   // slot column where the first drawn panel starts
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> small: array<f32>;
@group(0) @binding(3) var<storage, read_write> ranges: array<f32>;

fn finite(v: f32) -> bool { return v == v && abs(v) <= 3.0e38; }

fn shown(v: f32) -> f32 {
  if (params.log_scale == 1u) { return sign(v) * log(1.0 + abs(v)); }
  return v;
}

@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x; let py = gid.y; let panel = gid.z;
  if (px >= params.thumb_width || py >= params.thumb_height || panel >= params.panels) { return; }
  let offset = params.panel_offset + panel * params.panel_stride;
  let x0 = px * params.panel_src_width / params.thumb_width;
  let x1 = max(x0 + 1u, (px + 1u) * params.panel_src_width / params.thumb_width);
  let y0 = py * params.src_height / params.thumb_height;
  let y1 = max(y0 + 1u, (py + 1u) * params.src_height / params.thumb_height);
  let step_x = max(1u, (x1 - x0) / params.samples);
  let step_y = max(1u, (y1 - y0) / params.samples);
  var total = 0.0;
  var count = 0.0;
  for (var y = y0; y < y1; y += step_y) {
    for (var x = x0; x < x1; x += step_x) {
      let v = src[y * params.src_width + offset + x];
      if (finite(v)) { total += v; count += 1.0; }
    }
  }
  var value = -3.4e38; // no finite pixel in the block: drawn as background, ignored by the range
  if (count > 0.0) { value = shown(total / count); }
  small[(panel * params.thumb_height + py) * params.thumb_width + px] = value;
}

var<workgroup> lows: array<f32, 256>;
var<workgroup> highs: array<f32, 256>;
var<workgroup> sums: array<f32, 256>;
var<workgroup> squares: array<f32, 256>;
var<workgroup> counts: array<f32, 256>;

@compute @workgroup_size(256)
fn panel_range(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let panel = wid.x;
  let n = params.thumb_width * params.thumb_height;
  let base = panel * n;
  var lo = 3.0e38;
  var hi = -3.0e38;
  var total = 0.0;
  var square = 0.0;
  var count = 0.0;
  for (var i = lid; i < n; i += 256u) {
    let v = small[base + i];
    if (finite(v)) { lo = min(lo, v); hi = max(hi, v); total += v; square += v * v; count += 1.0; }
  }
  lows[lid] = lo; highs[lid] = hi; sums[lid] = total; squares[lid] = square; counts[lid] = count;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lid < stride) {
      lows[lid] = min(lows[lid], lows[lid + stride]);
      highs[lid] = max(highs[lid], highs[lid + stride]);
      sums[lid] += sums[lid + stride];
      squares[lid] += squares[lid + stride];
      counts[lid] += counts[lid + stride];
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    // mean +- 3 standard deviations, inside the min and max: a few bright pixels do not flatten the rest
    let c = max(counts[0], 1.0);
    let mean = sums[0] / c;
    let sigma = sqrt(max(squares[0] / c - mean * mean, 0.0));
    ranges[panel * 2u] = max(lows[0], mean - 3.0 * sigma);
    ranges[panel * 2u + 1u] = min(highs[0], mean + 3.0 * sigma);
  }
}
`;

const THUMBNAIL_DRAW_SHADER = /* wgsl */ `
struct Params {
  src_width: u32, src_height: u32, panel_src_width: u32, panel_stride: u32,
  panels: u32, thumb_width: u32, thumb_height: u32, log_scale: u32,
  gap: u32, bg_rgb: u32, samples: u32, panel_offset: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> small: array<f32>;
@group(0) @binding(2) var<storage, read> ranges: array<f32>;
@group(0) @binding(3) var<storage, read> lut: array<u32>;

struct VSOut { @builtin(position) pos: vec4f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
  out.pos = vec4f(x, y, 0.0, 1.0);
  return out;
}

fn rgb(packed: u32) -> vec4f {
  return vec4f(f32(packed & 0xFFu) / 255.0, f32((packed >> 8u) & 0xFFu) / 255.0, f32((packed >> 16u) & 0xFFu) / 255.0, 1.0);
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let x = u32(in.pos.x);
  let y = min(u32(in.pos.y), params.thumb_height - 1u);
  let cell = params.thumb_width + params.gap;
  let panel = x / cell;
  let px = x % cell;
  if (panel >= params.panels || px >= params.thumb_width) { return rgb(params.bg_rgb); }
  let v = small[(panel * params.thumb_height + y) * params.thumb_width + px];
  if (v < -3.0e38) { return rgb(params.bg_rgb); }
  let lo = ranges[panel * 2u];
  let hi = ranges[panel * 2u + 1u];
  var t = 0.5;
  if (hi > lo) { t = clamp((v - lo) / (hi - lo), 0.0, 1.0); }
  return rgb(lut[min(u32(t * 255.0), 255u)]);
}
`;

export type ThumbnailLayout = {
  /** Panels side by side in the slot (1 for a single stack). */
  panels: number;
  /** One panel's width in the slot; the slot's own width when there is one panel. */
  panelSourceWidth: number;
  /** Slot column where the first drawn panel starts (to draw panel k alone: k * panelSourceWidth). */
  panelOffset?: number;
  /** Every panel reads the same slot columns (Show3D's shared panel source). */
  sharedSource: boolean;
  /** One panel's thumbnail size in device pixels. */
  thumbWidth: number;
  thumbHeight: number;
  /** Device pixels between panels, and their packed 0xBBGGRR color. */
  gap: number;
  bgRgb: number;
  logScale: boolean;
  lutName: string;
  lut: Uint8Array;
};

/** The canvas size, in device pixels, that `layout` draws into. */
export function thumbnailCanvasSize(layout: ThumbnailLayout): { width: number; height: number } {
  const panels = Math.max(1, Math.round(layout.panels));
  return {
    width: panels * Math.max(1, Math.round(layout.thumbWidth)) + (panels - 1) * Math.max(0, Math.round(layout.gap)),
    height: Math.max(1, Math.round(layout.thumbHeight)),
  };
}

/** Draws frame thumbnails from the engine's resident float32 slots into small WebGPU canvases. */
export class ThumbnailRenderer {
  private readonly engine: GPUColormapEngine;
  private readonly device: GPUDevice;
  private readonly format: GPUTextureFormat;
  private downsamplePipeline: GPUComputePipeline | null = null;
  private rangePipeline: GPUComputePipeline | null = null;
  private drawPipeline: GPURenderPipeline | null = null;
  private readonly params: GPUBuffer;
  private readonly ranges: GPUBuffer;
  private small: GPUBuffer | null = null;
  private smallCapacity = 0;
  private readonly configured = new WeakSet<GPUCanvasContext>();
  /** Set when the pipelines fail GPU validation; every later draw is refused instead of leaving black canvases. */
  failure: string | null = null;

  constructor(engine: GPUColormapEngine) {
    this.engine = engine;
    this.device = engine.getDevice();
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.params = this.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.ranges = this.device.createBuffer({ size: MAX_PANELS * 2 * 4, usage: GPUBufferUsage.STORAGE });
  }

  private ensurePipelines(): boolean {
    if (this.failure) return false;
    if (!this.downsamplePipeline) {
      this.device.pushErrorScope("validation");
      this.downsamplePipeline = computePipeline(this.device, THUMBNAIL_SHADER, "downsample");
      this.rangePipeline = computePipeline(this.device, THUMBNAIL_SHADER, "panel_range");
      this.drawPipeline = renderPipeline(this.device, THUMBNAIL_DRAW_SHADER, this.format);
      void this.device.popErrorScope().then(error => {
        if (!error) return;
        this.failure = error.message;
        console.error("[thumbnails] WebGPU pipeline failed validation; thumbnails are not drawn:", error.message);
      });
    }
    return Boolean(this.downsamplePipeline && this.rangePipeline && this.drawPipeline);
  }

  private ensureSmall(values: number): GPUBuffer {
    if (!this.small || this.smallCapacity < values) {
      this.small?.destroy();
      this.smallCapacity = Math.max(values, this.smallCapacity * 2, 4096);
      this.small = this.device.createBuffer({ size: this.smallCapacity * 4, usage: GPUBufferUsage.STORAGE });
    }
    return this.small;
  }

  /**
   * Draw resident slot `slotIdx` into `context` (sized with thumbnailCanvasSize). Returns false when the
   * slot is not a resident float32 frame, so the caller can try again once it is uploaded.
   */
  draw(context: GPUCanvasContext, slotIdx: number, layout: ThumbnailLayout): boolean {
    const slot = this.engine.slots[slotIdx];
    if (!slot || slot.dataKind !== "f32" || !this.ensurePipelines()) return false;
    const panels = Math.max(1, Math.min(MAX_PANELS, Math.round(layout.panels)));
    const panelSourceWidth = Math.max(1, Math.min(slot.width, Math.round(layout.panelSourceWidth)));
    // never more thumbnail pixels than source pixels
    const thumbWidth = Math.max(1, Math.min(panelSourceWidth, Math.round(layout.thumbWidth)));
    const thumbHeight = Math.max(1, Math.min(slot.height, Math.round(layout.thumbHeight)));
    const gap = Math.max(0, Math.round(layout.gap));
    if (!this.configured.has(context)) {
      context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
      this.configured.add(context);
    }
    const small = this.ensureSmall(panels * thumbWidth * thumbHeight);
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([
      slot.width, slot.height, panelSourceWidth, layout.sharedSource ? 0 : panelSourceWidth,
      panels, thumbWidth, thumbHeight, layout.logScale ? 1 : 0,
      gap, layout.bgRgb & 0xFFFFFF, SAMPLES_PER_SIDE, Math.max(0, Math.round(layout.panelOffset ?? 0)),
    ]));
    const lut = this.engine.namedLutBuffer(layout.lutName, layout.lut);
    // "auto" layouts hold only the bindings an entry point uses: downsample reads src, panel_range writes ranges.
    const encoder = this.device.createCommandEncoder();
    const down = encoder.beginComputePass();
    down.setPipeline(this.downsamplePipeline!);
    down.setBindGroup(0, this.device.createBindGroup({
      layout: this.downsamplePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: small } },
      ],
    }));
    down.dispatchWorkgroups(Math.ceil(thumbWidth / 8), Math.ceil(thumbHeight / 8), panels);
    down.end();
    const range = encoder.beginComputePass();
    range.setPipeline(this.rangePipeline!);
    range.setBindGroup(0, this.device.createBindGroup({
      layout: this.rangePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 2, resource: { buffer: small } },
        { binding: 3, resource: { buffer: this.ranges } },
      ],
    }));
    range.dispatchWorkgroups(panels);
    range.end();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.drawPipeline!);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.drawPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: small } },
        { binding: 2, resource: { buffer: this.ranges } },
        { binding: 3, resource: { buffer: lut } },
      ],
    }));
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  /** Free a thumbnail canvas's GPU textures; the next draw configures it again. */
  release(context: GPUCanvasContext): void {
    if (!this.configured.has(context)) return;
    context.unconfigure();
    this.configured.delete(context);
  }

  destroy(): void {
    this.params.destroy();
    this.ranges.destroy();
    this.small?.destroy();
    this.small = null;
  }
}
