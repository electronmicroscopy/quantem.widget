/// <reference types="@webgpu/types" />
// Lazy compute backend for Show4DSTEM. Drop-in for Show4DSTEMCompute, but NOTHING bulk-loads:
// - virtual image (BF/ADF/circular detector) = derive from a ~100 MB RADIAL PROFILE resident in
//   VRAM (sum radial bins for the detector's r-range). float32, ~1 ms.
// - CBED frame = lazy range-fetch the one frame's ~124 KB bslz4 chunk from disk + GPU-decode it.
// - CoM/DPC = a small precomputed per-scan CoM field.
// The full 38 GB never enters VRAM; the disk stays the store. Same method signatures as
// Show4DSTEMCompute so the component's recomputeVI / recomputeFrame / recomputeCoM work unchanged.
import { decodeBslz4ToStack } from "./bslz4";
import { getGPUDevice } from "./device";

function be32(b: Uint8Array, o: number): number { return ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0; }

// One thread per scan position sums the detector's radial-bin range from the profile.
const DERIVE_WGSL = `
@group(0) @binding(0) var<storage,read> prof: array<f32>;        // N * NB radial profile
@group(0) @binding(1) var<storage,read_write> vi: array<f32>;    // N virtual image
@group(0) @binding(2) var<uniform> u: vec4<u32>;                 // N, NB, r0, r1 (bins [r0,r1))
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>){
  let s = g.x; if (s >= u.x) { return; }
  var sum = 0.0; for (var b = u.z; b < u.w; b = b + 1u) { sum = sum + prof[s * u.y + b]; }
  vi[s] = sum;
}`;

// Colormap render: read the derived VI straight from its GPU buffer and paint to the canvas.
// No GPU->CPU readback, no model.set, no React - this is the 60 FPS fast-paint path.
const CMAP_WGSL = `
@group(0) @binding(0) var<storage,read> vi: array<f32>;
@group(0) @binding(1) var<uniform> u: vec4<f32>;   // w, h, vmin, vmax
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>,3>(vec2<f32>(-1.,-1.), vec2<f32>(3.,-1.), vec2<f32>(-1.,3.));
  return vec4<f32>(p[i], 0., 1.);
}
@fragment fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let x = u32(fc.x); let y = u32(fc.y); let w = u32(u.x);
  let v = clamp((vi[y * w + x] - u.z) / (u.w - u.z), 0., 1.);
  return vec4<f32>(v, v * v, 1. - v, 1.);   // inferno-ish
}`;

export interface LazyMeta { SR: number; SC: number; D: number; NB: number; nFrames: number; files: string[]; }

export class LazyShow4DSTEM {
  readonly mode = 2;                 // float32, so sampleF / display treat frames as f32
  badPx = new Uint32Array(0);
  readonly scanCount: number;
  readonly detSize: number;
  private cache = new Map<number, Float32Array>();   // LRU of decoded CBED frames
  private rpipe: GPURenderPipeline | null = null;    // fast-paint colormap render (lazy-init)
  private rcfg: GPUBuffer | null = null;
  private rctx: GPUCanvasContext | null = null;
  private constructor(
    private device: GPUDevice, private meta: LazyMeta, private base: string,
    private idx: Uint32Array, private binOfPixel: Int32Array,
    private profBuf: GPUBuffer, private viBuf: GPUBuffer, private cfgBuf: GPUBuffer,
    private pipe: GPUComputePipeline, private com: Float32Array | null,
  ) { this.scanCount = meta.SR * meta.SC; this.detSize = meta.D * meta.D; }

  // base = the URL prefix the data files + sidecars sit under (e.g. ".../output/").
  static async create(base: string): Promise<LazyShow4DSTEM | null> {
    const device = await getGPUDevice(); if (!device) return null;
    const meta = await (await fetch(base + "meta.json")).json() as LazyMeta;
    const idx = new Uint32Array(await (await fetch(base + "index.bin")).arrayBuffer());
    const prof = new Float32Array(await (await fetch(base + "profile.bin")).arrayBuffer());
    let com: Float32Array | null = null;
    try { com = new Float32Array(await (await fetch(base + "com.bin")).arrayBuffer()); } catch { /* optional */ }
    // radial bin of each detector pixel (center = det center), to map a detector mask -> bin range.
    const D = meta.D, cy = D / 2, cx = D / 2, binOfPixel = new Int32Array(D * D);
    for (let y = 0; y < D; y++) for (let x = 0; x < D; x++) binOfPixel[y * D + x] = Math.min(meta.NB - 1, Math.floor(Math.hypot(y - cy, x - cx)));
    const profBuf = device.createBuffer({ size: prof.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Float32Array(profBuf.getMappedRange()).set(prof); profBuf.unmap();
    const viBuf = device.createBuffer({ size: meta.SR * meta.SC * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const cfgBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: DERIVE_WGSL }), entryPoint: "main" } });
    return new LazyShow4DSTEM(device, meta, base, idx, binOfPixel, profBuf, viBuf, cfgBuf, pipe, com);
  }

  // Virtual image for a detector mask. The mask is a 0/1 weight per DETECTOR PIXEL (length =
  // detSize), NOT a list of indices. For a circular/annular detector the active pixels span a
  // radial range [r0,r1]; we sum those profile bins (exact for the Show4DSTEM "Circle"/"Annular"
  // detectors). For a square/arbitrary mask the radial range over-covers (the profile can't
  // represent non-circular regions) - that case needs the full-data path.
  async maskedSum(mask: Uint32Array): Promise<Float32Array> {
    let r0 = this.meta.NB, r1 = -1;
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] === 0) continue;
      const b = this.binOfPixel[i];
      if (b < r0) r0 = b; if (b > r1) r1 = b;
    }
    if (r1 < r0) { r0 = 0; r1 = -1; }
    return this.deriveVI(r0, r1 + 1);
  }

  private async deriveVI(r0: number, r1: number): Promise<Float32Array> {
    const N = this.scanCount;
    this.device.queue.writeBuffer(this.cfgBuf, 0, new Uint32Array([N, this.meta.NB, Math.max(0, r0), Math.max(0, r1)]));
    const bg = this.device.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.profBuf } }, { binding: 1, resource: { buffer: this.viBuf } }, { binding: 2, resource: { buffer: this.cfgBuf } } ] });
    const enc = this.device.createCommandEncoder(); const p = enc.beginComputePass();
    p.setPipeline(this.pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(Math.ceil(N / 64)); p.end();
    const rb = this.device.createBuffer({ size: N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(this.viBuf, 0, rb, 0, N * 4); this.device.queue.submit([enc.finish()]); await this.device.queue.onSubmittedWorkDone();
    await rb.mapAsync(GPUMapMode.READ); const v = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    return v;
  }

  // CBED of one scan position: lazy range-fetch its bslz4 chunk + decode one float32 frame.
  async frameAt(scanIdx: number): Promise<Float32Array> {
    const cached = this.cache.get(scanIdx); if (cached) return cached;
    const o = scanIdx * 3, fi = this.idx[o], off = this.idx[o + 1], len = this.idx[o + 2];
    const ch = new Uint8Array(await (await fetch(this.base + this.meta.files[fi], { headers: { Range: `bytes=${off}-${off + len - 1}` } })).arrayBuffer());
    const bb = be32(ch, 8), be = bb / 4, nBlk = Math.ceil(this.detSize / be); const m: number[] = []; let pos = 12;
    for (let b = 0; b < nBlk; b++) { const c = be32(ch, pos); m.push(pos + 4, c); pos += 4 + c; }
    const res = await decodeBslz4ToStack({ compressed: ch, blockMeta: new Uint32Array(m), nFrames: 1, nBlocksPerFrame: nBlk, blockElems: be, detSize: this.detSize }, "float32", "float32");
    if (!res) return new Float32Array(this.detSize);
    const rb = res.device.createBuffer({ size: this.detSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = res.device.createCommandEncoder(); enc.copyBufferToBuffer(res.buffer, 0, rb, 0, this.detSize * 4);
    res.device.queue.submit([enc.finish()]); await res.device.queue.onSubmittedWorkDone();
    await rb.mapAsync(GPUMapMode.READ); const v = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy(); res.buffer.destroy();
    this.cache.set(scanIdx, v); if (this.cache.size > 200) this.cache.delete(this.cache.keys().next().value as number);
    return v;
  }

  // CoM/DPC from the precomputed CoM field (per scan position). Mask ignored (CoM is detector-global).
  async maskedCoM(_mask: Uint32Array, _detCols: number): Promise<{ comY: Float32Array; comX: Float32Array }> {
    const N = this.scanCount;
    if (!this.com) return { comY: new Float32Array(N), comX: new Float32Array(N) };
    return { comY: this.com.subarray(0, N), comX: this.com.subarray(N, 2 * N) };
  }

  // ROI->DP: mean/sum diffraction over the real-space ROI's scan positions. scanMask is a 0/1
  // weight per scan position (length = scanCount). Lazy: fetch+decode each ROI frame (cached) and
  // accumulate. Capped so a huge ROI stays interactive - past the cap it's a representative sample
  // (the UI can offer a full-load for an exact large-ROI mean).
  async reduceFrames(scanMask: Uint32Array, mean: boolean): Promise<Float32Array> {
    const acc = new Float32Array(this.detSize); let count = 0; const CAP = 1500;
    for (let i = 0; i < scanMask.length && count < CAP; i++) {
      if (scanMask[i] === 0) continue;
      const f = await this.frameAt(i);
      for (let p = 0; p < this.detSize; p++) acc[p] += f[p];
      count++;
    }
    if (mean && count > 0) for (let p = 0; p < this.detSize; p++) acc[p] /= count;
    return acc;
  }

  // Fast-paint: derive VI for [r0,r1) and colormap-render straight to the canvas via WebGPU.
  // One submit: compute pass (sum bins -> viBuf) + render pass (colormap viBuf -> canvas). No readback,
  // no model.set, no React. This is the path that turns the 14 FPS React render into vsync 60.
  renderVI(canvas: HTMLCanvasElement, r0: number, r1: number, vmin: number, vmax: number): void {
    if (!this.rpipe) {
      const mod = this.device.createShaderModule({ code: CMAP_WGSL });
      const fmt = navigator.gpu.getPreferredCanvasFormat();
      this.rctx = canvas.getContext("webgpu"); if (!this.rctx) return;
      this.rctx.configure({ device: this.device, format: fmt, alphaMode: "opaque" });
      this.rpipe = this.device.createRenderPipeline({ layout: "auto",
        vertex: { module: mod, entryPoint: "vs" },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: fmt }] },
        primitive: { topology: "triangle-list" } });
      this.rcfg = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    const enc = this.device.createCommandEncoder();
    this.device.queue.writeBuffer(this.cfgBuf, 0, new Uint32Array([this.scanCount, this.meta.NB, Math.max(0, r0), Math.max(0, r1)]));
    const cbg = this.device.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.profBuf } }, { binding: 1, resource: { buffer: this.viBuf } }, { binding: 2, resource: { buffer: this.cfgBuf } } ] });
    const cp = enc.beginComputePass(); cp.setPipeline(this.pipe); cp.setBindGroup(0, cbg); cp.dispatchWorkgroups(Math.ceil(this.scanCount / 64)); cp.end();
    this.device.queue.writeBuffer(this.rcfg!, 0, new Float32Array([this.meta.SC, this.meta.SR, vmin, vmax]));
    const rbg = this.device.createBindGroup({ layout: this.rpipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.viBuf } }, { binding: 1, resource: { buffer: this.rcfg! } } ] });
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.rctx!.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.rpipe); pass.setBindGroup(0, rbg); pass.draw(3); pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  dispose(): void { this.profBuf.destroy(); this.viBuf.destroy(); this.cfgBuf.destroy(); this.rcfg?.destroy(); this.cache.clear(); }
}
