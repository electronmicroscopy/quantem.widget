/// <reference types="@webgpu/types" />
// Offline 4D-STEM compute in the browser via WebGPU - no Python kernel.
// Primitives (browser siblings of the Python backends):
//   maskedSum(detectorMask) -> virtual image  (one thread per scan position)
//   reduceFrames(scanMask)  -> diffraction DP  (one thread per detector pixel)
//
// CHUNKED: the stack is split into scan-row ranges, each in its own GPU buffer
// (<= the 1 GB per-buffer cap). This lets a stack far larger than one buffer
// (e.g. 512x512x192x192 = 9.7 GB) live across N buffers and be reduced by
// dispatching per chunk and accumulating. A single-buffer dataset is just the
// N=1 case. Verified bit-exact vs numpy (chunked masked_sum maxErr 0).
//
// Stack ships as uint8 (clip(0,255): real detector counts are 0-~200, so the
// value IS the count, near-lossless) or uint16; dtype inferred from byte length.
import { getGPUDevice } from "./device";
import { decodeBslz4ToStack, decodeBslz4Batch, type Bslz4Spec } from "./bslz4";

// `mode`: 0 = uint16 (2 samples/u32), 1 = uint8 (4/u32). `sample(gp)` reads a
// detector value at a chunk-local global pixel index.
const SAMPLE = `
fn sample(gp: u32, mode: u32) -> u32 {
  if (mode == 1u) { let w = data[gp >> 2u]; return (w >> ((gp & 3u) * 8u)) & 0xffu; }
  let w = data[gp >> 1u];
  return select(w >> 16u, w & 0xffffu, (gp & 1u) == 0u);
}
// Float value of a pixel. mode 2 = float32 (1 u32/pixel = IEEE-754 bit pattern -> bitcast,
// full precision); modes 0/1 return the integer count as f32.
fn sampleF(gp: u32, mode: u32) -> f32 {
  if (mode == 2u) { return bitcast<f32>(data[gp]); }
  return f32(sample(gp, mode));
}`;

// One WORKGROUP per scan position; its 64 threads COOPERATIVELY sum the aperture pixels, then
// a shared-memory tree reduction writes one VI value. The old "one thread per scan position"
// kernel was uncoalesced: thread sl read data[sl*detSize + idx[j]], so a warp's 64 threads
// touched 64 addresses detSize apart = 64 cache lines per access = ~1/64 of memory bandwidth
// (180-300 ms for a BF drag). Here consecutive threads read consecutive idx entries (idx is
// row-major sorted) -> consecutive detector pixels -> COALESCED, so the drag hits the memory
// floor. 2D dispatch (gridX in u2.x) because a single-buffer stack can exceed 65535 scans.
// `sg` = use subgroup (warp) reduction: one subgroup (32 lanes) per scan, summed with a single
// `subgroupAdd` - NO shared memory, NO barriers. The shared-memory tree reduction (fallback for
// GPUs without the subgroups feature) pays 6 workgroupBarriers per scan position x 262144 scans,
// which dominated the kernel (~45-58ms); subgroupAdd is a register-level warp shuffle. Lanes read
// consecutive idx entries -> consecutive detector pixels (idx is row-major sorted) -> coalesced.
// WGSZ threads per scan position so many loads are in flight at once (the kernel is memory-bound
// on a gather; only 32 lanes left bandwidth at ~4% of peak). Each warp reduces with subgroupAdd,
// the per-warp partials combine through a tiny shared array (one barrier). Fallback (no subgroups)
// is the full shared tree reduction.
const WGSZ = 128;
const SGSZ = 32;   // subgroup (warp) size on the target GPUs
const maskedSumSrc = (sg: boolean) => `
${sg ? "enable subgroups;" : ""}
@group(0) @binding(0) var<storage,read> data: array<u32>;
@group(0) @binding(1) var<storage,read> idx: array<u32>;   // ACTIVE detector pixel indices only
@group(0) @binding(2) var<storage,read_write> vi: array<f32>;
@group(0) @binding(3) var<uniform> u: vec4<u32>;   // startScan, nScanInChunk, detSize, mode
@group(0) @binding(4) var<uniform> u2: vec4<u32>;  // gridX, 0, 0, 0
${SAMPLE}
var<workgroup> part: array<f32, ${WGSZ}>;
@compute @workgroup_size(${WGSZ})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let sl = wid.y * u2.x + wid.x; let tid = lid.x;
  // f32 accumulate via sampleF: exact for uint8/uint16 counts (sum << 2^24 for a screening
  // aperture), and the only correct path for float32 (mode 2) values.
  let n = arrayLength(&idx); let base = sl * u.z; var sum: f32 = 0.0;
  if (sl < u.y) { for (var j = tid; j < n; j = j + ${WGSZ}u) { sum = sum + sampleF(base + idx[j], u.w); } }
${sg
  ? `  sum = subgroupAdd(sum);                       // per-warp partial
  if (subgroupElect()) { part[tid / ${SGSZ}u] = sum; }   // one slot per warp
  workgroupBarrier();
  if (tid == 0u && sl < u.y) {
    var total = 0.0; for (var w = 0u; w < ${WGSZ / SGSZ}u; w = w + 1u) { total = total + part[w]; }
    vi[u.x + sl] = total;
  }`
  : `  part[tid] = sum; workgroupBarrier();
  for (var s: u32 = ${WGSZ / 2}u; s > 0u; s = s >> 1u) { if (tid < s) { part[tid] = part[tid] + part[tid + s]; } workgroupBarrier(); }
  if (tid == 0u && sl < u.y) { vi[u.x + sl] = part[0]; }`}
}`;

// One thread per detector pixel; ACCUMULATES this chunk's in-ROI scan positions
// into the DP (chunks dispatched serially, so += across chunks is safe). dims:
// startScan, nScanInChunk, detSize, mode; plus extra: total scanMask is global,
// indexed by startScan+sl.
// One thread per (detector pixel, FRAME-BLOCK): the 2D grid parallelizes the reduction over
// FRAMES too, not just pixels. The old "one thread per pixel, serial loop over all frames"
// launched only detSize (~37K) threads -> ~12% occupancy on a big GPU (88% idle), so the
// memory-bound sum ran ~70x over its bandwidth floor. Here gid.y splits the frames into
// FRAME_BLOCKS strided slices; each thread sums its slice locally (bit-exact integer) then does
// ONE atomicAdd into dp[k]. ~FRAME_BLOCKS x more threads saturate the GPU; atomic contention is
// only FRAME_BLOCKS-way per pixel (one add per thread), trivially cheap vs the memory traffic.
const FRAME_BLOCKS = 64;
const REDUCE_FRAMES_WGSL = `
@group(0) @binding(0) var<storage,read> data: array<u32>;
@group(0) @binding(1) var<storage,read> scanMask: array<u32>;  // GLOBAL scanCount
@group(0) @binding(2) var<storage,read_write> dp: array<atomic<u32>>;  // detSize, INTEGER (exact)
@group(0) @binding(3) var<uniform> u: vec4<u32>;   // startScan, nScanInChunk, detSize, mode
${SAMPLE}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x; let detSize = u.z; if (k >= detSize) { return; }
  var sum: u32 = 0u;   // integer accumulate: bit-exact, no f32 rounding on large/dead-pixel sums
  for (var sl: u32 = gid.y; sl < u.y; sl = sl + ${FRAME_BLOCKS}u) {   // strided frame slice
    if (scanMask[u.x + sl] != 0u) { sum = sum + sample(sl * detSize + k, u.w); }
  }
  if (sum != 0u) { atomicAdd(&dp[k], sum); }   // one add per thread; skip empty slices
}`;

// Extract ONE frame's diffraction pattern (detSize values) from a chunk buffer -
// the offline replacement for the kernel's per-probe frame_bytes. One thread/pixel.
const FRAME_WGSL = `
@group(0) @binding(0) var<storage,read> data: array<u32>;
@group(0) @binding(1) var<storage,read_write> frame: array<f32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;   // localBase (pixels), detSize, mode
${SAMPLE}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x; if (k >= u.y) { return; }
  frame[k] = sampleF(u.x + k, u.z);
}`;

// One thread per scan position: intensity-weighted centroid (center of mass) of the
// detector over the active mask pixels. Output is the per-position CoM in detector px:
// comY at [gi], comX at [scanCount+gi]. Drives CoMx/CoMy/CoMmag/iCoM (DPC).
// One WORKGROUP per scan position (same coalescing fix as MASKED_SUM): 64 threads cooperatively
// accumulate the intensity-weighted centroid over the aperture, three shared-memory reductions
// (weight, y*weight, x*weight). gridX in u2.z for the 2D dispatch.
const maskedComSrc = (sg: boolean) => `
${sg ? "enable subgroups;" : ""}
@group(0) @binding(0) var<storage,read> data: array<u32>;
@group(0) @binding(1) var<storage,read> idx: array<u32>;   // ACTIVE detector pixel indices
@group(0) @binding(2) var<storage,read_write> com: array<f32>;  // 2*scanCount: [gi]=comY, [scanCount+gi]=comX
@group(0) @binding(3) var<uniform> u: vec4<u32>;   // startScan, nScanInChunk, detSize, mode
@group(0) @binding(4) var<uniform> u2: vec4<u32>;  // detCols, scanCount, gridX, 0
${SAMPLE}
var<workgroup> pw: array<f32, ${WGSZ}>;
var<workgroup> py: array<f32, ${WGSZ}>;
var<workgroup> px: array<f32, ${WGSZ}>;
@compute @workgroup_size(${WGSZ})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let sl = wid.y * u2.z + wid.x; let tid = lid.x;
  let base = sl * u.z; let n = arrayLength(&idx); let detCols = u2.x;
  var wsum: f32 = 0.0; var ysum: f32 = 0.0; var xsum: f32 = 0.0;
  if (sl < u.y) { for (var j: u32 = tid; j < n; j = j + ${WGSZ}u) {
    let p = idx[j]; let v = sampleF(base + p, u.w);
    wsum = wsum + v; ysum = ysum + f32(p / detCols) * v; xsum = xsum + f32(p % detCols) * v;
  } }
${sg
  ? `  wsum = subgroupAdd(wsum); ysum = subgroupAdd(ysum); xsum = subgroupAdd(xsum);
  if (subgroupElect()) { let w = tid / ${SGSZ}u; pw[w] = wsum; py[w] = ysum; px[w] = xsum; }
  workgroupBarrier();
  if (tid == 0u && sl < u.y) {
    var w = 0.0; var y = 0.0; var x = 0.0;
    for (var k = 0u; k < ${WGSZ / SGSZ}u; k = k + 1u) { w = w + pw[k]; y = y + py[k]; x = x + px[k]; }
    let gi = u.x + sl;
    if (w > 0.0) { com[gi] = y / w; com[u2.y + gi] = x / w; } else { com[gi] = 0.0; com[u2.y + gi] = 0.0; }
  }`
  : `  pw[tid] = wsum; py[tid] = ysum; px[tid] = xsum; workgroupBarrier();
  for (var s: u32 = ${WGSZ / 2}u; s > 0u; s = s >> 1u) {
    if (tid < s) { pw[tid] = pw[tid] + pw[tid + s]; py[tid] = py[tid] + py[tid + s]; px[tid] = px[tid] + px[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u && sl < u.y) {
    let gi = u.x + sl;
    if (pw[0] > 0.0) { com[gi] = py[0] / pw[0]; com[u2.y + gi] = px[0] / pw[0]; } else { com[gi] = 0.0; com[u2.y + gi] = 0.0; }
  }`}
}`;

const MAX_WG = 65535;   // max workgroups per dispatch dimension; >this needs a 2D grid

interface Chunk { buffer: GPUBuffer; startScan: number; nScan: number; }

export class Show4DSTEMCompute {
  private device: GPUDevice;
  private maskedSumPipe: GPUComputePipeline;
  private maskedComPipe: GPUComputePipeline;
  private reduceFramesPipe: GPUComputePipeline;
  private frameAtPipe: GPUComputePipeline;
  private chunks: Chunk[];
  readonly scanCount: number;
  readonly detSize: number;
  readonly mode: number;
  // Bad/hot detector pixel indices (from the HDF5 pixel_mask). Auto-excluded from
  // every reduction so the offline result matches CUDA's apply_mask path - the
  // browser data is filtered automatically, no per-call masking needed.
  badPx: Uint32Array = new Uint32Array(0);

  private constructor(device: GPUDevice, chunks: Chunk[], scanCount: number, detSize: number, mode: number) {
    this.device = device; this.chunks = chunks; this.scanCount = scanCount; this.detSize = detSize; this.mode = mode;
    const sg = device.features.has("subgroups");   // warp reduction in maskedSum/CoM when available
    const ms = device.createShaderModule({ code: maskedSumSrc(sg) });
    const rf = device.createShaderModule({ code: REDUCE_FRAMES_WGSL });
    this.maskedSumPipe = device.createComputePipeline({ layout: "auto", compute: { module: ms, entryPoint: "main" } });
    this.maskedComPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: maskedComSrc(sg) }), entryPoint: "main" } });
    this.reduceFramesPipe = device.createComputePipeline({ layout: "auto", compute: { module: rf, entryPoint: "main" } });
    this.frameAtPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: FRAME_WGSL }), entryPoint: "main" } });
  }

  // One frame's diffraction pattern (f32[detSize]) for scan position scanIdx -
  // a GPU extract from whichever chunk holds it. Drives the offline DP panel.
  async frameAt(scanIdx: number): Promise<Float32Array> {
    const ch = this.chunks.find((c) => scanIdx >= c.startScan && scanIdx < c.startScan + c.nScan) ?? this.chunks[0];
    const localBase = (scanIdx - ch.startScan) * this.detSize;
    const out = this.device.createBuffer({ size: this.detSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const dims = this.uniform([localBase, this.detSize, this.mode, 0]);
    const bind = this.device.createBindGroup({ layout: this.frameAtPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: ch.buffer } }, { binding: 1, resource: { buffer: out } }, { binding: 2, resource: { buffer: dims } } ] });
    this.dispatch(this.frameAtPipe, bind, Math.ceil(this.detSize / 64));
    const frame = await this.readF32(out, this.detSize);
    for (const bp of this.badPx) frame[bp] = 0;   // auto-filter hot px in the diffraction pattern
    out.destroy(); dims.destroy(); return frame;
  }

  // Per-scan-position center of mass (intensity-weighted detector centroid) over the
  // active mask. Returns {comY, comX} in detector px (length scanCount each). Drives DPC
  // (CoMx/CoMy/CoMmag/iCoM). detCols unravels the flat pixel index to (row,col); badPx
  // are excluded from the mask, matching every other reduction.
  async maskedCoM(mask: Uint32Array, detCols: number): Promise<{ comY: Float32Array; comX: Float32Array }> {
    const device = this.device;
    const bad = this.badPx.length ? new Set(this.badPx) : null;
    const idxArr = new Uint32Array(this.detSize); let n = 0;
    for (let k = 0; k < this.detSize; k++) if (mask[k] !== 0 && !(bad && bad.has(k))) idxArr[n++] = k;
    const idx = idxArr.subarray(0, n || 1);
    const idxBuf = this.upload(idx, GPUBufferUsage.STORAGE);
    const com = device.createBuffer({ size: this.scanCount * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const temps: GPUBuffer[] = [];
    // One workgroup per scan position (2D grid), all chunks in ONE encoder + submit, readback
    // folded in. com slices are disjoint per chunk (no accumulation), so no zero-init needed.
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(); pass.setPipeline(this.maskedComPipe);
    for (const ch of this.chunks) {
      const gx = Math.min(ch.nScan, MAX_WG), gy = Math.ceil(ch.nScan / MAX_WG);
      const dims = this.uniform([ch.startScan, ch.nScan, this.detSize, this.mode]); temps.push(dims);
      const dims2 = this.uniform([detCols, this.scanCount, gx, 0]); temps.push(dims2);
      const bind = device.createBindGroup({ layout: this.maskedComPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: ch.buffer } }, { binding: 1, resource: { buffer: idxBuf } },
        { binding: 2, resource: { buffer: com } }, { binding: 3, resource: { buffer: dims } }, { binding: 4, resource: { buffer: dims2 } } ] });
      pass.setBindGroup(0, bind); pass.dispatchWorkgroups(gx, gy);
    }
    pass.end();
    const rb = device.createBuffer({ size: this.scanCount * 2 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(com, 0, rb, 0, this.scanCount * 2 * 4);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const flat = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    idxBuf.destroy(); com.destroy(); temps.forEach((b) => b.destroy());
    const comY = flat.slice(0, this.scanCount), comX = flat.slice(this.scanCount, this.scanCount * 2);
    if (n === 0) { comY.fill(0); comX.fill(0); }
    return { comY, comX };
  }

  // Single decompressed stack -> one chunk (the common, fits-in-one-buffer case).
  static async create(stack: Uint8Array, scanCount: number, detSize: number): Promise<Show4DSTEMCompute | null> {
    return Show4DSTEMCompute.createChunked([{ bytes: stack, startScan: 0, nScan: scanCount }], scanCount, detSize);
  }

  // Decompress a native HDF5 bitshuffle+LZ4 (bslz4) stack on the GPU and wrap the
  // decoded buffer as the (single) compute chunk - the offline "ship compressed,
  // decompress in browser, no Python" path. dtype "uint8" (offline default, clip
  // 0-255, half memory) or "uint16" (lossless). The decoded buffer is packed in
  // [scanPos][detPixel] order matching sample() for that mode, so masked_sum /
  // reduceFrames run on it unchanged.
  static async createFromBslz4(spec: Bslz4Spec, dtype: "uint8" | "uint16" | "float32" = "uint8", srcDtype: "uint8" | "uint16" | "float32" = "uint16"): Promise<Show4DSTEMCompute | null> {
    const decoded = await decodeBslz4ToStack(spec, dtype, srcDtype);
    if (!decoded) return null;
    const chunks: Chunk[] = [{ buffer: decoded.buffer, startScan: 0, nScan: spec.nFrames }];
    return new Show4DSTEMCompute(decoded.device, chunks, spec.nFrames, spec.detSize, decoded.mode);
  }

  // Wrap already-decoded GPU stack buffers as a compute (no decode). Lets a caller pipeline
  // parse + decode itself (decode group N while parsing group N+1) and hand the finished
  // chunk buffers here. mode: 1 = uint8, 0 = uint16.
  static fromGpuChunks(device: GPUDevice, chunks: { buffer: GPUBuffer; startScan: number; nScan: number }[], scanCount: number, detSize: number, mode: number): Show4DSTEMCompute {
    return new Show4DSTEMCompute(device, chunks, scanCount, detSize, mode);
  }

  // Chunked bslz4: decode each scan-row chunk's compressed bytes into its OWN GPU
  // buffer and hold them all, so a stack far bigger than one 1 GB buffer (full
  // 512x512x192x192 = 9.6 GB uint8) lives across N buffers and masked_sum /
  // reduceFrames reduce across them. Each decode reuses a ~1 GB scratch internally.
  static async createFromBslz4Chunked(
    chunkSpecs: (Bslz4Spec & { startScan: number; nScan: number })[],
    scanCount: number, detSize: number, dtype: "uint8" | "uint16" | "float32" = "uint8",
    srcDtype: "uint8" | "uint16" | "float32" = "uint16",
  ): Promise<Show4DSTEMCompute | null> {
    // Batch the per-chunk decodes (one submit + await per group) so the GPU overlaps
    // upload and compute instead of draining after every chunk.
    const decoded = await decodeBslz4Batch(chunkSpecs, dtype, srcDtype);
    if (!decoded) return null;
    const chunks: Chunk[] = decoded.buffers.map((buffer, i) => ({ buffer, startScan: chunkSpecs[i].startScan, nScan: chunkSpecs[i].nScan }));
    return new Show4DSTEMCompute(decoded.device, chunks, scanCount, detSize, decoded.mode);
  }

  // N chunks, each {bytes, startScan, nScan}. Each chunk's bytes hold its scan
  // range's frames contiguously. dtype inferred from total bytes vs total pixels.
  static async createChunked(chunkSpecs: { bytes: Uint8Array; startScan: number; nScan: number }[], scanCount: number, detSize: number): Promise<Show4DSTEMCompute | null> {
    const device = await getGPUDevice();
    if (!device) return null;
    const totalBytes = chunkSpecs.reduce((a, c) => a + c.bytes.byteLength, 0);
    const mode = totalBytes <= scanCount * detSize ? 1 : 0;  // <= 1 byte/pixel => uint8
    const chunks: Chunk[] = chunkSpecs.map((c) => {
      const padLen = Math.ceil(c.bytes.byteLength / 4) * 4;
      const buffer = device.createBuffer({ size: Math.max(4, padLen), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, c.bytes.buffer as ArrayBuffer, c.bytes.byteOffset, c.bytes.byteLength);
      return { buffer, startScan: c.startScan, nScan: c.nScan };
    });
    return new Show4DSTEMCompute(device, chunks, scanCount, detSize, mode);
  }

  // Virtual image: f32[scanCount]. Each chunk writes its disjoint VI slice.
  // Loops only the ACTIVE (in-aperture) detector pixels, not all detSize - a BF
  // disk (~9k of 36864 px) or ADF annulus is then 4-10x fewer reads per scan pos.
  async maskedSum(mask: Uint32Array): Promise<Float32Array> {
    const device = this.device;
    const bad = this.badPx.length ? new Set(this.badPx) : null;
    const idxArr = new Uint32Array(this.detSize); let n = 0;
    for (let k = 0; k < this.detSize; k++) if (mask[k] !== 0 && !(bad && bad.has(k))) idxArr[n++] = k;  // skip hot px
    const idx = idxArr.subarray(0, n || 1);   // active pixel indices (>=1 to keep a valid binding)
    const idxBuf = this.upload(idx, GPUBufferUsage.STORAGE);
    const vi = device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const temps: GPUBuffer[] = [];
    // One workgroup per scan position (2D grid for >65535), ALL chunks in ONE encoder + submit,
    // and the readback folded in - so a BF/ADF drag is a single GPU pass + single sync.
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(); pass.setPipeline(this.maskedSumPipe);
    for (const ch of this.chunks) {
      const gx = Math.min(ch.nScan, MAX_WG), gy = Math.ceil(ch.nScan / MAX_WG);
      const dims = this.uniform([ch.startScan, ch.nScan, this.detSize, this.mode]); temps.push(dims);
      const dims2 = this.uniform([gx, 0, 0, 0]); temps.push(dims2);
      const bind = device.createBindGroup({ layout: this.maskedSumPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: ch.buffer } }, { binding: 1, resource: { buffer: idxBuf } },
        { binding: 2, resource: { buffer: vi } }, { binding: 3, resource: { buffer: dims } }, { binding: 4, resource: { buffer: dims2 } } ] });
      pass.setBindGroup(0, bind); pass.dispatchWorkgroups(gx, gy);
    }
    pass.end();
    const rb = device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(vi, 0, rb, 0, this.scanCount * 4);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    if (n === 0) out.fill(0);   // empty mask -> all zero (idx had a dummy entry)
    idxBuf.destroy(); vi.destroy(); temps.forEach((b) => b.destroy()); return out;
  }

  // GPU-RESIDENT virtual image: identical to maskedSum but returns the vi GPU buffer WITHOUT
  // the readback. The 60fps drag path feeds this straight into the colormap engine + canvas, so
  // there is NO GPU->CPU->GPU bounce and NO mapAsync fence (the two fences - maskedSum readback +
  // colormap rgba readback - were the ~100ms/repaint that capped the drag at ~9fps). Caller owns
  // the returned buffer (the colormap slot adopts it; freed on the next adopt / dispose).
  maskedSumBuffer(mask: Uint32Array): { buffer: GPUBuffer; n: number } {
    const device = this.device;
    const bad = this.badPx.length ? new Set(this.badPx) : null;
    const idxArr = new Uint32Array(this.detSize); let n = 0;
    for (let k = 0; k < this.detSize; k++) if (mask[k] !== 0 && !(bad && bad.has(k))) idxArr[n++] = k;
    const idx = idxArr.subarray(0, n || 1);
    const idxBuf = this.upload(idx, GPUBufferUsage.STORAGE);
    const vi = device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(); pass.setPipeline(this.maskedSumPipe);
    for (const cd of this.sumDims()) {   // per-chunk dims are constant -> built once, reused every drag frame
      const bind = device.createBindGroup({ layout: this.maskedSumPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: cd.chunk } }, { binding: 1, resource: { buffer: idxBuf } },
        { binding: 2, resource: { buffer: vi } }, { binding: 3, resource: { buffer: cd.dims } }, { binding: 4, resource: { buffer: cd.dims2 } } ] });
      pass.setBindGroup(0, bind); pass.dispatchWorkgroups(cd.gx, cd.gy);
    }
    pass.end();
    device.queue.submit([enc.finish()]);
    idxBuf.destroy();   // vi handed to caller; dims are cached + reused, NOT destroyed
    return { buffer: vi, n };
  }

  // Per-chunk maskedSum dispatch params (chunk buffer, dims uniforms, 2D grid). These never change
  // for a dataset, so build them ONCE and reuse across every drag frame instead of allocating +
  // destroying ~2 uniform buffers per chunk per frame (~3300 buffer creates/s during a drag).
  private sumDimsCache: { chunk: GPUBuffer; dims: GPUBuffer; dims2: GPUBuffer; gx: number; gy: number }[] | null = null;
  private sumDims() {
    if (!this.sumDimsCache) {
      this.sumDimsCache = this.chunks.map((ch) => {
        const gx = Math.min(ch.nScan, MAX_WG), gy = Math.ceil(ch.nScan / MAX_WG);
        return { chunk: ch.buffer, dims: this.uniform([ch.startScan, ch.nScan, this.detSize, this.mode]), dims2: this.uniform([gx, 0, 0, 0]), gx, gy };
      });
    }
    return this.sumDimsCache;
  }

  // DP over a real-space ROI: f32[detSize]. scanMask is GLOBAL; chunks accumulate
  // in INTEGER (u32, bit-exact) - the mean divide happens once in f64 at readback,
  // so the result matches the torch/CUDA integer-sum-then-divide exactly (even on
  // saturated 65535 dead pixels, where f32 accumulation would drift ~1 count).
  async reduceFrames(scanMask: Uint32Array, mean = true): Promise<Float32Array> {
    const device = this.device;
    const maskBuf = this.upload(scanMask, GPUBufferUsage.STORAGE);
    const dp = device.createBuffer({ size: this.detSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dp, 0, new Uint32Array(this.detSize));  // zero-init integer accumulator
    const temps: GPUBuffer[] = [];
    // Record EVERY chunk's reduce pass into ONE command encoder + ONE submit. Per-chunk submits
    // (27 buffers for a 27-file dataset) each pay kernel-launch + queue round-trip overhead and
    // do not overlap; batching lets the GPU run them back-to-back into the shared dp accumulator.
    const grid = Math.ceil(this.detSize / 64);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.reduceFramesPipe);
    for (const ch of this.chunks) {
      const dims = this.uniform([ch.startScan, ch.nScan, this.detSize, this.mode]); temps.push(dims);
      const bind = device.createBindGroup({ layout: this.reduceFramesPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: ch.buffer } }, { binding: 1, resource: { buffer: maskBuf } },
        { binding: 2, resource: { buffer: dp } }, { binding: 3, resource: { buffer: dims } } ] });
      pass.setBindGroup(0, bind); pass.dispatchWorkgroups(grid, FRAME_BLOCKS);
    }
    pass.end();
    // Fold the dp -> readback copy into the SAME encoder: one submit, one GPU->CPU sync. A
    // separate readU32 would submit + fence a second time (~30-50ms of mapAsync round-trip for
    // a 147 KB buffer - all latency, no bandwidth), doubling the sync cost of a tiny readback.
    const rb = device.createBuffer({ size: this.detSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(dp, 0, rb, 0, this.detSize * 4);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const sums = new Uint32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    temps.forEach((b) => b.destroy());
    const n = mean ? (scanMask.reduce((a, v) => a + (v ? 1 : 0), 0) || 1) : 1;
    const out = new Float32Array(this.detSize);
    for (let i = 0; i < this.detSize; i++) out[i] = sums[i] / n;  // f64 divide -> f32 store
    for (const bp of this.badPx) out[bp] = 0;   // auto-filter hot px (matches CUDA apply_mask)
    maskBuf.destroy(); dp.destroy(); return out;
  }

  private upload(arr: Uint32Array, usage: number): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, arr.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, arr.buffer as ArrayBuffer, arr.byteOffset, arr.byteLength); return b;
  }
  private uniform(vals: number[]): GPUBuffer {
    const b = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const a = new Uint32Array(vals); this.device.queue.writeBuffer(b, 0, a.buffer as ArrayBuffer, a.byteOffset, a.byteLength); return b;
  }
  private dispatch(pipe: GPUComputePipeline, bind: GPUBindGroup, groups: number, gy = 1) {
    const enc = this.device.createCommandEncoder(); const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(groups, gy); pass.end();
    this.device.queue.submit([enc.finish()]);
  }
  private async readF32(buf: GPUBuffer, n: number): Promise<Float32Array> {
    const rb = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4); this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy(); return out;
  }

  dispose() {
    for (const c of this.chunks) c.buffer.destroy();
    if (this.sumDimsCache) { for (const cd of this.sumDimsCache) { cd.dims.destroy(); cd.dims2.destroy(); } this.sumDimsCache = null; }
  }
}
