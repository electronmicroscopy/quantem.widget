// Volume-resident orthogonal and oblique slices for Show3DSlices: the volume
// is uploaded once and each scrub runs one slice + colormap compute pass.
import type { GPUColormapEngine } from "./colormapEngine";
import { DISPLAY_NORMALIZE_WGSL } from "./displayShaders";
import { computePipeline } from "./pipelines";

// Smallest zoom of a volume slice view, Show3DSlices' MIN_ZOOM. The area
// average reads every source pixel an output pixel covers, (fullW / canvasW /
// zoom) per axis, so the clamp is what bounds that work per pixel.
export const VOLUME_SLICE_MIN_ZOOM = 0.5;

// Volume-resident orthogonal slice + colormap in ONE compute pass. The whole 3D
// volume lives on the GPU (uploaded once), in a storage buffer or, for 4k and
// row-padded stacks, a 2D texture array; per scrub only a tiny uniform (axis +
// slice index + vmin/vmax) changes, so there is NO per-frame CPU slice
// extraction and NO per-frame volume re-upload. axis: 0=top (slice fixed),
// 1=row fixed, 2=col fixed, 3=oblique cut along the alignment segment. Order
// matches the CPU path: log THEN flip.
function volumeSliceShader(source: "buffer" | "texture"): string {
  const volumeBinding = source === "buffer"
    ? "var<storage, read> vol: array<f32>"
    : "var volTex: texture_2d_array<f32>";
  const volumeLoad = source === "buffer"
    ? "vol[slice * p.numRows * p.numCols + row * p.numCols + col]"
    : "textureLoad(volTex, vec2<i32>(i32(col), i32(row)), i32(slice), 0).r";
  return DISPLAY_NORMALIZE_WGSL + /* wgsl */ `
struct VParams {
  numCols: u32, numRows: u32, numSlices: u32, axis: u32,
  index: u32, outW: u32, outH: u32, logScale: u32,
  flip: u32, viewMode: u32, canvasW: u32, canvasH: u32,
  vmin: f32, vmax: f32, zoom: f32, panX: f32,
  panY: f32, rowShift: f32, colShift: f32, segStartCol: f32,
  segStartRow: f32, segStopCol: f32, segStopRow: f32, _p0: f32,
};
@group(0) @binding(0) var<uniform> p: VParams;
@group(0) @binding(1) ${volumeBinding};
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

fn loadVolume(slice: u32, col: u32, row: u32) -> f32 {
  return ${volumeLoad};
}

fn sampleAligned(sliceRaw: f32, colRaw: f32, rowRaw: f32) -> f32 {
  let slice = u32(clamp(round(sliceRaw), 0.0, f32(p.numSlices - 1u)));
  let center = (f32(max(p.numSlices, 1u)) - 1.0) * 0.5;
  let col = clamp(colRaw - (f32(slice) - center) * p.colShift, 0.0, f32(p.numCols - 1u));
  let row = clamp(rowRaw - (f32(slice) - center) * p.rowShift, 0.0, f32(p.numRows - 1u));
  let col0 = u32(floor(col));
  let row0 = u32(floor(row));
  let col1 = min(p.numCols - 1u, col0 + 1u);
  let row1 = min(p.numRows - 1u, row0 + 1u);
  let colFrac = col - f32(col0);
  let rowFrac = row - f32(row0);
  let v00 = loadVolume(slice, col0, row0);
  let v10 = loadVolume(slice, col1, row0);
  let v01 = loadVolume(slice, col0, row1);
  let v11 = loadVolume(slice, col1, row1);
  return mix(mix(v00, v10, colFrac), mix(v01, v11, colFrac), rowFrac);
}

fn sampleSlice(p_axis: u32, p_index: u32, planeCol: u32, planeRow: u32) -> f32 {
  var col: f32; var row: f32; var slice: f32;
  if (p_axis == 0u) { col = f32(planeCol); row = f32(planeRow); slice = f32(p_index); }          // top
  else if (p_axis == 1u) { col = f32(planeCol); row = f32(p_index); slice = f32(planeRow); }     // row fixed
  else if (p_axis == 2u) { col = f32(p_index); row = f32(planeCol); slice = f32(planeRow); }     // col fixed
  else {
    let denom = max(f32(p.canvasW - 1u), 1.0);
    let t = f32(planeCol) / denom;
    col = mix(p.segStartCol, p.segStopCol, t);
    row = mix(p.segStartRow, p.segStopRow, t);
    slice = f32(planeRow);
  }
  return sampleAligned(slice, col, row);
}

fn signedLog1p(v: f32) -> f32 {
  if (v >= 0.0) { return log(1.0 + v); }
  return -log(1.0 - v);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.outW || gid.y >= p.outH) { return; }
  // Full slice dims for this axis (source resolution).
  var fullW: u32; var fullH: u32;
  if (p.axis == 0u) { fullW = p.numCols; fullH = p.numRows; }          // top
  else if (p.axis == 1u) { fullW = p.numCols; fullH = p.numSlices; }   // row fixed
  else if (p.axis == 2u) { fullW = p.numRows; fullH = p.numSlices; }   // col fixed
  else { fullW = max(1u, p.canvasW); fullH = p.numSlices; }            // oblique
  var x0: u32; var y0: u32; var x1: u32; var y1: u32;
  if (p.viewMode == 0u) {
    // AREA AVERAGE downsample: this output pixel covers the source block
    // [x0,x1) x [y0,y1). Average every covered source value, so no source pixels
    // are silently skipped when a 4k slice is displayed in a smaller panel. When
    // outW==fullW the block is 1x1 = exact native pixel.
    x0 = (gid.x * fullW) / p.outW;
    y0 = (gid.y * fullH) / p.outH;
    x1 = max(x0 + 1u, ((gid.x + 1u) * fullW) / p.outW);
    y1 = max(y0 + 1u, ((gid.y + 1u) * fullH) / p.outH);
  } else {
    let cw = max(f32(p.canvasW), 1.0);
    let ch = max(f32(p.canvasH), 1.0);
    let z = max(p.zoom, ${VOLUME_SLICE_MIN_ZOOM});
    let cx = cw * 0.5;
    let cy = ch * 0.5;
    let sx0 = (((f32(gid.x) - cx - p.panX) / z) + cx) * f32(fullW) / cw;
    let sy0 = (((f32(gid.y) - cy - p.panY) / z) + cy) * f32(fullH) / ch;
    let sx1 = (((f32(gid.x + 1u) - cx - p.panX) / z) + cx) * f32(fullW) / cw;
    let sy1 = (((f32(gid.y + 1u) - cy - p.panY) / z) + cy) * f32(fullH) / ch;
    let loX = min(sx0, sx1);
    let hiX = max(sx0, sx1);
    let loY = min(sy0, sy1);
    let hiY = max(sy0, sy1);
    if (hiX <= 0.0 || hiY <= 0.0 || loX >= f32(fullW) || loY >= f32(fullH)) {
      rgba[gid.y * p.outW + gid.x] = 0xFF000000u;
      return;
    }
    x0 = u32(clamp(floor(loX), 0.0, f32(fullW - 1u)));
    y0 = u32(clamp(floor(loY), 0.0, f32(fullH - 1u)));
    x1 = max(x0 + 1u, u32(clamp(ceil(hiX), 1.0, f32(fullW))));
    y1 = max(y0 + 1u, u32(clamp(ceil(hiY), 1.0, f32(fullH))));
  }
  var sum = 0.0; var cnt = 0.0;
  for (var yy = y0; yy < y1; yy++) {
    for (var xx = x0; xx < x1; xx++) {
      sum = sum + sampleSlice(p.axis, p.index, min(xx, fullW - 1u), min(yy, fullH - 1u));
      cnt = cnt + 1.0;
    }
  }
  var val = sum / max(cnt, 1.0);
  if (p.logScale == 1u) { val = signedLog1p(val); }
  if (p.flip == 1u) { val = -val; }
  let t = display_normalize(val, p.vmin, p.vmax);
  let li = min(u32(t * 255.0), 255u);
  rgba[gid.y * p.outW + gid.x] = lut[li] | 0xFF000000u;
}
`;
}

const VOLUME_PARAMS_BYTES = 96;

interface VolumeSliceView {
  zoom: number;
  panX: number;
  panY: number;
  canvasW: number;
  canvasH: number;
}

interface VolumeSliceAlignment {
  rowShift: number;
  colShift: number;
  segment?: {
    start: { row: number; col: number };
    stop: { row: number; col: number };
  };
}

/**
 * Volume-resident slice pipeline (Show3DSlices): volume uploaded once, slice +
 * colormap done on GPU per scrub - no per-frame CPU extract / re-upload. It
 * paints with the engine's current LUT and blit pass.
 */
export class VolumeSlices {
  private readonly engine: GPUColormapEngine;
  private readonly device: GPUDevice;
  private volumePipeline: GPUComputePipeline | null = null;
  private volumeTexturePipeline: GPUComputePipeline | null = null;
  private volumeBuffer: GPUBuffer | null = null;
  private volumeTexture: GPUTexture | null = null;
  private volTextureView: GPUTextureView | null = null;
  private volUseTexture = false;
  private volTextureWidth = 0;
  private volCols = 0;
  private volRows = 0;
  private volSlices = 0;
  private volCount = 0;
  private volParamsBuffer: GPUBuffer | null = null;
  private volRgbaBuffer: GPUBuffer | null = null;
  private volRgbaCapacity = 0;
  private volParams = new ArrayBuffer(VOLUME_PARAMS_BYTES);
  private volParamsU32 = new Uint32Array(this.volParams);
  private volParamsF32 = new Float32Array(this.volParams);
  private volBlitCanvas: OffscreenCanvas | null = null;
  private volBlitContext: GPUCanvasContext | null = null;
  private volBlitFormat: GPUTextureFormat | null = null;
  private volBlitWidth = 0;
  private volBlitHeight = 0;
  private volBlitParamsBuffer: GPUBuffer | null = null;
  private volBlitParams = new Uint32Array(2);
  private volBlitBindGroup: GPUBindGroup | null = null;
  volComputeBindGroup: GPUBindGroup | null = null;
  volTextureBindGroup: GPUBindGroup | null = null;

  constructor(engine: GPUColormapEngine) {
    this.engine = engine;
    this.device = engine.getDevice();
  }

  private ensureVolumePipeline(): void {
    this.volumePipeline ??= computePipeline(this.device, volumeSliceShader("buffer"));
    this.volParamsBuffer ??= this.device.createBuffer({
      size: VOLUME_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private ensureVolumeTexturePipeline(): void {
    this.volumeTexturePipeline ??= computePipeline(this.device, volumeSliceShader("texture"));
    this.volParamsBuffer ??= this.device.createBuffer({
      size: VOLUME_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Upload a 3D volume (slices, rows, cols) row-major float32 into GPU memory once.
   * 4k and row-padded large stacks use a 2D texture array to avoid
   * storage-buffer binding limits. The texture path keeps original float32
   * values; unaligned rows are padded only in the upload stride and never
   * sampled. Other shapes use the storage-buffer path.
   */
  uploadVolume(vol: Float32Array, numCols: number, numRows: number, numSlices: number): boolean {
    const rowBytes = numCols * 4;
    const paddedRowBytes = Math.ceil(rowBytes / 256) * 256;
    const textureWidth = paddedRowBytes / 4;
    const canTexture = textureWidth <= this.device.limits.maxTextureDimension2D &&
      numRows <= this.device.limits.maxTextureDimension2D &&
      numSlices <= this.device.limits.maxTextureArrayLayers;
    if (canTexture) {
      this.ensureVolumeTexturePipeline();
      const needsTexture = !this.volumeTexture || this.volCount !== vol.length ||
        this.volCols !== numCols || this.volRows !== numRows || this.volSlices !== numSlices ||
        this.volTextureWidth !== textureWidth;
      if (needsTexture) {
        this.volumeTexture?.destroy();
        this.volumeTexture = this.device.createTexture({
          size: { width: textureWidth, height: numRows, depthOrArrayLayers: numSlices },
          format: "r32float",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.volTextureView = this.volumeTexture.createView({ dimension: "2d-array" });
        this.volTextureBindGroup = null;
        this.volCount = vol.length;
      }
      const texture = this.volumeTexture!;
      if (paddedRowBytes === rowBytes) {
        this.device.queue.writeTexture(
          { texture },
          vol.buffer as ArrayBuffer,
          { offset: vol.byteOffset, bytesPerRow: rowBytes, rowsPerImage: numRows },
          { width: numCols, height: numRows, depthOrArrayLayers: numSlices },
        );
      } else {
        const layer = new Float32Array(textureWidth * numRows);
        const sliceStride = numCols * numRows;
        for (let slice = 0; slice < numSlices; slice++) {
          const sliceStart = slice * sliceStride;
          for (let row = 0; row < numRows; row++) {
            const rowStart = sliceStart + row * numCols;
            layer.set(vol.subarray(rowStart, rowStart + numCols), row * textureWidth);
          }
          this.device.queue.writeTexture(
            { texture, origin: { x: 0, y: 0, z: slice } },
            layer.buffer,
            { bytesPerRow: paddedRowBytes, rowsPerImage: numRows },
            { width: numCols, height: numRows, depthOrArrayLayers: 1 },
          );
        }
      }
      this.volUseTexture = true;
      this.volumeBuffer?.destroy();
      this.volumeBuffer = null;
      this.volComputeBindGroup = null;
      this.volTextureWidth = textureWidth;
      this.volCols = numCols; this.volRows = numRows; this.volSlices = numSlices;
      return true;
    }
    this.ensureVolumePipeline();
    this.volumeTexture?.destroy();
    this.volumeTexture = null;
    this.volTextureView = null;
    this.volTextureBindGroup = null;
    this.volTextureWidth = 0;
    const maxBind = this.device.limits.maxStorageBufferBindingSize;
    if (vol.byteLength > maxBind) return false;
    if (!this.volumeBuffer || this.volCount !== vol.length) {
      this.volumeBuffer?.destroy();
      this.volumeBuffer = this.device.createBuffer({
        size: vol.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.volCount = vol.length;
      this.volComputeBindGroup = null;
    }
    this.device.queue.writeBuffer(this.volumeBuffer, 0, vol.buffer as ArrayBuffer, vol.byteOffset, vol.byteLength);
    this.volUseTexture = false;
    this.volCols = numCols; this.volRows = numRows; this.volSlices = numSlices;
    return true;
  }

  /**
   * Slice the resident volume along `axis` (0=top, 1=row fixed, 2=col fixed) at `index`,
   * colormap with the current LUT + vmin/vmax (logScale/flip applied in-shader to
   * match the CPU path), and blit to an ImageBitmap. Returns null if the volume
   * isn't uploaded or the LUT/pipeline isn't ready (caller falls back to CPU).
   */
  renderVolumeSliceToImageBitmap(
    axis: number, index: number,
    range: { vmin: number; vmax: number },
    logScale: boolean, flip: boolean,
    maxOut?: number,
    view?: VolumeSliceView,
    alignment?: VolumeSliceAlignment,
  ): ImageBitmap | null {
    const texturePipeline = this.volUseTexture ? this.volumeTexturePipeline : null;
    const textureView = this.volUseTexture ? this.volTextureView : null;
    const useTexture = texturePipeline != null && textureView != null;
    const bufferPipeline = this.volumePipeline;
    const useBuffer = !useTexture && this.volumeBuffer != null && bufferPipeline != null;
    if ((!useTexture && !useBuffer) || !this.engine.lutBuffer || !this.volParamsBuffer) return null;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.engine.ensureBlitPipeline(format);
    if (!this.engine.blitPipeline) return null;
    const numCols = this.volCols, numRows = this.volRows, numSlices = this.volSlices;
    const segmentWidth = alignment?.segment
      ? Math.max(1, Math.ceil(Math.hypot(
        alignment.segment.stop.col - alignment.segment.start.col,
        alignment.segment.stop.row - alignment.segment.start.row,
      )) + 1)
      : 1;
    const fullW = axis === 0 ? numCols : axis === 1 ? numCols : axis === 2 ? numRows : segmentWidth;
    const fullH = axis === 0 ? numRows : numSlices;
    // If maxOut is supplied, cap the output raster while still sampling from
    // the full-resolution source. Callers that need native-pixel zoom leave it
    // undefined, so outW/outH stay at the full slice dimensions.
    const cap = maxOut && maxOut > 0 ? maxOut : Math.max(fullW, fullH);
    const scale = Math.min(1, cap / Math.max(fullW, fullH));
    const outW = view ? Math.max(1, Math.round(view.canvasW)) : Math.max(1, Math.round(fullW * scale));
    const outH = view ? Math.max(1, Math.round(view.canvasH)) : Math.max(1, Math.round(fullH * scale));
    const sliceIndex = Math.max(0, Math.min((axis === 2 ? numCols : axis === 1 ? numRows : numSlices) - 1, Math.round(index)));
    const rgbaCount = outW * outH;
    if (!this.volRgbaBuffer || this.volRgbaCapacity < rgbaCount) {
      this.volRgbaBuffer?.destroy();
      this.volRgbaBuffer = this.device.createBuffer({
        size: rgbaCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.volRgbaCapacity = rgbaCount;
      this.volBlitBindGroup = null;
      this.volComputeBindGroup = null;
      this.volTextureBindGroup = null;
    }
    const paramsBuffer = this.volParamsBuffer;
    const lutBuffer = this.engine.lutBuffer;
    const rgbaBuffer = this.volRgbaBuffer;
    if (!paramsBuffer || !lutBuffer || !rgbaBuffer) return null;
    // VParams: u32 control block + float contrast / viewport block.
    const paramsU32 = this.volParamsU32; const paramsF32 = this.volParamsF32;
    paramsU32[0] = numCols; paramsU32[1] = numRows; paramsU32[2] = numSlices; paramsU32[3] = axis;
    paramsU32[4] = sliceIndex; paramsU32[5] = outW; paramsU32[6] = outH; paramsU32[7] = logScale ? 1 : 0;
    paramsU32[8] = flip ? 1 : 0;
    paramsU32[9] = view ? 1 : 0;
    paramsU32[10] = view ? Math.max(1, Math.round(view.canvasW)) : outW;
    paramsU32[11] = view ? Math.max(1, Math.round(view.canvasH)) : outH;
    paramsF32[12] = range.vmin; paramsF32[13] = range.vmax;
    paramsF32[14] = view ? Math.max(1e-6, view.zoom) : 1;
    paramsF32[15] = view ? view.panX : 0;
    paramsF32[16] = view ? view.panY : 0;
    paramsF32[17] = alignment && Number.isFinite(alignment.rowShift) ? alignment.rowShift : 0;
    paramsF32[18] = alignment && Number.isFinite(alignment.colShift) ? alignment.colShift : 0;
    paramsF32[19] = alignment?.segment ? alignment.segment.start.col : 0;
    paramsF32[20] = alignment?.segment ? alignment.segment.start.row : 0;
    paramsF32[21] = alignment?.segment ? alignment.segment.stop.col : 0;
    paramsF32[22] = alignment?.segment ? alignment.segment.stop.row : 0;
    paramsF32[23] = 0;
    this.device.queue.writeBuffer(paramsBuffer, 0, this.volParams);
    const encoder = this.device.createCommandEncoder();
    if (useTexture) {
      if (!this.volTextureBindGroup) {
        this.volTextureBindGroup = this.device.createBindGroup({
          layout: texturePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: textureView },
            { binding: 2, resource: { buffer: lutBuffer } },
            { binding: 3, resource: { buffer: rgbaBuffer } },
          ],
        });
      }
    } else if (!this.volComputeBindGroup && bufferPipeline && this.volumeBuffer) {
      this.volComputeBindGroup = this.device.createBindGroup({
        layout: bufferPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: this.volumeBuffer } },
          { binding: 2, resource: { buffer: lutBuffer } },
          { binding: 3, resource: { buffer: rgbaBuffer } },
        ],
      });
    }
    const computePass = encoder.beginComputePass();
    if (useTexture) {
      computePass.setPipeline(texturePipeline);
      computePass.setBindGroup(0, this.volTextureBindGroup!);
    } else {
      if (!bufferPipeline || !this.volComputeBindGroup) { computePass.end(); return null; }
      computePass.setPipeline(bufferPipeline);
      computePass.setBindGroup(0, this.volComputeBindGroup);
    }
    computePass.dispatchWorkgroups(Math.ceil(outW / 16), Math.ceil(outH / 16));
    computePass.end();
    const sizeChanged = !this.volBlitCanvas || this.volBlitWidth !== outW || this.volBlitHeight !== outH;
    const formatChanged = this.volBlitFormat !== format;
    if (sizeChanged || formatChanged || !this.volBlitContext) {
      this.volBlitCanvas = new OffscreenCanvas(outW, outH);
      this.volBlitContext = this.volBlitCanvas.getContext("webgpu") as GPUCanvasContext | null;
      if (!this.volBlitContext) return null;
      this.volBlitContext.configure({ device: this.device, format, alphaMode: "opaque" });
      this.volBlitWidth = outW;
      this.volBlitHeight = outH;
      this.volBlitFormat = format;
    }
    if (!this.volBlitParamsBuffer) {
      this.volBlitParamsBuffer = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    this.volBlitParams[0] = outW;
    this.volBlitParams[1] = outH;
    this.device.queue.writeBuffer(this.volBlitParamsBuffer, 0, this.volBlitParams);
    if (!this.volBlitBindGroup) {
      this.volBlitBindGroup = this.device.createBindGroup({
        layout: this.engine.blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.volBlitParamsBuffer } },
          { binding: 1, resource: { buffer: rgbaBuffer } },
        ],
      });
    }
    const blitContext = this.volBlitContext;
    const blitCanvas = this.volBlitCanvas;
    if (!blitContext || !blitCanvas) return null;
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{ view: blitContext.getCurrentTexture().createView(), loadOp: "clear" as GPULoadOp, storeOp: "store" as GPUStoreOp, clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    renderPass.setPipeline(this.engine.blitPipeline);
    renderPass.setBindGroup(0, this.volBlitBindGroup);
    renderPass.draw(3);
    renderPass.end();
    this.device.queue.submit([encoder.finish()]);
    return blitCanvas.transferToImageBitmap();
  }

  /** Release the volume and its slice buffers; the engine's destroy calls this. */
  destroy(): void {
    this.volumeBuffer?.destroy(); this.volumeBuffer = null;
    this.volumeTexture?.destroy(); this.volumeTexture = null; this.volTextureView = null;
    this.volParamsBuffer?.destroy(); this.volParamsBuffer = null;
    this.volRgbaBuffer?.destroy(); this.volRgbaBuffer = null;
    this.volBlitParamsBuffer?.destroy(); this.volBlitParamsBuffer = null;
    this.volBlitCanvas = null; this.volBlitContext = null; this.volBlitFormat = null;
    this.volBlitBindGroup = null; this.volComputeBindGroup = null; this.volTextureBindGroup = null; this.volBlitWidth = 0; this.volBlitHeight = 0;
    this.volUseTexture = false;
    this.volCount = 0; this.volRgbaCapacity = 0; this.volTextureWidth = 0;
  }
}
