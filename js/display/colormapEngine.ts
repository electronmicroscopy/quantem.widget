// The WebGPU colormap engine: persistent per-image slots, the LUT and the
// blit pass. Range/histogram passes, direct canvas rendering and volume slices
// live in their own modules; the engine is the one object widgets call.
import { DirectRenderer } from "./directRender";
import { AVERAGE_SHADER, BLIT_SHADER, COLORMAP_SHADER, RGB_PASSTHROUGH_SHADER, SCALED_COLORMAP_SHADER, SCALED_UINT8_COLORMAP_SHADER } from "./displayShaders";
import { getHardwareGPUDevice } from "./device";
import { GPURanges } from "./gpuRange";
import { computePipeline, renderPipeline } from "./pipelines";
import { VolumeSlices } from "./volumeSlice";

export type GPUBufferOwnership = "owned" | "borrowed";

export type GPUSlot = {
  dataBuffer: GPUBuffer;
  dataOwnership: GPUBufferOwnership;
  rgbaBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  paramsBuffer: GPUBuffer;
  blitParamsBuffer: GPUBuffer;
  histBinsBuffer: GPUBuffer;
  // Lazily allocated per-slot 16-byte buffer holding { vmin, vmax, _p0, _p1 }.
  // Populated by computeRange* on GPU and consumed directly by the range-aware
  // colormap shader (no CPU readback between passes).
  rangeBuffer: GPUBuffer | null;
  liveRange?: {
    groups: number;
    partials: GPUBuffer;
    parameters: GPUBuffer;
    values: Uint32Array<ArrayBuffer>;
    reduceGroup: GPUBindGroup;
    finishGroup: GPUBindGroup;
    renderGroup: GPUBindGroup | null;
  };
  directGridBindGroup: GPUBindGroup | null;
  directSlotBindGroup: GPUBindGroup | null;
  directRegionParamsBuffers: (GPUBuffer | null)[];
  directRegionBindGroups: (GPUBindGroup | null)[];
  directRegionLutNames: string[];
  count: number;
  rgbaCapacity: number;
  width: number;
  height: number;
  directOnly: boolean;
  dataKind: "f32" | "u8";
};

/**
 * WebGPU colormap engine. Holds persistent data buffers on the GPU; histogram
 * slider changes only update a small uniform, with no data re-upload.
 * GPURanges, DirectRenderer and VolumeSlices draw through the engine's slots,
 * LUT and blit pass, so those members are public.
 */
export class GPUColormapEngine {
  readonly path = "WebGPU" as const;
  private device: GPUDevice;
  pipeline: GPUComputePipeline | null = null;
  scaledPipeline: GPUComputePipeline | null = null;
  scaledUint8Pipeline: GPUComputePipeline | null = null;
  blitPipeline: GPURenderPipeline | null = null;
  // GPU temporal-average state: a compute pipeline that means N window frames
  // into a slot's dataBuffer, plus a reused scratch buffer for the window frames.
  private avgPipeline: GPUComputePipeline | null = null;
  private avgScratch: GPUBuffer | null = null;
  private avgScratchSize = 0;
  private avgParamsBuffer: GPUBuffer | null = null;
  // True-color passthrough state: an RGB->rgba compute pipeline plus reused
  // input/output buffers, so RGB frames paint on the GPU like grayscale.
  private rgbPipeline: GPUComputePipeline | null = null;
  private rgbDataBuffer: GPUBuffer | null = null;
  private rgbDataCapacity = 0;
  private rgbRgbaBuffer: GPUBuffer | null = null;
  private rgbRgbaCapacity = 0;
  private rgbParamsBuffer: GPUBuffer | null = null;
  // Persistent per-image buffers: data, rgba, readback, params, histogram.
  slots: GPUSlot[] = [];
  lutBuffer: GPUBuffer | null = null;
  currentLutName: string = "";
  private namedLutBuffers = new Map<string, GPUBuffer>();
  // Scratch for the 24-byte colormap params; writeBuffer copies it before reuse.
  private colormapParams = new ArrayBuffer(24);
  private retiredSlots: GPUSlot[] = [];
  // The drawing sections, one module each; they draw through the slots, LUT and blit pass above.
  private readonly ranges: GPURanges;
  private readonly direct: DirectRenderer;
  private readonly volume: VolumeSlices;

  constructor(device: GPUDevice) {
    this.device = device;
    this.ranges = new GPURanges(this);
    this.direct = new DirectRenderer(this, this.ranges);
    this.volume = new VolumeSlices(this);
  }

  getDevice(): GPUDevice { return this.device; }

  private destroySlot(slot: GPUSlot): void {
    if (slot.dataOwnership === "owned") slot.dataBuffer.destroy();
    slot.rgbaBuffer.destroy();
    slot.readBuffer.destroy();
    slot.paramsBuffer.destroy();
    slot.blitParamsBuffer.destroy();
    slot.histBinsBuffer.destroy();
    slot.rangeBuffer?.destroy();
    slot.liveRange?.partials.destroy();
    slot.liveRange?.parameters.destroy();
    for (const buf of slot.directRegionParamsBuffers) buf?.destroy();
  }

  private createLutBuffer(lut: Uint8Array): GPUBuffer {
    const packed = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      packed[i] = lut[i * 3] | (lut[i * 3 + 1] << 8) | (lut[i * 3 + 2] << 16);
    }
    const buffer = this.device.createBuffer({
      size: packed.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, packed);
    return buffer;
  }

  namedLutBuffer(name: string, lut: Uint8Array): GPUBuffer {
    const cached = this.namedLutBuffers.get(name);
    if (cached) return cached;
    const buffer = this.createLutBuffer(lut);
    this.namedLutBuffers.set(name, buffer);
    return buffer;
  }

  private retireSlot(slot: GPUSlot): void {
    this.retiredSlots.push(slot);
    void this.device.queue.onSubmittedWorkDone()
      .catch(() => {})
      .finally(() => {
        const idx = this.retiredSlots.indexOf(slot);
        if (idx < 0) return;
        this.retiredSlots.splice(idx, 1);
        this.destroySlot(slot);
      });
  }

  releaseSlot(idx: number): void {
    const slot = this.slots[idx];
    if (!slot) return;
    this.destroySlot(slot);
    this.slots[idx] = null as never;
  }

  private ensurePipeline(): void {
    this.pipeline ??= computePipeline(this.device, COLORMAP_SHADER);
  }

  ensureScaledPipeline(): void {
    this.scaledPipeline ??= computePipeline(this.device, SCALED_COLORMAP_SHADER);
  }

  ensureScaledUint8Pipeline(): void {
    this.scaledUint8Pipeline ??= computePipeline(this.device, SCALED_UINT8_COLORMAP_SHADER);
  }

  private ensureAveragePipeline(): void {
    this.avgPipeline ??= computePipeline(this.device, AVERAGE_SHADER);
  }

  /**
   * Average already-resident frame slots entirely on the GPU.
   *
   * The source buffers are copied GPU-to-GPU into the reused average scratch
   * buffer, then the same temporal-mean compute shader writes the result into
   * `idx`. This keeps Show3D scrubbing off the JS CPU after residency completes.
   */
  averageResidentSlotsInto(idx: number, sourceIndices: number[]): boolean {
    const sources = sourceIndices.map(sourceIdx => this.slots[sourceIdx]).filter(Boolean);
    if (sources.length !== sourceIndices.length || sources.length === 0) return false;
    const first = sources[0];
    if (!sources.every(slot => (
      slot.count === first.count && slot.width === first.width && slot.height === first.height
    ))) return false;

    while (this.slots.length <= idx) this.slots.push(null as never);
    let target = this.slots[idx];
    if (
      !target || target.count !== first.count ||
      target.width !== first.width || target.height !== first.height
    ) {
      this.uploadData(idx, new Float32Array(first.count), first.width, first.height, first.rgbaCapacity);
      target = this.slots[idx];
    }

    this.ensureAveragePipeline();
    const frameBytes = first.count * 4;
    const scratchBytes = sources.length * frameBytes;
    if (!this.avgScratch || this.avgScratchSize < scratchBytes) {
      this.avgScratch?.destroy();
      this.avgScratch = this.device.createBuffer({
        size: scratchBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.avgScratchSize = scratchBytes;
    }
    if (!this.avgParamsBuffer) {
      this.avgParamsBuffer = this.device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(
      this.avgParamsBuffer,
      0,
      new Uint32Array([sources.length, first.count]),
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.avgPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.avgScratch } },
        { binding: 1, resource: { buffer: target.dataBuffer } },
        { binding: 2, resource: { buffer: this.avgParamsBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    sources.forEach((slot, sourceIdx) => {
      encoder.copyBufferToBuffer(slot.dataBuffer, 0, this.avgScratch!, sourceIdx * frameBytes, frameBytes);
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.avgPipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(first.count / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  /** Upload LUT to GPU (only when colormap name changes). */
  uploadLUT(lutName: string, lut: Uint8Array): void {
    if (this.currentLutName === lutName && this.lutBuffer) return;
    this.ensurePipeline();
    if (this.lutBuffer) {
      this.lutBuffer.destroy();
      this.volume.volComputeBindGroup = null;
      this.volume.volTextureBindGroup = null;
      for (const slot of this.slots) {
        if (!slot) continue;
        slot.directGridBindGroup = null;
        slot.directSlotBindGroup = null;
        if (slot.liveRange) slot.liveRange.renderGroup = null;
        slot.directRegionBindGroups = slot.directRegionBindGroups.map(() => null);
      }
    }
    this.lutBuffer = this.createLutBuffer(lut);
    this.currentLutName = lutName;
  }

  /**
   * Slot record around an already-written data buffer, with fresh per-slot
   * rgba, readback, params and histogram buffers and no cached bind groups.
   * The params buffer holds 64 bytes: the 24-byte colormap and histogram
   * structs, the 32-byte scaled colormap struct and the 64-byte direct grid struct.
   */
  private createSlot(
    dataBuffer: GPUBuffer, dataOwnership: GPUBufferOwnership, dataKind: "f32" | "u8",
    count: number, width: number, height: number, rgbaCapacity: number, readBytes: number, directOnly: boolean,
  ): GPUSlot {
    return {
      dataBuffer,
      dataOwnership,
      rgbaBuffer: this.device.createBuffer({ size: rgbaCapacity * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      readBuffer: this.device.createBuffer({ size: readBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      paramsBuffer: this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      blitParamsBuffer: this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      histBinsBuffer: this.device.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      rangeBuffer: null,
      directGridBindGroup: null,
      directSlotBindGroup: null,
      directRegionParamsBuffers: [],
      directRegionBindGroups: [],
      directRegionLutNames: [],
      count,
      rgbaCapacity,
      width,
      height,
      directOnly,
      dataKind,
    };
  }

  /** Upload a width x height float32 image for slot `idx`. Only call when data changes. */
  uploadData(idx: number, data: Float32Array, width: number, height: number, rgbaCapacityHint?: number, directOnly: boolean = false): void {
    // a guessed near-square shape would paint a non-square image as noise, so a mismatch is refused
    if (width < 1 || height < 1 || width * height !== data.length) {
      throw new Error(`display dimensions ${width}x${height} do not match ${data.length} pixels`);
    }
    this.ensurePipeline();
    while (this.slots.length <= idx) this.slots.push(null as never);
    const byteSize = data.byteLength;
    const rgbaCapacity = directOnly ? 1 : Math.max(1, Math.round(rgbaCapacityHint ?? data.length));
    const existing = this.slots[idx];
    if (existing && existing.dataOwnership === "owned" && existing.dataKind === "f32" && existing.directOnly === directOnly && existing.count === data.length && existing.width === width && existing.height === height && existing.rgbaCapacity >= rgbaCapacity) {
      this.device.queue.writeBuffer(existing.dataBuffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
      return;
    }
    if (existing) {
      this.destroySlot(existing);
    }
    const dataBuffer = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.device.queue.writeBuffer(dataBuffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    this.slots[idx] = this.createSlot(dataBuffer, "owned", "f32", data.length, width, height, rgbaCapacity, rgbaCapacity * 4, directOnly);
  }

  /** Upload one native uint8 image without expanding it to float32. */
  uploadUint8Data(
    idx: number,
    data: Uint8Array | Uint8ClampedArray,
    width: number,
    height: number,
    rgbaCapacityHint: number,
  ): void {
    this.ensureScaledUint8Pipeline();
    while (this.slots.length <= idx) this.slots.push(null as never);
    if (width <= 0 || height <= 0 || width * height !== data.length) {
      throw new Error(`uint8 display dimensions ${width}x${height} do not match ${data.length} pixels`);
    }
    const packedBytes = Math.max(4, Math.ceil(data.byteLength / 4) * 4);
    const upload = packedBytes === data.byteLength ? data : (() => {
      const padded = new Uint8Array(packedBytes);
      padded.set(data);
      return padded;
    })();
    const rgbaCapacity = Math.max(1, Math.round(rgbaCapacityHint));
    const existing = this.slots[idx];
    if (
      existing
      && existing.dataOwnership === "owned"
      && existing.dataKind === "u8"
      && existing.count === data.length
      && existing.width === width
      && existing.height === height
      && existing.rgbaCapacity >= rgbaCapacity
    ) {
      this.device.queue.writeBuffer(existing.dataBuffer, 0, upload.buffer as ArrayBuffer, upload.byteOffset, upload.byteLength);
      return;
    }
    if (existing) this.destroySlot(existing);
    const dataBuffer = this.device.createBuffer({
      size: packedBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.device.queue.writeBuffer(dataBuffer, 0, upload.buffer as ArrayBuffer, upload.byteOffset, upload.byteLength);
    this.slots[idx] = this.createSlot(dataBuffer, "owned", "u8", data.length, width, height, rgbaCapacity, rgbaCapacity * 4, false);
  }

  /** Display caller-owned storage; the caller retains its lifetime. */
  borrowBuffer(idx: number, buffer: GPUBuffer, width: number, height: number): void {
    this.adoptBuffer(idx, buffer, width, height, "borrowed");
  }

  /**
   * Adopt an externally produced float32 GPU buffer as a display slot. The slot
   * owns the buffer after this call unless ownership is explicitly "borrowed".
   * Borrowed storage remains the caller's responsibility and must outlive all
   * queued display work. Slot replacement, release, and engine destruction never
   * destroy borrowed storage. If the same buffer is adopted again, keep
   * the existing display resources so in-place GPU updates can repaint without
   * retiring their own source.
   *
   * @example
   * engine.adoptBuffer(41, residentDisplay.buffer, 512, 512, "borrowed");
   */
  adoptBuffer(idx: number, buffer: GPUBuffer, width: number, height: number, ownership: GPUBufferOwnership = "owned"): void {
    while (this.slots.length <= idx) this.slots.push(null as never);
    const old = this.slots[idx];
    if (old && old.dataBuffer === buffer) {
      if (old.width === width && old.height === height) {
        old.dataOwnership = ownership;
        return;
      }
      // The new slot retains the same storage while old display resources retire.
      old.dataOwnership = "borrowed";
    }
    if (old) this.retireSlot(old);
    const count = Math.max(1, width * height);
    // The 16-byte readback buffer is a placeholder: adopted slots draw without rgba readback.
    this.slots[idx] = this.createSlot(buffer, ownership, "f32", count, width, height, count, 16, false);
  }

  /** Fill the 24-byte colormap (and histogram) params: { width, height, vmin, vmax, log_scale, _pad }. */
  writeColormapParams(params: ArrayBuffer, width: number, height: number, vmin: number, vmax: number, logScale: boolean): void {
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = width;
    paramsU32[1] = height;
    paramsF32[2] = vmin;
    paramsF32[3] = vmax;
    paramsU32[4] = logScale ? 1 : 0;
    paramsU32[5] = 0;
  }

  /**
   * Encode the colormap compute pass that fills `slot`'s packed rgba buffer from
   * its data, display range and `lutBuffer`. The params go through the slot's
   * own uniform, written now, so slots encoded into one submit keep their ranges.
   */
  private encodeColormapPass(
    encoder: GPUCommandEncoder, slot: GPUSlot, vmin: number, vmax: number, logScale: boolean, lutBuffer: GPUBuffer,
  ): void {
    this.writeColormapParams(this.colormapParams, slot.width, slot.height, vmin, vmax, logScale);
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, this.colormapParams);
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: lutBuffer } },
        { binding: 3, resource: { buffer: slot.rgbaBuffer } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
    pass.end();
  }

  /** Apply one slot with an explicitly named LUT without mutating the engine-wide LUT. */
  async applySingleWithLut(
    idx: number,
    vmin: number,
    vmax: number,
    lutName: string,
    lut: Uint8Array,
    logScale: boolean = false,
  ): Promise<Uint8ClampedArray | null> {
    const slot = this.slots[idx];
    if (!this.pipeline || !slot || slot.directOnly || slot.rgbaCapacity < slot.count) return null;
    const lutBuffer = this.namedLutBuffer(lutName, lut);
    const encoder = this.device.createCommandEncoder();
    this.encodeColormapPass(encoder, slot, vmin, vmax, logScale, lutBuffer);
    encoder.copyBufferToBuffer(slot.rgbaBuffer, 0, slot.readBuffer, 0, slot.count * 4);
    this.device.queue.submit([encoder.finish()]);
    await slot.readBuffer.mapAsync(GPUMapMode.READ);
    // The read buffer holds rgbaCapacity pixels; only the first count are this image.
    const rgba = new Uint8ClampedArray(slot.readBuffer.getMappedRange(0, slot.count * 4).slice(0));
    slot.readBuffer.unmap();
    return rgba;
  }

  ensureBlitPipeline(format: GPUTextureFormat): void {
    this.blitPipeline ??= renderPipeline(this.device, BLIT_SHADER, format);
  }

  /**
   * Encode the fullscreen blit of a packed rgba buffer into a new width x height
   * OffscreenCanvas. Returns the canvas and its blit params buffer, which the
   * caller destroys once the encoder is submitted, or null when the canvas has
   * no WebGPU context. The blit pipeline must already exist.
   */
  encodeBlitToOffscreen(
    encoder: GPUCommandEncoder, rgbaBuffer: GPUBuffer, width: number, height: number, format: GPUTextureFormat,
  ): { canvas: OffscreenCanvas; blitParamsBuffer: GPUBuffer } | null {
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) return null;
    context.configure({ device: this.device, format, alphaMode: "opaque" });
    const blitParamsBuffer = this.device.createBuffer({
      size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(blitParamsBuffer, 0, new Uint32Array([width, height]));
    const blitGroup = this.device.createBindGroup({
      layout: this.blitPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: blitParamsBuffer } },
        { binding: 1, resource: { buffer: rgbaBuffer } },
      ],
    });
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    renderPass.setPipeline(this.blitPipeline!);
    renderPass.setBindGroup(0, blitGroup);
    renderPass.draw(3);
    renderPass.end();
    return { canvas, blitParamsBuffer };
  }

  /**
   * GPU colormap → OffscreenCanvas → ImageBitmap (zero mapAsync).
   * Compute shader writes RGBA, render pass blits to OffscreenCanvas texture,
   * transferToImageBitmap() returns ImageBitmap for drawImage on 2D canvas.
   * Eliminates the 35ms JS memcpy for 12×4K images.
   */
  renderSlotsToImageBitmap(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean = false,
  ): ImageBitmap[] | null {
    if (!this.lutBuffer) return null;
    const canvases = this.encodeSlotsToOffscreen(indices, ranges, logScale, this.lutBuffer);
    if (!canvases) return null;
    return this.transferOffscreens(canvases);
  }

  private ensureRgbPipeline(): void {
    this.rgbPipeline ??= computePipeline(this.device, RGB_PASSTHROUGH_SHADER);
  }

  /**
   * Paint an interleaved RGB float frame (3 channels in [0, 1]) to an
   * ImageBitmap entirely on the GPU: a passthrough compute shader packs the
   * channels into rgba, then the existing blit renders it. Returns null (so the
   * caller falls back to the CPU pack loop) if WebGPU is unavailable, the frame
   * is too large for the device's storage-buffer limit, or a validation error
   * fires. The device already requests the adapter's max buffer limits, so a 4K
   * color slot fits where the 128 MB default would have silently failed.
   */
  renderRgbToImageBitmap(rgb: Float32Array, width: number, height: number): ImageBitmap | null {
    if (width <= 0 || height <= 0) return null;
    const count = width * height;
    if (rgb.length < count * 3) return null;
    const rgbBytes = count * 3 * 4;
    const rgbaBytes = count * 4;
    const maxBind = this.device.limits.maxStorageBufferBindingSize;
    if (rgbBytes > maxBind || rgbaBytes > maxBind) return null;  // too big for one buffer -> CPU path
    try {
      this.ensureRgbPipeline();
      const format = navigator.gpu.getPreferredCanvasFormat();
      this.ensureBlitPipeline(format);
      if (!this.rgbPipeline || !this.blitPipeline) return null;

      if (!this.rgbDataBuffer || this.rgbDataCapacity < rgbBytes) {
        this.rgbDataBuffer?.destroy();
        this.rgbDataBuffer = this.device.createBuffer({
          size: rgbBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.rgbDataCapacity = rgbBytes;
      }
      if (!this.rgbRgbaBuffer || this.rgbRgbaCapacity < rgbaBytes) {
        this.rgbRgbaBuffer?.destroy();
        this.rgbRgbaBuffer = this.device.createBuffer({
          size: rgbaBytes, usage: GPUBufferUsage.STORAGE,
        });
        this.rgbRgbaCapacity = rgbaBytes;
      }
      if (!this.rgbParamsBuffer) {
        this.rgbParamsBuffer = this.device.createBuffer({
          size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
      }
      const src = rgb.length === count * 3 ? rgb : rgb.subarray(0, count * 3);
      this.device.queue.writeBuffer(this.rgbDataBuffer, 0, src.buffer, src.byteOffset, count * 3 * 4);
      this.device.queue.writeBuffer(this.rgbParamsBuffer, 0, new Uint32Array([width, height, 0, 0]));

      const encoder = this.device.createCommandEncoder();
      const computeGroup = this.device.createBindGroup({
        layout: this.rgbPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.rgbParamsBuffer } },
          { binding: 1, resource: { buffer: this.rgbDataBuffer } },
          { binding: 2, resource: { buffer: this.rgbRgbaBuffer } },
        ],
      });
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.rgbPipeline);
      computePass.setBindGroup(0, computeGroup);
      computePass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
      computePass.end();
      const blit = this.encodeBlitToOffscreen(encoder, this.rgbRgbaBuffer, width, height, format);
      if (!blit) return null;
      this.device.queue.submit([encoder.finish()]);
      blit.blitParamsBuffer.destroy();
      return blit.canvas.transferToImageBitmap();
    } catch {
      return null;
    }
  }

  /**
   * Async twin of renderSlotsToImageBitmap: waits for the submitted GPU work to
   * complete BEFORE snapshotting each OffscreenCanvas. Without the wait,
   * transferToImageBitmap() can snapshot the render pass's clear color (black)
   * before the blit executes, which is what produces the flaky black canvas
   * when the GPU queue is backed up (e.g. a 128 MB frame still draining over a
   * slow SSH tunnel keeps writeBuffer work ahead of the colormap pass).
   * The paint names its colormap instead of reading the engine-wide LUT, so
   * pipelines that paint from separate animation frames (image and FFT panels)
   * cannot repaint each other's panels in the wrong colormap.
   */
  async renderSlotsToImageBitmapAsync(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean,
    lutName: string,
    lut: Uint8Array,
  ): Promise<ImageBitmap[] | null> {
    this.ensurePipeline();
    const canvases = this.encodeSlotsToOffscreen(indices, ranges, logScale, this.namedLutBuffer(lutName, lut));
    if (!canvases) return null;
    await this.device.queue.onSubmittedWorkDone();
    return this.transferOffscreens(canvases);
  }

  /** Snapshot each encoded canvas as an ImageBitmap, keeping null slots in place. */
  transferOffscreens(canvases: (OffscreenCanvas | null)[]): ImageBitmap[] {
    return canvases.map((canvas) => canvas ? canvas.transferToImageBitmap() : null as never);
  }

  /**
   * Encode colormap + blit for each slot into its own OffscreenCanvas in one
   * submit. Slots that are missing, direct-only or under capacity get null.
   */
  private encodeSlotsToOffscreen(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean,
    lutBuffer: GPUBuffer,
  ): (OffscreenCanvas | null)[] | null {
    if (!this.pipeline || indices.length === 0) return null;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ensureBlitPipeline(format);
    if (!this.blitPipeline) return null;

    const encoder = this.device.createCommandEncoder();
    const canvases: (OffscreenCanvas | null)[] = [];
    const tempBuffers: GPUBuffer[] = [];
    for (let k = 0; k < indices.length; k++) {
      const slot = this.slots[indices[k]];
      if (!slot || slot.directOnly || slot.rgbaCapacity < slot.count) { canvases.push(null as never); continue; }
      const range = ranges[k] || { vmin: 0, vmax: 1 };
      this.encodeColormapPass(encoder, slot, range.vmin, range.vmax, logScale, lutBuffer);
      const blit = this.encodeBlitToOffscreen(encoder, slot.rgbaBuffer, slot.width, slot.height, format)!;
      tempBuffers.push(blit.blitParamsBuffer);
      canvases.push(blit.canvas);
    }
    this.device.queue.submit([encoder.finish()]);
    for (const buffer of tempBuffers) buffer.destroy();
    return canvases;
  }

  // Volume slices (volumeSlice.ts) and direct canvas rendering (directRender.ts).
  uploadVolume(...args: Parameters<VolumeSlices["uploadVolume"]>) { return this.volume.uploadVolume(...args); }
  renderVolumeSliceToImageBitmap(...args: Parameters<VolumeSlices["renderVolumeSliceToImageBitmap"]>) { return this.volume.renderVolumeSliceToImageBitmap(...args); }
  renderSlotScaledToImageBitmap(...args: Parameters<DirectRenderer["renderSlotScaledToImageBitmap"]>) { return this.direct.renderSlotScaledToImageBitmap(...args); }
  renderSlotScaledToImageBitmapAsync(...args: Parameters<DirectRenderer["renderSlotScaledToImageBitmapAsync"]>) { return this.direct.renderSlotScaledToImageBitmapAsync(...args); }
  renderSharedGridDirectToCanvas(...args: Parameters<DirectRenderer["renderSharedGridDirectToCanvas"]>) { return this.direct.renderSharedGridDirectToCanvas(...args); }
  renderSharedGridToImageBitmapAsync(...args: Parameters<DirectRenderer["renderSharedGridToImageBitmapAsync"]>) { return this.direct.renderSharedGridToImageBitmapAsync(...args); }
  renderPanelSlotsDirectToCanvas(...args: Parameters<DirectRenderer["renderPanelSlotsDirectToCanvas"]>) { return this.direct.renderPanelSlotsDirectToCanvas(...args); }
  renderPanelSlotsToImageBitmapAsync(...args: Parameters<DirectRenderer["renderPanelSlotsToImageBitmapAsync"]>) { return this.direct.renderPanelSlotsToImageBitmapAsync(...args); }
  renderSlotDirectWithGpuRangeToCanvas(...args: Parameters<DirectRenderer["renderSlotDirectWithGpuRangeToCanvas"]>) { return this.direct.renderSlotDirectWithGpuRangeToCanvas(...args); }
  renderCombinedPanelRegionsDirectToCanvas(...args: Parameters<DirectRenderer["renderCombinedPanelRegionsDirectToCanvas"]>) { return this.direct.renderCombinedPanelRegionsDirectToCanvas(...args); }
  renderCombinedPanelRegionsToImageBitmapAsync(...args: Parameters<DirectRenderer["renderCombinedPanelRegionsToImageBitmapAsync"]>) { return this.direct.renderCombinedPanelRegionsToImageBitmapAsync(...args); }

  /**
   * Configure a canvas for WebGPU zero-copy rendering.
   * Returns the GPUCanvasContext, or null if WebGPU canvas is not supported.
   */
  configureCanvas(canvas: HTMLCanvasElement, width: number, height: number): GPUCanvasContext | null {
    try {
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("webgpu") as GPUCanvasContext | null;
      if (!ctx) return null;
      ctx.configure({
        device: this.device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "opaque",
      });
      return ctx;
    } catch {
      return null;
    }
  }

  /** Release all GPU resources. */
  destroy(): void {
    for (const slot of this.slots) {
      if (slot) this.destroySlot(slot);
    }
    this.slots = [];
    for (const slot of this.retiredSlots) this.destroySlot(slot);
    this.retiredSlots = [];
    this.lutBuffer?.destroy();
    this.lutBuffer = null;
    for (const buffer of this.namedLutBuffers.values()) buffer.destroy();
    this.namedLutBuffers.clear();
    this.currentLutName = "";
    this.ranges.destroy();
    this.volume.destroy();
  }

  /** Number of uploaded image slots. */
  get slotCount(): number { return this.slots.filter(slot => slot).length; }

  /** Resolve once all GPU work submitted so far has completed. */
  async waitForSubmittedWork(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  // GPU ranges and histograms (gpuRange.ts).
  computeRangeBatch(...args: Parameters<GPURanges["computeRangeBatch"]>) { return this.ranges.computeRangeBatch(...args); }
  renderPerPanelGpuExplicit(...args: Parameters<GPURanges["renderPerPanelGpuExplicit"]>) { return this.ranges.renderPerPanelGpuExplicit(...args); }
  computeHistogramRegions(...args: Parameters<GPURanges["computeHistogramRegions"]>) { return this.ranges.computeHistogramRegions(...args); }
  computeHistogramBatch(...args: Parameters<GPURanges["computeHistogramBatch"]>) { return this.ranges.computeHistogramBatch(...args); }
  computeHistogramWithRange(...args: Parameters<GPURanges["computeHistogramWithRange"]>) { return this.ranges.computeHistogramWithRange(...args); }
}

/**
 * Create a colormap engine on the hardware adapter, or null when the browser
 * has no WebGPU; callers then draw with applyColormap/renderToOffscreen.
 */
export async function createGPUColormapEngine(): Promise<GPUColormapEngine | null> {
  const device = await getHardwareGPUDevice();
  if (!device) return null;
  // A display pipeline the adapter's shader compiler cannot build paints
  // black without an exception (FXC, below D3D feature level 12, rejected the
  // first level-edge shaders), so such a page draws with the CPU path instead.
  try {
    await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: COLORMAP_SHADER }), entryPoint: "main" },
    });
  } catch (error) {
    console.warn("WebGPU display shaders do not build on this adapter; drawing on the CPU.", error);
    return null;
  }
  return new GPUColormapEngine(device);
}
